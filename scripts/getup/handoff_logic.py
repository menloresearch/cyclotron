# Copyright (c) 2026, Menlo Research.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Pure-tensor state machine for the get-up <-> walking handoff demo.

Deliberately has **no Isaac Lab / Isaac Sim import**, so it can be unit-tested on its own (fast,
CPU, no GPU/sim needed) — see the ``if __name__ == "__main__":`` self-test at the bottom, run with
plain ``python3 scripts/getup/handoff_logic.py`` (only needs ``torch``).

Owns no simulation state. ``play_combined.py`` reads per-step tensors from the env (stand-hold
timer, per-joint-group deviation, angular velocity, projected-gravity z, base x position) and feeds
them to :meth:`HandoffStateMachine.step`; the result tells it whether to flip
``HandoffJointPositionAction.set_mode``, whether the 5 m walk leg is done, and whether to apply the
optional push. All comparisons are elementwise over the env batch, so this scales to any
``num_envs``.

Handoff condition (guards against handoff instability; the pose bound is **per joint group**
rather than a single 0.15 rad bound over all 23 joints): switch get-up -> walk only once *all* of
    stand_hold_s >= 1.0                     (``env.getup_state.stand_timer_s``, i.e. ``is_standing`` held)
    joint_dev_linf[group] < bound[group]    for *every* group in ``max_joint_dev_by_group``:
        legs_waist        <= 0.15 rad
        shoulders_elbows  <= 0.30 rad
        wrist_yaw         <= 0.50 rad
    ang_vel_norm < 0.3 rad/s                (L2, base angular velocity)
    not fallen                              (the lying detector below currently reads upright)
hold at once, and switch walk -> get-up only when the lying detector fires. Both transitions are
derived from the *same* per-tick snapshot of the mode and of ``fallen``, so they are mutually
exclusive within one ``.step()`` call: a handoff can never fire in the same tick (or the next) that
a re-fall would also fire, because a handoff *requires* ``not fallen`` and a re-fall *requires*
``fallen``. Two different signal groups gate the two different directions (stand_hold_s / pose /
ang_vel is a strictly-improving gate the walking direction owns; the lying detector is a
strictly-triggering gate the get-up direction owns), which gives real hysteresis rather than one
threshold with no memory.

Why per-group, not one scalar bound: a single 0.15 rad bound over *all* 23 joints was found
(running ``play_combined.py`` against an early training checkpoint) to reject every attempt whose
``is_standing`` (the get-up success definition: height/tilt/feet-contact/linear-velocity only, no
joint-pose check at all) was
otherwise satisfied, because the arms/wrists — which don't affect standing balance nearly as much
as the legs and waist do, and whose posture reward terms carry small weights and a delayed
curriculum ramp — settle far from the walking default pose long before the legs do. Splitting the
bound by body region lets the legs/waist (which *do* need to be near the walking pose for a clean
handoff — that's the whole point of the gate) stay strict, while giving the arms room to differ
without blocking every handoff.

An earlier version of this state machine did *not* require ``not fallen`` in the handoff gate, and
computed the walk -> get-up check from the *post-handoff* mode within the same tick. That let a
handoff fire while the gravity EMA still read "fallen" (plausible if the EMA lags the other three
signals even briefly), which then immediately re-triggered a fall on the same or next tick,
producing dozens of spurious transitions per second. The self-test's phase 2 (a deliberately
unrealistic ramp where gravity crosses its threshold slower than the other three signals) caught
this before it shipped — see "handoff steps" / "num_falls" assertions below.

Lying/fall detector (gravity-based, low-pass in the 100-300 ms range): a one-pole EMA of
``projected_gravity_b[:, 2]`` with a time constant ``lying_tau_s`` (default 0.2 s). Upright, this is close
to -1 (gravity points straight down in the body frame); tilted past ~60 degrees from vertical it
crosses the threshold (-0.5, i.e. cos(60 deg)). Because it is EMA-smoothed, a single noisy frame
during a hard landing does not immediately flip the mode — the usual anti-chatter idea behind alert
debouncing, just tuned to the much shorter timescale a fall-recovery trigger needs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

# Canonical joint-group names. `play_combined.py` maps ASIMOV_1_JOINT_NAMES to
# exactly these three groups (legs+waist, shoulders+elbows, wrist yaw); kept as plain string keys
# here (not an enum) so this file stays isaaclab-free and the mapping lives entirely on the caller
# side, where the joint names actually are.
GROUP_LEGS_WAIST = "legs_waist"
GROUP_SHOULDERS_ELBOWS = "shoulders_elbows"
GROUP_WRIST_YAW = "wrist_yaw"

DEFAULT_MAX_JOINT_DEV_BY_GROUP: dict[str, float] = {
    GROUP_LEGS_WAIST: 0.15,
    GROUP_SHOULDERS_ELBOWS: 0.30,
    GROUP_WRIST_YAW: 0.50,
}


@dataclass
class HandoffConfig:
    stand_hold_s: float = 1.0
    max_joint_dev_by_group: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_MAX_JOINT_DEV_BY_GROUP))
    max_ang_vel: float = 0.3
    walk_distance_m: float = 5.0
    lying_tau_s: float = 0.2
    lying_gravity_z_thresh: float = -0.5


class HandoffStateMachine:
    """Per-env hysteresis state. See module docstring."""

    def __init__(self, num_envs: int, device, cfg: HandoffConfig | None = None):
        self.cfg = cfg or HandoffConfig()
        self.mode_getup = torch.ones(num_envs, dtype=torch.bool, device=device)
        # Anchor (x, y) at the last handoff -- and, until the first handoff, at the first `.step()`
        # call (see the `_anchor_initialized` lazy-init below), NOT zero. Initializing this to zero
        # made pre-handoff "distance" read as the robot's raw world-frame position (e.g. "walked: 3.80m" during get-up, before any handoff happened at
        # all), because it was measured from the world origin, not from the robot's own position.
        self._anchor_initialized = False
        self.anchor_x = torch.zeros(num_envs, device=device)
        self.anchor_y = torch.zeros(num_envs, device=device)
        self.prev_x = torch.zeros(num_envs, device=device)
        self.prev_y = torch.zeros(num_envs, device=device)
        # Cumulative planar arc length walked since the last handoff (reset to 0 at every handoff),
        # separate from `net_displacement` (straight-line distance from the handoff point): a robot
        # that curves or backtracks can have path_length > net_displacement even though both are
        # measured from the same anchor.
        self.path_length = torch.zeros(num_envs, device=device)
        self.lying_ema = torch.full((num_envs,), -1.0, device=device)  # prior: upright
        self.push_done = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.num_handoffs = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.num_falls = torch.zeros(num_envs, dtype=torch.long, device=device)

    def step(
        self,
        dt: float,
        stand_hold_s: torch.Tensor,
        joint_dev_by_group: dict[str, torch.Tensor],
        ang_vel_norm: torch.Tensor,
        gravity_z: torch.Tensor,
        base_x: torch.Tensor,
        base_y: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """One control-tick update. Most args are [num_envs] tensors (or floats broadcastable to
        it); ``joint_dev_by_group`` is a ``{group_name: [num_envs] tensor}`` dict, one L_inf
        deviation per body-region group — every key in
        ``self.cfg.max_joint_dev_by_group`` must be present (a missing group is treated as
        "always failing," never as "no bound", so a caller can't accidentally skip a region).

        Returns a dict of [num_envs] tensors: ``mode_getup`` (bool, apply via ``set_mode``),
        ``fallen`` (bool, the lying detector's current verdict), ``walked_distance`` (float,
        meters since the last handoff), ``reached_distance`` (bool, this env just hit the walk
        target this tick), ``should_push`` (bool, apply the optional push this tick),
        ``pose_ok_by_group`` (``{group: bool tensor}``, this tick's per-group verdict — the
        reusable diagnostic ``play_combined.py --pose_diag`` reports).
        """
        cfg = self.cfg
        if base_y is None:
            base_y = torch.zeros_like(base_x)
        if not self._anchor_initialized:
            # Lazy-init on the first real tick (not in __init__, since __init__ has no position
            # yet): anchor at the robot's actual starting position, not the world origin, so a
            # `--pose_diag`/overlay read *before* the first handoff shows ~0m of drift instead of
            # the robot's raw world coordinate.
            self.anchor_x = base_x.clone()
            self.anchor_y = base_y.clone()
            self.prev_x = base_x.clone()
            self.prev_y = base_y.clone()
            self._anchor_initialized = True

        step_dist = torch.sqrt((base_x - self.prev_x) ** 2 + (base_y - self.prev_y) ** 2)
        self.path_length = self.path_length + step_dist
        self.prev_x, self.prev_y = base_x.clone(), base_y.clone()

        alpha = min(1.0, dt / max(cfg.lying_tau_s, 1e-6))
        self.lying_ema = self.lying_ema + alpha * (gravity_z - self.lying_ema)
        fallen = self.lying_ema > cfg.lying_gravity_z_thresh

        # Snapshot the mode once per tick and derive both transitions from it, so a single
        # `.step()` call can never fire *both* a handoff and an immediate re-fall for the same
        # env (which would otherwise double-count and thrash whenever the gravity EMA and the
        # other gate signals disagree for a few ticks — caught by this module's own self-test,
        # phase 2, before this snapshot was added).
        mode_at_start = self.mode_getup

        pose_ok_by_group: dict[str, torch.Tensor] = {}
        pose_ok = torch.ones_like(stand_hold_s, dtype=torch.bool)
        for group, bound in cfg.max_joint_dev_by_group.items():
            dev = joint_dev_by_group.get(group)
            ok = torch.zeros_like(pose_ok) if dev is None else dev < bound
            pose_ok_by_group[group] = ok
            pose_ok &= ok

        # get-up -> walk. Requiring `~fallen` here too (not just for the reverse direction) is
        # what actually gives this hysteresis: the handoff can never fire while the same
        # low-pass gravity signal used to trigger a re-fall still reads "fallen", so the two
        # transitions are mutually exclusive by construction, not just by the snapshot above.
        can_handoff = (
            mode_at_start
            & (stand_hold_s >= cfg.stand_hold_s)
            & pose_ok
            & (ang_vel_norm < cfg.max_ang_vel)
            & ~fallen
        )
        # Reset the anchor (and the path-length accumulator) at every handoff, so displacement is
        # measured from the robot's position AT HANDOFF. `step_dist` above was still accumulated into
        # `path_length` for this same tick using the *old* anchor/prev position before this reset,
        # which is fine (that last increment belongs to the get-up leg, not the walk leg; it's at
        # most one tick, ~1cm, of noise) and keeps the reset atomic with the mode flip.
        self.anchor_x = torch.where(can_handoff, base_x, self.anchor_x)
        self.anchor_y = torch.where(can_handoff, base_y, self.anchor_y)
        self.path_length = torch.where(can_handoff, torch.zeros_like(self.path_length), self.path_length)
        self.num_handoffs += can_handoff.long()

        # walk -> get-up (fall while walking).
        refall = (~mode_at_start) & fallen
        self.num_falls += refall.long()
        self.push_done = torch.where(refall, torch.zeros_like(self.push_done), self.push_done)

        self.mode_getup = torch.where(can_handoff, torch.zeros_like(mode_at_start), mode_at_start)
        self.mode_getup = torch.where(refall, torch.ones_like(self.mode_getup), self.mode_getup)

        net_displacement = torch.sqrt((base_x - self.anchor_x) ** 2 + (base_y - self.anchor_y) ** 2)
        reached = (~self.mode_getup) & (net_displacement >= cfg.walk_distance_m) & (~self.push_done)
        self.push_done = torch.where(reached, torch.ones_like(self.push_done), self.push_done)

        return {
            "mode_getup": self.mode_getup.clone(),
            "fallen": fallen.clone(),
            "walked_distance": net_displacement.clone(),  # kept name for callers; now handoff-anchored net displacement
            "net_displacement": net_displacement.clone(),
            "path_length": self.path_length.clone(),
            "reached_distance": reached.clone(),
            "should_push": reached.clone(),
            "pose_ok_by_group": pose_ok_by_group,
        }


if __name__ == "__main__":
    # Self-test (no Isaac Lab / GPU needed): python3 scripts/getup/handoff_logic.py
    torch.manual_seed(0)
    dev = "cpu"
    n = 3
    cfg = HandoffConfig(stand_hold_s=1.0, max_ang_vel=0.3, walk_distance_m=5.0, lying_tau_s=0.2)
    sm = HandoffStateMachine(n, dev, cfg)
    dt = 0.02  # 50 Hz policy tick

    def zeros():
        return torch.zeros(n)

    def dev_dict(legs_waist, shoulders_elbows, wrist_yaw):
        return {
            GROUP_LEGS_WAIST: torch.full((n,), float(legs_waist)),
            GROUP_SHOULDERS_ELBOWS: torch.full((n,), float(shoulders_elbows)),
            GROUP_WRIST_YAW: torch.full((n,), float(wrist_yaw)),
        }

    # Phase 1: fallen at start -> must stay in get-up mode even as noise briefly looks "upright".
    out = sm.step(dt, stand_hold_s=zeros(), joint_dev_by_group=dev_dict(1.0, 1.0, 1.0),
                  ang_vel_norm=torch.full((n,), 2.0), gravity_z=torch.tensor([0.9, 0.9, 0.9]), base_x=zeros())
    assert torch.all(out["mode_getup"]), "should still be in get-up mode while falling"

    # Phase 2a (per-group bound regression): legs/waist already tight, but arms/wrist still far
    # out -- must NOT hand off. This is the case a single all-joints bound handled badly
    # (is_standing satisfied, but the scalar bound rejected it); the per-group gate still
    # correctly blocks on wrist_yaw alone.
    out = sm.step(dt, stand_hold_s=torch.full((n,), 2.0), joint_dev_by_group=dev_dict(0.05, 0.20, 0.80),
                  ang_vel_norm=torch.full((n,), 0.05), gravity_z=torch.full((n,), -0.95), base_x=zeros())
    assert torch.all(out["mode_getup"]), "wrist_yaw (0.80 > 0.50 bound) alone must block the handoff"
    assert not bool(out["pose_ok_by_group"][GROUP_WRIST_YAW].any()), "wrist_yaw group should report not-ok"
    assert bool(out["pose_ok_by_group"][GROUP_LEGS_WAIST].all()), "legs_waist group should report ok"

    # Phase 2b: robot settles into a good standing state; ramp all three group deviations down/up
    # (still using different absolute values per group, matching the different per-group bounds).
    stand_hold = torch.full((n,), 2.0)
    dev_legs, dev_shoulders, dev_wrist = torch.full((n,), 0.05), torch.full((n,), 0.20), torch.full((n,), 0.80)
    ang_vel = torch.full((n,), 1.0)
    grav_z = torch.full((n,), -0.95)
    base_x = zeros()
    handed_off_step = [None] * n
    for step in range(120):
        stand_hold = stand_hold + dt
        dev_wrist = torch.clamp(dev_wrist - 0.01, min=0.05)  # only the wrist needs to come down
        ang_vel = torch.clamp(ang_vel - 0.02, min=0.05)
        out = sm.step(dt, stand_hold_s=stand_hold,
                      joint_dev_by_group={GROUP_LEGS_WAIST: dev_legs, GROUP_SHOULDERS_ELBOWS: dev_shoulders, GROUP_WRIST_YAW: dev_wrist},
                      ang_vel_norm=ang_vel, gravity_z=grav_z, base_x=base_x)
        for i in range(n):
            if handed_off_step[i] is None and not bool(out["mode_getup"][i]):
                handed_off_step[i] = step
    assert all(s is not None for s in handed_off_step), f"expected all envs to hand off, got {handed_off_step}"
    assert not bool(sm.mode_getup.any()), "all envs should be walking now"
    print(f"[selftest] handoff steps: {handed_off_step}")

    # Phase 3: walk forward at 1 m/s (dt=0.02s -> 0.02 m/step) until 5 m is reached.
    reached_step = None
    for step in range(400):
        base_x = base_x + 0.02
        out = sm.step(dt, stand_hold_s=stand_hold,
                      joint_dev_by_group={GROUP_LEGS_WAIST: dev_legs, GROUP_SHOULDERS_ELBOWS: dev_shoulders, GROUP_WRIST_YAW: dev_wrist},
                      ang_vel_norm=ang_vel, gravity_z=grav_z, base_x=base_x)
        if bool(out["reached_distance"].any()) and reached_step is None:
            reached_step = step
            assert bool(out["should_push"].all()), "all 3 envs walk at the same rate; should all trigger together"
    assert reached_step is not None, "never reached the 5 m target"
    print(f"[selftest] reached 5 m at step {reached_step} (~{reached_step * dt:.2f}s of walking)")

    # Phase 4: simulate a push -> the robot falls (gravity_z jumps positive and stays there).
    grav_z = torch.full((n,), 0.95)
    fell_step = None
    for step in range(50):
        out = sm.step(dt, stand_hold_s=stand_hold,
                      joint_dev_by_group={GROUP_LEGS_WAIST: dev_legs, GROUP_SHOULDERS_ELBOWS: dev_shoulders, GROUP_WRIST_YAW: dev_wrist},
                      ang_vel_norm=ang_vel, gravity_z=grav_z, base_x=base_x)
        if bool(out["mode_getup"].all()) and fell_step is None:
            fell_step = step
    assert fell_step is not None, "push should have triggered a re-fall back into get-up mode"
    print(f"[selftest] re-triggered get-up {fell_step} steps (~{fell_step * dt:.2f}s) after the push")
    assert int(sm.num_handoffs[0]) == 1 and int(sm.num_falls[0]) == 1

    # Phase 5: a single noisy upright-looking spike must NOT immediately mark the robot fallen
    # (i.e. the EMA actually smooths, it isn't just a same-tick threshold check).
    sm2 = HandoffStateMachine(1, dev, cfg)
    ok_dev = {GROUP_LEGS_WAIST: torch.zeros(1), GROUP_SHOULDERS_ELBOWS: torch.zeros(1), GROUP_WRIST_YAW: torch.zeros(1)}
    out = sm2.step(dt, stand_hold_s=torch.ones(1), joint_dev_by_group=ok_dev, ang_vel_norm=torch.zeros(1),
                   gravity_z=torch.tensor([-1.0]), base_x=torch.zeros(1))
    assert bool(out["mode_getup"][0]) is False  # handed off (all gates satisfied)
    out = sm2.step(dt, stand_hold_s=torch.ones(1), joint_dev_by_group=ok_dev, ang_vel_norm=torch.zeros(1),
                   gravity_z=torch.tensor([0.95]), base_x=torch.zeros(1))
    assert bool(out["fallen"][0]) is False, "one noisy frame should not trip the low-pass fall detector"
    print("[selftest] EMA smoothing confirmed: a single-frame gravity spike does not trigger a fall")

    # Phase 6: a missing group key must be treated as "always failing", never as "unbounded".
    sm3 = HandoffStateMachine(1, dev, cfg)
    out = sm3.step(dt, stand_hold_s=torch.ones(1), joint_dev_by_group={GROUP_LEGS_WAIST: torch.zeros(1)},
                   ang_vel_norm=torch.zeros(1), gravity_z=torch.tensor([-1.0]), base_x=torch.zeros(1))
    assert bool(out["mode_getup"][0]) is True, "a missing group must block the handoff, not be ignored"
    print("[selftest] missing-group-fails-closed confirmed")

    # Phase 7 (anchor regression): anchor must reset to the robot's position AT
    # handoff (not the world origin), and displacement/path-length must be planar (x, y), not
    # x-only. Spawn the robot far from the origin, let it walk in a diagonal (not axis-aligned)
    # line after handoff, and confirm (a) pre-handoff "distance" reads ~0 even though the robot
    # is nowhere near (0, 0), (b) net_displacement matches straight-line planar distance, and
    # (c) path_length (accumulated diagonal steps) exceeds net_displacement once the robot turns
    # partway through, proving the two are tracked independently.
    sm4 = HandoffStateMachine(1, dev, cfg)
    ok_dev1 = {GROUP_LEGS_WAIST: torch.zeros(1), GROUP_SHOULDERS_ELBOWS: torch.zeros(1), GROUP_WRIST_YAW: torch.zeros(1)}
    spawn_x, spawn_y = torch.tensor([37.0]), torch.tensor([-12.0])
    out = sm4.step(dt, stand_hold_s=torch.zeros(1), joint_dev_by_group=ok_dev1, ang_vel_norm=torch.full((1,), 2.0),
                   gravity_z=torch.tensor([0.9]), base_x=spawn_x, base_y=spawn_y)
    assert abs(float(out["net_displacement"][0])) < 1e-6, "pre-handoff distance must read ~0, not the spawn-vs-origin distance"
    # Satisfy every gate and hand off right where the robot spawned.
    out = sm4.step(dt, stand_hold_s=torch.ones(1), joint_dev_by_group=ok_dev1, ang_vel_norm=torch.zeros(1),
                   gravity_z=torch.tensor([-1.0]), base_x=spawn_x, base_y=spawn_y)
    assert bool(out["mode_getup"][0]) is False, "should have handed off at the spawn point"
    # Walk 3m due x, then 4m due y (a 3-4-5 right angle): net_displacement should end at 5m
    # (straight line from the handoff anchor) while path_length ends at 7m (3 + 4, the actual
    # distance traveled along the L-shaped path) -- confirming the two are genuinely different
    # accumulators, not aliases of each other.
    x, y = spawn_x.clone(), spawn_y.clone()
    for _ in range(150):  # 3m at 0.02m/step
        x = x + 0.02
        out = sm4.step(dt, stand_hold_s=torch.ones(1), joint_dev_by_group=ok_dev1, ang_vel_norm=torch.zeros(1),
                       gravity_z=torch.tensor([-1.0]), base_x=x, base_y=y)
    for _ in range(200):  # 4m at 0.02m/step
        y = y + 0.02
        out = sm4.step(dt, stand_hold_s=torch.ones(1), joint_dev_by_group=ok_dev1, ang_vel_norm=torch.zeros(1),
                       gravity_z=torch.tensor([-1.0]), base_x=x, base_y=y)
    assert abs(float(out["net_displacement"][0]) - 5.0) < 1e-3, f"expected 5.0m net displacement, got {float(out['net_displacement'][0])}"
    assert abs(float(out["path_length"][0]) - 7.0) < 1e-3, f"expected 7.0m path length, got {float(out['path_length'][0])}"
    print(f"[selftest] planar anchor/path-length confirmed: net_displacement={float(out['net_displacement'][0]):.3f}m, path_length={float(out['path_length'][0]):.3f}m")

    print("[selftest] ALL CHECKS PASSED")
