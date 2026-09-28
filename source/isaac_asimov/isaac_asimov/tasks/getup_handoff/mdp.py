# Copyright (c) 2026, Menlo Research.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""MDP terms for the get-up <-> walking handoff demo.

This module holds exactly one action term, ``HandoffJointPositionAction``, plus the one
observation helper it needs beyond what already exists (the walking path's ``actions`` term; the
get-up path reuses ``getup.mdp.filtered_action`` unmodified, see below). It exists because the
combined env (``Asimov1-GetUp-Handoff-Play-v0``) must carry *two* action paths on
the *same* 23 joints — the get-up path (``getup.mdp.FilteredRelativeJointPositionAction``'s
formula, ``q_meas + beta * s_j * LPF(clip(a))``) and the walking path (plain absolute
``JointPositionActionCfg`` at scale 0.25, ``velocity_env_cfg.ActionsCfg``) — and switch between
them per env, without a jump.

Design (see docs/getup/HANDOFF.md "Clean switching" for the full rationale):

- One action term, ``joint_pos``, with ``action_dim = 2 * num_joints``. The play script
  concatenates ``[getup_raw_action, walk_raw_action]`` every step, i.e. it runs *both* policies'
  forward passes every step (both are stateless MLPs, so this is cheap) and hands both raw
  outputs to the env, i.e. ``getup(obs["getup"])`` and ``walk(obs["walk"])`` are computed
  unconditionally every step.
- The term updates *both* internal states every step, regardless of which one is "live":
  the get-up low-pass filter is fed the get-up raw action every step (masked to zero while
  ``env.getup_state.policy_active`` is False, exactly like ``getup.mdp.FilteredRelativeJointPositionAction``,
  so a limp `mid_fall` window behaves the same way here as it does in the standalone get-up task),
  and the walk path's own last-raw-action buffer is fed the walk raw action every step. Only the
  *commanded joint target* is selected per env via ``mode`` (a bool tensor owned by the play
  script via :meth:`HandoffJointPositionAction.set_mode`).
- Consequence: at the instant ``mode`` flips for an env, the get-up LPF state is never stale (it
  has been tracking the get-up policy's own output continuously) and the walking group's
  ``actions`` observation is never a discontinuous jump from zero (it has been tracking the
  walking policy's own output continuously too). The hysteresis gate in the handoff
  condition (stand held >= 1s, ``|q-q_default|_inf < 0.15 rad``, ``|omega| < 0.3 rad/s``) also
  means the two paths' *targets* nearly agree by the time the mode actually flips, since the robot
  is already close to the walking default pose and nearly still.
- On the robot only one ONNX model is assumed to run at a time, so this
  "run both continuously" trick is a sim-side convenience. The state-machine section of
  docs/getup/FIRMWARE_SPEC.md covers how a real GETUP -> STAND/MOVE transition seeds the walking
  LPF instead.
- ``s_j`` (the get-up path's per-joint relative-action scale) is read from
  ``getup.mdp.state.joint_tables(env)`` — the get-up task's own nominal tau_max/Kp table — rather than
  re-derived from the asset here, so this term can never silently drift from the trained policy's
  actual scale.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.actions.actions_cfg import JointActionCfg
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass

from isaac_asimov.tasks.getup.mdp.limp import update_limp_schedule
from isaac_asimov.tasks.getup.mdp.state import ensure_state, joint_tables, pelvis_height

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class HandoffJointPositionAction(ActionTerm):
    """See module docstring. ``action_dim = 2 * num_joints``: ``[:, :J]`` is the get-up raw
    action, ``[:, J:]`` is the walking raw action. Both are always processed; ``mode`` (set
    externally via :meth:`set_mode`) selects which one is written to the articulation.
    """

    cfg: HandoffJointPositionActionCfg

    def __init__(self, cfg: HandoffJointPositionActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self._joint_ids, self._joint_names = self._asset.find_joints(
            cfg.joint_names, preserve_order=cfg.preserve_order
        )
        self._num_joints = len(self._joint_ids)
        ids_t = self._joint_ids
        if isinstance(ids_t, slice):
            ids_t = torch.arange(self._asset.num_joints, device=self.device)
        self._joint_ids_t = torch.as_tensor(ids_t, device=self.device, dtype=torch.long)

        dev = self.device
        n = self.num_envs
        j = self._num_joints

        self._raw_actions = torch.zeros(n, 2 * j, device=dev)
        self._processed_actions = torch.zeros(n, j, device=dev)

        # get-up path state (mirrors getup.mdp.actions.FilteredRelativeJointPositionAction).
        self._lpf_state = torch.zeros(n, j, device=dev)
        # clipped-action history, purely so the get-up task's own `action_rate_clipped_l2` /
        # `action_second_diff_l2` reward terms (tasks/getup/mdp/rewards.py, inherited unchanged
        # by the combined env's RewardsCfg) find the same `clipped_actions` / `prev_clipped_actions`
        # / `prev_prev_clipped_actions` properties on this term that they'd find on
        # `FilteredRelativeJointPositionAction` — without this they raise AttributeError on the
        # very first `env.step()`, since the reward manager always runs every declared term.
        self._clipped = torch.zeros(n, j, device=dev)
        self._prev_clipped = torch.zeros(n, j, device=dev)
        self._prev_prev_clipped = torch.zeros(n, j, device=dev)
        tab = joint_tables(env)
        self._tau_nom = tab.tau_max[self._joint_ids_t]
        self._kp_nom = tab.kp[self._joint_ids_t].clamp_min(1e-6)
        # Mirrors getup.mdp.actions.FilteredRelativeJointPositionAction's own attribute names
        # exactly (joint_scale / bound_scale / joint_scale_live / beta / set_bound_scale /
        # contract()), not just its formula: a startup event term (`apply_play_effort_scale`)
        # calls `action_manager.get_term("joint_pos").set_bound_scale(...)` unconditionally on
        # every task that registers a "joint_pos" term, including this combined one. Without a
        # compatible `set_bound_scale` this raises `AttributeError` at env *construction* time
        # (event manager's `startup` mode runs during `gym.make`, before any policy is even
        # loaded).
        self.joint_scale = cfg.getup_scale_torque_factor * self._tau_nom / self._kp_nom  # nominal, [J]
        self.bound_scale = torch.ones_like(self.joint_scale)
        self.joint_scale_live = self.joint_scale.clone()
        self.beta = float(cfg.getup_beta)

        # walking path state (last raw action fed to the walk half; used for its own obs term).
        self._last_walk_raw = torch.zeros(n, j, device=dev)
        self._default_joint_pos = self._asset.data.default_joint_pos[:, self._joint_ids].clone()

        # True = get-up path drives the joints for that env; False = walking path does.
        self._mode = torch.ones(n, dtype=torch.bool, device=dev)

        ensure_state(env)

    # --- Isaac Lab ActionTerm API -----------------------------------------------------------

    @property
    def action_dim(self) -> int:
        return 2 * self._num_joints

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        env = self._env
        st = ensure_state(env)
        # Mirror the standalone get-up action term (`FilteredRelativeJointPositionAction.
        # process_actions`, getup/mdp/actions.py): it calls `update_limp_schedule(env)` every step
        # and writes the result into `getup_state.policy_active`. That call does *two* things:
        # (1) recomputes `policy_active` for this tick, and (2) as a side effect, releases each
        # env's actuators from limp (Kp 0, Kd 0.5) back to nominal PD gains the instant
        # `step >= control_start_step`. Without it, `mid_fall` (the only category that starts with
        # `policy_active=False`, for a limp reset window) would never have its actuators released:
        # they would stay limp (Kp=0) for the whole episode and the get-up policy's joint targets
        # would have no effect. `getup_state.step` (this env's own step counter, distinct from
        # `episode_length_buf`) is incremented *only* inside the action term's `process_actions`,
        # so this term must do that too, or `update_limp_schedule`'s `step >= control_start_step`
        # check would never advance. It must run at the top of `process_actions`, before anything
        # reads `policy_active` this tick.
        # The `newly`/`control_started`/height-buffer bookkeeping is copied verbatim from
        # `FilteredRelativeJointPositionAction.process_actions` for the same reason: those buffers
        # (`start_height`, `max_height_ep`, `prev_max_height`, `assist_height0`) are read by the
        # get-up task's own reward/observation terms, inherited unchanged by this combined env.
        st.policy_active[:] = update_limp_schedule(env)
        newly = st.policy_active & ~st.control_started
        h = pelvis_height(env)
        for buf in (st.start_height, st.max_height_ep, st.prev_max_height, st.assist_height0):
            buf.copy_(torch.where(newly, h, buf))
        st.control_started |= newly
        st.step += 1

        self._raw_actions[:] = actions
        j = self._num_joints
        a_getup_raw = actions[:, :j]
        a_walk_raw = actions[:, j:]

        # get-up path: LPF update, masked by policy_active exactly like the standalone get-up
        # task's own action term (a `mid_fall` limp window before control starts).
        active = self._policy_active_mask()
        a_clipped = torch.clamp(a_getup_raw, -1.0, 1.0)
        if self.cfg.getup_use_lpf:
            alpha = self.cfg.getup_lpf_alpha
            new_lpf = alpha * a_clipped + (1.0 - alpha) * self._lpf_state
        else:
            new_lpf = a_clipped
        self._lpf_state = torch.where(active, new_lpf, torch.zeros_like(new_lpf))
        self._prev_prev_clipped[:] = self._prev_clipped
        self._prev_clipped[:] = self._clipped
        self._clipped[:] = torch.where(active, a_clipped, torch.zeros_like(a_clipped))

        # walking path: always update the last-raw-action buffer (this is its `actions` obs).
        self._last_walk_raw[:] = a_walk_raw

    def apply_actions(self) -> None:
        # re-read q every physics substep, matching FilteredRelativeJointPositionAction's
        # `relative_per_substep=True` default: a hard bound on the commanded P-torque. Uses the
        # *live* `self.beta`/`self.joint_scale_live` (settable via `set_bound_scale` / direct
        # assignment, like the real term), not the frozen cfg value, so a curriculum or startup
        # event that mutates them takes effect here exactly as it would on the standalone term.
        q_meas = self._asset.data.joint_pos[:, self._joint_ids_t]
        target_getup = q_meas + self.beta * self.joint_scale_live * self._lpf_state
        target_walk = self._default_joint_pos + self.cfg.walk_scale * self._last_walk_raw
        self._processed_actions = torch.where(self._mode.unsqueeze(-1), target_getup, target_walk)
        self._asset.set_joint_position_target(self._processed_actions, joint_ids=self._joint_ids_t)

    def set_bound_scale(self, scale: float | dict[str, float] | torch.Tensor) -> None:
        """Matches ``FilteredRelativeJointPositionAction.set_bound_scale`` exactly (same accepted
        formats, same semantics): sets the uniform per-joint bound scale and recomputes
        ``joint_scale_live``. Called by the `apply_play_effort_scale` startup event and by
        `getup.mdp.curriculums.effort_beta_schedule` during training; this term must accept the
        same calls as the standalone get-up action term since it replaces it in the combined env."""
        if isinstance(scale, torch.Tensor):
            self.bound_scale[:] = scale.to(self.device)
        else:
            from isaac_asimov.tasks.getup.mdp.state import per_joint_values

            self.bound_scale[:] = torch.tensor(per_joint_values(scale, self._joint_names), device=self.device)
        self.joint_scale_live[:] = self.joint_scale * self.bound_scale

    def contract(self) -> dict:
        """The get-up path's live action contract, same shape as
        ``FilteredRelativeJointPositionAction.contract()`` -- lets a consumer (e.g.
        `export_onnx.py`, if ever pointed at the handoff task) cross-check this term exactly like
        the standalone one."""
        return {
            "joint_names": list(self._joint_names),
            "target": "q_meas + beta * s_j * LPF(clip(a_getup, -1, 1))  [get-up path only]",
            "s_j": [float(x) for x in self.joint_scale_live.tolist()],
            "beta": float(self.beta),
            "bound_scale": [float(x) for x in self.bound_scale.tolist()],
            "scale_torque_factor": float(self.cfg.getup_scale_torque_factor),
            "tau_max_nominal": [float(x) for x in self._tau_nom.tolist()],
            "kp_nominal": [float(x) for x in self._kp_nom.tolist()],
            "use_lpf": bool(self.cfg.getup_use_lpf),
            "lpf_alpha": float(self.cfg.getup_lpf_alpha),
            "relative_per_substep": True,
        }

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0
        self._lpf_state[env_ids] = 0.0
        self._clipped[env_ids] = 0.0
        self._prev_clipped[env_ids] = 0.0
        self._prev_prev_clipped[env_ids] = 0.0
        self._last_walk_raw[env_ids] = 0.0
        # A reset always starts a fresh fallen episode in this demo; get-up mode drives first.
        self._mode[env_ids] = True

    # --- Handoff-specific API ---------------------------------------------------------------

    @property
    def filtered_actions(self) -> torch.Tensor:
        """The get-up path's LPF state. This is what ``getup.mdp.filtered_action`` (the get-up
        ``policy``/``critic`` groups' ``actions`` observation) reads, regardless
        of which path currently drives the joints — this term's property name matches
        ``FilteredRelativeJointPositionAction.filtered_actions`` on purpose so that obs function
        is reusable unmodified."""
        return self._lpf_state

    @property
    def clipped_actions(self) -> torch.Tensor:
        """Get-up path's raw action clipped to [-1, 1] this step (see class docstring)."""
        return self._clipped

    @property
    def prev_clipped_actions(self) -> torch.Tensor:
        return self._prev_clipped

    @property
    def prev_prev_clipped_actions(self) -> torch.Tensor:
        return self._prev_prev_clipped

    @property
    def last_walk_raw_action(self) -> torch.Tensor:
        """The walking path's last raw (pre-scale) action, i.e. the walking ``policy`` group's
        ``actions`` observation, matching ``locomotion.mdp.last_action`` semantics."""
        return self._last_walk_raw

    @property
    def mode(self) -> torch.Tensor:
        """Bool [num_envs]: True where the get-up path is currently commanding the joints."""
        return self._mode

    def set_mode(self, mode: torch.Tensor) -> None:
        """Set the per-env active path. Called by the play script's hysteresis state machine,
        never by a reward/observation term (this term has no opinion on *when* to switch)."""
        self._mode[:] = mode.to(dtype=torch.bool, device=self.device)

    def _policy_active_mask(self) -> torch.Tensor:
        """[N, 1] bool: mirrors ``getup_state.policy_active`` (True outside any limp window).
        Defaults to all-True if ``getup_state`` isn't attached yet (first construction step)."""
        st = getattr(self._env, "getup_state", None)
        if st is None:
            return torch.ones(self.num_envs, 1, dtype=torch.bool, device=self.device)
        return st.policy_active.unsqueeze(-1)


@configclass
class HandoffJointPositionActionCfg(JointActionCfg):
    """Config for :class:`HandoffJointPositionAction`. See module docstring for the design.

    ``joint_names`` must resolve to all 23 joints in ``ASIMOV_1_JOINT_NAMES`` order
    (``preserve_order=True``), matching both the get-up and the walking action term.
    """

    class_type: type[ActionTerm] = HandoffJointPositionAction

    # get-up path (mirrors getup.mdp.FilteredRelativeJointPositionActionCfg).
    getup_beta: float = 1.0
    """Curriculum bound on the relative-action excursion. Copy the trained value from the get-up
    task's own action term at env-construction time; do not hand-tune this file independently."""
    getup_lpf_alpha: float = 0.557
    """One-pole low-pass coefficient: 10 Hz cutoff at the 50 Hz policy rate, alpha = dt/(RC+dt)
    (`getup/getup_env_cfg.py` sets this same value on the standalone get-up task's own action
    term). This default is only used if `handoff_env_cfg.py`'s copy-from-parent logic can't find
    the parent's own value; in normal use the live value always comes from
    `Asimov1GetUpEnvCfg_PLAY`'s configured action term, never from this literal, so this file can't
    silently drift from the get-up task's configuration."""
    getup_use_lpf: bool = True
    getup_scale_torque_factor: float = 1.1
    """s_j = getup_scale_torque_factor * tau_max_j / Kp_j."""

    # walking path (mirrors locomotion.velocity_env_cfg.ActionsCfg.joint_pos).
    walk_scale: float = 0.25
    walk_use_default_offset: bool = True


# --- Observation helpers -----------------------------------------------------------------------
#
# The get-up `policy`/`critic` groups' `actions` term needs no new function: `getup.mdp.filtered_action`
# (tasks/getup/mdp/observations.py) already just does `env.action_manager.get_term(action_name).filtered_actions`,
# which this term exposes directly. Only the walking path's `actions` term is new.


def walk_last_raw_action(env: ManagerBasedEnv, action_name: str = "joint_pos") -> torch.Tensor:
    """Walking ``policy`` group's ``actions`` term: the (always-warm) walking raw action."""
    term: HandoffJointPositionAction = env.action_manager.get_term(action_name)
    return term.last_walk_raw_action.clone()
