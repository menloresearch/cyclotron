# Copyright (c) 2026, Menlo Research.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Get-up episode tracker.

A class-based **termination** term that always returns False. Termination terms always run (before rewards), and the
termination manager's ``reset`` runs after ``extras["log"]`` has been recreated in ``_reset_idx``, so per-episode
metrics written from :meth:`getup_tracker.reset` reach the rsl_rl logger.

Per step it updates ``env.getup_state``: ``prev_max_height``/``max_height_ep``, ``made_progress``, ``stand_timer_s``,
``success`` (latched after ``is_standing`` held 1.0 s), ``time_to_stand_s`` and the thermal proxy EMA. On reset it logs
the finished episodes and then clears the per-episode state (it is the last manager reset in ``_reset_idx``).

Logged keys (``extras["log"]``; every key is written on every reset so it is in the logger's first dict):

- ``GetUp/success_rate``, ``GetUp/max_height``, ``GetUp/arm_sat_{left,right}_s`` (seconds per episode with the left /
  right elbow or wrist at >= 99 % of its torque clip): per-episode tensors (true per-iteration means).
- ``GetUp/success_ema``, ``GetUp/success_assisted``, ``GetUp/success_unassisted``, ``GetUp/time_to_stand_s``,
  ``GetUp/success_<category>``, ``GetUp/share_<category>``: exponentially windowed scalars (~``window`` = 500
  episodes) over the **sampled-action** envs, never NaN.
- ``GetUp/ew_streak_max_s`` (per episode), ``GetUp/ew_streak_<category>`` (windowed): longest contiguous elbow/wrist
  saturation streak; ``GetUp/hold_success_rate`` (stood >= ``curriculum_hold_s``), ``GetUp/lost_after_success``.
- ``GetUp/det_*``: the same over the **deterministic** envs (``getup_state.deterministic``, policy mean), with
  shorter windows: ~``det_window`` = 200 episodes overall / assisted / unassisted and ~``det_category_window`` = 100
  per category. At 4096 envs × 5 % ≈ 205 deterministic envs and ~12 s episodes that is ≈ 10-15 deterministic episodes
  per iteration, so the overall window spans ~15-20 iterations, the unassisted window (20 % of them) ~70 iterations,
  and a rare category (side_*, 8 %) ~100 iterations (the category reweighting runs every 200). ``GetUp/det_episodes_window``
  is the effective episode count of the overall deterministic window.

The windowed rates are stored on the state; the curricula read ``getup_state.curr_*`` = the deterministic rates, per
slot falling back to the sampled rate while that deterministic window is still empty (no deterministic envs yet).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, TerminationTermCfg

from .state import (
    _CATEGORIES,
    current_effort_limits,
    ensure_state,
    is_standing,
    joint_tables,
    pelvis_height,
    torso_tilt,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class _Windowed:
    """GPU exponentially-windowed mean per slot, sync-free: S <- l^k S + sum(w v), N <- l^k N + sum(w), k = sum(w)."""

    def __init__(self, num_slots: int, window: float, device):
        self.lam = 1.0 - 1.0 / window
        self.s = torch.zeros(num_slots, device=device)
        self.n = torch.zeros(num_slots, device=device)

    def add(self, slot: torch.Tensor, value: torch.Tensor, weight: torch.Tensor):
        w = weight.float()
        k = torch.zeros_like(self.n).scatter_add_(0, slot, w)
        v = torch.zeros_like(self.s).scatter_add_(0, slot, value.float() * w)
        decay = torch.pow(torch.full_like(k, self.lam), k)
        self.s = decay * self.s + v
        self.n = decay * self.n + k

    def mean(self) -> torch.Tensor:
        return self.s / self.n.clamp(min=1e-6)


class getup_tracker(ManagerTermBase):
    """Termination term (``time_out=False``) that never terminates; tracks success and logs get-up metrics."""

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        ensure_state(env)
        n, dev = env.num_envs, env.device
        window = float(cfg.params.get("window", 500.0))
        self._false = torch.zeros(n, dtype=torch.bool, device=dev)
        self._cat_snap = torch.zeros(n, dtype=torch.long, device=dev)
        self._assist_snap = torch.zeros(n, dtype=torch.bool, device=dev)
        det_window = float(cfg.params.get("det_window", 200.0))
        det_cat_window = float(cfg.params.get("det_category_window", 100.0))
        self._det_snap = torch.zeros(n, dtype=torch.bool, device=dev)
        # sampled-action envs
        self._cat = _Windowed(len(_CATEGORIES), window, dev)
        self._cat_counts = torch.zeros(len(_CATEGORIES), device=dev)
        self._assist = _Windowed(2, window, dev)  # slot 0: unassisted, 1: assisted
        self._all = _Windowed(1, window, dev)
        self._tts = _Windowed(1, window, dev)
        # deterministic envs: far fewer episodes (5 % of the envs), so shorter windows
        self._d_cat = _Windowed(len(_CATEGORIES), det_cat_window, dev)
        self._d_assist = _Windowed(2, det_window, dev)
        self._d_all = _Windowed(1, det_window, dev)
        self._d_tts = _Windowed(1, det_window, dev)
        self._use_hold = float(cfg.params.get("curriculum_hold_s", 0.0)) > float(cfg.params.get("hold_time_s", 1.0))
        self._ew_cat = _Windowed(len(_CATEGORIES), window, dev)  # per-category mean of the per-episode max e/w streak
        self._tab = joint_tables(env)
        robot = env.scene["robot"]
        self._arm_ids = [
            robot.find_joints(["left_elbow_joint", "left_wrist_yaw_joint"], preserve_order=True)[0],
            robot.find_joints(["right_elbow_joint", "right_wrist_yaw_joint"], preserve_order=True)[0],
        ]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        hold_time_s: float = 1.0,
        progress_height: float = 0.2,
        thermal_time_constant_s: float = 2.0,
        window: float = 500.0,
        det_window: float = 200.0,
        det_category_window: float = 100.0,
        curriculum_hold_s: float = 0.0,
        lost_height: float = 0.5,
        lost_tilt: float = 0.35,
    ) -> torch.Tensor:
        st = ensure_state(env)
        active = st.policy_active
        h = pelvis_height(env)
        # heights
        st.prev_max_height.copy_(st.max_height_ep)
        st.max_height_ep.copy_(torch.where(active, torch.maximum(st.max_height_ep, h), st.max_height_ep))
        # standing / success
        standing = is_standing(env) & active
        st.stand_timer_s.copy_(torch.where(standing, st.stand_timer_s + env.step_dt, torch.zeros_like(h)))
        newly = (st.stand_timer_s >= hold_time_s - 1e-6) & ~st.success
        st.success |= newly
        t_ctrl = (st.step - st.control_start_step).clamp(min=0).float() * env.step_dt
        st.time_to_stand_s.copy_(torch.where(newly, t_ctrl - st.stand_timer_s, st.time_to_stand_s))
        st.made_progress |= active & ((h > st.start_height + progress_height) | st.success)
        # longer hold (curricula judged on it when curriculum_hold_s > 0) and loss of standing after success
        st.success_hold |= st.stand_timer_s >= max(hold_time_s, curriculum_hold_s) - 1e-6
        fallen = (h < lost_height) | (torso_tilt(env) > lost_tilt)
        st.lost_now.copy_(st.success & fallen & ~st.lost_after_success & active)
        st.lost_after_success |= st.lost_now
        # thermal proxy: EMA of (tau / tau_rated)^2, time constant ~2 s
        asset = env.scene["robot"]
        e = torch.square(asset.data.applied_torque / self._tab.rated.clamp(min=1e-6))
        alpha = min(1.0, env.step_dt / thermal_time_constant_s)
        st.thermal.add_(alpha * (e - st.thermal))
        # per-arm saturation time (elbow or wrist at >= 99 % of its current clip), to expose left/right asymmetry
        lim = current_effort_limits(env)
        tau_hat = asset.data.applied_torque.abs() / lim.clamp(min=1e-6)
        any_sat = torch.zeros_like(active)
        for side, ids in enumerate(self._arm_ids):
            sat = (tau_hat[:, ids] >= 0.99).any(dim=1) & active
            st.arm_sat_s[:, side] += sat.float() * env.step_dt
            any_sat |= sat
        # contiguous elbow/wrist saturation streak (safety check: never > 0.2 s)
        st.ew_streak_s.copy_(torch.where(any_sat, st.ew_streak_s + env.step_dt, torch.zeros_like(st.ew_streak_s)))
        st.ew_streak_max_s.copy_(torch.maximum(st.ew_streak_max_s, st.ew_streak_s))
        # snapshots of fields that other reset hooks overwrite before our reset() runs
        self._cat_snap.copy_(st.category)
        self._assist_snap.copy_(st.assist_enabled)
        self._det_snap.copy_(st.deterministic)
        return self._false

    def reset(self, env_ids=None):
        # Sync-free: no boolean indexing / .any() / .item(). Episodes whose control never started (only the
        # initial env.reset(), which the runner does not log, or a termination during a limp phase) get weight 0 in the
        # windows and count as failures in the per-episode tensors.
        env = self._env
        st = ensure_state(env)
        if env_ids is None or isinstance(env_ids, slice):
            env_ids = torch.arange(env.num_envs, device=env.device)
        env_ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
        started = st.control_started[env_ids]
        det = self._det_snap[env_ids]
        w_s = started & ~det  # sampled-action episodes
        w_d = started & det  # deterministic episodes
        succ = st.success[env_ids] & started
        hold = st.success_hold[env_ids] & started
        cat = self._cat_snap[env_ids]
        # the deterministic (curriculum-driving) windows judge the longer hold when configured (StageB2: 3 s)
        succ_d = hold if self._use_hold else succ
        asl = self._assist_snap[env_ids].long()
        zero_slot = torch.zeros_like(cat)
        tts = st.time_to_stand_s[env_ids]
        # sampled
        self._cat.add(cat, succ, w_s)
        n_done = started.sum()
        self._cat_counts = self._cat_counts * torch.pow(torch.tensor(self._cat.lam, device=env.device), n_done.float())
        self._cat_counts = self._cat_counts.scatter_add(0, cat, started.float())
        self._assist.add(asl, succ, w_s)
        self._all.add(zero_slot, succ, w_s)
        self._tts.add(zero_slot, tts, succ & w_s)
        # deterministic
        self._d_cat.add(cat, succ_d, w_d)
        self._d_assist.add(asl, succ_d, w_d)
        self._d_all.add(zero_slot, succ_d, w_d)
        self._d_tts.add(zero_slot, tts, succ & w_d)
        self._ew_cat.add(cat, st.ew_streak_max_s[env_ids], started)
        st.episodes_finished += n_done
        log = env.extras.setdefault("log", {})
        # per-episode tensors over all finished episodes of this call (sampled and deterministic; the deterministic
        # ones are ~5 % and are reported separately by the det_* windows)
        log["GetUp/success_rate"] = succ.float()
        log["GetUp/max_height"] = st.max_height_ep[env_ids]
        log["GetUp/arm_sat_left_s"] = st.arm_sat_s[env_ids, 0]
        log["GetUp/arm_sat_right_s"] = st.arm_sat_s[env_ids, 1]
        log["GetUp/ew_streak_max_s"] = st.ew_streak_max_s[env_ids]
        log["GetUp/hold_success_rate"] = hold.float()
        log["GetUp/lost_after_success"] = (st.lost_after_success[env_ids] & started).float()
        ew = self._ew_cat.mean()
        for i, c in enumerate(_CATEGORIES):
            log[f"GetUp/ew_streak_{c}"] = ew[i]
        # windowed scalars (never NaN)
        rates, d_rates = self._cat.mean(), self._d_cat.mean()
        share = self._cat_counts / self._cat_counts.sum().clamp(min=1e-6)
        a, d_a = self._assist.mean(), self._d_assist.mean()
        for i, c in enumerate(_CATEGORIES):
            log[f"GetUp/success_{c}"] = rates[i]
            log[f"GetUp/det_success_{c}"] = d_rates[i]
            log[f"GetUp/share_{c}"] = share[i]
        log["GetUp/success_unassisted"] = a[0]
        log["GetUp/success_assisted"] = a[1]
        log["GetUp/success_ema"] = self._all.mean()[0]
        log["GetUp/time_to_stand_s"] = self._tts.mean()[0]
        log["GetUp/det_success_unassisted"] = d_a[0]
        log["GetUp/det_success_assisted"] = d_a[1]
        log["GetUp/det_success_ema"] = self._d_all.mean()[0]
        log["GetUp/det_time_to_stand_s"] = self._d_tts.mean()[0]
        log["GetUp/det_episodes_window"] = self._d_all.n[0]
        st.success_by_category = rates
        st.success_unassisted_ema = a[0]
        st.success_ema = self._all.mean()[0]
        st.det_success_by_category = d_rates
        st.det_success_unassisted_ema = d_a[0]
        st.det_success_ema = self._d_all.mean()[0]
        # curriculum-driving metrics: deterministic, per-slot fallback to sampled while a det window is empty
        eps = 1e-3
        st.curr_success_by_category = torch.where(self._d_cat.n > eps, d_rates, rates)
        st.curr_success_unassisted_ema = torch.where(self._d_assist.n[0] > eps, d_a[0], a[0])
        st.curr_success_ema = torch.where(self._d_all.n[0] > eps, st.det_success_ema, st.success_ema)
        # effective episode counts of the windows the curricula read (curricula act only on filled windows)
        st.curr_category_n = torch.where(self._d_cat.n > eps, self._d_cat.n, self._cat.n)
        st.curr_unassisted_n = torch.where(self._d_assist.n[0] > eps, self._d_assist.n[0], self._assist.n[0])
        # clear per-episode state last
        st.reset(env_ids)
        env._getup_reset_count = getattr(env, "_getup_reset_count", 0) + 1  # invalidates cached terrain heights
