# Copyright (c) 2026, Menlo Research.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Get-up action term (relative joint-position targets with a torque-derived per-joint scale and an action LPF).

``target = q_meas + beta * s_j * LPF(clip(a, -1, 1))`` with ``s_j = 1.1 * tau_max_j * bound_scale_j / Kp_j`` (nominal
tau_max and Kp). ``bound_scale`` is the uniform curriculum/stage effort scale; per-env motor-strength DR and eval stress
tests change only the actuator torque clip, so DR teaches robustness to weak motors.
:meth:`FilteredRelativeJointPositionAction.contract` returns the live contract.

- The LPF is a one-pole filter ``y <- y + alpha (x - y)``, alpha = 0.557 (design choice: 10 Hz cutoff at the 50 Hz
  policy rate), applied once per policy step by default (``lpf_per_substep=False``). The target uses the filtered
  action. The ``actions`` observation returns the filtered action (the previous-action
  input at deploy time).
- ``q_meas`` is re-read every physics substep by default (Isaac Lab ``RelativeJointPositionAction`` semantics, which is
  what makes ``Kp * s_j * beta`` a hard bound on the P torque). ``relative_per_substep=False`` freezes it per policy
  step instead.
- While ``getup_state.policy_active`` is False (limp phase of ``mid_fall``) the action is ignored: the filter state is
  held at zero and the target equals the measured position.

This term also owns the per-env step counter and the control-start bookkeeping in ``env.getup_state``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.actions import JointAction, JointActionCfg
from isaaclab.utils import configclass

from .state import ensure_state, joint_tables, pelvis_height, per_joint_values

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class FilteredRelativeJointPositionAction(JointAction):
    """Relative joint-position action with a torque-derived per-joint scale, curriculum bound and action LPF."""

    cfg: FilteredRelativeJointPositionActionCfg

    def __init__(self, cfg: FilteredRelativeJointPositionActionCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        tab = joint_tables(env)
        ids = self._joint_ids
        if isinstance(ids, slice):
            ids = torch.arange(self._asset.num_joints, device=self.device)
        ids = torch.as_tensor(ids, device=self.device, dtype=torch.long)
        self._joint_ids_t = ids
        # s_j = factor * tau_max,nominal * bound_scale_j / Kp,nominal. `bound_scale` is the uniform
        # curriculum/stage effort scale (Stage 1: x1.2 on hips/knees/shoulders, then 1.0, Stage B 0.9), set with
        # `set_bound_scale`. It does NOT follow per-env motor-strength DR or eval stress tests, which change only the
        # actuator torque clip (on hardware s_j is a fixed constant from the ONNX metadata; a weak motor saturates).
        self._kp_nom = tab.kp[ids].clamp(min=1e-6)
        self._tau_nom = tab.tau_max[ids]
        self.joint_scale = cfg.scale_torque_factor * self._tau_nom / self._kp_nom  # nominal (bound_scale 1)
        self.bound_scale = torch.ones_like(self.joint_scale)
        self.joint_scale_live = self.joint_scale.clone()
        self.beta = float(cfg.beta)
        self._clip_val = float(cfg.action_clip)
        self._filtered = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._prev_clipped = torch.zeros_like(self._filtered)
        self._prev_prev_clipped = torch.zeros_like(self._filtered)
        self._clipped = torch.zeros_like(self._filtered)
        self._delta = torch.zeros_like(self._filtered)
        self._q_hold = torch.zeros_like(self._filtered)
        self._limp_schedule = _resolve_limp_schedule()
        ensure_state(env)

    # -- properties used by observations / rewards ---------------------------------------------------------------

    @property
    def filtered_actions(self) -> torch.Tensor:
        """Filtered, clipped action in [-1, 1] (the previous-action input at deploy time)."""
        return self._filtered

    @property
    def clipped_actions(self) -> torch.Tensor:
        """Raw policy action clipped to [-1, 1] (this step)."""
        return self._clipped

    @property
    def prev_clipped_actions(self) -> torch.Tensor:
        return self._prev_clipped

    @property
    def prev_prev_clipped_actions(self) -> torch.Tensor:
        return self._prev_prev_clipped

    # -- operations --------------------------------------------------------------------------------------------------

    def process_actions(self, actions: torch.Tensor):
        env = self._env
        st = ensure_state(env)
        # control schedule (own counter; never episode_length_buf). update_limp_schedule releases limp envs.
        st.policy_active[:] = self._limp_schedule(env)
        newly = st.policy_active & ~st.control_started
        # control-start bookkeeping, sync-free (masked writes instead of `if torch.any(newly)`)
        h = pelvis_height(env)
        for buf in (st.start_height, st.max_height_ep, st.prev_max_height, st.assist_height0):
            buf.copy_(torch.where(newly, h, buf))
        st.control_started |= newly
        st.step += 1

        active = st.policy_active.unsqueeze(1)
        self._raw_actions[:] = actions
        clipped = torch.clamp(actions, -self._clip_val, self._clip_val)
        clipped = torch.where(active, clipped, torch.zeros_like(clipped))
        self._prev_prev_clipped[:] = self._prev_clipped
        self._prev_clipped[:] = self._clipped
        self._clipped[:] = clipped
        # keep the action history equal to the current action on the first active step (no fake jump)
        nw = newly.unsqueeze(1)
        self._prev_clipped.copy_(torch.where(nw, clipped, self._prev_clipped))
        self._prev_prev_clipped.copy_(torch.where(nw, clipped, self._prev_prev_clipped))
        if not self.cfg.lpf_per_substep:
            if self.cfg.use_lpf:
                self._filtered[:] = self._filtered + self.cfg.lpf_alpha * (clipped - self._filtered)
            else:
                self._filtered[:] = clipped
            self._filtered[:] = torch.where(active, self._filtered, torch.zeros_like(self._filtered))
        self._q_hold[:] = self._asset.data.joint_pos[:, self._joint_ids_t]
        self._update_delta()

    def set_bound_scale(self, scale: float | dict[str, float] | torch.Tensor) -> None:
        """Set the uniform per-joint bound scale (float, ``{joint regex: scale}`` with 1.0 for unmatched joints, or a
        tensor [J] in this term's joint order) and recompute ``joint_scale_live``."""
        if isinstance(scale, torch.Tensor):
            self.bound_scale[:] = scale.to(self.device)
        else:
            self.bound_scale[:] = torch.tensor(per_joint_values(scale, self._joint_names), device=self.device)
        self.joint_scale_live[:] = self.joint_scale * self.bound_scale

    def contract(self) -> dict:
        """The live action contract: what export / firmware must reproduce."""
        return {
            "joint_names": list(self._joint_names),
            "target": "q_meas + beta * s_j * LPF(clip(a, -action_clip, action_clip))",
            "s_j": [float(x) for x in self.joint_scale_live.tolist()],
            "beta": float(self.beta),
            "bound_scale": [float(x) for x in self.bound_scale.tolist()],
            "scale_torque_factor": float(self.cfg.scale_torque_factor),
            "tau_max_nominal": [float(x) for x in self._tau_nom.tolist()],
            "kp_nominal": [float(x) for x in self._kp_nom.tolist()],
            "action_clip": float(self.cfg.action_clip),
            "use_lpf": bool(self.cfg.use_lpf),
            "lpf_alpha": float(self.cfg.lpf_alpha),
            "lpf_per_substep": bool(self.cfg.lpf_per_substep),
            "relative_per_substep": bool(self.cfg.relative_per_substep),
        }

    def _update_delta(self):
        self._delta[:] = self.beta * self.joint_scale_live * self._filtered
        self._processed_actions = self._delta

    def apply_actions(self):
        if self.cfg.lpf_per_substep:
            active = self._env.getup_state.policy_active.unsqueeze(1)
            if self.cfg.use_lpf:
                self._filtered[:] = self._filtered + self.cfg.lpf_alpha * (self._clipped - self._filtered)
            else:
                self._filtered[:] = self._clipped
            self._filtered[:] = torch.where(active, self._filtered, torch.zeros_like(self._filtered))
            self._update_delta()
        if self.cfg.relative_per_substep:
            q = self._asset.data.joint_pos[:, self._joint_ids_t]
        else:
            q = self._q_hold
        self._asset.set_joint_position_target(q + self._delta, joint_ids=self._joint_ids_t)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0
        self._filtered[env_ids] = 0.0
        self._clipped[env_ids] = 0.0
        self._prev_clipped[env_ids] = 0.0
        self._prev_prev_clipped[env_ids] = 0.0
        self._delta[env_ids] = 0.0


@configclass
class FilteredRelativeJointPositionActionCfg(JointActionCfg):
    """Configuration of :class:`FilteredRelativeJointPositionAction`.

    ``scale`` / ``offset`` / ``clip`` of the base cfg are ignored; the per-joint scale comes from the asset.
    """

    class_type: type = FilteredRelativeJointPositionAction

    scale_torque_factor: float = 1.1
    """s_j = scale_torque_factor * effort_limit_j (live, per env) / Kp_j (nominal)."""
    beta: float = 1.0
    """Global bound multiplier; the curriculum writes the live value to the term's ``beta`` attribute."""
    action_clip: float = 1.0
    """Raw actions are clipped to [-action_clip, action_clip] before filtering."""
    use_lpf: bool = True
    """One-pole low-pass filter on the clipped action (10 Hz at the 50 Hz policy rate)."""
    lpf_alpha: float = 0.557
    """Filter coefficient: 10 Hz one-pole at the 50 Hz policy rate, dt/(RC+dt) with RC = 1/(2*pi*10)."""
    lpf_per_substep: bool = False
    """Apply the filter every physics substep (200 Hz) instead of once per policy step (50 Hz); alpha must then be
    re-derived for dt = 5 ms (0.239)."""
    relative_per_substep: bool = True
    """Re-read q_meas every physics substep (hard P-torque bound) instead of once per policy step."""


# --- limp schedule dependency -----------------------------------------------------------------------------------------


def _resolve_limp_schedule():
    """``limp.update_limp_schedule`` (releases limp envs, returns ``step >= control_start_step``).

    Fallback if it is missing or a stub: the same predicate, without touching actuator gains.
    """
    try:
        from .limp import update_limp_schedule
    except ImportError:
        update_limp_schedule = None

    def _call(env):
        if update_limp_schedule is not None:
            try:
                return update_limp_schedule(env)
            except NotImplementedError:
                pass
        st = env.getup_state
        return st.step >= st.control_start_step

    return _call
