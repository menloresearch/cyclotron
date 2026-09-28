# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""The two frozen action contracts.

* :class:`WalkingAction` -- ``JointPositionActionCfg(scale=0.25, use_default_offset=True)``
  (``tasks/locomotion/velocity_env_cfg.py::ActionsCfg``): ``target = default_pos + scale * raw_action``, computed
  once per policy tick and held constant across physics substeps. The ``actions`` observation is the raw,
  unscaled policy output (``mdp.last_action``).
* :class:`GetupAction` -- ``FilteredRelativeJointPositionActionCfg``
  (``tasks/getup/mdp/actions.py``, mirrored exactly): ``target = q_meas + beta * s_j *
  LPF(clip(a, -1, 1))``, with ``q_meas`` re-read every physics substep by default (``relative_per_substep=True``),
  which is what makes ``Kp * s_j * beta`` act as a hard bound on the P-torque rather than a plain relative offset.
  The ``actions`` observation is the filtered value (the previous-action input at deploy time).
"""

from __future__ import annotations

import numpy as np

from . import constants as C
from .constants import JointArrays
from .filters import OnePoleLPF


class WalkingAction:
    action_dim = C.NUM_JOINTS

    def __init__(self, joints: JointArrays, default_qpos: np.ndarray, scale: float = C.ACTION_SCALE_WALKING):
        self.default_qpos = np.asarray(default_qpos, dtype=float)
        self.scale = float(scale)
        self._raw = np.zeros(self.action_dim)
        self._target = self.default_qpos.copy()

    def reset(self) -> None:
        self._raw[:] = 0.0
        self._target[:] = self.default_qpos

    def set_action(self, raw_action: np.ndarray, active: bool = True) -> None:
        """Called once per policy tick (50 Hz)."""
        self._raw[:] = raw_action
        self._target = self.default_qpos + self.scale * self._raw

    def target(self, q_now: np.ndarray) -> np.ndarray:
        """Called once per physics substep; walking's target does not depend on ``q_now`` (absolute action)."""
        return self._target

    def obs_action(self) -> np.ndarray:
        """``mdp.last_action``: the raw, unscaled policy output for this tick."""
        return self._raw


class GetupAction:
    action_dim = C.NUM_JOINTS

    def __init__(
        self,
        joints: JointArrays,
        beta: float = C.GETUP_BETA_DEFAULT,
        scale_torque_factor: float = C.GETUP_SCALE_TORQUE_FACTOR,
        action_clip: float = C.GETUP_ACTION_CLIP,
        use_lpf: bool = C.GETUP_USE_LPF,
        lpf_alpha: float = C.GETUP_LPF_ALPHA,
        lpf_per_substep: bool = C.GETUP_LPF_PER_SUBSTEP,
        relative_per_substep: bool = C.GETUP_RELATIVE_PER_SUBSTEP,
        s_j_override: np.ndarray | None = None,
    ):
        # s_j_override: the trained checkpoint's live s_j (bound_scale already baked in), read from ONNX metadata
        # when available -- takes precedence over the nominal scale_torque_factor*tau_max/Kp computation, so the
        # harness uses the exact bound the policy was trained with instead of assuming it.
        self.s_j = np.asarray(s_j_override, dtype=float) if s_j_override is not None else joints.s_j(scale_torque_factor)
        self.beta = float(beta)
        self.action_clip = float(action_clip)
        self.use_lpf = use_lpf
        self.lpf_per_substep = lpf_per_substep
        self.relative_per_substep = relative_per_substep
        self._lpf = OnePoleLPF(self.action_dim, alpha=lpf_alpha)
        self._clipped = np.zeros(self.action_dim)
        self._filtered = np.zeros(self.action_dim)
        self._delta = np.zeros(self.action_dim)
        self._q_hold = np.zeros(self.action_dim)

    def reset(self) -> None:
        self._lpf.reset(0.0)
        self._clipped[:] = 0.0
        self._filtered[:] = 0.0
        self._delta[:] = 0.0
        self._q_hold[:] = 0.0

    def set_action(self, raw_action: np.ndarray, active: bool = True, q_now: np.ndarray | None = None) -> None:
        """Called once per policy tick (50 Hz). ``active`` is ``env.getup_state.policy_active`` -- False during the
        limp phase of a ``mid_fall`` start, where the action is ignored and the filter state held at zero."""
        clipped = np.clip(raw_action, -self.action_clip, self.action_clip)
        clipped = clipped if active else np.zeros_like(clipped)
        self._clipped[:] = clipped
        if not self.lpf_per_substep:
            self._filtered[:] = self._lpf.step(clipped) if self.use_lpf else clipped
            self._filtered[:] = self._filtered if active else np.zeros_like(self._filtered)
        if q_now is not None:
            self._q_hold[:] = q_now
        self._update_delta()

    def _update_delta(self) -> None:
        self._delta[:] = self.beta * self.s_j * self._filtered

    def target(self, q_now: np.ndarray) -> np.ndarray:
        """Called once per physics substep."""
        if self.lpf_per_substep:
            self._filtered[:] = self._lpf.step(self._clipped) if self.use_lpf else self._clipped
            self._update_delta()
        q = q_now if self.relative_per_substep else self._q_hold
        return q + self._delta

    def obs_action(self) -> np.ndarray:
        """The LPF-filtered, clipped action (the previous-action input at deploy time)."""
        return self._filtered
