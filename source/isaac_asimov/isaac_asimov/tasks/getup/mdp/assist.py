# Copyright (c) 2026, Menlo Research.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Structure adapted from NVIDIA WBC-AGILE `agile/rl_env/mdp/actions/lift_action.py`
# (Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES, Apache-2.0, http://www.apache.org/licenses/LICENSE-2.0).
# Changes: world-frame force re-added every physics substep through the instantaneous wrench composer, PD with damping on
# the pelvis vertical velocity, target ramp from the control-start height, per-env unassisted fraction, policy-active
# gating, and a single curriculum-set ``scale``.
"""Assist harness: a vertical PD force on the torso toward a target height ramped to h* over 3 s.

The force is added to ``instantaneous_wrench_composer`` with ``is_global=True`` in every ``apply_actions`` call (every
physics substep), so it stays vertical while the torso rotates. It is a
zero-dimensional action term, so the policy output stays 23-D.

``F_z = scale * clip(k (h_target - h) - d * v_z, 0, F_max)``, ``h_target = h0 + (h* - h0) * clip(t / T_ramp, 0, 1)``
where h is the pelvis height, h0 its value at control start and t the time since control start. Yaw damping:
``tau_z = -scale * c * omega_z`` (clipped). Not gated by orientation. ``unassisted_fraction`` of the envs get no assist
in each episode (``getup_state.assist_enabled``).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from .state import ensure_state, pelvis_height

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class AssistForceAction(ActionTerm):
    """Zero-dimensional action term applying the assist harness force. ``scale`` is set by the curriculum."""

    cfg: AssistForceActionCfg
    _asset: Articulation

    def __init__(self, cfg: AssistForceActionCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        ids, _ = self._asset.find_bodies(cfg.body_name)
        if len(ids) != 1:
            raise ValueError(f"AssistForceAction: body '{cfg.body_name}' resolved to {ids}")
        self._body_ids = [ids[0]]
        self.scale: float = float(cfg.initial_scale)
        self._forces = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._torques = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._empty = torch.zeros(self.num_envs, 0, device=self.device)
        ensure_state(env)

    @property
    def action_dim(self) -> int:
        return 0

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._empty

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._empty

    def process_actions(self, actions: torch.Tensor):
        pass

    def target_height(self) -> torch.Tensor:
        st = self._env.getup_state
        t = (st.step - st.control_start_step).clamp(min=0).float() * self._env.step_dt
        ratio = (t / self.cfg.ramp_time_s).clamp(0.0, 1.0)
        return st.assist_height0 + (self.cfg.target_height - st.assist_height0) * ratio

    def apply_actions(self):
        st = self._env.getup_state
        if self.scale <= 0.0:
            st.assist_force.zero_()
            return
        mask = (st.assist_enabled & st.policy_active).float()
        h = pelvis_height(self._env)
        vz = self._asset.data.root_lin_vel_w[:, 2]
        fz = self.cfg.stiffness * (self.target_height() - h) - self.cfg.damping * vz
        fz = self.scale * fz.clamp(0.0, self.cfg.force_max) * mask
        wz = self._asset.data.root_ang_vel_w[:, 2]
        tz = (-self.scale * self.cfg.yaw_damping * wz).clamp(-self.cfg.yaw_torque_max, self.cfg.yaw_torque_max) * mask
        self._forces[:, 0, 2] = fz
        self._torques[:, 0, 2] = tz
        st.assist_force[:] = self._forces[:, 0]
        self._asset.instantaneous_wrench_composer.add_forces_and_torques(
            forces=self._forces, torques=self._torques, body_ids=self._body_ids, is_global=True
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        st = ensure_state(self._env)
        if env_ids is None or isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)
        n = len(env_ids)
        st.assist_enabled[env_ids] = torch.rand(n, device=self.device) >= self.cfg.unassisted_fraction
        st.assist_force[env_ids] = 0.0


@configclass
class AssistForceActionCfg(ActionTermCfg):
    class_type: type = AssistForceAction

    body_name: str = "waist_yaw_link"
    """Link the force is applied to (torso; the waist joint is yaw-only)."""
    target_height: float = 0.614
    """h*: pelvis standing height [m]."""
    ramp_time_s: float = 3.0
    stiffness: float = 5000.0
    damping: float = 500.0
    force_max: float = 250.0
    yaw_damping: float = 50.0
    yaw_torque_max: float = 50.0
    unassisted_fraction: float = 0.2
    initial_scale: float = 1.0
