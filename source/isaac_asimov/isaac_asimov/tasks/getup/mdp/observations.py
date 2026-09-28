# Copyright (c) 2026, Menlo Research.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Get-up observation terms.

Policy terms reuse the walking task's functions (`delayed_obs`, `joint_pos_rel`, `joint_vel_rel`); only ``actions`` is
replaced by :func:`filtered_action`. Critic-only terms are defined here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

from .state import current_effort_limits, ensure_state, pelvis_height

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# Bodies whose contact-force norm the critic sees, in L/R pairs then centre bodies (order matters for mirroring).
# Head collisions are merged into waist_yaw_link (merge_fixed_joints=True), so "torso" covers the head.
CRITIC_CONTACT_BODIES = [
    "left_ankle_roll_link", "right_ankle_roll_link",
    "left_knee_link", "right_knee_link",
    "left_elbow_link", "right_elbow_link",
    "left_wrist_yaw_link", "right_wrist_yaw_link",
    "waist_yaw_link",
    "pelvis_link",
]

# Mirror rules: every term here has a rule in the registry in `tasks/getup/symmetry.py`; `limp_remaining` is registered
# by `getup_env_cfg.py`.


def filtered_action(env: ManagerBasedRLEnv, action_name: str = "joint_pos") -> torch.Tensor:
    """Filtered action of the get-up action term (the previous-action input at deploy time)."""
    return env.action_manager.get_term(action_name).filtered_actions


def pelvis_height_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    return pelvis_height(env).unsqueeze(-1)


def root_quat_no_yaw(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root orientation with the heading removed, (w, x, y, z), w >= 0."""
    asset: Articulation = env.scene[asset_cfg.name]
    q = asset.data.root_quat_w
    q = math_utils.quat_mul(math_utils.quat_conjugate(math_utils.yaw_quat(q)), q)
    return math_utils.quat_unique(q)


def body_contact_forces(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """log(1 + |F|) of the max contact-force norm over the sensor history, per selected body."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    f = sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids].norm(dim=-1).max(dim=1).values
    return torch.log1p(f)


def effort_saturation(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Applied torque / current effort limit, per joint (signed)."""
    asset: Articulation = env.scene[asset_cfg.name]
    lim = current_effort_limits(env, asset_cfg.name)
    return asset.data.applied_torque[:, asset_cfg.joint_ids] / lim[:, asset_cfg.joint_ids].clamp(min=1e-6)


def thermal_proxy(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """EMA of (tau / tau_rated)^2 per joint (updated by the tracker)."""
    return ensure_state(env).thermal[:, asset_cfg.joint_ids]


def assist_force(env: ManagerBasedRLEnv, force_scale: float = 250.0) -> torch.Tensor:
    """Vertical assist force, normalized by its clip value."""
    return ensure_state(env).assist_force[:, 2:3] / force_scale


def limp_remaining(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Seconds left in the limp (no-control) phase; 0 once the policy is in control."""
    st = ensure_state(env)
    return ((st.control_start_step - st.step).clamp(min=0).float() * env.step_dt).unsqueeze(-1)
