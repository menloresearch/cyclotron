# Copyright (c) 2026, Menlo Research.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Several terms follow NVIDIA WBC-AGILE's stand-up rewards (Apache-2.0) in intent; the code here is an independent
# implementation, not copied.
"""Get-up reward terms (task, posture, safety, regularization), plus the ankle per-motor torque penalty and the Stage B
shaping terms.

Conventions:
- h = pelvis height above the env origin; h* = pelvis standing height (``target_height`` param, default 0.614 m, measured).
- g = projected gravity in the **torso** (``waist_yaw_link``) frame.
- Gates: S = standing gate (``is_standing`` without the velocity check, h >= 0.8 h*); R = rising gate (h > 0.38 m).
- Every term is multiplied by ``getup_state.policy_active`` (no reward during limp phases).
- Torque ratios use the nominal (1.0x) sim effort limit ``tau_max`` unless stated; the soft-limit term uses the
  current (curriculum-scaled) limit; the thermal proxy uses the rated torque.
- Rewards never mutate state (zero-weight terms are skipped by the reward manager); the tracker updates state.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

from .state import (
    BODY_CONTACT_SENSOR,
    TORSO_BODY_NAME,
    current_effort_limits,
    ensure_state,
    feet_in_contact,
    is_standing,
    joint_tables,
    pelvis_height,
    torso_tilt,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

H_STAR = 0.614
RISING_HEIGHT = 0.38
_FEET = SceneEntityCfg("robot", body_names=["left_ankle_roll_link", "right_ankle_roll_link"], preserve_order=True)
_TORSO = SceneEntityCfg("robot", body_names=[TORSO_BODY_NAME])
# Head collision box (shell-matched collision, `neck_pitch_link` merged into `waist_yaw_link` by merge_fixed_joints):
# centre and half extents in the waist_yaw_link frame (centre (-0.0036, 0, 0.4580), size 0.162x0.126x0.125).
_HEAD_BOX_CENTER = (-0.0036, 0.0, 0.4580)
_HEAD_BOX_HALF = (0.081, 0.063, 0.0625)
_HEAD_CORNERS = tuple(
    (_HEAD_BOX_CENTER[0] + sx * _HEAD_BOX_HALF[0], _HEAD_BOX_CENTER[1] + sy * _HEAD_BOX_HALF[1],
     _HEAD_BOX_CENTER[2] + sz * _HEAD_BOX_HALF[2])
    for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)
)  # fmt: skip


# --- helpers ----------------------------------------------------------------------------------------------------------


def _active(env: ManagerBasedRLEnv) -> torch.Tensor:
    return ensure_state(env).policy_active.float()


def _gate_rising(env: ManagerBasedRLEnv, rising_height: float = RISING_HEIGHT) -> torch.Tensor:
    return (pelvis_height(env) > rising_height).float()


def _gate_standing(env: ManagerBasedRLEnv, target_height: float = H_STAR) -> torch.Tensor:
    return is_standing(env, min_height=0.8 * target_height, require_velocity=False).float()


def _torso_gravity(env: ManagerBasedRLEnv) -> torch.Tensor:
    asset: Articulation = env.scene["robot"]
    body_id = _body_ids(env, _TORSO)[0]
    quat = asset.data.body_link_quat_w[:, body_id]
    g = asset.data.GRAVITY_VEC_W
    return math_utils.quat_apply_inverse(quat, g)


def _body_ids(env: ManagerBasedRLEnv, cfg: SceneEntityCfg) -> list[int]:
    cache = env.__dict__.setdefault("_getup_rew_cache", {})
    key = ("body", tuple(cfg.body_names))
    if key not in cache:
        cache[key] = env.scene[cfg.name].find_bodies(cfg.body_names, preserve_order=True)[0]
    return cache[key]


def _sensor_ids(env: ManagerBasedRLEnv, sensor: ContactSensor, names: list[str]) -> list[int]:
    cache = env.__dict__.setdefault("_getup_rew_cache", {})
    key = ("sensor", tuple(names))
    if key not in cache:
        cache[key] = sensor.find_bodies(names, preserve_order=True)[0]
    return cache[key]


def _feet_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    asset: Articulation = env.scene["robot"]
    return asset.data.body_link_pos_w[:, _body_ids(env, _FEET)]


def _joint_ids(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg):
    return asset_cfg.joint_ids


def _tau_ratio(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    tab = joint_tables(env)
    ids = asset_cfg.joint_ids
    return asset.data.applied_torque[:, ids] / tab.tau_max[ids].clamp(min=1e-6)


# --- task -------------------------------------------------------------------------------------------------------------


def height_exp(env: ManagerBasedRLEnv, std: float, target_height: float = H_STAR) -> torch.Tensor:
    """exp(-(h - h*)^2 / std^2)."""
    h = pelvis_height(env)
    return torch.exp(-torch.square(h - target_height) / std**2) * _active(env)


def height_record(env: ManagerBasedRLEnv, max_rate: float = 1.0) -> torch.Tensor:
    """max(0, h - h_max,ep) / dt, clipped to ``max_rate`` [m/s]. Non-exploitable: pays only for new records."""
    st = ensure_state(env)
    rate = (pelvis_height(env) - st.prev_max_height).clamp(min=0.0) / env.step_dt
    return rate.clamp(max=max_rate) * _active(env)


def upright(env: ManagerBasedRLEnv) -> torch.Tensor:
    """(1 - g_z) / 2 in [0, 1] (torso frame)."""
    return 0.5 * (1.0 - _torso_gravity(env)[:, 2]) * _active(env)


def torso_flat_orientation_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """||g_xy||^2 (torso frame)."""
    return torch.sum(torch.square(_torso_gravity(env)[:, :2]), dim=1) * _active(env)


def feet_support(
    env: ManagerBasedRLEnv, max_foot_height: float = 0.08, rising_height: float = RISING_HEIGHT
) -> torch.Tensor:
    """both feet in contact and both feet (ankle-roll link origins) below ``max_foot_height``; gate R."""
    contact = feet_in_contact(env).all(dim=1)
    low = (_feet_pos(env)[:, :, 2] - env.scene.env_origins[:, 2:3] < max_foot_height).all(dim=1)
    return (contact & low).float() * _gate_rising(env, rising_height) * _active(env)


def stand_bonus(env: ManagerBasedRLEnv, target_height: float = H_STAR) -> torch.Tensor:
    """1 under the standing gate S."""
    return _gate_standing(env, target_height) * _active(env)


# --- posture ----------------------------------------------------------------------------------------------------------


def joint_deviation_l1_standing(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), target_height: float = H_STAR
) -> torch.Tensor:
    """sum |q - q_default| under gate S."""
    asset: Articulation = env.scene[asset_cfg.name]
    ids = asset_cfg.joint_ids
    dev = torch.sum(torch.abs(asset.data.joint_pos[:, ids] - asset.data.default_joint_pos[:, ids]), dim=1)
    return dev * _gate_standing(env, target_height) * _active(env)


def not_moving_standing(env: ManagerBasedRLEnv, target_height: float = H_STAR) -> torch.Tensor:
    """||v||^2 + ||omega||^2 (root, world/base) under gate S."""
    asset: Articulation = env.scene["robot"]
    m = torch.sum(torch.square(asset.data.root_lin_vel_w), dim=1) + torch.sum(
        torch.square(asset.data.root_ang_vel_b), dim=1
    )
    return m * _gate_standing(env, target_height) * _active(env)


def feet_lateral_distance_standing(
    env: ManagerBasedRLEnv, target_distance: float = 0.215, target_height: float = H_STAR
) -> torch.Tensor:
    """|d_y - d_y,default| (lateral feet distance in the pelvis heading frame), L1, under gate S."""
    asset: Articulation = env.scene["robot"]
    feet = _feet_pos(env)
    diff = feet[:, 0] - feet[:, 1]
    diff_b = math_utils.quat_apply_inverse(math_utils.yaw_quat(asset.data.root_quat_w), diff)
    return torch.abs(diff_b[:, 1] - target_distance) * _gate_standing(env, target_height) * _active(env)


def feet_yaw_vs_base_standing(env: ManagerBasedRLEnv, target_height: float = H_STAR) -> torch.Tensor:
    """(mean feet yaw - base yaw)^2 (wrapped), under gate S."""
    asset: Articulation = env.scene["robot"]
    fq = asset.data.body_link_quat_w[:, _body_ids(env, _FEET)]
    _, _, yaw_l = math_utils.euler_xyz_from_quat(fq[:, 0])
    _, _, yaw_r = math_utils.euler_xyz_from_quat(fq[:, 1])
    _, _, yaw_b = math_utils.euler_xyz_from_quat(asset.data.root_quat_w)
    el = math_utils.wrap_to_pi(yaw_l - yaw_b)
    er = math_utils.wrap_to_pi(yaw_r - yaw_b)
    mean_err = 0.5 * (el + er)
    return torch.square(mean_err) * _gate_standing(env, target_height) * _active(env)


def ang_vel_xy_rising(env: ManagerBasedRLEnv, rising_height: float = RISING_HEIGHT) -> torch.Tensor:
    """||omega_xy||^2 (base frame), gate R."""
    asset: Articulation = env.scene["robot"]
    return (
        torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)
        * _gate_rising(env, rising_height)
        * _active(env)
    )


def joint_deviation_l2_rising(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, rising_height: float = RISING_HEIGHT
) -> torch.Tensor:
    """sum (q - q_default)^2 over the selected joints (hip roll/yaw, or waist), gate R."""
    asset: Articulation = env.scene[asset_cfg.name]
    ids = asset_cfg.joint_ids
    dev = torch.sum(torch.square(asset.data.joint_pos[:, ids] - asset.data.default_joint_pos[:, ids]), dim=1)
    return dev * _gate_rising(env, rising_height) * _active(env)


def feet_distance_bounds(env: ManagerBasedRLEnv, min_dist: float = 0.12, max_dist: float = 0.5) -> torch.Tensor:
    """1[d < min] + 1[d > max], d = horizontal feet distance. Always on."""
    feet = _feet_pos(env)
    d = torch.norm(feet[:, 0, :2] - feet[:, 1, :2], dim=1)
    return ((d < min_dist).float() + (d > max_dist).float()) * _active(env)


# --- safety -----------------------------------------------------------------------------------------------------------


def impact_nonfoot(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 150.0,
    max_excess: float = 2000.0,
) -> torch.Tensor:
    """sum_b clip(max_history ||F_b|| - threshold, 0, max_excess) over head/torso/pelvis bodies.

    Masked by ``policy_active`` like every get-up reward (no reward during limp phases): the limp part of a
    ``mid_fall`` is not penalized, only impacts after the policy takes control.

    The max over the 4-substep history catches impacts between policy steps; the clip bounds a single PhysX contact
    spike (5-23 kN measured on limp drops) to ``max_excess`` per body, so one spike cannot dominate the return
    (per step at weight -1e-3 and dt 0.02: at most 0.04 per body).
    """
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    f = sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids].norm(dim=-1).max(dim=1).values
    return torch.sum((f - threshold).clamp(0.0, max_excess), dim=1) * _active(env)


def head_contact(
    env: ManagerBasedRLEnv, sensor_name: str = BODY_CONTACT_SENSOR, force_threshold: float = 1.0, margin: float = 0.02
) -> torch.Tensor:
    """head touches the ground.

    The head box is merged into ``waist_yaw_link`` (merge_fixed_joints=True), so no head-only contact force exists.
    Proxy: the torso link reports contact **and** the lowest corner of the head collision box is within ``margin`` of
    the ground plane.
    """
    asset: Articulation = env.scene["robot"]
    sensor: ContactSensor = env.scene.sensors[sensor_name]
    torso_s = _sensor_ids(env, sensor, [TORSO_BODY_NAME])
    f = sensor.data.net_forces_w_history[:, :, torso_s].norm(dim=-1).max(dim=1).values[:, 0]
    body_id = _body_ids(env, _TORSO)[0]
    pos = asset.data.body_link_pos_w[:, body_id]
    quat = asset.data.body_link_quat_w[:, body_id]
    corners = torch.tensor(_HEAD_CORNERS, device=env.device)  # [8, 3]
    n = env.num_envs
    pts = math_utils.quat_apply(quat.unsqueeze(1).expand(n, 8, 4).reshape(-1, 4), corners.repeat(n, 1)).view(n, 8, 3)
    z_min = (pts[..., 2] + pos[:, 2:3]).min(dim=1).values - env.scene.env_origins[:, 2]
    return ((f > force_threshold) & (z_min < margin)).float() * _active(env)


# --- regularization ---------------------------------------------------------------------------------------------------


def action_rate_clipped_l2(env: ManagerBasedRLEnv, action_name: str = "joint_pos") -> torch.Tensor:
    """||a_t - a_{t-1}||^2 on the clipped raw policy action."""
    term = env.action_manager.get_term(action_name)
    return torch.sum(torch.square(term.clipped_actions - term.prev_clipped_actions), dim=1) * _active(env)


def action_out_of_bounds_l2(env: ManagerBasedRLEnv, action_name: str = "joint_pos", bound: float = 1.0) -> torch.Tensor:
    """sum relu(|a_raw| - bound)^2 on the raw policy output (before the +-1 clip). Keeps the policy mean inside
    the clip range so that exploration noise (std) is not wasted beyond it."""
    term = env.action_manager.get_term(action_name)
    return torch.sum(torch.square((term.raw_actions.abs() - bound).clamp(min=0.0)), dim=1) * _active(env)


def action_second_diff_l2(env: ManagerBasedRLEnv, action_name: str = "joint_pos") -> torch.Tensor:
    """||a_t - 2 a_{t-1} + a_{t-2}||^2 on the clipped raw policy action."""
    term = env.action_manager.get_term(action_name)
    d2 = term.clipped_actions - 2.0 * term.prev_clipped_actions + term.prev_prev_clipped_actions
    return torch.sum(torch.square(d2), dim=1) * _active(env)


def joint_acc_l2_active(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """||qdd||^2."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_acc[:, asset_cfg.joint_ids]), dim=1) * _active(env)


def torque_tiredness(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """sum min(tau_hat^2, 1), tau_hat = tau / tau_max (nominal)."""
    r = _tau_ratio(env, asset_cfg)
    return torch.sum(torch.square(r).clamp(max=1.0), dim=1) * _active(env)


def torque_soft_limit(
    env: ManagerBasedRLEnv, soft_ratio: float = 0.85, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """sum relu(|tau| / tau_limit - soft_ratio), against the **current** (curriculum-scaled) effort limit."""
    asset: Articulation = env.scene[asset_cfg.name]
    ids = asset_cfg.joint_ids
    lim = current_effort_limits(env, asset_cfg.name)[:, ids]
    r = torch.abs(asset.data.applied_torque[:, ids]) / lim.clamp(min=1e-6)
    return torch.sum((r - soft_ratio).clamp(min=0.0), dim=1) * _active(env)


def upper_torque_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """sum tau_hat^2 over elbow and wrist joints (12 Nm actuators)."""
    return torch.sum(torch.square(_tau_ratio(env, asset_cfg)), dim=1) * _active(env)


def positive_power(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """sum relu(tau * qd)."""
    asset: Articulation = env.scene[asset_cfg.name]
    ids = asset_cfg.joint_ids
    p = asset.data.applied_torque[:, ids] * asset.data.joint_vel[:, ids]
    return torch.sum(p.clamp(min=0.0), dim=1) * _active(env)


def joint_pos_limits_soft(
    env: ManagerBasedRLEnv, soft_ratio: float = 0.95, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """distance outside the central ``soft_ratio`` of the hard joint range (independent of the asset's soft
    factor, so the 0.98 asset factor does not change the penalty band)."""
    asset: Articulation = env.scene[asset_cfg.name]
    ids = asset_cfg.joint_ids
    lim = asset.data.joint_pos_limits[:, ids]
    mid = 0.5 * (lim[..., 0] + lim[..., 1])
    half = 0.5 * (lim[..., 1] - lim[..., 0]) * soft_ratio
    q = asset.data.joint_pos[:, ids]
    out = (q - (mid + half)).clamp(min=0.0) + ((mid - half) - q).clamp(min=0.0)
    return torch.sum(out, dim=1) * _active(env)


def joint_vel_limits_soft(
    env: ManagerBasedRLEnv, soft_ratio: float = 0.9, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """sum relu(|qd| - soft_ratio * v_max) with the PhysX (URDF) velocity limits, clipped at 1 rad/s per joint."""
    asset: Articulation = env.scene[asset_cfg.name]
    ids = asset_cfg.joint_ids
    over = torch.abs(asset.data.joint_vel[:, ids]) - soft_ratio * asset.data.joint_vel_limits[:, ids]
    return torch.sum(over.clamp(min=0.0, max=1.0), dim=1) * _active(env)


def root_acc_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """||a_torso||^2 (linear acceleration of the torso link)."""
    asset: Articulation = env.scene["robot"]
    body_id = _body_ids(env, _TORSO)[0]
    return torch.sum(torch.square(asset.data.body_lin_acc_w[:, body_id]), dim=1) * _active(env)


def thermal_proxy_penalty(
    env: ManagerBasedRLEnv, threshold: float = 0.6, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """sum relu(E_j - threshold), E_j = EMA of (tau_j / tau_rated_j)^2 (tracker-updated, ~2 s time constant)."""
    e = ensure_state(env).thermal[:, asset_cfg.joint_ids]
    return torch.sum((e - threshold).clamp(min=0.0), dim=1) * _active(env)


# --- ankle differential -----------------------------------------------------------------------------------------------

ANKLE_MOTOR_RATED = 12.0  # EC-A4310 rated torque [N m]


def ankle_motor_torque(
    env: ManagerBasedRLEnv,
    soft_ratio: float = 0.85,
    motor_rated: float = ANKLE_MOTOR_RATED,
    pitch_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", joint_names=["left_ankle_pitch_joint", "right_ankle_pitch_joint"], preserve_order=True
    ),
    roll_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", joint_names=["left_ankle_roll_joint", "right_ankle_roll_joint"], preserve_order=True
    ),
    k_pitch: float = 1.0,
    k_roll: float = 1.0,
) -> torch.Tensor:
    """Per-motor ankle torque penalty through the differential: sum_m relu(|tau_m| / tau_rated - soft_ratio).

    ``k_pitch``/``k_roll`` are the **torque-side** differential gains (``getup_actuators.ankle_motor_torques``). Virtual work on the
    ankle position map gives (2.02, 0.80); the conservative alternative treats 2.02 as a position-only ratio (joint
    torque = 2x motor torque), i.e. (1.0, 1.0), which predicts larger motor torques. The env uses the conservative
    (1.0, 1.0) until the hardware team confirms the transmission.
    """
    asset: Articulation = env.scene["robot"]
    tp = asset.data.applied_torque[:, pitch_cfg.joint_ids]
    tr = asset.data.applied_torque[:, roll_cfg.joint_ids]
    ma, mb = _ankle_motor_torques(tp, tr, k_pitch, k_roll)
    r = torch.cat([ma, mb], dim=1).abs() / motor_rated
    return torch.sum((r - soft_ratio).clamp(min=0.0), dim=1) * _active(env)


def _ankle_motor_torques(
    tau_pitch: torch.Tensor, tau_roll: torch.Tensor, k_pitch: float, k_roll: float
) -> tuple[torch.Tensor, torch.Tensor]:
    try:
        from isaac_asimov.assets.robots.getup_actuators import ankle_motor_torques
    except ImportError:  # fallback: same formula as ``ankle_motor_torques`` (virtual work, torque-side gains)
        a = tau_pitch / k_pitch
        b = tau_roll / k_roll
        return 0.5 * (a - b), 0.5 * (-a - b)
    return ankle_motor_torques(tau_pitch, tau_roll, k_pitch, k_roll)


# --- Stage B: rise-speed shaping (smoother, less explosive rise) ------------------------------------------------------


def pelvis_vertical_speed_excess(env: ManagerBasedRLEnv, max_speed: float = 1.0) -> torch.Tensor:
    """relu(v_z - max_speed)^2 on the pelvis (root) vertical velocity [m/s]; only upward speed is penalized."""
    asset: Articulation = env.scene["robot"]
    return torch.square((asset.data.root_lin_vel_w[:, 2] - max_speed).clamp(min=0.0)) * _active(env)


def torso_ang_vel_excess(env: ManagerBasedRLEnv, max_rate: float = 2.0) -> torch.Tensor:
    """relu(||omega_torso|| - max_rate)^2 [rad/s] on the torso link (world-frame angular speed)."""
    asset: Articulation = env.scene["robot"]
    body_id = _body_ids(env, _TORSO)[0]
    w = torch.norm(asset.data.body_link_ang_vel_w[:, body_id], dim=-1)
    return torch.square((w - max_rate).clamp(min=0.0)) * _active(env)


# --- Stage B: arm loading against stops / self-contact (root cause of elbow/wrist saturation) -------------------------

_ELBOW_WRIST = SceneEntityCfg("robot", joint_names=[".*_elbow_joint", ".*_wrist_yaw_joint"])
_WRIST_YAW = SceneEntityCfg("robot", joint_names=[".*_wrist_yaw_joint"])


def arm_limit_under_load(
    env: ManagerBasedRLEnv, near_limit: float = 0.9, asset_cfg: SceneEntityCfg = _ELBOW_WRIST
) -> torch.Tensor:
    """sum_j relu(|q_j - q_center,j| / half_range_j - near_limit) * |tau_j| / tau_max,j over elbow/wrist joints:
    being near the mechanical stop **while torqued** (the arm loaded against its stops)."""
    asset: Articulation = env.scene[asset_cfg.name]
    ids = asset_cfg.joint_ids
    lim = asset.data.joint_pos_limits[:, ids]
    center = 0.5 * (lim[..., 0] + lim[..., 1])
    half = (0.5 * (lim[..., 1] - lim[..., 0])).clamp(min=1e-6)
    near = (torch.abs(asset.data.joint_pos[:, ids] - center) / half - near_limit).clamp(min=0.0)
    tau_hat = torch.abs(_tau_ratio(env, asset_cfg))
    return torch.sum(near * tau_hat, dim=1) * _active(env)


def wrist_yaw_deviation(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = _WRIST_YAW) -> torch.Tensor:
    """sum |q_wrist_yaw - q_default| in every phase (the wrist yaw has no functional role in the push-off)."""
    asset: Articulation = env.scene[asset_cfg.name]
    ids = asset_cfg.joint_ids
    return torch.sum(torch.abs(asset.data.joint_pos[:, ids] - asset.data.default_joint_pos[:, ids]), dim=1) * _active(env)


# Collision sample points (link frame) and radii of the forearm links (URDF elbow spheres; wrist stub along the
# wrist-yaw axis). Used to decide whether a contact on the link can be with the ground.
_ARM_LINKS = {
    "left_elbow_link": (((0.0, 0.01, 0.0), (0.09, 0.01, -0.07)), 0.03),
    "right_elbow_link": (((0.0, -0.01, 0.0), (0.09, -0.01, -0.07)), 0.03),
    "left_wrist_yaw_link": (((0.0, 0.0, 0.0), (0.0306, 0.0, -0.0257)), 0.033),
    "right_wrist_yaw_link": (((0.0, 0.0, 0.0), (0.0306, 0.0, -0.0257)), 0.033),
}


def arm_self_contact(
    env: ManagerBasedRLEnv,
    sensor_name: str = BODY_CONTACT_SENSOR,
    force_threshold: float = 5.0,
    ground_margin: float = 0.03,
) -> torch.Tensor:
    """Number of forearm links (elbow / wrist, both arms) in a **non-ground** contact above ``force_threshold``.

    Pairwise (filtered) contact reporting is one-to-many only in this Isaac Lab version, so the partner body is
    inferred geometrically: the link reports a net contact force > threshold (max over the substep history) while its
    lowest collision point is more than ``ground_margin`` above the ground plane, i.e. it is pressed against another
    body part (torso / shoulder, or a leg). Flat-ground approximation (ground = env origin height).
    """
    asset: Articulation = env.scene["robot"]
    sensor: ContactSensor = env.scene.sensors[sensor_name]
    names = list(_ARM_LINKS)
    s_ids = _sensor_ids(env, sensor, names)
    f = sensor.data.net_forces_w_history[:, :, s_ids].norm(dim=-1).max(dim=1).values  # [N, 4]
    b_ids = _body_ids(env, SceneEntityCfg("robot", body_names=names))
    pos = asset.data.body_link_pos_w[:, b_ids]  # [N, 4, 3]
    quat = asset.data.body_link_quat_w[:, b_ids]
    ground = env.scene.env_origins[:, 2:3]
    lows = []
    for k, n in enumerate(names):
        pts, r = _ARM_LINKS[n]
        z = None
        for p in pts:
            off = torch.tensor(p, device=env.device).expand(env.num_envs, 3)
            zz = (pos[:, k] + math_utils.quat_apply(quat[:, k], off))[:, 2] - r
            z = zz if z is None else torch.minimum(z, zz)
        lows.append(z)
    low = torch.stack(lows, dim=1) - ground
    return torch.sum(((f > force_threshold) & (low > ground_margin)).float(), dim=1) * _active(env)


# --- StageB2: robust hold ----------------------------------------------------------------------------------------------


def hold_after_success(env: ManagerBasedRLEnv, target_height: float = H_STAR) -> torch.Tensor:
    """1 per step under the standing gate S (no velocity term) once the episode has succeeded: rewards holding the
    stand (pushes may move the robot; only falling is penalized)."""
    st = ensure_state(env)
    return _gate_standing(env, target_height) * st.success.float() * _active(env)


def lost_standing_event(env: ManagerBasedRLEnv) -> torch.Tensor:
    """1 on the step the robot first loses standing after success (pelvis < 0.5 m or torso tilt > 0.35 rad; set by the
    tracker, once per episode). Note: rewards are scaled by dt, so weight -250 = -5 per event."""
    return ensure_state(env).lost_now.float() * _active(env)


def post_success_drift(
    env: ManagerBasedRLEnv,
    tilt_free: float = 0.2,
    height_margin: float = 0.05,
    target_height: float = H_STAR,
) -> torch.Tensor:
    """Dense shaping after success: relu(tilt - tilt_free) + 2 * relu((h* - margin) - h) (height drop in m)."""
    st = ensure_state(env)
    tilt = torso_tilt(env)
    drop = ((target_height - height_margin) - pelvis_height(env)).clamp(min=0.0)
    return ((tilt - tilt_free).clamp(min=0.0) + 2.0 * drop) * st.success.float() * _active(env)



# --- StageB3: stable stance and faster rise under weak motors ---------------------------------------------------------


def standing_motion(env: ManagerBasedRLEnv, target_height: float = H_STAR) -> torch.Tensor:
    """||omega_torso||^2 + v_z,pelvis^2 under the standing gate S (a quiet, stable stance)."""
    asset: Articulation = env.scene["robot"]
    body_id = _body_ids(env, _TORSO)[0]
    w2 = torch.sum(torch.square(asset.data.body_link_ang_vel_w[:, body_id]), dim=1)
    vz2 = torch.square(asset.data.root_lin_vel_w[:, 2])
    return (w2 + vz2) * _gate_standing(env, target_height) * _active(env)


def not_yet_standing(env: ManagerBasedRLEnv) -> torch.Tensor:
    """1 per step under policy control before the episode's first success (time penalty; masked while limp)."""
    return (~ensure_state(env).success).float() * _active(env)
