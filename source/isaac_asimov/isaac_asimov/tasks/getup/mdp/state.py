# Copyright (c) 2026, Menlo Research.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Shared per-env get-up state.

Every get-up term (reset event, action, assist, rewards, tracker, curricula) reads and writes one
``GetUpState`` object attached to the env as ``env.getup_state``. It is created lazily by
:func:`ensure_state`, so it works whatever order the managers build their class terms in.

Never use ``env.episode_length_buf`` for timing: ``train.py`` randomizes it at start. Use
``getup_state.step`` (env steps since reset) and ``getup_state.control_start_step`` instead.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

# Fixed category order. The canonical `GETUP_CATEGORIES` lives in `resets.py`; this private copy only sizes buffers
# here and must stay identical.
_CATEGORIES = ("supine", "prone", "side_left", "side_right", "sitting", "kneeling", "mid_fall", "standing")

# Success thresholds (the standing condition used by training and evaluation).
STAND_PELVIS_HEIGHT = 0.50
STAND_MAX_TILT = 0.35
STAND_MAX_LIN_VEL = 0.3
STAND_HOLD_S = 1.0
FOOT_CONTACT_FORCE = 1.0

BODY_CONTACT_SENSOR = "body_contact"
FEET_BODY_NAMES = ("left_ankle_roll_link", "right_ankle_roll_link")
PELVIS_BODY_NAME = "pelvis_link"
TORSO_BODY_NAME = "waist_yaw_link"


class GetUpState:
    """Per-env buffers shared by all get-up terms.

    Core fields (shared by all terms and evaluation):
        category (long): index into the category key order.
        control_start_step (long): value of ``step`` at which the policy takes control (limp phase before it).
        policy_active (bool): True when ``step >= control_start_step``; rewards and the assist are masked otherwise.
        max_height_ep (float): max pelvis height since control start (initialized at control start).
        stand_timer_s (float): consecutive seconds ``is_standing`` has held.
        success (bool): latched once ``stand_timer_s >= 1.0``.
        time_to_stand_s (float): seconds from control start to the start of the successful hold (-1 if none).
        assist_enabled (bool): this env receives the assist harness this episode.
        assist_force (float, [N, 3]): world-frame assist force applied in the last substep.

    Additional fields:
        step (long): env steps since the last reset (own counter; not ``episode_length_buf``).
        start_height (float): pelvis height at control start (for ``no_height_progress``).
        control_started (bool): control-start bookkeeping has run for this episode.
        assist_height0 (float): pelvis height the assist target ramps from.
        prev_max_height (float): ``max_height_ep`` before this step's update (for the height-record reward).
        made_progress (bool): pelvis rose 0.2 m above its control-start height, or reached standing height.
        thermal (float, [N, J]): EMA of (tau / tau_rated)^2 per joint (thermal proxy).

    ``deterministic`` (bool [N]): env acts with the policy mean (set by GetUpPPO); its episodes feed the curricula.

    Run-level fields: ``success_by_category`` [C], ``success_ema``, ``success_unassisted_ema`` (windowed over sampled-
    action envs), ``det_*`` (same over deterministic envs), ``curr_*`` (what the curricula use: ``det_*`` with a per-slot
    fallback to the sampled value while the deterministic window is empty),
    ``episodes_finished`` (0-dim long tensor), ``assist_scale`` (float, assist curriculum), ``stage`` (int, effort
    curriculum), ``iteration_offset`` (int, resume offset set by ``curriculum_checkpoint``).

    Update order within an env step: action ``process_actions`` (step counter, ``policy_active``, control-start
    bookkeeping) -> physics -> tracker termination term (heights, stand timer, success, thermal) -> rewards -> resets.
    """

    def __init__(self, env: ManagerBasedEnv):
        n, dev = env.num_envs, env.device
        num_joints = env.scene["robot"].num_joints
        self.num_envs = n
        self.device = dev
        # core fields
        self.category = torch.zeros(n, dtype=torch.long, device=dev)
        self.control_start_step = torch.zeros(n, dtype=torch.long, device=dev)
        self.policy_active = torch.ones(n, dtype=torch.bool, device=dev)
        self.max_height_ep = torch.zeros(n, device=dev)
        self.stand_timer_s = torch.zeros(n, device=dev)
        self.success = torch.zeros(n, dtype=torch.bool, device=dev)
        self.time_to_stand_s = torch.full((n,), -1.0, device=dev)
        self.assist_enabled = torch.zeros(n, dtype=torch.bool, device=dev)
        self.assist_force = torch.zeros(n, 3, device=dev)
        # additional fields
        self.step = torch.zeros(n, dtype=torch.long, device=dev)
        self.start_height = torch.zeros(n, device=dev)
        self.control_started = torch.zeros(n, dtype=torch.bool, device=dev)
        self.assist_height0 = torch.zeros(n, device=dev)
        self.prev_max_height = torch.zeros(n, device=dev)
        self.made_progress = torch.zeros(n, dtype=torch.bool, device=dev)
        self.thermal = torch.zeros(n, num_joints, device=dev)
        self.arm_sat_s = torch.zeros(n, 2, device=dev)  # per episode: seconds with elbow/wrist at the clip [left, right]
        self.ew_streak_s = torch.zeros(n, device=dev)  # current contiguous elbow/wrist saturation streak
        self.ew_streak_max_s = torch.zeros(n, device=dev)  # per-episode max streak
        self.success_hold = torch.zeros(n, dtype=torch.bool, device=dev)  # latched: stood >= curriculum hold time
        self.lost_after_success = torch.zeros(n, dtype=torch.bool, device=dev)  # latched: fell after success
        self.lost_now = torch.zeros(n, dtype=torch.bool, device=dev)  # this step: first loss of standing after success
        self.num_categories = len(_CATEGORIES)
        # envs that act with the policy mean (set by GetUpPPO); default: none
        self.deterministic = torch.zeros(n, dtype=torch.bool, device=dev)
        # run-level (not per env): written by the tracker / curricula, read by curricula and eval
        # sampled-action envs (logging)
        self.success_by_category = torch.zeros(len(_CATEGORIES), device=dev)
        self.success_ema = torch.zeros((), device=dev)
        self.success_unassisted_ema = torch.zeros((), device=dev)
        # deterministic envs (logging) and the curriculum-driving metrics: deterministic windows, falling back to the
        # sampled ones slot by slot while a deterministic window is still empty (e.g. no deterministic envs)
        self.det_success_by_category = torch.zeros(len(_CATEGORIES), device=dev)
        self.det_success_ema = torch.zeros((), device=dev)
        self.det_success_unassisted_ema = torch.zeros((), device=dev)
        self.curr_success_by_category = torch.zeros(len(_CATEGORIES), device=dev)
        self.curr_success_ema = torch.zeros((), device=dev)
        self.curr_success_unassisted_ema = torch.zeros((), device=dev)
        self.curr_category_n = torch.zeros(len(_CATEGORIES), device=dev)
        self.curr_unassisted_n = torch.zeros((), device=dev)
        self.episodes_finished = torch.zeros((), dtype=torch.long, device=dev)  # GPU counter (no host sync)
        self.assist_scale = 1.0
        self.stage = 0
        self.iteration_offset = 0

    def reset(self, env_ids: torch.Tensor | slice):
        """Clear per-episode fields. Called from the tracker's ``reset`` (termination manager, the last manager reset in
        ``_reset_idx``), after the tracker has logged the finished episodes.

        ``category``, ``control_start_step``, ``policy_active`` (written by the reset event) and ``assist_enabled``
        (written by the assist term's reset) are left untouched.
        """
        self.step[env_ids] = 0
        self.control_started[env_ids] = False
        self.max_height_ep[env_ids] = 0.0
        self.prev_max_height[env_ids] = 0.0
        self.made_progress[env_ids] = False
        self.stand_timer_s[env_ids] = 0.0
        self.success[env_ids] = False
        self.time_to_stand_s[env_ids] = -1.0
        self.assist_force[env_ids] = 0.0
        self.thermal[env_ids] = 0.0
        self.arm_sat_s[env_ids] = 0.0
        self.ew_streak_s[env_ids] = 0.0
        self.ew_streak_max_s[env_ids] = 0.0
        self.success_hold[env_ids] = False
        self.lost_after_success[env_ids] = False
        self.lost_now[env_ids] = False


def ensure_state(env: ManagerBasedEnv) -> GetUpState:
    """Return ``env.getup_state``, creating it on first use."""
    st = getattr(env, "getup_state", None)
    if st is None:
        st = GetUpState(env)
        env.getup_state = st
    return st


def pelvis_height(env: ManagerBasedEnv, asset_name: str = "robot") -> torch.Tensor:
    """Pelvis (root) height above the ground **under the pelvis**.

    Plane: root z - env origin z (exact). Generator terrain (rough / slopes): root z - terrain height at the root xy,
    ray-cast against the terrain mesh with ``TerrainHeight`` from ``resets.py`` (the same lookup the reset lift uses), so
    ``is_standing``, the height rewards and the success metric stay correct on bumps and slopes.
    """
    asset: Articulation = env.scene[asset_name]
    root = asset.data.root_pos_w
    if not _terrain_is_generator(env):
        return root[:, 2] - env.scene.env_origins[:, 2]
    # one ray cast per physics state (many terms read it per step); key on the physics-step counter
    cache = env.__dict__.setdefault("_getup_idx_cache", {})
    key = (int(getattr(env, "_sim_step_counter", -1)), getattr(env, "_getup_reset_count", 0))
    hit = cache.get("pelvis_h")
    if hit is not None and hit[0] == key:
        return hit[1]
    from .resets import _terrain_height

    h = root[:, 2] - _terrain_height(env)(root[:, :2])
    cache["pelvis_h"] = (key, h)
    return h


def _terrain_is_generator(env: ManagerBasedEnv) -> bool:
    cache = env.__dict__.setdefault("_getup_idx_cache", {})
    if "terrain_gen" not in cache:
        terrain = getattr(env.scene, "terrain", None)
        cache["terrain_gen"] = terrain is not None and terrain.cfg.terrain_type != "plane"
    return cache["terrain_gen"]


def torso_tilt(env: ManagerBasedEnv, asset_name: str = "robot") -> torch.Tensor:
    """Angle (rad) between the torso z axis and world up, from the torso link's projected gravity."""
    asset: Articulation = env.scene[asset_name]
    body_id = _torso_body_id(env, asset)
    quat = asset.data.body_link_quat_w[:, body_id]
    # z axis of the body in world frame: third column of R(q); its world-z component is 1 - 2(x^2 + y^2)
    cos_tilt = 1.0 - 2.0 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)
    return torch.acos(cos_tilt.clamp(-1.0, 1.0))


def feet_in_contact(
    env: ManagerBasedEnv, sensor_name: str = BODY_CONTACT_SENSOR, threshold: float = FOOT_CONTACT_FORCE
) -> torch.Tensor:
    """[N, 2] bool: each foot's max contact-force norm over the sensor history exceeds ``threshold``."""
    sensor: ContactSensor = env.scene.sensors[sensor_name]
    ids = _feet_sensor_ids(env, sensor)
    hist = sensor.data.net_forces_w_history
    if hist is not None:
        f = hist[:, :, ids].norm(dim=-1).max(dim=1).values
    else:
        f = sensor.data.net_forces_w[:, ids].norm(dim=-1)
    return f > threshold


def is_standing(
    env: ManagerBasedEnv,
    min_height: float = STAND_PELVIS_HEIGHT,
    max_tilt: float = STAND_MAX_TILT,
    max_lin_vel: float = STAND_MAX_LIN_VEL,
    require_velocity: bool = True,
    asset_name: str = "robot",
    sensor_name: str = BODY_CONTACT_SENSOR,
) -> torch.Tensor:
    """Standing condition: [N] bool.

    pelvis height >= 0.50 m, torso tilt <= 0.35 rad, both feet in contact, base linear speed < 0.3 m/s.
    ``require_velocity=False`` gives the reward gate S (no velocity check).
    """
    asset: Articulation = env.scene[asset_name]
    ok = pelvis_height(env, asset_name) >= min_height
    ok &= torso_tilt(env, asset_name) <= max_tilt
    ok &= feet_in_contact(env, sensor_name).all(dim=1)
    if require_velocity:
        ok &= asset.data.root_lin_vel_w.norm(dim=-1) < max_lin_vel
    return ok


# --- per-joint tables -----------------------------------------------------------------------------------------------

# Fallback used only if the asset's `ASIMOV_1_RATED_TORQUE` is empty. Datasheet rated (continuous) joint torque,
# motor datasheet values (ankle values are joint-space: 2·K·12 Nm through the differential).
_RATED_TORQUE_SHIM = {
    ".*_hip_pitch_joint": 40.0,
    ".*_hip_roll_joint": 30.0,
    ".*_hip_yaw_joint": 20.0,
    ".*_knee_joint": 25.0,
    ".*_ankle_pitch_joint": 48.5,
    ".*_ankle_roll_joint": 19.2,
    "waist_yaw_joint": 40.0,
    ".*_shoulder_pitch_joint": 30.0,
    ".*_shoulder_roll_joint": 25.0,
    ".*_shoulder_yaw_joint": 20.0,
    ".*_elbow_joint": 12.0,
    ".*_wrist_yaw_joint": 12.0,
}


class JointTables:
    """Nominal per-joint constants in articulation joint order, captured once before any curriculum/DR changes.

    Attributes: ``tau_max`` (nominal sim effort limit, i.e. the 1.0x baseline), ``kp`` (nominal stiffness),
    ``rated`` (datasheet rated torque), ``names`` (joint names).
    """

    def __init__(self, env: ManagerBasedEnv, asset_name: str = "robot"):
        import isaaclab.utils.string as string_utils

        asset: Articulation = env.scene[asset_name]
        n_j = asset.num_joints
        dev = env.device
        self.names = list(asset.joint_names)
        self.tau_max = torch.zeros(n_j, device=dev)
        self.kp = torch.zeros(n_j, device=dev)
        for act in asset.actuators.values():
            ids = act.joint_indices
            if isinstance(ids, slice):
                ids = torch.arange(n_j, device=dev)
            self.tau_max[ids] = act.effort_limit[0].to(dev)
            self.kp[ids] = act.stiffness[0].to(dev)
        rated_map = _load_rated_torque()
        self.rated = self.tau_max.clone()
        idx, _, vals = string_utils.resolve_matching_names_values(rated_map, self.names, strict=False)
        if len(idx) > 0:
            self.rated[idx] = torch.tensor(vals, device=dev, dtype=torch.float32)


def _load_rated_torque() -> dict[str, float]:
    try:
        from isaac_asimov.assets.robots.asimov_1 import ASIMOV_1_RATED_TORQUE
    except ImportError:
        ASIMOV_1_RATED_TORQUE = {}
    return dict(ASIMOV_1_RATED_TORQUE) if ASIMOV_1_RATED_TORQUE else dict(_RATED_TORQUE_SHIM)


def joint_tables(env: ManagerBasedEnv) -> JointTables:
    """Nominal joint tables (cached on the env the first time; call it early, e.g. from an action term init)."""
    tab = env.__dict__.get("_getup_joint_tables")
    if tab is None:
        tab = JointTables(env)
        env._getup_joint_tables = tab
    return tab


def current_effort_limits(env: ManagerBasedEnv, asset_name: str = "robot") -> torch.Tensor:
    """[N, J] effort limits currently used by the actuator models (reflects `set_effort_scale`)."""
    asset: Articulation = env.scene[asset_name]
    out = torch.empty(env.num_envs, asset.num_joints, device=env.device)
    for act in asset.actuators.values():
        ids = act.joint_indices
        if isinstance(ids, slice):
            out[:] = act.effort_limit
        else:
            out[:, ids] = act.effort_limit
    return out


def per_joint_values(scale: float | dict[str, float], names: list[str], default: float = 1.0) -> list[float]:
    """Per-joint values in ``names`` order from a float or ``{joint regex: value}`` (``re.fullmatch``; first match wins,
    unmatched joints get ``default``)."""
    import re

    if not isinstance(scale, dict):
        return [float(scale)] * len(names)
    out = []
    for n in names:
        v = default
        for k, val in scale.items():
            if re.fullmatch(k, n):
                v = float(val)
                break
        out.append(v)
    return out


# --- cached index helpers -------------------------------------------------------------------------------------------


def _torso_body_id(env, asset: Articulation) -> int:
    cache = env.__dict__.setdefault("_getup_idx_cache", {})
    if "torso" not in cache:
        cache["torso"] = asset.find_bodies(TORSO_BODY_NAME)[0][0]
    return cache["torso"]


def _feet_sensor_ids(env, sensor: ContactSensor) -> list[int]:
    cache = env.__dict__.setdefault("_getup_idx_cache", {})
    key = ("feet", sensor.cfg.prim_path)
    if key not in cache:
        cache[key] = sensor.find_bodies(list(FEET_BODY_NAMES), preserve_order=True)[0]
    return cache[key]
