# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Initial-state generators for sim2sim episodes: standing, the fallen categories via a limp
drop, and mid-fall (push from standing, limp for a random duration, then policy control with no settle).

This is a *lightweight*, single-instance MuJoCo generator for sim2sim smoke testing -- not a replacement for the
training-cache builder (``scripts/getup/build_fallen_cache.py``), which does massively-parallel rejection
sampling in Isaac. Supine/prone/side_left/side_right/mid_fall/standing follow the training-cache recipe directly
(random orientation + random joints within 0.9 of the limits + limp settle + reject on penetration/residual speed).
Sitting/kneeling are seeded from a hand-picked nominal joint pose (Gaussian-jittered) because a pure random-drop is
very unlikely to converge to either pose in a handful of attempts on a single instance -- a heuristic, not a validated
match to the training cache's sitting/kneeling entries.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from . import constants as C
from .actuator import DelayedPDActuatorSim
from .mj_model import RobotModel


def _quat_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    s = np.sin(angle / 2.0)
    return np.array([np.cos(angle / 2.0), axis[0] * s, axis[1] * s, axis[2] * s])


def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


# Nominal base rotation per category, applied to the identity (standing-upright) orientation, before per-attempt
# jitter. Heuristic: rotating the pelvis +/-90 deg about its local Y (pitch) axis lays the robot flat on its front
# or back; +/-90 deg about local X (roll) lays it on a side. NOT verified against the URDF's forward-axis
# convention by rendering.
_CATEGORY_BASE_ROTATION = {
    "supine": _quat_from_axis_angle((0, 1, 0), -np.pi / 2),
    "prone": _quat_from_axis_angle((0, 1, 0), np.pi / 2),
    "side_left": _quat_from_axis_angle((1, 0, 0), np.pi / 2),
    "side_right": _quat_from_axis_angle((1, 0, 0), -np.pi / 2),
    "mid_fall": np.array([1.0, 0.0, 0.0, 0.0]),
    "standing": np.array([1.0, 0.0, 0.0, 0.0]),
    # near-upright poses (bent legs do the rest): identity base orientation, small jitter only.
    "sitting": np.array([1.0, 0.0, 0.0, 0.0]),
    "kneeling": np.array([1.0, 0.0, 0.0, 0.0]),
}

_NOMINAL_HEIGHT = {
    "supine": 0.15, "prone": 0.15, "side_left": 0.18, "side_right": 0.18,
    "sitting": 0.35, "kneeling": 0.40, "mid_fall": C.STANDING_INIT_POS[2], "standing": C.STANDING_INIT_POS[2],
}

# Sitting/kneeling: nominal joint pose (everything else defaults to the standing pose), Gaussian-jittered.
# Explicit per-joint values (not a sign-flip rule): left/right hip_pitch, knee, ankle_pitch, elbow have mirrored
# joint axes on this robot (see MJCF <joint axis=...>; the L/R sign conventions are asymmetric),
# so a "bend the same way" pose needs opposite-signed values, following ASIMOV_1_STANDING_INIT_STATE's own signs
# (left_hip_pitch=-0.15/right=+0.15, left_knee=+0.45/right=-0.45, left_ankle_pitch=-0.30/right=+0.30).
_SITTING_JOINT_OVERRIDES = {
    "left_hip_pitch_joint": -1.6, "right_hip_pitch_joint": 1.6,
    "left_knee_joint": 1.4, "right_knee_joint": -1.4,
    "left_ankle_pitch_joint": 0.0, "right_ankle_pitch_joint": 0.0,
    "left_shoulder_pitch_joint": -0.3, "right_shoulder_pitch_joint": 0.3,
    "left_elbow_joint": 0.8, "right_elbow_joint": -0.8,
}
_KNEELING_JOINT_OVERRIDES = {
    "left_hip_pitch_joint": -0.3, "right_hip_pitch_joint": 0.3,
    "left_knee_joint": 1.45, "right_knee_joint": -1.45,
    "left_ankle_pitch_joint": 0.35, "right_ankle_pitch_joint": -0.35,
}


def _nominal_pose(overrides: dict[str, float]) -> np.ndarray:
    out = C.standing_init_joint_pos().copy()
    for name, val in overrides.items():
        out[C.JOINT_INDEX[name]] = val
    return out


def _sample_joint_pos(category: str, robot: RobotModel, rng: np.random.Generator) -> np.ndarray:
    jnt_ids = [mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in C.ASIMOV_1_JOINT_NAMES]
    lo = np.array([robot.model.jnt_range[j, 0] for j in jnt_ids])
    hi = np.array([robot.model.jnt_range[j, 1] for j in jnt_ids])
    span = hi - lo
    lo90, hi90 = lo + 0.05 * span, hi - 0.05 * span  # "within 0.9 of the limits" -> central 90% of the range

    if category == "sitting":
        nominal = _nominal_pose(_SITTING_JOINT_OVERRIDES)
        q = rng.normal(nominal, 0.08)
    elif category == "kneeling":
        nominal = _nominal_pose(_KNEELING_JOINT_OVERRIDES)
        q = rng.normal(nominal, 0.08)
    elif category == "standing":
        q = C.standing_init_joint_pos()
    else:
        q = rng.uniform(lo90, hi90)
    return np.clip(q, lo, hi)


@dataclass
class FallenState:
    qpos_joints: np.ndarray
    root_pos: np.ndarray
    root_quat_wxyz: np.ndarray
    qvel_joints: np.ndarray
    root_lin_vel: np.ndarray
    root_ang_vel: np.ndarray
    category: str
    accepted: bool
    attempts: int
    max_residual_speed: float


def standing_state() -> FallenState:
    return FallenState(
        qpos_joints=C.standing_init_joint_pos(),
        root_pos=np.array(C.STANDING_INIT_POS),
        root_quat_wxyz=np.array(C.STANDING_INIT_QUAT_WXYZ),
        qvel_joints=np.zeros(C.NUM_JOINTS),
        root_lin_vel=np.zeros(3),
        root_ang_vel=np.zeros(3),
        category="standing",
        accepted=True,
        attempts=0,
        max_residual_speed=0.0,
    )


def sample_mid_fall_limp_duration(rng: np.random.Generator) -> float:
    """``mid_fall`` starts from standing with actuators limp for U(0.04, 1.0) s, then policy."""
    return float(rng.uniform(0.04, 1.0))


def generate_fallen_state(
    category: str,
    robot: RobotModel,
    physics_dt: float,
    rng: np.random.Generator,
    drop_height_range: tuple[float, float] = (0.3, 0.8),
    settle_s: float = 2.0,
    max_speed_reject: float = 0.05,
    max_attempts: int = 20,
) -> FallenState:
    """Training-cache recipe: random orientation + random joints (within 0.9 of the limits) + random small
    velocity, dropped from ``drop_height_range``, limp (Kp=0, Kd=0.5) for ``settle_s``, reject on non-finite state
    or residual speed above ``max_speed_reject``; ``sitting``/``kneeling`` seed from a nominal pose instead of a
    uniform joint sample (see module docstring)."""
    if category not in C.CATEGORY_KEYS or category in ("standing", "mid_fall"):
        raise ValueError(f"generate_fallen_state is for the dropped categories, got {category!r}")

    model, data = robot.model, robot.data
    model.opt.timestep = physics_dt
    actuator = DelayedPDActuatorSim(robot.joints, physics_dt, effort_scale=1.0, rng=rng)
    n_steps = int(round(settle_s / physics_dt))
    base_quat = _CATEGORY_BASE_ROTATION[category]

    last = None
    for attempt in range(1, max_attempts + 1):
        qpos_j = _sample_joint_pos(category, robot, rng)
        jitter = _quat_from_axis_angle(rng.normal(size=3) + 1e-6, rng.uniform(-0.15, 0.15))
        yaw = _quat_from_axis_angle((0, 0, 1), rng.uniform(-np.pi, np.pi))
        root_quat = _quat_mul(yaw, _quat_mul(base_quat, jitter))
        root_quat /= np.linalg.norm(root_quat)
        height = rng.uniform(*drop_height_range) if category not in ("sitting", "kneeling") else _NOMINAL_HEIGHT[category]
        root_pos = np.array([0.0, 0.0, height])
        root_lin_vel = rng.uniform(-0.5, 0.5, size=3)
        root_ang_vel = rng.uniform(-1.0, 1.0, size=3)

        mujoco.mj_resetData(model, data)
        data.qpos[robot.free_joint_qpos_adr : robot.free_joint_qpos_adr + 3] = root_pos
        data.qpos[robot.free_joint_qpos_adr + 3 : robot.free_joint_qpos_adr + 7] = root_quat
        data.qpos[robot.joint_qpos_adr] = qpos_j
        data.qvel[robot.free_joint_dof_adr : robot.free_joint_dof_adr + 3] = root_lin_vel
        data.qvel[robot.free_joint_dof_adr + 3 : robot.free_joint_dof_adr + 6] = root_ang_vel
        mujoco.mj_forward(model, data)
        actuator.reset(data.qpos[robot.joint_qpos_adr].copy())

        diverged = False
        for _ in range(n_steps):
            q_now = data.qpos[robot.joint_qpos_adr]
            qdot_now = data.qvel[robot.joint_dof_adr]
            torque = actuator.compute(q_now, q_now, qdot_now, limp=True)
            data.ctrl[robot.actuator_id] = torque
            mujoco.mj_step(model, data)
            # PhysX hard-enforces the joint velocity limit and it binds during limp falls; MuJoCo has
            # no native per-hinge velocity limit, so clamp post-integration to match.
            qvel_ids = robot.joint_dof_adr
            vlim = robot.joints.velocity_limit
            data.qvel[qvel_ids] = np.clip(data.qvel[qvel_ids], -vlim, vlim)
            if not np.all(np.isfinite(data.qpos)) or not np.all(np.isfinite(data.qvel)):
                diverged = True
                break

        if diverged:
            last = FallenState(qpos_j, root_pos, root_quat, np.zeros(C.NUM_JOINTS), root_lin_vel, root_ang_vel,
                                category, False, attempt, float("inf"))
            continue

        speed = max(
            float(np.max(np.abs(data.qvel[robot.joint_dof_adr]))),
            float(np.linalg.norm(data.qvel[robot.free_joint_dof_adr : robot.free_joint_dof_adr + 3])),
        )
        final = FallenState(
            qpos_joints=data.qpos[robot.joint_qpos_adr].copy(),
            root_pos=data.qpos[robot.free_joint_qpos_adr : robot.free_joint_qpos_adr + 3].copy(),
            root_quat_wxyz=data.qpos[robot.free_joint_qpos_adr + 3 : robot.free_joint_qpos_adr + 7].copy(),
            qvel_joints=data.qvel[robot.joint_dof_adr].copy(),
            root_lin_vel=data.qvel[robot.free_joint_dof_adr : robot.free_joint_dof_adr + 3].copy(),
            root_ang_vel=data.qvel[robot.free_joint_dof_adr + 3 : robot.free_joint_dof_adr + 6].copy(),
            category=category,
            accepted=speed <= max_speed_reject,
            attempts=attempt,
            max_residual_speed=speed,
        )
        last = final
        if final.accepted:
            return final

    return last  # exhausted attempts: return the last try anyway (caller should check .accepted)
