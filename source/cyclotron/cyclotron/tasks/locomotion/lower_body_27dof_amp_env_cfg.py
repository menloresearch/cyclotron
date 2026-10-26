# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Leg-only AMP policy on the 27-DOF Asimov-1, balancing through randomized upper-body motion.

The blind lower-body framework (asimov_velocity_upper_body_blind_amp): the policy drives the
12 leg joints and sees only them; the waist, arms, 6DOF wrists and neck follow sampled motion
it must balance against. The grippers are locked and not controlled. The IMU is read on the
torso (waist_yaw_link). No hand payload.
"""

import re

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from cyclotron.assets.robots.asimov_1 import (
    ASIMOV_1_27DOF_DELAYED_CFG,
    ASIMOV_1_27DOF_EXTRA_JOINT_NAMES,
    ASIMOV_1_27DOF_JOINT_NAMES,
    ASIMOV_1_ACTION_SCALE,
    asimov_1_27dof_urdf_path,
)

from . import mdp
from .amp_env_cfg import Asimov1AmpEnvCfg, Asimov1AmpEnvCfg_PLAY
from .velocity_env_cfg import TORSO_BODY, Asimov1VelocityEnvCfg

ASIMOV_1_LOWER_BODY_JOINT_NAMES = ASIMOV_1_27DOF_JOINT_NAMES[:12]
# Waist, both arms (including the 6DOF wrists) and neck: driven by the motion sampler.
ASIMOV_1_27DOF_UPPER_BODY_JOINT_NAMES = ASIMOV_1_27DOF_JOINT_NAMES[12:]

# The IMU is mounted on the torso, above the waist yaw joint, not on the pelvis.
IMU_BODY = TORSO_BODY

SELF_COLLISION_BODIES = tuple(
    f"{side}_{joint}_link"
    for side in ("left", "right")
    for joint in ("wrist_yaw", "elbow", "knee", "ankle_roll")
)
SELF_COLLISION_SENSORS = tuple(f"self_collision_{body}" for body in SELF_COLLISION_BODIES)


def _joints_cfg(names) -> SceneEntityCfg:
    return SceneEntityCfg("robot", joint_names=list(names), preserve_order=True)


def _imu_cfg() -> SceneEntityCfg:
    return SceneEntityCfg("robot", body_names=[IMU_BODY])


def _lower_body_patterns(values: dict) -> dict:
    return {
        pattern: value
        for pattern, value in values.items()
        if any(re.fullmatch(pattern, name) for name in ASIMOV_1_LOWER_BODY_JOINT_NAMES)
    }


def _self_collision_cfg(body: str) -> ContactSensorCfg:
    # PhysX filtered contact reports require one sensor body per environment
    # and one body per filter expression, not a many-to-many regex.
    targets = ["pelvis_link", "waist_yaw_link"] + [
        f"{side}_{joint}_link"
        for side in ("left", "right")
        for joint in ("hip_pitch", "hip_roll", "hip_yaw", "shoulder_roll")
    ]
    return ContactSensorCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{body}",
        filter_prim_paths_expr=[f"{{ENV_REGEX_NS}}/Robot/{target}" for target in targets],
    )


def _configure_27dof_lower_body(env_cfg: Asimov1VelocityEnvCfg):
    scene = env_cfg.scene
    scene.robot = ASIMOV_1_27DOF_DELAYED_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    scene.robot.spawn.asset_path = asimov_1_27dof_urdf_path()
    scene.self_collision = None
    for name, body in zip(SELF_COLLISION_SENSORS, SELF_COLLISION_BODIES):
        setattr(scene, name, _self_collision_cfg(body))
    env_cfg.rewards.self_collisions.params["sensor_name"] = SELF_COLLISION_SENSORS

    # Keep the base action's ankle scales/clipping, restricted to the legs.
    action = env_cfg.actions.joint_pos
    action.joint_names = list(ASIMOV_1_LOWER_BODY_JOINT_NAMES)
    for field in ("scale", "offset", "clip"):
        values = getattr(action, field)
        if isinstance(values, dict):
            setattr(action, field, _lower_body_patterns(values))

    # Actor: torso IMU, command, leg joints and the last leg action.
    policy = env_cfg.observations.policy
    policy.base_ang_vel.params["asset_cfg"] = _imu_cfg()
    policy.projected_gravity.params["asset_cfg"] = _imu_cfg()
    for quantity in ("joint_pos", "joint_vel"):
        for slot in ("01", "23", "45"):
            term = getattr(policy, f"{quantity}_slot{slot}")
            term.params["asset_cfg"] = _joints_cfg(
                name for name in term.params["asset_cfg"].joint_names if name in ASIMOV_1_LOWER_BODY_JOINT_NAMES
            )
    policy.actions = ObsTerm(func=mdp.last_action)

    # Privileged critic: the torso IMU without delay, and every controlled joint, including
    # the externally driven neck and 6DOF wrists.
    critic = env_cfg.observations.critic
    critic.base_ang_vel = ObsTerm(func=mdp.body_ang_vel_b, params={"asset_cfg": _imu_cfg()}, scale=0.25)
    critic.projected_gravity = ObsTerm(func=mdp.body_projected_gravity_b, params={"asset_cfg": _imu_cfg()})
    critic.joint_pos_slot67 = ObsTerm(
        func=mdp.joint_pos_rel, params={"asset_cfg": _joints_cfg(ASIMOV_1_27DOF_EXTRA_JOINT_NAMES)}
    )
    critic.joint_vel_slot67 = ObsTerm(
        func=mdp.joint_vel_rel, params={"asset_cfg": _joints_cfg(ASIMOV_1_27DOF_EXTRA_JOINT_NAMES)}
    )

    env_cfg.observations.amp.joint_pos.params["asset_cfg"] = _joints_cfg(ASIMOV_1_LOWER_BODY_JOINT_NAMES)
    env_cfg.observations.amp.joint_vel.params["asset_cfg"] = _joints_cfg(ASIMOV_1_LOWER_BODY_JOINT_NAMES)

    # Do not penalize externally commanded upper-body motion against the standing pose.
    env_cfg.rewards.pose.params["asset_cfg"] = _joints_cfg(ASIMOV_1_LOWER_BODY_JOINT_NAMES)
    for key in ("std_standing", "std_walking", "std_running"):
        env_cfg.rewards.pose.params[key] = _lower_body_patterns(env_cfg.rewards.pose.params[key])
    env_cfg.rewards.joint_deviation_l1.params["asset_cfg"] = _joints_cfg(ASIMOV_1_LOWER_BODY_JOINT_NAMES)


# Coordinated sampler settings of the blind lower-body run (2026-10-17, 16k envs). Joints a
# preset leaves out (here the neck and 6DOF wrists) take their default pose plus jitter.
_UPPER_BODY_MOTION_CONFIG = {
    "sampling_weights": (0.5, 0.2, 0.2, 0.1),  # independent / mirrored / preset / neutral
    "transition_time_range_s": (1.0, 3.0),
    "hold_time_range_s": (1.0, 6.0),
    "max_joint_speed": 2.0,  # rad/s; long moves extend the sampled transition time
    "limit_margin": 0.02,  # rad inside the articulation's existing soft limits
    "waist_yaw_range": (-0.5, 0.5),
    "pose_jitter": 0.1,
    "neutral_range": ASIMOV_1_ACTION_SCALE,
    "pose_presets": (
        # Raised arms, bent elbows, wrists brought inward with joint-limit margin.
        {
            "right_shoulder_pitch_joint": 1.4,
            "left_shoulder_pitch_joint": -1.4,
            "right_shoulder_roll_joint": 0.15,
            "left_shoulder_roll_joint": -0.15,
            "right_shoulder_yaw_joint": -1.35,
            "left_shoulder_yaw_joint": 1.35,
            "right_elbow_joint": -1.48,
            "left_elbow_joint": 1.48,
        },
        # Forward reach.
        {
            "right_shoulder_pitch_joint": 1.4,
            "left_shoulder_pitch_joint": -1.4,
            "right_shoulder_roll_joint": 0.15,
            "left_shoulder_roll_joint": -0.15,
            "right_elbow_joint": -0.3,
            "left_elbow_joint": 0.3,
        },
        # Overhead reach.
        {
            "right_shoulder_pitch_joint": 2.7,
            "left_shoulder_pitch_joint": -2.7,
            "right_shoulder_roll_joint": 0.2,
            "left_shoulder_roll_joint": -0.2,
            "right_elbow_joint": -0.4,
            "left_elbow_joint": 0.4,
        },
        # Arms spread sideways.
        {
            "right_shoulder_roll_joint": 1.3,
            "left_shoulder_roll_joint": -1.3,
            "right_elbow_joint": -0.5,
            "left_elbow_joint": 0.5,
        },
    ),
}


@configclass
class Asimov1LowerBody27DofAmpEnvCfg(Asimov1AmpEnvCfg):
    """45 actor inputs, 12 leg actions, and a full-body privileged critic."""

    def __post_init__(self):
        super().__post_init__()
        _configure_27dof_lower_body(self)
        self.events.upper_body_pd_targets = EventTerm(
            func=mdp.randomize_upper_body_motion,
            mode="interval",
            interval_range_s=(0.0, 0.0),
            params={
                "asset_cfg": _joints_cfg(ASIMOV_1_27DOF_UPPER_BODY_JOINT_NAMES),
                "motion_config": _UPPER_BODY_MOTION_CONFIG,
            },
        )


@configclass
class Asimov1LowerBody27DofAmpEnvCfg_PLAY(Asimov1AmpEnvCfg_PLAY):
    """Same actor inputs for evaluation, with the upper body held at its default pose."""

    def __post_init__(self):
        super().__post_init__()
        _configure_27dof_lower_body(self)
        self.events.reset_upper_body_pd_targets = EventTerm(
            func=mdp.randomize_joint_position_targets,
            mode="reset",
            params={
                "asset_cfg": _joints_cfg(ASIMOV_1_27DOF_UPPER_BODY_JOINT_NAMES),
                "position_range": (0.0, 0.0),
            },
        )
