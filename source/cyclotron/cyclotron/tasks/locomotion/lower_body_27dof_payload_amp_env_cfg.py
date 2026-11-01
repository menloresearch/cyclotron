# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""27-DOF leg-only AMP policy carrying a 5-8 kg load split between both hands.

Same actor/critic I/O as ``lower_body_27dof_amp_env_cfg``. The payload follows the 25-DOF
lower-body payload run (2026-10-21): a mass/COM/inertia proxy added to each hand link at reset,
not a separate object or grasp, with a reset-time torque-capacity derating. Loaded environments
switch the upper body to joint-space carrying poses.
"""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from . import mdp
from .lower_body_27dof_amp_env_cfg import (
    _UPPER_BODY_MOTION_CONFIG,
    Asimov1LowerBody27DofAmpEnvCfg,
    Asimov1LowerBody27DofAmpEnvCfg_PLAY,
)

# Distal arm links; the locked gripper links are merged into these.
HAND_BODIES = ["left_6dof_link", "right_6dof_link"]


def _hand_payload_params(**overrides) -> dict:
    return {
        "asset_cfg": SceneEntityCfg("robot", body_names=HAND_BODIES, preserve_order=True),
        "min_total_mass": 5.0,
        "max_total_mass": 8.0,
        "empty_probability": 0.3,
        "left_fraction_range": (0.4, 0.6),
        "com_offset": 0.05,
        "radius_range": (0.04, 0.08),
        "initial_scale": 0.2,
        "curriculum_steps": 48_000,
        **overrides,
    }


# Representative joint-space carrying poses. These do not impose a rigid two-hand grasp
# constraint; the 6DOF wrists take their default pose plus jitter.
_CARRYING_MOTION_CONFIG = {
    **_UPPER_BODY_MOTION_CONFIG,
    "sampling_weights": (0.0, 0.0, 1.0, 0.0),
    "transition_time_range_s": (2.0, 4.0),
    "hold_time_range_s": (4.0, 10.0),
    "max_joint_speed": 0.75,
    "max_joint_acceleration": 1.5,
    "pose_jitter": 0.05,
    "pose_presets": tuple(
        {
            "right_shoulder_pitch_joint": pitch,
            "left_shoulder_pitch_joint": -pitch,
            "right_shoulder_roll_joint": 0.15,
            "left_shoulder_roll_joint": -0.15,
            "right_shoulder_yaw_joint": 0.0,
            "left_shoulder_yaw_joint": 0.0,
            "right_elbow_joint": -elbow,
            "left_elbow_joint": elbow,
            "neck_yaw_joint": neck_yaw,
        }
        for pitch, elbow, neck_yaw in ((0.6, 1.2, -0.5), (0.9, 1.0, 0.0), (1.2, 0.7, 0.5))
    ),
}


@configclass
class Asimov1LowerBody27DofPayloadAmpEnvCfg(Asimov1LowerBody27DofAmpEnvCfg):
    """45 actor inputs, 12 leg actions, a full-body privileged critic, and a two-hand payload."""

    def __post_init__(self):
        super().__post_init__()
        self.events.qpos0_rand.params["ranges"] = (-0.03, 0.03)
        self.events.hand_payload = EventTerm(
            func=mdp.randomize_hand_payload, mode="reset", params=_hand_payload_params()
        )
        self.events.torque_capacity = EventTerm(
            func=mdp.randomize_torque_capacity, mode="reset",
            params={"asset_cfg": SceneEntityCfg("robot"), "scale_range": (0.8, 1.0)},
        )
        self.events.upper_body_pd_targets.params["carrying_motion_config"] = _CARRYING_MOTION_CONFIG
        self.rewards.angular_momentum.params["mass_event_name"] = "hand_payload"
        # Allow balance corrections under a sustained load, including at zero command.
        self.rewards.pose.params["std_standing"] = {
            ".*_hip_.*": 0.15, ".*_knee_joint": 0.2, ".*_ankle_pitch_joint": 0.1, ".*_ankle_roll_joint": 0.08,
        }
        self.rewards.pose.params["pose_weight_slow"] = 1.0
        self.rewards.joint_deviation_l1.weight = -0.25


@configclass
class Asimov1LowerBody27DofPayloadAmpEnvCfg_PLAY(Asimov1LowerBody27DofAmpEnvCfg_PLAY):
    """Evaluate with the full 5-8 kg load in every environment, upper body at its default pose."""

    def __post_init__(self):
        super().__post_init__()
        self.events.hand_payload = EventTerm(
            func=mdp.randomize_hand_payload,
            mode="reset",
            params=_hand_payload_params(empty_probability=0.0, curriculum_steps=0),
        )
