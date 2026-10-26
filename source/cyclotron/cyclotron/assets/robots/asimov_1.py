# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Asimov-1 configurations."""

import os
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from cyclotron.gripper_lock import spawn_gripper_locked_urdf

_REPOSITORY_ROOT = Path(__file__).resolve().parents[5]
ASIMOV_1_MODEL_DIR = str(_REPOSITORY_ROOT / "third_party" / "asimov-1" / "sim-model")
ASIMOV_1_URDF_PATH = str(
    Path(os.environ.get("ASIMOV_1_MODEL_DIR", ASIMOV_1_MODEL_DIR)).expanduser() / "urdf" / "asimov_1.urdf"
)


DELAY_MIN_LAG = 0
DELAY_MAX_LAG = 5
ASIMOV_1_ACTION_SCALE = 0.25


ASIMOV_1_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_yaw_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_yaw_joint",
]


ASIMOV_1_ACTUATORS = {
    "hip_pitch": DelayedPDActuatorCfg(
        joint_names_expr=[".*_hip_pitch_joint"],
        stiffness=150.0,
        damping=5.0,
        effort_limit=45.0,
        armature=0.0698,
        friction=0.70,
        min_delay=DELAY_MIN_LAG,
        max_delay=DELAY_MAX_LAG,
    ),
    "hip_roll": DelayedPDActuatorCfg(
        joint_names_expr=[".*_hip_roll_joint"],
        stiffness=150.0,
        damping=5.0,
        effort_limit=45.0,
        armature=0.1400,
        friction=0.20,
        min_delay=DELAY_MIN_LAG,
        max_delay=DELAY_MAX_LAG,
    ),
    "hip_yaw": DelayedPDActuatorCfg(
        joint_names_expr=[".*_hip_yaw_joint"],
        stiffness=150.0,
        damping=5.0,
        effort_limit=28.0,
        armature=0.0687,
        friction=0.70,
        min_delay=DELAY_MIN_LAG,
        max_delay=DELAY_MAX_LAG,
    ),
    "knee": DelayedPDActuatorCfg(
        joint_names_expr=[".*_knee_joint"],
        stiffness=150.0,
        damping=5.0,
        effort_limit=45.0,
        armature=0.0330,
        friction=0.70,
        min_delay=DELAY_MIN_LAG,
        max_delay=DELAY_MAX_LAG,
    ),
    "ankle_pitch": DelayedPDActuatorCfg(
        joint_names_expr=[".*_ankle_pitch_joint"],
        stiffness=224.42,
        damping=10.201,
        effort_limit=72.0,
        armature=0.42984,
        friction=0.40,
        min_delay=DELAY_MIN_LAG,
        max_delay=DELAY_MAX_LAG,
    ),
    "ankle_roll": DelayedPDActuatorCfg(
        joint_names_expr=[".*_ankle_roll_joint"],
        stiffness=35.20,
        damping=1.600,
        effort_limit=18.0,
        armature=0.05814,
        friction=0.40,
        min_delay=DELAY_MIN_LAG,
        max_delay=DELAY_MAX_LAG,
    ),
    "waist": DelayedPDActuatorCfg(
        joint_names_expr=["waist_yaw_joint"],
        stiffness=65.0,
        damping=5.0,
        effort_limit=40.0,
        armature=0.0698,
        friction=0.70,
        min_delay=DELAY_MIN_LAG,
        max_delay=DELAY_MAX_LAG,
    ),
    "shoulder_pitch": DelayedPDActuatorCfg(
        joint_names_expr=[".*_shoulder_pitch_joint"],
        stiffness=57.0,
        damping=5.0,
        effort_limit=30.0,
        armature=0.1400,
        friction=0.20,
        min_delay=DELAY_MIN_LAG,
        max_delay=DELAY_MAX_LAG,
    ),
    "shoulder_roll": DelayedPDActuatorCfg(
        joint_names_expr=[".*_shoulder_roll_joint"],
        stiffness=86.0,
        damping=5.0,
        effort_limit=25.0,
        armature=0.0330,
        friction=0.70,
        min_delay=DELAY_MIN_LAG,
        max_delay=DELAY_MAX_LAG,
    ),
    "shoulder_yaw": DelayedPDActuatorCfg(
        joint_names_expr=[".*_shoulder_yaw_joint"],
        stiffness=96.0,
        damping=5.0,
        effort_limit=20.0,
        armature=0.0687,
        friction=0.70,
        min_delay=DELAY_MIN_LAG,
        max_delay=DELAY_MAX_LAG,
    ),
    "elbow_wrist": DelayedPDActuatorCfg(
        joint_names_expr=[".*_elbow_joint", ".*_wrist_yaw_joint"],
        stiffness=40.0,
        damping=2.0,
        effort_limit=12.0,
        armature=0.0242,
        friction=0.40,
        min_delay=DELAY_MIN_LAG,
        max_delay=DELAY_MAX_LAG,
    ),
}


ASIMOV_1_STANDING_INIT_STATE = ArticulationCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.639),
    joint_pos={
        "left_hip_pitch_joint": -0.15,
        "right_hip_pitch_joint": 0.15,
        ".*_hip_roll_joint": 0.0,
        ".*_hip_yaw_joint": 0.0,
        "left_knee_joint": 0.45,
        "right_knee_joint": -0.45,
        "left_ankle_pitch_joint": -0.30,
        "right_ankle_pitch_joint": 0.30,
        ".*_ankle_roll_joint": 0.0,
        "waist_yaw_joint": 0.0,
        "left_shoulder_pitch_joint": -0.25,
        "right_shoulder_pitch_joint": 0.25,
        "left_shoulder_roll_joint": -0.05,
        "right_shoulder_roll_joint": 0.05,
        ".*_shoulder_yaw_joint": 0.0,
        "left_elbow_joint": 0.40,
        "right_elbow_joint": -0.40,
        ".*_wrist_yaw_joint": 0.0,
    },
    joint_vel={".*": 0.0},
)


ASIMOV_1_DELAYED_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=ASIMOV_1_URDF_PATH,
        fix_base=False,
        merge_fixed_joints=True,
        joint_drive=sim_utils.UrdfFileCfg.JointDriveCfg(
            target_type="position",
            gains=sim_utils.UrdfFileCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
        ),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ASIMOV_1_STANDING_INIT_STATE,
    soft_joint_pos_limit_factor=0.9,
    actuators=ASIMOV_1_ACTUATORS,
)


# 27-DOF model: the 23 joints above, the neck, a sixth joint on each arm, and grippers.
# The grippers are not controlled: their joints are locked at zero when the URDF is
# imported, keeping their links, mass and collision geometry.
ASIMOV_1_27DOF_MODEL = {
    "repo": "menloresearch/asimov-1",
    "commit": "96a6807c163e4e6cabdc124eefcc45b0ae2cb57a",
    "urdf_filepath": "sim-model/asimov_1_27dof/urdf/asimov_1_27dof.urdf",
    "sha256": "03a5e87ac971b944cbc1a36cbdada9ed3b6d9aeb2ba519df91eefd8fea86b46c",
    "dirty": False,
}
_ASIMOV_1_27DOF_LOCAL_URDF = _REPOSITORY_ROOT / "third_party" / "asimov-1" / ASIMOV_1_27DOF_MODEL["urdf_filepath"]

ASIMOV_1_27DOF_EXTRA_JOINT_NAMES = [
    "neck_yaw_joint",
    "neck_pitch_joint",
    "right_6DOF_joint",
    "left_6dof_joint",
]
ASIMOV_1_27DOF_JOINT_NAMES = ASIMOV_1_JOINT_NAMES + ASIMOV_1_27DOF_EXTRA_JOINT_NAMES


def asimov_1_27dof_urdf_path() -> str:
    """The 27-DOF URDF: ``ASIMOV_1_27DOF_URDF_PATH`` if set, else the copy in third_party/asimov-1 if present,
    else asimov-1 at the pinned commit, fetched once into cyclotron's cache."""
    if "ASIMOV_1_27DOF_URDF_PATH" in os.environ:
        return str(Path(os.environ["ASIMOV_1_27DOF_URDF_PATH"]).expanduser())
    if _ASIMOV_1_27DOF_LOCAL_URDF.is_file():
        return str(_ASIMOV_1_27DOF_LOCAL_URDF)
    from cyclotron.robot_model import materialize

    return materialize(ASIMOV_1_27DOF_MODEL)


ASIMOV_1_27DOF_ACTUATORS = {
    **ASIMOV_1_ACTUATORS,
    # Initial simulation gains for the new joints, on the small-joint motor model;
    # effort is the URDF limit. These are not hardware-calibrated.
    "neck": DelayedPDActuatorCfg(
        joint_names_expr=["neck_yaw_joint", "neck_pitch_joint"],
        stiffness=40.0,
        damping=2.0,
        effort_limit=40.0,
        armature=0.0242,
        friction=0.40,
        min_delay=DELAY_MIN_LAG,
        max_delay=DELAY_MAX_LAG,
    ),
    "arm_6dof": DelayedPDActuatorCfg(
        joint_names_expr=["right_6DOF_joint", "left_6dof_joint"],
        stiffness=40.0,
        damping=2.0,
        effort_limit=12.0,
        armature=0.0242,
        friction=0.40,
        min_delay=DELAY_MIN_LAG,
        max_delay=DELAY_MAX_LAG,
    ),
}


ASIMOV_1_27DOF_STANDING_INIT_STATE = ASIMOV_1_STANDING_INIT_STATE.replace(
    joint_pos={**ASIMOV_1_STANDING_INIT_STATE.joint_pos, **{name: 0.0 for name in ASIMOV_1_27DOF_EXTRA_JOINT_NAMES}}
)


# The asset path is resolved by the task (``asimov_1_27dof_urdf_path``), so importing
# this module never downloads anything.
ASIMOV_1_27DOF_DELAYED_CFG = ASIMOV_1_DELAYED_CFG.replace(
    spawn=ASIMOV_1_DELAYED_CFG.spawn.replace(func=spawn_gripper_locked_urdf, asset_path=None),
    init_state=ASIMOV_1_27DOF_STANDING_INIT_STATE,
    actuators=ASIMOV_1_27DOF_ACTUATORS,
)
