# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Asimov-1 AMP configurations."""

from pathlib import Path

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from cyclotron.assets.robots.asimov_1 import ASIMOV_1_JOINT_NAMES

from . import mdp
from .velocity_env_cfg import Asimov1VelocityEnvCfg, Asimov1VelocityEnvCfg_PLAY

_AMP_JOINTS_CFG = SceneEntityCfg(
    "robot", joint_names=list(ASIMOV_1_JOINT_NAMES), preserve_order=True
)

ASIMOV_1_AMP_OBS_TERMS = ["joint_pos", "joint_vel"]

_MOTIONS_DIR = Path(__file__).resolve().parent / "motions"

ASIMOV_1_MOTION_FILES = [
    str(_MOTIONS_DIR / "policy_delay_walk_slow.npz"),
]

ASIMOV_1_KEY_BODY_NAMES = [
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
    "waist_yaw_link",
]
ASIMOV_1_ANCHOR_NAME = "pelvis_link"


@configclass
class AmpObsCfg(ObsGroup):

    joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": _AMP_JOINTS_CFG})
    joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": _AMP_JOINTS_CFG})

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True
        term_names = [name for name, value in self.__dict__.items() if isinstance(value, ObsTerm)]
        assert term_names == ASIMOV_1_AMP_OBS_TERMS, (
            f"AMP obs group terms {term_names} drifted from ASIMOV_1_AMP_OBS_TERMS "
            f"{ASIMOV_1_AMP_OBS_TERMS}; the discriminator's policy and expert features "
            "would be misaligned."
        )


@configclass
class Asimov1AmpEnvCfg(Asimov1VelocityEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations.amp = AmpObsCfg()


@configclass
class Asimov1AmpEnvCfg_PLAY(Asimov1VelocityEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        self.observations.amp = AmpObsCfg()
