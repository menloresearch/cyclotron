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
        stiffness=110.0,
        damping=5.0,
        effort_limit=40.0,
        armature=0.0484,
        friction=0.40,
        min_delay=DELAY_MIN_LAG,
        max_delay=DELAY_MAX_LAG,
    ),
    "ankle_roll": DelayedPDActuatorCfg(
        joint_names_expr=[".*_ankle_roll_joint"],
        stiffness=110.0,
        damping=5.0,
        effort_limit=17.0,
        armature=0.0484,
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


# ---------------------------------------------------------------------------------------------------------------------
# Get-up asset (additive: nothing above this line is changed, the walking configs are untouched).
# Patched collision geometry (see getup_urdf.py) plus limpable, sync-free delayed PD actuators.
# ---------------------------------------------------------------------------------------------------------------------

import copy  # noqa: E402
import dataclasses  # noqa: E402
import warnings  # noqa: E402

from .getup_actuators import (  # noqa: E402
    ANKLE_K_PITCH,
    ANKLE_K_ROLL,
    MOTOR_DATASHEET,
    DelayedDCMotorLimpableCfg,
    DelayedPDLimpableActuatorCfg,
    no_load_speed,
)
from .getup_urdf import DEFAULT_CACHE_ROOT, build_getup_urdf  # noqa: E402

ASIMOV_1_JOINT_MOTOR: dict[str, str] = {
    ".*_hip_pitch_joint": "EC-A6416-P2-25",
    "waist_yaw_joint": "EC-A6416-P2-25",
    ".*_hip_roll_joint": "EC-A5013-H17-100",
    ".*_shoulder_pitch_joint": "EC-A5013-H17-100",
    ".*_hip_yaw_joint": "EC-A3814-H14-107",
    ".*_shoulder_yaw_joint": "EC-A3814-H14-107",
    ".*_knee_joint": "EC-A4315-P2-36",
    ".*_shoulder_roll_joint": "EC-A4315-P2-36",
    ".*_ankle_pitch_joint": "EC-A4310-P2-36",  # two motors per ankle, differential
    ".*_ankle_roll_joint": "EC-A4310-P2-36",  # two motors per ankle, differential
    ".*_elbow_joint": "EC-A4310-P2-36",
    ".*_wrist_yaw_joint": "EC-A4310-P2-36",
}
"""Motor model driving each joint (ENCOS EC-series part numbers; values from the motor datasheets)."""

ASIMOV_1_ANKLE_MOTOR_RATED_TORQUE = MOTOR_DATASHEET["EC-A4310-P2-36"]["rated_torque"]
"""Rated torque of one ankle motor [N m]. Prefer this with ``ankle_motor_torques`` for the ankle thermal proxy."""


def _joint_torque(key: str, which: str) -> float:
    motor = MOTOR_DATASHEET[ASIMOV_1_JOINT_MOTOR[key]][which]
    # joint-space equivalent through the ankle differential (virtual work, both motors equally loaded)
    if "ankle_pitch" in key:
        return 2.0 * ANKLE_K_PITCH * motor
    if "ankle_roll" in key:
        return 2.0 * ANKLE_K_ROLL * motor
    return motor


ASIMOV_1_RATED_TORQUE: dict[str, float] = {k: _joint_torque(k, "rated_torque") for k in ASIMOV_1_JOINT_MOTOR}
"""Datasheet rated (continuous) joint torque [N m] per joint-name regex, for the thermal proxy.

Ankle entries are joint-space equivalents (pitch 48.5, roll 19.2) that assume both motors share the load; the exact
per-motor check is ``ankle_motor_torques(tau_p, tau_r)`` against ``ASIMOV_1_ANKLE_MOTOR_RATED_TORQUE`` (12 N m).
Note several sim effort limits exceed rated torque (knee 45 vs 25, hip roll 45 vs 30), so sustained torques near the
sim limit are a real thermal risk.
"""

ASIMOV_1_PEAK_TORQUE: dict[str, float] = {k: _joint_torque(k, "peak_torque") for k in ASIMOV_1_JOINT_MOTOR}
"""Datasheet peak/stall joint torque [N m] per joint-name regex (ankle: joint-space equivalents)."""

ASIMOV_1_PEAK_SPEED: dict[str, float] = {k: MOTOR_DATASHEET[m]["peak_speed"] for k, m in ASIMOV_1_JOINT_MOTOR.items()}
"""Datasheet peak joint speed [rad/s] per joint-name regex. Identical to the URDF ``<limit velocity>`` values (the URDF
applies the ankle *motor* speed 9.32 rad/s to the ankle joints as well)."""


def _limpable(src: DelayedPDActuatorCfg, cls=DelayedPDLimpableActuatorCfg, **overrides):
    kw = {f.name: copy.deepcopy(getattr(src, f.name)) for f in dataclasses.fields(src) if f.name != "class_type"}
    kw.update(overrides)
    return cls(**kw)


ASIMOV_1_GETUP_GROUPED_ACTUATORS: dict[str, DelayedPDLimpableActuatorCfg] = {
    name: _limpable(cfg) for name, cfg in ASIMOV_1_ACTUATORS.items()
}
"""The walking groups (identical gains, effort limits, delays, armature and friction) as limpable delayed PD, one
actuator per group. Kept for reference/tests; the get-up asset uses the merged :data:`ASIMOV_1_GETUP_ACTUATORS`."""

_PER_JOINT_FIELDS = ("stiffness", "damping", "effort_limit", "armature", "friction")


def _merge_groups(groups: dict[str, DelayedPDActuatorCfg], cls=DelayedPDLimpableActuatorCfg, **extra):
    """One actuator for all joints: per-joint values as regex dicts, the groups' independent lags as delay_groups."""
    exprs, delay_groups = [], []
    kw = {f: {} for f in _PER_JOINT_FIELDS}
    min_d = {g.min_delay for g in groups.values()}
    max_d = {g.max_delay for g in groups.values()}
    if len(min_d) != 1 or len(max_d) != 1:
        raise ValueError("cannot merge actuator groups with different delay ranges")
    for g in groups.values():
        delay_groups.append(list(g.joint_names_expr))
        for e in g.joint_names_expr:
            exprs.append(e)
            for f in _PER_JOINT_FIELDS:
                v = getattr(g, f)
                if not isinstance(v, (int, float)):
                    raise ValueError(f"cannot merge non-scalar {f}={v!r}")
                kw[f][e] = float(v)
    return cls(
        joint_names_expr=exprs,
        min_delay=min_d.pop(),
        max_delay=max_d.pop(),
        delay_groups=delay_groups,
        **kw,
        **extra,
    )


ASIMOV_1_GETUP_ACTUATORS: dict[str, DelayedPDLimpableActuatorCfg] = {"all": _merge_groups(ASIMOV_1_ACTUATORS)}
"""Stage A actuators: ONE limpable delayed-PD actuator covering all 23 joints, with per-joint gains, effort limits,
armature and friction equal to the walking groups (``ASIMOV_1_ACTUATORS``), and the walking groups kept as
``delay_groups`` (one random lag per env and group in 0..5 physics steps, as with one ``DelayedPDActuator`` per group).
One group instead of 11 cuts the per-substep Python overhead; the sync-free delay removes the GPU->CPU syncs.
``velocity_limit_sim`` is left to the URDF values exactly as in walking (the importer writes them and PhysX enforces
them: hip roll / shoulder pitch capped at 3.98 rad/s)."""

_ANKLE_V0 = no_load_speed("EC-A4310-P2-36")


def _dc_params() -> tuple[dict[str, float], dict[str, float]]:
    import re

    sat, v0 = {}, {}
    for cfg in ASIMOV_1_ACTUATORS.values():
        for expr in cfg.joint_names_expr:
            if "ankle_pitch" in expr:
                v0[expr] = _ANKLE_V0 / ANKLE_K_PITCH  # unused (ankle modelled per motor) but must be set
                continue
            if "ankle_roll" in expr:
                v0[expr] = _ANKLE_V0 / ANKLE_K_ROLL
                continue
            (motor,) = [m for k, m in ASIMOV_1_JOINT_MOTOR.items() if re.fullmatch(k, expr) or k == expr]
            sat[expr] = MOTOR_DATASHEET[motor]["peak_torque"]
            v0[expr] = no_load_speed(motor)
    return sat, v0


_DC_SAT, _DC_V0 = _dc_params()

ASIMOV_1_GETUP_DC_ACTUATORS: dict[str, DelayedDCMotorLimpableCfg] = {
    "all": _merge_groups(
        ASIMOV_1_ACTUATORS, cls=DelayedDCMotorLimpableCfg, saturation_effort=_DC_SAT, velocity_limit=_DC_V0
    )
}
"""Stage B actuators: the merged actuator with torque-speed curves (stall = datasheet peak torque, no-load speed =
line through the peak and rated points) and the two ankle motors of each ankle clipped in motor space (both ankle
joints are in the one group). UNVERIFIED curve shape; see ``getup_actuators.no_load_speed``."""


try:
    ASIMOV_1_GETUP_URDF_PATH = build_getup_urdf(ASIMOV_1_URDF_PATH)
except Exception as _err:  # never break the walking task because of the get-up asset
    warnings.warn(f"[asimov_1] Could not build the get-up URDF ({_err!r}); ASIMOV_1_GETUP_CFG will fail to spawn.")
    ASIMOV_1_GETUP_URDF_PATH = str(DEFAULT_CACHE_ROOT / "BUILD_FAILED" / "asimov_1_getup.urdf")


ASIMOV_1_GETUP_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=ASIMOV_1_GETUP_URDF_PATH,
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
            # 1 m/s = at most 5 mm of depenetration per 5 ms substep: no popping when resting on the ground or when a
            # limb is pinned under the body. Resets must therefore be penetration-free (cache states / FK lift).
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            # TGS 8/4 as walking: 64-env limp drop tests settle the same at 8 and 12 position iterations (95 % by 6 s,
            # the rest are limp limbs still swinging in the air), so 12 buys nothing; cost is ~linear in it.
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
            # never let a robot lying still fall asleep (it would then ignore actuator torques until woken)
            sleep_threshold=0.0,
        ),
    ),
    init_state=ASIMOV_1_STANDING_INIT_STATE,
    # hip pitch must reach 2.09 rad and knee 1.5 rad for the deep crouch (0.98 -> 2.05 / 1.47)
    soft_joint_pos_limit_factor=0.98,
    actuators=ASIMOV_1_GETUP_ACTUATORS,
)
"""Get-up robot: patched URDF (wrist stub, upper arm and body-shell boxes), limpable delayed PD actuators
(walking gains). The shapes are listed in ``getup_urdf.SHELL_BOXES``."""

ASIMOV_1_GETUP_DC_CFG = ASIMOV_1_GETUP_CFG.replace(actuators=ASIMOV_1_GETUP_DC_ACTUATORS)
"""Stage B variant of :data:`ASIMOV_1_GETUP_CFG` with :class:`DelayedDCMotorLimpable` actuators."""
