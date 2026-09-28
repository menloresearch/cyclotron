# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Frozen constants, duplicated as plain Python literals so this package needs no ``isaaclab``/``torch`` import.

Every value here is cross-checked against a specific source file; see the comment above each block. If Isaac's own
source changes, re-run `capture_isaac_obs.py` / `compare_isaac.py` (needs Isaac Lab) to check against this file.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------------------------------------------------
# Joint order (source/isaac_asimov/isaac_asimov/assets/robots/asimov_1.py: ASIMOV_1_JOINT_NAMES).
# Verified identical to the MJCF's own <joint> declaration order (third_party/asimov-1/sim-model/xmls/asimov_1.xml),
# so joint mapping between MuJoCo and Isaac can go purely by name (mj_name2id), with this order used only to build
# per-joint arrays / slot indices.
# ---------------------------------------------------------------------------------------------------------------------

ASIMOV_1_JOINT_NAMES: list[str] = [
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
NUM_JOINTS = len(ASIMOV_1_JOINT_NAMES)  # 23
JOINT_INDEX = {name: i for i, name in enumerate(ASIMOV_1_JOINT_NAMES)}

FEET_BODIES: tuple[str, str] = ("left_ankle_roll_link", "right_ankle_roll_link")
TORSO_BODY = "waist_yaw_link"
PELVIS_BODY = "pelvis_link"
IMU_SITE = "imu_in_pelvis"

# ---------------------------------------------------------------------------------------------------------------------
# Actuator groups (source/isaac_asimov/isaac_asimov/assets/robots/asimov_1.py: ASIMOV_1_ACTUATORS).
# Get-up (Stage A, ASIMOV_1_GETUP_ACTUATORS in getup_actuators.py::_limpable) uses IDENTICAL numeric fields (gains,
# effort limits, delays, armature, friction) to the walking groups below -- only the actuator *class* differs
# (limp-capable). So one table serves both layouts.
# min_delay/max_delay are in ISAAC PHYSICS STEPS (dt = ISAAC_PHYSICS_DT), NOT this harness's own physics dt.
# ---------------------------------------------------------------------------------------------------------------------


_RPM = 2.0 * np.pi / 60.0  # rad/s per RPM

# Datasheet peak joint speed [rad/s] (ENCOS EC-series motor datasheet values / getup_actuators.py::MOTOR_DATASHEET),
# identical to the URDF <limit velocity=...> values. PhysX DOES enforce this hard (a step test saturates hip roll /
# shoulder pitch at 3.98 rad/s), so this harness clamps qvel to it post-integration
# every physics step (MuJoCo has no native per-hinge velocity limit) -- see episode.py / init_states.py.
_VEL_HIP_PITCH_WAIST = 120 * _RPM  # EC-A6416-P2-25, ~12.566 (PhysX-measured: 12.57)
_VEL_HIP_ROLL_SH_PITCH = 38 * _RPM  # EC-A5013-H17-100, ~3.976 (PhysX-measured: 3.98)
_VEL_HIP_YAW_SH_YAW = 52 * _RPM  # EC-A3814-H14-107, ~5.445 (PhysX-measured: 5.45)
_VEL_KNEE_SH_ROLL = 117 * _RPM  # EC-A4315-P2-36, ~12.252 (PhysX-measured: 12.25)
_VEL_ANKLE_ELBOW_WRIST = 89 * _RPM  # EC-A4310-P2-36, ~9.320 (PhysX-measured: 9.32)


@dataclass(frozen=True)
class ActuatorGroup:
    pattern: str  # matched with re.fullmatch against a joint name
    stiffness: float
    damping: float
    effort_limit: float
    armature: float
    friction: float
    velocity_limit: float  # rad/s, PhysX-enforced hard clamp (see above)
    min_delay: int = 0
    max_delay: int = 5


ASIMOV_1_ACTUATOR_GROUPS: list[ActuatorGroup] = [
    ActuatorGroup(r".*_hip_pitch_joint", stiffness=150.0, damping=5.0, effort_limit=45.0, armature=0.0698, friction=0.70, velocity_limit=_VEL_HIP_PITCH_WAIST),
    ActuatorGroup(r".*_hip_roll_joint", stiffness=150.0, damping=5.0, effort_limit=45.0, armature=0.1400, friction=0.20, velocity_limit=_VEL_HIP_ROLL_SH_PITCH),
    ActuatorGroup(r".*_hip_yaw_joint", stiffness=150.0, damping=5.0, effort_limit=28.0, armature=0.0687, friction=0.70, velocity_limit=_VEL_HIP_YAW_SH_YAW),
    ActuatorGroup(r".*_knee_joint", stiffness=150.0, damping=5.0, effort_limit=45.0, armature=0.0330, friction=0.70, velocity_limit=_VEL_KNEE_SH_ROLL),
    ActuatorGroup(r".*_ankle_pitch_joint", stiffness=110.0, damping=5.0, effort_limit=40.0, armature=0.0484, friction=0.40, velocity_limit=_VEL_ANKLE_ELBOW_WRIST),
    ActuatorGroup(r".*_ankle_roll_joint", stiffness=110.0, damping=5.0, effort_limit=17.0, armature=0.0484, friction=0.40, velocity_limit=_VEL_ANKLE_ELBOW_WRIST),
    ActuatorGroup(r"waist_yaw_joint", stiffness=65.0, damping=5.0, effort_limit=40.0, armature=0.0698, friction=0.70, velocity_limit=_VEL_HIP_PITCH_WAIST),
    ActuatorGroup(r".*_shoulder_pitch_joint", stiffness=57.0, damping=5.0, effort_limit=30.0, armature=0.1400, friction=0.20, velocity_limit=_VEL_HIP_ROLL_SH_PITCH),
    ActuatorGroup(r".*_shoulder_roll_joint", stiffness=86.0, damping=5.0, effort_limit=25.0, armature=0.0330, friction=0.70, velocity_limit=_VEL_KNEE_SH_ROLL),
    ActuatorGroup(r".*_shoulder_yaw_joint", stiffness=96.0, damping=5.0, effort_limit=20.0, armature=0.0687, friction=0.70, velocity_limit=_VEL_HIP_YAW_SH_YAW),
    ActuatorGroup(r".*_elbow_joint", stiffness=40.0, damping=2.0, effort_limit=12.0, armature=0.0242, friction=0.40, velocity_limit=_VEL_ANKLE_ELBOW_WRIST),
    ActuatorGroup(r".*_wrist_yaw_joint", stiffness=40.0, damping=2.0, effort_limit=12.0, armature=0.0242, friction=0.40, velocity_limit=_VEL_ANKLE_ELBOW_WRIST),
]

# MJCF's own baked-in <joint armature="..."> values (third_party/asimov-1/sim-model/xmls/asimov_1.xml), keyed the
# same way, kept only for the `--armature-source mjcf` sensitivity check. These do NOT match ASIMOV_1_ACTUATORS
# (the checkpoint trained against the Isaac values, so those are the default).
MJCF_ARMATURE_GROUPS: dict[str, float] = {
    r".*_hip_pitch_joint": 0.095625,
    r".*_hip_roll_joint": 0.11,
    r".*_hip_yaw_joint": 0.038,
    r".*_knee_joint": 0.0339552,
    r".*_ankle_pitch_joint": 0.0565056,
    r".*_ankle_roll_joint": 0.0565056,
    r"waist_yaw_joint": 0.095625,
    r".*_shoulder_pitch_joint": 0.11,
    r".*_shoulder_roll_joint": 0.0339552,
    r".*_shoulder_yaw_joint": 0.038,
    r".*_elbow_joint": 0.0282528,
    r".*_wrist_yaw_joint": 0.0282528,
}


def _match_one(patterns: dict[str, float] | list[ActuatorGroup], joint: str):
    if isinstance(patterns, dict):
        hits = [(k, v) for k, v in patterns.items() if re.fullmatch(k, joint)]
    else:
        hits = [(g.pattern, g) for g in patterns if re.fullmatch(g.pattern, joint)]
    if len(hits) != 1:
        raise ValueError(f"joint {joint!r} matched {len(hits)} actuator-group patterns (need exactly 1): {hits}")
    return hits[0][1]


@dataclass
class JointArrays:
    """Per-joint arrays in ``ASIMOV_1_JOINT_NAMES`` order, shape (23,) float64."""

    stiffness: np.ndarray
    damping: np.ndarray
    effort_limit: np.ndarray  # nominal, 1.0x -- scale at runtime via --effort-scale
    armature: np.ndarray
    friction: np.ndarray
    velocity_limit: np.ndarray  # rad/s, PhysX-enforced hard clamp (datasheet peak speed = URDF velocity limit)
    min_delay: np.ndarray  # Isaac physics steps (int)
    max_delay: np.ndarray  # Isaac physics steps (int)

    def s_j(self, scale_torque_factor: float = 1.1) -> np.ndarray:
        """Get-up action scale ``s_j = scale_torque_factor * tau_max_j / Kp_j`` (mdp/actions.py)."""
        return scale_torque_factor * self.effort_limit / np.clip(self.stiffness, 1e-6, None)


def build_joint_arrays(armature_source: str = "isaac") -> JointArrays:
    """Build the per-joint arrays for :data:`ASIMOV_1_JOINT_NAMES`.

    Args:
        armature_source: ``"isaac"`` (default, matches what the checkpoint trained against) or ``"mjcf"`` (the
            MJCF's own baked-in values, for a sensitivity check).
    """
    n = NUM_JOINTS
    out = JointArrays(
        stiffness=np.zeros(n), damping=np.zeros(n), effort_limit=np.zeros(n),
        armature=np.zeros(n), friction=np.zeros(n), velocity_limit=np.zeros(n),
        min_delay=np.zeros(n, dtype=int), max_delay=np.zeros(n, dtype=int),
    )
    for i, name in enumerate(ASIMOV_1_JOINT_NAMES):
        g = _match_one(ASIMOV_1_ACTUATOR_GROUPS, name)
        out.stiffness[i] = g.stiffness
        out.damping[i] = g.damping
        out.effort_limit[i] = g.effort_limit
        out.friction[i] = g.friction
        out.velocity_limit[i] = g.velocity_limit
        out.min_delay[i] = g.min_delay
        out.max_delay[i] = g.max_delay
        if armature_source == "mjcf":
            out.armature[i] = _match_one(MJCF_ARMATURE_GROUPS, name)
        elif armature_source == "isaac":
            out.armature[i] = g.armature
        else:
            raise ValueError(f"armature_source must be 'isaac' or 'mjcf', got {armature_source!r}")
    return out


# ---------------------------------------------------------------------------------------------------------------------
# Simulation rates. Isaac: dt=0.005 (200 Hz), decimation=4 -> policy 50 Hz (tasks/locomotion/velocity_env_cfg.py
# Asimov1VelocityEnvCfg.__post_init__; the get-up env uses the same dt/decimation).
# ---------------------------------------------------------------------------------------------------------------------

ISAAC_PHYSICS_DT = 0.005
ISAAC_POLICY_DECIMATION = 4
ISAAC_POLICY_DT = ISAAC_PHYSICS_DT * ISAAC_POLICY_DECIMATION  # 0.02 s = 50 Hz

POLICY_DT = ISAAC_POLICY_DT  # 50 Hz policy, both layouts

MJ_PHYSICS_DT_1KHZ = 0.001
MJ_PHYSICS_DT_500HZ = 0.002
MJ_PHYSICS_DT_ISAAC = ISAAC_PHYSICS_DT  # MuJoCo at 1 kHz, 500 Hz and Isaac's own dt are all exposed via
# --physics-hz / --physics-dt on the CLI.

# ---------------------------------------------------------------------------------------------------------------------
# Walking layout (tasks/locomotion/velocity_env_cfg.py: SLOT_0_1/2_3/4_5, ActionsCfg, ObservationsCfg.PolicyCfg).
# ---------------------------------------------------------------------------------------------------------------------

SLOT_0_1: tuple[str, ...] = (
    "left_hip_pitch_joint", "left_hip_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint",
    "waist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
)
SLOT_2_3: tuple[str, ...] = (
    "left_hip_yaw_joint", "left_knee_joint",
    "right_hip_yaw_joint", "right_knee_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint",
)
SLOT_4_5: tuple[str, ...] = (
    "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "right_wrist_yaw_joint", "left_wrist_yaw_joint",
)
ALL_SLOTS = (SLOT_0_1, SLOT_2_3, SLOT_4_5)
assert sorted(sum((list(s) for s in ALL_SLOTS), [])) == sorted(ASIMOV_1_JOINT_NAMES), "slots must partition all 23 joints"

ACTION_SCALE_WALKING = 0.25  # ASIMOV_1_ACTION_SCALE, flat, absolute default-offset (JointPositionActionCfg)

# Command ranges (CommandsCfg.twist.ranges) -- only used to size/clip a manually-supplied velocity command in
# walking-layout sim2sim; not sampled automatically (get-up mode has no command term at all).
COMMAND_RANGES = {"lin_vel_x": (-0.6, 0.8), "lin_vel_y": (-0.5, 0.5), "ang_vel_z": (-0.8, 0.8)}

# Noise (Unoise n_min/n_max, additive uniform) per walking obs term, and per-term output scale, in the exact
# ObservationsCfg.PolicyCfg declaration order (this order also fixes the concatenation order for both layouts).
WALKING_OBS_TERM_ORDER = (
    "base_ang_vel", "projected_gravity", "command",
    "joint_pos_slot01", "joint_pos_slot23", "joint_pos_slot45",
    "joint_vel_slot01", "joint_vel_slot23", "joint_vel_slot45",
    "actions",
)
GETUP_OBS_TERM_ORDER = (  # get-up "policy" obs group: identical but no `command` term.
    "base_ang_vel", "projected_gravity",
    "joint_pos_slot01", "joint_pos_slot23", "joint_pos_slot45",
    "joint_vel_slot01", "joint_vel_slot23", "joint_vel_slot45",
    "actions",
)
GETUP_HISTORY_LENGTH = 5  # get-up obs history; walking layout has no history (history_length unset -> 0 in Isaac Lab)

OBS_NOISE_UNIFORM = {  # (n_min, n_max), applied AFTER any obs-delay and BEFORE scale (ObservationManager.compute_group)
    "base_ang_vel": (-0.01, 0.01),
    "projected_gravity": (-0.02, 0.02),
    "joint_pos_slot01": (-0.01, 0.01),
    "joint_pos_slot23": (-0.01, 0.01),
    "joint_pos_slot45": (-0.01, 0.01),
    "joint_vel_slot01": (-0.5, 0.5),
    "joint_vel_slot23": (-0.5, 0.5),
    "joint_vel_slot45": (-0.5, 0.5),
    # "command" and "actions": no noise
}
OBS_SCALE = {
    "base_ang_vel": 0.25,
    "projected_gravity": 1.0,
    "command": 1.0,
    "joint_pos_slot01": 1.0, "joint_pos_slot23": 1.0, "joint_pos_slot45": 1.0,
    "joint_vel_slot01": 0.1, "joint_vel_slot23": 0.1, "joint_vel_slot45": 0.1,
    "actions": 1.0,
}
# Per-policy-tick observation delay (tasks/locomotion/mdp/observations.py::delayed_obs), in POLICY TICKS (not
# physics steps), min/max inclusive, one random lag per env drawn at episode reset and held fixed all episode.
OBS_DELAY_TICKS = {
    "base_ang_vel": (0, 1),
    "projected_gravity": (0, 2),
    # every other term: no delay model in Isaac (straight through)
}

# ---------------------------------------------------------------------------------------------------------------------
# Get-up action contract (tasks/getup/mdp/actions.py::FilteredRelativeJointPositionActionCfg, real implementation).
# ---------------------------------------------------------------------------------------------------------------------

GETUP_SCALE_TORQUE_FACTOR = 1.1
GETUP_BETA_DEFAULT = 1.0
GETUP_ACTION_CLIP = 1.0
GETUP_USE_LPF = True
GETUP_LPF_ALPHA = 0.557
"""10 Hz one-pole action filter at the 50 Hz policy rate (dt=0.02s -> alpha = dt/(RC+dt) = 0.557), NOT at 200 Hz
(which would give 0.24). Applies identically in training, this harness and the deploy spec.
``tasks/getup/getup_env_cfg.py`` sets ``lpf_alpha=0.557`` explicitly on the ``joint_pos`` action term (the action
*class*'s own default is 0.24, but the env cfg overrides it -- this constant matches the training config)."""
GETUP_LPF_PER_SUBSTEP = False
GETUP_RELATIVE_PER_SUBSTEP = True

# Stage-1 curriculum effort-scale groups (tasks/getup/mdp/curriculums.py::effort_beta_schedule, _STRONG_KEYS/
# _OTHER_KEYS, copied verbatim): hips/knees/shoulders get the Stage-1 boost, everything else (elbow, wrist, ankle,
# waist) stays at 1.0 while stage==0 (Stage 1: beta=1.0, strong=1.2).
GETUP_EFFORT_STRONG_PATTERNS: tuple[str, ...] = (r".*_hip_.*_joint", r".*_knee_joint", r".*_shoulder_.*_joint")
GETUP_EFFORT_OTHER_PATTERNS: tuple[str, ...] = (r".*_(elbow|wrist_yaw|ankle_pitch|ankle_roll)_joint", r"waist_yaw_joint")


def stage1_effort_scale(strong: float = 1.2, other: float = 1.0) -> dict[str, float]:
    """The get-up curriculum's per-group effort-scale dict at Stage 1 (effort limits x1.2 on hips,
    knees and shoulders; elbow, wrist and ankle x1.0). Use this instead of a flat ``--effort-scale 1.2`` for
    parity with a Stage-1 checkpoint -- a flat scale would also boost elbow/wrist/ankle, which training did not."""
    d = {k: strong for k in GETUP_EFFORT_STRONG_PATTERNS}
    d.update({k: other for k in GETUP_EFFORT_OTHER_PATTERNS})
    return d

# ---------------------------------------------------------------------------------------------------------------------
# Standing init state (ASIMOV_1_STANDING_INIT_STATE, shared by walking and get-up handoff target).
# ---------------------------------------------------------------------------------------------------------------------

STANDING_INIT_POS = (0.0, 0.0, 0.639)
STANDING_INIT_QUAT_WXYZ = (1.0, 0.0, 0.0, 0.0)
STANDING_INIT_JOINT_POS_PATTERNS: dict[str, float] = {
    "left_hip_pitch_joint": -0.15,
    "right_hip_pitch_joint": 0.15,
    r".*_hip_roll_joint": 0.0,
    r".*_hip_yaw_joint": 0.0,
    "left_knee_joint": 0.45,
    "right_knee_joint": -0.45,
    "left_ankle_pitch_joint": -0.30,
    "right_ankle_pitch_joint": 0.30,
    r".*_ankle_roll_joint": 0.0,
    "waist_yaw_joint": 0.0,
    "left_shoulder_pitch_joint": -0.25,
    "right_shoulder_pitch_joint": 0.25,
    "left_shoulder_roll_joint": -0.05,
    "right_shoulder_roll_joint": 0.05,
    r".*_shoulder_yaw_joint": 0.0,
    "left_elbow_joint": 0.40,
    "right_elbow_joint": -0.40,
    r".*_wrist_yaw_joint": 0.0,
}


def standing_init_joint_pos() -> np.ndarray:
    """Default joint position vector (23,) in ``ASIMOV_1_JOINT_NAMES`` order.

    Exact joints are matched first (dict lookup), regex patterns second, mirroring Isaac Lab's
    ``InitialStateCfg.joint_pos`` resolution (specific keys win over wildcard keys).
    """
    out = np.zeros(NUM_JOINTS)
    for i, name in enumerate(ASIMOV_1_JOINT_NAMES):
        if name in STANDING_INIT_JOINT_POS_PATTERNS:
            out[i] = STANDING_INIT_JOINT_POS_PATTERNS[name]
            continue
        hits = [v for k, v in STANDING_INIT_JOINT_POS_PATTERNS.items() if k != name and re.fullmatch(k, name)]
        if len(hits) != 1:
            raise ValueError(f"joint {name!r}: expected exactly one default-pos match, got {hits}")
        out[i] = hits[0]
    return out


# ---------------------------------------------------------------------------------------------------------------------
# Get-up success / gates (mirrors scripts/getup/_common.py's shared eval constants -- duplicated, not imported, to
# keep this package isaaclab/torch-free; kept numerically identical on purpose).
# ---------------------------------------------------------------------------------------------------------------------

STANDING_PELVIS_HEIGHT_M = 0.50
STANDING_TILT_RAD = 0.35
STANDING_MAX_LIN_VEL = 0.3
STANDING_HOLD_S = 1.0

G1_TIME_TO_STAND_MAX_S = 6.0
G1_HOLD_S = 5.0
G2_TAU_HAT_SAT_THRESHOLD = 0.9
G2_TAU_HAT_SAT_STEP_FRAC_MAX = 0.05
G2_ELBOW_WRIST_SAT_MAX_S = 0.2
G3_EFFORT_SCALE = 0.9

CATEGORY_KEYS: list[str] = [
    "supine", "prone", "side_left", "side_right", "sitting", "kneeling", "mid_fall", "standing",
]

# ---------------------------------------------------------------------------------------------------------------------
# Wrist stub / upper-arm collision patch (assets/robots/getup_urdf.py -- exact numbers, same patch, applied to the
# MJCF instead of the URDF).
# ---------------------------------------------------------------------------------------------------------------------

WRIST_AXIS = (0.7660444431, 0.0, -0.6427876097)  # link-frame wrist-yaw joint axis, both sides (URDF, same in MJCF)
WRIST_STUB_RADIUS = 0.0188
WRIST_STUB_S_MIN = -0.0040
WRIST_STUB_S_MAX = 0.0150
WRIST_STUB_LENGTH = WRIST_STUB_S_MAX - WRIST_STUB_S_MIN
WRIST_STUB_CENTER_S = 0.5 * (WRIST_STUB_S_MIN + WRIST_STUB_S_MAX)  # 0.0055, measured along WRIST_AXIS

UPPER_ARM_RADIUS = 0.033
UPPER_ARM_Z_TOP = 0.0  # local z of the shoulder_yaw_link frame; bottom = elbow joint z (read from the model)

# Body-shell collision boxes (torso, head, pelvis, thigh, shank), on by default. Copied verbatim from assets/robots/getup_urdf.py::SHELL_BOXES (PATCH_VERSION "getup-v2"): (link, name,
# center xyz, FULL size xyz) in the link/body frame -- URDF box size is full dimensions; MuJoCo box size is
# half-extents, so mj_model.py halves it when adding these. "{side}"-templated entries mirror y for "right".
SHELL_BOXES: list[tuple[str, str, tuple[float, float, float], tuple[float, float, float]]] = [
    ("waist_yaw_link", "torso_abdomen", (0.0, 0.0, 0.075), (0.16, 0.19, 0.09)),
    ("waist_yaw_link", "torso_chest", (-0.0035, 0.0, 0.195), (0.193, 0.24, 0.15)),
    ("waist_yaw_link", "torso_upper_back", (-0.03, 0.0, 0.3075), (0.15, 0.24, 0.075)),
    ("neck_pitch_link", "head", (0.013, 0.0, 0.0425), (0.162, 0.126, 0.125)),
    ("pelvis_link", "pelvis", (-0.0575, 0.0, -0.005), (0.125, 0.12, 0.16)),
    ("{side}_hip_yaw_link", "{side}_thigh", (0.006, 0.0, -0.095), (0.112, 0.10, 0.11)),
    ("{side}_knee_link", "{side}_shank", (-0.005, 0.0, -0.085), (0.114, 0.094, 0.23)),
]
