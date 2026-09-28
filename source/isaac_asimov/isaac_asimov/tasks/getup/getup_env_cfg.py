# Copyright (c) 2026, Menlo Research.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Asimov 1 get-up task configuration.

- ``Asimov1GetUpEnvCfg``: training (``Asimov1-GetUp-v0``).
- ``Asimov1GetUpEnvCfg_PLAY``: evaluation (``Asimov1-GetUp-Play-v0``): no DR, no assist, no curricula, no
  no-progress termination, noise-free policy obs, nominal (1.0x) effort limits. Callers set the category mix through
  ``env_cfg.events.reset_fallen.params["category_probs"]`` (and ``["assignment"] = "round_robin"`` for equal splits),
  and the action bound through ``env_cfg.actions.joint_pos.beta``.
"""

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import math

import isaaclab.terrains as terrain_gen

from isaac_asimov.assets.robots.asimov_1 import ASIMOV_1_GETUP_CFG, ASIMOV_1_GETUP_DC_CFG, ASIMOV_1_JOINT_NAMES
from isaac_asimov.tasks.locomotion.velocity_env_cfg import SLOT_0_1, SLOT_2_3, SLOT_4_5

from . import mdp

# ---------------------------------------------------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------------------------------------------------

H_STAR = 0.614
"""Pelvis standing height h* [m]: measured in simulation with the default standing pose resting on flat ground (feet
spheres touching); the init-state root height (0.639) floats the feet 2.5 cm."""

def _default_cache_path() -> str:
    """``$ASIMOV_GETUP_CACHE``, else the full cache ``~/getup_cache/fallen_v1.pt``, else ``fallen_v0_small.pt``."""
    env_path = os.environ.get("ASIMOV_GETUP_CACHE")
    if env_path:
        return env_path
    v1 = "~/getup_cache/fallen_v1.pt"
    return v1 if os.path.exists(os.path.expanduser(v1)) else "~/getup_cache/fallen_v0_small.pt"


GETUP_CACHE_PATH = _default_cache_path()
"""Fallen-state cache. Override with the ``ASIMOV_GETUP_CACHE`` env var or the event param."""

STEPS_PER_ITER = 24
"""PPO rollout length; converts ``common_step_counter`` to iterations in the curricula."""

TORSO_BODY = "waist_yaw_link"
FEET_BODIES = ["left_ankle_roll_link", "right_ankle_roll_link"]
IMPACT_BODIES = ["waist_yaw_link", "pelvis_link"]  # head collisions are merged into waist_yaw_link

HIP_ROLL_YAW = [".*_hip_roll_joint", ".*_hip_yaw_joint"]
ELBOW_WRIST = [".*_elbow_joint", ".*_wrist_yaw_joint"]


def _slot_cfg(names: tuple[str, ...]) -> SceneEntityCfg:
    return SceneEntityCfg("robot", joint_names=list(names), preserve_order=True)


def _all_joints() -> SceneEntityCfg:
    return SceneEntityCfg("robot", joint_names=list(ASIMOV_1_JOINT_NAMES), preserve_order=True)


# Register mirror rules for critic terms that the symmetry module's defaults do not cover.
try:
    from isaac_asimov.tasks.getup import symmetry as _sym

    _sym.register_mirror_rule("limp_remaining", _sym.Invariant())
except (ImportError, AttributeError):  # pragma: no cover - symmetry module missing
    pass


# ---------------------------------------------------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------------------------------------------------


@configclass
class Asimov1GetUpSceneCfg(InteractiveSceneCfg):
    """Flat plane (the fallen-state cache is settled on flat ground), the get-up robot and one all-body contact sensor.

    The walking task's filtered ``self_collision`` sensor is deliberately not used (its filter patterns do not match
    and PhysX reports errors); non-foot impacts are penalized from the all-body sensor instead.
    """

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )

    robot: ArticulationCfg = ASIMOV_1_GETUP_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    body_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=4,
        track_air_time=False,
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(intensity=750.0, color=(0.9, 0.9, 0.9)),
    )


# ---------------------------------------------------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------------------------------------------------


@configclass
class ActionsCfg:
    joint_pos = mdp.FilteredRelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=list(ASIMOV_1_JOINT_NAMES),
        preserve_order=True,
        scale_torque_factor=1.1,
        beta=1.0,
        use_lpf=True,
        lpf_alpha=0.557,  # 10 Hz one-pole at 50 Hz
    )
    assist = mdp.AssistForceActionCfg(
        asset_name="robot",
        body_name=TORSO_BODY,
        target_height=H_STAR,
        ramp_time_s=3.0,
        stiffness=5000.0,
        damping=500.0,
        force_max=250.0,
        yaw_damping=50.0,
        unassisted_fraction=0.2,
        initial_scale=1.0,
    )


# ---------------------------------------------------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------------------------------------------------


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        """Deployable: the walking sensors, noise and delays, minus ``command``; ``actions`` is the filtered action."""

        base_ang_vel = ObsTerm(
            func=mdp.delayed_obs,
            params={"quantity": "base_ang_vel", "min_lag": 0, "max_lag": 1},
            noise=Unoise(n_min=-0.01, n_max=0.01),
            scale=0.25,
        )
        projected_gravity = ObsTerm(
            func=mdp.delayed_obs,
            params={"quantity": "projected_gravity", "min_lag": 0, "max_lag": 2},
            noise=Unoise(n_min=-0.02, n_max=0.02),
        )
        joint_pos_slot01 = ObsTerm(
            func=mdp.joint_pos_rel, params={"asset_cfg": _slot_cfg(SLOT_0_1)}, noise=Unoise(n_min=-0.01, n_max=0.01)
        )
        joint_pos_slot23 = ObsTerm(
            func=mdp.joint_pos_rel, params={"asset_cfg": _slot_cfg(SLOT_2_3)}, noise=Unoise(n_min=-0.01, n_max=0.01)
        )
        joint_pos_slot45 = ObsTerm(
            func=mdp.joint_pos_rel, params={"asset_cfg": _slot_cfg(SLOT_4_5)}, noise=Unoise(n_min=-0.01, n_max=0.01)
        )
        joint_vel_slot01 = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": _slot_cfg(SLOT_0_1)},
            noise=Unoise(n_min=-0.5, n_max=0.5),
            scale=0.1,
        )
        joint_vel_slot23 = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": _slot_cfg(SLOT_2_3)},
            noise=Unoise(n_min=-0.5, n_max=0.5),
            scale=0.1,
        )
        joint_vel_slot45 = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": _slot_cfg(SLOT_4_5)},
            noise=Unoise(n_min=-0.5, n_max=0.5),
            scale=0.1,
        )
        actions = ObsTerm(func=mdp.filtered_action, params={"action_name": "joint_pos"})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 5

    @configclass
    class CriticCfg(ObsGroup):
        """Noise-free policy terms plus privileged state."""

        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.25)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        joint_pos_slot01 = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": _slot_cfg(SLOT_0_1)})
        joint_pos_slot23 = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": _slot_cfg(SLOT_2_3)})
        joint_pos_slot45 = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": _slot_cfg(SLOT_4_5)})
        joint_vel_slot01 = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": _slot_cfg(SLOT_0_1)}, scale=0.1)
        joint_vel_slot23 = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": _slot_cfg(SLOT_2_3)}, scale=0.1)
        joint_vel_slot45 = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": _slot_cfg(SLOT_4_5)}, scale=0.1)
        actions = ObsTerm(func=mdp.filtered_action, params={"action_name": "joint_pos"})
        # privileged
        pelvis_height = ObsTerm(func=mdp.pelvis_height_obs)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        root_quat = ObsTerm(func=mdp.root_quat_no_yaw)
        body_contact_forces = ObsTerm(
            func=mdp.body_contact_forces,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "body_contact", body_names=list(mdp.CRITIC_CONTACT_BODIES), preserve_order=True
                )
            },
        )
        effort_saturation = ObsTerm(func=mdp.effort_saturation, params={"asset_cfg": _all_joints()})
        thermal_proxy = ObsTerm(func=mdp.thermal_proxy, params={"asset_cfg": _all_joints()})
        assist_force = ObsTerm(func=mdp.assist_force, params={"force_scale": 250.0})
        limp_remaining = ObsTerm(func=mdp.limp_remaining)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
            self.history_length = 5

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


# ---------------------------------------------------------------------------------------------------------------------
# Events: start states and domain randomization (narrow; widened by `curriculum.widen_dr`)
# ---------------------------------------------------------------------------------------------------------------------

DEFAULT_CATEGORY_PROBS = {
    "supine": 0.22,
    "prone": 0.22,
    "side_left": 0.08,
    "side_right": 0.08,
    "sitting": 0.10,
    "kneeling": 0.08,
    "mid_fall": 0.14,
    "standing": 0.08,
}

WIDE_DR_OVERRIDES = {
    # wide ranges: applied by `widen_dr` at the effort milestone, and from the start in Stage B
    "physics_material": {
        "static_friction_range": (0.3, 1.5),
        "dynamic_friction_range": (0.3, 1.5),
        "restitution_range": (0.0, 0.2),
    },
    "link_mass": {"mass_distribution_params": (0.9, 1.1)},
    "torso_mass": {"mass_distribution_params": (-1.0, 3.0)},
    "torso_com": {"com_range": {"x": (-0.03, 0.03), "y": (-0.03, 0.03), "z": (-0.03, 0.03)}},
    "joint_armature": {"armature_distribution_params": (0.8, 1.2)},
    "actuator_gains": {"stiffness_distribution_params": (0.85, 1.15), "damping_distribution_params": (0.8, 1.5)},
    "torso_wrench": {"force_range": (-10.0, 10.0), "torque_range": (-3.0, 3.0)},
    "push_robot": {
        "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-0.5, 0.5)},
    },
}


@configclass
class EventCfg:
    # -- start states (the curricula and evaluation address this term by name)
    reset_fallen = EventTerm(
        func=mdp.reset_fallen_state,
        mode="reset",
        params={"category_probs": dict(DEFAULT_CATEGORY_PROBS), "cache_path": GETUP_CACHE_PATH},
    )

    # -- startup DR (narrow)
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.7, 1.2),
            "dynamic_friction_range": (0.7, 1.2),
            "restitution_range": (0.0, 0.05),
            "num_buckets": 64,
            "make_consistent": True,
        },
    )
    link_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (0.97, 1.03),
            "operation": "scale",
        },
    )
    torso_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[TORSO_BODY]),
            "mass_distribution_params": (-0.3, 0.5),
            "operation": "add",
        },
    )
    torso_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[TORSO_BODY]),
            "com_range": {"x": (-0.005, 0.005), "y": (-0.005, 0.005), "z": (-0.005, 0.005)},
        },
    )
    joint_armature = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "armature_distribution_params": (0.95, 1.05),
            "operation": "scale",
        },
    )

    # -- reset DR (narrow)
    actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.95, 1.05),
            "damping_distribution_params": (0.9, 1.1),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    torso_wrench = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[TORSO_BODY]),
            "force_range": (-2.0, 2.0),
            "torque_range": (-0.5, 0.5),
        },
    )

    # -- interval DR (narrow)
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(3.0, 8.0),
        params={"velocity_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "yaw": (-0.1, 0.1)}},
    )


# ---------------------------------------------------------------------------------------------------------------------
# Rewards: task, posture, safety, regularization (+ ankle motor torque). All masked while the policy is not in control.
# ---------------------------------------------------------------------------------------------------------------------


@configclass
class RewardsCfg:
    # -- task
    height_coarse = RewTerm(func=mdp.height_exp, weight=2.0, params={"std": 0.5, "target_height": H_STAR})
    height_medium = RewTerm(func=mdp.height_exp, weight=6.0, params={"std": 0.25, "target_height": H_STAR})
    height_fine = RewTerm(func=mdp.height_exp, weight=10.0, params={"std": 0.1, "target_height": H_STAR})
    height_record = RewTerm(func=mdp.height_record, weight=1.5, params={"max_rate": 1.0})
    upright = RewTerm(func=mdp.upright, weight=1.0)
    flat_orientation = RewTerm(func=mdp.torso_flat_orientation_l2, weight=-2.0)
    feet_support = RewTerm(func=mdp.feet_support, weight=2.0, params={"max_foot_height": 0.08, "rising_height": 0.38})
    stand_bonus = RewTerm(func=mdp.stand_bonus, weight=2.0, params={"target_height": H_STAR})
    # -- posture
    joint_deviation = RewTerm(
        func=mdp.joint_deviation_l1_standing,
        weight=-0.05,
        params={"asset_cfg": _all_joints(), "target_height": H_STAR},
    )
    not_moving = RewTerm(func=mdp.not_moving_standing, weight=-0.5, params={"target_height": H_STAR})
    feet_lateral_distance = RewTerm(
        func=mdp.feet_lateral_distance_standing,
        weight=-10.0,
        params={"target_distance": 0.215, "target_height": H_STAR},  # measured default stance
    )
    feet_yaw = RewTerm(func=mdp.feet_yaw_vs_base_standing, weight=-2.0, params={"target_height": H_STAR})
    ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_rising, weight=-0.2)
    hip_roll_yaw_dev = RewTerm(
        func=mdp.joint_deviation_l2_rising,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=HIP_ROLL_YAW)},
    )
    waist_dev = RewTerm(
        func=mdp.joint_deviation_l2_rising,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["waist_yaw_joint"])},
    )
    feet_distance = RewTerm(func=mdp.feet_distance_bounds, weight=-2.0, params={"min_dist": 0.12, "max_dist": 0.5})
    # -- safety
    impact_nonfoot = RewTerm(
        func=mdp.impact_nonfoot,
        weight=-1.0e-3,
        params={"sensor_cfg": SceneEntityCfg("body_contact", body_names=IMPACT_BODIES), "threshold": 150.0},
    )
    head_contact = RewTerm(func=mdp.head_contact, weight=-1.0)
    # -- regularization
    action_rate = RewTerm(func=mdp.action_rate_clipped_l2, weight=-0.01)
    action_second_diff = RewTerm(func=mdp.action_second_diff_l2, weight=-0.01)
    action_out_of_bounds = RewTerm(func=mdp.action_out_of_bounds_l2, weight=-0.5)  # raw |a| beyond the clip
    joint_acc = RewTerm(func=mdp.joint_acc_l2_active, weight=-2.5e-7)
    torque_tiredness = RewTerm(func=mdp.torque_tiredness, weight=-0.02)
    torque_soft_limit = RewTerm(func=mdp.torque_soft_limit, weight=-2.0, params={"soft_ratio": 0.85})
    elbow_wrist_torque = RewTerm(
        func=mdp.upper_torque_l2, weight=-0.05, params={"asset_cfg": SceneEntityCfg("robot", joint_names=ELBOW_WRIST)}
    )
    positive_power = RewTerm(func=mdp.positive_power, weight=-2.0e-3)
    joint_pos_limits = RewTerm(func=mdp.joint_pos_limits_soft, weight=-5.0, params={"soft_ratio": 0.95})
    joint_vel_limits = RewTerm(func=mdp.joint_vel_limits_soft, weight=-1.0, params={"soft_ratio": 0.9})
    root_acc = RewTerm(func=mdp.root_acc_l2, weight=-5.0e-4)
    thermal_proxy = RewTerm(func=mdp.thermal_proxy_penalty, weight=-0.5, params={"threshold": 0.6})
    # per-motor torque beyond the 12 Nm rating (through the ankle differential). -2 @ 0.85 dominated the smoke run's
    # return (random actions saturate pitch+roll combinations), so it starts at -1 @ 1.0
    ankle_motor_torque = RewTerm(
        func=mdp.ankle_motor_torque,
        weight=-1.0,
        # conservative torque model (joint torque = 2x motor torque) until the manufacturer confirms the transmission
        params={"soft_ratio": 1.0, "k_pitch": 1.0, "k_roll": 1.0},
    )
    # -- termination penalty
    no_height_progress = RewTerm(
        func=mdp.is_terminated_term, weight=-5.0, params={"term_keys": "no_height_progress"}
    )


# ---------------------------------------------------------------------------------------------------------------------
# Terminations
# ---------------------------------------------------------------------------------------------------------------------


@configclass
class TerminationsCfg:
    # tracker first: it updates the shared state that `no_height_progress` reads (never terminates)
    getup_tracker = DoneTerm(
        func=mdp.getup_tracker,
        params={"hold_time_s": 1.0, "progress_height": 0.2, "thermal_time_constant_s": 2.0, "window": 500.0,
                "det_window": 200.0, "det_category_window": 100.0},  # fmt: skip
    )
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    no_height_progress = DoneTerm(func=mdp.no_height_progress, params={"time_limit_s": 8.0})
    invalid_state = DoneTerm(func=mdp.root_state_invalid, params={"max_lin_vel": 20.0, "max_height": 5.0})


# ---------------------------------------------------------------------------------------------------------------------
# Curricula (iterations = common_step_counter // 24)
# ---------------------------------------------------------------------------------------------------------------------


@configclass
class CurriculumCfg:
    assist = CurrTerm(
        func=mdp.assist_decay,
        params={
            "action_name": "assist",
            "ema_alpha": 0.05,
            "threshold": 0.6,
            "decay": 0.997,  # per iteration, driven by assisted success (not per reset)
            "disable_below": 0.02,
            "allow_recover": True,
            "recover_factor": 1.002,
            "recover_below_scale": 0.3,
            "recover_unassisted_below": 0.1,
            "steps_per_iter": STEPS_PER_ITER,
        },
    )
    action_rate = CurrTerm(
        func=mdp.reward_weight_ramp,
        params={"term_name": "action_rate", "start_iter": 1000, "end_iter": 3000, "end_weight": -0.1,
                "log_space": True, "steps_per_iter": STEPS_PER_ITER},  # fmt: skip
    )
    joint_deviation = CurrTerm(
        func=mdp.reward_weight_ramp,
        params={"term_name": "joint_deviation", "start_iter": 2000, "end_iter": 4000, "end_weight": -0.5,
                "log_space": True, "steps_per_iter": STEPS_PER_ITER},  # fmt: skip
    )
    torque_soft_limit = CurrTerm(
        func=mdp.reward_weight_ramp,
        params={"term_name": "torque_soft_limit", "start_iter": 2000, "end_iter": 4000, "end_weight": -5.0,
                "log_space": False, "steps_per_iter": STEPS_PER_ITER},  # fmt: skip
    )
    no_progress_penalty = CurrTerm(
        func=mdp.reward_weight_ramp,
        params={"term_name": "no_height_progress", "start_iter": 2500, "end_iter": 5000, "end_weight": -100.0,
                "log_space": True, "steps_per_iter": STEPS_PER_ITER},  # fmt: skip
    )
    effort = CurrTerm(
        func=mdp.effort_beta_schedule,
        params={
            "action_name": "joint_pos",
            "stage1_scale": 1.2,
            "final_scale": 1.0,
            "beta_start": 1.0,
            "beta_end": 0.8,
            "transition_iters": 1000,
            "milestone_success": 0.7,
            "min_iters_after_assist": 50,  # iterations at assist = 0 before the milestone can fire
            "strength_range": (1.0, 1.0),
            "strength_range_wide": (0.85, 1.05),
            "num_strength_buckets": 5,
            "start_stage": 0,
            "steps_per_iter": STEPS_PER_ITER,
        },
    )
    widen_dr = CurrTerm(func=mdp.domain_rand_widen, params={"overrides": WIDE_DR_OVERRIDES, "force": False})
    categories = CurrTerm(
        func=mdp.category_reweighting,
        params={"every_iters": 200, "floor": 0.03, "offset": 0.1, "min_episodes": 500, "event_term": "reset_fallen",
                "steps_per_iter": STEPS_PER_ITER},  # fmt: skip
    )
    # must stay last: saves/restores the state of the terms above (RUNBOOK "Resuming a get-up run")
    persist = CurrTerm(
        func=mdp.curriculum_checkpoint,
        params={"save_interval": 250, "restore_path": "", "start_iteration": -1, "steps_per_iter": STEPS_PER_ITER},
    )


# ---------------------------------------------------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------------------------------------------------


def eval_terrain_generator(kind: str) -> terrain_gen.TerrainGeneratorCfg | None:
    """Small evaluation terrains (<= 16 envs; few tiles and a thin border keep PhysX GPU buffers small).

    ``"flat"``: None (plane). ``"rough"``: random-uniform +-2 cm everywhere. ``"mix"``: the training terrain mix
    (flat 60 %, rough +-2 cm 30 %, pyramid slopes <= 8 deg 10 %; 10 columns so the mix is exact by column).
    """
    if kind == "flat":
        return None
    rough = terrain_gen.HfRandomUniformTerrainCfg(
        proportion=1.0, noise_range=(-0.02, 0.02), noise_step=0.005, border_width=0.25
    )
    common = dict(size=(8.0, 8.0), border_width=2.0, horizontal_scale=0.1, vertical_scale=0.005,
                  slope_threshold=0.75, difficulty_range=(0.0, 1.0), use_cache=False, curriculum=False)  # fmt: skip
    if kind == "rough":
        return terrain_gen.TerrainGeneratorCfg(num_rows=2, num_cols=4, sub_terrains={"rough": rough}, **common)
    if kind == "mix":
        return terrain_gen.TerrainGeneratorCfg(
            num_rows=2,
            num_cols=10,
            sub_terrains={
                "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.6),
                "rough": rough.replace(proportion=0.3),
                "slope": terrain_gen.HfPyramidSlopedTerrainCfg(
                    proportion=0.1, slope_range=(0.0, math.tan(math.radians(8.0))), platform_width=2.0,
                    border_width=0.25,
                ),  # fmt: skip
            },
            **common,
        )
    raise ValueError(f"unknown eval terrain '{kind}' (flat | rough | mix)")


@configclass
class Asimov1GetUpEnvCfg(ManagerBasedRLEnvCfg):
    scene: Asimov1GetUpSceneCfg = Asimov1GetUpSceneCfg(num_envs=4096, env_spacing=2.5)
    play_motor_strength: float | str | dict | None = None
    """Stress test (PLAY / eval): multiplies the actuator effort **clip** only, the action bound stays at
    ``play_effort_scale``, e.g. 0.9 for the weak-motor check."""
    play_effort_scale: float | str | dict | None = None
    """Fixed effort scale applied at startup when there are no curricula (PLAY): a float (all joints), a preset
    (``"nominal"``, ``"stage1"`` = hips/knees/shoulders x1.2, ``"stageB"`` = x0.9) or ``{joint regex: scale}``.
    ``None`` in training (the effort curriculum owns the limits). A **trained condition**: sets both the action bound
    and the effort clip. ``mdp.action_contract(env)`` returns the resulting contract."""
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def set_eval_terrain(self, kind: str) -> None:
        """Evaluation terrain switch, call **before** ``gym.make``: ``"flat"`` | ``"rough"`` (+-2 cm) | ``"mix"``
        (the training terrain mix). Small generator for <= 16 envs; resets place states on the terrain (ray-cast lift)
        and ``pelvis_height`` measures against the terrain under the robot."""
        gen = eval_terrain_generator(kind)
        if gen is None:
            self.scene.terrain.terrain_type = "plane"
            self.scene.terrain.terrain_generator = None
            return
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = gen
        self.scene.terrain.max_init_terrain_level = None

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 12.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        # PhysX GPU buffers: lying robots touch the ground with many shapes (+ self-collision pairs)
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        self.sim.physx.gpu_found_lost_pairs_capacity = 2**22
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 2**23
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 2**26
        # a robot lying still must not fall asleep; self-collisions stay on (asset)
        self.scene.robot.spawn.articulation_props.sleep_threshold = 0.0
        self.scene.robot.spawn.articulation_props.enabled_self_collisions = True
        # contact sensor updates every physics substep (history_length = decimation)
        self.scene.body_contact.update_period = 0.0


@configclass
class Asimov1GetUpEnvCfg_PLAY(Asimov1GetUpEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 64
        self.episode_length_s = 15.0
        self.observations.policy.enable_corruption = False
        # no assist
        self.actions.assist.initial_scale = 0.0
        self.actions.assist.unassisted_fraction = 1.0
        # no DR
        for name in ("physics_material", "link_mass", "torso_mass", "torso_com", "joint_armature", "actuator_gains",
                     "torso_wrench", "push_robot"):  # fmt: skip
            setattr(self.events, name, None)
        # no curricula: effort from `play_effort_scale` (1.0 = nominal certification setting; "stage1" reproduces the
        # Stage-1 training limits), beta from actions.joint_pos.beta; no early termination
        self.curriculum = None
        self.play_effort_scale = 1.0
        self.events.play_effort = EventTerm(func=mdp.apply_play_effort_scale, mode="startup", params={})
        self.terminations.no_height_progress = None
        self.rewards.no_height_progress = None


# ---------------------------------------------------------------------------------------------------------------------
# Stage B fine-tune: resumes from a Stage A checkpoint (see agents/warm_start.py)
# ---------------------------------------------------------------------------------------------------------------------

STAGE_B_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    curriculum=False,
    sub_terrains={
        # terrain mix: flat 60 %, mild rough (+-2 cm) 30 %, slopes <= 8 deg 10 %
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.6),
        "rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.3, noise_range=(-0.02, 0.02), noise_step=0.005, border_width=0.25
        ),
        "slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, math.tan(math.radians(8.0))), platform_width=2.0, border_width=0.25
        ),
    },
)
"""Stage B terrain. Resets lift every start state out of the terrain (ray-cast ``TerrainHeight``). Height terms
measure the pelvis against the env origin (tile centre height), so on the +-2 cm tiles they are off by <= 2 cm; the
slope tiles keep a 2 m flat platform at the origin, where the robots start (xy offset <= 0.25 m)."""

STAGE_B_SCALES = {
    "elbow_wrist_torque": -0.2,  # Stage A -0.05 (x4): elbow/wrist saturated most of supine/sitting episodes
    "torque_soft_limit": -8.0,  # Stage A final -5 (x1.6)
    "thermal_proxy": -2.0,  # Stage A -0.5 (x4): thermal proxy peaked at 1.27 in Stage A evals
    "impact_nonfoot": -2.0e-3,  # Stage A -1e-3 (x2): non-foot impact peaks 6-10 kN
    "pelvis_vz_excess": -1.0,  # new: relu(v_z - 1 m/s)^2
    "torso_ang_vel_excess": -0.2,  # new: relu(|w_torso| - 2 rad/s)^2
    # root cause of the elbow/wrist saturation: the arm (mostly the left) is loaded against its stops / the torso
    "arm_limit_under_load": -2.0,  # sum relu(|q - q_c| / half_range - 0.9) * tau_hat over elbow/wrist
    "wrist_yaw_deviation": -0.1,  # sum |q_wrist_yaw - q_default|, all phases
    "arm_self_contact": -1.0,  # per forearm link in a non-ground contact > 5 N
}
"""Stage B hardware-safety emphasis (actuator load, impacts, arm stops). Everything else starts at its post-ramp
Stage A value."""


@configclass
class Asimov1GetUpStageBEnvCfg(Asimov1GetUpEnvCfg):
    """Stage B fine-tune: DC-motor actuators (torque-speed curve + per-motor ankle clip, conservative ankle ratio),
    effort 1.0 -> 0.9 and beta 0.8 -> 0.7 over 1k iterations, no assist, full DR from the start, terrain mix (opt-in),
    final regularization weights plus the hardware-safety emphasis above. Observation and action layouts are identical
    to Stage A, so Stage A weights load unchanged."""

    def __post_init__(self):
        super().__post_init__()
        # -- robot: Stage B actuators; ankle motor clip with the conservative torque ratio (joint = 2 x motor)
        robot = ASIMOV_1_GETUP_DC_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        robot.actuators = {
            name: act.replace(ankle_torque_k_pitch=1.0, ankle_torque_k_roll=1.0) for name, act in robot.actuators.items()
        }
        self.scene.robot = robot
        # -- terrain mix: configured but OFF by default (plane). The generator terrain can exhaust PhysX's GPU contact
        #    buffers when the GPU is shared with another 4096-env run. Enable with the Hydra override
        #    `env.scene.terrain.terrain_type=generator`.
        self.scene.terrain = self.scene.terrain.replace(
            terrain_type="plane", terrain_generator=STAGE_B_TERRAIN_CFG, max_init_terrain_level=None
        )
        # -- no assist: the action term is removed (0-dim, so the policy output is unchanged); the critic's
        #    `assist_force` obs stays (always 0) so the observation layout matches Stage A
        self.actions.assist = None
        self.curriculum.assist = None
        # -- full domain randomization from the start (friction buckets are sampled from these ranges at init)
        for term_name, params in WIDE_DR_OVERRIDES.items():
            term = getattr(self.events, term_name, None)
            if term is not None:
                term.params.update(params)
        self.curriculum.widen_dr = None
        # -- regularization at the post-ramp (final) Stage A weights, no ramps
        self.rewards.action_rate.weight = -0.1
        self.rewards.joint_deviation.weight = -0.5
        self.rewards.torque_soft_limit.weight = -5.0
        self.rewards.no_height_progress.weight = -100.0
        for name in ("action_rate", "joint_deviation", "torque_soft_limit", "no_progress_penalty"):
            setattr(self.curriculum, name, None)
        # -- hardware-safety emphasis + rise-speed shaping
        self.rewards.elbow_wrist_torque.weight = STAGE_B_SCALES["elbow_wrist_torque"]
        self.rewards.torque_soft_limit.weight = STAGE_B_SCALES["torque_soft_limit"]
        self.rewards.thermal_proxy.weight = STAGE_B_SCALES["thermal_proxy"]
        self.rewards.impact_nonfoot.weight = STAGE_B_SCALES["impact_nonfoot"]
        self.rewards.pelvis_vz_excess = RewTerm(
            func=mdp.pelvis_vertical_speed_excess, weight=STAGE_B_SCALES["pelvis_vz_excess"], params={"max_speed": 1.0}
        )
        self.rewards.torso_ang_vel_excess = RewTerm(
            func=mdp.torso_ang_vel_excess, weight=STAGE_B_SCALES["torso_ang_vel_excess"], params={"max_rate": 2.0}
        )
        self.rewards.arm_limit_under_load = RewTerm(
            func=mdp.arm_limit_under_load, weight=STAGE_B_SCALES["arm_limit_under_load"], params={"near_limit": 0.9}
        )
        self.rewards.wrist_yaw_deviation = RewTerm(
            func=mdp.wrist_yaw_deviation, weight=STAGE_B_SCALES["wrist_yaw_deviation"]
        )
        self.rewards.arm_self_contact = RewTerm(
            func=mdp.arm_self_contact,
            weight=STAGE_B_SCALES["arm_self_contact"],
            params={"force_threshold": 5.0, "ground_margin": 0.03},
        )
        # -- effort / action bound: post-milestone ramp (stage 1 -> 2): effort 1.0 -> 0.9 on all joints and
        #    beta 0.8 -> 0.7 over 1000 iterations; per-env motor strength x0.85-1.05 (wide DR)
        eff = self.curriculum.effort.params
        eff.update({
            "start_stage": 1, "stage1_scale": 1.0, "final_scale": 0.9, "beta_start": 0.8, "beta_end": 0.7,
            "transition_iters": 1000,
        })  # fmt: skip
        self.actions.joint_pos.beta = 0.8


@configclass
class Asimov1GetUpStageBEnvCfg_PLAY(Asimov1GetUpStageBEnvCfg):
    """Stage B evaluation: DC actuators, effort x0.9 (``play_effort_scale="stageB"``), beta 0.7, no DR, no curricula,
    terrain mix kept (evaluate flat by setting ``scene.terrain.terrain_type = "plane"``)."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 64
        self.episode_length_s = 15.0
        self.observations.policy.enable_corruption = False
        for name in ("physics_material", "link_mass", "torso_mass", "torso_com", "joint_armature", "actuator_gains",
                     "torso_wrench", "push_robot"):  # fmt: skip
            setattr(self.events, name, None)
        self.curriculum = None
        self.terminations.no_height_progress = None
        self.rewards.no_height_progress = None
        self.actions.joint_pos.beta = 0.7
        self.play_effort_scale = "stageB"
        self.events.play_effort = EventTerm(func=mdp.apply_play_effort_scale, mode="startup", params={})


# ---------------------------------------------------------------------------------------------------------------------
# Stage B2 "robust hold": warm start from Stage B (contract bound x0.9 / beta 0.7), terrain on, hold emphasis
# ---------------------------------------------------------------------------------------------------------------------

STAGE_B2_TERRAIN_CFG = STAGE_B_TERRAIN_CFG.replace(num_rows=8, num_cols=10, border_width=5.0)
"""Training terrain for B2: the Stage B mix on 8 x 10 tiles of 8 m with a 5 m border (smaller mesh than Stage B's
10 x 20 / 20 m, which ran out of GPU memory on a shared GPU). 10 columns make the 60/30/10 mix exact by column."""

STAGE_B2_SCALES = {
    "hold_after_success": 3.0,  # +1/step (x dt) under S after the first success (stand_bonus +2 stays)
    "lost_standing_event": -250.0,  # -5 per event after the x dt scaling, once per episode
    "post_success_drift": -2.0,  # relu(tilt - 0.2) + 2 relu(h* - 0.05 - h) after success
    "elbow_wrist_torque": -0.3,  # Stage B -0.2 (+50 %), kneeling e/w streaks
    "arm_limit_under_load": -3.0,  # Stage B -2 (+50 %)
}


@configclass
class Asimov1GetUpStageB2EnvCfg(Asimov1GetUpStageBEnvCfg):
    """Stage B2: same contract as Stage B (bound x0.9, beta 0.7, DC motors, conservative ankle ratio), no assist, full
    DR incl. pushes, the terrain mix in training, 16 s episodes, hold emphasis; curricula (category reweighting) judged
    on a 3 s hold of the deterministic envs."""

    def __post_init__(self):
        super().__post_init__()
        # contract: final Stage B values from the start (stage 2), no ramp
        self.curriculum.effort.params.update({"start_stage": 2, "final_scale": 0.9, "beta_end": 0.7})
        self.actions.joint_pos.beta = 0.7
        # terrain mix on (terrain-aware pelvis height, reset lift on the spawn terrain)
        self.scene.terrain = self.scene.terrain.replace(
            terrain_type="generator", terrain_generator=STAGE_B2_TERRAIN_CFG, max_init_terrain_level=None
        )
        # long holds
        self.episode_length_s = 16.0
        self.terminations.getup_tracker.params["curriculum_hold_s"] = 3.0
        # hold emphasis
        self.rewards.hold_after_success = RewTerm(
            func=mdp.hold_after_success, weight=STAGE_B2_SCALES["hold_after_success"], params={"target_height": H_STAR}
        )
        self.rewards.lost_standing = RewTerm(func=mdp.lost_standing_event, weight=STAGE_B2_SCALES["lost_standing_event"])
        self.rewards.post_success_drift = RewTerm(
            func=mdp.post_success_drift, weight=STAGE_B2_SCALES["post_success_drift"], params={"target_height": H_STAR}
        )
        # kneeling elbow/wrist streaks: +50 %
        self.rewards.elbow_wrist_torque.weight = STAGE_B2_SCALES["elbow_wrist_torque"]
        self.rewards.arm_limit_under_load.weight = STAGE_B2_SCALES["arm_limit_under_load"]


@configclass
class Asimov1GetUpStageB2EnvCfg_PLAY(Asimov1GetUpStageB2EnvCfg):
    """B2 evaluation: contract x0.9 / beta 0.7 (``play_effort_scale="stageB"``), no DR, no curricula, plane by default
    (``set_eval_terrain("rough"|"mix")`` for the robustness evaluation)."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 64
        self.episode_length_s = 16.0
        self.observations.policy.enable_corruption = False
        for name in ("physics_material", "link_mass", "torso_mass", "torso_com", "joint_armature", "actuator_gains",
                     "torso_wrench", "push_robot"):  # fmt: skip
            setattr(self.events, name, None)
        self.curriculum = None
        self.terminations.no_height_progress = None
        self.rewards.no_height_progress = None
        self.play_effort_scale = "stageB"
        self.events.play_effort = EventTerm(func=mdp.apply_play_effort_scale, mode="startup", params={})
        self.set_eval_terrain("flat")


# ---------------------------------------------------------------------------------------------------------------------
# Stage B3: B2 + weak-motor robustness (clip only), stronger hold, faster rise
# ---------------------------------------------------------------------------------------------------------------------

STAGE_B3_SCALES = {
    "hold_after_success": 5.0,  # B2 +3
    "lost_standing_event": -500.0,  # B2 -250 (= -10 per event after x dt)
    "standing_motion": -0.5,  # new: (|w_torso|^2 + v_z^2) under S
    "not_yet_standing": -0.5,  # new: per step under control before the first success
    "height_record": 2.0,  # Stage A/B 1.5
}


@configclass
class Asimov1GetUpStageB3EnvCfg(Asimov1GetUpStageB2EnvCfg):
    """Stage B3: B2 (contract bound x0.9 / beta 0.7, terrain mix, pushes, full DR) with motor-strength DR biased low
    (clip scale 0.80-1.00, half of the draws from 0.85-0.92; the action bound is unchanged), a stronger hold
    reward / fall penalty, a quiet-stance penalty and a mild time-to-stand penalty."""

    def __post_init__(self):
        super().__post_init__()
        self.curriculum.effort.params.update(
            {"strength_range_wide": (0.80, 1.00), "strength_focus_range": (0.85, 0.92), "strength_focus_frac": 0.5}
        )
        self.rewards.hold_after_success.weight = STAGE_B3_SCALES["hold_after_success"]
        self.rewards.lost_standing.weight = STAGE_B3_SCALES["lost_standing_event"]
        self.rewards.height_record.weight = STAGE_B3_SCALES["height_record"]
        self.rewards.standing_motion = RewTerm(
            func=mdp.standing_motion, weight=STAGE_B3_SCALES["standing_motion"], params={"target_height": H_STAR}
        )
        self.rewards.not_yet_standing = RewTerm(func=mdp.not_yet_standing, weight=STAGE_B3_SCALES["not_yet_standing"])


@configclass
class Asimov1GetUpStageB3EnvCfg_PLAY(Asimov1GetUpStageB2EnvCfg_PLAY):
    """B3 evaluation = B2 evaluation (same contract, x0.9 / beta 0.7); use ``play_motor_strength`` for weak-motor tests."""


# ---------------------------------------------------------------------------------------------------------------------
# Stage B2R: Stage B2 with a rough-heavy training terrain (ablation: stacked DR hurts only on rough ground)
# ---------------------------------------------------------------------------------------------------------------------

STAGE_B2R_TERRAIN_CFG = STAGE_B2_TERRAIN_CFG.replace(
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.2),
        "rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.6, noise_range=(-0.02, 0.02), noise_step=0.005, border_width=0.25
        ),
        "slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.2, slope_range=(0.0, math.tan(math.radians(8.0))), platform_width=2.0, border_width=0.25
        ),
    }
)
"""Rough-heavy mix: flat 20 % / rough +-2 cm 60 % / slopes <= 8 deg 20 %; 10 columns -> exactly 2 / 6 / 2 by column."""


@configclass
class Asimov1GetUpStageB2REnvCfg(Asimov1GetUpStageB2EnvCfg):
    """Stage B2 with the rough-heavy terrain; DR, rewards and contract identical to B2."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = STAGE_B2R_TERRAIN_CFG


@configclass
class Asimov1GetUpStageB2REnvCfg_PLAY(Asimov1GetUpStageB2EnvCfg_PLAY):
    """Same as the B2 evaluation cfg (plane by default; ``set_eval_terrain("rough"|"mix")`` for robustness evals)."""
