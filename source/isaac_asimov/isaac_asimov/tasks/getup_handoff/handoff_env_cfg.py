# Copyright (c) 2026, Menlo Research.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""``Asimov1-GetUp-Handoff-Play-v0``.

Builds on ``Asimov1GetUpEnvCfg_PLAY`` (get-up categories, get-up ``policy``/``critic`` obs
groups, get-up rewards/terminations/tracker) and adds everything needed to demo a handoff to the
walking policy in the same env:

- a ``walk`` observation group with the walking policy's exact layout (base_ang_vel,
  projected_gravity, ``twist`` velocity command, joint_pos/vel slots, actions) so the walking
  checkpoint can run unmodified;
- a ``twist`` velocity command term (absent from the get-up task) that the play script drives
  directly rather than letting it auto-resample;
- a single combined action term, ``HandoffJointPositionAction`` (mdp.py), replacing the get-up
  task's ``FilteredRelativeJointPositionActionCfg`` action term. It carries both action paths
  (get-up relative+filtered, walking absolute+0.25) and switches which one drives the joints per
  env with no jump — see mdp.py's module docstring;
- get-up terminations are disabled except a generous time_out, so the demo's own state machine
  (in play_combined.py) controls the fall -> get-up -> walk -> push -> get-up sequence without the
  env resetting out from under it.

This module imports ``isaac_asimov.tasks.getup.getup_env_cfg`` at module scope, which is why the
task's ``__init__.py`` only registers it via string entry points (see that module's docstring).
"""

from __future__ import annotations

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from isaac_asimov.assets.robots.asimov_1 import ASIMOV_1_JOINT_NAMES
from isaac_asimov.tasks.getup.getup_env_cfg import Asimov1GetUpEnvCfg_PLAY  # noqa: E402  (see docstring)
from isaac_asimov.tasks.locomotion import mdp as walk_mdp
from isaac_asimov.tasks.locomotion.velocity_env_cfg import SLOT_0_1, SLOT_2_3, SLOT_4_5
from isaac_asimov.tasks.locomotion.velocity_env_cfg import CommandsCfg as WalkCommandsCfg

from . import mdp as handoff_mdp


def _slot_cfg(names: tuple[str, ...]) -> SceneEntityCfg:
    # Local copy of velocity_env_cfg's private helper (not re-imported: that one is prefixed
    # `_` and private to the locomotion task file, not a frozen cross-task export).
    return SceneEntityCfg("robot", joint_names=list(names), preserve_order=True)

# Handoff hysteresis thresholds: switch to walking only once the robot has held a stand close to
# the walking default pose and is nearly still, so the walking policy never starts from an
# out-of-distribution state. Exposed here (not just hardcoded in play_combined.py) so the sim demo
# and the firmware spec cite one source.
HANDOFF_STAND_HOLD_S = 1.0
HANDOFF_MAX_JOINT_DEV = 0.15  # rad, L_inf over all joints vs. the walking default pose.
HANDOFF_MAX_ANG_VEL = 0.3  # rad/s, L2 over base angular velocity.

# Re-trigger get-up if the walking policy is driving and the robot falls (the get-up success
# condition's complement, used loosely — the real lying detector for the demo/firmware lives in
# play_combined.py / the deploy spec and works off gravity, not this pelvis-height/tilt gate).


@configclass
class WalkObsCfg(ObsGroup):
    """The walking policy's exact obs layout (locomotion.velocity_env_cfg.ObservationsCfg.PolicyCfg),
    with ``actions`` re-pointed at the handoff action term's walking path (mdp.walk_last_raw_action)
    instead of the generic ``mdp.last_action``, which would read the combined 46-dim raw action."""

    base_ang_vel = ObsTerm(
        func=walk_mdp.delayed_obs,
        params={"quantity": "base_ang_vel", "min_lag": 0, "max_lag": 1},
        noise=Unoise(n_min=-0.01, n_max=0.01),
        scale=0.25,
    )
    projected_gravity = ObsTerm(
        func=walk_mdp.delayed_obs,
        params={"quantity": "projected_gravity", "min_lag": 0, "max_lag": 2},
        noise=Unoise(n_min=-0.02, n_max=0.02),
    )
    command = ObsTerm(func=walk_mdp.generated_commands, params={"command_name": "twist"})
    joint_pos_slot01 = ObsTerm(
        func=walk_mdp.joint_pos_rel, params={"asset_cfg": _slot_cfg(SLOT_0_1)}, noise=Unoise(n_min=-0.01, n_max=0.01)
    )
    joint_pos_slot23 = ObsTerm(
        func=walk_mdp.joint_pos_rel, params={"asset_cfg": _slot_cfg(SLOT_2_3)}, noise=Unoise(n_min=-0.01, n_max=0.01)
    )
    joint_pos_slot45 = ObsTerm(
        func=walk_mdp.joint_pos_rel, params={"asset_cfg": _slot_cfg(SLOT_4_5)}, noise=Unoise(n_min=-0.01, n_max=0.01)
    )
    joint_vel_slot01 = ObsTerm(
        func=walk_mdp.joint_vel_rel,
        params={"asset_cfg": _slot_cfg(SLOT_0_1)},
        noise=Unoise(n_min=-0.5, n_max=0.5),
        scale=0.1,
    )
    joint_vel_slot23 = ObsTerm(
        func=walk_mdp.joint_vel_rel,
        params={"asset_cfg": _slot_cfg(SLOT_2_3)},
        noise=Unoise(n_min=-0.5, n_max=0.5),
        scale=0.1,
    )
    joint_vel_slot45 = ObsTerm(
        func=walk_mdp.joint_vel_rel,
        params={"asset_cfg": _slot_cfg(SLOT_4_5)},
        noise=Unoise(n_min=-0.5, n_max=0.5),
        scale=0.1,
    )
    actions = ObsTerm(func=handoff_mdp.walk_last_raw_action, params={"action_name": "joint_pos"})

    def __post_init__(self):
        # PLAY: no corruption, matching Asimov1VelocityEnvCfg_PLAY.
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class Asimov1GetUpHandoffPlayEnvCfg(Asimov1GetUpEnvCfg_PLAY):
    """Combined get-up + walking env for the handoff demo. Frozen gym id: ``Asimov1-GetUp-Handoff-Play-v0``."""

    def __post_init__(self):
        super().__post_init__()

        # --- commands: add the walking task's velocity command (absent from get-up). -----------
        self.commands = WalkCommandsCfg()
        # The play script drives `command_manager.get_term("twist").command[:] = ...` directly
        # (zero during get-up, forward during walk); make auto-resampling a no-op by giving it an
        # effectively infinite period so the script's writes are never overwritten mid-episode.
        self.commands.twist.resampling_time_range = (1.0e9, 1.0e9)
        self.commands.twist.rel_standing_envs = 0.0
        # Disable heading control: with it on, `UniformVelocityCommand._update_command()` recomputes
        # the yaw-rate component (index 2) from a heading target every step regardless of the
        # resample period above, which would curve the demo's walk instead of a straight line. The
        # script writes index 2 directly (0.0) instead.
        self.commands.twist.heading_command = False

        # --- observations: keep the get-up `policy`/`critic` groups, add `walk`. ----------------
        # No override needed for the get-up groups' own `actions` term: `getup.mdp.filtered_action`
        # already just reads `action_manager.get_term("joint_pos").filtered_actions`, and
        # `HandoffJointPositionAction` exposes that exact property name on purpose (mdp.py).
        self.observations.walk = WalkObsCfg()

        # --- actions: one combined term instead of the get-up task's own action term. -----------
        # Copy the trained get-up numbers from whatever the parent class configured, so this file
        # never drifts from the get-up task's tuned values.
        getup_action_cfg = self.actions.joint_pos
        self.actions.joint_pos = handoff_mdp.HandoffJointPositionActionCfg(
            asset_name="robot",
            joint_names=list(ASIMOV_1_JOINT_NAMES),
            preserve_order=True,
            getup_beta=getattr(getup_action_cfg, "beta", 1.0),
            getup_lpf_alpha=getattr(getup_action_cfg, "lpf_alpha", 0.24),
            getup_use_lpf=getattr(getup_action_cfg, "use_lpf", True),
            getup_scale_torque_factor=getattr(getup_action_cfg, "scale_torque_factor", 1.1),
            walk_scale=0.25,
            walk_use_default_offset=True,
        )

        # --- terminations: keep the parent's, just make them irrelevant for a long demo run.
        # `Asimov1GetUpEnvCfg_PLAY` already sets `no_height_progress = None` (both the termination
        # and its reward), so the only env-driven resets left are `time_out` and the `invalid_state`
        # safety net (NaN / absurd velocity or height) — both fine to keep, since the demo script
        # owns the fall -> get-up -> walk -> push -> get-up sequence and just needs `time_out` not
        # to fire mid-demo.
        self.episode_length_s = 1.0e6

        # Both `walk` (78-dim, unnoised for PLAY) and get-up's own `policy`/`critic` groups are
        # computed every manager step regardless of which is "active" (manager_based_rl_env.py
        # computes every declared group each step), which is exactly what keeps both action paths'
        # internal filters warm for a clean handoff — see mdp.py's module docstring.
