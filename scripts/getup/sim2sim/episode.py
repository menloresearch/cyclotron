# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Ties the MJCF model, actuator, observation builder, action term and policy into one episode runner.

Per-tick order mirrors ``ManagerBasedRLEnv.step``: read state -> build obs -> policy(obs) -> action.set_action
(once) -> for each of ``decimation`` physics substeps: recompute the per-substep target, run the (delayed) PD
actuator, ``mj_step`` -> next tick.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import mujoco
import numpy as np

from . import constants as C
from .actions import GetupAction, WalkingAction
from .actuator import DelayedPDActuatorSim
from .mj_model import RobotModel
from .obs import ObservationBuilder


@dataclass
class TickRecord:
    t: float
    qpos: np.ndarray
    qvel: np.ndarray
    root_pos: np.ndarray
    root_quat: np.ndarray
    root_lin_vel: np.ndarray
    torque: np.ndarray
    effort_limit: np.ndarray
    action_raw: np.ndarray
    action_obs: np.ndarray
    feet_force: tuple[float, float]  # peak contact-force norm this tick, per foot [N]
    non_foot_impact_force: float  # raw, single-substep peak this tick [N] -- can include solver spikes
    non_foot_impact_force_smoothed5: float  # peak of a 5-substep moving-average window this tick [N]
    policy_active: bool


@dataclass
class EpisodeConfig:
    mode: str  # "walking" | "getup"
    physics_dt: float = C.MJ_PHYSICS_DT_1KHZ
    effort_scale: float | dict[str, float] = 1.0
    armature_source: str = "isaac"
    enable_obs_noise: bool = True
    enable_action_delay: bool = True
    seed: int = 0
    # get-up action term knobs (tasks/getup/mdp/actions.py)
    beta: float = C.GETUP_BETA_DEFAULT
    use_lpf: bool = C.GETUP_USE_LPF
    lpf_alpha: float = C.GETUP_LPF_ALPHA
    s_j_override: np.ndarray | None = None  # from ONNX metadata's action_s_j, if present -- see actions.py


class Sim2SimEpisode:
    def __init__(self, robot: RobotModel, cfg: EpisodeConfig, policy=None):
        self.robot = robot
        self.cfg = cfg
        self.policy = policy
        self.model = robot.model
        self.data = robot.data
        self.model.opt.timestep = cfg.physics_dt
        self.rng = np.random.default_rng(cfg.seed)
        self.default_qpos = C.standing_init_joint_pos()

        if not cfg.enable_action_delay:
            robot.joints.min_delay[:] = 0
            robot.joints.max_delay[:] = 0
        self.actuator = DelayedPDActuatorSim(robot.joints, cfg.physics_dt, cfg.effort_scale, self.rng)
        self.obs_builder = ObservationBuilder(cfg.mode, self.default_qpos, self.rng, cfg.enable_obs_noise)

        if cfg.mode == "walking":
            self.action = WalkingAction(robot.joints, self.default_qpos, scale=C.ACTION_SCALE_WALKING)
        else:
            self.action = GetupAction(
                robot.joints, beta=cfg.beta, use_lpf=cfg.use_lpf, lpf_alpha=cfg.lpf_alpha,
                s_j_override=cfg.s_j_override,
            )

        self.decimation = round(C.POLICY_DT / cfg.physics_dt)
        if abs(self.decimation * cfg.physics_dt - C.POLICY_DT) > 1e-9:
            raise ValueError(f"policy_dt={C.POLICY_DT} is not an integer multiple of physics_dt={cfg.physics_dt}")

        self.t = 0.0
        self.policy_active = True
        self.limp_until_s = 0.0
        self._history: list[TickRecord] = []
        self._impact_window: deque[float] = deque(maxlen=5)  # 5-substep moving window (spans tick boundaries)

    # -- setup ------------------------------------------------------------------------------------------------------

    def reset(
        self,
        qpos_joints: np.ndarray,
        root_pos: np.ndarray,
        root_quat_wxyz: np.ndarray,
        qvel_joints: np.ndarray | None = None,
        root_lin_vel: np.ndarray | None = None,
        root_ang_vel: np.ndarray | None = None,
        limp_until_s: float = 0.0,
    ) -> None:
        mujoco.mj_resetData(self.model, self.data)
        r = self.robot
        self.data.qpos[r.free_joint_qpos_adr : r.free_joint_qpos_adr + 3] = root_pos
        self.data.qpos[r.free_joint_qpos_adr + 3 : r.free_joint_qpos_adr + 7] = root_quat_wxyz
        self.data.qpos[r.joint_qpos_adr] = qpos_joints
        if root_lin_vel is not None:
            self.data.qvel[r.free_joint_dof_adr : r.free_joint_dof_adr + 3] = root_lin_vel
        if root_ang_vel is not None:
            self.data.qvel[r.free_joint_dof_adr + 3 : r.free_joint_dof_adr + 6] = root_ang_vel
        if qvel_joints is not None:
            self.data.qvel[r.joint_dof_adr] = qvel_joints
        mujoco.mj_forward(self.model, self.data)

        self.actuator.reset(self.data.qpos[r.joint_qpos_adr].copy())
        self.action.reset()
        self.obs_builder.reset()
        self.t = 0.0
        self.limp_until_s = limp_until_s
        self.policy_active = limp_until_s <= 0.0
        self._impact_window.clear()
        self._history = []

    # -- per-tick sensor readout --------------------------------------------------------------------------------

    def _read_state(self):
        r, d = self.robot, self.data
        qpos = d.qpos[r.joint_qpos_adr].copy()
        qvel = d.qvel[r.joint_dof_adr].copy()
        root_pos = d.qpos[r.free_joint_qpos_adr : r.free_joint_qpos_adr + 3].copy()
        gyro = d.sensordata[r.gyro_sensor_adr : r.gyro_sensor_adr + 3].copy()
        quat = d.sensordata[r.quat_sensor_adr : r.quat_sensor_adr + 4].copy()
        lin_vel = d.sensordata[r.vel_sensor_adr : r.vel_sensor_adr + 3].copy()
        return qpos, qvel, root_pos, gyro, quat, lin_vel

    def _feet_contact_and_impact(self) -> tuple[tuple[float, float], float]:
        """Returns ``((left_foot_peak_N, right_foot_peak_N), non_foot_peak_N)`` for this substep -- the caller
        maxes these over the tick's substeps. Peak force (not just "any contact"), so the caller can apply
        Isaac's own ``feet_in_contact`` threshold (``FOOT_CONTACT_FORCE = 1.0`` N, ``tasks/getup/mdp/state.py``)
        instead of flagging a near-zero grazing contact as "in contact"."""
        r, d, m = self.robot, self.data, self.model
        foot_ids = set(r.foot_body_id)
        foot_force = [0.0, 0.0]
        non_foot_peak = 0.0
        for i in range(d.ncon):
            con = d.contact[i]
            b1 = m.geom_bodyid[con.geom1]
            b2 = m.geom_bodyid[con.geom2]
            force6 = np.zeros(6)
            mujoco.mj_contactForce(m, d, i, force6)
            force_norm = float(np.linalg.norm(force6[:3]))
            for side_idx, fid in enumerate(r.foot_body_id):
                if b1 == fid or b2 == fid:
                    foot_force[side_idx] = max(foot_force[side_idx], force_norm)
            if b1 not in foot_ids and b2 not in foot_ids:
                non_foot_peak = max(non_foot_peak, force_norm)
        return (foot_force[0], foot_force[1]), non_foot_peak

    # -- stepping -----------------------------------------------------------------------------------------------

    def step(self, command: np.ndarray | None = None) -> TickRecord:
        qpos, qvel, root_pos, gyro, quat, lin_vel = self._read_state()
        self.policy_active = self.t >= self.limp_until_s

        obs = self.obs_builder.build(
            base_ang_vel=gyro, quat_wxyz=quat, qpos=qpos, qvel=qvel,
            action_obs=self.action.obs_action(), command=command,
        )
        raw_action = self._policy(obs)

        if isinstance(self.action, GetupAction):
            self.action.set_action(raw_action, active=self.policy_active, q_now=qpos)
        else:
            self.action.set_action(raw_action)

        torque_last = np.zeros(C.NUM_JOINTS)
        max_non_foot_impact = 0.0
        max_non_foot_impact_smoothed5 = 0.0
        feet_force = [0.0, 0.0]
        for _ in range(self.decimation):
            q_now = self.data.qpos[self.robot.joint_qpos_adr]
            qdot_now = self.data.qvel[self.robot.joint_dof_adr]
            target = self.action.target(q_now)
            torque = self.actuator.compute(target, q_now, qdot_now, limp=not self.policy_active)
            self.data.ctrl[self.robot.actuator_id] = torque
            mujoco.mj_step(self.model, self.data)
            # PhysX hard-enforces the URDF/datasheet joint velocity limit; MuJoCo has no
            # native per-hinge velocity limit, so clamp post-integration to match.
            qvel_ids = self.robot.joint_dof_adr
            vlim = self.robot.joints.velocity_limit
            self.data.qvel[qvel_ids] = np.clip(self.data.qvel[qvel_ids], -vlim, vlim)
            self.t += self.cfg.physics_dt
            torque_last = torque
            fc, impact = self._feet_contact_and_impact()
            feet_force = [max(feet_force[0], fc[0]), max(feet_force[1], fc[1])]
            max_non_foot_impact = max(max_non_foot_impact, impact)
            # 5-substep moving-average window: the raw per-substep peak can be a single
            # rigid-contact solver spike; the windowed mean is a more robust "was there a sustained hard impact"
            # signal. The window spans tick boundaries (never reset mid-episode), so it also smooths the exact
            # instant a policy-tick boundary falls mid-impact.
            self._impact_window.append(impact)
            windowed_mean = float(np.mean(self._impact_window))
            max_non_foot_impact_smoothed5 = max(max_non_foot_impact_smoothed5, windowed_mean)

        qpos_f, qvel_f, root_pos_f, _, quat_f, lin_vel_f = self._read_state()
        rec = TickRecord(
            t=self.t, qpos=qpos_f, qvel=qvel_f, root_pos=root_pos_f, root_quat=quat_f, root_lin_vel=lin_vel_f,
            torque=torque_last, effort_limit=self.actuator.effort_limit.copy(),
            action_raw=raw_action, action_obs=self.action.obs_action().copy(),
            feet_force=(feet_force[0], feet_force[1]), non_foot_impact_force=max_non_foot_impact,
            non_foot_impact_force_smoothed5=max_non_foot_impact_smoothed5,
            policy_active=self.policy_active,
        )
        self._history.append(rec)
        return rec

    def _policy(self, obs: np.ndarray) -> np.ndarray:
        if self.policy is None:
            raise RuntimeError("Sim2SimEpisode.policy is not set; pass policy= to the constructor or set_policy().")
        return self.policy(obs)

    def set_policy(self, policy) -> None:
        self.policy = policy

    @property
    def history(self) -> list[TickRecord]:
        return self._history
