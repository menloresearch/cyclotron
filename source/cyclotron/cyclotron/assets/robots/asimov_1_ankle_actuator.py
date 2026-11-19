"""Actuator model of Asimov-1's parallel ankles: the motor-side PD loops, simulated through the nonlinear linkage.

On the robot each ankle's two motors (A, B) run their own PD loops on the drivers at fixed gains, and the firmware
maps the policy's virtual (pitch, roll) targets to motor targets. The resulting joint impedance,
``K_j(q) = J(q)^T K_m J(q)``, depends on the pose: at the robot's standing pose it is about 1.7x (pitch) and 1.5x
(roll) the value of the linear map at zero, which a constant joint-space PD gain does not reproduce.

This actuator reproduces the robot instead of assuming constant joint gains. Per physics step and ankle:

* the (delayed) virtual targets are clamped like the firmware and mapped to motor targets, through the firmware's
  linear map ``T0`` (``target_map="linear"``) or the full linkage (``target_map="nonlinear"``),
* the motor angles and velocities come from the joint state through the linkage, ``q_m = g(q)``, ``dq_m = J dq``,
* each motor's PD torque ``tau_m = k_p (q_m* - q_m) + k_d (dq_m* - dq_m)`` is clipped to the motor's torque limit,
* the joint torques are ``tau_j = J^T tau_m``.

The stiffness and damping configured for the ankle joints are the gains of the motor in that slot, as the firmware
assigns them: motor A sits in the pitch slot and motor B in the roll slot. Gain randomization therefore perturbs
each motor's gains, and unequal motor gains give the pitch-roll coupling they give on the robot.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import Literal

import torch
from isaaclab.actuators import DelayedPDActuator, DelayedPDActuatorCfg
from isaaclab.utils import configclass
from isaaclab.utils.types import ArticulationActions

from .asimov_1_ankle_kinematics import linear_motor_angles, motor_angles_and_jacobian

_SIDES = ("left", "right")


class ParallelAnkleActuator(DelayedPDActuator):
    """Delayed motor-side PD for the two parallel ankles, mapped through the linkage (see the module docstring)."""

    cfg: ParallelAnkleActuatorCfg

    def __init__(self, cfg: ParallelAnkleActuatorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        names = list(self.joint_names)
        expected = {f"{side}_ankle_{joint}_joint" for side in _SIDES for joint in ("pitch", "roll")}
        if set(names) != expected:
            raise ValueError(
                f"ParallelAnkleActuator drives exactly the four ankle joints {sorted(expected)}, got {names}"
            )
        # Column indices of (pitch, roll) for each leg within this actuator's joints.
        self._pitch_ids = torch.tensor([names.index(f"{s}_ankle_pitch_joint") for s in _SIDES], device=self._device)
        self._roll_ids = torch.tensor([names.index(f"{s}_ankle_roll_joint") for s in _SIDES], device=self._device)
        self._right = torch.tensor([False, True], device=self._device)

    def reset(self, env_ids: Sequence[int]):
        super().reset(env_ids)

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        # Actuation delay, as in DelayedPDActuator (the robot delays the targets).
        target_pos = self.positions_delay_buffer.compute(control_action.joint_positions)
        target_vel = self.velocities_delay_buffer.compute(control_action.joint_velocities)
        feedforward = self.efforts_delay_buffer.compute(control_action.joint_efforts)

        pi, ri = self._pitch_ids, self._roll_ids
        # (num_envs, 2 legs) each
        pitch, roll = joint_pos[:, pi], joint_pos[:, ri]
        pitch_vel, roll_vel = joint_vel[:, pi], joint_vel[:, ri]
        pitch_t = torch.clamp(target_pos[:, pi], -self.cfg.pitch_target_limit, self.cfg.pitch_target_limit)
        roll_t = torch.clamp(target_pos[:, ri], -self.cfg.roll_target_limit, self.cfg.roll_target_limit)

        # Motor state from the joint state, through the linkage.
        qm, jac = motor_angles_and_jacobian(pitch, roll, self._right)  # (N, 2, 2), (N, 2, 2, 2)
        dq = torch.stack((pitch_vel, roll_vel), dim=-1)
        dqm = (jac @ dq.unsqueeze(-1)).squeeze(-1)
        # Motor targets, as the firmware computes them, clamped to the motors' range.
        if self.cfg.target_map == "linear":
            qm_t = linear_motor_angles(pitch_t, roll_t)
        else:
            qm_t, _ = motor_angles_and_jacobian(pitch_t, roll_t, self._right)
        qm_t = torch.clamp(qm_t, -self.cfg.motor_position_limit, self.cfg.motor_position_limit)
        dq_t = torch.stack((target_vel[:, pi], target_vel[:, ri]), dim=-1)
        dqm_t = (jac @ dq_t.unsqueeze(-1)).squeeze(-1)

        # Each motor's own PD loop, with its (possibly randomized) gains: motor A in the pitch slot, B in the roll slot.
        kp = torch.stack((self.stiffness[:, pi], self.stiffness[:, ri]), dim=-1)
        kd = torch.stack((self.damping[:, pi], self.damping[:, ri]), dim=-1)
        tau_m = kp * (qm_t - qm) + kd * (dqm_t - dqm)
        tau_m_applied = torch.clamp(tau_m, -self.cfg.motor_effort_limit, self.cfg.motor_effort_limit)

        # Back to the virtual joints: tau_j = J^T tau_m.
        jt = jac.transpose(-1, -2)
        tau_j = (jt @ tau_m.unsqueeze(-1)).squeeze(-1)
        tau_j_applied = (jt @ tau_m_applied.unsqueeze(-1)).squeeze(-1)

        computed = torch.zeros_like(joint_pos)
        applied = torch.zeros_like(joint_pos)
        computed[:, pi], computed[:, ri] = tau_j[..., 0], tau_j[..., 1]
        applied[:, pi], applied[:, ri] = tau_j_applied[..., 0], tau_j_applied[..., 1]
        self.computed_effort = computed + feedforward
        self.applied_effort = applied + feedforward
        self.motor_effort = tau_m_applied  # (num_envs, 2 legs, motors A/B), for logging

        control_action.joint_efforts = self.applied_effort
        control_action.joint_positions = None
        control_action.joint_velocities = None
        return control_action


@configclass
class ParallelAnkleActuatorCfg(DelayedPDActuatorCfg):
    """Configuration of :class:`ParallelAnkleActuator`.

    ``stiffness`` and ``damping`` are the motors' PD gains (motor A in the pitch slot, B in the roll slot). The
    joint-space effort limits of the base configuration are not used for clipping; the motor torque limit is.
    """

    class_type: type = ParallelAnkleActuator

    motor_effort_limit: float = MISSING
    """Torque limit of each ankle motor (N m), as the firmware clamps it."""

    motor_position_limit: float = 1.22
    """Range of each ankle motor's target (rad), as the firmware clamps it."""

    pitch_target_limit: float = 0.35
    """Virtual pitch target clamp (rad), as the firmware applies it before the motor map."""

    roll_target_limit: float = 0.09
    """Virtual roll target clamp (rad), as the firmware applies it before the motor map."""

    target_map: Literal["linear", "nonlinear"] = "linear"
    """How the firmware maps virtual targets to motor targets: its linear map T0, or the full linkage kinematics."""
