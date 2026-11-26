"""Actuator model of Asimov-1's parallel ankles: a joint-space PD whose gains follow the nonlinear linkage.

Everything stays in joint space, as with :class:`DelayedPDActuator`: the PD acts on the joint errors ``e = q* - q`` and
``de = dq* - dq``, and the torque is clipped per joint to ``actuator_effort_limit``. The only change is the gains. Each
ankle's two motors (A, B) have fixed PD gains ``K_m``, ``D_m``, so the joint gains are

    K_j(q) = J(q)^T K_m J(q),    D_j(q) = J(q)^T D_m J(q),

with ``J(q)`` the linkage Jacobian (motor angles per joint angle). They depend on the pose: at the robot's standing pose
they are about 1.7x (pitch) and 1.5x (roll) their value at zero, which a constant joint-space gain does not reproduce,
and they couple pitch and roll slightly. Per physics step and ankle: ``tau_j = clip(K_j(q) e + D_j(q) de)``.

The stiffness and damping configured for the ankle joints are the gains of the motor in that slot: motor A sits in the
pitch slot and motor B in the roll slot. Gain randomization therefore perturbs each motor's gains, and unequal motor
gains give the pitch-roll coupling they give on the robot.
"""

from __future__ import annotations

from collections.abc import Sequence
import torch
from isaaclab.actuators import DelayedPDActuator, DelayedPDActuatorCfg
from isaaclab.utils import configclass
from isaaclab.utils.types import ArticulationActions

from .asimov_1_ankle_kinematics import motor_angles_and_jacobian

_SIDES = ("left", "right")


class ParallelAnkleActuator(DelayedPDActuator):
    """Delayed joint-space PD for the two parallel ankles, with pose-dependent gains (see the module docstring)."""

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

        # Joint-space errors, as in DelayedPDActuator.
        err = torch.stack((pitch_t - pitch, roll_t - roll), dim=-1)  # (N, 2 legs, 2)
        derr = torch.stack((target_vel[:, pi] - pitch_vel, target_vel[:, ri] - roll_vel), dim=-1)

        # Pose-dependent joint gains J^T K_m J and J^T D_m J, from each motor's (possibly randomized) gains:
        # motor A in the pitch slot, B in the roll slot.
        _, jac = motor_angles_and_jacobian(pitch, roll, self._right)  # (N, 2, 2, 2)
        jt = jac.transpose(-1, -2)
        km = torch.stack((self.stiffness[:, pi], self.stiffness[:, ri]), dim=-1)
        dm = torch.stack((self.damping[:, pi], self.damping[:, ri]), dim=-1)
        kj = jt @ (km.unsqueeze(-1) * jac)
        dj = jt @ (dm.unsqueeze(-1) * jac)
        self.joint_stiffness_matrix = kj  # (num_envs, 2 legs, 2, 2), for logging
        tau = (kj @ err.unsqueeze(-1) + dj @ derr.unsqueeze(-1)).squeeze(-1)

        computed = torch.zeros_like(joint_pos)
        computed[:, pi], computed[:, ri] = tau[..., 0], tau[..., 1]
        self.computed_effort = computed + feedforward
        # Per-joint clip to actuator_effort_limit, as in DelayedPDActuator.
        self.applied_effort = self._clip_effort(self.computed_effort, joint_vel)

        control_action.joint_efforts = self.applied_effort
        control_action.joint_positions = None
        control_action.joint_velocities = None
        return control_action


@configclass
class ParallelAnkleActuatorCfg(DelayedPDActuatorCfg):
    """Configuration of :class:`ParallelAnkleActuator`.

    ``stiffness`` and ``damping`` are the motors' PD gains (motor A in the pitch slot, B in the roll slot); the joint
    gains are ``J(q)^T K_m J(q)`` and ``J(q)^T D_m J(q)``. ``actuator_effort_limit`` clips the joint torques, per joint.
    """

    class_type: type = ParallelAnkleActuator

    pitch_target_limit: float = 0.35
    """Joint pitch target clamp (rad), as the firmware applies it."""

    roll_target_limit: float = 0.09
    """Joint roll target clamp (rad), as the firmware applies it."""
