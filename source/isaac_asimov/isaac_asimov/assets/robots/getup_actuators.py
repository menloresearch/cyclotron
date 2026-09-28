# Copyright (c) 2026, Menlo Research.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Actuator models and runtime actuator controls for the Asimov 1 get-up task.

Contents
--------
* :class:`DelayedPDLimpableActuator` / :class:`DelayedPDLimpableActuatorCfg`
    Same model as Isaac Lab's ``DelayedPDActuator`` (identical gains, delays, armature, friction, clip), but with a
    **host-sync-free** command delay (:class:`SyncFreeDelayBuffer`, bit-identical to Isaac Lab's ``DelayBuffer``, see
    ``tests/getup/test_getup_actuators.py``), optional per-joint-group lags so one actuator can cover all 23 joints
    (``delay_groups``), and a per-env ``gain_scale`` tensor that blends between nominal PD and a *limp* mode:

    .. math::

        s &= \\max(\\text{gain\\_scale}, 0) \\\\
        K_p^{eff} &= s \\cdot K_p \\\\
        K_d^{eff} &= s \\cdot K_d + \\max(1 - s, 0) \\cdot K_d^{limp}

    with ``K_d^{limp} = cfg.limp_damping = 0.5`` N m s/rad (absolute). So ``gain_scale = 1`` is the nominal
    controller, ``gain_scale = 0`` is exactly *limp* (Kp = 0, Kd = 0.5 absolute, for every joint), and values in
    between blend linearly, which gives a smooth ramp out of limp. Values > 1 scale both gains up (no limp term).
    ``gain_scale`` is reset to 1 in :meth:`reset` (``scene.reset`` runs *before* the reset events, so a reset event
    such as ``reset_fallen_state`` can set limp afterwards). The gains are applied at compute time, not written
    into ``stiffness``/``damping``, so gain randomization events that edit those tensors keep working.

* :class:`DelayedDCMotorLimpable` / :class:`DelayedDCMotorLimpableCfg` (Stage B)
    Everything above, plus

    1. a linear four-quadrant torque-speed curve per joint (same formula as Isaac Lab's ``DCMotor``) with per-joint
       ``saturation_effort`` (stall torque) and ``velocity_limit`` (no-load speed), then the flat ``effort_limit``
       clip; and
    2. for each ankle whose pitch **and** roll joints are in the group, the torque-speed curve and a hard clip are
       applied to the **two physical motors** (A, B) of the differential instead of to the joints (see
       :func:`ankle_motor_torques`).

* :func:`set_effort_scale` - runtime per-joint effort-limit scaling relative to the cfg's base effort limits.
* :func:`set_gain_scale` - sets ``gain_scale`` on every limpable actuator of the robot (used by ``set_limp``).
* :func:`get_effort_limits` - current (scaled) effort limits in articulation joint order.
* :func:`ankle_motor_torques` - joint-space ankle torques to per-motor torques.

Ankle differential (derivation)
-------------------------------
The ankle differential transmission maps ankle joint angles to motor angles with the *position* transform
(ratios are configurable; confirm with the hardware team)

.. math::

    m_A = K_p\\,p - K_r\\,r, \\qquad m_B = -K_p\\,p - K_r\\,r, \\qquad K_p = 2.02,\\; K_r = 0.80,

i.e. :math:`m = J q` with :math:`J = [[K_p, -K_r], [-K_p, -K_r]]`. For an ideal (lossless) transmission, virtual work
:math:`\\tau_q^T \\dot q = \\tau_m^T \\dot m` gives :math:`\\tau_q = J^T \\tau_m`:

.. math::

    \\tau_p = K_p(\\tau_A - \\tau_B), \\qquad \\tau_r = -K_r(\\tau_A + \\tau_B)

and inverting (:math:`\\det J = -2 K_p K_r \\neq 0`):

.. math::

    \\tau_A = \\tfrac12\\left(\\frac{\\tau_p}{K_p} - \\frac{\\tau_r}{K_r}\\right), \\qquad
    \\tau_B = \\tfrac12\\left(-\\frac{\\tau_p}{K_p} - \\frac{\\tau_r}{K_r}\\right).

Motor speeds follow the position map directly:
:math:`\\dot m_A = K_p \\dot p - K_r \\dot r`, :math:`\\dot m_B = -K_p \\dot p - K_r \\dot r`.

.. warning::
    An alternative reading of the transmission treats 2.02 as a position-only ratio and not a torque multiplier
    (max joint torque = 2 x motor torque). That model is inconsistent with the position map above under energy
    conservation (it would require :math:`\\tau_p = \\tau_A - \\tau_B`). If it is right, per-motor
    torques are :math:`K_p` (resp. :math:`K_r`) times larger than computed here for a given joint torque, i.e. the
    ankle pitch saturates ~2x sooner. The torque-side gains are therefore configurable (``ankle_torque_k_pitch``,
    ``ankle_torque_k_roll``; set both to 1.0 for the conservative model). Default: virtual work (2.02, 0.80). This must
    be confirmed with the hardware team.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.actuators import ActuatorBase, DelayedPDActuatorCfg, IdealPDActuator
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.utils.types import ArticulationActions

# ---------------------------------------------------------------------------------------------------------------------
# Motor datasheet values (ENCOS EC series, manufacturer datasheets)
# ---------------------------------------------------------------------------------------------------------------------

ANKLE_K_PITCH = 2.02
"""Ankle differential position ratio for pitch (motor rad per joint rad)."""
ANKLE_K_ROLL = 0.80
"""Ankle differential position ratio for roll (motor rad per joint rad)."""

_RPM = 2.0 * math.pi / 60.0

MOTOR_DATASHEET: dict[str, dict[str, float]] = {
    # rated (continuous) torque [Nm], peak/stall torque [Nm], rated speed [rad/s], peak speed [rad/s]
    "EC-A6416-P2-25": {"rated_torque": 40.0, "peak_torque": 120.0, "rated_speed": 107 * _RPM, "peak_speed": 120 * _RPM},
    "EC-A5013-H17-100": {"rated_torque": 30.0, "peak_torque": 90.0, "rated_speed": 33 * _RPM, "peak_speed": 38 * _RPM},
    "EC-A3814-H14-107": {"rated_torque": 20.0, "peak_torque": 60.0, "rated_speed": 47 * _RPM, "peak_speed": 52 * _RPM},
    "EC-A4315-P2-36": {"rated_torque": 25.0, "peak_torque": 75.0, "rated_speed": 109 * _RPM, "peak_speed": 117 * _RPM},
    "EC-A4310-P2-36": {"rated_torque": 12.0, "peak_torque": 36.0, "rated_speed": 75 * _RPM, "peak_speed": 89 * _RPM},
}


def no_load_speed(motor: str) -> float:
    """No-load speed [rad/s] of the linear torque-speed line through (0, peak torque) and (rated speed, rated torque).

    The datasheets give only these two operating points. A linear DC-motor line through both gives
    ``w0 = w_rated * T_peak / (T_peak - T_rated)`` (= 1.5 x rated speed for every ENCOS motor used here, since
    peak = 3 x rated). The datasheet "peak speed" (the URDF velocity limits) is lower than ``w0``; it is kept as the
    PhysX joint velocity limit. UNVERIFIED: replace with measured curves when the team provides them.
    """
    d = MOTOR_DATASHEET[motor]
    return d["rated_speed"] * d["peak_torque"] / (d["peak_torque"] - d["rated_torque"])


# ---------------------------------------------------------------------------------------------------------------------
# Ankle transform helpers
# ---------------------------------------------------------------------------------------------------------------------


def ankle_motor_torques(
    tau_pitch: torch.Tensor,
    tau_roll: torch.Tensor,
    k_pitch: float = ANKLE_K_PITCH,
    k_roll: float = ANKLE_K_ROLL,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-motor torques (A, B) of an ankle differential from its joint-space pitch and roll torques.

    ``tau_A = (tau_p / Kp - tau_r / Kr) / 2`` and ``tau_B = (-tau_p / Kp - tau_r / Kr) / 2`` (see module docstring for
    the derivation). Works elementwise on tensors of any matching shape.

    Example (reward): ``ta, tb = ankle_motor_torques(tau[:, l_pitch], tau[:, l_roll])`` then penalize
    ``relu(|ta| - 12)^2``.
    """
    a = tau_pitch / k_pitch
    b = tau_roll / k_roll
    return 0.5 * (a - b), 0.5 * (-a - b)


def ankle_joint_torques(
    tau_a: torch.Tensor, tau_b: torch.Tensor, k_pitch: float = ANKLE_K_PITCH, k_roll: float = ANKLE_K_ROLL
) -> tuple[torch.Tensor, torch.Tensor]:
    """Inverse of :func:`ankle_motor_torques`: ``tau_p = Kp (tau_A - tau_B)``, ``tau_r = -Kr (tau_A + tau_B)``."""
    return k_pitch * (tau_a - tau_b), -k_roll * (tau_a + tau_b)


def ankle_motor_velocities(
    qd_pitch: torch.Tensor, qd_roll: torch.Tensor, k_pitch: float = ANKLE_K_PITCH, k_roll: float = ANKLE_K_ROLL
) -> tuple[torch.Tensor, torch.Tensor]:
    """Motor (output-shaft) speeds of an ankle differential: ``m_A' = Kp p' - Kr r'``, ``m_B' = -Kp p' - Kr r'``."""
    return k_pitch * qd_pitch - k_roll * qd_roll, -k_pitch * qd_pitch - k_roll * qd_roll


def _torque_speed_bounds(
    vel: torch.Tensor, stall: torch.Tensor, no_load: torch.Tensor, cont: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Isaac Lab ``DCMotor`` four-quadrant bounds, with tensors for stall / no-load / continuous torque."""
    vel_at_lim = no_load * (1.0 + cont / stall)
    v = torch.maximum(torch.minimum(vel, vel_at_lim), -vel_at_lim)
    top = stall * (1.0 - v / no_load)
    bottom = stall * (-1.0 - v / no_load)
    return torch.maximum(bottom, -cont), torch.minimum(top, cont)


# ---------------------------------------------------------------------------------------------------------------------
# Limpable delayed PD
# ---------------------------------------------------------------------------------------------------------------------


class SyncFreeDelayBuffer:
    """Drop-in for Isaac Lab's ``DelayBuffer`` (``utils/buffers/delay_buffer.py``) without GPU->CPU syncs.

    Semantics are identical: ``compute(x)`` appends ``x`` to a ring of ``max_delay + 1`` slots and returns the entry
    ``lag`` appends ago; right after a :meth:`reset` of an env, its first append fills every slot (Isaac Lab's
    ``CircularBuffer.append`` first-push rule), so a lag larger than the number of appends since the reset returns the
    oldest one, i.e. the command is *held* at its first post-reset value. Isaac Lab clamps the lag to
    ``num_pushes - 1`` instead; with the first-push fill both give the same value, so no push counter is needed.

    Differences (implementation only): the ring length and pointer are Python ints (Isaac Lab reads them with
    ``.item()``), the first-push fill is an unconditional ``torch.where`` (Isaac Lab: ``if torch.any(...)``), there is no
    emptiness check, and lags can be per (env, joint) (gathered with ``torch.gather``) so several joint groups with
    independent lags fit in one buffer.
    """

    def __init__(self, max_delay: int, num_envs: int, num_joints: int, device: str):
        self.length = int(max(0, max_delay)) + 1
        self.buffer = torch.zeros(self.length, num_envs, num_joints, device=device)
        self.pointer = -1
        self.fresh = torch.ones(num_envs, 1, dtype=torch.bool, device=device)
        self.lags = torch.zeros(num_envs, num_joints, dtype=torch.long, device=device)

    def reset(self, env_ids: slice | torch.Tensor, lags: torch.Tensor | None = None) -> None:
        """Mark ``env_ids`` as fresh (next append fills their history) and optionally set their lags (n, num_joints)."""
        self.fresh[env_ids] = True
        if lags is not None:
            self.lags[env_ids] = lags

    def compute(self, data: torch.Tensor) -> torch.Tensor:
        self.pointer = (self.pointer + 1) % self.length
        self.buffer[self.pointer] = data
        self.buffer = torch.where(self.fresh.unsqueeze(0), data.unsqueeze(0), self.buffer)
        self.fresh.zero_()
        index = torch.remainder(self.pointer - self.lags, self.length).unsqueeze(0)
        return torch.gather(self.buffer, 0, index).squeeze(0)


class DelayedPDLimpableActuator(IdealPDActuator):
    """Delayed PD actuator (Isaac Lab ``DelayedPDActuator`` semantics) without host syncs, with limp and effort scaling.

    Differences to Isaac Lab's ``DelayedPDActuator`` (whose ``DelayBuffer`` costs ~4 GPU->CPU syncs per call):

    * the delay uses :class:`SyncFreeDelayBuffer`;
    * only the **position** target is delayed unless ``cfg.delay_velocity_effort`` is True. In the get-up task the
      velocity and effort targets are always zero (actions write position targets only; resets write zero velocity
      targets), and a delayed zero is zero, so this is exact there;
    * ``cfg.delay_groups`` lets one actuator hold several groups with independent lags (one lag per env and group,
      re-sampled at reset exactly like one ``DelayedPDActuator`` per group). ``None`` = one lag for all joints.

    Attributes:
        gain_scale: Tensor (num_envs,). 1 = nominal PD, 0 = limp (Kp 0, Kd ``cfg.limp_damping``), linear in between.
        effort_scale: Tensor (num_envs, num_joints). Multiplier on the cfg effort limit (see :func:`set_effort_scale`).
        base_effort_limit: Tensor (num_envs, num_joints). The effort limit parsed from the cfg at construction.
    """

    cfg: DelayedPDLimpableActuatorCfg

    def __init__(self, cfg: DelayedPDLimpableActuatorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self.gain_scale = torch.ones(self._num_envs, device=self._device)
        self.base_effort_limit = self.effort_limit.clone()
        self.effort_scale = torch.ones_like(self.effort_limit)
        self._limp_damping = float(cfg.limp_damping)
        # -- delay
        if cfg.min_delay < 0 or cfg.max_delay < cfg.min_delay:
            raise ValueError(f"Invalid delay range [{cfg.min_delay}, {cfg.max_delay}]")
        n, j = self._num_envs, self.num_joints
        self.positions_delay_buffer = SyncFreeDelayBuffer(cfg.max_delay, n, j, self._device)
        self.velocities_delay_buffer = self.efforts_delay_buffer = None
        if cfg.delay_velocity_effort:
            self.velocities_delay_buffer = SyncFreeDelayBuffer(cfg.max_delay, n, j, self._device)
            self.efforts_delay_buffer = SyncFreeDelayBuffer(cfg.max_delay, n, j, self._device)
        group_of_joint = [0] * j
        self._num_delay_groups = 1
        if cfg.delay_groups is not None:
            self._num_delay_groups = len(cfg.delay_groups)
            assigned = [None] * j
            for g, exprs in enumerate(cfg.delay_groups):
                for k, name in enumerate(self.joint_names):
                    if any(re.fullmatch(e, name) for e in exprs):
                        if assigned[k] is not None:
                            raise ValueError(f"delay_groups: joint {name} in groups {assigned[k]} and {g}")
                        assigned[k] = g
            missing = [self.joint_names[k] for k in range(j) if assigned[k] is None]
            if missing:
                raise ValueError(f"delay_groups: joints not in any group: {missing}")
            group_of_joint = assigned
        self._delay_group_of_joint = torch.tensor(group_of_joint, dtype=torch.long, device=self._device)

    def reset(self, env_ids: Sequence[int]):
        if env_ids is None or (isinstance(env_ids, slice) and env_ids == slice(None)):
            env_ids = slice(None)
            num = self._num_envs
        else:
            if not isinstance(env_ids, torch.Tensor):
                env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self._device)
            num = len(env_ids)
        # one lag per (env, delay group), as Isaac Lab samples one lag per env for each DelayedPDActuator group
        lags = torch.randint(
            low=self.cfg.min_delay,
            high=self.cfg.max_delay + 1,
            size=(num, self._num_delay_groups),
            dtype=torch.long,
            device=self._device,
        )[:, self._delay_group_of_joint]
        for buf in (self.positions_delay_buffer, self.velocities_delay_buffer, self.efforts_delay_buffer):
            if buf is not None:
                buf.reset(env_ids, lags)
        self.gain_scale[env_ids] = 1.0

    def apply_effort_scale(self, scale: torch.Tensor, env_ids: slice | torch.Tensor = slice(None)) -> None:
        """Set ``effort_scale[env_ids] = scale`` (broadcastable to (n, num_joints)) and update the limits."""
        self.effort_scale[env_ids] = scale
        self.effort_limit[env_ids] = self.base_effort_limit[env_ids] * self.effort_scale[env_ids]
        self._on_effort_scale_changed()

    def _on_effort_scale_changed(self) -> None:
        """Hook for subclasses with extra limits derived from the effort scale."""
        pass

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        q_des = self.positions_delay_buffer.compute(control_action.joint_positions)
        qd_des = control_action.joint_velocities
        tau_ff = control_action.joint_efforts
        if self.velocities_delay_buffer is not None:
            qd_des = self.velocities_delay_buffer.compute(qd_des)
            tau_ff = self.efforts_delay_buffer.compute(tau_ff)
        # effective gains (limp blend); the nominal stiffness/damping tensors are not modified, so gain randomization
        # events that edit them in place compose with the limp blend
        s = torch.clamp(self.gain_scale, min=0.0).unsqueeze(1)
        kp = self.stiffness * s
        kd = self.damping * s + torch.clamp(1.0 - s, min=0.0) * self._limp_damping
        # same expression order as IdealPDActuator.compute
        self.computed_effort = kp * (q_des - joint_pos) + kd * (qd_des - joint_vel) + tau_ff
        self.applied_effort = self._clip_effort(self.computed_effort)
        control_action.joint_efforts = self.applied_effort
        control_action.joint_positions = None
        control_action.joint_velocities = None
        return control_action


@configclass
class DelayedPDLimpableActuatorCfg(DelayedPDActuatorCfg):
    """Configuration for :class:`DelayedPDLimpableActuator`."""

    class_type: type = DelayedPDLimpableActuator

    limp_damping: float = 0.5
    """Absolute joint damping [N m s/rad] when ``gain_scale == 0`` (limp). Defaults to 0.5."""

    delay_groups: list[list[str]] | None = None
    """Joint-name regex groups with independent delays (one lag per env and group). ``None`` = one lag for all joints
    of this actuator (Isaac Lab ``DelayedPDActuator`` behaviour)."""

    delay_velocity_effort: bool = False
    """Also delay the velocity and effort targets (Isaac Lab does). Off: they are zero in the get-up task."""


# ---------------------------------------------------------------------------------------------------------------------
# Stage B: delayed DC motor with ankle differential
# ---------------------------------------------------------------------------------------------------------------------


class DelayedDCMotorLimpable(DelayedPDLimpableActuator):
    """Limpable delayed PD with a torque-speed curve and per-motor ankle-differential saturation (Stage B).

    Clipping order in :meth:`_clip_effort`:

    1. joint-space four-quadrant torque-speed curve (``saturation_effort``, ``velocity_limit``) capped by the flat
       ``effort_limit``, for every joint that is **not** part of a complete ankle pair in this group;
    2. flat ``effort_limit`` for ankle joints (same as Stage A);
    3. for each complete ankle pair: map to motor torques, apply the motor torque-speed curve
       (``ankle_motor_saturation_effort``, ``ankle_motor_velocity_limit``) capped by ``ankle_motor_effort_limit``,
       and map back. Clamping each motor toward zero never increases \\|tau_p\\| or \\|tau_r\\| (clamp is 1-Lipschitz
       and odd), so step 3 never undoes step 2.

    All torque limits scale with :func:`set_effort_scale` (ankle motor limits use the mean of that ankle's pitch and
    roll scales).
    """

    cfg: DelayedDCMotorLimpableCfg

    def __init__(self, cfg: DelayedDCMotorLimpableCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        if cfg.velocity_limit is None:
            raise ValueError("DelayedDCMotorLimpableCfg.velocity_limit (no-load joint speed) must be set.")
        # stall torque per joint; joints not covered by the cfg get +inf (no curve, flat clip only)
        sat = torch.full_like(self.effort_limit, math.inf)
        if isinstance(cfg.saturation_effort, (float, int)):
            sat[:] = float(cfg.saturation_effort)
        elif isinstance(cfg.saturation_effort, dict):
            for key, val in cfg.saturation_effort.items():
                ids = [j for j, n in enumerate(self.joint_names) if re.fullmatch(key, n)]
                if not ids:
                    raise ValueError(f"saturation_effort key '{key}' matches no joint in {self.joint_names}")
                sat[:, ids] = float(val)
        self.base_saturation_effort = sat
        self.saturation_effort = self.base_saturation_effort.clone()
        self._joint_vel = torch.zeros_like(self.computed_effort)
        # -- ankle pairs (local indices)
        pairs = []
        for side in ("left", "right"):
            names = self.joint_names
            p, r = f"{side}_ankle_pitch_joint", f"{side}_ankle_roll_joint"
            if p in names and r in names:
                pairs.append((names.index(p), names.index(r)))
            elif p in names or r in names:
                raise ValueError(
                    f"Actuator group has only one of {p}/{r}; put both ankle joints in the same group so the"
                    " differential can be modelled."
                )
        self._ankle_pitch_ids = torch.tensor([p for p, _ in pairs], dtype=torch.long, device=self._device)
        self._ankle_roll_ids = torch.tensor([r for _, r in pairs], dtype=torch.long, device=self._device)
        self._has_ankles = len(pairs) > 0
        # joints that use the joint-space curve
        curve = torch.ones(self.num_joints, dtype=torch.bool, device=self._device)
        if self._has_ankles:
            curve[self._ankle_pitch_ids] = False
            curve[self._ankle_roll_ids] = False
        # joints without a finite stall torque (or without a no-load speed) keep the flat clip
        curve &= torch.isfinite(self.base_saturation_effort[0]) & torch.isfinite(self.velocity_limit[0])
        curve &= self.velocity_limit[0] > 0
        self._joint_curve_mask = curve
        self._on_effort_scale_changed()

    def _on_effort_scale_changed(self) -> None:
        if not hasattr(self, "base_saturation_effort"):
            return  # called from the base __init__ path before our buffers exist
        self.saturation_effort = self.base_saturation_effort * self.effort_scale
        if self._has_ankles:
            s = 0.5 * (self.effort_scale[:, self._ankle_pitch_ids] + self.effort_scale[:, self._ankle_roll_ids])
            self.ankle_motor_effort_limit = self.cfg.ankle_motor_effort_limit * s
            self.ankle_motor_saturation_effort = self.cfg.ankle_motor_saturation_effort * s

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        self._joint_vel[:] = joint_vel
        return super().compute(control_action, joint_pos, joint_vel)

    def _clip_effort(self, effort: torch.Tensor) -> torch.Tensor:
        # 1-2. joint space
        m = self._joint_curve_mask
        stall = torch.where(m, self.saturation_effort, 1.0)  # dummy finite values where the curve is unused
        v0 = torch.where(m, self.velocity_limit, 1.0)
        lo, hi = _torque_speed_bounds(self._joint_vel, stall, v0, self.effort_limit)
        lo = torch.where(m, lo, -self.effort_limit)
        hi = torch.where(m, hi, self.effort_limit)
        out = torch.clip(effort, min=lo, max=hi)
        # 3. ankle motors
        if self._has_ankles:
            cfg = self.cfg
            pi, ri = self._ankle_pitch_ids, self._ankle_roll_ids
            ta, tb = ankle_motor_torques(out[:, pi], out[:, ri], cfg.ankle_torque_k_pitch, cfg.ankle_torque_k_roll)
            va, vb = ankle_motor_velocities(self._joint_vel[:, pi], self._joint_vel[:, ri])
            v0 = torch.full_like(ta, cfg.ankle_motor_velocity_limit)
            lo_a, hi_a = _torque_speed_bounds(va, self.ankle_motor_saturation_effort, v0, self.ankle_motor_effort_limit)
            lo_b, hi_b = _torque_speed_bounds(vb, self.ankle_motor_saturation_effort, v0, self.ankle_motor_effort_limit)
            ta = torch.clip(ta, min=lo_a, max=hi_a)
            tb = torch.clip(tb, min=lo_b, max=hi_b)
            self.ankle_motor_effort = torch.stack((ta, tb), dim=-1)  # (num_envs, n_ankles, 2), for logging
            tp, tr = ankle_joint_torques(ta, tb, cfg.ankle_torque_k_pitch, cfg.ankle_torque_k_roll)
            out = out.clone()
            out[:, pi] = tp
            out[:, ri] = tr
        return out


@configclass
class DelayedDCMotorLimpableCfg(DelayedPDLimpableActuatorCfg):
    """Configuration for :class:`DelayedDCMotorLimpable` (Stage B).

    ``velocity_limit`` (inherited) is the joint-space **no-load speed** of the torque-speed line and is required.
    It is used only by the actuator model; the PhysX joint velocity limit is ``velocity_limit_sim``.
    """

    class_type: type = DelayedDCMotorLimpable

    saturation_effort: float | dict[str, float] | None = None
    """Joint-space stall torque [N m] per joint (regex dict) or for all joints. ``None`` = no curve (flat clip)."""

    ankle_motor_effort_limit: float = 12.0
    """Continuous torque clip per ankle motor [N m] (EC-A4310 rated 12 Nm)."""

    ankle_motor_saturation_effort: float = 36.0
    """Stall torque per ankle motor [N m] (EC-A4310 peak 36 Nm)."""

    ankle_motor_velocity_limit: float = no_load_speed("EC-A4310-P2-36")
    """No-load speed per ankle motor [rad/s] (motor output shaft)."""

    ankle_torque_k_pitch: float = ANKLE_K_PITCH
    """Torque-side pitch ratio (virtual work: equals the position ratio 2.02; 1.0 = conservative 2x-motor-torque model)."""

    ankle_torque_k_roll: float = ANKLE_K_ROLL
    """Torque-side roll ratio (virtual work: equals the position ratio 0.80; 1.0 = conservative 2x-motor-torque model)."""


# ---------------------------------------------------------------------------------------------------------------------
# Runtime controls
# ---------------------------------------------------------------------------------------------------------------------


def _resolve_asset(env_or_asset, asset_name: str) -> Articulation:
    if hasattr(env_or_asset, "actuators") and hasattr(env_or_asset, "joint_names"):
        return env_or_asset
    return env_or_asset.scene[asset_name]


def _resolve_env_ids(env_ids, device) -> slice | torch.Tensor:
    if env_ids is None or (isinstance(env_ids, slice) and env_ids == slice(None)):
        return slice(None)
    if isinstance(env_ids, torch.Tensor):
        return env_ids.to(device=device, dtype=torch.long)
    return torch.as_tensor(list(env_ids), device=device, dtype=torch.long)


def _ensure_base_effort(actuator: ActuatorBase) -> None:
    """Give a plain Isaac Lab explicit actuator the attributes used by :func:`set_effort_scale`."""
    if not hasattr(actuator, "base_effort_limit"):
        actuator.base_effort_limit = actuator.effort_limit.clone()
        actuator.effort_scale = torch.ones_like(actuator.effort_limit)


def set_effort_scale(
    env: ManagerBasedEnv | Articulation,
    scale: float | dict[str, float],
    asset_name: str = "robot",
    env_ids: Sequence[int] | torch.Tensor | None = None,
) -> None:
    """Scale actuator effort limits at runtime, relative to the cfg's base effort limits.

    ``effort_limit = base_effort_limit * scale`` (not cumulative: calling with 1.2 twice gives 1.2x, not 1.44x).

    Args:
        env: The env (the articulation is ``env.scene[asset_name]``) or the articulation itself.
        scale: A float applied to every joint, or a dict ``{joint-name regex: scale}`` (``re.fullmatch`` on joint
            names). Joints matched by no key keep their current scale. Every key must match at least one joint and no
            joint may be matched by two keys (raises ``ValueError``).
        asset_name: Scene entity name of the robot. Defaults to ``"robot"``.
        env_ids: Envs to change (default: all). Allows per-env strength randomization.

    Works with every explicit actuator (limpable or not). Implicit actuators are rejected, since their limit lives in
    PhysX. For Stage B actuators the stall torques and ankle motor limits scale too.
    """
    asset = _resolve_asset(env, asset_name)
    ids = _resolve_env_ids(env_ids, asset.device)
    plan = _effort_scale_plan(asset, tuple(scale)) if isinstance(scale, dict) else None
    for k, actuator in enumerate(asset.actuators.values()):
        if actuator.is_implicit_model:
            raise TypeError("set_effort_scale: implicit actuators are not supported.")
        _ensure_base_effort(actuator)
        if plan is not None:
            local_ids, keys = plan[k]
            if not local_ids:
                continue
            vals = torch.tensor([float(scale[key]) for key in keys], device=asset.device)
            new = actuator.effort_scale[ids]
            new[:, local_ids] = vals
        else:
            new = float(scale)
        if isinstance(actuator, DelayedPDLimpableActuator):
            actuator.apply_effort_scale(new, ids)
        else:
            actuator.effort_scale[ids] = new
            actuator.effort_limit[ids] = actuator.base_effort_limit[ids] * actuator.effort_scale[ids]


_EFFORT_PLAN_CACHE: dict[tuple[int, tuple[str, ...]], list[tuple[list[int], list[str]]]] = {}


def _effort_scale_plan(asset, keys: tuple[str, ...]) -> list[tuple[list[int], list[str]]]:
    """Per actuator: (local joint ids, matching key per id) for a dict of regex keys. Cached (no work, no syncs)."""
    cache_key = (id(asset), keys)
    plan = _EFFORT_PLAN_CACHE.get(cache_key)
    if plan is not None:
        return plan
    patterns = {k: re.compile(k) for k in keys}
    used = dict.fromkeys(keys, False)
    plan = []
    for actuator in asset.actuators.values():
        local_ids, hit_keys = [], []
        for j, jname in enumerate(actuator.joint_names):
            hits = [k for k, p in patterns.items() if p.fullmatch(jname)]
            if len(hits) > 1:
                raise ValueError(f"set_effort_scale: joint '{jname}' matched by several keys {hits}")
            if hits:
                used[hits[0]] = True
                local_ids.append(j)
                hit_keys.append(hits[0])
        plan.append((local_ids, hit_keys))
    unused = [k for k, u in used.items() if not u]
    if unused:
        raise ValueError(f"set_effort_scale: keys matched no joint: {unused}")
    _EFFORT_PLAN_CACHE[cache_key] = plan
    return plan


def set_gain_scale(
    env: ManagerBasedEnv | Articulation,
    env_ids: Sequence[int] | torch.Tensor | None,
    scale: float | torch.Tensor,
    asset_name: str = "robot",
) -> None:
    """Set ``gain_scale[env_ids] = scale`` on every :class:`DelayedPDLimpableActuator` of the robot.

    ``scale = 0`` is limp (Kp 0, Kd 0.5 absolute), ``1`` is nominal. ``scale`` may be a float or a tensor of shape
    (len(env_ids),). Raises ``TypeError`` if the robot has no limpable actuator (e.g. the walking asset).
    """
    asset = _resolve_asset(env, asset_name)
    ids = _resolve_env_ids(env_ids, asset.device)
    n = 0
    for actuator in asset.actuators.values():
        if isinstance(actuator, DelayedPDLimpableActuator):
            actuator.gain_scale[ids] = scale
            n += 1
    if n == 0:
        raise TypeError("set_gain_scale: the robot has no DelayedPDLimpableActuator groups (use ASIMOV_1_GETUP_CFG).")


def get_gain_scale(env: ManagerBasedEnv | Articulation, asset_name: str = "robot") -> torch.Tensor:
    """Return the ``gain_scale`` (num_envs,) of the first limpable actuator group (all groups are set together)."""
    asset = _resolve_asset(env, asset_name)
    for actuator in asset.actuators.values():
        if isinstance(actuator, DelayedPDLimpableActuator):
            return actuator.gain_scale
    raise TypeError("get_gain_scale: the robot has no DelayedPDLimpableActuator groups.")


def get_effort_limits(env: ManagerBasedEnv | Articulation, asset_name: str = "robot") -> torch.Tensor:
    """Current (scaled) flat effort limits, shape (num_envs, num_joints), in articulation joint order.

    Use for ``tau / tau_max`` observations and saturation penalties. For Stage B this is the flat continuous clip;
    velocity-dependent limits are lower.
    """
    asset = _resolve_asset(env, asset_name)
    out = torch.full((asset.num_instances, asset.num_joints), float("inf"), device=asset.device)
    for actuator in asset.actuators.values():
        out[:, actuator.joint_indices] = actuator.effort_limit
    return out
