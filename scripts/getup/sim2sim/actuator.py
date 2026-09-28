# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Python-side per-joint PD actuator with Isaac's ``DelayedPDActuator`` delay model and a runtime effort scale.

Mirrors ``isaaclab.actuators.actuator_pd.{IdealPDActuator,DelayedPDActuator}`` (verified directly against
``third_party/IsaacLab/source/isaaclab/isaaclab/actuators/actuator_pd.py``):

* ``torque = Kp * (delayed_target_pos - q_now) + Kd * (0 - qdot_now)``, clipped to ``+/- effort_limit``. The
  velocity target is always 0 (no ``JointVelocityAction`` in either layout here).
* The delay buffer delays the *commanded target position* (not the joint state) by a fixed number of Isaac physics
  steps (``ASIMOV_1_ACTUATORS`` / ``ASIMOV_1_GETUP_ACTUATORS``: 0-5 steps at Isaac's dt=0.005s = 0-25 ms), drawn once
  per episode and held fixed. Converted here to a step count at *this* simulator's own physics dt (1 kHz/500 Hz/
  Isaac's own 0.005), rounding to the nearest step (the delay is a time, not a step count).
"""

from __future__ import annotations

import re

import numpy as np

from . import constants as C
from .constants import JointArrays
from .filters import TimeDelayBuffer


def _effort_scale_vector(scale: float | dict[str, float], joint_names: list[str]) -> np.ndarray:
    if isinstance(scale, (int, float)):
        return np.full(len(joint_names), float(scale))
    out = np.full(len(joint_names), np.nan)
    used = {k: False for k in scale}
    for i, name in enumerate(joint_names):
        hits = [k for k in scale if re.fullmatch(k, name)]
        if len(hits) > 1:
            raise ValueError(f"joint {name!r} matched by several effort_scale keys {hits}")
        if hits:
            used[hits[0]] = True
            out[i] = float(scale[hits[0]])
    if not np.all([used[k] for k in scale]):
        raise ValueError(f"effort_scale keys matched no joint: {[k for k, u in used.items() if not u]}")
    if np.any(np.isnan(out)):
        missing = [joint_names[i] for i in range(len(joint_names)) if np.isnan(out[i])]
        raise ValueError(f"effort_scale dict does not cover every joint; missing {missing}")
    return out


class DelayedPDActuatorSim:
    """Vectorized (23-joint) equivalent of ``DelayedPDActuator``, running at this simulator's own physics dt."""

    def __init__(
        self,
        joints: JointArrays,
        physics_dt: float,
        effort_scale: float | dict[str, float] = 1.0,
        rng: np.random.Generator | None = None,
        joint_names: list[str] = C.ASIMOV_1_JOINT_NAMES,
        limp_damping: float = 0.5,
    ):
        self.joints = joints
        self.physics_dt = float(physics_dt)
        self.joint_names = joint_names
        self.rng = rng or np.random.default_rng()
        self.limp_damping = float(limp_damping)
        self.base_effort_limit = joints.effort_limit.copy()
        self.effort_scale = np.ones(len(joint_names))
        self.effort_limit = self.base_effort_limit.copy()
        self.set_effort_scale(effort_scale)

        steps_per_isaac_step = C.ISAAC_PHYSICS_DT / self.physics_dt
        min_lag = np.round(joints.min_delay * steps_per_isaac_step).astype(int)
        max_lag = np.round(joints.max_delay * steps_per_isaac_step).astype(int)
        max_lag = np.maximum(max_lag, min_lag)
        self._delay = TimeDelayBuffer(len(joint_names), min_lag, max_lag, self.rng)

    def set_effort_scale(self, scale: float | dict[str, float]) -> None:
        self.effort_scale = _effort_scale_vector(scale, self.joint_names)
        self.effort_limit = self.base_effort_limit * self.effort_scale

    def reset(self, q0: np.ndarray) -> None:
        """Resample the per-joint delay lag and fill the delay buffer with the initial position (no startup torque
        transient -- the first delayed target equals the actual starting position)."""
        self._delay.resample()
        self._delay.reset(value=q0)

    def compute(self, target_pos: np.ndarray, q_now: np.ndarray, qdot_now: np.ndarray, limp: bool = False) -> np.ndarray:
        """One physics-substep PD update. ``target_pos`` is this substep's *commanded* target (already includes
        whatever the action term wants -- absolute for walking, ``q_meas + delta`` for get-up).

        ``limp=True`` mirrors ``DelayedPDLimpableActuator`` at ``gain_scale=0``: Kp=0, Kd=``limp_damping``
        (0.5 N m s/rad, absolute) on every joint, used for the ``mid_fall`` limp phase and for the fallen-state
        generator's settle drop. The delay buffer still runs (so it doesn't desync once control
        resumes) but ``target_pos`` is irrelevant when Kp=0.
        """
        delayed_target = self._delay.push_and_read(np.asarray(target_pos, dtype=float))
        if limp:
            torque = self.limp_damping * (0.0 - qdot_now)
            return np.clip(torque, -self.effort_limit, self.effort_limit)
        torque = self.joints.stiffness * (delayed_target - q_now) + self.joints.damping * (0.0 - qdot_now)
        return np.clip(torque, -self.effort_limit, self.effort_limit)
