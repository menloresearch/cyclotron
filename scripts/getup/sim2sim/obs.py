# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Reproduces ``ObservationsCfg.PolicyCfg`` (walking, 78-dim, no history) and the get-up policy obs group
(no command, ``history_length=5`` -> 375-dim) term-for-term.

Verified against ``isaaclab.managers.observation_manager.ObservationManager.compute_group`` and
``isaaclab.utils.buffers.CircularBuffer``: per term, the pipeline is ``func() -> noise -> clip -> scale ->
(history append + flatten)``, and terms are concatenated in declaration order *after* each term's own history is
flattened (history is never interleaved across terms). ``func()`` is where a term's own obs-delay (if any, e.g.
``base_ang_vel``/``projected_gravity``) is applied, i.e. delay happens before noise/scale/history.
"""

from __future__ import annotations

import numpy as np

from . import constants as C
from .filters import SharedLagDelayBuffer, TermHistory


def _slot_indices(slot_names: tuple[str, ...]) -> np.ndarray:
    return np.array([C.JOINT_INDEX[n] for n in slot_names], dtype=int)


SLOT_INDICES = {
    "01": _slot_indices(C.SLOT_0_1),
    "23": _slot_indices(C.SLOT_2_3),
    "45": _slot_indices(C.SLOT_4_5),
}


def projected_gravity(quat_wxyz: np.ndarray) -> np.ndarray:
    """Body-frame projection of the world "down" vector ``[0, 0, -1]`` (Isaac/IsaacGym convention, upright ->
    ``[0, 0, -1]``), from the pelvis orientation quaternion. ``R(q)^T @ [0,0,-1]`` computed directly (no scipy dep).
    """
    w, x, y, z = quat_wxyz
    # rotation matrix (body_from_world = R(q)); we need world_from_body^T @ g_world = R(q)^T @ g_world.
    R = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )
    g_world = np.array([0.0, 0.0, -1.0])
    return R.T @ g_world


class ObservationBuilder:
    """Builds one policy-tick's flattened obs vector for ``mode in {"walking", "getup"}``."""

    def __init__(
        self,
        mode: str,
        default_qpos: np.ndarray,
        rng: np.random.Generator,
        enable_noise: bool = True,
    ):
        if mode not in ("walking", "getup"):
            raise ValueError(f"mode must be 'walking' or 'getup', got {mode!r}")
        self.mode = mode
        self.default_qpos = np.asarray(default_qpos, dtype=float)
        self.rng = rng
        self.enable_noise = enable_noise
        self.term_order = C.WALKING_OBS_TERM_ORDER if mode == "walking" else C.GETUP_OBS_TERM_ORDER
        self.history_length = 0 if mode == "walking" else C.GETUP_HISTORY_LENGTH

        self._delay = {
            name: SharedLagDelayBuffer(3, lo, hi, rng)
            for name, (lo, hi) in C.OBS_DELAY_TICKS.items()
            if name in self.term_order
        }
        term_dims = {
            "base_ang_vel": 3, "projected_gravity": 3, "command": 3,
            "joint_pos_slot01": 9, "joint_pos_slot23": 8, "joint_pos_slot45": 6,
            "joint_vel_slot01": 9, "joint_vel_slot23": 8, "joint_vel_slot45": 6,
            "actions": C.NUM_JOINTS,
        }
        self._dims = {name: term_dims[name] for name in self.term_order}
        self._history = (
            {name: TermHistory(dim, self.history_length) for name, dim in self._dims.items()}
            if self.history_length > 0
            else {}
        )
        self.obs_dim = sum(self._dims.values()) * max(self.history_length, 1)

    def reset(self) -> None:
        for buf in self._delay.values():
            buf.resample()
            buf.reset()  # lazy fill: primed on the next push_and_read with that call's raw value
        for hist in self._history.values():
            hist.reset()

    def _term_value(
        self,
        name: str,
        *,
        base_ang_vel: np.ndarray,
        proj_grav: np.ndarray,
        qpos_rel: np.ndarray,
        qvel: np.ndarray,
        command: np.ndarray | None,
        action_obs: np.ndarray,
    ) -> np.ndarray:
        if name == "base_ang_vel":
            v = self._delay["base_ang_vel"].push_and_read(base_ang_vel) if "base_ang_vel" in self._delay else base_ang_vel
        elif name == "projected_gravity":
            v = self._delay["projected_gravity"].push_and_read(proj_grav) if "projected_gravity" in self._delay else proj_grav
        elif name == "command":
            v = np.zeros(3) if command is None else command
        elif name == "joint_pos_slot01":
            v = qpos_rel[SLOT_INDICES["01"]]
        elif name == "joint_pos_slot23":
            v = qpos_rel[SLOT_INDICES["23"]]
        elif name == "joint_pos_slot45":
            v = qpos_rel[SLOT_INDICES["45"]]
        elif name == "joint_vel_slot01":
            v = qvel[SLOT_INDICES["01"]]
        elif name == "joint_vel_slot23":
            v = qvel[SLOT_INDICES["23"]]
        elif name == "joint_vel_slot45":
            v = qvel[SLOT_INDICES["45"]]
        elif name == "actions":
            v = action_obs
        else:
            raise ValueError(f"unknown obs term {name!r}")
        return np.asarray(v, dtype=float).copy()

    def build(
        self,
        base_ang_vel: np.ndarray,
        quat_wxyz: np.ndarray,
        qpos: np.ndarray,
        qvel: np.ndarray,
        action_obs: np.ndarray,
        command: np.ndarray | None = None,
    ) -> np.ndarray:
        """One policy tick. Returns the flattened, concatenated obs vector (78-dim walking / 375-dim get-up)."""
        proj_grav = projected_gravity(quat_wxyz)
        qpos_rel = qpos - self.default_qpos
        parts = []
        for name in self.term_order:
            v = self._term_value(
                name,
                base_ang_vel=base_ang_vel, proj_grav=proj_grav,
                qpos_rel=qpos_rel, qvel=qvel, command=command, action_obs=action_obs,
            )
            if self.enable_noise and name in C.OBS_NOISE_UNIFORM:
                lo, hi = C.OBS_NOISE_UNIFORM[name]
                v = v + self.rng.uniform(lo, hi, size=v.shape)
            v = v * C.OBS_SCALE.get(name, 1.0)
            if self.history_length > 0:
                v = self._history[name].append(v)
            parts.append(v)
        return np.concatenate(parts)
