# Copyright (c) 2026, Menlo Research.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Limp (passive) actuator control for the get-up task.

* :func:`set_limp` switches envs between limp (Kp = 0, Kd = 0.5 N·m·s/rad absolute) and the nominal PD gains, through
  the ``DelayedPDLimpableActuator.gain_scale`` (0 = limp, 1 = nominal; reset to 1 by ``scene.reset``).
* :func:`update_limp_schedule` releases the limp phase once ``getup_state.step >= control_start_step``. The action term
  calls it once per env step (in ``process_actions``) and may use the returned mask as ``policy_active``.
* :func:`is_limp` returns the per-env limp flag.

Fallback for assets without limpable actuators (the walking asset, early development only): the actuator's per-env
``stiffness`` / ``damping`` tensors are overwritten and restored from a copy taken on first use. This fallback does not
compose with ``randomize_actuator_gains``; use ``ASIMOV_1_GETUP_CFG``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

__all__ = ["LIMP_KP", "LIMP_KD", "is_limp", "set_limp", "set_limp_mask", "update_limp_schedule"]

LIMP_KP: float = 0.0
"""Stiffness while limp [N·m/rad]."""

LIMP_KD: float = 0.5
"""Damping while limp [N·m·s/rad] (absolute, not a scale). Matches ``DelayedPDLimpableActuatorCfg.limp_damping``."""


def _resolve_ids(env, env_ids) -> torch.Tensor:
    if env_ids is None:
        return torch.arange(env.num_envs, device=env.device)
    if isinstance(env_ids, torch.Tensor):
        return env_ids.to(device=env.device, dtype=torch.long).flatten()
    return torch.as_tensor(list(env_ids), device=env.device, dtype=torch.long)


def _limp_mask(env) -> torch.Tensor:
    mask = env.__dict__.get("_getup_limp_mask")
    if mask is None:
        mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        env.__dict__["_getup_limp_mask"] = mask
    return mask


def is_limp(env: ManagerBasedEnv) -> torch.Tensor:
    """[num_envs] bool: env is currently limp (set by :func:`set_limp`)."""
    return _limp_mask(env)


def set_limp(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    limp: bool,
    asset_name: str = "robot",
) -> None:
    """Switch the actuators of ``env_ids`` (None = all) to limp (Kp 0, Kd 0.5) or back to nominal gains.

    Takes effect at the next actuator ``compute`` (next physics substep). Must be called after ``scene.reset`` (which
    resets ``gain_scale`` to 1), e.g. from a reset event. No GPU->host sync.
    """
    ids = _resolve_ids(env, env_ids)
    set_limp_mask(env, ids, torch.full(ids.shape, bool(limp), dtype=torch.bool, device=env.device), asset_name)


def set_limp_mask(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    limp: torch.Tensor,
    asset_name: str = "robot",
) -> None:
    """Per-env version of :func:`set_limp`: ``limp`` is a bool tensor aligned with ``env_ids`` (None = all envs).

    Envs with ``limp`` True go limp, the others get nominal gains. No GPU->host sync.
    """
    ids = _resolve_ids(env, env_ids)
    if ids.shape[0] == 0:
        return
    limp = limp.to(device=env.device, dtype=torch.bool)
    robot = env.scene[asset_name]
    for act in robot.actuators.values():
        scale = getattr(act, "gain_scale", None)
        if isinstance(scale, torch.Tensor):
            scale[ids] = (~limp).to(scale.dtype)
            continue
        # fallback: plain explicit PD actuator (walking asset); stash nominal gains once
        nominal = getattr(act, "_getup_nominal_gains", None)
        if nominal is None:
            nominal = (act.stiffness.clone(), act.damping.clone())
            act._getup_nominal_gains = nominal
        lm = limp.unsqueeze(1)
        act.stiffness[ids] = torch.where(lm, torch.full_like(nominal[0][ids], LIMP_KP), nominal[0][ids])
        act.damping[ids] = torch.where(lm, torch.full_like(nominal[1][ids], LIMP_KD), nominal[1][ids])
    _limp_mask(env)[ids] = limp


def update_limp_schedule(env: ManagerBasedEnv, asset_name: str = "robot") -> torch.Tensor:
    """Release limp envs whose limp window has ended; return ``active = step >= control_start_step`` [num_envs].

    Uses ``env.getup_state.step`` (env steps since reset, maintained by the action term) and ``control_start_step``
    (written by :func:`~.resets.reset_fallen_state`). Call once per env step before the actions are applied, with ``step`` still
    counting the steps completed since the reset (0 on the first step after a reset). No GPU->host sync.
    """
    st = env.getup_state
    active = st.step >= st.control_start_step
    mask = _limp_mask(env)
    release = active & mask
    robot = env.scene[asset_name]
    for act in robot.actuators.values():
        scale = getattr(act, "gain_scale", None)
        if isinstance(scale, torch.Tensor):
            scale.copy_(torch.where(release, torch.ones_like(scale), scale))
        else:
            nominal = getattr(act, "_getup_nominal_gains", None)
            if nominal is not None:
                r = release.unsqueeze(1)
                act.stiffness.copy_(torch.where(r, nominal[0], act.stiffness))
                act.damping.copy_(torch.where(r, nominal[1], act.damping))
    mask &= ~release
    return active
