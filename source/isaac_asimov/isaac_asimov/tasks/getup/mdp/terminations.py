# Copyright (c) 2026, Menlo Research.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# `no_height_progress` follows NVIDIA WBC-AGILE `agile/rl_env/mdp/terminations.py::no_height_progress` (Apache-2.0)
# in intent; rewritten to use the shared get-up state and time since control start.
"""Get-up terminations: time-out (from Isaac Lab), no-height-progress, non-finite state.

Falling after standing is **not** terminated. Declare the ``getup_tracker`` term before ``no_height_progress`` so
``made_progress`` is current.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

from .state import ensure_state

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def no_height_progress(env: ManagerBasedRLEnv, time_limit_s: float = 8.0) -> torch.Tensor:
    """Terminate when, ``time_limit_s`` after control start, the pelvis never rose ``progress_height`` (tracker param,
    0.2 m) above its control-start height and the episode has not succeeded."""
    st = ensure_state(env)
    t_ctrl = (st.step - st.control_start_step).float() * env.step_dt
    return st.policy_active & (t_ctrl > time_limit_s) & ~st.made_progress


def root_state_invalid(
    env: ManagerBasedRLEnv,
    max_lin_vel: float = 20.0,
    max_height: float = 5.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Safety termination: non-finite root/joint state, |v| > ``max_lin_vel`` or |z| > ``max_height`` (sim blew up)."""
    asset: Articulation = env.scene[asset_cfg.name]
    root = asset.data.root_state_w
    bad = ~torch.isfinite(root).all(dim=1)
    bad |= ~torch.isfinite(asset.data.joint_pos).all(dim=1) | ~torch.isfinite(asset.data.joint_vel).all(dim=1)
    bad |= torch.nan_to_num(root[:, 7:10], nan=0.0).norm(dim=1) > max_lin_vel
    bad |= torch.abs(torch.nan_to_num(root[:, 2] - env.scene.env_origins[:, 2], nan=0.0)) > max_height
    return bad
