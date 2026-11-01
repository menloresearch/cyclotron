# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, SceneEntityCfg

from .upper_body_motion import UpperBodyMotionSampler

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def randomize_joint_default_pos(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    ranges: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    action_name: str | None = "joint_pos",
):

    asset: Articulation = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    if not hasattr(asset.data, "nominal_default_joint_pos"):
        asset.data.nominal_default_joint_pos = asset.data.default_joint_pos.clone()

    joint_ids = asset_cfg.joint_ids
    if isinstance(joint_ids, slice):
        num_joints = asset.num_joints
        joint_ids = torch.arange(num_joints, device=asset.device)
    else:
        joint_ids = torch.tensor(joint_ids, device=asset.device)

    offsets = torch.empty(
        (len(env_ids), len(joint_ids)), device=asset.device
    ).uniform_(ranges[0], ranges[1])
    asset.data.default_joint_pos[env_ids[:, None], joint_ids] += offsets

    if action_name is not None and hasattr(env, "action_manager"):
        try:
            term = env.action_manager.get_term(action_name)
        except (KeyError, ValueError):
            return
        offset = getattr(term, "_offset", None)
        if isinstance(offset, torch.Tensor):
            term._offset = asset.data.default_joint_pos[:, term._joint_ids].clone()


def randomize_joint_position_targets(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    position_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Sample PD position targets around each joint default pose.

    The target is held in the articulation target buffer until the next sample,
    so the existing actuator PD loop tracks it at every physics step.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)
    elif isinstance(env_ids, slice):
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)[env_ids]
    else:
        env_ids = torch.as_tensor(env_ids, device=asset.device, dtype=torch.long)

    joint_ids = asset_cfg.joint_ids
    if isinstance(joint_ids, slice):
        joint_ids = torch.arange(asset.num_joints, device=asset.device)[joint_ids]
    else:
        joint_ids = torch.as_tensor(joint_ids, device=asset.device, dtype=torch.long)

    target = asset.data.default_joint_pos[env_ids[:, None], joint_ids].clone()
    target += torch.empty_like(target).uniform_(position_range[0], position_range[1])
    limits = asset.data.soft_joint_pos_limits[env_ids[:, None], joint_ids]
    target.clamp_(limits[..., 0], limits[..., 1])
    asset.set_joint_position_target(target, joint_ids=joint_ids, env_ids=env_ids)


class randomize_upper_body_motion(ManagerTermBase):
    """Advance coordinated targets every control step; reset only selected environments."""

    def __init__(self, cfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        asset_cfg = cfg.params["asset_cfg"]
        self.asset = env.scene[asset_cfg.name]
        self.joint_ids = torch.arange(self.asset.num_joints, device=self.asset.device)[asset_cfg.joint_ids]
        names = [self.asset.joint_names[i] for i in self.joint_ids.tolist()]
        self.sampler = UpperBodyMotionSampler(
            names,
            self.asset.data.default_joint_pos[:, self.joint_ids],
            self.asset.data.soft_joint_pos_limits[:, self.joint_ids],
            **cfg.params["motion_config"],
        )
        carrying_config = cfg.params.get("carrying_motion_config")
        self.carry_sampler = (
            UpperBodyMotionSampler(
                names,
                self.asset.data.default_joint_pos[:, self.joint_ids],
                self.asset.data.soft_joint_pos_limits[:, self.joint_ids],
                **carrying_config,
            )
            if carrying_config is not None else None
        )
        self.reset()

    def reset(self, env_ids=None):
        ids = self.sampler._ids(env_ids)
        current = self.asset.data.joint_pos[ids[:, None], self.joint_ids]
        defaults = self.asset.data.default_joint_pos[ids[:, None], self.joint_ids]
        self.sampler.reset(ids, current, defaults)
        if self.carry_sampler is not None:
            self.carry_sampler.reset(ids, current, defaults)
        self.asset.set_joint_position_target(current, joint_ids=self.joint_ids, env_ids=ids)

    def __call__(
        self, env: ManagerBasedEnv, env_ids, asset_cfg: SceneEntityCfg, motion_config: dict,
        carrying_motion_config: dict | None = None, payload_event_name: str = "hand_payload",
    ):
        ids = self.sampler._ids(env_ids)
        if self.carry_sampler is None:
            targets = self.sampler.advance(env.step_dt, ids)
            self.asset.set_joint_position_target(targets, joint_ids=self.joint_ids, env_ids=ids)
            return
        payload = env.event_manager.get_term_cfg(payload_event_name).func
        loaded = payload.mass[ids].sum(dim=-1) > 0
        for sampler, selected in ((self.sampler, ids[~loaded]), (self.carry_sampler, ids[loaded])):
            if len(selected):
                sampler.goal_scale[selected] = payload.difficulty[selected]
                targets = sampler.advance(env.step_dt, selected)
                self.asset.set_joint_position_target(targets, joint_ids=self.joint_ids, env_ids=selected)
        env.extras.setdefault("log", {}).update({
            "Metrics/payload_kg": payload.mass.sum(dim=-1).mean(),
            "Metrics/loaded_fraction": (payload.mass.sum(dim=-1) > 0).float().mean(),
            "Metrics/payload_difficulty": payload.difficulty.mean(),
            "Metrics/torque_saturation_fraction": (
                self.asset.data.computed_torque.abs() > self.asset.data.applied_torque.abs() + 1e-4
            ).float().mean(),
        })
