# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Task-local payload and torque-capacity randomization."""

import math

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg

from .payload import combine_payload_inertia, sample_hand_payloads


class randomize_hand_payload(ManagerTermBase):
    """Resample a persistent two-hand load at reset, always from nominal inertia."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.asset = env.scene[cfg.params["asset_cfg"].name]
        self.body_ids = cfg.params["asset_cfg"].body_ids
        if isinstance(self.body_ids, slice) or len(self.body_ids) != 2:
            raise ValueError("Payload requires exactly two wrist bodies, ordered left then right.")
        view = self.asset.root_physx_view
        self.nominal_mass = view.get_masses().clone()
        self.nominal_com = view.get_coms().clone()
        # PhysX exposes full inertia tensors in the link frame, about the COM.
        self.nominal_inertia = view.get_inertias().clone()
        self.mass = torch.zeros(env.num_envs, 2, device=env.device)
        self.body_mass = self.nominal_mass.to(env.device).clone()
        self.difficulty = torch.ones(env.num_envs, device=env.device)

    def __call__(
        self, env, env_ids, asset_cfg: SceneEntityCfg,
        max_total_mass: float = 5.0,
        empty_probability: float = 0.3,
        left_fraction_range: tuple[float, float] = (0.4, 0.6),
        com_offset: float = 0.05,
        radius_range: tuple[float, float] = (0.04, 0.08),
        curriculum_steps: int = 48_000,
        initial_scale: float = 0.2,
        min_total_mass: float = 0.0,
    ):
        if curriculum_steps < 0 or not math.isfinite(initial_scale) or not 0 <= initial_scale <= 1:
            raise ValueError("Payload curriculum steps and initial scale must be nonnegative and bounded.")
        ids = torch.arange(env.num_envs, device="cpu")
        if env_ids is not None:
            ids = ids[env_ids.cpu() if isinstance(env_ids, torch.Tensor) else env_ids]
        progress = min(env.common_step_counter / curriculum_steps, 1.0) if curriculum_steps else 1.0
        difficulty = initial_scale + (1.0 - initial_scale) * progress
        payload_mass, payload_com, payload_inertia = sample_hand_payloads(
            len(ids), min_total_mass=min_total_mass, max_total_mass=max_total_mass,
            empty_probability=empty_probability, left_fraction_range=left_fraction_range, com_offset=com_offset,
            radius_range=radius_range, difficulty=difficulty,
        )
        selection = (ids[:, None], self.body_ids)
        mass, center, inertia = combine_payload_inertia(
            self.nominal_mass[selection], self.nominal_com[selection][..., :3],
            self.nominal_inertia[selection].reshape(-1, 2, 3, 3),
            payload_mass, payload_com, payload_inertia,
        )
        view = self.asset.root_physx_view
        masses, coms, inertias = view.get_masses().clone(), view.get_coms().clone(), view.get_inertias().clone()
        masses[selection] = mass
        coms[selection] = self.nominal_com[selection]
        coms[ids[:, None], self.body_ids, :3] = center
        inertias[selection] = inertia.reshape(-1, 2, 9)
        view.set_masses(masses, ids)
        view.set_coms(coms, ids)
        # Must follow set_coms: PhysX diagonalizes the tensor and sets its principal axes.
        view.set_inertias(inertias, ids)
        self.mass[ids.to(env.device)] = payload_mass.to(env.device)
        self.body_mass[ids.to(env.device)] = masses[ids].to(env.device)
        self.difficulty[ids.to(env.device)] = difficulty


class randomize_torque_capacity(ManagerTermBase):
    """Derate explicit PD actuator torque limits without exceeding nominal hardware limits."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.asset = env.scene[cfg.params["asset_cfg"].name]
        self.nominal = {name: actuator.effort_limit.clone() for name, actuator in self.asset.actuators.items()}

    def __call__(self, env, env_ids, asset_cfg: SceneEntityCfg, scale_range: tuple[float, float] = (0.8, 1.0)):
        if not all(math.isfinite(value) for value in scale_range) or not 0 < scale_range[0] <= scale_range[1] <= 1:
            raise ValueError("Torque-capacity factors must be ordered in (0, 1].")
        ids = torch.arange(env.num_envs, device=env.device)
        if env_ids is not None:
            ids = ids[env_ids]
        for name, actuator in self.asset.actuators.items():
            scale = torch.empty(len(ids), 1, device=env.device).uniform_(*scale_range)
            actuator.effort_limit[ids] = self.nominal[name][ids] * scale
