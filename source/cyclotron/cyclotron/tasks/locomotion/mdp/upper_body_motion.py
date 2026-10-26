# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Batched upper-body target trajectories, independent of the simulator runtime."""

from __future__ import annotations

import math

import torch


class UpperBodyMotionSampler:
    """Mix broad arm configurations with coordinated poses and sustained holds.

    Targets use the articulation's soft limits, inset by ``limit_margin`` radians.
    These are joint-space samples, not collision-free Cartesian plans. A quintic
    blend gives zero target velocity and acceleration at either end of each move.
    Each environment has its own move/hold clock; resetting one never changes another.
    """

    def __init__(
        self,
        joint_names: list[str],
        default_pos: torch.Tensor,
        soft_limits: torch.Tensor,
        *,
        sampling_weights: tuple[float, float, float, float] = (0.5, 0.2, 0.2, 0.1),
        pose_presets: tuple[dict[str, float], ...] = (),
        transition_time_range_s: tuple[float, float] = (1.0, 3.0),
        hold_time_range_s: tuple[float, float] = (1.0, 6.0),
        max_joint_speed: float = 2.0,
        max_joint_acceleration: float | None = None,
        limit_margin: float = 0.02,
        waist_yaw_range: tuple[float, float] = (-0.5, 0.5),
        pose_jitter: float = 0.1,
        neutral_range: float = 0.25,
    ):
        self.device = default_pos.device
        self.joint_names = list(joint_names)
        self.default_pos = default_pos.clone()
        self.num_envs, self.num_joints = default_pos.shape
        if len(joint_names) != self.num_joints or soft_limits.shape != (*default_pos.shape, 2):
            raise ValueError("Joint names, defaults, and soft limits must have matching shapes.")
        for name, bounds in (
            ("transition_time_range_s", transition_time_range_s),
            ("hold_time_range_s", hold_time_range_s),
        ):
            if len(bounds) != 2 or not all(math.isfinite(x) for x in bounds) or not 0 < bounds[0] <= bounds[1]:
                raise ValueError(f"{name} must be an ordered, positive range.")
        if not math.isfinite(max_joint_speed) or max_joint_speed <= 0:
            raise ValueError("max_joint_speed must be finite and positive.")
        if max_joint_acceleration is not None and (
            not math.isfinite(max_joint_acceleration) or max_joint_acceleration <= 0
        ):
            raise ValueError("max_joint_acceleration must be finite and positive.")
        if any(not math.isfinite(x) or x < 0 for x in (limit_margin, pose_jitter, neutral_range)):
            raise ValueError("Margins and jitter ranges must be finite and nonnegative.")
        if len(sampling_weights) != 4 or any(not math.isfinite(x) or x < 0 for x in sampling_weights):
            raise ValueError("Expected four nonnegative weights: independent, mirrored, preset, neutral.")
        if not math.isclose(sum(sampling_weights), 1.0):
            raise ValueError("Sampling weights must sum to one.")
        if sampling_weights[2] > 0 and not pose_presets:
            raise ValueError("Preset sampling requires at least one pose.")
        if (
            len(waist_yaw_range) != 2
            or not all(math.isfinite(x) for x in waist_yaw_range)
            or waist_yaw_range[0] > waist_yaw_range[1]
        ):
            raise ValueError("waist_yaw_range must be finite and ordered.")

        self.lower = soft_limits[..., 0].clone() + limit_margin
        self.upper = soft_limits[..., 1].clone() - limit_margin
        if "waist_yaw_joint" in joint_names:
            waist = joint_names.index("waist_yaw_joint")
            self.lower[:, waist].clamp_(min=waist_yaw_range[0])
            self.upper[:, waist].clamp_(max=waist_yaw_range[1])
        if (
            not torch.isfinite(self.lower).all()
            or not torch.isfinite(self.upper).all()
            or (self.lower > self.upper).any()
        ):
            raise ValueError("Soft limits, margin, and waist range must leave a finite sampling interval.")

        self.right_ids, self.left_ids = [], []
        # Case-insensitive: the 27-DOF model names its sixth arm joints
        # right_6DOF_joint and left_6dof_joint.
        lowered = [name.lower() for name in joint_names]
        for right, name in enumerate(lowered):
            if name.startswith("right_"):
                left_name = "left_" + name.removeprefix("right_")
                if left_name in lowered:
                    self.right_ids.append(right)
                    self.left_ids.append(lowered.index(left_name))
        if sampling_weights[1] > 0 and not self.right_ids:
            raise ValueError("Mirrored sampling requires matching right/left joint names.")

        self.presets = torch.full((len(pose_presets), self.num_joints), float("nan"), device=self.device)
        for row, pose in enumerate(pose_presets):
            for name, value in pose.items():
                if name not in joint_names or not math.isfinite(value):
                    raise ValueError(f"Invalid preset joint or angle: {name}={value}.")
                self.presets[row, joint_names.index(name)] = value
        self.weights = sampling_weights
        self.transition_range = transition_time_range_s
        self.hold_range = hold_time_range_s
        self.max_joint_speed = max_joint_speed
        self.max_joint_acceleration = max_joint_acceleration
        self.pose_jitter = pose_jitter
        self.neutral_range = neutral_range
        self.start = default_pos.clone()
        self.goal = default_pos.clone()
        self.target = default_pos.clone()
        self.elapsed = torch.zeros(self.num_envs, device=self.device)
        self.duration = torch.ones(self.num_envs, device=self.device)
        self.hold = torch.zeros(self.num_envs, device=self.device)
        self.needs_sample = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.goal_scale = torch.ones(self.num_envs, device=self.device)
        self.all_env_ids = torch.arange(self.num_envs, device=self.device)

    def _ids(self, env_ids):
        if env_ids is None:
            return self.all_env_ids
        if isinstance(env_ids, slice):
            return self.all_env_ids[env_ids]
        return torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

    def reset(self, env_ids, current_pos: torch.Tensor, default_pos: torch.Tensor):
        """Start from the actual reset pose, refreshing any randomized defaults."""
        ids = self._ids(env_ids)
        self.start[ids] = current_pos
        self.goal[ids] = current_pos
        self.target[ids] = current_pos
        self.default_pos[ids] = default_pos
        self.elapsed[ids] = 0
        self.needs_sample[ids] = True

    def _uniform(self, count, bounds):
        return torch.rand(count, device=self.device) * (bounds[1] - bounds[0]) + bounds[0]

    def _resample(self, ids):
        lo, hi = self.lower[ids], self.upper[ids]
        goal = lo + torch.rand_like(lo) * (hi - lo)
        mode = torch.rand(len(ids), device=self.device)
        independent, mirrored, preset, _ = self.weights
        mirror_mask = (mode >= independent) & (mode < independent + mirrored)
        # Asimov's paired arm joint coordinates have opposite signs. Restrict
        # mirrored draws to the intersection if per-joint limits differ.
        for right, left in zip(self.right_ids, self.left_ids):
            mirror_lo = torch.maximum(lo[:, right], -hi[:, left])
            mirror_hi = torch.minimum(hi[:, right], -lo[:, left])
            angle = mirror_lo + torch.rand(len(ids), device=self.device) * (mirror_hi - mirror_lo).clamp_min(0)
            goal[:, right] = torch.where(mirror_mask, angle, goal[:, right])
            goal[:, left] = torch.where(mirror_mask, -angle, goal[:, left])
        preset_mask = (mode >= independent + mirrored) & (mode < independent + mirrored + preset)
        if len(self.presets):
            poses = self.presets[torch.randint(len(self.presets), (len(ids),), device=self.device)]
            poses = torch.where(torch.isnan(poses), self.default_pos[ids], poses)
            poses += (2 * torch.rand_like(poses) - 1) * self.pose_jitter
            goal = torch.where(preset_mask[:, None], poses, goal)
        neutral_mask = mode >= independent + mirrored + preset
        neutral = self.default_pos[ids] + (2 * torch.rand_like(goal) - 1) * self.neutral_range
        goal = torch.where(neutral_mask[:, None], neutral, goal)
        goal = torch.lerp(self.default_pos[ids], goal.clamp(lo, hi), self.goal_scale[ids, None])
        self.start[ids] = self.target[ids]
        self.goal[ids] = goal.clamp(lo, hi)
        # max d/du (6u^5 - 15u^4 + 10u^3) = 1.875; lengthen large moves
        # so even a full-range wrist move respects the target speed bound.
        distance = (self.goal[ids] - self.start[ids]).abs().amax(dim=1)
        min_duration = 1.875 * distance / self.max_joint_speed
        if self.max_joint_acceleration is not None:
            # max |d²/du² (6u^5 - 15u^4 + 10u^3)| = 10 / sqrt(3).
            min_duration = torch.maximum(
                min_duration, torch.sqrt((10.0 / math.sqrt(3.0)) * distance / self.max_joint_acceleration)
            )
        self.duration[ids] = torch.maximum(self._uniform(len(ids), self.transition_range), min_duration)
        self.hold[ids] = self._uniform(len(ids), self.hold_range)
        self.elapsed[ids] = 0
        self.needs_sample[ids] = False

    def advance(self, dt: float, env_ids=None):
        """Advance once per policy step and return targets for the selected rows."""
        ids = self._ids(env_ids)
        due = self.needs_sample[ids] | (self.elapsed[ids] >= self.duration[ids] + self.hold[ids])
        self._resample(ids[due])
        self.elapsed[ids] += dt
        phase = (self.elapsed[ids] / self.duration[ids]).clamp(0, 1)
        blend = phase**3 * (10 + phase * (-15 + 6 * phase))
        self.target[ids] = torch.lerp(self.start[ids], self.goal[ids], blend[:, None])
        return self.target[ids]
