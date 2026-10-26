# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""CPU tests for the 27-DOF lower-body task's upper-body motion sampling; run without launching Isaac Sim."""

import ast
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

CONFIG_DIR = Path(__file__).resolve().parents[1] / "source/cyclotron/cyclotron/tasks/locomotion"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


motion_module = load_module("upper_body_motion", CONFIG_DIR / "mdp/upper_body_motion.py")
Sampler = motion_module.UpperBodyMotionSampler
# Load the literal task settings without importing Isaac Sim / registering tasks.
config_ast = ast.parse((CONFIG_DIR / "lower_body_27dof_amp_env_cfg.py").read_text())
config_node = next(
    node.value
    for node in config_ast.body
    if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "_UPPER_BODY_MOTION_CONFIG" for t in node.targets)
)
MOTION_CONFIG = eval(
    compile(ast.Expression(config_node), "lower_body_27dof_amp_env_cfg.py", "eval"), {"ASIMOV_1_ACTION_SCALE": 0.25}
)
NAMES = ["waist_yaw_joint"] + [
    f"{side}_{joint}_joint"
    for side in ("right", "left")
    for joint in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_yaw")
] + ["neck_yaw_joint", "neck_pitch_joint", "right_6DOF_joint", "left_6dof_joint"]
DEFAULTS = [0, 0.25, 0.05, 0, -0.4, 0, -0.25, -0.05, 0, 0.4, 0, 0, 0, 0, 0]
# asimov_1_27dof.urdf joint ranges, contracted to the configured 90% soft limits.
HARD_LIMITS = torch.tensor(
    [
        [-1.5708, 1.5708],  # waist_yaw_joint
        [-0.87267, 3.1416],  # right_shoulder_pitch_joint
        [0, 1.5708],  # right_shoulder_roll_joint
        [-1.5708, 1.5708],  # right_shoulder_yaw_joint
        [-1.48357, 0.61083],  # right_elbow_joint
        [0, 3.1416],  # right_wrist_yaw_joint
        [-3.1416, 0.87267],  # left_shoulder_pitch_joint
        [-1.5708, 0],  # left_shoulder_roll_joint
        [-1.5708, 1.5708],  # left_shoulder_yaw_joint
        [-0.61083, 1.4836],  # left_elbow_joint
        [-3.1416, 0],  # left_wrist_yaw_joint
        [-1.57, 1.57],  # neck_yaw_joint
        [-0.34, 0.4],  # neck_pitch_joint
        [-1.5, 1.5],  # right_6DOF_joint
        [-1.5, 1.5],  # left_6dof_joint
    ]
)
MID = HARD_LIMITS.mean(dim=1, keepdim=True)
SOFT_LIMITS = MID + 0.9 * (HARD_LIMITS - MID)


def make_sampler(count=64, **overrides):
    config = dict(MOTION_CONFIG, **overrides)
    defaults = torch.tensor(DEFAULTS).repeat(count, 1)
    limits = SOFT_LIMITS.repeat(count, 1, 1)
    return Sampler(NAMES, defaults, limits, **config)


class TestUpperBodyMotion(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(1)

    def test_broad_coverage_and_limit_margin(self):
        sampler = make_sampler(8192)
        sampler.advance(0.02)
        self.assertTrue(torch.all(sampler.goal >= sampler.lower))
        self.assertTrue(torch.all(sampler.goal <= sampler.upper))
        covered = sampler.goal.max(dim=0).values - sampler.goal.min(dim=0).values
        self.assertTrue(torch.all(covered > 0.98 * (sampler.upper[0] - sampler.lower[0])))
        self.assertTrue(torch.all(sampler.goal[:, 0].abs() <= 0.5))
        # The problematic raised/bent/inward pose must receive substantial mass,
        # not just have a tiny chance of occurring in a 15-dimensional box.
        raised = (
            (sampler.goal[:, 1] > 1.2)
            & (sampler.goal[:, 3] < -1.2)
            & (sampler.goal[:, 4] < -1.3)
            & (sampler.goal[:, 6] < -1.2)
            & (sampler.goal[:, 8] > 1.2)
            & (sampler.goal[:, 9] > 1.3)
        )
        self.assertGreater(int(raised.sum()), 200)

    def test_mirrored_poses_and_joint_order(self):
        sampler = make_sampler(sampling_weights=(0, 1, 0, 0))
        sampler.advance(0.02)
        torch.testing.assert_close(sampler.goal[:, 1:6], -sampler.goal[:, 6:11])
        # The 6DOF wrists pair despite their differently cased names.
        torch.testing.assert_close(sampler.goal[:, 13], -sampler.goal[:, 14])
        # Resolve names rather than assuming actuator order.
        order = torch.tensor([8, 0, 12, 4, 14, 2, 7, 10, 1, 5, 13, 11, 3, 9, 6])
        shuffled = Sampler(
            [NAMES[i] for i in order],
            sampler.default_pos[:, order],
            SOFT_LIMITS[order].repeat(64, 1, 1),
            **dict(MOTION_CONFIG, sampling_weights=(0, 1, 0, 0)),
        )
        shuffled.advance(0.02)
        self.assertEqual(len(shuffled.right_ids), 6)
        for right, left in zip(shuffled.right_ids, shuffled.left_ids):
            torch.testing.assert_close(shuffled.goal[:, right], -shuffled.goal[:, left])

    def test_speed_bound_hold_and_resume(self):
        sampler = make_sampler(16, hold_time_range_s=(1, 1), transition_time_range_s=(1, 1))
        sampler.advance(0.02)
        original_goal = sampler.goal.clone()
        previous = sampler.target.clone()
        held_steps = torch.zeros(16, dtype=torch.long)
        for _ in range(700):
            target = sampler.advance(0.02)
            self.assertLessEqual(float((target - previous).abs().max()), 2.0 * 0.02 + 1e-5)
            holding = (sampler.elapsed >= sampler.duration) & (sampler.elapsed < sampler.duration + sampler.hold)
            torch.testing.assert_close(target[holding], sampler.goal[holding], atol=1e-6, rtol=0)
            held_steps += holding
            previous = target.clone()
        self.assertTrue(torch.all(held_steps >= 49))
        self.assertFalse(torch.equal(sampler.goal, original_goal))

    def test_partial_reset_and_partial_update(self):
        sampler = make_sampler(4)
        for _ in range(100):
            sampler.advance(0.02)
        before = sampler.target.clone()
        elapsed = sampler.elapsed.clone()
        ids = torch.tensor([1, 3])
        current = torch.tensor(DEFAULTS).repeat(2, 1) + 0.01
        defaults = torch.tensor(DEFAULTS).repeat(2, 1) + 0.02
        sampler.reset(ids, current, defaults)
        torch.testing.assert_close(sampler.target[ids], current)
        torch.testing.assert_close(sampler.default_pos[ids], defaults)
        sampler.advance(0.02, ids)
        torch.testing.assert_close(sampler.target[[0, 2]], before[[0, 2]])
        torch.testing.assert_close(sampler.elapsed[[0, 2]], elapsed[[0, 2]])
        sampler.advance(0.02, [])  # no environments due is valid
        sampler.reset(slice(0, 1), current[:1], defaults[:1])
        torch.testing.assert_close(sampler.target[:1], current[:1])

    def test_acceleration_bound_and_goal_scale(self):
        sampler = make_sampler(16, max_joint_speed=0.75, max_joint_acceleration=1.5)
        sampler.goal_scale[:] = 0.25
        sampler.advance(0.02)
        lo = sampler.default_pos + 0.25 * (sampler.lower - sampler.default_pos)
        hi = sampler.default_pos + 0.25 * (sampler.upper - sampler.default_pos)
        # Defaults may sit outside inset soft limits, so account for final clamping.
        assert torch.all(sampler.goal >= lo.clamp(sampler.lower, sampler.upper) - 1e-6)
        assert torch.all(sampler.goal <= hi.clamp(sampler.lower, sampler.upper) + 1e-6)
        previous = sampler.target.clone()
        velocity = torch.zeros_like(previous)
        for _ in range(500):
            target = sampler.advance(0.02)
            new_velocity = (target - previous) / 0.02
            assert new_velocity.abs().max() <= 0.75 + 1e-4
            assert ((new_velocity - velocity) / 0.02).abs().max() <= 1.5 + 0.02
            previous, velocity = target.clone(), new_velocity

    def test_presets_hold_unlisted_joints_near_default(self):
        sampler = make_sampler(4096, sampling_weights=(0, 0, 1, 0))
        sampler.advance(0.02)
        unlisted = [NAMES.index(n) for n in ("neck_yaw_joint", "right_6DOF_joint", "left_6dof_joint")]
        self.assertTrue(torch.all(sampler.goal[:, unlisted].abs() <= MOTION_CONFIG["pose_jitter"] + 1e-6))

    def test_invalid_configuration(self):
        for overrides in (
            {"max_joint_speed": 0},
            {"limit_margin": 10},
            {"hold_time_range_s": (2, 1)},
            {"sampling_weights": (1, 1, 0, 0)},
            {"pose_presets": ()},
            {"pose_presets": ({"missing_joint": 0},)},
        ):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                make_sampler(**overrides)

    def test_event_writes_only_upper_joints_and_resets_selected_rows(self):
        # Isolate only Isaac's import-time runtime dependency. Exercise the real
        # event adapter against an articulation with deliberately shuffled joints.
        class ManagerTermBase:
            def __init__(self, cfg, env):
                self.cfg, self._env = cfg, env

        class SceneEntityCfg:
            def __init__(self, name):
                self.name = name

        assets = types.ModuleType("isaaclab.assets")
        assets.Articulation = object
        managers = types.ModuleType("isaaclab.managers")
        managers.ManagerTermBase, managers.SceneEntityCfg = ManagerTermBase, SceneEntityCfg
        with patch.dict(
            sys.modules,
            {
                "isaaclab.assets": assets,
                "isaaclab.managers": managers,
                "sampler_test.upper_body_motion": motion_module,
            },
        ):
            events = load_module("sampler_test.events", CONFIG_DIR / "mdp/events.py")
        num_joints = 2 * len(NAMES) + 1
        ids = list(range(1, num_joints, 2))
        data = types.SimpleNamespace(
            default_joint_pos=torch.zeros(3, num_joints),
            joint_pos=torch.zeros(3, num_joints),
            soft_joint_pos_limits=torch.zeros(3, num_joints, 2),
        )
        data.default_joint_pos[:, ids] = torch.tensor(DEFAULTS)
        data.joint_pos.copy_(data.default_joint_pos)
        data.soft_joint_pos_limits[:, ids] = SOFT_LIMITS
        targets = torch.full((3, num_joints), 7.0)

        def set_targets(value, joint_ids, env_ids):
            rows = torch.arange(3) if env_ids is None else env_ids
            targets[rows[:, None], joint_ids] = value

        joint_names = ["leg"] * num_joints
        for i, name in zip(ids, NAMES):
            joint_names[i] = name
        asset = types.SimpleNamespace(
            device="cpu",
            num_joints=num_joints,
            joint_names=joint_names,
            data=data,
            set_joint_position_target=set_targets,
        )
        cfg = types.SimpleNamespace(
            params={
                "asset_cfg": types.SimpleNamespace(name="robot", joint_ids=ids),
                "motion_config": MOTION_CONFIG,
            }
        )
        env = types.SimpleNamespace(scene={"robot": asset}, step_dt=0.02)
        event = events.randomize_upper_body_motion(cfg, env)
        for _ in range(100):
            event(env, None, **cfg.params)
        self.assertTrue(torch.all(targets[:, ::2] == 7))
        before = targets.clone()
        data.joint_pos[1, ids] += 0.03
        event.reset(torch.tensor([1]))
        torch.testing.assert_close(targets[1, ids], data.joint_pos[1, ids])
        torch.testing.assert_close(targets[[0, 2]], before[[0, 2]])


if __name__ == "__main__":
    unittest.main()
