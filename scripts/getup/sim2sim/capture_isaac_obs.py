#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Capture a short, deterministic Isaac rollout's raw ingredients + ground-truth obs vector, for
``compare_isaac.py`` to check against this package's own (isaaclab-free) obs pipeline.

Needs Isaac Sim / Isaac Lab -- run via
``python scripts/getup/sim2sim/capture_isaac_obs.py --headless``.
Never imported by the rest of this package (which is isaaclab-free by design); only this one file needs Isaac Lab.

Uses ``Asimov1-Velocity-AMP-Play-v0`` (``enable_corruption=False`` -- no obs noise, no domain randomization, no
push events -- see ``Asimov1VelocityEnvCfg_PLAY``), 1 env, a fixed small-sinusoid action sequence (not the trained
policy: this checks the *obs pipeline*, not the policy), and dumps every raw ingredient the walking obs group reads
(joint pos/vel, base ang vel, projected gravity, command, the action) plus Isaac's own resulting 78-dim obs vector,
per tick, to an ``.npz`` file that needs no ``isaaclab``/``torch`` to read back.
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num-steps", type=int, default=20)
parser.add_argument("--out", type=str, default="getup_isaac_obs_capture.npz")
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch

import gymnasium as gym
import isaaclab.envs.mdp as base_mdp

import isaac_asimov.tasks  # noqa: F401 (gym registration)
from isaac_asimov.tasks.locomotion.amp_env_cfg import Asimov1AmpEnvCfg_PLAY


def main() -> None:
    env_cfg = Asimov1AmpEnvCfg_PLAY()
    env_cfg.scene.num_envs = 1
    env = gym.make("Asimov1-Velocity-AMP-Play-v0", cfg=env_cfg)
    obs_dict, _ = env.reset(seed=args_cli.seed)
    robot = env.unwrapped.scene["robot"]

    T = args_cli.num_steps
    action_dim = robot.num_joints
    log = {
        "obs_isaac": [], "qpos": [], "qvel": [], "default_qpos": [],
        "base_ang_vel_raw": [], "proj_grav_raw": [], "command": [], "action": [],
    }
    for t in range(T):
        action = 0.1 * torch.sin(torch.full((1, action_dim), float(t) * 0.3)) + 0.0
        obs_out, _, _, _, _ = env.step(action)
        obs_policy = obs_out["policy"] if isinstance(obs_out, dict) else obs_out
        log["obs_isaac"].append(obs_policy[0].detach().cpu().numpy().copy())
        log["qpos"].append(robot.data.joint_pos[0].detach().cpu().numpy().copy())
        log["qvel"].append(robot.data.joint_vel[0].detach().cpu().numpy().copy())
        log["default_qpos"].append(robot.data.default_joint_pos[0].detach().cpu().numpy().copy())
        log["base_ang_vel_raw"].append(base_mdp.base_ang_vel(env.unwrapped)[0].detach().cpu().numpy().copy())
        log["proj_grav_raw"].append(base_mdp.projected_gravity(env.unwrapped)[0].detach().cpu().numpy().copy())
        cmd = env.unwrapped.command_manager.get_command("twist")
        log["command"].append(cmd[0].detach().cpu().numpy().copy())
        log["action"].append(action[0].detach().cpu().numpy().copy())

    out = {k: np.stack(v) for k, v in log.items()}
    out["joint_names"] = np.array(list(robot.joint_names))
    np.savez(args_cli.out, **out)
    print(f"[capture_isaac_obs] wrote {args_cli.out} ({T} ticks, obs_isaac shape {out['obs_isaac'].shape})")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
