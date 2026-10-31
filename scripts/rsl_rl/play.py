# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL evaluation."""


import argparse
import os
import sys

from isaaclab.app import launch_simulation
from isaaclab.utils.dict import class_to_dict
from isaaclab_rl.entrypoints.common import (
    add_video_args,
    apply_video_recording,
    close_env,
    create_isaaclab_env,
    enable_cameras_for_video,
    pre_launch_video_config,
    run_playback,
    set_hydra_args,
)
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, create_rsl_rl_runner, handle_deprecated_rsl_rl_cfg
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli

import cyclotron.tasks  # noqa: F401
from cyclotron.code_state import check_out_hint, describe_changes
from cyclotron.policy_io import resolve_policy_io
from cyclotron.policy_loading import load_policy
from cyclotron.robot_model import load_training_robot_model, overrides_robot_model

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
add_video_args(parser, action="play")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--target", type=str, default=None, help="Direct path to the checkpoint file to play (alias for --checkpoint)."
)
parser.add_argument(
    "--strict", action="store_true", help="Stop, instead of warning, if the code changed since the run was trained."
)
parser.add_argument(
    "--current_robot_model",
    action="store_true",
    help="Simulate the robot model the current code loads, instead of the one the run was trained with.",
)
cli_args.add_rsl_rl_args(parser)
cli_args.add_launcher_args(parser)
args_cli, hydra_args = setup_preset_cli(parser)
# Unknown flags go to Hydra, which would only reject these after Isaac Sim has started.
REMOVED_EXPORT_FLAGS = (
    "--export-only",
    "--export_only",
    "--onnx-output",
    "--onnx_output",
    "--onnx-filename",
    "--onnx_filename",
)
removed = [arg.split("=")[0] for arg in hydra_args if arg.split("=")[0] in REMOVED_EXPORT_FLAGS]
if removed:
    sys.exit(f"[ERROR] {removed[0]} was removed from --play, which no longer writes ONNX. Use ./cyclotron.sh --export.")
if args_cli.target and not args_cli.checkpoint:
    args_cli.checkpoint = args_cli.target
# Isaac Lab 3.0 runs headless unless a visualizer is requested; play opens the Isaac Sim viewport as it did in 2.x.
if args_cli.visualizer is None and not args_cli.headless:
    args_cli.visualizer = ["kit"]
args_cli.frontend = "torch"
enable_cameras_for_video(args_cli)
set_hydra_args(hydra_args)


import importlib.metadata as metadata

installed_version = metadata.version("rsl-rl-lib")

CODE_CHANGE_CONSEQUENCE = (
    "--play runs the checkpoint in an environment built from the current code, so it may behave differently than in"
    " training."
)


def main():
    env_cfg, agent_cfg = resolve_task_config(args_cli.task, args_cli.agent)
    pre_launch_video_config(env_cfg, args_cli)
    with launch_simulation(env_cfg, args_cli):
        play(env_cfg, agent_cfg)


def play(env_cfg, agent_cfg):
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if args_cli.disable_fabric:
        env_cfg.sim.use_fabric = False

    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    else:
        resume_path = cli_args.resolve_checkpoint(log_root_path, agent_cfg, args_cli)

    log_dir = os.path.dirname(resume_path)

    env_cfg.log_dir = log_dir
    apply_video_recording(env_cfg, log_dir, args_cli, subdir="play", checkpoint_path=resume_path)
    robot_note = None
    # Like --current_robot_model, a urdf named by a Hydra override is loaded as it is.
    if not args_cli.current_robot_model and not overrides_robot_model(hydra_args):
        robot_loaded, robot_note = load_training_robot_model(env_cfg, log_dir)
        if robot_loaded:
            print(f"[INFO] {robot_loaded}")

    env = create_isaaclab_env(args_cli.task, env_cfg, args_cli, convert_marl_to_single_agent=False)
    # Compared where train.py saves params/: after the environment is created, which resolves parts of the config.
    agent_dict = class_to_dict(agent_cfg)
    changes, level = describe_changes(
        log_dir,
        class_to_dict(env_cfg),
        agent_dict,
        CODE_CHANGE_CONSEQUENCE,
        policy_io=resolve_policy_io(env.unwrapped, agent_dict),
        robot_model_note=robot_note,
    )
    print(changes)
    if level == "error":
        close_env(env)
        sys.exit(1)
    if level == "warning" and args_cli.strict:
        # Isaac Sim replaces sys.exit with a version that only takes an exit code, so the message is printed first.
        print("[ERROR] Stopped by --strict: the code changed since the run was trained (see above).")
        close_env(env)
        sys.exit(1)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    runner = create_rsl_rl_runner(env, agent_cfg)
    try:
        load_policy(runner, resume_path, agent_cfg.class_name, check_out_hint(log_dir))
    except ValueError as error:
        print(f"[ERROR] {error}")
        close_env(env)
        sys.exit(1)

    policy = runner.get_inference_policy(device=env.unwrapped.device)

    obs = env.get_observations()

    def step() -> None:
        nonlocal obs
        obs, _, dones, _ = env.step(policy(obs))
        policy.reset(dones)

    run_playback(step, dt=env.unwrapped.step_dt, args_cli=args_cli, env_cfg=env_cfg)

    close_env(env)


if __name__ == "__main__":
    main()
