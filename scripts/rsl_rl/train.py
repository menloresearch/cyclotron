# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL training."""


import argparse
import logging
import os
import time
from datetime import datetime

import torch
import torch.distributed as dist

from isaaclab.app import launch_simulation
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.dict import class_to_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.entrypoints.common import (
    add_video_args,
    apply_video_recording,
    close_env,
    create_isaaclab_env,
    enable_cameras_for_video,
    pre_launch_video_config,
    set_hydra_args,
    validate_distributed_device,
)
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, create_rsl_rl_runner, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli

import cyclotron.tasks  # noqa: F401
from cyclotron.code_state import hash_run_configs, record_code_state, write_code_state
from cyclotron.policy_io import resolve_policy_io

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
add_video_args(parser, action="training")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
parser.add_argument(
    "--ray-proc-id", "-rid", type=int, default=None, help="Automatically configured by Ray integration, otherwise None."
)
cli_args.add_rsl_rl_args(parser)
cli_args.add_launcher_args(parser)
args_cli, hydra_args = setup_preset_cli(parser)
args_cli.frontend = "torch"
enable_cameras_for_video(args_cli)
set_hydra_args(hydra_args)

RSL_RL_VERSION = "5.5.1"


import importlib.metadata as metadata

from packaging import version

installed_version = metadata.version("rsl-rl-lib")
if version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\tpip install rsl-rl-lib=={RSL_RL_VERSION}\n"
    )
    exit(1)


logger = logging.getLogger(__name__)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

# The rank of this process with --distributed, None otherwise; set by torch.distributed.run.
RANK = int(os.getenv("RANK", "0")) if args_cli.distributed else None


def print_run_info(log_dir: str) -> None:
    """Print where the run is stored and how to resume it."""
    if RANK not in (None, 0):
        return
    resume_cmd = f"./cyclotron.sh --train --task {args_cli.task} --resume --load_run {os.path.basename(log_dir)}"
    if args_cli.num_envs is not None:
        resume_cmd += f" --num_envs {args_cli.num_envs}"
    if args_cli.headless:
        resume_cmd += " --headless"
    print(f"[INFO] Run directory: {log_dir}", flush=True)
    print(f"[INFO] To resume this run from its latest checkpoint: {resume_cmd}", flush=True)


def main():
    env_cfg, agent_cfg = resolve_task_config(args_cli.task, args_cli.agent)
    pre_launch_video_config(env_cfg, args_cli)
    with launch_simulation(env_cfg, args_cli):
        train(env_cfg, agent_cfg)


def train(env_cfg, agent_cfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    validate_distributed_device(args_cli)
    if not args_cli.distributed and args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    if RANK is not None:
        # launch_simulation put this process's environment on its own GPU; the agent follows it.
        agent_cfg.device = env_cfg.sim.device
        agent_cfg.seed += RANK
    env_cfg.seed = agent_cfg.seed

    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    else:
        logger.warning(
            "IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
        )

    env_cfg.log_dir = log_dir
    apply_video_recording(env_cfg, log_dir, args_cli)

    env = create_isaaclab_env(args_cli.task, env_cfg, args_cli, convert_marl_to_single_agent=True)

    # Passing --checkpoint implies --resume.
    agent_cfg.resume = agent_cfg.resume or args_cli.checkpoint is not None
    loads_checkpoint = agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation"
    if loads_checkpoint:
        resume_path = cli_args.resolve_checkpoint(log_root_path, agent_cfg, args_cli)

    start_time = time.time()

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = create_rsl_rl_runner(env, agent_cfg, log_dir=log_dir)
    runner.add_git_repo_to_log(__file__)
    if loads_checkpoint:
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        runner.load(resume_path, map_location=agent_cfg.device)

    # Only the main process writes params/, like rsl_rl's logger: ranks that start in the same second share log_dir,
    # and another rank's env.yaml (its own device and seed) wouldn't match the hashes this one records.
    if RANK in (None, 0):
        dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
        dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
        # The sha256 of the two files just written, so --export and --share notice if they are edited later.
        run_configs = hash_run_configs(os.path.join(log_dir, "params"))
        try:
            # What --export and --play compare against to warn about code changes; <run>/<file>, not a local path.
            loaded = os.path.join(*resume_path.split(os.sep)[-2:]) if loads_checkpoint else None
            agent_dict = class_to_dict(agent_cfg)
            code_state = record_code_state(
                class_to_dict(env_cfg),
                loaded_checkpoint=loaded,
                agent=agent_dict,
                policy_io=resolve_policy_io(env.unwrapped, agent_dict),
            )
        except Exception as error:
            # Bookkeeping never stops training; --export then reports what the run couldn't record.
            logger.warning(f"Couldn't record the code this run is trained with: {error!r}")
            code_state = {"error": repr(error)}
        code_state["run_configs"] = run_configs
        write_code_state(os.path.join(log_dir, "params"), code_state)

    print_run_info(log_dir)
    try:
        runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    finally:
        print_run_info(log_dir)

    print(f"Training time: {round(time.time() - start_time, 2)} seconds")

    close_env(env)
    # The RL library creates the process group but never destroys it, which torch warns about at exit.
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
