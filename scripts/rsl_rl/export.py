"""Export a trained checkpoint to ONNX, for ``--view``, ``--share`` and deployment on the robot.

Writes ``policy.onnx``, a TorchScript ``policy.pt`` and copies of the run's ``env.yaml``, ``agent.yaml`` and
``code_state.yaml``, so the folder holds the same files as a policy shared on the Hugging Face Hub. Runs Isaac Sim
headless with one environment to build the policy, warns if the code changed since the run was trained, loads only
the policy from the checkpoint, then checks that the ONNX file gives the same actions as the PyTorch policy.
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

from cyclotron.hub import EXPERIMENT_TASKS, infer_task

import cli_args  # isort: skip

# Largest action difference allowed between the PyTorch policy and the ONNX file; float32 rounding stays far below it.
ONNX_TOLERANCE = 1e-4

parser = argparse.ArgumentParser(description="Export a trained Cyclotron checkpoint to ONNX.")
parser.add_argument(
    "--task",
    type=str,
    default=None,
    help="Task used to build the policy. Inferred from the run's agent.yaml or from --experiment_name if omitted.",
)
parser.add_argument(
    "--checkpoint",
    type=str,
    default=None,
    help="Checkpoint to export: a full path to a .pt file, or a filename inside the --load_run folder.",
)
parser.add_argument("--load_run", type=str, default=None, help="Run folder to export from. Defaults to the latest.")
parser.add_argument(
    "--experiment_name", type=str, default=None, help="Experiment folder under logs/rsl_rl/. Defaults to the task's."
)
parser.add_argument(
    "--output", type=str, default=None, help="Folder to write to. Defaults to <checkpoint folder>/exported."
)
parser.add_argument(
    "--strict", action="store_true", help="Stop, instead of warning, if the code changed since the run was trained."
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.headless = True

# Check what can be checked before Isaac Sim starts, which takes a while.
checkpoint_is_path = args_cli.checkpoint is not None and os.sep in args_cli.checkpoint
if checkpoint_is_path:
    args_cli.checkpoint = os.path.abspath(os.path.expanduser(args_cli.checkpoint))
    if not os.path.isfile(args_cli.checkpoint):
        sys.exit(f"[ERROR] Checkpoint not found: {args_cli.checkpoint}")
elif args_cli.checkpoint is not None and args_cli.load_run is None:
    sys.exit(
        f"[ERROR] --checkpoint '{args_cli.checkpoint}' is not a path. Pass a full path to the .pt file, or add"
        " --load_run <run> to pick it from that run folder."
    )
if args_cli.task is None:
    try:
        if checkpoint_is_path:
            args_cli.task = infer_task(os.path.dirname(args_cli.checkpoint))
        elif args_cli.experiment_name in EXPERIMENT_TASKS:
            args_cli.task = EXPERIMENT_TASKS[args_cli.experiment_name]
        else:
            sys.exit("[ERROR] Pass --task, or a full --checkpoint path so the task can be read from the run.")
    except ValueError as error:
        sys.exit(f"[ERROR] {error}")

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import importlib.metadata as metadata

import gymnasium as gym
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import class_to_dict
from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    handle_deprecated_rsl_rl_cfg,
    handle_deprecated_rsl_rl_checkpoint,
)
from isaaclab_tasks.utils.hydra import hydra_task_config

import cyclotron.tasks  # noqa: F401
from cyclotron.code_state import describe_changes, load_policy
from cyclotron.onnx_export import BUNDLE_YAMLS, OPTIONAL_BUNDLE_YAMLS, copy_run_yamls, max_onnx_difference

installed_version = metadata.version("rsl-rl-lib")

CODE_CHANGE_CONSEQUENCE = (
    "policy.onnx comes from the checkpoint and env.yaml from the run, so they still match each other, but the ONNX"
    " check runs on observations from the current code."
)


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    if args_cli.experiment_name is not None:
        agent_cfg.experiment_name = args_cli.experiment_name
    if args_cli.load_run is not None:
        agent_cfg.load_run = args_cli.load_run
    if args_cli.checkpoint is not None:
        agent_cfg.load_checkpoint = args_cli.checkpoint
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    env_cfg.scene.num_envs = 1
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    checkpoint = cli_args.resolve_checkpoint(log_root_path, agent_cfg, args_cli)
    run_dir = os.path.dirname(checkpoint)
    output_dir = os.path.abspath(os.path.expanduser(args_cli.output or os.path.join(run_dir, "exported")))

    env = gym.make(args_cli.task, cfg=env_cfg)
    # Compared where train.py saves params/: after the environment is created, which resolves parts of the config.
    changes, level = describe_changes(
        run_dir, class_to_dict(env_cfg), class_to_dict(agent_cfg), CODE_CHANGE_CONSEQUENCE
    )
    print(changes)
    if level == "error":
        env.close()
        sys.exit(1)
    if level == "warning" and args_cli.strict:
        print("[ERROR] Stopped by --strict: the code changed since the run was trained (see above).")
        env.close()
        sys.exit(1)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO] Loading model checkpoint from: {checkpoint}")
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    try:
        load_policy(
            runner, handle_deprecated_rsl_rl_checkpoint(checkpoint, installed_version), agent_cfg.class_name, run_dir
        )
    except ValueError as error:
        print(f"[ERROR] {error}")
        env.close()
        sys.exit(1)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    runner.export_policy_to_onnx(path=output_dir, filename="policy.onnx")
    runner.export_policy_to_jit(path=output_dir, filename="policy.pt")
    missing = copy_run_yamls(run_dir, output_dir)
    for name in missing:
        print(f"[WARNING] {run_dir}/params/{name} not found; the viewer and --share need it next to policy.onnx.")

    difference = None
    if policy.is_recurrent:
        print("[INFO] Skipping the ONNX check: it does not support recurrent policies.")
    else:
        difference = max_onnx_difference(policy, env.get_observations(), os.path.join(output_dir, "policy.onnx"))
    env.close()

    print(f"[INFO] Exported {os.path.basename(checkpoint)} to: {output_dir}")
    for name in ("policy.onnx", "policy.pt", *BUNDLE_YAMLS, *OPTIONAL_BUNDLE_YAMLS):
        if name not in missing and os.path.isfile(os.path.join(output_dir, name)):
            print(f"  {name}")
    if level == "warning":
        print("[WARNING] The code changed since this run was trained; see the warning before the export.")
    if difference is not None:
        # Exit here rather than after simulation_app.close(): hydra_task_config drops main's return value. Isaac Sim
        # replaces sys.exit with a version that only takes an exit code, so the message is printed first.
        if difference > ONNX_TOLERANCE:
            print(f"[ERROR] policy.onnx gives different actions than the checkpoint (max difference {difference:.2e}).")
            sys.exit(1)
        print(f"[INFO] Checked policy.onnx against the checkpoint: max action difference {difference:.1e}.")


if __name__ == "__main__":
    main()
    simulation_app.close()
