"""Export a trained checkpoint to ONNX, for ``--view``, ``--share`` and deployment on the robot.

Writes ``policy.onnx``, a TorchScript ``policy.pt`` and copies of the run's ``env.yaml``, ``agent.yaml`` and
``code_state.yaml``, so the folder holds the same files as a policy shared on the Hugging Face Hub. Runs Isaac Sim
headless with one environment to build the policy, with the policy settings set back to the ones the run saved in its
``env.yaml`` and ``agent.yaml`` (Hydra overrides on the command line apply on top), warns if the code changed since
the run was trained, loads only the policy from the checkpoint, then checks that the ONNX file gives the same actions
as the PyTorch policy.
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
# Settings given as Hydra overrides keep their value when the run's saved settings are restored.
OVERRIDDEN = [arg.split("=")[0].lstrip("+~") for arg in hydra_args]

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
from cyclotron.code_state import check_out_hint, describe_changes, load_policy, training_commit
from cyclotron.onnx_export import (
    BUNDLE_YAMLS,
    OPTIONAL_BUNDLE_YAMLS,
    attach_deploy_metadata,
    copy_run_yamls,
    deploy_metadata,
    existing_export_note,
    export_log,
    max_onnx_difference,
)
from cyclotron.run_config import load_run_configs, restore_policy_settings

installed_version = metadata.version("rsl-rl-lib")

CODE_CHANGE_CONSEQUENCE = (
    "Export set the policy settings back to the run's env.yaml and agent.yaml, but the code behind them (observation"
    " and action functions, the robot model) is the current code."
)


def _per_joint(value, count: int) -> list[float]:
    """A term's resolved scale or offset (a float, or a tensor row per env) as one float per joint."""
    import torch

    if isinstance(value, torch.Tensor):
        return [float(v) for v in value[0]]
    return [float(value)] * count


def _configured_offset(term) -> list[float]:
    """A term's offset as configured, one float per joint.

    With ``use_default_offset`` the live offset is the default pose of the one environment export builds, which
    startup events such as ``randomize_joint_default_pos`` perturb. Resolve the robot's configured
    ``init_state.joint_pos`` the way Isaac Lab builds the default pose instead.
    """
    from isaaclab.envs.mdp.actions import JointPositionAction
    from isaaclab.utils.string import resolve_matching_names_values

    if not (isinstance(term, JointPositionAction) and term.cfg.use_default_offset):
        return _per_joint(term._offset, len(term._joint_names))
    asset = term._asset
    pose = [0.0] * asset.num_joints
    indices, _, values = resolve_matching_names_values(asset.cfg.init_state.joint_pos, asset.joint_names)
    for index, value in zip(indices, values):
        pose[index] = float(value)
    joint_ids = range(asset.num_joints) if isinstance(term._joint_ids, slice) else term._joint_ids
    return [pose[int(i)] for i in joint_ids]


def gather_deploy_metadata(env, policy, run_dir: str) -> dict[str, str] | None:
    """Resolve the deployment contract from the live environment, or None (with a message) if an action term
    is not a joint action and the contract cannot describe it."""
    manager = getattr(env.unwrapped, "action_manager", None)
    if manager is None:
        print("[WARNING] The environment has no action manager (direct workflow); not attaching deploy metadata.")
        return None
    joint_names, scale, offset, clip, stiffness, damping = [], [], [], [], [], []
    clipped = False
    for name in manager.active_terms:
        term = manager.get_term(name)
        names = getattr(term, "_joint_names", None)
        if names is None:
            print(f"[WARNING] Action term {name} is not a joint action; not attaching deploy metadata.")
            return None
        joint_names += list(names)
        scale += _per_joint(term._scale, len(names))
        offset += _configured_offset(term)
        # The configured gains, not the simulated ones, which startup randomization events can perturb.
        stiffness += [float(v) for v in term._asset.data.default_joint_stiffness[0, term._joint_ids]]
        damping += [float(v) for v in term._asset.data.default_joint_damping[0, term._joint_ids]]
        if term.cfg.clip is None:
            clip += [[None, None]] * len(names)
        else:
            clipped = True
            # None for an unclipped side: the resolver fills joints the config does not name with +-inf.
            clip += [[None if abs(side) == float("inf") else side for side in pair] for pair in term._clip[0].tolist()]
    observations = env.unwrapped.observation_manager.active_terms
    groups = [group for group in policy.obs_groups if group in observations]
    observation_names = [
        name if len(groups) == 1 else f"{group}/{name}" for group in groups for name in observations[group]
    ]
    return deploy_metadata(
        joint_names=joint_names,
        action_scale=scale,
        action_offset=offset,
        action_clip=clip if clipped else None,
        joint_stiffness=stiffness,
        joint_damping=damping,
        sim_dt=env.unwrapped.physics_dt,
        decimation=env.unwrapped.cfg.decimation,
        observation_names=observation_names,
        trained_commit=training_commit(run_dir),
    )


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    if args_cli.experiment_name is not None:
        agent_cfg.experiment_name = args_cli.experiment_name
    if args_cli.load_run is not None:
        agent_cfg.load_run = args_cli.load_run
    if args_cli.checkpoint is not None:
        agent_cfg.load_checkpoint = args_cli.checkpoint

    env_cfg.scene.num_envs = 1
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    checkpoint = cli_args.resolve_checkpoint(log_root_path, agent_cfg, args_cli)
    run_dir = os.path.dirname(checkpoint)
    output_dir = os.path.abspath(os.path.expanduser(args_cli.output or os.path.join(run_dir, "exported")))
    log = export_log(output_dir, f"export of {checkpoint} with task {args_cli.task}")
    note = existing_export_note(run_dir, output_dir)
    if note:
        log(f"[INFO] {note}")

    saved = load_run_configs(run_dir)
    if saved is not None:
        problems = restore_policy_settings(env_cfg, agent_cfg, *saved, overridden=OVERRIDDEN)
        if problems:
            log("[ERROR] The current code can't rebuild the policy settings this run was trained with:")
            for line in problems:
                log(f"    {line}")
            log(f"  {check_out_hint(run_dir)}")
            sys.exit(1)
        log("[INFO] Set the policy settings back to the ones in the run's env.yaml and agent.yaml.")
    # After the restore, which can bring back a network config of another kind (e.g. an LSTM actor).
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    env = gym.make(args_cli.task, cfg=env_cfg)
    # Compared where train.py saves params/: after the environment is created, which resolves parts of the config.
    changes, level = describe_changes(
        run_dir, class_to_dict(env_cfg), class_to_dict(agent_cfg), CODE_CHANGE_CONSEQUENCE
    )
    log(changes)
    if level == "error":
        env.close()
        sys.exit(1)
    if level == "warning" and args_cli.strict:
        log("[ERROR] Stopped by --strict: the code changed since the run was trained (see above).")
        env.close()
        sys.exit(1)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    log(f"[INFO] Loading model checkpoint from: {checkpoint}")
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
        log(f"[ERROR] {error}")
        env.close()
        sys.exit(1)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    runner.export_policy_to_onnx(path=output_dir, filename="policy.onnx")
    runner.export_policy_to_jit(path=output_dir, filename="policy.pt")
    missing = copy_run_yamls(run_dir, output_dir)
    for name in missing:
        log(f"[WARNING] {run_dir}/params/{name} not found; the viewer and --share need it next to policy.onnx.")

    metadata = gather_deploy_metadata(env, policy, run_dir)
    if metadata is not None:
        attach_deploy_metadata(os.path.join(output_dir, "policy.onnx"), metadata)
        log(f"[INFO] Attached deploy metadata to policy.onnx: {', '.join(metadata)}, obs_dim, action_dim.")

    difference = None
    if policy.is_recurrent:
        log("[INFO] Skipping the ONNX check: it does not support recurrent policies.")
    else:
        difference = max_onnx_difference(policy, env.get_observations(), os.path.join(output_dir, "policy.onnx"))
    env.close()

    log(f"[INFO] Exported {os.path.basename(checkpoint)} to: {output_dir}")
    for name in ("policy.onnx", "policy.pt", *BUNDLE_YAMLS, *OPTIONAL_BUNDLE_YAMLS):
        if name not in missing and os.path.isfile(os.path.join(output_dir, name)):
            log(f"  {name}")
    if level == "warning":
        log("[WARNING] The code changed since this run was trained; see the warning before the export.")
    if difference is not None:
        # Exit here rather than after simulation_app.close(): hydra_task_config drops main's return value. Isaac Sim
        # replaces sys.exit with a version that only takes an exit code, so the message is printed first.
        if difference > ONNX_TOLERANCE:
            log(f"[ERROR] policy.onnx gives different actions than the checkpoint (max difference {difference:.2e}).")
            sys.exit(1)
        log(f"[INFO] Checked policy.onnx against the checkpoint: max action difference {difference:.1e}.")


if __name__ == "__main__":
    main()
    simulation_app.close()
