"""Export a trained checkpoint to ONNX, for ``--view``, ``--share`` and deployment on the robot.

Writes ``policy.onnx``, a TorchScript ``policy.pt`` and copies of the run's ``env.yaml``, ``agent.yaml`` and
``code_state.yaml``, so the folder holds the same files as a policy shared on the Hugging Face Hub. Runs Isaac Sim
headless with one environment to build the policy, like a restart of the run: the policy settings are set back to
the ones the run saved in its ``env.yaml`` and ``agent.yaml``, which are their only source (no overrides), and are
checked against those files once the environment is built. Warns if code that can change the exported policy (the
package files behind the policy settings, the robot model, Isaac Lab, library versions) changed since the run was
trained, loads only the policy from the checkpoint, then checks that the ONNX file gives the same actions as the PyTorch
policy. A run without ``env.yaml`` or ``agent.yaml`` can have them written from the current code, if you agree.
"""

import argparse
import os
import shutil
import sys
import tempfile

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
    "--strict",
    action="store_true",
    help="Stop, instead of warning, if code that can change the exported policy changed since the run was trained,"
    " or anything else doesn't come from the run.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, unknown_args = parser.parse_known_args()
args_cli.headless = True

# Check what can be checked before Isaac Sim starts, which takes a while.
if unknown_args:
    sys.exit(
        f"[ERROR] --export takes no setting overrides or other extra arguments ({' '.join(unknown_args)}): the run's"
        " env.yaml and agent.yaml are the only source of its settings."
    )
if args_cli.checkpoint is not None and "://" in args_cli.checkpoint:
    sys.exit(
        "[ERROR] --export needs the run folder around the checkpoint (params/env.yaml and agent.yaml), so it can't"
        " export a checkpoint URL. Download the run folder and pass the checkpoint's path."
    )
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

sys.argv = [sys.argv[0]]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import importlib.metadata as metadata
from datetime import datetime

import gymnasium as gym
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import class_to_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg, handle_deprecated_rsl_rl_checkpoint
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

import cyclotron.tasks  # noqa: F401
from cyclotron.code_state import (
    CODE_STATE_FILE,
    RUN_CONFIGS,
    check_out_hint,
    code_differences,
    current_code,
    edited_run_configs,
    load_policy,
    load_run_configs,
    normalize_config,
    policy_code_files,
    rebuild_differences,
    record_code_state,
    training_commit,
)
from cyclotron.onnx_export import (
    BUNDLE_YAMLS,
    OPTIONAL_BUNDLE_YAMLS,
    RECURRENT_STEPS,
    attach_deploy_metadata,
    copy_run_yamls,
    deploy_metadata,
    existing_export_note,
    export_log,
    max_onnx_difference,
    replace_export,
)
from cyclotron.run_config import (
    generated_run_configs,
    mark_generated,
    missing_run_configs,
    restore_policy_settings,
    trained_outside_cyclotron,
)

installed_version = metadata.version("rsl-rl-lib")

CODE_CHANGE_CONSEQUENCE = (
    "The policy settings come from the run's env.yaml and agent.yaml, but the code behind them is the current code."
    " Only code that can change the exported policy is listed: the package files defining the functions the policy"
    " settings name, the robot model, Isaac Lab and the library versions."
)


def ask_to_generate(run_dir: str, missing: list[str], log) -> bool:
    """Ask in the terminal whether to write the run's missing configs from the current code. No answer is a no."""
    files = " and ".join(missing)
    log(f"[WARNING] {run_dir}/params has no {files}, so the settings this run was trained with are unknown.")
    log(
        f"  Export can write {files} from the current code ({current_code()}) into params/, marked as generated."
        " The exported policy then gets the current code's settings, which may not be the ones it was trained with."
        " Only do this from the same codebase and commit the policy was trained with: nothing can verify it."
    )
    try:
        answer = input(f"Generate {files} from the current code? [y/N] ").strip()
    except EOFError:
        answer = ""
    log(f"  Answer: {answer or 'none'}")
    return answer.lower() in ("y", "yes")


def _per_joint(value, count: int) -> list[float]:
    """A term's resolved scale or offset (a float, or a tensor row per env) as one float per joint."""
    import torch

    if isinstance(value, torch.Tensor):
        return [float(v) for v in value[0]]
    return [float(value)] * count


def _configured_offset(term) -> list[float]:
    """A joint position action's offset as configured, one float per joint.

    With ``use_default_offset`` the live offset is the default pose of the one environment export builds, which
    startup events such as ``randomize_joint_default_pos`` perturb. Resolve the robot's configured
    ``init_state.joint_pos`` the way Isaac Lab builds the default pose instead.
    """
    from isaaclab.utils.string import resolve_matching_names_values

    if not term.cfg.use_default_offset:
        return _per_joint(term._offset, len(term._joint_names))
    asset = term._asset
    pose = [0.0] * asset.num_joints
    indices, _, values = resolve_matching_names_values(asset.cfg.init_state.joint_pos, asset.joint_names)
    for index, value in zip(indices, values):
        pose[index] = float(value)
    joint_ids = range(asset.num_joints) if isinstance(term._joint_ids, slice) else term._joint_ids
    return [pose[int(i)] for i in joint_ids]


def gather_deploy_metadata(env, policy, run_dir: str, raw_action_clip: float | None, log) -> dict[str, str] | None:
    """Resolve the deployment contract from the live environment, or None (with a message) if an action term
    is not a joint position action and the contract cannot describe it. ``raw_action_clip`` is the run's
    ``clip_actions``, which rsl_rl's environment wrapper applies to the raw actions."""
    from isaaclab.envs.mdp.actions import JointPositionAction

    manager = getattr(env.unwrapped, "action_manager", None)
    if manager is None:
        log("[WARNING] The environment has no action manager (direct workflow); not attaching deploy metadata.")
        return None
    joint_names, scale, offset, clip, stiffness, damping = [], [], [], [], [], []
    clipped = False
    for name in manager.active_terms:
        term = manager.get_term(name)
        # Velocity, effort and relative position actions turn actions into something other than position targets.
        if not isinstance(term, JointPositionAction):
            log(
                f"[WARNING] Action term {name} ({type(term).__name__}) is not a joint position action; not attaching"
                " deploy metadata."
            )
            return None
        names = term._joint_names
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
        raw_action_clip=raw_action_clip,
        action_scale=scale,
        action_offset=offset,
        action_clip=clip if clipped else None,
        joint_stiffness=stiffness,
        joint_damping=damping,
        sim_dt=env.unwrapped.physics_dt,
        decimation=env.unwrapped.cfg.decimation,
        observation_names=observation_names,
        trained_commit=training_commit(run_dir),
        trained_outside_cyclotron=trained_outside_cyclotron(run_dir),
    )


def main():
    # Straight from the task's registration, without Hydra: nothing on the command line changes a setting.
    env_cfg = load_cfg_from_registry(args_cli.task.split(":")[-1], "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(args_cli.task.split(":")[-1], "rsl_rl_cfg_entry_point")
    if args_cli.experiment_name is not None:
        agent_cfg.experiment_name = args_cli.experiment_name
    if args_cli.load_run is not None:
        agent_cfg.load_run = args_cli.load_run
    if args_cli.checkpoint is not None:
        agent_cfg.load_checkpoint = args_cli.checkpoint

    env_cfg.scene.num_envs = 1
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    # The policy runs where the environment does.
    agent_cfg.device = env_cfg.sim.device

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    checkpoint = cli_args.resolve_checkpoint(log_root_path, agent_cfg, args_cli)
    run_dir = os.path.dirname(checkpoint)
    output_dir = os.path.abspath(os.path.expanduser(args_cli.output or os.path.join(run_dir, "exported")))
    log = export_log(output_dir, f"export of {checkpoint} with task {args_cli.task}")
    note = existing_export_note(run_dir, output_dir)
    if note:
        log(f"[INFO] {note}")

    # A policy is exported with the env.yaml and agent.yaml it was trained with, as training wrote them.
    edited = edited_run_configs(run_dir)
    if edited:
        log(
            f"[ERROR] {run_dir}/params/{' and '.join(edited)} changed or went missing since training: it no longer"
            f" matches the sha256 the run's {CODE_STATE_FILE} recorded. These files are the record of how the run was"
            " trained, so they are never edited by hand; restore them from a backup or the run's Hub repo, or train a"
            " new run."
        )
        sys.exit(1)
    if edited is None:
        log(
            "[INFO] The run records no sha256 of env.yaml and agent.yaml (it was trained before training recorded them"
            f" in {CODE_STATE_FILE}), so edits since training can't be detected."
        )
    not_saved = missing_run_configs(run_dir)
    if not_saved:
        if args_cli.strict:
            log(
                f"[ERROR] Stopped by --strict: {run_dir}/params has no {' and '.join(not_saved)}, so the settings this"
                " run was trained with are unknown. Without --strict, --export asks whether to write it from the"
                " current code."
            )
            sys.exit(1)
        if not ask_to_generate(run_dir, not_saved, log):
            log(
                "[ERROR] Stopped: a policy is exported with its env.yaml and agent.yaml, and this run has no"
                f" {' and '.join(not_saved)}."
            )
            sys.exit(1)
    generated = generated_run_configs(run_dir)
    if generated:
        # Recorded, not stopped, also with --strict: the file's first line says it was generated, and it is shared
        # with the policy, so whoever receives it can see where the settings came from.
        log(
            f"[WARNING] {run_dir}/params/{' and '.join(generated)} was written from the code by an earlier --export,"
            " not by training."
        )
    if trained_outside_cyclotron(run_dir):
        log(
            f"[WARNING] The run has no {CODE_STATE_FILE}, so it was trained outside cyclotron (or before cyclotron"
            " recorded one), and nothing can check this export against the code it was trained with. Export from the"
            " same codebase and commit the policy was trained with, as far as possible. The exported policy is tagged"
            " as trained outside cyclotron."
        )

    problems = restore_policy_settings(env_cfg, agent_cfg, *load_run_configs(run_dir))
    if problems:
        log("[ERROR] The current code can't rebuild the policy settings this run was trained with:")
        for line in problems:
            log(f"    {line}")
        log(f"  {check_out_hint(run_dir)}")
        sys.exit(1)
    saved = [name for name in RUN_CONFIGS if name not in not_saved]
    if saved:
        log(f"[INFO] Set the policy settings back to the ones in the run's {' and '.join(saved)}.")
    # After the restore, which can bring back a network config of another kind (e.g. an LSTM actor).
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    env = gym.make(args_cli.task, cfg=env_cfg)
    # Written and compared where train.py saves params/: after the environment is created, which resolves parts of
    # the config.
    env_dict, agent_dict = class_to_dict(env_cfg), class_to_dict(agent_cfg)
    for name, cfg in zip(RUN_CONFIGS, (env_cfg, agent_cfg)):
        if name in not_saved:
            path = os.path.join(run_dir, "params", name)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            dump_yaml(path, cfg)
            mark_generated(path, current_code(), datetime.now().strftime("%Y-%m-%d %H:%M"))
            log(f"[WARNING] Wrote {path} from the current code, marked as generated.")
    saved_configs = load_run_configs(run_dir)
    mismatches, new = rebuild_differences(saved_configs, env_dict, agent_dict)
    if mismatches:
        log("[ERROR] The rebuilt policy settings don't match the run's env.yaml and agent.yaml:")
        for line in mismatches:
            log(f"    {line}")
        env.close()
        sys.exit(1)
    log("[INFO] Checked the rebuilt policy settings against the run's env.yaml and agent.yaml: they match.")
    if new:
        log("[WARNING] The current code has policy settings the run didn't save; they keep the current code's value:")
        for line in new:
            log(f"    {line}")
        if args_cli.strict:
            log("[ERROR] Stopped by --strict: these settings don't come from the run (see above).")
            env.close()
            sys.exit(1)
    # Rewards, the training algorithm and the like can't change what is exported, so they aren't compared here.
    behind_settings = policy_code_files(*saved_configs)
    code = code_differences(run_dir, record_code_state(normalize_config(env_dict)), behind_settings)
    if code:
        log("[WARNING] Code behind the policy settings changed since this run was trained:")
        for line in code:
            log(f"    {line}")
        log(f"  {CODE_CHANGE_CONSEQUENCE}")
        log(f"  For the exact training code: {check_out_hint(run_dir)}")
        if args_cli.strict:
            log("[ERROR] Stopped by --strict: code behind the policy settings changed since training (see above).")
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

    # Written to a staging folder and moved into place only once the check below passes, so a failed export leaves
    # the previous one as it was.
    staging = tempfile.mkdtemp(prefix=".staging-", dir=output_dir)
    try:
        onnx_path = os.path.join(staging, "policy.onnx")
        runner.export_policy_to_onnx(path=staging, filename="policy.onnx")
        runner.export_policy_to_jit(path=staging, filename="policy.pt")
        missing = copy_run_yamls(run_dir, staging)
        for name in missing:
            log(f"[WARNING] {run_dir}/params/{name} not found; the viewer and --share need it next to policy.onnx.")

        metadata = gather_deploy_metadata(env, policy, run_dir, agent_cfg.clip_actions, log)
        if metadata is not None:
            attach_deploy_metadata(onnx_path, metadata)
            log(f"[INFO] Attached deploy metadata to policy.onnx: {', '.join(metadata)}, obs_dim, action_dim.")

        difference = max_onnx_difference(policy, env.get_observations(), onnx_path)
        checked = f"actions and memory over {RECURRENT_STEPS} steps" if policy.is_recurrent else "actions"
        env.close()
        # Isaac Sim replaces sys.exit with a version that only takes an exit code, so the message is printed first.
        if difference > ONNX_TOLERANCE:
            shutil.rmtree(staging)
            log(
                "[ERROR] policy.onnx gives different actions than the checkpoint"
                f" (max difference {difference:.2e}; checked {checked}). Nothing was written to {output_dir}."
            )
            sys.exit(1)
        replace_export(staging, output_dir)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    log(f"[INFO] Checked policy.onnx against the checkpoint ({checked}): max difference {difference:.1e}.")

    log(f"[INFO] Exported {os.path.basename(checkpoint)} to: {output_dir}")
    for name in ("policy.onnx", "policy.pt", *BUNDLE_YAMLS, *OPTIONAL_BUNDLE_YAMLS):
        if name not in missing and os.path.isfile(os.path.join(output_dir, name)):
            log(f"  {name}")
    if not_saved or generated:
        log(f"[WARNING] {' and '.join(not_saved or generated)} came from the current code, not from training.")
    if new:
        log("[WARNING] The current code has policy settings the run didn't save; see the warning before the export.")
    if code:
        log("[WARNING] Code behind the policy settings changed since training; see the warning before the export.")


if __name__ == "__main__":
    main()
    simulation_app.close()
