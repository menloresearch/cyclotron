"""Export a trained checkpoint to ONNX, for ``--view``, ``--share`` and deployment on the robot.

Writes ``policy.onnx``, a TorchScript ``policy.pt`` and copies of the run's ``env.yaml``, ``agent.yaml`` and
``code_state.yaml``, so the folder holds the same files as a policy shared on the Hugging Face Hub. Runs Isaac Sim
headless with one environment to build the policy, like a restart of the run: the policy settings are set back to
the ones the run saved in its ``env.yaml`` and ``agent.yaml``, which are their only source (no overrides), and are
checked against those files once the environment is built. Stops if the joints and gains the policy's inputs and
outputs resolve to (from the robot model) differ from the ones training recorded in ``code_state.yaml``, and warns if
the robot model or the code of the policy's network (its rsl_rl models and modules) changed since the run was trained.
Then it loads only the policy from the checkpoint, and checks that policy.onnx and policy.pt give the same actions as
the PyTorch policy. A run without ``env.yaml`` or ``agent.yaml`` can have them written from the current code, if you agree.
"""

import argparse
import os
import shutil
import sys
import tempfile
import traceback

from isaaclab.app import AppLauncher

from cyclotron.hub import EXPERIMENT_TASKS, infer_task

import cli_args  # isort: skip

# Largest action difference allowed between the PyTorch policy and policy.onnx or policy.pt, relative to the action
# where it is above 1; float32 rounding stays far below it.
EXPORT_TOLERANCE = 1e-4

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
    help="Stop, instead of warning, if the code of the policy's network or the robot model changed since the run was"
    " trained, or anything else doesn't come from the run.",
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
    ACTOR_CODE_CHANGED,
    ACTOR_CODE_UNRECORDED,
    CODE_STATE_FILE,
    ROBOT_MODEL_CHANGED,
    ROBOT_MODEL_UNCHECKED,
    actor_code_differences,
    check_out_hint,
    code_state_missing,
    current_code,
    edited_run_configs,
    read_code_state,
    robot_model_check,
)
from cyclotron.onnx_export import (
    BUNDLE_YAMLS,
    OPTIONAL_BUNDLE_YAMLS,
    RECURRENT_STEPS,
    attach_deploy_metadata,
    copy_run_yamls,
    deploy_metadata_from_io,
    existing_export_note,
    export_log,
    max_jit_difference,
    max_onnx_difference,
    replace_export,
)
from cyclotron.policy_io import policy_io_differences, resolve_policy_io
from cyclotron.policy_loading import load_policy
from cyclotron.run_config import (
    RUN_CONFIGS,
    generated_run_configs,
    load_config,
    load_run_configs,
    mark_generated,
    missing_run_configs,
    rebuild_differences,
    restore_policy_settings,
)

installed_version = metadata.version("rsl-rl-lib")

ACTOR_CODE_CONSEQUENCE = (
    "The weights are the run's, but the network that runs them is the current code, so the exported policy may"
    " compute something else. Only the code of the network is compared (rsl_rl's models and modules, and a custom"
    " network's own module), not the rest of the library or of this package."
)


def ask_to_generate(run_dir: str, missing: list[str], log) -> bool:
    """Ask in the terminal whether to write the run's missing configs from the current code. No answer, or no
    terminal to answer in, is a no."""
    files = " and ".join(missing)
    log(f"[WARNING] {run_dir}/params has no {files}, so the settings this run was trained with are unknown.")
    log(
        f"  Export can write {files} from the current code ({current_code()}) into params/, marked as generated, once"
        " the export succeeds. The exported policy then gets the current code's settings, which may not be the ones it"
        " was trained with. Only do this from the same codebase and commit the policy was trained with: nothing can"
        " verify it."
    )
    if not sys.stdin.isatty():
        log("  No terminal to answer in.")
        return False
    try:
        answer = input(f"Generate {files} from the current code? [y/N] ").strip()
    except EOFError:
        answer = ""
    log(f"  Answer: {answer or 'none'}")
    return answer.lower() in ("y", "yes")


def generated_config(cfg) -> str:
    """A run config written from the current code, starting with a line saying so."""
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "config.yaml")
        dump_yaml(path, cfg)
        mark_generated(path, current_code(), datetime.now().strftime("%Y-%m-%d %H:%M"))
        with open(path) as f:
            return f.read()


def check_policy_io(env, run_dir: str, env_cfg: dict, agent: dict, log) -> tuple[dict, str]:
    """Compare the joints and gains the policy's inputs and outputs resolve to, and the robot model, with the run's.

    Stops (closing ``env``) when the joints or gains changed, and on a changed robot model with --strict. Returns the
    resolved inputs and outputs (``resolve_policy_io``) and the robot model's status (``robot_model_check``).
    """
    # The settings name joints by pattern; the robot model decides which joints they match, in which order, and with
    # which gains. The check of policy.onnx against the checkpoint can't see a change here: both run in this
    # environment.
    io = resolve_policy_io(env.unwrapped, agent)
    recorded_io = (read_code_state(run_dir) or {}).get("policy_io")
    robot_model, robot_change = robot_model_check(run_dir, env_cfg)
    io_changes = policy_io_differences(recorded_io, io) if recorded_io else []
    if io_changes:
        log("[ERROR] The joints and gains the policy's inputs and outputs resolve to changed since training:")
        for line in io_changes:
            log(f"    {line}")
        if robot_change:
            log(f"  The robot model changed: {robot_change}")
        log("  The policy settings match the run's, so the robot model the current code loads resolves them")
        log("  differently: the exported policy would drive the wrong joints or gains. Never exported, also without")
        log(f"  --strict. Load the robot model the run was trained with. {check_out_hint(run_dir)}")
        env.close()
        sys.exit(1)
    if recorded_io:
        log("[INFO] Checked the joints and gains the policy resolves to against the run's: they match.")
    if robot_model == ROBOT_MODEL_CHANGED:
        log(f"[WARNING] The robot model changed since training: {robot_change}")
        if recorded_io:
            log("  The joints and gains the policy uses still match the run's, but the simulated robot doesn't.")
        else:
            log("  The run records no joints and gains (it was trained before training recorded them), so whether the")
            log("  policy's actions still reach the same joints can't be checked.")
        if args_cli.strict:
            log("[ERROR] Stopped by --strict: the robot model changed since training (see above).")
            env.close()
            sys.exit(1)
    elif robot_model == ROBOT_MODEL_UNCHECKED and not recorded_io:
        log(
            "[INFO] The run records neither its robot model's sha256 nor the joints and gains the policy resolved to,"
            " so which joints the policy's actions reach can't be checked against training."
        )
    return io, robot_model


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
    if code_state_missing(run_dir):
        log(
            f"[WARNING] The run has no {CODE_STATE_FILE}, so it was trained outside cyclotron (or before cyclotron"
            " recorded one), and nothing can check this export against the code it was trained with. Export from the"
            " same codebase and commit the policy was trained with, as far as possible. The exported policy is tagged"
            " code_state_missing."
        )
    elif "error" in (code_state := read_code_state(run_dir)):
        log(
            f"[WARNING] Training couldn't record the code in {CODE_STATE_FILE} ({code_state['error']}),"
            " so only the env.yaml and agent.yaml hashes can be checked against it."
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
    # Kept here and written into params/ only once the export succeeds, so a stopped export leaves the run as it was.
    generated_now = {
        name: generated_config(cfg) for name, cfg in zip(RUN_CONFIGS, (env_cfg, agent_cfg)) if name in not_saved
    }
    saved_configs = tuple(
        load_config(generated_now[name]) if name in generated_now else config
        for name, config in zip(RUN_CONFIGS, load_run_configs(run_dir))
    )
    mismatches, new = rebuild_differences(saved_configs, env_dict, agent_dict)
    if mismatches:
        log("[ERROR] The rebuilt policy settings don't match the run's env.yaml and agent.yaml:")
        for line in mismatches:
            log(f"    {line}")
        log("  The current code changes these settings while building the environment, so the exported policy wouldn't")
        log(f"  see what it was trained with. {check_out_hint(run_dir)}")
        env.close()
        sys.exit(1)
    log("[INFO] Checked the rebuilt policy settings against the run's env.yaml and agent.yaml: they match.")
    io, robot_model = check_policy_io(env, run_dir, env_dict, agent_dict, log)
    if new:
        log("[WARNING] The current code has policy settings the run didn't save; they keep the current code's value:")
        for line in new:
            log(f"    {line}")
        log("  They were added to the code after this run was trained, so its env.yaml and agent.yaml can't say what")
        log("  they were. If these values reproduce how the run was trained (a new setting whose default keeps the old")
        log("  behaviour), the export is right; otherwise it isn't the trained policy.")
        log(f"  {check_out_hint(run_dir)}")
        if args_cli.strict:
            log(
                "[ERROR] Stopped by --strict: these settings don't come from the run (see above). Export from the"
                " commit the run was trained with, or, once you have checked the values above, without --strict"
                " (--share always uses --strict, so share from the training commit)."
            )
            env.close()
            sys.exit(1)
    # Only the network's code can change what the weights compute without showing in the settings or the check at the
    # end; rewards, observation functions and the rest of the library can't change what is exported.
    actor_code, actor_changes = actor_code_differences(run_dir, saved_configs[1])
    if actor_code == ACTOR_CODE_CHANGED:
        log("[WARNING] The code of the policy's network changed since this run was trained:")
        for line in actor_changes:
            log(f"    {line}")
        log(f"  {ACTOR_CODE_CONSEQUENCE}")
        log(f"  For the exact training code: {check_out_hint(run_dir)}")
        if args_cli.strict:
            log("[ERROR] Stopped by --strict: the code of the policy's network changed since training (see above).")
            env.close()
            sys.exit(1)
    elif actor_code == ACTOR_CODE_UNRECORDED and not code_state_missing(run_dir):
        # A run without code_state.yaml was warned about above; this one has it but recorded no network code.
        log(
            f"[WARNING] The run's {CODE_STATE_FILE} records no network code (it was trained before training recorded"
            " it), so changes to the network since training can't be detected. Never a stop, also with --strict."
        )
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
            runner,
            handle_deprecated_rsl_rl_checkpoint(checkpoint, installed_version),
            agent_cfg.class_name,
            check_out_hint(run_dir),
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
        for name, text in generated_now.items():
            with open(os.path.join(staging, name), "w") as f:
                f.write(text)
        missing = [name for name in missing if name not in generated_now]
        for name in missing:
            log(f"[WARNING] {run_dir}/params/{name} not found; the viewer and --share need it next to policy.onnx.")

        unwrapped = env.unwrapped
        deploy = deploy_metadata_from_io(
            io, agent_cfg.clip_actions, unwrapped.physics_dt, unwrapped.cfg.decimation, run_dir, actor_code, robot_model
        )
        if deploy is None:
            log(f"[WARNING] {io['unsupported']} Not attaching deploy metadata.")
        else:
            attach_deploy_metadata(onnx_path, deploy)
            log(f"[INFO] Attached deploy metadata to policy.onnx: {', '.join(deploy)}, obs_dim, action_dim.")

        # rsl_rl writes each file with its own exporter, so each is checked.
        obs = env.get_observations()
        differences = {
            "policy.onnx": max_onnx_difference(policy, obs, onnx_path),
            "policy.pt": max_jit_difference(policy, obs, os.path.join(staging, "policy.pt")),
        }
        checked = f"actions and memory over {RECURRENT_STEPS} steps" if policy.is_recurrent else "actions"
        env.close()
        # Isaac Sim replaces sys.exit with a version that only takes an exit code, so the message is printed first.
        wrong = {name: difference for name, difference in differences.items() if difference > EXPORT_TOLERANCE}
        if wrong:
            shutil.rmtree(staging)
            for name, difference in wrong.items():
                log(
                    f"[ERROR] {name} gives different actions than the checkpoint"
                    f" (max difference {difference:.2e}; checked {checked})."
                )
            log(f"[ERROR] Nothing was written to {output_dir}.")
            sys.exit(1)
        replace_export(staging, output_dir)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    for name, text in generated_now.items():
        path = os.path.join(run_dir, "params", name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(text)
        log(f"[WARNING] Wrote {path} from the current code, marked as generated.")
    for name, difference in differences.items():
        log(f"[INFO] Checked {name} against the checkpoint ({checked}): max difference {difference:.1e}.")

    log(f"[INFO] Exported {os.path.basename(checkpoint)} to: {output_dir}")
    for name in ("policy.onnx", "policy.pt", *BUNDLE_YAMLS, *OPTIONAL_BUNDLE_YAMLS):
        if name not in missing and os.path.isfile(os.path.join(output_dir, name)):
            log(f"  {name}")
    from_code = [name for name in RUN_CONFIGS if name in not_saved or name in generated]
    if from_code:
        log(f"[WARNING] {' and '.join(from_code)} came from the current code, not from training.")
    if new:
        log("[WARNING] The current code has policy settings the run didn't save; see the warning before the export.")
    if actor_code == ACTOR_CODE_CHANGED:
        log("[WARNING] The code of the policy's network changed since training; see the warning before the export.")
    if robot_model == ROBOT_MODEL_CHANGED:
        log("[WARNING] The robot model changed since training; see the warning before the export.")


if __name__ == "__main__":
    # close() ends the process with the status of a sys.exit in flight, and with 0 otherwise, so an unexpected error
    # is turned into sys.exit(1) first: --share trusts this status before uploading.
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
    finally:
        simulation_app.close()
