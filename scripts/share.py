"""Upload a completed training run to the Hugging Face Hub.

Uploads ``params/agent.yaml``, ``params/env.yaml``, ``params/code_state.yaml`` (for runs that have one) and the
policy from a run directory, together with a generated model card. The checkpoint is always exported first with
``export.py --strict`` into a temporary folder, so an ``exported/policy.onnx`` of unknown vintage is never published
and export's own checks stop the upload: strict turns export's warnings (code that can change the exported policy
changed since training, or the code has policy settings the run didn't save) into stops, and a policy.onnx or
policy.pt whose actions differ from the checkpoint is always one. ``--onnx`` uploads a given file as is instead,
without a policy.pt. Files an earlier share put in the repo that this one doesn't upload are removed, so the repo
never mixes two runs. A run whose
``env.yaml`` or ``agent.yaml`` is missing, or was edited since training, is never uploaded. One generated from the code
by an earlier export is uploaded with a warning; its first line says it was generated.
Requires a prior ``hf auth login`` (or ``HF_TOKEN``).

File paths from the training machine are reduced to file names in the uploaded yaml files; the run directory
itself is not changed.
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile

from cyclotron.code_state import edited_run_configs, hash_run_configs
from cyclotron.hub import checkpoints, infer_task, rehash_uploaded_run_configs, strip_local_paths
from cyclotron.run_config import generated_run_configs

README_TEMPLATE = """---
library_name: asimov
pipeline_tag: robotics
license: bsd-3-clause
---

# {title}

{summary}

- `policy.onnx` is the exported policy for inference, producing joint-position actions.
{policy_pt}- `agent.yaml` records the actor-critic and AMP training settings, including the motion data configuration.
- `env.yaml` records the simulation and locomotion task settings, including observations, actions, commands, and rewards.
{code_state}- Training and evaluation code: [menloresearch/cyclotron](https://github.com/menloresearch/cyclotron).
"""
CODE_STATE_LINE = (
    "- `code_state.yaml` records the code the policy was trained with: the commit, the robot model, the package"
    " versions, a hash of the network's code, the joints and gains the policy's inputs and outputs resolved to, and"
    " (for runs that record them) the sha256 of the uploaded `env.yaml` and `agent.yaml`.\n"
)

POLICY_PT_LINE = "- `policy.pt` is the same policy as TorchScript, for running it from PyTorch.\n"

# Files a share uploads besides the model card; any of them a share leaves out is removed from the repo.
SHARED_FILES = ("policy.onnx", "policy.pt", "agent.yaml", "env.yaml", "code_state.yaml")

DEFAULT_TITLE = "Asimov 1 locomotion policy checkpoint"
DEFAULT_SUMMARY = (
    "Basic velocity-commanded locomotion policy for Asimov 1, trained in Isaac Lab with PPO and adversarial motion"
    " priors (AMP)."
)

EXPORT_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rsl_rl", "export.py")

parser = argparse.ArgumentParser(description="Upload a trained Cyclotron run to the Hugging Face Hub.")
parser.add_argument("run_dir", type=str, help="Run directory, e.g. logs/rsl_rl/<experiment_name>/<run>.")
parser.add_argument("--repo-id", "--repo_id", dest="repo_id", required=True, help="Target repo, e.g. user/name.")
parser.add_argument("--title", type=str, default=DEFAULT_TITLE, help="Title of the model card.")
parser.add_argument(
    "--summary",
    type=str,
    default=DEFAULT_SUMMARY,
    help="Paragraph shown under the model card title, e.g. describing the training method.",
)
parser.add_argument(
    "--onnx",
    type=str,
    default=None,
    help="Upload this ONNX file as is, instead of exporting the checkpoint.",
)
parser.add_argument(
    "--checkpoint",
    type=str,
    default=None,
    help=(
        "Checkpoint to export and upload: a filename inside run_dir, or a full path to one."
        " Defaults to the latest model_*.pt in the run."
    ),
)
parser.add_argument(
    "--task",
    type=str,
    default=None,
    help="Task used to export the policy. Inferred from the experiment name in agent.yaml if omitted.",
)
parser.add_argument("--private", action="store_true", help="Create the repo as private if it does not exist.")
parser.add_argument("--dry-run", "--dry_run", dest="dry_run", action="store_true", help="Print the card and exit.")


def latest_checkpoint(run_dir: str) -> str:
    names = checkpoints(run_dir)
    if not names:
        sys.exit(f"[ERROR] No model_*.pt checkpoints found in: {run_dir}")
    return os.path.join(run_dir, names[-1])


def resolve_checkpoint(run_dir: str, checkpoint: str) -> str:
    """Resolve --checkpoint the same way as train.py and play.py, with run_dir standing in for --load_run."""
    path = os.path.expanduser(checkpoint) if os.sep in checkpoint else os.path.join(run_dir, checkpoint)
    if not os.path.isfile(path):
        sys.exit(f"[ERROR] Checkpoint not found: {path}")
    path = os.path.abspath(path)
    # The run's settings are uploaded with the policy, so a checkpoint from another run would be published with
    # settings it wasn't trained with.
    if os.path.realpath(os.path.dirname(path)) != os.path.realpath(run_dir):
        sys.exit(
            f"[ERROR] {path} is not in {run_dir}.\n--share uploads the run's settings with the policy, so share the"
            " checkpoint's own run:\n"
            f"  ./cyclotron.sh --share {os.path.dirname(path)} --checkpoint {os.path.basename(path)}"
        )
    return path


def export_policy(checkpoint: str, task: str, output_dir: str) -> None:
    print(f"[INFO] Exporting {checkpoint} to ONNX using task {task}")
    # Sharing publishes the policy, so --strict turns the warnings --export alone prints into stops.
    command = [sys.executable, EXPORT_SCRIPT, "--task", task, "--checkpoint", checkpoint, "--strict"]
    if subprocess.run([*command, "--output", output_dir]).returncode != 0:
        sys.exit("[ERROR] ONNX export failed.")
    missing = [name for name in ("policy.onnx", "policy.pt") if not os.path.isfile(os.path.join(output_dir, name))]
    if missing:
        sys.exit(f"[ERROR] The export finished without writing {' and '.join(missing)}.")


def stale_files(repo_files: list[str], uploaded: list[str]) -> list[str]:
    """The files an earlier share put in the repo that this one doesn't upload, e.g. a code_state.yaml or policy.pt
    that would otherwise be published next to a policy they don't belong to."""
    return [name for name in SHARED_FILES if name in repo_files and name not in uploaded]


def main() -> None:
    args_cli = parser.parse_args()
    run_dir = os.path.abspath(os.path.expanduser(args_cli.run_dir))
    onnx_path = os.path.abspath(os.path.expanduser(args_cli.onnx)) if args_cli.onnx else None

    files = {
        "agent.yaml": os.path.join(run_dir, "params", "agent.yaml"),
        "env.yaml": os.path.join(run_dir, "params", "env.yaml"),
    }
    if onnx_path:
        files["policy.onnx"] = onnx_path
    missing = [path for path in files.values() if not os.path.isfile(path)]
    if missing:
        sys.exit("[ERROR] Missing files:\n  " + "\n  ".join(missing))
    generated = generated_run_configs(run_dir)
    if generated:
        print(
            f"[WARNING] {run_dir}/params/{' and '.join(generated)} was written from the code by an earlier --export,"
            " not by training. It is uploaded with the line saying so."
        )
    edited = edited_run_configs(run_dir)
    if edited:
        sys.exit(
            f"[ERROR] {run_dir}/params/{' and '.join(edited)} changed since training: it no longer matches the sha256"
            " the run's code_state.yaml recorded, and --share only publishes the settings a run was trained with."
        )
    code_state = os.path.join(run_dir, "params", "code_state.yaml")
    if os.path.isfile(code_state):
        files["code_state.yaml"] = code_state
    else:
        print("[INFO] The run has no params/code_state.yaml (it was trained before training recorded one).")

    needs_export = not onnx_path
    if needs_export:
        if args_cli.checkpoint:
            checkpoint = resolve_checkpoint(run_dir, args_cli.checkpoint)
        else:
            checkpoint = latest_checkpoint(run_dir)
        try:
            task = args_cli.task or infer_task(run_dir)
        except ValueError as error:
            sys.exit(f"[ERROR] {error}")

    readme = README_TEMPLATE.format(
        title=args_cli.title,
        summary=args_cli.summary,
        policy_pt=POLICY_PT_LINE if needs_export else "",
        code_state=CODE_STATE_LINE if "code_state.yaml" in files else "",
    )

    # Upload copies of the yaml files without the training machine's file paths; the run directory is left as is.
    yamls = {}
    for name, yaml_path in files.items():
        if not name.endswith(".yaml"):
            continue
        with open(yaml_path) as f:
            yamls[name], removed = strip_local_paths(f.read())
        for path in removed:
            print(f"[INFO] Removing local path from {name}: {path}")
    # code_state.yaml records the sha256 of the run's env.yaml and agent.yaml; the Hub copy records the uploaded ones.
    if "code_state.yaml" in yamls:
        recorded = hash_run_configs(os.path.join(run_dir, "params"))
        yamls["code_state.yaml"] = rehash_uploaded_run_configs(yamls["code_state.yaml"], recorded, yamls)

    if args_cli.dry_run:
        if needs_export:
            print(f"[INFO] Would export {checkpoint} to ONNX using task {task}")
            print("[INFO] <export> -> policy.onnx\n[INFO] <export> -> policy.pt")
        for repo_path, local_path in files.items():
            print(f"[INFO] {local_path} -> {repo_path}")
        print(readme)
        return

    try:
        from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi
    except ImportError:
        sys.exit("[ERROR] huggingface_hub is not installed. Install it with: ./cyclotron.sh --install")

    api = HfApi()
    try:
        user = api.whoami()["name"]
    except Exception:
        sys.exit("[ERROR] Not logged in to Hugging Face. Run: hf auth login")
    print(f"[INFO] Logged in as: {user}")

    # Exported into a fresh folder, so only what this export wrote can be uploaded.
    export_dir = tempfile.mkdtemp(prefix="cyclotron-share-")
    try:
        if needs_export:
            export_policy(checkpoint, task, export_dir)
            files["policy.onnx"] = os.path.join(export_dir, "policy.onnx")
            files["policy.pt"] = os.path.join(export_dir, "policy.pt")

        repo_url = api.create_repo(args_cli.repo_id, repo_type="model", private=args_cli.private, exist_ok=True)
        operations = [
            CommitOperationAdd(path_in_repo=name, path_or_fileobj=yamls[name].encode() if name in yamls else path)
            for name, path in files.items()
        ]
        operations.append(CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=readme.encode()))
        for name in stale_files(api.list_repo_files(args_cli.repo_id, repo_type="model"), list(files)):
            print(f"[INFO] Removing {name} from the repo: an earlier share uploaded it, and this one doesn't.")
            operations.append(CommitOperationDelete(path_in_repo=name))
        api.create_commit(
            repo_id=args_cli.repo_id,
            repo_type="model",
            operations=operations,
            commit_message=f"Upload Asimov policy from {os.path.basename(run_dir)}",
        )
    finally:
        shutil.rmtree(export_dir, ignore_errors=True)
    print(f"[INFO] Uploaded to: {repo_url}")


if __name__ == "__main__":
    main()
