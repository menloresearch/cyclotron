"""Upload a completed training run to the Hugging Face Hub.

Uploads ``params/agent.yaml``, ``params/env.yaml`` and ``exported/policy.onnx`` from a run directory,
together with a generated model card. If the run has no ONNX export yet, the latest checkpoint is exported
first with ``export.py``. Requires a prior ``hf auth login`` (or ``HF_TOKEN``).

File paths from the training machine are reduced to file names in the uploaded yaml files; the run directory
itself is not changed.
"""

import argparse
import os
import re
import subprocess
import sys

from cyclotron.hub import infer_task, strip_local_paths

README_TEMPLATE = """---
library_name: asimov
pipeline_tag: robotics
license: bsd-3-clause
---

# {title}

{summary}

- `policy.onnx` is the exported policy for inference, producing joint-position actions.
- `agent.yaml` records the actor-critic and AMP training settings, including the motion data configuration.
- `env.yaml` records the simulation and locomotion task settings, including observations, actions, commands, and rewards.
- Training and evaluation code: [menloresearch/cyclotron](https://github.com/menloresearch/cyclotron).
"""

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
    help="Path to the ONNX policy. Defaults to <run_dir>/exported/policy.onnx, exported automatically if missing.",
)
parser.add_argument(
    "--checkpoint",
    type=str,
    default=None,
    help=(
        "Checkpoint to export and upload: a full path to a .pt file, or a filename inside run_dir."
        " Defaults to the latest model_*.pt in the run if policy.onnx is missing."
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
    checkpoints = [f for f in os.listdir(run_dir) if re.fullmatch(r"model_\d+\.pt", f)]
    if not checkpoints:
        sys.exit(f"[ERROR] No model_*.pt checkpoints found in: {run_dir}")
    return os.path.join(run_dir, max(checkpoints, key=lambda f: int(f[len("model_") : -len(".pt")])))


def resolve_checkpoint(run_dir: str, checkpoint: str) -> str:
    """Resolve --checkpoint the same way as train.py and play.py, with run_dir standing in for --load_run."""
    path = os.path.expanduser(checkpoint) if os.sep in checkpoint else os.path.join(run_dir, checkpoint)
    if not os.path.isfile(path):
        sys.exit(f"[ERROR] Checkpoint not found: {path}")
    return os.path.abspath(path)


def export_onnx(checkpoint: str, task: str) -> None:
    print(f"[INFO] Exporting {checkpoint} to ONNX using task {task}")
    command = [sys.executable, EXPORT_SCRIPT, "--task", task, "--checkpoint", checkpoint]
    if subprocess.run(command).returncode != 0:
        sys.exit("[ERROR] ONNX export failed.")


def main() -> None:
    args_cli = parser.parse_args()
    run_dir = os.path.abspath(os.path.expanduser(args_cli.run_dir))
    onnx_path = os.path.abspath(os.path.expanduser(args_cli.onnx)) if args_cli.onnx else None

    files = {
        "agent.yaml": os.path.join(run_dir, "params", "agent.yaml"),
        "env.yaml": os.path.join(run_dir, "params", "env.yaml"),
        "policy.onnx": onnx_path or os.path.join(run_dir, "exported", "policy.onnx"),
    }
    missing = [path for name, path in files.items() if name != "policy.onnx" and not os.path.isfile(path)]
    if onnx_path and not os.path.isfile(onnx_path):
        missing.append(onnx_path)
    if missing:
        sys.exit("[ERROR] Missing files:\n  " + "\n  ".join(missing))

    needs_export = not onnx_path and (args_cli.checkpoint or not os.path.isfile(files["policy.onnx"]))
    if needs_export:
        if args_cli.checkpoint:
            checkpoint = resolve_checkpoint(run_dir, args_cli.checkpoint)
        else:
            checkpoint = latest_checkpoint(run_dir)
        try:
            task = args_cli.task or infer_task(run_dir)
        except ValueError as error:
            sys.exit(f"[ERROR] {error}")
        # export.py writes next to the checkpoint, so upload from there.
        files["policy.onnx"] = os.path.join(os.path.dirname(checkpoint), "exported", "policy.onnx")

    readme = README_TEMPLATE.format(title=args_cli.title, summary=args_cli.summary)

    # Upload copies of the yaml files without the training machine's file paths; the run directory is left as is.
    yamls = {}
    for name in ("agent.yaml", "env.yaml"):
        with open(files[name]) as f:
            yamls[name], removed = strip_local_paths(f.read())
        for path in removed:
            print(f"[INFO] Removing local path from {name}: {path}")

    if args_cli.dry_run:
        if needs_export:
            print(f"[INFO] Would export {checkpoint} to ONNX using task {task}")
        for repo_path, local_path in files.items():
            print(f"[INFO] {local_path} -> {repo_path}")
        print(readme)
        return

    try:
        from huggingface_hub import CommitOperationAdd, HfApi
    except ImportError:
        sys.exit("[ERROR] huggingface_hub is not installed. Install it with: ./cyclotron.sh --install")

    api = HfApi()
    try:
        user = api.whoami()["name"]
    except Exception:
        sys.exit("[ERROR] Not logged in to Hugging Face. Run: hf auth login")
    print(f"[INFO] Logged in as: {user}")

    if needs_export:
        export_onnx(checkpoint, task)

    repo_url = api.create_repo(args_cli.repo_id, repo_type="model", private=args_cli.private, exist_ok=True)
    operations = [
        CommitOperationAdd(path_in_repo=name, path_or_fileobj=yamls[name].encode() if name in yamls else path)
        for name, path in files.items()
    ]
    operations.append(CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=readme.encode()))
    api.create_commit(
        repo_id=args_cli.repo_id,
        repo_type="model",
        operations=operations,
        commit_message=f"Upload Asimov policy from {os.path.basename(run_dir)}",
    )
    print(f"[INFO] Uploaded to: {repo_url}")


if __name__ == "__main__":
    main()
