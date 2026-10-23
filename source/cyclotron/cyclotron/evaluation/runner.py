"""Resolve inputs once, then hand the selected harness local paths only."""

import os
from datetime import datetime, timezone
from pathlib import Path

import yaml

from .harnesses.base import select_harness
from .hub import editable_copy, is_dataset, local_path, resolve_input
from .provenance import SCHEMA, file_manifest, git_state, read_json, sha256, write_json

# Cyclotron's own default suite, maintained in this repository rather than read from a harness.
DEFAULT_BENCHMARK = "benchmarks/default"
EXPORT_HINT = "./cyclotron.sh --export --checkpoint <run>/model_<n>.pt"


def resolve_benchmark(root, target, harness_name=None):
    adapter = select_harness(root, harness_name)
    if target:
        item = resolve_input(target, "dataset", adapter.dataset_files)
    else:
        item = {"source": "local", "repo_type": "dataset", "path": str(Path(root).resolve() / DEFAULT_BENCHMARK)}
    directory = Path(item["path"])
    if not directory.is_dir() or not (directory / "benchmark.yaml").is_file():
        raise ValueError(f"Not a benchmark folder: {directory}")
    raw = yaml.safe_load((directory / "benchmark.yaml").read_text())
    if not isinstance(raw, dict):
        raise ValueError("benchmark.yaml must contain a mapping")
    # Published benchmarks name their harness in eval.yaml; local folders have none until publication.
    spec = yaml.safe_load((directory / "eval.yaml").read_text()) if (directory / "eval.yaml").is_file() else None
    if spec is not None and not isinstance(spec, dict):
        raise ValueError("eval.yaml must contain a mapping")
    adapter = select_harness(root, harness_name, raw, spec)
    adapter.prepare()
    item["files"] = file_manifest(directory)
    return adapter, item


def find_runs(directory):
    """Training runs below `directory`, newest first: those exported (<run>/exported/ holding
    policy.onnx and env.yaml, as --export writes it) and those not."""
    runs = {}
    for folder, subfolders, files in os.walk(directory):
        folder = Path(folder)
        subfolders[:] = [name for name in subfolders if not name.startswith(".") and name not in ("exported", "node_modules")]
        export = folder / "exported"
        if export.is_dir() or any(name.startswith("model_") and name.endswith(".pt") for name in files):
            runs[folder] = (export / "policy.onnx").is_file() and (export / "env.yaml").is_file()

    def started(run):
        # rsl_rl names a run after the time it started: 2026-10-22_09-30-00[_<run_name>].
        try:
            return datetime.strptime(run.name[:19], "%Y-%m-%d_%H-%M-%S").timestamp()
        except ValueError:
            return run.stat().st_mtime

    ordered = sorted(runs, key=lambda run: (started(run), str(run)), reverse=True)
    return [run for run in ordered if runs[run]], [run for run in ordered if not runs[run]]


def exported_runs(word, directory):
    """The exported policies of a folder of training runs, newest first, after a summary."""
    exported, unexported = find_runs(directory)

    def names(runs):
        shown = [run.relative_to(directory).as_posix() if run != directory else run.name for run in runs[:3]]
        return ", ".join(shown) + (f" and {len(runs) - 3} more" if len(runs) > 3 else "")

    if not exported:
        if unexported:
            raise ValueError(f"No exported runs under {word}: {len(unexported)} not exported ({names(unexported)}); "
                             f"export them with {EXPORT_HINT}")
        raise ValueError(f"{word} is not a policy and holds no exported runs (<run>/exported/ with policy.onnx and env.yaml)")
    total = len(exported) + len(unexported)
    print(f"Found {total} run{'s' * (total != 1)} under {word}: {len(exported)} exported"
          + (f", {len(unexported)} not (export them with {EXPORT_HINT})" if unexported else ""))
    if unexported:
        print(f"  Not exported: {names(unexported)}")
    return [run / "exported" for run in exported]


def view(root, policies, benchmark=None, harness_name=None, revision=None, options=()):
    harness, dataset = resolve_benchmark(root, benchmark, harness_name)
    folders = []
    for word in policies:
        path = local_path(word, policy=True)
        if path is None:
            # --revision pins the first Hub policy only, as for --benchmark.
            folders.append(resolve_input(word, "model", harness.model_files, revision)["path"])
            revision = None
        elif path.is_dir() and not any(child.is_file() and child.suffix.lower() == ".onnx" for child in path.iterdir()):
            # Not a policy itself: a folder of training runs, such as logs/rsl_rl.
            folders.extend(str(folder) for folder in exported_runs(word, path))
        else:
            folders.append(str(path))
    # Local folders, the default included, are edited in place. A Hub benchmark opens read-only
    # from the pinned cache; pull copies it into a folder to edit.
    hub = dataset if dataset["source"] == "hub" else None
    if hub:
        reference = f"datasets/{hub['id']}" + (f"@{hub['revision']}" if hub["revision"] != "main" else "")
        print(f"{reference} ({hub['sha'][:12]}) opens read-only; to edit it: ./cyclotron.sh --benchmark pull {reference}")
    return harness.view(folders, dataset["path"], options, hub=hub)


def pull(root, target, destination=None, harness_name=None):
    """Copy a Hub benchmark into benchmarks/<name> (or `destination`) to edit, commit and publish."""
    if not is_dataset(target):
        raise ValueError(f"pull copies a Hub benchmark: give datasets/<org>/<name>[@revision], not {target}")
    harness = select_harness(root, harness_name)
    dataset = resolve_input(target, "dataset", harness.dataset_files)
    if not (Path(dataset["path"]) / "benchmark.yaml").is_file():
        raise ValueError(f"datasets/{dataset['id']} is not a benchmark: it has no benchmark.yaml")
    return editable_copy(dataset, destination or harness.benchmarks / dataset["id"].split("/")[1])


def run(root, policies, benchmark=None, harness_name=None, revision=None, output=None, options=()):
    harness, dataset = resolve_benchmark(root, benchmark, harness_name)
    inspection = harness.inspect_benchmark(dataset["path"])
    if inspection["errors"]:
        raise ValueError("Invalid benchmark:\n" + "\n".join(inspection["errors"]))
    if inspection["problems"] and "force" not in options:
        raise ValueError("Incompatible benchmark:\n" + "\n".join(inspection["problems"]))
    models = []
    for word in policies:
        item = resolve_input(word, "model", harness.model_files, revision)
        if item.get("source") == "hub":
            # --revision pins the first Hub policy only, as for --view.
            revision = None
        models.append(harness.inspect_policy(item))
    directory = Path(output).expanduser().resolve() if output else harness.results / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    directory.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema": SCHEMA, "status": "running", "started_at": datetime.now(timezone.utc).isoformat(),
        "cyclotron": git_state(root), "harness": harness.state(), "benchmark": dataset, "policies": models,
        "options": list(options), "inspection": inspection,
    }
    sidecar = directory / "cyclotron.json"
    write_json(sidecar, manifest)
    try:
        harness.run([str(Path(model["path"]) / model["primary"]) for model in models], dataset["path"], directory, options)
        native = harness.normalize_result(read_json(directory / "result.json"))
        errors = harness.binding_errors(native, manifest)
        # A policy records only the files the harness loads; the benchmark, its whole folder.
        for item, names in [(dataset, None), *((model, list(model["files"])) for model in models)]:
            try:
                changed = file_manifest(item["path"], names) != item["files"]
            except ValueError:
                changed = True
            if changed:
                errors.append(f"Input changed during execution: {item['path']}")
        if harness.state() != manifest["harness"] or git_state(root) != manifest["cyclotron"]:
            errors.append("Code checkout changed during execution")
        manifest.update(status="complete", result_sha256=sha256(directory / "result.json"), binding_errors=errors,
                        finished_at=datetime.now(timezone.utc).isoformat())
        if errors:
            manifest["status"] = "invalid"
            raise ValueError("\n".join(errors))
    except BaseException:
        if manifest["status"] == "running":
            manifest["status"] = "failed"
        raise
    finally:
        write_json(sidecar, manifest)
    print(f"\nCyclotron result: {directory}")
    print(f"View: ./cyclotron.sh --benchmark view {directory}")
    print(f"Check submission: ./cyclotron.sh --benchmark submit {directory} --dry-run")
    return directory
