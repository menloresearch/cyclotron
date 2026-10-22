"""Resolve inputs once, then hand the selected harness local paths only."""

from datetime import datetime, timezone
from pathlib import Path

import yaml

from .harnesses.base import select_harness
from .hub import resolve_input, working_copy
from .provenance import SCHEMA, file_manifest, git_state, read_json, sha256, write_json

# Cyclotron's own default suite, maintained in this repository rather than read from a harness.
DEFAULT_BENCHMARK = "benchmarks/default"


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


def view(root, policy, benchmark=None, harness_name=None, revision=None, options=()):
    harness, dataset = resolve_benchmark(root, benchmark, harness_name)
    model = resolve_input(policy, "model", harness.model_files, revision)
    # The pinned Hub cache is never edited; local folders, the default included, are edited in place.
    directory = dataset["path"]
    if dataset["source"] == "hub":
        directory = working_copy(dataset, harness.workspaces)
    return harness.view(model["path"], str(directory), options)


def run(root, policies, benchmark=None, harness_name=None, revision=None, output=None, options=()):
    harness, dataset = resolve_benchmark(root, benchmark, harness_name)
    inspection = harness.inspect_benchmark(dataset["path"])
    if inspection["errors"]:
        raise ValueError("Invalid benchmark:\n" + "\n".join(inspection["errors"]))
    if inspection["problems"] and "force" not in options:
        raise ValueError("Incompatible benchmark:\n" + "\n".join(inspection["problems"]))
    models = [harness.inspect_policy(resolve_input(word, "model", harness.model_files, revision if index == 0 else None))
              for index, word in enumerate(policies)]
    if not models:
        raise ValueError("Give at least one policy to benchmark")
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
        for item in [dataset, *models]:
            if file_manifest(item["path"]) != item["files"]:
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
    print("HPV's local-source upload hint does not apply; use cyclotron --benchmark submit for provenance checks.")
    print(f"View: ./cyclotron.sh --benchmark view {directory}")
    print(f"Check submission: ./cyclotron.sh --benchmark submit {directory} --dry-run")
    return directory
