"""Hub eval-results serialization and optimistic, provenance-checked submission."""

import math

import yaml

from .harnesses.base import select_harness
from .hub import download, hub_api, on_main, read_repo_text, commit_files
from .provenance import core_blockers, read_json, result_dir


class _Dumper(yaml.SafeDumper):
    def ignore_aliases(self, data):
        return True


class _Quoted(str):
    pass


_Dumper.add_representer(_Quoted, lambda dumper, value: dumper.represent_scalar("tag:yaml.org,2002:str", value, style='"'))


def eval_path(dataset_id):
    return f".eval_results/{dataset_id.replace('/', '__')}.yaml"


def merge_results(existing, entries):
    old = yaml.safe_load(existing) if existing is not None else []
    if not isinstance(old, list):
        raise ValueError("Existing eval-results must be a YAML list; refusing to overwrite")
    def identity(entry):
        if not isinstance(entry, dict) or not isinstance(entry.get("dataset"), dict):
            raise ValueError("Malformed existing eval-results entry")
        dataset = entry["dataset"]
        if not isinstance(dataset.get("id"), str) or not isinstance(dataset.get("task_id"), str) or "value" not in entry:
            raise ValueError("Malformed existing eval-results entry")
        return dataset["id"], dataset["task_id"]
    keys = {identity(entry) for entry in entries}
    merged = [entry for entry in old if identity(entry) not in keys] + entries
    for entry in merged:
        if "date" in entry:
            entry["date"] = _Quoted(str(entry["date"]))
    return "# Evaluation results submitted by cyclotron.\n" + yaml.dump(merged, Dumper=_Dumper, sort_keys=False, width=100000)


def build_entries(run, manifest, key, policy):
    dataset = manifest["benchmark"]
    entries = []
    for task in run.get("tasks", {}).get(key, []):
        value = task.get("value")
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"Non-finite or non-numeric score for {task.get('taskId')}")
        notes = [f"metric={task.get('metric')}", f"tests={task.get('tests')}", f"runs={task.get('cells')}"]
        if task.get("ci95"):
            notes.append(f"95%CI={task['ci95']}")
        notes.extend((f"cyclotron@{manifest['cyclotron']['commit'][:12]}",
                      f"{manifest['harness']['name']}@{manifest['harness']['commit'][:12]}",
                      f"protocol={run['provenance']['hpv']['protocol']}",
                      f"policy@{policy['sha'][:12]}"))
        robot = run.get("provenance", {}).get("asimov1", {}).get("commit")
        if robot:
            notes.append(f"asimov-1@{robot[:12]}")
        entries.append({"dataset": {"id": dataset["id"], "task_id": task["taskId"], "revision": dataset["sha"]},
                        "value": value, "date": run["generatedAt"], "notes": " ".join(notes)})
    if not entries:
        raise ValueError(f"No scores to submit for {key}")
    return entries


def submit(root, directory, pr=False, dry_run=False, api=None):
    directory = result_dir(directory)
    if not (directory / "cyclotron.json").is_file():
        raise ValueError("This result has no cyclotron provenance; native/browser results are viewable but not submittable")
    manifest = read_json(directory / "cyclotron.json")
    harness = select_harness(root, manifest.get("harness", {}).get("name"))
    run = harness.normalize_result(read_json(directory / "result.json"))
    reasons = core_blockers(manifest, directory / "result.json")
    if manifest.get("inspection") and manifest.get("policies"):
        reasons.extend(harness.submit_blockers(run, manifest))
    else:
        reasons.append("Missing recorded inspection or policies")
    if reasons:
        raise ValueError("Cannot submit:\n" + "\n".join(f"  - {reason}" for reason in dict.fromkeys(reasons)))
    api = api or hub_api()
    for item in [manifest["benchmark"], *manifest["policies"]]:
        if item.get("endpoint") != api.endpoint.rstrip("/"):
            raise ValueError("The recorded Hub endpoint differs from the configured endpoint")
        checked = download({"id": item["id"], "repo_type": item["repo_type"], "revision": item["sha"]},
                           harness.dataset_files if item["repo_type"] == "dataset" else harness.model_files, api=api)
        if checked["sha"] != item["sha"] or checked["files"] != item["files"]:
            raise ValueError(f"Recorded input differs from pinned Hub bytes: {item['id']}")
    dataset = manifest["benchmark"]
    # A pending PR's revision (refs/pr/N) can be pinned too, but its content was never merged.
    if not on_main(api, dataset["id"], dataset["sha"]):
        raise ValueError(f"datasets/{dataset['id']}@{dataset['sha']} is not in the dataset's main history "
                         "(an unmerged PR revision?); rerun against a merged revision, such as a release tag")
    published = read_repo_text(api, dataset["id"], "eval.yaml", dataset["sha"], "dataset")
    spec = yaml.safe_load(published or "")
    if not isinstance(spec, dict) or spec.get("evaluation_framework") != "cyclotron":
        raise ValueError("The pinned dataset must declare evaluation_framework: cyclotron in eval.yaml")
    tasks = {task["id"]: task for task in spec.get("tasks", [])}
    proposals = []
    repos = set()
    for index, policy in enumerate(manifest["policies"]):
        if policy["id"] in repos:
            raise ValueError("Submit one evaluated revision per model repo; this run contains repeated model repos")
        repos.add(policy["id"])
        entries = build_entries(run, manifest, f"m{index}", policy)
        for entry in entries:
            task = tasks.get(entry["dataset"]["task_id"])
            if task is None:
                raise ValueError("Result task IDs are not declared in the pinned dataset's eval.yaml")
            declared = (task.get("harness") or {}).get("name")
            if declared not in (None, harness.name):
                raise ValueError(f"eval.yaml says {task['id']} is scored by {declared}, but this run used {harness.name}")
        path = eval_path(dataset["id"])
        old = read_repo_text(api, policy["id"], path, policy["sha"])
        content = merge_results(old, entries)
        proposals.append((policy, path, content))
    for policy, path, content in proposals:
        if dry_run:
            print(f"\nWould submit {policy['id']}@{policy['sha']}:{path}\n{content}")
        else:
            result = commit_files(api, policy["id"], {path: content}, policy["sha"],
                                  f"Add {dataset['id']} benchmark results", pr=pr)
            print(f"Submitted {policy['id']}: {result.pr_url or result.commit_url}")
    return proposals
