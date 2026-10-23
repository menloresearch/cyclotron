"""Evaluation orchestration can be tested without Isaac Lab, Node or the network."""

import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml

from cyclotron.evaluation import hub, provenance, runner
from cyclotron.evaluation.harnesses.base import select_harness
from cyclotron.evaluation.harnesses.hpv import HPV
from cyclotron.evaluation.cli import main, parse_run
from cyclotron.evaluation.eval_results import build_entries, merge_results, submit

ROOT = Path(__file__).resolve().parents[1]
SHA = "a" * 40


@pytest.mark.parametrize("text,kind,repo,revision", [
    ("org/model", "model", "org/model", "main"),
    ("org/model@refs/pr/2", "model", "org/model", "refs/pr/2"),
    ("datasets/org/suite@v1", "dataset", "org/suite", "v1"),
    ("https://huggingface.co/datasets/org/suite/tree/v2", "dataset", "org/suite", "v2"),
])
def test_references(text, kind, repo, revision):
    assert hub.parse_reference(text) == {"id": repo, "repo_type": kind, "revision": revision}


@pytest.mark.parametrize("text", ["../org/repo", "https://other.example/org/repo", "org/../repo", "org/repo@"])
def test_bad_references(text):
    with pytest.raises(ValueError):
        hub.parse_reference(text)


def test_local_run_and_conflicting_revision(tmp_path):
    (tmp_path / "exported").mkdir()
    assert hub.local_path(str(tmp_path), policy=True) == tmp_path / "exported"
    with pytest.raises(ValueError, match="conflicting"):
        hub.parse_reference("org/repo@v1", revision="v2")
    with pytest.raises(ValueError, match="does not exist"):
        hub.local_path("./missing")


def test_cli_translation_and_input_aliases(tmp_path):
    (tmp_path / "benchmark.yaml").write_text("name: test")
    args, policies, benchmark, words = parse_run([
        "org/first", "--jobs", "2", f"benchmark={tmp_path}", "policy=org/second",
        "--no-open", "--harness", HPV.name, "--", "robots=3", "future=one",
    ])
    assert policies == ["org/first", "org/second"] and args.harness == HPV.name
    assert benchmark == str(tmp_path)
    assert words == ["jobs=2", "robots=3", "no-open", "future=one"]
    assert args.out is None
    with pytest.raises(ValueError, match="only once"):
        parse_run(["org/first", "--out", "one", "out=two"])
    with pytest.raises(ValueError, match="positionals"):
        parse_run(["org/first", "--", "org/unresolved"])
    with pytest.raises(ValueError, match="exactly one"):
        parse_run(["org/first", "datasets/org/suite", "--benchmark", str(tmp_path)])


def test_view_known_options():
    _, policies, _, words = parse_run(["org/first", "--port", "4321", "--no-open"], viewing=True)
    assert policies == ["org/first"] and words == ["port=4321", "no-open"]
    for option in (["jobs=2"], ["--verbose"], ["verbose"], ["--watch"], ["--out", "x"]):
        with pytest.raises(ValueError, match="Run options belong to --benchmark"):
            parse_run(["org/first", *option], viewing=True)


@pytest.mark.parametrize("word", ["org/m@main=x", "https://huggingface.co/org/m/tree/v=1", "--future=one", "--future",
                                  "Robots=3", "datasets/org/suite@v=1"])
def test_only_harness_words_pass_through(word):
    # HPV would take any of these for a Hub input and fetch it itself.
    for argv in (["org/first", word], ["org/first", "--", word]):
        with pytest.raises(ValueError, match="key=value words"):
            parse_run(argv)
    # Inputs whose names contain "=" are given through their aliases, and cyclotron resolves them.
    _, policies, benchmark, words = parse_run(["policy=org/m@main=x", "benchmark=datasets/org/suite@v=1", "future-key=a/b"])
    assert policies == ["org/m@main=x"] and benchmark == "datasets/org/suite@v=1" and words == ["future-key=a/b"]


def test_view_takes_several_policies_and_folders(tmp_path):
    (tmp_path / "runs").mkdir()
    (tmp_path / "suite").mkdir()
    (tmp_path / "suite/benchmark.yaml").write_text("name: test")
    _, policies, benchmark, _ = parse_run(
        ["org/first", str(tmp_path / "runs"), str(tmp_path / "suite"), "policy=org/second", "--revision", "v1"], viewing=True)
    assert policies == ["org/first", str(tmp_path / "runs"), "org/second"]
    assert benchmark == str(tmp_path / "suite")
    with pytest.raises(SystemExit):
        parse_run([str(tmp_path / "suite")], viewing=True)


def test_adapter_registry(monkeypatch):
    monkeypatch.delenv("CYCLOTRON_HARNESS", raising=False)
    assert select_harness(ROOT, benchmark={"harness": {"repo": HPV.repo}}).name == HPV.name
    for raw in ({"harness": {"repo": "https://evil.example/code"}},
                {"harness": {"repo": HPV.repo, "name": "other"}}):
        with pytest.raises(ValueError):
            select_harness(ROOT, benchmark=raw)
    with pytest.raises(ValueError, match="available"):
        select_harness(ROOT, "other")
    # A blank harness block is no declaration; a wrongly typed one is a clean error, not a traceback.
    assert select_harness(ROOT, benchmark={"harness": None}).name == HPV.name
    assert select_harness(ROOT, benchmark={"harness": {"repo": None}}).name == HPV.name
    for raw in ({"harness": "hpv"}, {"harness": {"repo": 1}}, {"harness": {"name": ["hpv"]}}):
        with pytest.raises(ValueError, match="benchmark.yaml's harness"):
            select_harness(ROOT, benchmark=raw)


def test_published_harness_is_selected(monkeypatch):
    monkeypatch.delenv("CYCLOTRON_HARNESS", raising=False)
    def spec(*names):
        return {"tasks": [{"id": f"t{i}", **({"harness": {"name": n}} if n else {})} for i, n in enumerate(names)]}
    legacy = {"harness": {"repo": HPV.repo}}
    assert select_harness(ROOT, spec=spec(HPV.name)).name == HPV.name
    assert select_harness(ROOT, benchmark=legacy, spec=spec(None)).name == HPV.name
    for raw, published in ((legacy, spec("other")), ({}, spec(HPV.name, "other")), ({}, spec("other"))):
        with pytest.raises(ValueError):
            select_harness(ROOT, benchmark=raw, spec=published)
    with pytest.raises(ValueError, match="conflicts"):
        select_harness(ROOT, "other", spec=spec(HPV.name))
    block = HPV(ROOT).task_harness({"raw": {"harness": {"protocol": 1}}}, SHA)
    assert block == {"name": HPV.name, "repo": HPV.repo, "protocol": 1, "commit": SHA}


def test_manifest_rejects_symlinks_and_missing_files(tmp_path):
    (tmp_path / "a").write_text("hello")
    assert provenance.file_manifest(tmp_path) == {"a": hashlib.sha256(b"hello").hexdigest()}
    (tmp_path / "link").symlink_to(tmp_path / "a")
    with pytest.raises(ValueError, match="symlinks"):
        provenance.file_manifest(tmp_path)
    with pytest.raises(ValueError, match="Missing"):
        provenance.file_manifest(tmp_path, ["missing"])


def test_untracked_files_leave_the_harness_checkout_clean(monkeypatch):
    # An untracked logs/ (HPV's viewer writes one) is all `git status` reports here.
    def check_output(command, **kwargs):
        if "rev-parse" in command:
            return SHA
        return "" if "--untracked-files=no" in command else "?? logs/\n"
    monkeypatch.setattr(provenance.subprocess, "check_output", check_output)
    assert provenance.git_state(ROOT) == {"commit": SHA, "dirty": True}
    harness = HPV(ROOT)
    monkeypatch.setattr(harness, "pin", lambda: SHA)
    assert harness.state()["dirty"] is False


def test_untracked_files_in_a_submodule_leave_cyclotron_clean(tmp_path):
    def git(*args, cwd=tmp_path / "root"):
        return subprocess.run(["git", "-c", "user.name=test", "-c", "user.email=test@example.com", "-c", "commit.gpgsign=false",
                               *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()
    # A checkout like cyclotron's: a pinned submodule (a gitlink) beside its own files.
    (tmp_path / "root/sub").mkdir(parents=True)
    git("init", "-q", cwd=tmp_path / "root/sub")
    (tmp_path / "root/sub/tracked").write_text("a")
    git("add", "tracked", cwd=tmp_path / "root/sub")
    git("commit", "-qm", "one", cwd=tmp_path / "root/sub")
    pin = git("rev-parse", "HEAD", cwd=tmp_path / "root/sub")
    git("init", "-q")
    (tmp_path / "root/own").write_text("r")
    git("add", "own")
    git("update-index", "--add", "--cacheinfo", f"160000,{pin},sub")
    git("commit", "-qm", "root")
    def dirty():
        return provenance.git_state(tmp_path / "root")["dirty"]
    assert dirty() is False
    (tmp_path / "root/sub/logs").mkdir()
    (tmp_path / "root/sub/logs/viewer.log").write_text("untracked")
    assert dirty() is False
    (tmp_path / "root/sub/tracked").write_text("edited")
    assert dirty() is True
    git("checkout", "-q", "tracked", cwd=tmp_path / "root/sub")
    git("commit", "-q", "--allow-empty", "-m", "two", cwd=tmp_path / "root/sub")
    assert dirty() is True
    git("checkout", "-q", pin, cwd=tmp_path / "root/sub")
    assert dirty() is False
    (tmp_path / "root/untracked").write_text("new")
    assert dirty() is True


@pytest.fixture
def fake_hub(tmp_path, monkeypatch):
    monkeypatch.setenv("CYCLOTRON_CACHE_DIR", str(tmp_path / "cache"))
    source = tmp_path / "source"
    source.mkdir()
    contents = {"policy.onnx": b"model", "params/env.yaml": b"{}", "params/tracking_policy.json": b"{}"}
    siblings = []
    for name, data in contents.items():
        file = source / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(data)
        blob = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
        siblings.append(SimpleNamespace(rfilename=name, blob_id=blob, lfs=None))
    api = Mock(endpoint="https://huggingface.co", token=None)
    api.repo_info.return_value = SimpleNamespace(id="org/model", sha=SHA, siblings=siblings)
    fetch = Mock(return_value=str(source))
    monkeypatch.setattr("huggingface_hub.snapshot_download", fetch)
    return api, fetch


def test_cache_pins_verifies_and_materializes(fake_hub):
    api, fetch = fake_hub
    reference = {"id": "org/model", "revision": "main", "repo_type": "model"}
    item = hub.download(reference, HPV.model_files, api)
    assert item["sha"] == SHA
    assert fetch.call_args.kwargs["revision"] == SHA
    # HPV labels a policy by its folder name: the repo, as HPV's own Hub download did.
    assert Path(item["path"]).name == "org__model" and Path(item["path"]).parent.name == SHA
    assert not (Path(item["path"]) / "policy.onnx").is_symlink()
    assert "params/tracking_policy.json" in item["files"]
    assert hub.download(reference, HPV.model_files, api)["files"] == item["files"]
    fetch.assert_called_once()
    (Path(item["path"]) / "policy.onnx").write_bytes(b"edited")
    with pytest.raises(ValueError, match="verify pinned"):
        hub.download(reference, HPV.model_files, api)


def test_cache_keeps_the_hubs_repo_id_and_literal_filenames(fake_hub):
    from huggingface_hub.utils import filter_repo_objects

    api, fetch = fake_hub
    source = Path(fetch.return_value)
    # snapshot_download glob-matches allow_patterns; these names must still download as themselves.
    for name in ("policy[1].onnx", "policy*?.onnx"):
        (source / name).write_bytes(b"other")
        blob = hashlib.sha1(b"blob 5\0other").hexdigest()
        api.repo_info.return_value.siblings.append(SimpleNamespace(rfilename=name, blob_id=blob, lfs=None))
    # Provenance records the Hub's spelling of the repo, not the user's.
    item = hub.download({"id": "ORG/Model", "revision": "main", "repo_type": "model"}, HPV.model_files, api)
    assert item["id"] == "org/model" and Path(item["path"]).name == "org__model"
    names = ["policy.onnx", "policy1.onnx", "policyab.onnx", "policy[1].onnx", "policy*?.onnx"]
    selected = list(filter_repo_objects(names, allow_patterns=fetch.call_args.kwargs["allow_patterns"]))
    assert selected == ["policy.onnx", "policy[1].onnx", "policy*?.onnx"]
    assert {"policy[1].onnx", "policy*?.onnx"} <= set(item["files"])


def test_endpoint_caches_do_not_collide(fake_hub):
    api, fetch = fake_hub
    reference = {"id": "org/model", "revision": "main", "repo_type": "model"}
    first = hub.download(reference, HPV.model_files, api)
    api.endpoint = "https://private.example"
    second = hub.download(reference, HPV.model_files, api)
    assert first["path"] != second["path"]
    assert fetch.call_count == 2


def test_failed_download_leaves_no_completed_cache(fake_hub):
    api, fetch = fake_hub
    fetch.side_effect = RuntimeError("interrupted")
    with pytest.raises(RuntimeError):
        hub.download({"id": "org/model", "revision": "main", "repo_type": "model"}, HPV.model_files, api)
    assert not list(hub.cache_root().rglob(".cyclotron-cache.json"))


def test_pull_copies_a_hub_benchmark_to_edit(tmp_path):
    source = tmp_path / "cache/org__suite"
    (source / "data/flat").mkdir(parents=True)
    for name in ("benchmark.yaml", "README.md", "eval.yaml", "data/flat/test.jsonl", ".cyclotron-cache.json", ".hf-dataset.json"):
        (source / name).write_text(name)
    item = {"path": str(source), "id": "org/suite", "sha": SHA, "revision": "main", "source": "hub",
            "endpoint": "https://huggingface.co", "files": {}}
    pulled = hub.editable_copy(item, tmp_path / "benchmarks/suite")
    # The cache's records stay behind (HPV's would make the copy read-only); publication builds eval.yaml again.
    assert provenance.file_names(pulled) == ["README.md", "benchmark.yaml", "data/flat/test.jsonl"]
    assert [path.name for path in pulled.glob(".*")] == [".cyclotron-origin.json"]
    assert provenance.read_json(pulled / ".cyclotron-origin.json") == {"origin": {
        "id": "org/suite", "endpoint": "https://huggingface.co", "revision": "main", "sha": SHA}}
    (pulled / "benchmark.yaml").write_text("edited")
    with pytest.raises(ValueError, match="already exists"):
        hub.editable_copy({**item, "sha": "b" * 40}, pulled)
    assert (pulled / "benchmark.yaml").read_text() == "edited"
    assert (source / "benchmark.yaml").read_text() == "benchmark.yaml"


def test_pull_puts_a_hub_benchmark_in_benchmarks(monkeypatch, tmp_path):
    (tmp_path / "benchmark.yaml").write_text("name: suite\n")
    item = {"path": str(tmp_path), "id": "org/suite"}
    monkeypatch.setattr(runner, "resolve_input", lambda word, kind, select: item if word == "datasets/org/suite@v1" else None)
    copy = Mock()
    monkeypatch.setattr(runner, "editable_copy", copy)
    runner.pull(ROOT, "datasets/org/suite@v1")
    copy.assert_called_once_with(item, ROOT / "benchmarks/suite")
    runner.pull(ROOT, "datasets/org/suite@v1", tmp_path / "elsewhere")
    assert copy.call_args.args == (item, tmp_path / "elsewhere")
    with pytest.raises(ValueError, match="pull copies a Hub benchmark"):
        runner.pull(ROOT, str(tmp_path))
    (tmp_path / "benchmark.yaml").unlink()
    with pytest.raises(ValueError, match="not a benchmark"):
        runner.pull(ROOT, "datasets/org/suite@v1")


def test_pull_is_a_benchmark_action(monkeypatch):
    pull = Mock()
    monkeypatch.setattr(runner, "pull", pull)
    main(ROOT, ["pull", "datasets/org/suite", "./mine"])
    pull.assert_called_once_with(ROOT, "datasets/org/suite", "./mine", None)


def test_an_empty_cache_variable_is_unset(tmp_path, monkeypatch):
    from cyclotron import robot_model

    monkeypatch.setenv("CYCLOTRON_CACHE_DIR", "")
    monkeypatch.setenv("XDG_CACHE_HOME", "")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path / "..")
    assert hub.cache_root() == (tmp_path / ".cache" / "cyclotron").resolve()
    assert hub.cache_root() == Path(robot_model.cache_dir()).resolve()


def test_a_lone_onnx_records_only_the_files_the_viewer_loads(tmp_path, monkeypatch):
    # ~/Downloads/policy.onnx: the rest of the folder is neither hashed nor checked for symlinks.
    (tmp_path / "policy.onnx").write_bytes(b"model")
    (tmp_path / "env.yaml").write_text("{}")
    (tmp_path / "unrelated.zip").write_bytes(b"x")
    (tmp_path / "link").symlink_to(tmp_path / "unrelated.zip")
    (tmp_path / "other").mkdir()
    (tmp_path / "other/file").write_text("y")
    harness = HPV(ROOT)
    monkeypatch.setattr(harness, "inspect", lambda kind, path: {
        "errors": [], "policyDir": str(tmp_path), "primary": "policy.onnx", "hashes": {}})
    policy = harness.inspect_policy({"source": "local", "path": str(tmp_path / "policy.onnx")})
    assert set(policy["files"]) == {"policy.onnx", "env.yaml"}


@pytest.fixture
def recorded_result(tmp_path):
    raw = {"name": "suite", "version": 1, "harness": {"repo": HPV.repo, "protocol": 1}}
    cell = {"cellId": "stand:0", "testId": "stand", "repeat": 0, "seed": 123}
    policy = {"source": "hub", "id": "org/model", "sha": SHA, "repo_type": "model",
              "endpoint": "https://huggingface.co", "path": str(tmp_path), "files": {},
              "primary": "policy.onnx", "hpv_hashes": {"onnx": "hash", "envYaml": "env"}}
    manifest = {"schema": provenance.SCHEMA, "status": "complete", "options": [],
                "cyclotron": {"commit": SHA, "dirty": False},
                "harness": {"name": HPV.name, "commit": SHA, "dirty": False},
                "policies": [policy], "benchmark": {**policy, "id": "org/suite", "repo_type": "dataset"},
                "inspection": {"raw": raw, "errors": [], "problems": [], "cells": [cell],
                               "tests": [{"id": "stand"}], "tasks": [{"id": "score_v1"}]}}
    run = {"schema": "hpv-benchmark-result/1", "generatedAt": "2026-10-15T00:00:00Z",
           "provenance": {"hpv": {"commit": SHA, "dirty": False, "protocol": 1}},
           "benchmark": {"raw": raw, "problems": []}, "overrides": {}, "shard": None,
           "policies": [{"key": "m0", "primary": "policy.onnx", "files": policy["hpv_hashes"]}],
           "cells": [{**cell, "policyKey": "m0", "error": None}], "expectedCells": 1,
           "tests": [{"id": "stand"}], "tasks": {"m0": [{"taskId": "score_v1", "value": 0.9, "metric": "success", "tests": 1, "cells": 1}]}}
    return run, manifest


def test_full_suite_blockers(recorded_result):
    run, manifest = recorded_result
    harness = HPV(ROOT)
    assert not harness.submit_blockers(run, manifest)
    for mutation in (lambda r: r["cells"].append(copy.deepcopy(r["cells"][0])),
                     lambda r: r["cells"][0].update(seed=9),
                     lambda r: r["cells"][0].update(error="failed"),
                     lambda r: r.update(tests=[]),
                     lambda r: r.update(overrides={"repeats": 1}),
                     lambda r: r["policies"][0].update(files={}),
                     lambda r: r["provenance"]["hpv"].update(dirty=True),
                     lambda r: r["provenance"]["hpv"].update(protocol=2),
                     lambda r: r.update(shard={"index": 1, "count": 2})):
        changed = copy.deepcopy(run)
        mutation(changed)
        assert harness.submit_blockers(changed, manifest)
    # No protocol on either side is a blocker, not a KeyError when the entries are built.
    unversioned = copy.deepcopy(manifest)
    del unversioned["inspection"]["raw"]["harness"]["protocol"]
    changed = copy.deepcopy(run)
    del changed["provenance"]["hpv"]["protocol"]
    assert "Benchmark protocol differs from the HPV runtime protocol" in harness.submit_blockers(changed, unversioned)
    manifest["options"] = ["force"]
    assert harness.submit_blockers(run, manifest)


def test_result_binding(recorded_result, tmp_path):
    run, manifest = recorded_result
    path = tmp_path / "result.json"
    provenance.write_json(path, run)
    manifest["result_sha256"] = provenance.sha256(path)
    assert not provenance.core_blockers(manifest, path)
    path.write_text("{}")
    assert provenance.core_blockers(manifest, path)
    manifest["status"] = "failed"
    manifest["cyclotron"]["dirty"] = True
    manifest["benchmark"]["source"] = "local"
    assert len(provenance.core_blockers(manifest, path)) >= 4


def test_eval_yaml_merge_and_quoted_date(recorded_result):
    run, manifest = recorded_result
    entries = build_entries(run, manifest, "m0", manifest["policies"][0])
    existing = yaml.safe_dump([
        {"dataset": {"id": "org/suite", "task_id": "score_v1"}, "value": 0},
        {"dataset": {"id": "other/suite", "task_id": "score_v1"}, "value": 0.2},
    ])
    text = merge_results(existing, entries)
    assert 'date: "2026-10-15T00:00:00Z"' in text
    assert len(yaml.safe_load(text)) == 2
    assert yaml.safe_load(text)[-1]["value"] == 0.9
    assert "cyclotron@" in text
    for malformed in ("oops: true", "- nope"):
        with pytest.raises(ValueError):
            merge_results(malformed, entries)
    # An existing but empty file holds no entries to keep.
    for empty in ("", "# cleared\n"):
        assert len(yaml.safe_load(merge_results(empty, entries))) == 1
    run["tasks"]["m0"][0]["value"] = float("nan")
    with pytest.raises(ValueError, match="Non-finite"):
        build_entries(run, manifest, "m0", manifest["policies"][0])


@pytest.mark.parametrize("harness", ["", "\n  harness: {name: humanoid-policy-viewer}", "\n  harness: {name: other-harness}"])
def test_submit_dry_run_pinned_inputs_and_no_writes(recorded_result, tmp_path, monkeypatch, harness):
    from cyclotron.evaluation import eval_results

    run, manifest = recorded_result
    provenance.write_json(tmp_path / "result.json", run)
    manifest["result_sha256"] = provenance.sha256(tmp_path / "result.json")
    provenance.write_json(tmp_path / "cyclotron.json", manifest)
    monkeypatch.setattr(eval_results, "download", lambda ref, *a, **kw: {"sha": ref["revision"], "files": {}})
    read = Mock(side_effect=[f"evaluation_framework: cyclotron\ntasks:\n- id: score_v1{harness}", None])
    monkeypatch.setattr(eval_results, "read_repo_text", read)
    commit = Mock()
    monkeypatch.setattr(eval_results, "commit_files", commit)
    api = Mock(endpoint="https://huggingface.co")
    api.list_repo_commits.return_value = [SimpleNamespace(commit_id="b" * 40), SimpleNamespace(commit_id=SHA)]
    if "other-harness" in harness:
        # A task's results only count from the harness its eval.yaml names.
        with pytest.raises(ValueError, match="scored by other-harness"):
            submit(ROOT, tmp_path, dry_run=True, api=api)
    else:
        assert len(submit(ROOT, tmp_path, dry_run=True, api=api)) == 1
        assert read.call_args.args[3] == SHA
    commit.assert_not_called()


def test_submit_requires_a_merged_dataset_revision(recorded_result, tmp_path, monkeypatch):
    from cyclotron.evaluation import eval_results

    run, manifest = recorded_result
    provenance.write_json(tmp_path / "result.json", run)
    manifest["result_sha256"] = provenance.sha256(tmp_path / "result.json")
    provenance.write_json(tmp_path / "cyclotron.json", manifest)
    monkeypatch.setattr(eval_results, "download", lambda ref, *a, **kw: {"sha": ref["revision"], "files": {}})
    read, commit = Mock(), Mock()
    monkeypatch.setattr(eval_results, "read_repo_text", read)
    monkeypatch.setattr(eval_results, "commit_files", commit)
    # A pinned refs/pr/N revision resolves to a commit that main does not contain.
    api = Mock(endpoint="https://huggingface.co")
    api.list_repo_commits.return_value = [SimpleNamespace(commit_id="b" * 40)]
    with pytest.raises(ValueError, match="not in the dataset's main history"):
        submit(ROOT, tmp_path, api=api)
    assert api.list_repo_commits.call_args.args[0] == "org/suite"
    assert api.list_repo_commits.call_args.kwargs == {"repo_type": "dataset", "revision": "main"}
    read.assert_not_called()
    commit.assert_not_called()


def test_native_results_cannot_be_adopted(tmp_path):
    (tmp_path / "result.json").write_text("{}")
    with pytest.raises(ValueError, match="no cyclotron provenance"):
        submit(ROOT, tmp_path, dry_run=True)


def test_cli_help_imports_no_training():
    code = "from cyclotron.evaluation.cli import main; import sys; assert 'torch' not in sys.modules; assert 'isaaclab' not in sys.modules"
    result = subprocess.run([sys.executable, "-c", code], env={**__import__("os").environ, "PYTHONPATH": str(ROOT / "source/cyclotron")})
    assert result.returncode == 0


def test_child_args_are_local_and_headless(monkeypatch, tmp_path):
    monkeypatch.setenv("CYCLOTRON_CACHE_DIR", str(tmp_path / "cache"))
    harness = HPV(ROOT)
    execute = Mock()
    monkeypatch.setattr(harness, "execute", execute)
    monkeypatch.setattr(harness, "write_template", lambda: tmp_path / "template")
    monkeypatch.setenv("SSH_CONNECTION", "test")
    monkeypatch.setenv("HF_TOKEN", "hf_secret")
    monkeypatch.setenv("HF_ENDPOINT", "https://huggingface.co")
    harness.view([str(tmp_path / "a"), str(tmp_path / "b")], str(tmp_path / "suite"), [])
    assert execute.call_args.args[0] == ["scripts/hf.mjs", f"policy={tmp_path / 'a'}", f"policy={tmp_path / 'b'}",
                                         f"benchmark={tmp_path / 'suite'}", "no-open"]
    harness.run([str(tmp_path / "policy")], str(tmp_path / "suite"), tmp_path / "out", ["jobs=2"])
    assert f"out={tmp_path / 'out'}" in execute.call_args.args[0]
    env = harness.env()
    # New benchmarks start from the template view() writes and are saved beside the default suite; HPV's
    # debug logs stay out of its checkout.
    assert env["HPV_BENCHMARKS_DIR"] == str(ROOT / "benchmarks")
    assert env["HPV_BENCHMARK_TEMPLATE"] == str(tmp_path / "template")
    assert env["HPV_LOG_DIR"] == str(tmp_path / "cache/harnesses" / HPV.name / "logs")
    assert "HF_TOKEN" not in env
    # Should a Hub ID reach HPV anyway, it finds no saved token and no Hub.
    assert env["HF_TOKEN_PATH"].startswith(str(tmp_path / "cache")) and not Path(env["HF_TOKEN_PATH"]).exists()
    assert env["HF_ENDPOINT"] == "http://127.0.0.1:9"


@pytest.mark.parametrize("moved", [True, False])
def test_moving_the_harness_pin_also_moves_its_robot_model(monkeypatch, moved):
    from cyclotron.evaluation.harnesses import hpv

    harness = HPV(ROOT)
    monkeypatch.setattr(harness, "pin", lambda: SHA)
    monkeypatch.setattr(harness, "checkout", lambda: {"commit": "b" * 40 if moved else SHA, "dirty": False})
    monkeypatch.setattr(hpv.subprocess, "check_output", lambda *a, **kw: "22.0.0")
    calls = []
    monkeypatch.setattr(hpv.subprocess, "run", lambda command, **kw: calls.append((command, kw.get("cwd"))))
    harness.prepare()
    update = ["git", "-C", str(ROOT), "submodule", "update", "--checkout", "--depth", "1", "third_party/" + HPV.name]
    if moved:
        assert calls == [(update, None), (["bash", "scripts/init-asimov-1.sh"], harness.path), (["npm", "ci"], harness.path)]
    else:
        assert all(command[:2] != ["bash", "scripts/init-asimov-1.sh"] and command != update for command, _ in calls)


@pytest.fixture
def run_tree(tmp_path):
    """logs/rsl_rl-like runs: exported and not, nested experiments, and files that are no run."""
    root = tmp_path / "rsl_rl"
    def run(name, *files):
        folder = root / name
        for file in files:
            (folder / file).parent.mkdir(parents=True, exist_ok=True)
            (folder / file).write_text("x")
        return folder
    exported = ("model_10.pt", "exported/policy.onnx", "exported/env.yaml")
    old = run("walk/2026-10-01_10-00-00_first", *exported)
    nested = run("walk/sweep/2026-10-02_10-00-00", *exported)
    undated = run("custom/hand-named", "exported/policy.onnx", "exported/env.yaml")
    unexported = run("walk/2026-10-03_10-00-00", "model_5.pt", "params/env.yaml")
    no_env = run("walk/2026-10-04_10-00-00_no_env", "model_5.pt", "exported/policy.onnx")
    run("misc", "stray.onnx")
    run(".hidden/2026-10-09_10-00-00", *exported)
    # A run name without a timestamp falls back to the folder's mtime: here between the others.
    stamp = datetime(2026, 10, 1, 12).timestamp()
    os.utime(undated, (stamp, stamp))
    return root, [nested, undated, old], [no_env, unexported]


def test_find_runs_newest_first(run_tree):
    root, exported, unexported = run_tree
    assert runner.find_runs(root) == (exported, unexported)
    # A single run folder is its own run.
    assert runner.find_runs(exported[0]) == ([exported[0]], [])


def test_folder_of_runs_summary_and_errors(run_tree, capsys):
    root, exported, unexported = run_tree
    assert runner.exported_runs("logs", root) == [run / "exported" for run in exported]
    out = capsys.readouterr().out
    assert out.startswith("Found 5 runs under logs: 3 exported, 2 not (export them with ./cyclotron.sh --export")
    assert "Not exported: walk/2026-10-04_10-00-00_no_env, walk/2026-10-03_10-00-00\n" in out
    with pytest.raises(ValueError, match="No exported runs under walk: 1 not exported"):
        runner.exported_runs("walk", unexported[1])
    with pytest.raises(ValueError, match="holds no exported runs"):
        runner.exported_runs("misc", root / "misc")


def test_view_passes_every_policy_in_one_call(monkeypatch, run_tree):
    root, exported, _ = run_tree
    monkeypatch.setattr(HPV, "prepare", lambda self: None)
    # Hub policies are resolved by cyclotron; --revision pins the first of them only.
    monkeypatch.setattr(runner, "resolve_input", lambda word, kind, select, revision: {"path": f"/cache/{word}@{revision}"})
    execute = Mock()
    monkeypatch.setattr(HPV, "execute", lambda self, args: execute(args))
    monkeypatch.setattr(HPV, "write_template", lambda self: None)
    monkeypatch.setenv("SSH_CONNECTION", "test")
    policy = exported[1] / "exported"
    runner.view(ROOT, [str(policy), "org/first", str(root), "org/second"], revision="v1", options=["port=3131"])
    assert execute.call_args.args[0] == [
        "scripts/hf.mjs", f"policy={policy}", "policy=/cache/org/first@v1",
        *(f"policy={run / 'exported'}" for run in exported), "policy=/cache/org/second@None",
        f"benchmark={ROOT / 'benchmarks/default'}", "port=3131", "no-open",
    ]


def test_default_benchmark_is_cyclotrons_own_and_opens_in_place(monkeypatch, tmp_path):
    monkeypatch.setattr(HPV, "prepare", lambda self: None)
    harness, item = runner.resolve_benchmark(ROOT, None)
    assert Path(item["path"]) == ROOT / "benchmarks/default"
    assert harness.name == HPV.name and {"benchmark.yaml", "data/locomotion/test.jsonl"} <= set(item["files"])
    monkeypatch.setattr(runner, "resolve_input", lambda *a: {"path": str(tmp_path / "policy")})
    opened = Mock()
    monkeypatch.setattr(HPV, "view", lambda self, policies, benchmark, options, hub: opened(policies, benchmark, hub))
    runner.view(ROOT, ["org/model"])
    opened.assert_called_once_with([str(tmp_path / "policy")], str(ROOT / "benchmarks/default"), None)



def test_new_benchmarks_start_from_the_pinned_hpvs_own_code(monkeypatch, tmp_path):
    monkeypatch.setenv("CYCLOTRON_CACHE_DIR", str(tmp_path / "cache"))
    harness = HPV(ROOT)
    context = {"protocol": 3, "robots": ["asimov-1"], "interfaces": ["velocity-command"],
               "metrics": ["survival_rate@3", "survival_time@4"]}
    monkeypatch.setattr(harness, "execute", lambda args, capture=False: context)
    folder = harness.write_template()
    raw = yaml.safe_load((folder / "benchmark.yaml").read_text())
    assert raw["version"] == 1
    assert raw["harness"] == {"repo": HPV.repo, "protocol": 3, "tested_commit": None}
    assert raw["requires"] == {"robot": {"name": "asimov-1"}, "policy_interface": "velocity-command"}
    assert [task["metric"] for task in raw["tasks"]] == ["survival_rate@3", "survival_time@4"]
    assert (folder / "README.md").read_text().startswith("# new-benchmark\n\nDescribe what this benchmark measures.\n")
    # Named by its contents: the same pin reuses it, another pin gets its own.
    assert harness.write_template() == folder
    context["protocol"] = 4
    assert harness.write_template() != folder
    context["robots"] = ["asimov-1", "asimov-2"]
    with pytest.raises(ValueError, match="needs choosing"):
        harness.write_template()


@pytest.mark.skipif(not shutil.which("node") or not (ROOT / "third_party/humanoid-policy-viewer/node_modules/yaml").is_dir(),
                    reason="needs node and the harness's npm dependencies")
def test_the_viewers_new_benchmark_is_version_1_under_the_pinned_hpv(monkeypatch, tmp_path):
    monkeypatch.setenv("CYCLOTRON_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(HPV, "prepare", lambda self: None)
    harness = HPV(ROOT)
    template = harness.write_template()
    # What the viewer's New benchmark makes from it (HPV's initBenchmark, without example tests).
    made = tmp_path / "my-bench"
    script = ("const { initBenchmark } = await import('./scripts/benchmarkInit.mjs');"
              "initBenchmark(process.argv[1], { title: 'My bench', exampleTest: false, template: process.argv[2], log: () => {} });")
    subprocess.run(["node", "--input-type=module", "-e", script, str(made), str(template)], cwd=harness.path, check=True)
    raw = yaml.safe_load((made / "benchmark.yaml").read_text())
    written = yaml.safe_load((template / "benchmark.yaml").read_text())
    assert (raw["name"], raw["title"], raw["version"]) == ("my-bench", "My bench", 1)
    assert raw["harness"]["tested_commit"] == harness.checkout()["commit"]
    assert {key: raw[key] for key in ("requires", "defaults", "tests", "tasks")} == {
        key: written[key] for key in ("requires", "defaults", "tests", "tasks")}
    assert (made / "README.md").read_text().startswith("# My bench\n\nDescribe what this benchmark measures.\n")


def test_a_hub_benchmark_opens_read_only_from_the_cache(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(HPV, "prepare", lambda self: None)
    cached = tmp_path / "cache/org__suite"
    shutil.copytree(ROOT / "benchmarks/default", cached)
    item = {"source": "hub", "repo_type": "dataset", "id": "org/suite", "revision": "v1", "sha": SHA,
            "endpoint": "https://huggingface.co", "path": str(cached)}
    monkeypatch.setattr(runner, "resolve_input", lambda word, kind, select: item)
    execute = Mock()
    monkeypatch.setattr(HPV, "execute", lambda self, args: execute(args))
    monkeypatch.setattr(HPV, "write_template", lambda self: None)
    monkeypatch.setenv("SSH_CONNECTION", "test")
    policy = tmp_path / "policy.onnx"
    policy.write_bytes(b"onnx")
    runner.view(ROOT, [str(policy)], "datasets/org/suite@v1")
    assert f"benchmark={cached}" in execute.call_args.args[0]
    # HPV's viewer refuses edits to a folder holding its Hub download record; the cache checks skip it.
    assert provenance.read_json(cached / ".hf-dataset.json") == {"id": "org/suite", "revision": "v1", "sha": SHA}
    assert ".hf-dataset.json" not in provenance.file_names(cached)
    assert "to edit it: ./cyclotron.sh --benchmark pull datasets/org/suite@v1" in capsys.readouterr().out


@pytest.mark.skipif(not __import__("shutil").which("node")
                    or not (ROOT / "third_party/humanoid-policy-viewer/node_modules/yaml").is_dir(),
                    reason="needs node and the harness's npm dependencies")
def test_default_benchmark_is_valid_for_the_pinned_app(monkeypatch, tmp_path):
    # After a harness upgrade, this is what says cyclotron's own copy needs updating.
    monkeypatch.setenv("CYCLOTRON_CACHE_DIR", str(tmp_path))
    harness = HPV(ROOT)
    monkeypatch.setattr(harness, "prepare", lambda: None)
    inspection = harness.inspect_benchmark(ROOT / "benchmarks/default")
    assert inspection["schema"] == "hpv-inspection/1"
    assert not inspection["errors"] and not inspection["problems"]
    assert [task["id"] for task in inspection["tasks"]] == ["survival_rate_v2", "survival_time_v2"]


@pytest.mark.skipif(not __import__("shutil").which("node")
                    or not (ROOT / "third_party/humanoid-policy-viewer/node_modules/yaml").is_dir(),
                    reason="needs node and the harness's npm dependencies")
def test_policy_inspection_records_hashes_and_stops_on_errors(monkeypatch, tmp_path):
    monkeypatch.setenv("CYCLOTRON_CACHE_DIR", str(tmp_path / "cache"))
    harness = HPV(ROOT)
    monkeypatch.setattr(harness, "prepare", lambda: None)
    bundled = harness.path / "public/examples/checkpoints/asimov/model_aug_18_1"
    policy = harness.inspect_policy({"path": str(bundled)})
    assert policy["primary"] == "policy_isaaclab_new.onnx" and set(policy["hpv_hashes"]) == {"onnx", "envYaml"}
    assert policy["files"]["env.yaml"] == policy["hpv_hashes"]["envYaml"]
    # Without env.yaml the viewer would guess the policy's gains: cyclotron stops with HPV's reason.
    (tmp_path / "no-env").mkdir()
    (tmp_path / "no-env/policy.onnx").write_bytes((bundled / policy["primary"]).read_bytes())
    with pytest.raises(ValueError, match="cannot be run:\n  - missing env.yaml"):
        harness.inspect_policy({"path": str(tmp_path / "no-env")})


@pytest.mark.skipif(not __import__("shutil").which("node"), reason="needs node")
@pytest.mark.parametrize("sig,forwarded", [("SIGTERM", True), ("SIGINT", False)])
def test_stopping_the_harness_is_quiet_and_signals_it_once(monkeypatch, tmp_path, capfd, sig, forwarded):
    import os
    import signal
    import threading

    monkeypatch.setenv("CYCLOTRON_CACHE_DIR", str(tmp_path))
    harness = HPV(ROOT)
    monkeypatch.setattr(harness, "prepare", lambda: None)
    # Ctrl+C reaches the harness through the terminal itself; only the wrapper is signalled here.
    child = ("for (const s of ['SIGINT', 'SIGTERM']) process.on(s, () => { console.log('got ' + s); process.exit(130); });"
             "setTimeout(() => process.exit(0), 1500);")
    threading.Timer(0.5, os.kill, (os.getpid(), getattr(signal, sig))).start()
    with pytest.raises(KeyboardInterrupt):
        harness.execute(["-e", child])
    assert ("got " + sig in capfd.readouterr().out) is forwarded
