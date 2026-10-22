"""Evaluation orchestration can be tested without Isaac Lab, Node or the network."""

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml

from cyclotron.evaluation import hub, provenance, runner
from cyclotron.evaluation.harnesses.base import select_harness
from cyclotron.evaluation.harnesses.hpv import HPV
from cyclotron.evaluation.cli import parse_run
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
    with pytest.raises(ValueError, match="Run options"):
        parse_run(["org/first", "jobs=2"], viewing=True)


def test_adapter_registry(monkeypatch):
    monkeypatch.delenv("CYCLOTRON_HARNESS", raising=False)
    assert select_harness(ROOT, benchmark={"harness": {"repo": HPV.repo}}).name == HPV.name
    for raw in ({"harness": {"repo": "https://evil.example/code"}},
                {"harness": {"repo": HPV.repo, "name": "other"}}):
        with pytest.raises(ValueError):
            select_harness(ROOT, benchmark=raw)
    with pytest.raises(ValueError, match="available"):
        select_harness(ROOT, "other")


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
    api.repo_info.return_value = SimpleNamespace(sha=SHA, siblings=siblings)
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


def test_working_copy_preserves_edits(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "benchmark.yaml").write_text("original")
    item = {"path": str(source), "id": "org/suite", "sha": SHA, "revision": "main", "source": "hub",
            "endpoint": "https://huggingface.co"}
    working = hub.working_copy(item, tmp_path / "workspaces")
    (working / "benchmark.yaml").write_text("edited")
    again = hub.working_copy({**item, "sha": "b" * 40}, tmp_path / "workspaces")
    assert again == working and (again / "benchmark.yaml").read_text() == "edited"
    assert (source / "benchmark.yaml").read_text() == "original"


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
    for malformed in ("oops: true", "", "- nope"):
        with pytest.raises(ValueError):
            merge_results(malformed, entries)
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
    if "other-harness" in harness:
        # A task's results only count from the harness its eval.yaml names.
        with pytest.raises(ValueError, match="scored by other-harness"):
            submit(ROOT, tmp_path, dry_run=True, api=api)
    else:
        assert len(submit(ROOT, tmp_path, dry_run=True, api=api)) == 1
        assert read.call_args.args[3] == SHA
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
    harness = HPV(ROOT)
    execute = Mock()
    monkeypatch.setattr(harness, "execute", execute)
    monkeypatch.setenv("SSH_CONNECTION", "test")
    harness.view(str(tmp_path / "policy"), str(tmp_path / "suite"), [])
    assert execute.call_args.args[0] == ["scripts/hf.mjs", str(tmp_path / "policy"), f"benchmark={tmp_path / 'suite'}", "no-open"]
    harness.run([str(tmp_path / "policy")], str(tmp_path / "suite"), tmp_path / "out", ["jobs=2"])
    assert f"out={tmp_path / 'out'}" in execute.call_args.args[0]
    env = harness.env()
    assert env["HPV_BENCHMARKS_DIR"].startswith(str(ROOT / "logs"))
    assert "HF_TOKEN" not in env


def test_default_benchmark_is_cyclotrons_own_and_opens_in_place(monkeypatch, tmp_path):
    monkeypatch.setattr(HPV, "prepare", lambda self: None)
    harness, item = runner.resolve_benchmark(ROOT, None)
    assert Path(item["path"]) == ROOT / "benchmarks/default"
    assert harness.name == HPV.name and {"benchmark.yaml", "data/locomotion/test.jsonl"} <= set(item["files"])
    monkeypatch.setattr(runner, "resolve_input", lambda *a: {"path": str(tmp_path / "policy")})
    monkeypatch.setattr(runner, "working_copy", Mock(side_effect=AssertionError("the default was copied")))
    opened = Mock()
    monkeypatch.setattr(HPV, "view", lambda self, policy, benchmark, options: opened(benchmark))
    runner.view(ROOT, "org/model")
    opened.assert_called_once_with(str(ROOT / "benchmarks/default"))


@pytest.mark.skipif(not __import__("shutil").which("node")
                    or not (ROOT / "third_party/humanoid-policy-viewer/node_modules/yaml").is_dir(),
                    reason="needs node and the harness's npm dependencies")
def test_default_benchmark_is_valid_for_the_pinned_app(monkeypatch, tmp_path):
    # After a harness upgrade, this is what says cyclotron's own copy needs updating.
    monkeypatch.setenv("CYCLOTRON_CACHE_DIR", str(tmp_path))
    harness = HPV(ROOT)
    monkeypatch.setattr(harness, "prepare", lambda: None)
    inspection = harness.inspect_benchmark(ROOT / "benchmarks/default")
    assert not inspection["errors"] and not inspection["problems"]
    assert [task["id"] for task in inspection["tasks"]] == ["survival_rate_v2", "survival_time_v2"]


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
