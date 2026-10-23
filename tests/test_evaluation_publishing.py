import copy
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import requests
import yaml
from huggingface_hub.errors import EntryNotFoundError, HfHubHTTPError, LocalEntryNotFoundError, RepositoryNotFoundError

from cyclotron.evaluation import hub, publishing, runner
from cyclotron.evaluation.harnesses.hpv import HPV
from cyclotron.evaluation.provenance import file_manifest, read_json, write_json

SHA = "a" * 40
ROOT = Path(__file__).resolve().parents[1]
START = {"pos": [0, 0, 0.16], "rot": [0.7071068, 0, -0.7071068, 0], "joint_pos": {".*_knee_joint": 0.6}}
HARNESS = {"name": "humanoid-policy-viewer", "repo": "https://github.com/menloresearch/humanoid-policy-viewer",
           "protocol": 1, "commit": SHA}


def http_error(status, cls=HfHubHTTPError):
    response = requests.Response()
    response.status_code = status
    response.url = "https://huggingface.co/test"
    return cls("test", response=response)


@pytest.mark.parametrize("status,attempts", [(403, 2), (409, 1), (401, 1), (500, 1)])
def test_commit_fallback_only_for_permission_denial(status, attempts):
    api = Mock()
    api.create_commit.side_effect = [http_error(status), SimpleNamespace(pr_url="pr")]
    if status == 403:
        hub.commit_files(api, "org/repo", {"README.md": "hello"}, SHA, "Update")
    else:
        with pytest.raises(HfHubHTTPError):
            hub.commit_files(api, "org/repo", {"README.md": "hello"}, SHA, "Update")
    assert api.create_commit.call_count == attempts
    for call in api.create_commit.call_args_list:
        assert call.kwargs["parent_commit"] == SHA
    if attempts == 2:
        assert api.create_commit.call_args.kwargs["create_pr"] is True


def test_only_the_hubs_404_reads_as_an_absent_file(monkeypatch):
    from huggingface_hub import errors

    api = Mock(endpoint="https://huggingface.co", token=None)
    # The Hub's own "entry not found" (RemoteEntryNotFoundError in huggingface_hub 1.x).
    fetch = Mock(side_effect=http_error(404, getattr(errors, "RemoteEntryNotFoundError", EntryNotFoundError)))
    monkeypatch.setattr("huggingface_hub.hf_hub_download", fetch)
    assert hub.read_repo_text(api, "org/model", "eval.yaml", SHA) is None
    # hf_hub_download raises LocalEntryNotFoundError, an EntryNotFoundError, for connection errors,
    # timeouts, 5xx, 429 and 403: the file may exist, so a submission must not merge into nothing.
    fetch.side_effect = LocalEntryNotFoundError("An error happened while trying to locate the file on the Hub")
    with pytest.raises(LocalEntryNotFoundError):
        hub.read_repo_text(api, "org/model", "eval.yaml", SHA)


@pytest.fixture
def inspection():
    return {"raw": {"name": "suite", "title": "Test Suite", "description": "test", "version": 2,
                    "license": "bsd-3-clause", "harness": {"protocol": 1, "tested_commit": SHA}},
            "tasks": [{"id": "score_v2", "config": "all", "split": "test"}], "scored_hash": "new",
            "card": {"license": "bsd-3-clause", "tags": ["humanoid-policy-viewer"], "configs": []},
            "files": ["benchmark.yaml", "data/flat/test.jsonl"], "errors": [], "problems": [], "public": True,
            "publish_files": {"benchmark.yaml": "name: suite\nversion: 2\nharness:\n  tested_commit: " + SHA + "\n",
                              "data/flat/test.jsonl": json.dumps({"id": "flat/a", "start": json.dumps(START)}) + "\n"}}


def test_version_guard(inspection):
    previous = copy.deepcopy(inspection)
    previous["scored_hash"] = "old"
    with pytest.raises(ValueError, match="increase"):
        publishing.version_check(inspection, previous)
    previous["raw"]["version"] = 1
    publishing.version_check(inspection, previous)
    previous["raw"]["version"] = 3
    with pytest.raises(ValueError, match="decrease"):
        publishing.version_check(inspection, previous)


def test_metadata_preserves_prose_and_historical_tasks(tmp_path, inspection):
    previous = tmp_path / "previous"
    previous.mkdir()
    (previous / "eval.yaml").write_text("tasks:\n- id: score_v1\n  config: all\n- id: score_v2\n  config: old\n")
    (tmp_path / "README.md").write_text(
        "---\ncustom: preserved\ntags: [custom, humanoid-policy-viewer]\n---\n\nResearch notes.\n")
    generated = publishing.hub_files(inspection, tmp_path, previous, harness=HARNESS)
    spec = yaml.safe_load(generated["eval.yaml"])
    assert spec["evaluation_framework"] == "cyclotron"
    assert set(spec) == {"name", "description", "evaluation_framework", "tasks"}
    assert [task["id"] for task in spec["tasks"]] == ["score_v2", "score_v1"]
    # New tasks name their harness; a historical task keeps what it was published with.
    assert spec["tasks"][0]["harness"] == HARNESS and "harness" not in spec["tasks"][1]
    assert "&" not in generated["eval.yaml"]
    front, body = publishing.split_card(generated["README.md"])
    assert front["custom"] == "preserved"
    # The framework is cyclotron: the harness's own tag is dropped, even from an earlier card.
    assert front["tags"] == ["cyclotron", "custom"]
    assert body == "Research notes.\n"


def test_license_requires_deliberate_choice(tmp_path, inspection):
    inspection["raw"].pop("license")
    inspection["card"].pop("license")
    with pytest.raises(ValueError, match="license"):
        publishing.hub_files(inspection, tmp_path, harness=HARNESS)
    (tmp_path / "README.md").write_text("---\nlicense: mit\n---\nNotes")
    assert "license: mit" in publishing.hub_files(inspection, tmp_path, harness=HARNESS)["README.md"]
    inspection["raw"]["license"] = None
    assert "license:" not in publishing.hub_files(inspection, tmp_path, harness=HARNESS)["README.md"]


def test_publish_offers_a_license_and_never_assumes_one(tmp_path, monkeypatch):
    from cyclotron.evaluation import cli

    folder = tmp_path / "suite"
    folder.mkdir()
    original = "name: suite\nversion: 1                # bump\nharness: {}\n"
    calls = []

    def publish(root, directory, *rest):
        calls.append(rest)
        if "license:" not in (folder / "benchmark.yaml").read_text():
            raise publishing.MissingLicense("Set license in benchmark.yaml before publishing")
        return {"blockers": []}

    monkeypatch.setattr(publishing, "publish", publish)
    monkeypatch.setattr(cli.sys, "stdin", SimpleNamespace(isatty=lambda: True))
    # Declined, a dry run, or no terminal: the folder is left as it is.
    for argv, tty, answer in [([], True, "n"), (["--dry-run"], True, "y"), ([], False, "y")]:
        (folder / "benchmark.yaml").write_text(original)
        monkeypatch.setattr(cli.sys.stdin, "isatty", lambda tty=tty: tty)
        monkeypatch.setattr("builtins.input", lambda prompt, answer=answer: answer)
        with pytest.raises(publishing.MissingLicense):
            cli.main(ROOT, ["publish", str(folder), "org/suite", *argv])
        assert (folder / "benchmark.yaml").read_text() == original
    # Accepted: the line goes after version, as HPV's docs place it, and publication goes on.
    calls.clear()
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda prompt: "y")
    cli.main(ROOT, ["publish", str(folder), "org/suite", "--create"])
    assert (folder / "benchmark.yaml").read_text() == (
        "name: suite\nversion: 1                # bump\nlicense: bsd-3-clause\nharness: {}\n")
    assert len(calls) == 2


def test_tag_is_idempotent_but_never_moved():
    api = Mock()
    api.list_repo_refs.return_value.tags = [SimpleNamespace(name="v2", target_commit=SHA)]
    publishing.ensure_tag(api, "org/suite", SHA, 2)
    api.create_tag.assert_not_called()
    with pytest.raises(ValueError, match="another commit"):
        publishing.ensure_tag(api, "org/suite", "b" * 40, 2)


@pytest.fixture
def publication(tmp_path, monkeypatch, inspection):
    monkeypatch.setenv("CYCLOTRON_CACHE_DIR", str(tmp_path / "cache"))
    directory = tmp_path / "working"
    directory.mkdir()
    (directory / "benchmark.yaml").write_text(yaml.safe_dump(inspection["raw"]))
    (directory / "data/flat").mkdir(parents=True)
    (directory / "data/flat/test.jsonl").write_text("{}\n")
    state = {"commit": SHA, "dirty": False}
    harness = Mock()
    harness.name = HPV.name
    harness.state.return_value = state
    harness.inspect_benchmark.return_value = inspection
    harness.task_harness.return_value = HARNESS
    harness.fetch_history.return_value = None
    monkeypatch.setattr(publishing, "select_harness", lambda *a, **kw: harness)
    monkeypatch.setattr(publishing, "git_state", lambda *a: state.copy())
    api = Mock(endpoint="https://huggingface.co")
    api.repo_info.side_effect = http_error(404, RepositoryNotFoundError)
    return directory, harness, api


def test_publish_dry_run_does_not_mutate_folder_or_hub(publication):
    directory, harness, api = publication
    before = file_manifest(directory)
    result = publishing.publish(ROOT, directory, "org/suite", create=True, dry_run=True, api=api)
    assert not result["blockers"]
    spec = yaml.safe_load(result["files"]["eval.yaml"])
    assert spec["evaluation_framework"] == "cyclotron"
    # The harness block records the harness commit the benchmark was checked with.
    assert harness.task_harness.call_args.args[1] == SHA and spec["tasks"][0]["harness"] == HARNESS
    assert file_manifest(directory) == before
    api.create_repo.assert_not_called()
    api.create_commit.assert_not_called()
    api.create_tag.assert_not_called()


@pytest.mark.parametrize("included", [False, True])
def test_publish_requires_custom_license_in_upload(publication, included):
    directory, harness, api = publication
    inspection = harness.inspect_benchmark.return_value
    inspection["raw"]["license"] = "LICENSE.txt"
    inspection["card"].update(license="other", license_link="LICENSE.txt")
    (directory / "LICENSE.txt").write_text("Custom license terms.\n")
    if included:
        inspection["publish_files"]["LICENSE.txt"] = "Custom license terms.\n"
        result = publishing.publish(ROOT, directory, "org/suite", create=True, dry_run=True, api=api)
        assert result["files"]["LICENSE.txt"] == "Custom license terms.\n"
    else:
        with pytest.raises(ValueError, match="publishable license filename"):
            publishing.publish(ROOT, directory, "org/suite", create=True, dry_run=True, api=api)
    api.create_repo.assert_not_called()
    api.create_commit.assert_not_called()
    api.create_tag.assert_not_called()


def test_publish_uploads_what_the_harness_gives_not_the_folder_bytes(publication):
    directory, harness, api = publication
    api.repo_info.side_effect = [http_error(404, RepositoryNotFoundError), SimpleNamespace(sha=SHA)]
    api.create_commit.return_value = SimpleNamespace(pr_url="https://hub/pr/1", commit_url=None, oid="b" * 40)
    publishing.publish(ROOT, directory, "org/suite", create=True, pr=True, api=api)
    uploaded = {op.path_in_repo: op.path_or_fileobj for op in api.create_commit.call_args.kwargs["operations"]}
    published = harness.inspect_benchmark.return_value["publish_files"]
    assert uploaded["benchmark.yaml"] == published["benchmark.yaml"].encode()
    assert uploaded["data/flat/test.jsonl"] == published["data/flat/test.jsonl"].encode()
    assert uploaded["data/flat/test.jsonl"] != (directory / "data/flat/test.jsonl").read_bytes()
    assert {"eval.yaml", "README.md"} <= set(uploaded)


@pytest.mark.skipif(not shutil.which("node") or not (ROOT / "third_party/humanoid-policy-viewer/node_modules/yaml").is_dir(),
                    reason="needs node and the harness's npm dependencies")
def test_start_poses_upload_as_json_text_and_score_the_same(tmp_path, monkeypatch):
    monkeypatch.setenv("CYCLOTRON_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(HPV, "prepare", lambda self: None)
    # No network: whether the commit is public is not what this checks.
    monkeypatch.setattr(HPV, "fetch_history", lambda self: None)
    directory = tmp_path / "suite"
    shutil.copytree(ROOT / "benchmarks/default", directory)
    (directory / "benchmark.yaml").write_text((directory / "benchmark.yaml").read_text() + "license: bsd-3-clause\n")
    # Locally HPV's editor writes a start pose as an object; the card declares the column a string.
    rows = directory / "data/locomotion/test.jsonl"
    lines = rows.read_text().splitlines()
    lines[0] = json.dumps({**json.loads(lines[0]), "start": START})
    rows.write_text("\n".join(lines) + "\n")
    api = Mock(endpoint="https://huggingface.co")
    api.repo_info.side_effect = http_error(404, RepositoryNotFoundError)
    files = publishing.publish(ROOT, directory, "org/suite", create=True, dry_run=True, api=api)["files"]
    uploaded = [json.loads(line) for line in files["data/locomotion/test.jsonl"].splitlines()]
    assert isinstance(uploaded[0]["start"], str) and json.loads(uploaded[0]["start"]) == START
    assert uploaded[1:] == [json.loads(line) for line in lines[1:]]
    # benchmark.yaml records the harness commit it was checked with.
    assert yaml.safe_load(files["benchmark.yaml"])["harness"]["tested_commit"] == HPV(ROOT).checkout()["commit"]
    api.create_commit.assert_not_called()
    # Read back in its Hub form, the benchmark is the same scored content as the local folder.
    published = tmp_path / "published"
    for name, text in files.items():
        (published / name).parent.mkdir(parents=True, exist_ok=True)
        (published / name).write_text(text)
    harness = HPV(ROOT)
    local, hub_form = harness.inspect_benchmark(directory), harness.inspect_benchmark(published)
    assert not local["errors"] and not hub_form["errors"]
    assert hub_form["scored_hash"] == local["scored_hash"]


@pytest.fixture
def existing(publication, tmp_path, monkeypatch):
    """The dataset already exists on the Hub at SHA, published from the same suite."""
    directory, harness, api = publication
    previous = tmp_path / "previous"
    previous.mkdir()
    (previous / "benchmark.yaml").write_text("name: suite\n")
    (previous / "README.md").write_text("---\nlicense: bsd-3-clause\n---\n\nNotes.\n")
    api.repo_info.side_effect = lambda repo_id, repo_type, revision, **kw: SimpleNamespace(
        id="org/suite", sha=SHA if revision == "main" else revision)
    monkeypatch.setattr(publishing, "download", lambda reference, *a: {"path": str(previous)})
    harness.dataset_files = HPV.dataset_files
    return directory, harness, api, previous


def origin(directory, sha, dataset="org/suite", endpoint="https://huggingface.co"):
    # What hub.editable_copy records when --benchmark pull copies a Hub dataset to edit.
    write_json(directory / ".cyclotron-origin.json", {"origin": {
        "id": dataset, "endpoint": endpoint, "revision": "main", "sha": sha}})


def test_a_pulled_benchmark_publishes_onto_the_commit_it_matches(existing):
    directory, harness, api, _ = existing
    origin(directory, "b" * 40)
    # Main moved since the folder was pulled: publishing it would overwrite those changes.
    with pytest.raises(ValueError, match=f"main moved since this folder was pulled from main \\({'b' * 40}\\).*pull datasets/org/suite.*--base {SHA}"):
        publishing.publish(ROOT, directory, "org/suite", dry_run=True, api=api)
    harness.inspect_benchmark.assert_not_called()
    # An explicit --base is the researcher's decision; a copy of another dataset or Hub has no say.
    assert not publishing.publish(ROOT, directory, "org/suite", base=SHA, dry_run=True, api=api)["blockers"]
    for other in (dict(dataset="org/other"), dict(endpoint="https://private.example")):
        origin(directory, "b" * 40, **other)
        assert not publishing.publish(ROOT, directory, "org/suite", dry_run=True, api=api)["blockers"]
    origin(directory, SHA, dataset="Org/Suite")
    api.create_commit.return_value = SimpleNamespace(pr_url="https://hub/pr/1", commit_url=None, oid="c" * 40)
    publishing.publish(ROOT, directory, "org/suite", pr=True, api=api)
    assert api.create_commit.call_args.kwargs["parent_commit"] == SHA
    # The origin record is cyclotron's, never uploaded; a PR leaves main, and so the record, as it was.
    assert ".cyclotron-origin.json" not in {op.path_in_repo for op in api.create_commit.call_args.kwargs["operations"]}
    assert read_json(directory / ".cyclotron-origin.json")["origin"]["sha"] == SHA
    # Published directly, the folder is the new main, so its next publication builds on that commit.
    api.create_commit.return_value = SimpleNamespace(pr_url=None, commit_url="https://hub/commit", oid="c" * 40)
    publishing.publish(ROOT, directory, "org/suite", api=api)
    assert read_json(directory / ".cyclotron-origin.json")["origin"] == {
        "id": "Org/Suite", "endpoint": "https://huggingface.co", "revision": "main", "sha": "c" * 40}


def test_an_inherited_license_link_must_be_uploaded(existing):
    directory, harness, api, previous = existing
    inspection = harness.inspect_benchmark.return_value
    inspection["raw"].pop("license")
    inspection["card"].pop("license")
    # The earlier card links a LICENSE that is not in this folder: it would be deleted as obsolete.
    (previous / "README.md").write_text("---\nlicense: other\nlicense_link: LICENSE\n---\n\nNotes.\n")
    (previous / "LICENSE").write_text("Terms.\n")
    with pytest.raises(ValueError, match="license_link LICENSE is not in the upload"):
        publishing.publish(ROOT, directory, "org/suite", dry_run=True, api=api)
    inspection["publish_files"]["LICENSE"] = "Terms.\n"
    result = publishing.publish(ROOT, directory, "org/suite", dry_run=True, api=api)
    assert result["files"]["LICENSE"] == "Terms.\n" and "LICENSE" not in result["deletions"]
    # A link to a web page names no file.
    (previous / "README.md").write_text("---\nlicense: other\nlicense_link: https://example.com/terms\n---\n")
    del inspection["publish_files"]["LICENSE"]
    assert "LICENSE" in publishing.publish(ROOT, directory, "org/suite", dry_run=True, api=api)["deletions"]


def test_publish_uses_the_staged_benchmark_yaml(publication, monkeypatch):
    directory, harness, api = publication
    # benchmark.yaml as first read says version 1; the staged copy the harness inspects says 2.
    (directory / "benchmark.yaml").write_text(yaml.safe_dump({**harness.inspect_benchmark.return_value["raw"], "version": 1}))
    selected = []
    monkeypatch.setattr(publishing, "select_harness", lambda root, name, raw: selected.append(raw) or harness)
    api.repo_info.side_effect = [http_error(404, RepositoryNotFoundError), SimpleNamespace(sha=SHA)]
    api.create_commit.return_value = SimpleNamespace(pr_url=None, commit_url="https://hub/commit", oid="b" * 40)
    api.list_repo_refs.return_value.tags = []
    publishing.publish(ROOT, directory, "org/suite", create=True, api=api)
    assert [raw["version"] for raw in selected] == [1, 2]
    assert api.create_commit.call_args.kwargs["commit_message"] == "Publish suite v2"
    assert api.create_tag.call_args.kwargs["tag"] == "v2"
    # A harness that differs between the first read and the staged copy stops publication.
    other = Mock()
    other.name = "other-harness"
    picks = iter([harness, other])
    monkeypatch.setattr(publishing, "select_harness", lambda *a: next(picks))
    api.repo_info.side_effect = http_error(404, RepositoryNotFoundError)
    with pytest.raises(ValueError, match="harness changed while staging"):
        publishing.publish(ROOT, directory, "org/other", create=True, dry_run=True, api=api)


def test_publish_fetches_harness_history_and_says_how_when_it_cannot(publication, monkeypatch, capsys):
    from cyclotron.evaluation import cli

    directory, harness, api = publication
    harness.inspect_benchmark.return_value["public"] = False
    harness.fetch_history.return_value = "git -C hpv fetch --quiet --unshallow origin"
    result = publishing.publish(ROOT, directory, "org/suite", create=True, dry_run=True, api=api)
    harness.fetch_history.assert_called_once()
    assert result["blockers"] == ["Harness commit is not on a fetched public upstream branch; "
                                  "fetch the harness's history with: git -C hpv fetch --quiet --unshallow origin"]
    # A dry run prints every blocker, then exits 1 as the publication would.
    monkeypatch.setattr(publishing, "hub_api", lambda: api)
    api.repo_info.side_effect = http_error(404, RepositoryNotFoundError)
    with pytest.raises(SystemExit) as stopped:
        cli.main(ROOT, ["publish", str(directory), "org/suite", "--create", "--dry-run"])
    assert stopped.value.code == 1
    assert "Publication blockers:\n  - Harness commit" in capsys.readouterr().out
    harness.inspect_benchmark.return_value["public"] = True
    assert cli.main(ROOT, ["publish", str(directory), "org/suite", "--create", "--dry-run"])["blockers"] == []


@pytest.mark.parametrize("shallow,failed", [(True, False), (False, False), (True, True)])
def test_hpv_fetches_its_full_history_for_the_public_check(monkeypatch, shallow, failed):
    from cyclotron.evaluation.harnesses import hpv

    calls = []
    def run(command, **kwargs):
        calls.append(command)
        if "rev-parse" in command:
            return SimpleNamespace(returncode=0, stdout="true\n" if shallow else "false\n")
        return SimpleNamespace(returncode=128 if failed else 0)
    monkeypatch.setattr(hpv.subprocess, "run", run)
    harness = HPV(ROOT)
    hint = harness.fetch_history()
    fetch = ["git", "-C", str(harness.path), "fetch", "--quiet", *(["--unshallow"] if shallow else []), "origin"]
    assert calls[-1] == fetch
    assert hint == (" ".join(fetch) if failed else None)


def test_publish_refuses_concurrent_ui_edit(publication):
    directory, harness, api = publication
    inspection = harness.inspect_benchmark.return_value
    def edit_during_inspection(*args):
        (directory / "data/flat/test.jsonl").write_text('{"changed":true}\n')
        return inspection
    harness.inspect_benchmark.side_effect = edit_during_inspection
    with pytest.raises(ValueError, match="changed during"):
        publishing.publish(ROOT, directory, "org/suite", create=True, dry_run=True, api=api)
    api.create_commit.assert_not_called()


def test_publish_new_dataset_private_and_pr_not_tagged(publication):
    directory, harness, api = publication
    api.repo_info.side_effect = [http_error(404, RepositoryNotFoundError), SimpleNamespace(sha=SHA)]
    api.create_commit.return_value = SimpleNamespace(pr_url="https://hub/pr/1", commit_url=None, oid="b" * 40)
    publishing.publish(ROOT, directory, "org/suite", create=True, pr=True, api=api)
    assert api.create_repo.call_args.kwargs["private"] is True
    assert api.create_commit.call_args.kwargs["parent_commit"] == SHA
    assert api.create_commit.call_args.kwargs["create_pr"] is True
    api.create_tag.assert_not_called()


def test_publish_partial_success_reports_tag_retry(publication):
    directory, harness, api = publication
    api.repo_info.side_effect = [http_error(404, RepositoryNotFoundError), SimpleNamespace(sha=SHA)]
    api.create_commit.return_value = SimpleNamespace(pr_url=None, commit_url="https://hub/commit", oid="b" * 40)
    api.list_repo_refs.side_effect = RuntimeError("offline")
    with pytest.raises(ValueError, match="Commit succeeded.*Retry with"):
        publishing.publish(ROOT, directory, "org/suite", create=True, api=api)
    api.create_commit.assert_called_once()


def test_release_requires_main_ancestry():
    api = Mock()
    api.repo_info.return_value = SimpleNamespace(sha=SHA)
    api.list_repo_commits.return_value = [SimpleNamespace(commit_id="b" * 40)]
    with pytest.raises(ValueError, match="main history"):
        publishing.release(ROOT, "org/suite", SHA, api=api)
    api.create_tag.assert_not_called()
    with pytest.raises(ValueError, match="pending PR"):
        publishing.release(ROOT, "org/suite", "refs/pr/3", api=api)


def test_failed_run_never_gets_completed_manifest(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset"
    model = tmp_path / "model"
    dataset.mkdir()
    model.mkdir()
    (dataset / "benchmark.yaml").write_text("name: test")
    (model / "policy.onnx").write_bytes(b"policy")
    harness = Mock()
    harness.inspect_benchmark.return_value = {"errors": [], "problems": []}
    harness.inspect_policy.return_value = {"path": str(model), "primary": "policy.onnx", "files": file_manifest(model)}
    harness.state.return_value = {"commit": SHA, "dirty": False}
    harness.run.side_effect = RuntimeError("interrupted")
    monkeypatch.setattr(runner, "resolve_benchmark", lambda *a: (harness, {"path": str(dataset), "files": file_manifest(dataset)}))
    monkeypatch.setattr(runner, "resolve_input", lambda *a: {})
    out = tmp_path / "result"
    with pytest.raises(RuntimeError, match="interrupted"):
        runner.run(ROOT, ["org/model"], output=out)
    assert read_json(out / "cyclotron.json")["status"] == "failed"
    with pytest.raises(FileExistsError):
        runner.run(ROOT, ["org/model"], output=out)


def test_revision_pins_the_first_hub_policy_after_a_local_one(tmp_path, monkeypatch):
    harness = Mock()
    harness.inspect_benchmark.return_value = {"errors": [], "problems": []}
    harness.inspect_policy.side_effect = lambda item: item
    harness.state.return_value = {"commit": SHA, "dirty": False}
    harness.run.side_effect = RuntimeError("stop")
    monkeypatch.setattr(runner, "resolve_benchmark", lambda *a: (harness, {"path": str(tmp_path), "files": {}}))
    revisions = []
    def resolve_input(word, kind, select, revision):
        revisions.append((word, revision))
        return {"source": "hub" if "/" in word else "local", "path": str(tmp_path), "primary": "policy.onnx"}
    monkeypatch.setattr(runner, "resolve_input", resolve_input)
    with pytest.raises(RuntimeError, match="stop"):
        runner.run(ROOT, ["local", "org/a", "org/b"], revision="v2", output=tmp_path / "result")
    assert revisions == [("local", "v2"), ("org/a", "v2"), ("org/b", None)]
