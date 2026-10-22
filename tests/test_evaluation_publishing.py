import copy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import requests
import yaml
from huggingface_hub.errors import EntryNotFoundError, HfHubHTTPError, RepositoryNotFoundError

from cyclotron.evaluation import hub, publishing, runner
from cyclotron.evaluation.provenance import file_manifest, read_json, write_json

SHA = "a" * 40
ROOT = Path(__file__).resolve().parents[1]


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


def test_existing_file_read_never_swallows_auth_failure(monkeypatch):
    api = Mock(endpoint="https://huggingface.co", token=None)
    fetch = Mock(side_effect=http_error(403))
    monkeypatch.setattr("huggingface_hub.hf_hub_download", fetch)
    with pytest.raises(HfHubHTTPError):
        hub.read_repo_text(api, "org/model", "eval.yaml", SHA)
    fetch.side_effect = http_error(404, EntryNotFoundError)
    assert hub.read_repo_text(api, "org/model", "eval.yaml", SHA) is None


@pytest.fixture
def inspection():
    return {"raw": {"name": "suite", "title": "Test Suite", "description": "test", "version": 2,
                    "license": "bsd-3-clause", "harness": {"protocol": 1, "tested_commit": SHA}},
            "tasks": [{"id": "score_v2", "config": "all", "split": "test"}], "scored_hash": "new",
            "card": {"license": "bsd-3-clause", "tags": ["humanoid-policy-viewer"], "configs": []},
            "files": ["benchmark.yaml", "data/flat/test.jsonl"], "errors": [], "problems": [], "public": True,
            "benchmark_yaml": "name: suite\nversion: 2\nharness:\n  tested_commit: " + SHA + "\n"}


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
    (tmp_path / "README.md").write_text("---\ncustom: preserved\ntags: [custom]\n---\n\nResearch notes.\n")
    generated = publishing.hub_files(inspection, tmp_path, previous)
    spec = yaml.safe_load(generated["eval.yaml"])
    assert spec["evaluation_framework"] == "cyclotron"
    assert [task["id"] for task in spec["tasks"]] == ["score_v2", "score_v1"]
    front, body = publishing.split_card(generated["README.md"])
    assert front["custom"] == "preserved"
    assert front["tags"] == ["cyclotron", "custom", "humanoid-policy-viewer"]
    assert body == "Research notes.\n"


def test_license_requires_deliberate_choice(tmp_path, inspection):
    inspection["raw"].pop("license")
    inspection["card"].pop("license")
    with pytest.raises(ValueError, match="license"):
        publishing.hub_files(inspection, tmp_path)
    (tmp_path / "README.md").write_text("---\nlicense: mit\n---\nNotes")
    assert "license: mit" in publishing.hub_files(inspection, tmp_path)["README.md"]
    inspection["raw"]["license"] = None
    assert "license:" not in publishing.hub_files(inspection, tmp_path)["README.md"]


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
    app = Mock()
    app.state.return_value = state
    app.inspect_benchmark.return_value = inspection
    monkeypatch.setattr(publishing, "select_app", lambda *a, **kw: app)
    monkeypatch.setattr(publishing, "git_state", lambda *a: state.copy())
    api = Mock(endpoint="https://huggingface.co")
    api.repo_info.side_effect = http_error(404, RepositoryNotFoundError)
    return directory, app, api


def test_publish_dry_run_does_not_mutate_folder_or_hub(publication):
    directory, app, api = publication
    before = file_manifest(directory)
    result = publishing.publish(ROOT, directory, "org/suite", create=True, dry_run=True, api=api)
    assert not result["blockers"]
    assert yaml.safe_load(result["files"]["eval.yaml"])["evaluation_framework"] == "cyclotron"
    assert file_manifest(directory) == before
    api.create_repo.assert_not_called()
    api.create_commit.assert_not_called()
    api.create_tag.assert_not_called()


@pytest.mark.parametrize("included", [False, True])
def test_publish_requires_custom_license_in_upload(publication, included):
    directory, app, api = publication
    inspection = app.inspect_benchmark.return_value
    inspection["raw"]["license"] = "LICENSE.txt"
    inspection["card"].update(license="other", license_link="LICENSE.txt")
    (directory / "LICENSE.txt").write_text("Custom license terms.\n")
    if included:
        inspection["files"].append("LICENSE.txt")
        result = publishing.publish(ROOT, directory, "org/suite", create=True, dry_run=True, api=api)
        assert result["files"]["LICENSE.txt"] == b"Custom license terms.\n"
    else:
        with pytest.raises(ValueError, match="publishable license filename"):
            publishing.publish(ROOT, directory, "org/suite", create=True, dry_run=True, api=api)
    api.create_repo.assert_not_called()
    api.create_commit.assert_not_called()
    api.create_tag.assert_not_called()


def test_publish_refuses_concurrent_ui_edit(publication):
    directory, app, api = publication
    inspection = app.inspect_benchmark.return_value
    def edit_during_inspection(*args):
        (directory / "data/flat/test.jsonl").write_text('{"changed":true}\n')
        return inspection
    app.inspect_benchmark.side_effect = edit_during_inspection
    with pytest.raises(ValueError, match="changed during"):
        publishing.publish(ROOT, directory, "org/suite", create=True, dry_run=True, api=api)
    api.create_commit.assert_not_called()


def test_publish_new_dataset_private_and_pr_not_tagged(publication):
    directory, app, api = publication
    api.repo_info.side_effect = [http_error(404, RepositoryNotFoundError), SimpleNamespace(sha=SHA)]
    api.create_commit.return_value = SimpleNamespace(pr_url="https://hub/pr/1", commit_url=None, oid="b" * 40)
    publishing.publish(ROOT, directory, "org/suite", create=True, pr=True, api=api)
    assert api.create_repo.call_args.kwargs["private"] is True
    assert api.create_commit.call_args.kwargs["parent_commit"] == SHA
    assert api.create_commit.call_args.kwargs["create_pr"] is True
    api.create_tag.assert_not_called()


def test_publish_partial_success_reports_tag_retry(publication):
    directory, app, api = publication
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
    app = Mock()
    app.inspect_benchmark.return_value = {"errors": [], "problems": []}
    app.inspect_policy.return_value = {"path": str(model), "primary": "policy.onnx", "files": file_manifest(model)}
    app.state.return_value = {"commit": SHA, "dirty": False}
    app.run.side_effect = RuntimeError("interrupted")
    monkeypatch.setattr(runner, "resolve_benchmark", lambda *a: (app, {"path": str(dataset), "files": file_manifest(dataset)}))
    monkeypatch.setattr(runner, "resolve_input", lambda *a: {})
    out = tmp_path / "result"
    with pytest.raises(RuntimeError, match="interrupted"):
        runner.run(ROOT, ["org/model"], output=out)
    assert read_json(out / "cyclotron.json")["status"] == "failed"
    with pytest.raises(FileExistsError):
        runner.run(ROOT, ["org/model"], output=out)
