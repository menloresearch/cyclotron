import importlib.util
import os

import pytest

SHARE_SCRIPT = os.path.join(os.path.dirname(__file__), os.pardir, "scripts", "share.py")
spec = importlib.util.spec_from_file_location("share", SHARE_SCRIPT)
share = importlib.util.module_from_spec(spec)
spec.loader.exec_module(share)


@pytest.fixture
def runs(tmp_path):
    for run in ("runA", "runB"):
        (tmp_path / run).mkdir()
        (tmp_path / run / "model_100.pt").write_bytes(b"")
    return tmp_path


def test_a_checkpoint_of_the_run_resolves_by_name_or_path(runs):
    run_dir = str(runs / "runA")
    expected = str(runs / "runA" / "model_100.pt")
    assert share.resolve_checkpoint(run_dir, "model_100.pt") == expected
    assert share.resolve_checkpoint(run_dir, expected) == expected
    assert share.resolve_checkpoint(run_dir, os.path.join(run_dir, ".", "model_100.pt")) == expected


def test_a_checkpoint_of_another_run_is_refused(runs):
    with pytest.raises(SystemExit) as stop:
        share.resolve_checkpoint(str(runs / "runA"), str(runs / "runB" / "model_100.pt"))
    assert "is not in" in str(stop.value)
    assert f"--share {runs / 'runB'} --checkpoint model_100.pt" in str(stop.value)
    with pytest.raises(SystemExit, match="Checkpoint not found"):
        share.resolve_checkpoint(str(runs / "runA"), "model_200.pt")


def test_files_an_earlier_share_uploaded_and_this_one_doesnt_are_removed():
    repo = ["README.md", ".gitattributes", "policy.onnx", "policy.pt", "env.yaml", "agent.yaml", "code_state.yaml"]
    # An --onnx share of a run without a code_state.yaml: the old policy.pt and code_state.yaml would not match it.
    uploaded = ["agent.yaml", "env.yaml", "policy.onnx"]
    assert share.stale_files(repo, uploaded) == ["policy.pt", "code_state.yaml"]
    # Only the files a share writes are touched; the model card is always replaced, other files are left alone.
    assert share.stale_files(["notes.md", "policy.onnx"], ["policy.onnx"]) == []


def test_a_dry_run_lists_the_exported_policy_and_the_run_settings(runs, monkeypatch, capsys):
    params = runs / "runA" / "params"
    params.mkdir()
    (params / "agent.yaml").write_text("experiment_name: asimov_velocity_amp\n")
    (params / "env.yaml").write_text("seed: 42\n")
    monkeypatch.setattr("sys.argv", ["share.py", str(runs / "runA"), "--repo-id", "me/policy", "--dry-run"])
    share.main()
    out = capsys.readouterr().out
    assert f"Would export {runs / 'runA' / 'model_100.pt'} to ONNX using task Asimov1-Velocity-AMP-v0" in out
    for name in ("policy.onnx", "policy.pt", "agent.yaml", "env.yaml"):
        assert f"-> {name}\n" in out
    assert "`policy.pt` is the same policy as TorchScript" in out
