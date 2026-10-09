from __future__ import annotations

import hashlib
import os
import shutil
import subprocess

import pytest
import yaml
from config_samples import dump, make_agent, make_env

from cyclotron.code_state import (
    CODE_STATE_FILE,
    _git_state,
    _model_where,
    _where,
    _without_credentials,
    actor_code,
    actor_code_differences,
    check_out_hint,
    code_state_missing,
    describe_changes,
    edited_run_configs,
    hash_run_configs,
    read_code_state,
    record_code_state,
    robot_model_check,
    training_code,
    training_commit,
    training_robot_model,
    write_code_state,
)

TRAINED_ON = {"commit": "32aef5c5ec11641f785bfd3c4aebb9c4de6bdc6b", "branch": "exp/drift", "dirty": False}


def with_robot(env: dict, urdf) -> dict:
    env["scene"]["robot"]["spawn"] = {"asset_path": str(urdf)}
    return env


def test_code_state_records_git_versions_and_the_robot_model_but_no_files(tmp_path):
    urdf = tmp_path / "asimov_1.urdf"
    urdf.write_text("<robot/>")
    io = {"actions": {"joint_names": ["hip"], "scale": [0.25]}, "observations": {}}
    state = record_code_state(
        with_robot(make_env(), urdf), "2026-09-26_base/model_500.pt", agent=make_agent(), policy_io=io
    )
    assert list(state) == [
        "cyclotron",
        "isaaclab_commit",
        "packages",
        "robot_model",
        "actor_code",
        "policy_io",
        "loaded_checkpoint",
    ]
    # Git's view of the package, not a hash of each of its files.
    assert set(state["cyclotron"]) == {"commit", "branch", "remote", "dirty"}
    assert state["policy_io"] == io
    write_code_state(str(tmp_path / "params"), state)
    text = (tmp_path / "params" / CODE_STATE_FILE).read_text()
    assert state["robot_model"]["urdf_filepath"] == "asimov_1.urdf" and str(tmp_path) not in text
    assert yaml.safe_load(text)["loaded_checkpoint"] == "2026-09-26_base/model_500.pt"
    # Without an agent or a resolved environment there is nothing to record for them.
    assert not {"actor_code", "policy_io", "loaded_checkpoint"} & set(record_code_state(make_env()))


def test_read_code_state_is_none_without_the_file(tmp_path):
    assert read_code_state(str(tmp_path)) is None
    assert code_state_missing(str(tmp_path))
    (tmp_path / "params").mkdir()
    (tmp_path / "params" / CODE_STATE_FILE).write_text("")
    assert read_code_state(str(tmp_path)) == {}
    assert not code_state_missing(str(tmp_path))


def test_actor_code_hashes_the_network_modules_and_not_the_rest_of_the_library():
    modules = actor_code(make_agent())
    assert {"rsl_rl.models.mlp_model", "rsl_rl.modules.mlp", "rsl_rl.modules.normalization"} <= set(modules)
    assert all(name.startswith(("rsl_rl.models", "rsl_rl.modules")) for name in modules)
    # Every network module, whichever the run uses; the distilled student is found under its own key.
    assert "rsl_rl.modules.rnn" in modules
    student = {"class_name": "DistillationRunner", "student": {"class_name": "MLPModel"}}
    assert actor_code(student) == modules
    assert actor_code({"actor": {}}) is None and actor_code({"actor": {"class_name": "NoSuchModel"}}) is None


def test_actor_code_adds_the_module_of_a_custom_network(tmp_path, monkeypatch):
    source = "from rsl_rl.models import MLPModel\n\n\nclass WideModel(MLPModel):\n    pass\n"
    (tmp_path / "wide_model.py").write_text(source)
    monkeypatch.syspath_prepend(str(tmp_path))
    modules = actor_code({"actor": {"class_name": "wide_model:WideModel"}})
    assert modules["wide_model"] == hashlib.sha256(source.encode()).hexdigest()
    assert set(modules) - {"wide_model"} == set(actor_code(make_agent()))


def test_actor_code_differences_compare_with_what_the_run_recorded(tmp_path, monkeypatch):
    agent = make_agent()
    # Runs that recorded no network code, or no code_state.yaml at all, can't be checked.
    assert actor_code_differences(str(tmp_path), agent) == ("unrecorded", [])
    write_code_state(str(tmp_path / "params"), record_code_state(make_env()))
    assert actor_code_differences(str(tmp_path), agent) == ("unrecorded", [])

    write_code_state(str(tmp_path / "params"), record_code_state(make_env(), agent=agent))
    assert actor_code_differences(str(tmp_path), agent) == ("unchanged", [])

    recorded = actor_code(agent)
    monkeypatch.setattr(
        "cyclotron.code_state.actor_code",
        lambda _: {**recorded, "rsl_rl.modules.mlp": "0" * 64, "rsl_rl.modules.extra": "1" * 64},
    )
    status, lines = actor_code_differences(str(tmp_path), agent)
    assert status == "changed"
    assert lines == ["rsl_rl.modules.mlp: changed", "rsl_rl.modules.extra: added"]
    monkeypatch.setattr("cyclotron.code_state.actor_code", lambda _: None)
    assert actor_code_differences(str(tmp_path), agent)[0] == "changed"


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_robot_model_records_its_path_commit_and_whether_the_model_repository_is_dirty(tmp_path):
    def git(*args):
        return subprocess.run(["git", "-C", str(tmp_path), *args], check=True, capture_output=True, text=True).stdout

    git("init", "-q")
    git("config", "user.email", "test@example.com")
    git("config", "user.name", "test")
    git("config", "commit.gpgsign", "false")
    (tmp_path / "sim-model" / "urdf").mkdir(parents=True)
    (tmp_path / "sim-model" / "assets").mkdir()
    urdf = tmp_path / "sim-model" / "urdf" / "asimov_1.urdf"
    urdf.write_text('<robot name="asimov_1"/>')
    (tmp_path / "sim-model" / "assets" / "leg.STL").write_text("mesh")
    git("add", ".")
    git("commit", "-q", "-m", "model")
    git("remote", "add", "origin", "https://user:token@github.com/menloresearch/asimov-1.git")
    env = {"scene": {"robot": {"spawn": {"asset_path": str(urdf)}}}}

    robot = record_code_state(env)["robot_model"]
    assert set(robot) == {"name", "repo", "urdf_filepath", "sha256", "commit", "dirty"}
    assert robot["name"] == "asimov_1"
    # The path from the repository's root, and the repository it belongs to, without credentials.
    assert robot["urdf_filepath"] == "sim-model/urdf/asimov_1.urdf" and robot["repo"] == "menloresearch/asimov-1"
    git("remote", "set-url", "origin", "git@github.com:menloresearch/asimov-1.git")
    assert record_code_state(env)["robot_model"]["repo"] == "menloresearch/asimov-1"
    assert robot["commit"] == git("rev-parse", "HEAD").strip() and robot["dirty"] is False

    # A mesh next to the urdf changed: the commit alone isn't the model.
    (tmp_path / "sim-model" / "assets" / "leg.STL").write_text("edited")
    assert record_code_state(env)["robot_model"]["dirty"] is True


def test_robot_model_outside_git_has_only_its_file_name_and_hash(tmp_path):
    urdf = tmp_path / "asimov_1.urdf"
    urdf.write_text("<robot/>")
    env = {"scene": {"robot": {"spawn": {"asset_path": str(urdf)}}}}
    robot = record_code_state(env)["robot_model"]
    assert robot["urdf_filepath"] == "asimov_1.urdf" and robot["sha256"]
    assert robot["repo"] is None and robot["commit"] is None and robot["dirty"] is None
    # <robot/> has no name; a file that isn't xml has none either.
    assert robot["name"] is None
    urdf.write_text("not xml")
    assert record_code_state(env)["robot_model"]["name"] is None


def test_training_robot_model_reads_the_run_record(tmp_path):
    assert training_robot_model(str(tmp_path)) is None
    robot = {"name": "asimov_1", "repo": "o/r", "urdf_filepath": "urdf/asimov_1.urdf"}
    robot.update(sha256="abc", commit="123", dirty=False)
    write_code_state(str(tmp_path / "params"), {"robot_model": robot})
    assert training_robot_model(str(tmp_path)) == robot
    write_code_state(str(tmp_path / "params"), {"robot_model": None})
    assert training_robot_model(str(tmp_path)) is None


def test_robot_model_check_compares_the_urdf_the_current_config_loads(tmp_path):
    urdf = tmp_path / "asimov_1.urdf"
    urdf.write_text('<robot name="asimov_1"/>')
    env = with_robot(make_env(), urdf)
    run = str(tmp_path / "run")
    # Nothing recorded, or no local robot model now: there is nothing to compare.
    assert robot_model_check(run, env) == ("unchecked", None)
    write_code_state(os.path.join(run, "params"), record_code_state(env))
    assert robot_model_check(run, make_env()) == ("unchecked", None)

    assert robot_model_check(run, env) == ("matched", None)
    before = hashlib.sha256(urdf.read_bytes()).hexdigest()[:12]
    urdf.write_text('<robot name="asimov_1"><link name="base"/></robot>')
    after = hashlib.sha256(urdf.read_bytes()).hexdigest()[:12]
    line = f"asimov_1.urdf @ unknown (sha256 {before}) -> asimov_1.urdf @ unknown (sha256 {after})"
    assert robot_model_check(run, env) == ("changed", line)


def test_a_robot_model_edited_at_the_same_commit_names_the_uncommitted_changes():
    record = {"urdf_filepath": "urdf/asimov_1.urdf", "commit": "732cc60dcb8f", "sha256": "bcd9a911c193de"}
    assert _model_where({**record, "dirty": False}) == "urdf/asimov_1.urdf @ 732cc60 (sha256 bcd9a911c193)"
    assert _model_where({**record, "dirty": True}) == (
        "urdf/asimov_1.urdf @ 732cc60 with uncommitted changes (sha256 bcd9a911c193)"
    )
    assert "(uncommitted changes unknown)" in _model_where({**record, "dirty": None})


def test_remote_credentials_are_not_recorded():
    assert _without_credentials("https://user:token@github.com/org/repo.git") == "https://github.com/org/repo.git"
    assert _without_credentials("git@github.com:org/repo.git") == "git@github.com:org/repo.git"
    # No part of a password with an @ in it is left behind.
    assert _without_credentials("https://user:p@ss@host.com/org/repo.git") == "https://host.com/org/repo.git"
    assert _without_credentials("https://token@github.com/org/repo.git") == "https://github.com/org/repo.git"
    # An @ after the host belongs to the path, not to the credentials.
    assert _without_credentials("https://host.com/org/repo@v1.git") == "https://host.com/org/repo@v1.git"


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_dirty_counts_new_files_in_the_package_but_not_elsewhere(tmp_path):
    def git(*args):
        return subprocess.run(["git", "-C", str(tmp_path), *args], check=True, capture_output=True, text=True).stdout

    git("init", "-q")
    git("config", "user.email", "test@example.com")
    git("config", "user.name", "test")
    git("config", "commit.gpgsign", "false")
    package = tmp_path / "source" / "pkg"
    package.mkdir(parents=True)
    (package / "obs.py").write_text("scale = 0.25\n")
    git("add", ".")
    git("commit", "-q", "-m", "trained here")
    assert _git_state(str(package))["dirty"] is False
    (tmp_path / "README.md").write_text("notes\n")
    assert _git_state(str(package))["dirty"] is False
    # A new module training imports but nobody added to git: the commit alone isn't the code.
    (package / "new_reward.py").write_text("weight = 1.0\n")
    assert _git_state(str(package))["dirty"] is True


def test_edited_run_configs_compares_with_the_sha256_training_recorded(tmp_path):
    params = tmp_path / "params"
    params.mkdir()
    (params / "env.yaml").write_text(dump(make_env()))
    (params / "agent.yaml").write_text(dump(make_agent()))
    assert edited_run_configs(str(tmp_path)) is None  # no code_state.yaml
    write_code_state(str(params), record_code_state(make_env()))
    assert edited_run_configs(str(tmp_path)) is None  # trained before training recorded the hashes

    state = record_code_state(make_env())
    state["run_configs"] = hash_run_configs(str(params))
    write_code_state(str(params), state)
    assert edited_run_configs(str(tmp_path)) == []
    (params / "env.yaml").write_text(dump(make_env()).replace("decimation: 4", "decimation: 8"))
    os.remove(params / "agent.yaml")
    assert edited_run_configs(str(tmp_path)) == ["env.yaml", "agent.yaml"]


def test_describe_changes_is_quiet_when_nothing_changed_and_warns_otherwise(tmp_path):
    params = tmp_path / "params"
    params.mkdir()
    (params / "env.yaml").write_text(dump(make_env()))
    (params / "agent.yaml").write_text(dump(make_agent()))
    io = {"actions": {"joint_names": ["left_hip", "right_hip"], "scale": [0.25, 0.25]}, "observations": {}}
    write_code_state(str(params), {"cyclotron": TRAINED_ON, "policy_io": io})
    assert describe_changes(str(tmp_path), make_env(), make_agent(), "Behaviour may differ.", io) == (
        "[INFO] Checked the run against the current code: nothing changed since training.",
        "ok",
    )

    # The robot model resolves the actions to other joints.
    swapped = {**io, "actions": {"joint_names": ["right_hip", "left_hip"], "scale": [0.25, 0.25]}}
    message, level = describe_changes(str(tmp_path), make_env(), make_agent(), "Behaviour may differ.", swapped)
    assert level == "warning"
    assert message.splitlines() == [
        "[WARNING] The code has changed since this run was trained.",
        "  Joints and gains the policy's inputs and outputs resolve to:",
        "    action joint 0: left_hip -> right_hip",
        "    action joint 1: right_hip -> left_hip",
        "  Behaviour may differ.",
    ]

    env, agent = make_env(), make_agent()
    env["actions"]["joint_pos"]["scale"] = 0.5
    message, level = describe_changes(str(tmp_path), env, agent, "Behaviour may differ.")
    assert level == "warning"
    assert message.splitlines() == [
        "[WARNING] The code has changed since this run was trained.",
        "  Policy settings (what the policy sees and does):",
        "    env.actions.joint_pos.scale: 0.25 -> 0.5",
        "  Behaviour may differ.",
    ]
    agent["actor"]["activation"] = "relu"
    message, level = describe_changes(str(tmp_path), env, agent, "Behaviour may differ.")
    assert level == "error" and "agent.actor.activation: elu -> relu" in message and "Behaviour" not in message
    # The error names the commit to check out, from the run's code_state.yaml.
    assert message.splitlines()[-1] == (
        "  Check out the code the run was trained with (exp/drift @ 32aef5c), or train a new run."
    )

    os.remove(params / "env.yaml")
    message, level = describe_changes(str(tmp_path), env, agent, "Behaviour may differ.")
    assert level == "warning" and "can't be checked" in message


def write_git_record(run_dir, name: str, commit: str, branch: str, diff: str = "") -> None:
    git = run_dir / "git"
    git.mkdir(parents=True, exist_ok=True)
    (git / name).write_text(
        f"--- git commit ---\n{commit}\n\n\n--- git status ---\nOn branch {branch}\n\n\n--- git diff ---\n{diff}"
    )


def test_where_says_when_uncommitted_changes_are_unknown():
    assert _where(TRAINED_ON) == "exp/drift @ 32aef5c"
    assert _where({**TRAINED_ON, "dirty": True}) == "exp/drift @ 32aef5c with uncommitted changes"
    assert _where({**TRAINED_ON, "dirty": None}) == "exp/drift @ 32aef5c (uncommitted changes unknown)"
    assert _where({**TRAINED_ON, "branch": None}) == "a detached HEAD @ 32aef5c"
    assert _where({"commit": None}) == "code git doesn't track"


def test_training_code_names_the_commit_to_check_out(tmp_path):
    new_run, old_run, unknown = tmp_path / "new", tmp_path / "old", tmp_path / "unknown"
    write_code_state(
        str(new_run / "params"), {"cyclotron": {"commit": "32aef5c5ec11", "branch": "exp/drift", "dirty": True}}
    )
    assert training_code(str(new_run)) == "exp/drift @ 32aef5c with uncommitted changes"
    # Runs from before code_state.yaml: every commit rsl_rl logged, since either could be the training code.
    write_git_record(old_run, "drift.diff", "32aef5c5ec11641f785bfd3c4aebb9c4de6bdc6b", "exp/drift")
    write_git_record(old_run, "isaac_asimov.diff", "bdf28f5e8b60584fd6b8b50b7433d639c5d8b958", "main", "+w = 1\n")
    assert training_code(str(old_run)) == (
        "exp/drift @ 32aef5c (git/drift.diff) or main @ bdf28f5 with uncommitted changes (git/isaac_asimov.diff)"
    )
    assert check_out_hint(str(old_run)).startswith("Check out the code the run was trained with (exp/drift @")
    unknown.mkdir()
    assert training_code(str(unknown)) is None
    assert check_out_hint(str(unknown)) == (
        "Train a new run, or check out the code the run was trained with (the run doesn't record its commit)."
    )


def test_training_commit_returns_a_single_raw_commit_or_none(tmp_path):
    new_run, old_run, ambiguous = tmp_path / "new", tmp_path / "old", tmp_path / "ambiguous"
    write_code_state(str(new_run / "params"), {"cyclotron": {"commit": "32aef5c5ec11", "dirty": False}})
    assert training_commit(str(new_run)) == "32aef5c5ec11"
    write_code_state(str(new_run / "params"), {"cyclotron": {"commit": "32aef5c5ec11", "dirty": True}})
    assert training_commit(str(new_run)) == "32aef5c5ec11-dirty"
    # Unknown isn't clean: the commit alone isn't known to be the code.
    write_code_state(str(new_run / "params"), {"cyclotron": {"commit": "32aef5c5ec11", "dirty": None}})
    assert training_commit(str(new_run)) == "32aef5c5ec11-dirty"
    write_git_record(old_run, "isaac_asimov.diff", "bdf28f5e8b60584fd6b8b50b7433d639c5d8b958", "main")
    assert training_commit(str(old_run)) == "bdf28f5e8b60584fd6b8b50b7433d639c5d8b958"
    # Two logged repositories: either could be the training code, so no single commit is named.
    write_git_record(ambiguous, "a.diff", "32aef5c5ec11641f785bfd3c4aebb9c4de6bdc6b", "exp/drift")
    write_git_record(ambiguous, "b.diff", "bdf28f5e8b60584fd6b8b50b7433d639c5d8b958", "main")
    assert training_commit(str(ambiguous)) is None
