from __future__ import annotations

import copy
import os
import re
import shutil
import subprocess
from types import SimpleNamespace

import pytest
import torch
import yaml

from cyclotron.code_state import (
    CODE_STATE_FILE,
    _without_credentials,
    code_differences,
    compare_code_state,
    describe_changes,
    files_changed_since,
    hash_files,
    interface_differences,
    load_config,
    load_policy,
    normalize_config,
    policy_interface,
    policy_shape_errors,
    read_git_records,
    rebuild_differences,
    record_code_state,
    training_code,
    training_commit,
    write_code_state,
)

SLOT_JOINTS = ["left_hip_pitch_joint", "right_hip_pitch_joint"]


def dump(config: dict) -> str:
    """Write a config the way Isaac Lab's dump_yaml does: Python tags, keys in their original order."""
    return yaml.dump(config, default_flow_style=False, sort_keys=False)


def term(func: str, scale=None, **params) -> dict:
    return {
        "func": func,
        "params": params,
        "modifiers": None,
        "noise": {"func": "isaaclab.utils.noise.noise_model:uniform_noise", "n_min": -0.01, "n_max": 0.01},
        "clip": None,
        "scale": scale,
        "history_length": 0,
        "flatten_history_dim": True,
    }


def make_env() -> dict:
    """A trimmed env config shaped like Isaac Lab's class_to_dict output, with tuples and slices left in."""
    asset_cfg = {"name": "robot", "joint_names": list(SLOT_JOINTS), "joint_ids": slice(None), "preserve_order": True}
    return {
        "sim": {"dt": 0.005, "gravity": (0.0, 0.0, -9.81)},
        "decimation": 4,
        "seed": 42,
        "scene": {
            "num_envs": 8192,
            "robot": {
                "init_state": {"pos": (0.0, 0.0, 0.639), "joint_pos": {".*_hip_pitch_joint": -0.15}},
                "actuators": {
                    "hip_pitch": {
                        "class_type": "isaaclab.actuators.actuator_pd:DelayedPDActuator",
                        "joint_names_expr": [".*_hip_pitch_joint"],
                        "effort_limit": 45.0,
                        "stiffness": 150.0,
                        "damping": 5.0,
                    }
                },
            },
        },
        "observations": {
            "policy": {
                "concatenate_terms": True,
                "enable_corruption": True,
                "history_length": None,
                "base_ang_vel": term(
                    "cyclotron.tasks.locomotion.mdp.observations:delayed_obs", 0.25, quantity="base_ang_vel"
                ),
                "joint_pos_slot01": term(
                    "isaaclab.envs.mdp.observations:joint_pos_rel", 1.0, asset_cfg=copy.deepcopy(asset_cfg)
                ),
                "actions": term("isaaclab.envs.mdp.observations:last_action"),
            },
            "critic": {"base_lin_vel": term("isaaclab.envs.mdp.observations:base_lin_vel")},
        },
        "actions": {
            "joint_pos": {
                "class_type": "isaaclab.envs.mdp.actions.joint_actions:JointPositionAction",
                "debug_vis": False,
                "joint_names": list(SLOT_JOINTS),
                "scale": 0.25,
                "offset": 0.0,
                "preserve_order": True,
                "use_default_offset": True,
            }
        },
        "commands": {"twist": {"ranges": {"lin_vel_x": (-0.6, 0.8)}}},
        "events": {"push_robot": {"func": "isaaclab.envs.mdp.events:push_by_setting_velocity"}},
    }


def make_agent() -> dict:
    distribution = {"class_name": "GaussianDistribution", "init_std": 1.0}
    return {
        "seed": 1,
        "obs_groups": {"actor": ["policy"], "critic": ["critic"]},
        "clip_actions": None,
        "class_name": "OnPolicyRunner",
        "actor": {
            "class_name": "MLPModel",
            "hidden_dims": [512, 256, 128],
            "activation": "elu",
            "distribution_cfg": distribution,
        },
        "critic": {"class_name": "MLPModel", "hidden_dims": [512, 256, 128], "activation": "elu"},
    }


def differences(env: dict, agent: dict | None = None) -> tuple[list[str], list[str]]:
    """Compare a changed config with the one a run saved, as export and play do."""
    saved = load_config(dump(make_env())), load_config(dump(make_agent()))
    current = normalize_config(env), normalize_config(agent or make_agent())
    return interface_differences(policy_interface(*saved), policy_interface(*current))


def test_saved_yaml_with_python_tags_reads_like_the_current_config():
    text = dump(make_env())
    assert "!!python/tuple" in text and "builtins.slice" in text
    assert differences(make_env()) == ([], [])
    assert load_config(text)["sim"]["gravity"] == [0.0, 0.0, -9.81]


def test_changes_to_what_the_policy_sees_and_does_are_named_as_overrides():
    env, agent = make_env(), make_agent()
    env["observations"]["policy"]["base_ang_vel"]["scale"] = 0.5
    env["observations"]["policy"]["base_ang_vel"]["params"]["max_lag"] = 2
    env["actions"]["joint_pos"]["joint_names"].reverse()
    env["scene"]["robot"]["actuators"]["hip_pitch"]["stiffness"] = 200.0
    env["decimation"] = 2
    agent["actor"]["hidden_dims"] = [256, 128]
    agent["clip_actions"] = 1.0
    # Settings only the current config has come last.
    assert differences(env, agent) == (
        [],
        [
            "env.observations.policy.base_ang_vel.scale: 0.25 -> 0.5",
            "env.actions.joint_pos.joint_names: same entries in a different order",
            "env.scene.robot.actuators.hip_pitch.stiffness: 150.0 -> 200.0",
            "env.decimation: 4 -> 2",
            "agent.actor.hidden_dims: [512, 256, 128] -> [256, 128]",
            "agent.clip_actions: none -> 1.0",
            "env.observations.policy.base_ang_vel.params.max_lag: none -> 2",
        ],
    )


def test_a_network_change_that_keeps_the_weight_shapes_is_an_error():
    agent = make_agent()
    agent["actor"]["activation"] = "relu"
    assert differences(make_env(), agent) == (["agent.actor.activation: elu -> relu"], [])


def test_added_removed_and_reordered_terms_are_reported_once():
    env = make_env()
    policy = env["observations"]["policy"]
    policy["foot_height"] = term("cyclotron.tasks.locomotion.mdp.observations:foot_height")
    del policy["actions"]
    assert differences(env) == (
        [],
        [
            "env.observations.policy: term actions removed",
            "env.observations.policy: term foot_height added",
        ],
    )
    reordered = make_env()
    reordered["observations"]["policy"] = dict(reversed(list(reordered["observations"]["policy"].items())))
    assert differences(reordered) == (
        [],
        [
            "env.observations.policy: terms reordered, base_ang_vel, joint_pos_slot01, actions"
            " -> actions, joint_pos_slot01, base_ang_vel"
        ],
    )


def test_training_only_and_play_changes_are_ignored():
    env, agent = make_env(), make_agent()
    # The experiment branches' privileged critic input, and what the play tasks change.
    env["observations"]["critic"]["path_error"] = term("cyclotron.tasks.locomotion.mdp.drift:path_error")
    env["observations"]["policy"]["enable_corruption"] = False
    for name in ("base_ang_vel", "joint_pos_slot01", "actions"):
        env["observations"]["policy"][name]["noise"] = None
    env["commands"]["twist"]["ranges"]["lin_vel_x"] = (0.6, 0.8)
    env["events"]["push_robot"] = None
    env["actions"]["joint_pos"]["debug_vis"] = True
    env["scene"]["num_envs"] = 1
    env["seed"] = 7
    agent["critic"]["hidden_dims"] = [1024]
    agent["actor"]["distribution_cfg"]["init_std"] = 0.5
    assert differences(env, agent) == ([], [])


def test_renamed_modules_and_new_unset_fields_are_not_changes():
    env = make_env()
    env["observations"]["policy"]["base_ang_vel"]["func"] = "isaac_asimov.tasks.locomotion.mdp.observations:delayed_obs"
    env["actions"]["joint_pos"]["clip"] = None  # a field a newer Isaac Lab adds, unset
    assert differences(env) == ([], [])


def test_hashes_cover_training_code_only(tmp_path):
    (tmp_path / "tasks" / "__pycache__").mkdir(parents=True)
    (tmp_path / "tasks" / "env_cfg.py").write_text("scale = 0.25\n")
    (tmp_path / "tasks" / "__pycache__" / "env_cfg.cpython-311.pyc").write_bytes(b"")
    (tmp_path / "hub.py").write_text("")
    assert list(hash_files(str(tmp_path))) == ["tasks/env_cfg.py"]


def test_compare_code_state_lists_files_dependencies_and_versions(tmp_path):
    package = tmp_path / "pkg"
    (package / "tasks").mkdir(parents=True)
    (package / "tasks" / "env_cfg.py").write_text("scale = 0.25\n")
    (package / "old.py").write_text("")
    saved = record_code_state(package=str(package))
    assert compare_code_state(saved, record_code_state(package=str(package))) == []

    (package / "tasks" / "env_cfg.py").write_text("scale = 0.5\n")
    (package / "old.py").unlink()
    (package / "new.py").write_text("")
    current = record_code_state(package=str(package))
    current["packages"]["rsl-rl-lib"] = "9.9.9"
    current["robot_model"] = {"file": "asimov_1.urdf", "sha256": "abc", "commit": "1234567890"}
    assert compare_code_state(saved, current) == [
        "trained on a detached HEAD @ unknown, now a detached HEAD @ unknown",
        "cyclotron files changed: tasks/env_cfg.py",
        "cyclotron files added: new.py",
        "cyclotron files removed: old.py",
        "robot model changed: unknown @ unknown -> asimov_1.urdf @ 1234567",
        f"rsl-rl-lib: {saved['packages']['rsl-rl-lib']} -> 9.9.9",
    ]


def test_code_state_records_the_robot_model_and_no_local_paths(tmp_path):
    urdf = tmp_path / "asimov_1.urdf"
    urdf.write_text("<robot/>")
    env = {"scene": {"robot": {"spawn": {"asset_path": str(urdf)}}}}
    state = record_code_state(env, loaded_checkpoint="2026-09-26_base/model_500.pt")
    write_code_state(str(tmp_path / "params"), state)
    text = (tmp_path / "params" / CODE_STATE_FILE).read_text()
    assert state["robot_model"]["file"] == "asimov_1.urdf" and str(tmp_path) not in text
    assert yaml.safe_load(text)["loaded_checkpoint"] == "2026-09-26_base/model_500.pt"


def test_remote_credentials_are_not_recorded():
    assert _without_credentials("https://user:token@github.com/org/repo.git") == "https://github.com/org/repo.git"
    assert _without_credentials("git@github.com:org/repo.git") == "git@github.com:org/repo.git"


def test_old_runs_fall_back_to_the_git_records_rsl_rl_wrote(tmp_path):
    git = tmp_path / "git"
    git.mkdir()
    header = "--- git commit ---\n{}\n\n\n--- git status ---\nOn branch {}\nnothing to commit\n\n\n--- git diff ---\n{}"
    (git / "drift.diff").write_text(header.format("32aef5c5ec11641f785bfd3c4aebb9c4de6bdc6b", "exp/drift", ""))
    (git / "wip.diff").write_text(header.format("bdf28f5e8b60584fd6b8b50b7433d639c5d8b958", "main", "+scale = 1\n"))
    assert read_git_records(str(tmp_path)) == [
        {
            "file": "drift.diff",
            "commit": "32aef5c5ec11641f785bfd3c4aebb9c4de6bdc6b",
            "branch": "exp/drift",
            "dirty": False,
        },
        {"file": "wip.diff", "commit": "bdf28f5e8b60584fd6b8b50b7433d639c5d8b958", "branch": "main", "dirty": True},
    ]
    empty = tmp_path / "empty"
    empty.mkdir()
    assert code_differences(str(empty), record_code_state()) == [
        "the run has no record of its code (no params/code_state.yaml or git/), so code changes can't be checked"
    ]


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_files_changed_since_ignores_pure_renames(tmp_path):
    def git(*args):
        return subprocess.run(["git", "-C", str(tmp_path), *args], check=True, capture_output=True, text=True).stdout

    git("init", "-q")
    git("config", "user.email", "test@example.com")
    git("config", "user.name", "test")
    git("config", "commit.gpgsign", "false")
    (tmp_path / "source" / "old_name").mkdir(parents=True)
    (tmp_path / "source" / "old_name" / "rewards.py").write_text("weight = 1.0\n" * 20)
    (tmp_path / "source" / "old_name" / "obs.py").write_text("scale = 0.25\n" * 20)
    git("add", ".")
    git("commit", "-q", "-m", "trained here")
    commit = git("rev-parse", "HEAD").strip()
    git("mv", "source/old_name", "source/new_name")
    (tmp_path / "source" / "new_name" / "obs.py").write_text("scale = 0.5\n" + "scale = 0.25\n" * 19)
    assert files_changed_since(str(tmp_path), commit) == ["source/new_name/obs.py"]
    assert files_changed_since(str(tmp_path), "0" * 40) is None


def test_rebuild_differences_flags_values_unlike_the_run_and_lists_settings_it_did_not_save(tmp_path):
    params = tmp_path / "params"
    params.mkdir()
    (params / "env.yaml").write_text(dump(make_env()))
    (params / "agent.yaml").write_text(dump(make_agent()))
    assert rebuild_differences(str(tmp_path), make_env(), make_agent()) == ([], [])

    env, agent = make_env(), make_agent()
    env["actions"]["joint_pos"]["scale"] = 0.5
    env["actions"]["joint_pos"]["clip"] = {".*": (-1.0, 1.0)}  # a setting added to the code since training
    agent["actor"]["activation"] = "relu"
    assert rebuild_differences(str(tmp_path), env, agent) == (
        ["env.actions.joint_pos.scale: 0.25 -> 0.5", "agent.actor.activation: elu -> relu"],
        ["env.actions.joint_pos.clip[.*]: none -> [-1.0, 1.0]"],
    )


def test_describe_changes_is_quiet_when_nothing_changed_and_warns_otherwise(tmp_path):
    params = tmp_path / "params"
    params.mkdir()
    (params / "env.yaml").write_text(dump(make_env()))
    (params / "agent.yaml").write_text(dump(make_agent()))
    write_code_state(str(params), record_code_state(make_env()))
    assert describe_changes(str(tmp_path), make_env(), make_agent(), "Behaviour may differ.") == (
        "[INFO] Checked the run against the current code: nothing changed since training.",
        "ok",
    )

    # Only the code changed.
    state = record_code_state(make_env())
    state["cyclotron"]["files"]["removed_since.py"] = "0" * 64
    write_code_state(str(params), state)
    message, level = describe_changes(str(tmp_path), make_env(), make_agent(), "Behaviour may differ.")
    assert level == "warning" and "  Code:" in message and "Policy settings" not in message
    write_code_state(str(params), record_code_state(make_env()))

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
    assert re.fullmatch(
        r"  Check out the code the run was trained with \(.+ @ [0-9a-f]{7}.*\), or train a new run\.",
        message.splitlines()[-1],
    )

    os.remove(params / "env.yaml")
    message, level = describe_changes(str(tmp_path), env, agent, "Behaviour may differ.")
    assert level == "warning" and "can't be checked" in message


def test_policy_shape_errors_name_the_inputs_outputs_and_layers():
    saved = {"mlp.0.weight": torch.zeros(512, 78), "mlp.2.weight": torch.zeros(23, 512), "std": torch.zeros(23)}
    current = {"mlp.0.weight": torch.zeros(512, 55), "mlp.2.weight": torch.zeros(12, 512), "std": torch.zeros(12)}
    assert policy_shape_errors(saved, current) == [
        "mlp.0.weight: checkpoint [512, 78], current code [512, 55] (the policy takes 78 inputs; the current"
        " observations give 55)",
        "mlp.2.weight: checkpoint [23, 512], current code [12, 512] (the policy gives 23 actions; the current code"
        " expects 12)",
        "std: checkpoint [23], current code [12]",
    ]
    rnn = {"rnn.weight_ih_l0": torch.zeros(256, 78), "mlp.0.weight": torch.zeros(23, 256)}
    assert policy_shape_errors(rnn, {"mlp.0.weight": torch.zeros(512, 78)}) == [
        "the network's layers changed:",
        "  only in the checkpoint: rnn.weight_ih_l0",
        "mlp.0.weight: checkpoint [23, 256], current code [512, 78]",
    ]
    assert policy_shape_errors(saved, saved) == []


def write_git_record(run_dir, name: str, commit: str, branch: str, diff: str = "") -> None:
    git = run_dir / "git"
    git.mkdir(exist_ok=True)
    (git / name).write_text(
        f"--- git commit ---\n{commit}\n\n\n--- git status ---\nOn branch {branch}\n\n\n--- git diff ---\n{diff}"
    )


def test_training_code_names_the_commit_to_check_out(tmp_path):
    new_run, old_run, unknown = tmp_path / "new", tmp_path / "old", tmp_path / "unknown"
    write_code_state(
        str(new_run / "params"), {"cyclotron": {"commit": "32aef5c5ec11", "branch": "exp/drift", "dirty": True}}
    )
    assert training_code(str(new_run)) == "exp/drift @ 32aef5c with uncommitted changes"
    # Runs from before code_state.yaml: every commit rsl_rl logged, since either could be the training code.
    old_run.mkdir()
    write_git_record(old_run, "drift.diff", "32aef5c5ec11641f785bfd3c4aebb9c4de6bdc6b", "exp/drift")
    write_git_record(old_run, "isaac_asimov.diff", "bdf28f5e8b60584fd6b8b50b7433d639c5d8b958", "main")
    assert training_code(str(old_run)) == (
        "exp/drift @ 32aef5c (git/drift.diff) or main @ bdf28f5 (git/isaac_asimov.diff)"
    )
    unknown.mkdir()
    assert training_code(str(unknown)) is None


def test_training_commit_returns_a_single_raw_commit_or_none(tmp_path):
    new_run, old_run, ambiguous = tmp_path / "new", tmp_path / "old", tmp_path / "ambiguous"
    write_code_state(str(new_run / "params"), {"cyclotron": {"commit": "32aef5c5ec11", "branch": "exp/drift"}})
    assert training_commit(str(new_run)) == "32aef5c5ec11"
    old_run.mkdir()
    write_git_record(old_run, "isaac_asimov.diff", "bdf28f5e8b60584fd6b8b50b7433d639c5d8b958", "main")
    assert training_commit(str(old_run)) == "bdf28f5e8b60584fd6b8b50b7433d639c5d8b958"
    # Two logged repositories: either could be the training code, so no single commit is named.
    ambiguous.mkdir()
    write_git_record(ambiguous, "a.diff", "32aef5c5ec11641f785bfd3c4aebb9c4de6bdc6b", "exp/drift")
    write_git_record(ambiguous, "b.diff", "bdf28f5e8b60584fd6b8b50b7433d639c5d8b958", "main")
    assert training_commit(str(ambiguous)) is None


def test_load_policy_loads_only_the_actor_or_names_the_training_code(tmp_path):
    torch.manual_seed(0)
    trained = torch.nn.Sequential(torch.nn.Linear(78, 8), torch.nn.ELU(), torch.nn.Linear(8, 23))
    checkpoint = tmp_path / "model_1.pt"
    torch.save(
        {"actor_state_dict": trained.state_dict(), "critic_state_dict": {"0.weight": torch.zeros(1, 96)}}, checkpoint
    )
    write_git_record(tmp_path, "drift.diff", "32aef5c5ec11641f785bfd3c4aebb9c4de6bdc6b", "exp/drift")

    loads = []
    policy = {"network": torch.nn.Sequential(torch.nn.Linear(78, 8), torch.nn.ELU(), torch.nn.Linear(8, 23))}
    runner = SimpleNamespace(
        alg=SimpleNamespace(get_policy=lambda: policy["network"]),
        load=lambda path, load_cfg=None: loads.append(load_cfg),
    )
    load_policy(runner, str(checkpoint), "OnPolicyRunner", str(tmp_path))
    assert loads == [{"actor": True}]

    policy["network"] = torch.nn.Sequential(torch.nn.Linear(55, 8), torch.nn.ELU(), torch.nn.Linear(8, 23))
    with pytest.raises(ValueError) as error:
        load_policy(runner, str(checkpoint), "OnPolicyRunner", str(tmp_path))
    assert str(error.value).splitlines() == [
        "The checkpoint's policy doesn't fit the network the current code builds:",
        "  0.weight: checkpoint [8, 78], current code [8, 55] (the policy takes 78 inputs; the current observations"
        " give 55)",
        "The policy's inputs or network changed since training; see the warning above.",
        "Check out the code the run was trained with (exp/drift @ 32aef5c (git/drift.diff)), or train a new run.",
    ]
    assert loads == [{"actor": True}]
