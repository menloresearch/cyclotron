from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest
import yaml

from cyclotron.run_config import (
    GENERATED_HEADER,
    generated_run_configs,
    load_run_configs,
    mark_generated,
    missing_run_configs,
    restore_policy_settings,
)

# Plain dataclasses shaped like the Isaac Lab and rsl_rl configs export restores. Like Isaac Lab's configclass, some
# defaults are factories.


def joint_pos_rel():
    """Stands in for an observation function."""


@dataclass
class Term:
    func: object = None
    scale: float = 1.0
    clip: tuple | None = None


@dataclass
class Group:
    history_length: int = 0
    base_ang_vel: Term | None = None
    joint_pos: Term | None = None
    foot_contact: Term | None = None


@dataclass
class ActionTerm:
    class_type: object = None
    scale: float = 1.0


@dataclass
class Actions:
    joint_pos: ActionTerm | None = None


@dataclass
class Actuator:
    class_type: object = None
    stiffness: dict = field(default_factory=dict)


@dataclass
class ModelCfg:
    class_name: str = field(default_factory=lambda: "MLPModel")
    hidden_dims: list = field(default_factory=lambda: [512, 256])
    activation: str = "elu"


@dataclass
class RNNModelCfg(ModelCfg):
    class_name: str = field(default_factory=lambda: "RNNModel")
    rnn_type: str = "gru"
    rnn_hidden_dim: int = 64


def make_cfgs():
    """The task's configs as the current code builds them."""
    policy = Group(
        base_ang_vel=Term(func=math.sin, scale=0.25),
        joint_pos=Term(func=joint_pos_rel),
        foot_contact=Term(func=joint_pos_rel),  # added since the run was trained
    )
    robot = SimpleNamespace(
        init_state=SimpleNamespace(joint_pos={".*": 0.0, ".*_knee": 0.3}),
        actuators={
            "legs": Actuator(class_type=Actuator, stiffness={".*": 100.0}),
            "arms": Actuator(class_type=Actuator),
        },
    )
    env = SimpleNamespace(
        observations=SimpleNamespace(policy=policy, critic=Group()),
        actions=Actions(joint_pos=ActionTerm(class_type=ActionTerm, scale=0.25)),
        scene=SimpleNamespace(robot=robot),
        sim=SimpleNamespace(dt=0.005),
        decimation=4,
    )
    agent = SimpleNamespace(obs_groups={"actor": ["policy"], "critic": ["critic"]}, actor=ModelCfg(), clip_actions=None)
    return env, agent


def saved_configs():
    """What the run saved, as Isaac Lab's dump_yaml writes it (functions as "module:name")."""
    env = {
        "observations": {
            "policy": {
                "history_length": 3,
                "base_ang_vel": {"func": "math:cos", "scale": 0.2, "clip": (-5.0, 5.0)},
                # The package was renamed since; the current code has the same function under its new name.
                "joint_pos": {"func": "isaac_asimov.tasks.mdp:joint_pos_rel", "scale": 1.0, "clip": None},
            },
            "critic": {"history_length": 7},
        },
        "actions": {"joint_pos": {"class_type": "isaac_asimov.actions:ActionTerm", "scale": 0.5}},
        "scene": {
            "robot": {
                "init_state": {"joint_pos": {"left_hip": -0.15, ".*_knee": 0.3}},
                "actuators": {"legs": {"class_type": "isaac_asimov.actuators:Actuator", "stiffness": {".*_hip": 80.0}}},
            }
        },
        "sim": {"dt": 0.004},
        "decimation": 5,
    }
    agent = {
        "class_name": "OnPolicyRunner",
        "obs_groups": {"actor": ["policy"], "critic": ["critic"]},
        "actor": {"class_name": "RNNModel", "hidden_dims": [256, 128], "rnn_type": "lstm", "rnn_hidden_dim": 256},
        "clip_actions": 1.0,
    }
    return env, agent


def test_restore_sets_the_policy_settings_back_to_the_run():
    env, agent = make_cfgs()
    assert restore_policy_settings(env, agent, *saved_configs()) == []

    policy = env.observations.policy
    assert policy.history_length == 3
    assert policy.base_ang_vel.func is math.cos and policy.base_ang_vel.scale == 0.2
    assert policy.base_ang_vel.clip == (-5.0, 5.0)
    assert policy.joint_pos.func is joint_pos_rel
    assert policy.foot_contact is None
    # Only the actor's observation groups are restored.
    assert env.observations.critic.history_length == 0

    assert env.actions.joint_pos.class_type is ActionTerm and env.actions.joint_pos.scale == 0.5
    assert env.scene.robot.init_state.joint_pos == {"left_hip": -0.15, ".*_knee": 0.3}
    assert list(env.scene.robot.actuators) == ["legs"]
    assert env.scene.robot.actuators["legs"].stiffness == {".*_hip": 80.0}
    assert env.sim.dt == 0.004 and env.decimation == 5

    # The actor is rebuilt as the config class of the network the run trained: an LSTM, not the code's MLP.
    assert isinstance(agent.actor, RNNModelCfg)
    assert (agent.actor.hidden_dims, agent.actor.rnn_type, agent.actor.rnn_hidden_dim) == ([256, 128], "lstm", 256)
    assert agent.obs_groups == {"actor": ["policy"], "critic": ["critic"]}
    assert agent.clip_actions == 1.0


def test_restore_leaves_a_side_the_run_did_not_save_as_the_code_has_it():
    env, agent = make_cfgs()
    assert restore_policy_settings(env, agent, saved_configs()[0], {}) == []
    assert env.decimation == 5
    assert type(agent.actor) is ModelCfg and agent.clip_actions is None


def test_restore_lists_the_settings_the_current_code_cannot_take():
    env, agent = make_cfgs()
    saved_env, saved_agent = saved_configs()
    saved_env["observations"]["policy"]["height_scan"] = {"func": "math:cos", "scale": 1.0}
    saved_env["observations"]["policy"]["joint_pos"]["func"] = "isaac_asimov.tasks.mdp:joint_pos_abs"
    saved_env["scene"]["robot"]["actuators"]["waist"] = {"class_type": "isaac_asimov.actuators:Actuator"}
    saved_agent["actor"]["class_name"] = "TransformerModel"
    assert restore_policy_settings(env, agent, saved_env, saved_agent) == [
        "env.observations.policy.joint_pos.func: the run used isaac_asimov.tasks.mdp:joint_pos_abs, which the current"
        " code doesn't have",
        "env.observations.policy.height_scan: the run has this setting, the current code doesn't",
        "env.scene.robot.actuators.waist: the run has this entry, the current code doesn't",
        "agent.actor.class_name: the run used TransformerModel, which no config class in the current code has",
        "agent.actor.rnn_type: the run has this setting, the current code doesn't",
        "agent.actor.rnn_hidden_dim: the run has this setting, the current code doesn't",
    ]


def test_load_run_configs_reads_tuples_and_slices_but_no_other_python_objects(tmp_path):
    params = tmp_path / "params"
    params.mkdir()
    assert load_run_configs(str(tmp_path)) == ({}, {})
    (params / "env.yaml").write_text(
        "clip: !!python/tuple\n- -5.0\n- 5.0\njoint_ids: !!python/object/apply:builtins.slice\n- null\n- null\n- null\n"
    )
    (params / "agent.yaml").write_text("seed: 42\n")
    assert load_run_configs(str(tmp_path)) == ({"clip": (-5.0, 5.0), "joint_ids": slice(None)}, {"seed": 42})
    (params / "agent.yaml").write_text("seed: !!python/object/apply:os.getcwd []\n")
    with pytest.raises(yaml.constructor.ConstructorError):
        load_run_configs(str(tmp_path))


def test_generated_run_configs_are_found_by_their_first_line(tmp_path):
    params = tmp_path / "params"
    params.mkdir()
    assert missing_run_configs(str(tmp_path)) == ["env.yaml", "agent.yaml"]
    (params / "env.yaml").write_text("decimation: 4\n")
    (params / "agent.yaml").write_text("seed: 42\n")
    assert missing_run_configs(str(tmp_path)) == [] and generated_run_configs(str(tmp_path)) == []
    mark_generated(str(params / "agent.yaml"), "main @ 1234567", "2026-10-07 12:00")
    assert generated_run_configs(str(tmp_path)) == ["agent.yaml"]
    assert (params / "agent.yaml").read_text().startswith(f"{GENERATED_HEADER} (main @ 1234567) on 2026-10-07 12:00:")
    # The header is a yaml comment, so the file still loads as before.
    assert load_run_configs(str(tmp_path)) == ({"decimation": 4}, {"seed": 42})
