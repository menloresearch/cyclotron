from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch

from cyclotron import policy_io
from cyclotron.policy_io import observation_names, policy_io_differences, resolve_policy_io

ROBOT_JOINTS = ["left_hip", "right_hip", "left_knee", "right_knee"]
ROBOT_BODIES = ["pelvis", "left_foot", "right_foot"]


class FakeJointPositionAction:
    """Shaped like the parts of Isaac Lab's JointPositionAction that are resolved."""

    def __init__(self, joint_ids, scale=0.25, offset=0.0, clip=None):
        self._joint_ids = joint_ids
        ids = range(len(ROBOT_JOINTS)) if isinstance(joint_ids, slice) else joint_ids
        self._joint_names = [ROBOT_JOINTS[i] for i in ids]
        self._scale, self._offset = scale, offset
        self.cfg = SimpleNamespace(use_default_offset=False, clip=clip)
        if clip is not None:
            self._clip = torch.tensor([clip])
        data = SimpleNamespace(
            default_joint_stiffness=torch.tensor([[150.0, 150.0, 200.0, 200.0]]),
            default_joint_damping=torch.tensor([[5.0, 5.0, 8.0, 8.0]]),
        )
        self._asset = SimpleNamespace(data=data)


class OtherAction:
    pass


def entity(joint_ids=slice(None), joint_names=None, body_ids=slice(None), body_names=None):
    """A resolved SceneEntityCfg: the patterns as configured, the ids as resolved."""
    return SimpleNamespace(
        name="robot", joint_ids=joint_ids, joint_names=joint_names, body_ids=body_ids, body_names=body_names
    )


def make_env(*terms, observations=None):
    actions = {f"term{i}": term for i, term in enumerate(terms)}
    observations = observations or {
        "policy": {
            "base_ang_vel": SimpleNamespace(params={}),
            "joint_pos": SimpleNamespace(params={"asset_cfg": entity([2, 3, 0, 1], [".*_knee", ".*_hip"])}),
            "feet": SimpleNamespace(params={"sensor_cfg": entity(body_ids=[1, 2], body_names=[".*_foot"])}),
            "everything": SimpleNamespace(params={"asset_cfg": entity(slice(None), [".*"])}),
        },
        "critic": {"base_lin_vel": SimpleNamespace(params={"asset_cfg": entity()})},
    }
    return SimpleNamespace(
        action_manager=SimpleNamespace(active_terms=list(actions), get_term=actions.__getitem__),
        observation_manager=SimpleNamespace(
            active_terms={group: list(terms) for group, terms in observations.items()},
            _group_obs_term_cfgs={group: list(terms.values()) for group, terms in observations.items()},
        ),
        scene={"robot": SimpleNamespace(joint_names=ROBOT_JOINTS, body_names=ROBOT_BODIES)},
    )


AGENT = {"class_name": "OnPolicyRunner", "obs_groups": {"actor": ["policy"], "critic": ["critic"]}}


@pytest.fixture(autouse=True)
def fake_isaaclab(monkeypatch):
    monkeypatch.setattr(policy_io, "_joint_position_action", lambda: FakeJointPositionAction)


def test_actions_resolve_to_joints_in_action_order_with_their_values():
    env = make_env(
        FakeJointPositionAction([2, 3], scale=0.5, clip=[[-1.0, 1.0], [float("-inf"), float("inf")]]),
        FakeJointPositionAction([0, 1], offset=-0.1),
    )
    actions = resolve_policy_io(env, AGENT)["actions"]
    assert actions == {
        "joint_names": ["left_knee", "right_knee", "left_hip", "right_hip"],
        "scale": [0.5, 0.5, 0.25, 0.25],
        "offset": [0.0, 0.0, -0.1, -0.1],
        # An unclipped side reads as None; a term without a clip leaves its joints unclipped.
        "clip": [[-1.0, 1.0], [None, None], [None, None], [None, None]],
        "stiffness": [200.0, 200.0, 150.0, 150.0],
        "damping": [8.0, 8.0, 5.0, 5.0],
    }
    assert resolve_policy_io(env := make_env(FakeJointPositionAction(slice(None))), AGENT)["actions"]["clip"] is None
    assert resolve_policy_io(env, AGENT)["actions"]["joint_names"] == ROBOT_JOINTS


def test_an_action_that_isnt_a_joint_position_target_is_not_resolved():
    io = resolve_policy_io(make_env(FakeJointPositionAction([0]), OtherAction()), AGENT)
    assert io["actions"] is None
    assert io["unsupported"] == "Action term term1 (OtherAction) is not a joint position action."
    direct = resolve_policy_io(SimpleNamespace(), AGENT)
    assert direct["actions"] is None and direct["observations"] == {}
    assert "direct workflow" in direct["unsupported"]


def test_observations_record_the_joints_and_bodies_each_actor_term_reads():
    io = resolve_policy_io(make_env(FakeJointPositionAction([0])), AGENT)
    # Only the actor's groups; terms that select no joints or bodies (all of them, unnamed) record nothing.
    assert io["observations"] == {
        "policy": {
            "base_ang_vel": {},
            "joint_pos": {"asset_cfg": {"joint_names": ["left_knee", "right_knee", "left_hip", "right_hip"]}},
            "feet": {"sensor_cfg": {"body_names": ["left_foot", "right_foot"]}},
            "everything": {"asset_cfg": {"joint_names": ROBOT_JOINTS}},
        }
    }
    assert observation_names(io) == ["base_ang_vel", "joint_pos", "feet", "everything"]
    # A distilled student reads its own groups; with several, the names say which group.
    student = {"class_name": "DistillationRunner", "obs_groups": {"student": ["policy", "critic"]}}
    both = resolve_policy_io(make_env(FakeJointPositionAction([0])), student)
    assert observation_names(both) == [
        "policy/base_ang_vel",
        "policy/joint_pos",
        "policy/feet",
        "policy/everything",
        "critic/base_lin_vel",
    ]


def recorded_io() -> dict:
    return resolve_policy_io(make_env(FakeJointPositionAction([0, 1, 2, 3])), AGENT)


def test_the_same_resolution_has_no_differences_within_float_tolerance():
    io = recorded_io()
    assert policy_io_differences(io, copy.deepcopy(io)) == []
    nearly = copy.deepcopy(io)
    nearly["actions"]["stiffness"][0] += 1e-5
    assert policy_io_differences(io, nearly) == []


def test_reordered_joints_and_changed_gains_are_named_by_joint():
    io, now = recorded_io(), recorded_io()
    actions = now["actions"]
    for key in ("joint_names", "scale", "offset", "stiffness", "damping"):
        actions[key][0], actions[key][1] = actions[key][1], actions[key][0]
    actions["stiffness"][2] = 180.0
    actions["clip"] = [[-1.0, 1.0]] + [[None, None]] * 3
    # A reorder is reported once, not as every value changing.
    assert policy_io_differences(io, now) == [
        "action joint 0: left_hip -> right_hip",
        "action joint 1: right_hip -> left_hip",
        "stiffness left_knee: 200 -> 180",
        "clip right_hip: [none, none] -> [-1, 1]",
    ]
    fewer = recorded_io()
    fewer["actions"]["joint_names"] = ["left_hip", "right_hip", "left_knee"]
    assert policy_io_differences(io, fewer) == ["action joint: 4 -> 3; removed right_knee"]


def test_observation_order_changes_and_unsupported_actions_are_reported():
    io, now = recorded_io(), recorded_io()
    now["observations"]["policy"]["joint_pos"]["asset_cfg"]["joint_names"].reverse()
    del now["observations"]["policy"]["feet"]
    # A removed term is a settings change, which the configs show.
    assert policy_io_differences(io, now) == [
        "observation policy/joint_pos asset_cfg joint 0: left_knee -> right_hip",
        "observation policy/joint_pos asset_cfg joint 1: right_knee -> left_hip",
        "observation policy/joint_pos asset_cfg joint 2: left_hip -> right_knee",
        "observation policy/joint_pos asset_cfg joint 3: right_hip -> left_knee",
    ]
    unsupported = resolve_policy_io(make_env(OtherAction()), AGENT)
    assert policy_io_differences(io, unsupported) == [
        "actions: training resolved them as joint position targets, now: Action term term0 (OtherAction) is not a"
        " joint position action."
    ]
    # Training couldn't resolve them either: nothing to compare.
    assert policy_io_differences(unsupported, io) == []
