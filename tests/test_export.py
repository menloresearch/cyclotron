from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
import torch
from rsl_rl.models import MLPModel
from rsl_rl.runners import OnPolicyRunner
from tensordict import TensorDict

from cyclotron.hub import infer_task
from cyclotron.onnx_export import (
    attach_deploy_metadata,
    copy_run_yamls,
    deploy_metadata,
    existing_export_note,
    max_onnx_difference,
)


def make_run(
    tmp_path, agent_yaml: str | None = "experiment_name: asimov_velocity_amp\n", env_yaml: str | None = "seed: 42\n"
):
    params = tmp_path / "run" / "params"
    params.mkdir(parents=True)
    if agent_yaml is not None:
        (params / "agent.yaml").write_text(agent_yaml)
    if env_yaml is not None:
        (params / "env.yaml").write_text(env_yaml)
    return str(tmp_path / "run")


def test_infer_task_reads_experiment_name(tmp_path):
    assert infer_task(make_run(tmp_path)) == "Asimov1-Velocity-AMP-v0"


def test_infer_task_rejects_unknown_experiment_and_missing_agent_yaml(tmp_path):
    with pytest.raises(ValueError, match="experiment 'other'"):
        infer_task(make_run(tmp_path / "a", agent_yaml="experiment_name: other\n"))
    with pytest.raises(ValueError, match="agent.yaml not found"):
        infer_task(make_run(tmp_path / "b", agent_yaml=None))


def test_copy_run_yamls_copies_unchanged_and_reports_missing(tmp_path):
    run_dir = make_run(tmp_path, agent_yaml=None, env_yaml="asset_path: /home/me/asimov_1.urdf\n")
    output_dir = tmp_path / "exported"
    output_dir.mkdir()
    assert copy_run_yamls(run_dir, str(output_dir)) == ["agent.yaml"]
    assert (output_dir / "env.yaml").read_text() == "asset_path: /home/me/asimov_1.urdf\n"
    assert not (output_dir / "code_state.yaml").exists()


def test_copy_run_yamls_copies_code_state_when_the_run_has_one(tmp_path):
    run_dir = make_run(tmp_path)
    with open(os.path.join(run_dir, "params", "code_state.yaml"), "w") as f:
        f.write("packages: {}\n")
    output_dir = tmp_path / "exported"
    output_dir.mkdir()
    assert copy_run_yamls(run_dir, str(output_dir)) == []
    assert (output_dir / "code_state.yaml").read_text() == "packages: {}\n"


def make_policy(obs: TensorDict, obs_normalization: bool) -> MLPModel:
    # The actor reads two observation groups, so the check has to concatenate them in the policy's order.
    obs_groups = {"actor": ["policy", "extra"], "critic": ["policy"]}
    distribution = {"class_name": "GaussianDistribution", "init_std": 1.0}
    torch.manual_seed(0)
    return MLPModel(
        obs, obs_groups, "actor", 4, [16, 16], obs_normalization=obs_normalization, distribution_cfg=distribution
    )


def export(policy: MLPModel, path) -> str:
    # The runner's own export, which only needs the policy from the algorithm.
    runner = SimpleNamespace(alg=SimpleNamespace(get_policy=lambda: policy))
    OnPolicyRunner.export_policy_to_onnx(runner, path=str(path), filename="policy.onnx")
    return os.path.join(path, "policy.onnx")


@pytest.fixture
def trained():
    obs = TensorDict(
        {"policy": torch.randn(32, 6) * 3 + 2, "extra": torch.randn(32, 2), "critic": torch.zeros(32, 1)},
        batch_size=[32],
    )
    policy = make_policy(obs, obs_normalization=True)
    policy.update_normalization(obs)
    return policy.eval(), obs


def test_exported_onnx_matches_policy(tmp_path, trained):
    policy, obs = trained
    assert max_onnx_difference(policy, obs, export(policy, tmp_path)) < 1e-5


def test_onnx_without_the_normalizer_is_caught(tmp_path, trained):
    policy, obs = trained
    unnormalized = make_policy(obs, obs_normalization=False)
    unnormalized.mlp.load_state_dict(policy.mlp.state_dict())
    unnormalized.distribution.load_state_dict(policy.distribution.state_dict())
    assert max_onnx_difference(policy, obs, export(unnormalized, tmp_path)) > 1e-2


def sample_metadata(**overrides):
    values = dict(
        joint_names=["hip", "knee", "ankle", "toe"],
        action_scale=[0.25, 0.25, 0.25, 0.5],
        action_offset=[0.1, -0.4, 0.3, 0.0],
        action_clip=None,
        joint_stiffness=[100.0, 100.0, 40.0, 20.0],
        joint_damping=[5.0, 5.0, 2.0, 1.0],
        sim_dt=0.005,
        decimation=4,
        observation_names=["base_ang_vel", "joint_pos"],
        trained_commit="32aef5c5ec11",
    )
    return deploy_metadata(**{**values, **overrides})


def test_deploy_metadata_round_trip_leaves_the_graph_unchanged(tmp_path, trained):
    import json

    import onnx

    policy, _ = trained
    path = export(policy, tmp_path)
    graph_before = onnx.load(path).graph.SerializeToString()
    attach_deploy_metadata(path, sample_metadata(action_clip=[[None, None], [-1.0, 1.0], [None, 2.0], [None, None]]))
    model = onnx.load(path)
    assert model.graph.SerializeToString() == graph_before
    read = {entry.key: entry.value for entry in model.metadata_props}
    # The policy reads the 6-wide "policy" group and the 2-wide "extra" group; the graph outputs 4 actions.
    assert read["obs_dim"] == "8" and read["action_dim"] == "4"
    assert read["deploy_metadata_version"] == "1"
    assert json.loads(read["joint_names"]) == ["hip", "knee", "ankle", "toe"]
    assert json.loads(read["action_clip"])[1] == [-1.0, 1.0]
    assert float(read["policy_rate_hz"]) == 50.0 and read["decimation"] == "4"
    assert read["trained_commit"] == "32aef5c5ec11"


def test_deploy_metadata_attach_replaces_earlier_values(tmp_path, trained):
    policy, _ = trained
    path = export(policy, tmp_path)
    attach_deploy_metadata(path, sample_metadata())
    attach_deploy_metadata(path, sample_metadata(action_scale=[1.0, 1.0, 1.0, 1.0]))
    import onnx

    entries = [entry for entry in onnx.load(path).metadata_props if entry.key == "action_scale"]
    assert len(entries) == 1 and entries[0].value == "[1.0, 1.0, 1.0, 1.0]"


def test_deploy_metadata_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="one entry per action"):
        sample_metadata(action_scale=[0.25])
    with pytest.raises(ValueError, match="pair per action"):
        sample_metadata(action_clip=[[None, None]])


def test_existing_export_note_is_silent_without_an_export(tmp_path):
    run_dir = make_run(tmp_path)
    assert existing_export_note(run_dir, os.path.join(run_dir, "exported")) is None


def make_run_with_export(tmp_path, exported_at: int, checkpoints: dict[str, int]):
    run_dir = make_run(tmp_path)
    for name, written_at in checkpoints.items():
        path = os.path.join(run_dir, name)
        open(path, "w").close()
        os.utime(path, (written_at, written_at))
    exported = os.path.join(run_dir, "exported")
    os.makedirs(exported)
    onnx_path = os.path.join(exported, "policy.onnx")
    open(onnx_path, "w").close()
    os.utime(onnx_path, (exported_at, exported_at))
    return run_dir, exported


def test_existing_export_note_names_a_checkpoint_newer_than_the_export(tmp_path):
    run_dir, exported = make_run_with_export(
        tmp_path, exported_at=100, checkpoints={"model_50.pt": 90, "model_2500.pt": 300, "model_1000.pt": 200}
    )
    note = existing_export_note(run_dir, exported)
    assert "model_2500.pt" in note and "latest" in note


def test_existing_export_note_on_an_export_of_the_latest_checkpoint(tmp_path):
    run_dir, exported = make_run_with_export(tmp_path, exported_at=100, checkpoints={"model_50.pt": 90})
    assert existing_export_note(run_dir, exported) == "Overwriting the run's existing export."
