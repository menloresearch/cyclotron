from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
import torch
from rsl_rl.models import MLPModel
from rsl_rl.runners import OnPolicyRunner
from tensordict import TensorDict

from cyclotron.hub import infer_task
from cyclotron.onnx_export import copy_run_yamls, max_onnx_difference


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
