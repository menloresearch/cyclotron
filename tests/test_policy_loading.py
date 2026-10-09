from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from cyclotron.policy_loading import inference_load_cfg, load_policy, policy_shape_errors


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


def test_inference_restores_only_the_deployed_network():
    assert inference_load_cfg("OnPolicyRunner") == {"actor": True}
    assert inference_load_cfg("DistillationRunner") == {"student": True}


def test_load_policy_loads_only_the_actor_or_ends_with_the_hint(tmp_path):
    torch.manual_seed(0)
    trained = torch.nn.Sequential(torch.nn.Linear(78, 8), torch.nn.ELU(), torch.nn.Linear(8, 23))
    checkpoint = tmp_path / "model_1.pt"
    torch.save(
        {"actor_state_dict": trained.state_dict(), "critic_state_dict": {"0.weight": torch.zeros(1, 96)}}, checkpoint
    )

    loads = []
    policy = {"network": torch.nn.Sequential(torch.nn.Linear(78, 8), torch.nn.ELU(), torch.nn.Linear(8, 23))}
    runner = SimpleNamespace(
        alg=SimpleNamespace(get_policy=lambda: policy["network"]),
        load=lambda path, load_cfg=None: loads.append(load_cfg),
    )
    hint = "Check out the code the run was trained with (exp/drift @ 32aef5c), or train a new run."
    load_policy(runner, str(checkpoint), "OnPolicyRunner", hint)
    assert loads == [{"actor": True}]

    policy["network"] = torch.nn.Sequential(torch.nn.Linear(55, 8), torch.nn.ELU(), torch.nn.Linear(8, 23))
    with pytest.raises(ValueError) as error:
        load_policy(runner, str(checkpoint), "OnPolicyRunner", hint)
    assert str(error.value).splitlines() == [
        "The checkpoint's policy doesn't fit the network the current code builds:",
        "  0.weight: checkpoint [8, 78], current code [8, 55] (the policy takes 78 inputs; the current observations"
        " give 55)",
        "The policy's inputs or network changed since training; see the warning above.",
        hint,
    ]
    assert loads == [{"actor": True}]

    # A checkpoint without this runner's layout is left to rsl_rl.
    torch.save({"model_state_dict": trained.state_dict()}, checkpoint)
    load_policy(runner, str(checkpoint), "OnPolicyRunner", hint)
    assert loads == [{"actor": True}, None]
