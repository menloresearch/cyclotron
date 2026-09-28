# Copyright (c) 2026, Menlo Research.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""CPU tests for algorithms/getup_ppo.py (rsl-rl-lib 5.0.1, no Isaac Lab needed)."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from isaac_asimov.algorithms.getup_ppo import (
    DiscountedReturnNormalizer,
    GetUpPPO,
    interpolate_obs,
    l2c2_loss,
)
from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage

NUM_ENVS, T, A = 8, 6, 3
OBS_GROUPS = {"actor": ["policy"], "critic": ["policy", "critic"]}


def _obs(n=NUM_ENVS):
    return TensorDict({"policy": torch.randn(n, 4), "critic": torch.randn(n, 2)}, batch_size=[n])


def _toy_mirror(env=None, obs=None, actions=None):
    """Toy symmetry for the 4-dim policy / 2-dim critic obs: negate everything."""
    obs_aug = act_aug = None
    if obs is not None:
        mirrored = obs.clone()
        for k in obs.keys():
            mirrored[k] = -obs[k]
        obs_aug = torch.cat([obs, mirrored], dim=0)
    if actions is not None:
        act_aug = torch.cat([actions, -actions], dim=0)
    return obs_aug, act_aug


def _make_alg(**kwargs) -> GetUpPPO:
    torch.manual_seed(0)
    obs = _obs()
    actor = MLPModel(
        obs, OBS_GROUPS, "actor", A, hidden_dims=[16], obs_normalization=True,
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0},
    )  # fmt: skip
    critic = MLPModel(obs, OBS_GROUPS, "critic", 1, hidden_dims=[16], obs_normalization=True)
    storage = RolloutStorage("rl", NUM_ENVS, T, obs, [A], "cpu")
    defaults = dict(num_learning_epochs=2, num_mini_batches=2, gamma=0.995, device="cpu")
    defaults.update(kwargs)
    return GetUpPPO(actor, critic, storage, **defaults)


def _rollout(alg: GetUpPPO, seed=1, reward_scale=1.0, with_timeouts=False):
    g = torch.Generator().manual_seed(seed)
    obs = _obs()
    raw = []
    with torch.inference_mode():  # exactly like OnPolicyRunner.learn
        for t in range(T):
            alg.act(obs)
            obs = TensorDict(
                {"policy": torch.randn(NUM_ENVS, 4, generator=g), "critic": torch.randn(NUM_ENVS, 2, generator=g)},
                batch_size=[NUM_ENVS],
            )
            r = reward_scale * (torch.randn(NUM_ENVS, generator=g) + 0.5)
            dones = (torch.rand(NUM_ENVS, generator=g) < 0.2).float()
            extras = {"time_outs": dones * (torch.rand(NUM_ENVS, generator=g) < 0.5)} if with_timeouts else {}
            raw.append(r)
            alg.process_env_step(obs, r, dones, extras)
        alg.compute_returns(obs)
    return raw


# ---------------------------------------------------------------------------------------------------------------------
# Reward normalization numerics
# ---------------------------------------------------------------------------------------------------------------------


def _reference_normalize(rewards: np.ndarray, dones: np.ndarray, gamma, decay, eps):
    """Float64 numpy reference of DiscountedReturnNormalizer."""
    n_steps, n_envs = rewards.shape
    G = np.zeros(n_envs)
    mean, var, count = 0.0, 1.0, 0
    out = np.zeros_like(rewards)
    for t in range(n_steps):
        G = gamma * G + rewards[t]
        bm, bv = G.mean(), G.var()
        w = max(1.0 / (count + 1), 1.0 - decay)
        delta = bm - mean
        mean = mean + w * delta
        var = (1 - w) * var + w * bv + w * (1 - w) * delta**2
        count += 1
        out[t] = rewards[t] / (np.sqrt(var) + eps)
        G = G * (1 - dones[t])
    return out, np.sqrt(var)


def test_reward_normalizer_matches_reference():
    rng = np.random.default_rng(3)
    steps, n = 300, 64
    rewards = rng.normal(0.2, 1.5, size=(steps, n))
    dones = (rng.random((steps, n)) < 0.02).astype(np.float64)
    gamma, decay, eps = 0.99, 0.98, 1e-2
    expected, expected_std = _reference_normalize(rewards, dones, gamma, decay, eps)
    norm = DiscountedReturnNormalizer(n, gamma, decay=decay, eps=eps).double()
    got = np.stack(
        [norm(torch.from_numpy(rewards[t]), torch.from_numpy(dones[t])).numpy() for t in range(steps)]
    )
    np.testing.assert_allclose(got, expected, rtol=1e-10, atol=1e-12)
    assert float(norm.std) == pytest.approx(expected_std, rel=1e-10)


def test_reward_normalizer_scale_invariance_and_unit_return_std():
    """Normalized rewards are invariant to the raw scale and give discounted returns of ~unit std."""
    torch.manual_seed(0)
    gamma, n, steps = 0.95, 4096, 400
    r = torch.randn(steps, n) + 0.3
    outs = []
    for scale in (1.0, 250.0):
        norm = DiscountedReturnNormalizer(n, gamma, decay=0.99, eps=1e-8)
        outs.append(torch.stack([norm(scale * r[t]) for t in range(steps)]))
    torch.testing.assert_close(outs[0], outs[1], rtol=1e-4, atol=1e-5)
    # returns of the normalized rewards at stationarity
    G = torch.zeros(n)
    for t in range(steps):
        G = gamma * G + outs[0][t]
    assert G.std().item() == pytest.approx(1.0, abs=0.05)


def test_reward_normalizer_resets_on_done_and_eval_freezes():
    norm = DiscountedReturnNormalizer(2, 0.9)
    norm(torch.tensor([1.0, 1.0]), torch.tensor([1.0, 0.0]))
    assert norm._returns.tolist() == [0.0, 1.0]
    norm(torch.tensor([1.0, 1.0]), torch.tensor([0.0, 0.0]))
    assert norm._returns.tolist() == pytest.approx([1.0, 1.9])
    state = copy.deepcopy(norm.state_dict())
    norm.eval()
    norm(torch.tensor([100.0, -100.0]))
    for k, v in norm.state_dict().items():
        assert torch.equal(v, state[k]), k
    assert "_returns" not in state  # per-env accumulator is not checkpointed


def test_process_env_step_stores_normalized_rewards_in_inference_mode():
    alg = _make_alg(reward_norm=True, reward_norm_decay=0.9)
    raw = _rollout(alg)
    ref = DiscountedReturnNormalizer(NUM_ENVS, alg.gamma, decay=0.9)
    # replay the dones recorded in storage
    for t in range(T):
        expected = ref(raw[t], alg.storage.dones[t, :, 0].float())
        torch.testing.assert_close(alg.storage.rewards[t, :, 0], expected)
    stats = alg.reward_norm_stats()
    assert stats["reward_norm_return_std"] == pytest.approx(float(ref.std))
    assert stats["reward_raw_mean"] == pytest.approx(float(torch.stack(raw).mean()), rel=1e-5)


def test_reward_norm_off_is_plain_ppo():
    alg = _make_alg(reward_norm=False)
    raw = _rollout(alg)
    for t in range(T):
        torch.testing.assert_close(alg.storage.rewards[t, :, 0], raw[t])
    assert alg.reward_norm_stats() == {}


def test_timeout_bootstrap_uses_normalized_scale():
    alg = _make_alg(reward_norm=True)
    obs = _obs()
    with torch.inference_mode():
        alg.act(obs)
        values = alg.transition.values.clone()
        r = torch.ones(NUM_ENVS)
        dones = torch.ones(NUM_ENVS)
        time_outs = torch.zeros(NUM_ENVS)
        time_outs[0] = 1.0
        alg.process_env_step(_obs(), r, dones, {"time_outs": time_outs})
    normalized = alg.storage.rewards[0, :, 0]
    base = normalized[1]
    assert normalized[0].item() == pytest.approx(base.item() + alg.gamma * values[0, 0].item(), rel=1e-5)


def test_checkpoint_roundtrip():
    alg = _make_alg(reward_norm=True)
    _rollout(alg)
    alg.update()
    saved = alg.save()
    assert "reward_normalizer_state_dict" in saved
    alg2 = _make_alg(reward_norm=True)
    alg2.load(saved, None, strict=True)
    for k, v in alg.reward_normalizer.state_dict().items():
        assert torch.equal(v, alg2.reward_normalizer.state_dict()[k])
    # checkpoints of plain PPO (e.g. without the key) still load
    del saved["reward_normalizer_state_dict"]
    _make_alg(reward_norm=True).load(saved, None, strict=True)


# ---------------------------------------------------------------------------------------------------------------------
# L2C2
# ---------------------------------------------------------------------------------------------------------------------


def test_l2c2_zero_for_identical_inputs():
    alg = _make_alg()
    obs = _obs(32)
    for u in (torch.zeros(32, 1), torch.rand(32, 1)):
        interp = interpolate_obs(obs, obs.clone(), u)
        assert l2c2_loss(alg.actor, obs, interp).item() == 0.0
        assert l2c2_loss(alg.critic, obs, interp).item() == 0.0
    # u = 0 is the identity even when the next state differs
    assert l2c2_loss(alg.actor, obs, interpolate_obs(obs, _obs(32), torch.zeros(32, 1))).item() == 0.0


def test_l2c2_positive_and_differentiable():
    alg = _make_alg()
    obs, obs_next = _obs(32), _obs(32)
    interp = interpolate_obs(obs, obs_next, torch.full((32, 1), 0.5))
    torch.testing.assert_close(interp["policy"], 0.5 * (obs["policy"] + obs_next["policy"]))
    loss = l2c2_loss(alg.actor, obs, interp)
    assert loss.item() > 0.0
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in alg.actor.parameters())


def test_l2c2_pairs_are_consecutive_and_skip_resets():
    alg = _make_alg(l2c2_enabled=True)
    st = alg.storage
    for k in st.observations.keys():
        t_code = torch.arange(T).view(T, 1, 1) * 100.0
        n_code = torch.arange(NUM_ENVS).view(1, NUM_ENVS, 1).float()
        st.observations[k] = (t_code + n_code).expand_as(st.observations[k]).clone()
    st.dones.zero_()
    st.dones[2, :, 0] = 1  # obs[3] is a post-reset state for every env
    obs, obs_next = alg._sample_l2c2_pairs(2000)
    t = (obs["policy"][:, 0] // 100).long()
    assert torch.all(obs_next["policy"][:, 0] - obs["policy"][:, 0] == 100.0)
    assert not torch.any(t == 2) and torch.all(t <= T - 2)
    st.dones.fill_(1)
    assert alg._sample_l2c2_pairs(100) is None
    la, lc = alg._l2c2_losses(100)
    assert la.item() == 0.0 and lc.item() == 0.0


# ---------------------------------------------------------------------------------------------------------------------
# Full iteration (as the runner drives it) with symmetry + L2C2 + reward norm
# ---------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("l2c2", [False, True])
def test_full_update_with_symmetry(l2c2):
    symmetry_cfg = {
        "use_data_augmentation": True,
        "use_mirror_loss": False,
        "data_augmentation_func": _toy_mirror,
        "mirror_loss_coeff": 0.0,
        "_env": SimpleNamespace(),
    }
    alg = _make_alg(reward_norm=True, l2c2_enabled=l2c2, symmetry_cfg=symmetry_cfg)
    alg.train_mode()
    before = [p.detach().clone() for p in alg.actor.parameters()]
    _rollout(alg, with_timeouts=True)
    loss = alg.update()
    for key in ("value", "surrogate", "entropy", "symmetry", "reward_norm_return_std", "reward_raw_mean"):
        assert key in loss and np.isfinite(loss[key]), key
    assert ("l2c2_actor" in loss) == l2c2 and ("l2c2_critic" in loss) == l2c2
    if l2c2:
        assert loss["l2c2_actor"] > 0.0 and loss["l2c2_critic"] > 0.0
    assert any(not torch.equal(a, b) for a, b in zip(before, alg.actor.parameters()))
    assert alg.storage.step == 0  # cleared
    # the next rollout starts fresh stats
    assert alg._reward_steps == 0


def test_construct_algorithm_from_runner_cfg_dict():
    """What OnPolicyRunner does with GetUpPPOAlgorithmCfg().to_dict(): every field becomes a kwarg of GetUpPPO."""
    alg_cfg = {
        "class_name": "isaac_asimov.algorithms.getup_ppo:GetUpPPO",
        "num_learning_epochs": 5, "num_mini_batches": 4, "learning_rate": 1e-3, "schedule": "adaptive",
        "gamma": 0.995, "lam": 0.95, "entropy_coef": 0.005, "desired_kl": 0.01, "max_grad_norm": 1.0,
        "optimizer": "adam", "value_loss_coef": 1.0, "use_clipped_value_loss": True, "clip_param": 0.2,
        "normalize_advantage_per_mini_batch": False, "share_cnn_encoders": False, "rnd_cfg": None,
        "symmetry_cfg": {"use_data_augmentation": True, "use_mirror_loss": False,
                         "data_augmentation_func": _toy_mirror, "mirror_loss_coeff": 0.0},
        "reward_norm": True, "reward_norm_decay": 0.999, "reward_norm_eps": 1e-2, "reward_norm_clip": None,
        "l2c2_enabled": False, "l2c2_lambda_actor": 1.0, "l2c2_lambda_critic": 0.1, "l2c2_interp_max": 1.0,
        "l2c2_num_samples": None,
    }  # fmt: skip
    try:  # on the server, take the real dict from the Isaac Lab configclass
        from isaac_asimov.algorithms.getup_ppo import GetUpPPOAlgorithmCfg

        if GetUpPPOAlgorithmCfg is not None:
            from isaaclab_rl.rsl_rl import RslRlSymmetryCfg

            real = GetUpPPOAlgorithmCfg(
                symmetry_cfg=RslRlSymmetryCfg(use_data_augmentation=True, data_augmentation_func=_toy_mirror)
            ).to_dict()
            assert set(alg_cfg) <= set(real) | {"symmetry_cfg"}, set(alg_cfg) ^ set(real)
            real["symmetry_cfg"]["data_augmentation_func"] = _toy_mirror  # to_dict stored a "module:func" string
            alg_cfg = real
    except ImportError:
        pass
    cfg = {
        "algorithm": copy.deepcopy(alg_cfg),
        "actor": {"class_name": "MLPModel", "hidden_dims": [16], "activation": "elu", "obs_normalization": True,
                  "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": 1.0}},
        "critic": {"class_name": "MLPModel", "hidden_dims": [16], "activation": "elu", "obs_normalization": True},
        "obs_groups": OBS_GROUPS,
        "num_steps_per_env": T,
        "multi_gpu": None,
    }  # fmt: skip
    env = SimpleNamespace(num_envs=NUM_ENVS, num_actions=A)
    alg = PPO.construct_algorithm(_obs(), env, cfg, "cpu")
    assert isinstance(alg, GetUpPPO)
    assert alg.gamma == 0.995 and alg.reward_normalizer is not None and not alg.l2c2_enabled
    assert alg.symmetry["_env"] is env


# ---------------------------------------------------------------------------------------------------------------------
# action-std clamp
# ---------------------------------------------------------------------------------------------------------------------


def _make_alg_std(std_type="scalar", init_std=1.0, **kwargs) -> GetUpPPO:
    torch.manual_seed(0)
    obs = _obs()
    actor = MLPModel(
        obs, OBS_GROUPS, "actor", A, hidden_dims=[16], obs_normalization=True,
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": init_std, "std_type": std_type},
    )  # fmt: skip
    critic = MLPModel(obs, OBS_GROUPS, "critic", 1, hidden_dims=[16], obs_normalization=True)
    storage = RolloutStorage("rl", NUM_ENVS, T, obs, [A], "cpu")
    defaults = dict(num_learning_epochs=2, num_mini_batches=2, gamma=0.995, device="cpu")
    defaults.update(kwargs)
    return GetUpPPO(actor, critic, storage, **defaults)


@pytest.mark.parametrize("std_type", ["scalar", "log"])
def test_std_clamped_at_init_and_after_every_step(std_type):
    alg = _make_alg_std(std_type, init_std=2.5, max_action_std=1.0, min_action_std=0.05, entropy_coef=50.0)
    assert torch.allclose(alg.action_std(), torch.ones(A))  # clamped at construction
    # a huge entropy bonus pushes the std up every step; it must stay <= max
    _rollout(alg)
    loss = alg.update()
    std = alg.action_std()
    assert torch.all(std <= 1.0 + 1e-6) and torch.all(std >= 0.05 - 1e-6)
    assert torch.allclose(std, torch.ones(A), atol=1e-5)  # pinned at the upper bound
    assert np.isfinite(loss["value"])
    # and down: entropy penalty drives it to the floor
    alg = _make_alg_std(
        std_type, init_std=0.06, max_action_std=1.0, min_action_std=0.05, entropy_coef=-50.0,
        schedule="fixed", learning_rate=0.05,
    )  # fmt: skip
    _rollout(alg)
    alg.update()
    assert torch.allclose(alg.action_std(), torch.full((A,), 0.05), atol=1e-5)


def test_std_clamp_off_and_heteroscedastic_rejected():
    alg = _make_alg_std("scalar", init_std=2.5, max_action_std=None, min_action_std=None)
    assert torch.allclose(alg.action_std(), torch.full((A,), 2.5))
    obs = _obs()
    actor = MLPModel(
        obs, OBS_GROUPS, "actor", A, hidden_dims=[16],
        distribution_cfg={"class_name": "HeteroscedasticGaussianDistribution", "init_std": 1.0},
    )  # fmt: skip
    critic = MLPModel(obs, OBS_GROUPS, "critic", 1, hidden_dims=[16])
    with pytest.raises(ValueError, match="state-independent"):
        GetUpPPO(actor, critic, RolloutStorage("rl", NUM_ENVS, T, obs, [A], "cpu"), device="cpu")


def test_std_logged_through_extras():
    alg = _make_alg_std("scalar", init_std=0.7)
    extras = {"log": {"Episode_Reward/x": torch.tensor(1.0)}}
    with torch.inference_mode():
        alg.act(_obs())
        alg.process_env_step(_obs(), torch.zeros(NUM_ENVS), torch.zeros(NUM_ENVS), extras)
    assert extras["log"]["Policy/std_mean"].item() == pytest.approx(0.7)
    assert extras["log"]["Policy/std_max"].item() == pytest.approx(0.7)
    assert "Episode_Reward/x" in extras["log"]


# ---------------------------------------------------------------------------------------------------------------------
# deterministic evaluation envs
# ---------------------------------------------------------------------------------------------------------------------


def test_deterministic_mask_layout():
    for n_envs, frac, expected in [(8, 0.05, 1), (8, 0.0, 0), (100, 0.05, 5), (101, 0.05, 6), (8, 0.5, 4)]:
        obs = _obs(n_envs)
        actor = MLPModel(obs, OBS_GROUPS, "actor", A, hidden_dims=[8],
                         distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0})  # fmt: skip
        critic = MLPModel(obs, OBS_GROUPS, "critic", 1, hidden_dims=[8])
        alg = GetUpPPO(actor, critic, RolloutStorage("rl", n_envs, T, obs, [A], "cpu"), device="cpu",
                       deterministic_env_fraction=frac)  # fmt: skip
        assert alg.num_deterministic_envs == expected
        assert alg.deterministic_mask.sum().item() == expected
        assert not alg.deterministic_mask[: n_envs - expected].any()  # the LAST ceil(f*N) envs


def test_deterministic_envs_act_with_mean():
    alg = _make_alg_std("scalar", init_std=1.0, deterministic_env_fraction=0.25)  # envs 6, 7
    det = alg.deterministic_mask
    obs = _obs()
    with torch.inference_mode():
        actions = alg.act(obs)
        mean = alg.actor(obs)  # deterministic forward = mean
    torch.testing.assert_close(actions[det], mean[det])
    assert not torch.allclose(actions[~det], mean[~det])  # std 1.0: sampled
    torch.testing.assert_close(alg.transition.actions[det], mean[det])


def test_deterministic_envs_excluded_from_update(monkeypatch):
    alg = _make_alg_std("scalar", deterministic_env_fraction=0.25)
    _rollout(alg)
    st = alg.storage
    stoch = ~alg.deterministic_mask
    # advantages normalized over stochastic envs only; deterministic envs zeroed
    adv = st.advantages[:, stoch]
    assert adv.mean().item() == pytest.approx(0.0, abs=1e-5) and adv.std().item() == pytest.approx(1.0, rel=1e-4)
    assert torch.all(st.advantages[:, alg.deterministic_mask] == 0.0)
    # tag every stored transition with its env index and check which ones reach the loss
    for k in st.observations.keys():
        st.observations[k][..., 0] = torch.arange(NUM_ENVS, dtype=torch.float).view(1, NUM_ENVS)
    seen = []
    gen = alg._mini_batch_generator()
    for batch in gen:
        seen.append(batch.observations["policy"][:, 0])
    seen = torch.cat(seen).long()
    n_stoch = int(stoch.sum())
    epochs = alg.num_learning_epochs
    assert seen.numel() == epochs * (n_stoch * T // alg.num_mini_batches) * alg.num_mini_batches
    assert not torch.any(alg.deterministic_mask[seen])
    # each epoch covers every stochastic transition exactly once
    counts = torch.bincount(seen, minlength=NUM_ENVS)
    assert torch.all(counts[stoch] == epochs * T)


def test_deterministic_envs_excluded_with_symmetry():
    seen_sizes = []

    def recording_mirror(env=None, obs=None, actions=None):
        if obs is not None:
            seen_sizes.append(obs.batch_size[0])
            assert torch.all(obs["policy"][:, 0] < NUM_ENVS - 2)  # envs 6, 7 are deterministic
        return _toy_mirror(env=env, obs=obs, actions=actions)

    symmetry_cfg = {"use_data_augmentation": True, "use_mirror_loss": False,
                    "data_augmentation_func": recording_mirror, "mirror_loss_coeff": 0.0, "_env": SimpleNamespace()}  # fmt: skip
    alg = _make_alg_std("scalar", deterministic_env_fraction=0.25, symmetry_cfg=symmetry_cfg, l2c2_enabled=True)
    _rollout(alg)
    for k in alg.storage.observations.keys():
        alg.storage.observations[k][..., 0] = torch.arange(NUM_ENVS, dtype=torch.float).view(1, NUM_ENVS)
    loss = alg.update()
    assert seen_sizes and all(s == 6 * T // 2 for s in seen_sizes)
    assert np.isfinite(loss["symmetry"]) and np.isfinite(loss["l2c2_actor"])


def test_deterministic_fraction_zero_is_plain_rsl_rl():
    alg = _make_alg_std("scalar", deterministic_env_fraction=0.0)
    assert alg.num_deterministic_envs == 0
    _rollout(alg)
    adv = alg.storage.advantages
    assert adv.mean().item() == pytest.approx(0.0, abs=1e-5) and adv.std().item() == pytest.approx(1.0, rel=1e-4)
    gen = alg._mini_batch_generator()
    total = sum(b.observations.batch_size[0] for b in gen)
    assert total == alg.num_learning_epochs * NUM_ENVS * T


def test_construct_algorithm_publishes_mask_to_env():
    cfg = {
        "algorithm": {"class_name": "isaac_asimov.algorithms.getup_ppo:GetUpPPO", "gamma": 0.995,
                      "deterministic_env_fraction": 0.25, "rnd_cfg": None, "symmetry_cfg": None},
        "actor": {"class_name": "MLPModel", "hidden_dims": [16], "activation": "elu",
                  "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": 1.0}},
        "critic": {"class_name": "MLPModel", "hidden_dims": [16], "activation": "elu"},
        "obs_groups": OBS_GROUPS, "num_steps_per_env": T, "multi_gpu": None,
    }  # fmt: skip
    # the runner resolves class_name and calls <class>.construct_algorithm
    state = SimpleNamespace(category=torch.zeros(NUM_ENVS))
    env = SimpleNamespace(num_envs=NUM_ENVS, num_actions=A, unwrapped=SimpleNamespace(getup_state=state, device="cpu"))
    alg = GetUpPPO.construct_algorithm(_obs(), env, copy.deepcopy(cfg), "cpu")
    assert isinstance(alg, GetUpPPO)
    assert state.deterministic.dtype == torch.bool and state.deterministic.shape == (NUM_ENVS,)
    assert state.deterministic.tolist() == [False] * 6 + [True] * 2
    assert state.deterministic is not alg.deterministic_mask  # env-side copy


# ---------------------------------------------------------------------------------------------------------------------
# load re-clamps std; learning rate persisted / reset
# ---------------------------------------------------------------------------------------------------------------------


def test_load_reclamps_std_and_refreshes_log():
    src = _make_alg_std("scalar", init_std=1.0, max_action_std=1.0)
    saved = src.save()
    dst = _make_alg_std("scalar", init_std=0.3, max_action_std=0.6)  # Stage-B-like tighter bound
    dst.load(saved, None, strict=True)
    assert torch.allclose(dst.action_std(), torch.full((A,), 0.6))
    assert dst._std_log["Policy/std_max"].item() == pytest.approx(0.6)
    # also when only the actor is loaded (warm start)
    dst = _make_alg_std("log", init_std=0.3, max_action_std=0.6)
    src = _make_alg_std("log", init_std=1.0, max_action_std=1.0)
    dst.load(src.save(), {"actor": True}, strict=True)
    assert torch.allclose(dst.action_std(), torch.full((A,), 0.6), atol=1e-6)


def _alg_with_adapted_lr(lr):
    alg = _make_alg_std("scalar")
    alg.learning_rate = lr
    for g in alg.optimizer.param_groups:
        g["lr"] = lr
    return alg


def test_learning_rate_persisted_and_resumed():
    saved = _alg_with_adapted_lr(3.4e-4).save()
    assert saved["learning_rate"] == pytest.approx(3.4e-4)
    dst = _make_alg_std("scalar", learning_rate=1e-3)
    dst.load(saved, None, strict=True)
    assert dst.learning_rate == pytest.approx(3.4e-4)
    assert all(g["lr"] == pytest.approx(3.4e-4) for g in dst.optimizer.param_groups)


def test_learning_rate_from_old_checkpoint_optimizer():
    saved = _alg_with_adapted_lr(2.2e-4).save()
    del saved["learning_rate"]  # older checkpoints saved without the learning rate
    dst = _make_alg_std("scalar", learning_rate=1e-3)
    dst.load(saved, None, strict=True)
    assert dst.learning_rate == pytest.approx(2.2e-4)
    # optimizer not loaded and no key -> configured lr
    dst = _make_alg_std("scalar", learning_rate=1e-3)
    dst.load(saved, {"actor": True, "critic": True}, strict=True)
    assert dst.learning_rate == pytest.approx(1e-3)


def test_learning_rate_reset_on_load_for_warm_start():
    saved = _alg_with_adapted_lr(3.4e-4).save()
    dst = _make_alg_std("scalar", learning_rate=5e-4, reset_learning_rate_on_load=True)
    dst.load(saved, None, strict=True)  # optimizer state loaded, but lr reset
    assert dst.learning_rate == pytest.approx(5e-4)
    assert all(g["lr"] == pytest.approx(5e-4) for g in dst.optimizer.param_groups)
    # fixed schedule always uses the configured lr
    dst = _make_alg_std("scalar", learning_rate=5e-4, schedule="fixed")
    dst.load(saved, None, strict=True)
    assert dst.learning_rate == pytest.approx(5e-4)
    assert all(g["lr"] == pytest.approx(5e-4) for g in dst.optimizer.param_groups)
