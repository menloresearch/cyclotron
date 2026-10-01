from __future__ import annotations

import numpy as np
import torch
from tensordict import TensorDict

from cyclotron.algorithms.amp_ppo import AMPPPO
from cyclotron.algorithms.discriminator import AMPDiscriminator, AMPFeatureNormalizer
from cyclotron.algorithms.replay_buffer import AMPReplayBuffer
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage


def test_feature_normalizer_matches_baseline_numpy_equations():
    rng = np.random.default_rng(7)
    batches = [
        rng.normal(size=(17, 4)).astype(np.float32),
        rng.normal(1.0, 2.0, size=(9, 4)).astype(np.float32),
    ]
    expected_mean = np.zeros(4, np.float64)
    expected_var = np.ones(4, np.float64)
    expected_count = 1.0e-4
    normalizer = AMPFeatureNormalizer(4)

    for batch in batches:
        expected_normalized = np.clip(
            (batch - expected_mean) / np.sqrt(expected_var + 1.0e-4), -10.0, 10.0
        )
        actual_normalized = normalizer(torch.from_numpy(batch)).numpy()
        np.testing.assert_allclose(actual_normalized, expected_normalized, rtol=1.0e-6, atol=1.0e-6)

        batch_mean = np.mean(expected_normalized, axis=0)
        batch_var = np.var(expected_normalized, axis=0)
        batch_count = len(batch)
        delta = batch_mean - expected_mean
        total_count = expected_count + batch_count
        new_mean = expected_mean + delta * batch_count / total_count
        moment_2 = (
            expected_var * expected_count
            + batch_var * batch_count
            + np.square(delta) * expected_count * batch_count / total_count
        )
        expected_mean = new_mean
        expected_var = moment_2 / total_count
        expected_count = total_count

        normalizer.update(torch.from_numpy(expected_normalized))
        np.testing.assert_allclose(normalizer._mean.numpy().squeeze(0), expected_mean, rtol=1.0e-12)
        np.testing.assert_allclose(normalizer._var.numpy().squeeze(0), expected_var, rtol=1.0e-12)
        assert normalizer.count.item() == expected_count


def test_discriminator_reward_and_raw_gradient_penalty_match_baseline_formulas():
    torch.manual_seed(3)
    discriminator = AMPDiscriminator(observation_dim=3, hidden_dims=[8, 4])
    state = torch.randn(6, 3)
    next_state = torch.randn(6, 3)

    reward, prediction = discriminator.predict_amp_reward(state, next_state)
    normalized_transition = torch.cat((discriminator.normalize(state), discriminator.normalize(next_state)), dim=-1)
    expected_prediction = discriminator(normalized_transition).squeeze(-1)
    expected_reward = torch.clamp(1.0 - 0.25 * torch.square(expected_prediction - 1.0), min=0.0)
    torch.testing.assert_close(prediction, expected_prediction)
    torch.testing.assert_close(reward, expected_reward)

    raw_transition = torch.cat((state, next_state), dim=-1).detach().requires_grad_(True)
    raw_prediction = discriminator(raw_transition)
    raw_gradient = torch.autograd.grad(
        raw_prediction,
        raw_transition,
        grad_outputs=torch.ones_like(raw_prediction),
        create_graph=True,
    )[0]
    expected_penalty = 10.0 * raw_gradient.norm(2, dim=1).pow(2).mean()
    actual_penalty = discriminator.compute_grad_pen(state, next_state, lambda_=10.0)
    torch.testing.assert_close(actual_penalty, expected_penalty)


def test_replay_buffer_keeps_state_pairs_aligned_across_wraparound():
    replay = AMPReplayBuffer(observation_dim=2, capacity=5, device="cpu")
    first = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    second = torch.arange(8, dtype=torch.float32).reshape(4, 2) + 20.0
    replay.insert(first, first + 100.0)
    replay.insert(second, second + 100.0)

    assert replay.num_samples == 5
    torch.testing.assert_close(replay.next_states, replay.states + 100.0)
    sampled_states, sampled_next_states = next(replay.generator(num_batches=1, batch_size=20))
    torch.testing.assert_close(sampled_next_states, sampled_states + 100.0)


def test_amp_ppo_pairs_frames_and_blends_rewards_without_custom_runner():
    class Commands:
        def get_command(self, _name):
            return torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

    num_envs = 2
    obs = TensorDict(
        {
            "policy": torch.randn(num_envs, 4),
            "critic": torch.randn(num_envs, 5),
            "amp": torch.randn(num_envs, 3),
        },
        batch_size=[num_envs],
    )
    obs_groups = {"actor": ["policy"], "critic": ["critic"]}
    actor = MLPModel(
        obs,
        obs_groups,
        "actor",
        2,
        hidden_dims=[8],
        obs_normalization=False,
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0},
    )
    critic = MLPModel(obs, obs_groups, "critic", 1, hidden_dims=[8], obs_normalization=False)
    storage = RolloutStorage("rl", num_envs, 1, obs, [2], "cpu")
    algorithm = AMPPPO(
        actor,
        critic,
        storage,
        amp_data=object(),
        amp_observation_dim=3,
        command_manager=Commands(),
        amp_reward_command_gate=True,
        amp_reward_command_threshold=0.1,
        device="cpu",
    )

    initial_amp_state = obs["amp"].clone()
    algorithm.act(obs)
    next_obs = TensorDict(
        {
            "policy": torch.full((num_envs, 4), 1000.0),
            "critic": torch.full((num_envs, 5), -1000.0),
            "amp": torch.randn(num_envs, 3),
        },
        batch_size=[num_envs],
    )
    raw_amp_reward, _ = algorithm.discriminator.predict_amp_reward(initial_amp_state, next_obs["amp"])
    task_reward = torch.tensor([2.0, 3.0])
    gate = torch.tensor([1.0, 0.0])
    expected_reward = 0.3 * (0.3 * raw_amp_reward * gate) + 0.7 * task_reward

    algorithm.process_env_step(next_obs, task_reward, torch.zeros(num_envs), {"log": {}})

    torch.testing.assert_close(algorithm.amp_storage.states[:num_envs], initial_amp_state)
    torch.testing.assert_close(algorithm.amp_storage.next_states[:num_envs], next_obs["amp"])
    torch.testing.assert_close(storage.rewards[0, :, 0], expected_reward)
    assert next_obs["policy"].max() == 500.0
    assert next_obs["critic"].min() == -500.0


def test_amp_ppo_clips_one_combined_actor_critic_gradient_norm():
    algorithm = AMPPPO.__new__(AMPPPO)
    algorithm.actor = torch.nn.Linear(3, 2, bias=False)
    algorithm.critic = torch.nn.Linear(3, 1, bias=False)
    algorithm.max_grad_norm = 1.0

    parameters = list(algorithm.actor.parameters()) + list(algorithm.critic.parameters())
    for parameter in parameters:
        parameter.grad = torch.ones_like(parameter)

    original_norm = algorithm._clip_actor_critic_gradients()
    clipped_norm = torch.linalg.vector_norm(
        torch.cat([parameter.grad.flatten() for parameter in parameters])
    )

    torch.testing.assert_close(original_norm, torch.tensor(3.0))
    assert clipped_norm <= algorithm.max_grad_norm + 1.0e-6
