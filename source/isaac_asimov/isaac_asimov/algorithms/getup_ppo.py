# Copyright (c) 2026, Menlo Research.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# ``GetUpPPO.update`` is adapted from rsl-rl-lib 5.0.1 ``rsl_rl/algorithms/ppo.py`` (PPO.update),
# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION, BSD-3-Clause.
# The L2C2 regularizer follows the formulation used in NVIDIA WBC-AGILE
# (third_party/rsl_rl/patches/rsl_rl_5_4_1_agile.patch, Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES,
# Apache-2.0) and HoST (rsl_rl/rsl_rl/algorithms/ppo.py, Copyright (c) 2025 OpenRobotLab, MIT); the code here is
# a re-implementation, not a copy.
"""PPO for the Asimov 1 get-up task.

Adds to rsl-rl 5.0.1 ``PPO``:

* **Running reward normalization** (on by default): rewards are divided by a running estimate of the standard
  deviation of the discounted return ``G_t = r_t + gamma * G_{t-1}`` (per-env accumulator, reset on ``done``;
  cross-env moments tracked with an EMA of decay ``reward_norm_decay``, count-averaged during warm-up). This is the
  classic return-based scaling (Engstrom et al. 2020, "Implementation Matters in Deep Policy Gradients"; OpenAI
  baselines ``VecNormalize``) that WBC-AGILE enables for ``StandUp-T1`` via ``RslRlRewardNormalizationCfg``
  (AGILE estimates the same quantity analytically as ``std(r)/sqrt(1-gamma^2)`` with a measured correction; we
  measure the discounted-return spread directly). Normalization happens before the time-out bootstrap, so the
  critic lives entirely in normalized units. Episode rewards logged by the runner stay raw.
* **L2C2** (Kobayashi et al. 2022, arXiv:2202.07152; off by default, Stage B): for random consecutive pairs
  ``(s_t, s_{t+1})`` of the rollout that do not cross a reset, ``s~ = s_t + u (s_{t+1} - s_t)``,
  ``u ~ U(0, l2c2_interp_max)``, and ``loss += l2c2_lambda_actor * MSE(mu(s~), mu(s_t))
  + l2c2_lambda_critic * MSE(V(s~), V(s_t))`` (mean over batch and output dims, as in AGILE).
* **Symmetry**: unchanged rsl-rl 5.0.1 ``symmetry_cfg`` handling (use ``tasks/getup/symmetry.py:mirror_getup``).
* **Action-std clamp**: after every optimizer step (and at construction) the policy's state-independent std
  parameter is clamped to ``[min_action_std, max_action_std]``: rsl-rl 5.0.1 ``GaussianDistribution`` keeps a
  per-action-dim ``std_param`` (``std_type="scalar"``) or ``log_std_param`` (``std_type="log"``, clamped in log
  space). ``HeteroscedasticGaussianDistribution`` (state-dependent std) cannot be clamped this way and is rejected.
* **Deterministic evaluation envs**: the last ``ceil(deterministic_env_fraction * N)`` envs act with the policy
  MEAN during rollouts, so success metrics measured on them reflect the deployable policy. Their transitions are
  **excluded from the PPO update**: the mini-batch generator only indexes stochastic envs (flat index
  ``t * N + n``, same layout as ``RolloutStorage.mini_batch_generator``), advantages are normalized over stochastic
  envs only, and L2C2 pairs are drawn from stochastic envs only. Symmetry augmentation therefore never sees them.
  Their observations still update the empirical obs normalizers and the reward-normalization statistics (same state
  distribution; harmless), and their episodes still appear in the runner's ``Train/mean_reward``. The mask is
  exposed to the env as ``env.unwrapped.getup_state.deterministic`` (bool ``[N]``, env device), created at
  ``construct_algorithm`` time.
* Logging: extra keys in the returned loss dict (TensorBoard ``Loss/<key>``): ``reward_norm_return_std``,
  ``reward_raw_mean``, ``reward_normalized_mean``, ``l2c2_actor``, ``l2c2_critic`` (+ ``symmetry`` from rsl-rl);
  ``Policy/std_mean`` and ``Policy/std_max`` (std of the rollout policy) through ``extras["log"]``.

Instantiated by the rsl-rl 5.0.1 ``OnPolicyRunner`` through ``PPO.construct_algorithm`` with
``cfg["algorithm"]["class_name"] = "isaac_asimov.algorithms.getup_ppo:GetUpPPO"``; every extra field of
``GetUpPPOAlgorithmCfg`` arrives as a keyword argument of ``GetUpPPO.__init__``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.algorithms import PPO
from rsl_rl.storage import RolloutStorage

__all__ = ["GetUpPPO", "GetUpPPOAlgorithmCfg", "DiscountedReturnNormalizer", "interpolate_obs", "l2c2_loss"]


class DiscountedReturnNormalizer(nn.Module):
    """Scale rewards by the running standard deviation of the per-env discounted return.

    Per step: ``G <- gamma * G + r`` (per env), update the cross-env mean/variance of ``G``, output
    ``r / (std(G) + eps)`` (optionally clipped to ``+-clip``), then zero ``G`` for envs that are done.
    Moments use a count average for the first ``1 / (1 - decay)`` updates and an EMA with ``decay`` afterwards,
    so the scale tracks curriculum changes.
    """

    def __init__(
        self,
        num_envs: int,
        gamma: float,
        decay: float = 0.999,
        eps: float = 1.0e-2,
        clip: float | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        if not 0.0 < decay < 1.0:
            raise ValueError(f"reward_norm_decay must be in (0, 1), got {decay}")
        self.gamma = float(gamma)
        self.decay = float(decay)
        self.eps = float(eps)
        self.clip = clip
        self.register_buffer("_mean", torch.zeros(1, device=device))
        self._mean: torch.Tensor
        self.register_buffer("_var", torch.ones(1, device=device))
        self._var: torch.Tensor
        self.register_buffer("_count", torch.zeros(1, device=device))
        self._count: torch.Tensor
        self.register_buffer("_returns", torch.zeros(num_envs, device=device), persistent=False)
        self._returns: torch.Tensor

    @property
    def std(self) -> torch.Tensor:
        """Current running std of the discounted return (raw reward units)."""
        return torch.sqrt(self._var.clamp(min=0.0))

    def forward(self, rewards: torch.Tensor, dones: torch.Tensor | None = None) -> torch.Tensor:
        rewards = rewards.reshape(-1)
        if self.training:
            self.update(rewards, dones)
        out = rewards / (self.std + self.eps)
        if self.clip is not None:
            out = out.clamp(-self.clip, self.clip)
        return out

    @torch.no_grad()
    def update(self, rewards: torch.Tensor, dones: torch.Tensor | None = None) -> None:
        rewards = rewards.reshape(-1).to(self._returns.dtype)
        self._returns.mul_(self.gamma).add_(rewards)
        batch_mean = self._returns.mean()
        batch_var = self._returns.var(unbiased=False)
        # count average during warm-up, EMA afterwards (w = weight of the new batch)
        w = torch.clamp(1.0 / (self._count + 1.0), min=1.0 - self.decay)
        delta = batch_mean - self._mean
        new_mean = self._mean + w * delta
        new_var = (1.0 - w) * self._var + w * batch_var + w * (1.0 - w) * delta.square()
        self._mean.copy_(new_mean)
        self._var.copy_(new_var)
        self._count.add_(1.0)
        if dones is not None:
            self._returns.mul_(1.0 - dones.reshape(-1).to(self._returns.dtype))


def interpolate_obs(obs: TensorDict, obs_next: TensorDict, u: torch.Tensor) -> TensorDict:
    """``obs + u * (obs_next - obs)`` for every floating-point group; ``u`` has shape ``[B, 1]``."""
    out = {}
    for key, value in obs.items():
        if not torch.is_floating_point(value):
            out[key] = value
            continue
        uu = u.to(value.dtype).reshape(u.shape[0], *([1] * (value.dim() - 1)))
        out[key] = value + uu * (obs_next[key] - value)
    return TensorDict(out, batch_size=obs.batch_size, device=obs.device)


def l2c2_loss(model: nn.Module, obs: TensorDict, obs_interp: TensorDict) -> torch.Tensor:
    """L2C2 term ``MSE(f(s~), f(s))`` for a deterministic model call (actor mean or critic value)."""
    return nn.functional.mse_loss(model(obs_interp), model(obs))


class GetUpPPO(PPO):
    """rsl-rl 5.0.1 PPO + running reward normalization + L2C2 (+ unchanged symmetry augmentation)."""

    def __init__(
        self,
        actor,
        critic,
        storage: RolloutStorage,
        *,
        reward_norm: bool = True,
        reward_norm_decay: float = 0.999,
        reward_norm_eps: float = 1.0e-2,
        reward_norm_clip: float | None = None,
        l2c2_enabled: bool = False,
        l2c2_lambda_actor: float = 1.0,
        l2c2_lambda_critic: float = 0.1,
        l2c2_interp_max: float = 1.0,
        l2c2_num_samples: int | None = None,
        max_action_std: float | None = 1.0,
        min_action_std: float | None = 0.05,
        deterministic_env_fraction: float = 0.05,
        reset_learning_rate_on_load: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(actor, critic, storage, **kwargs)
        self.reset_learning_rate_on_load = bool(reset_learning_rate_on_load)
        self.initial_learning_rate = float(self.learning_rate)
        # action-std clamp
        self.max_action_std = max_action_std
        self.min_action_std = min_action_std
        if max_action_std is not None and min_action_std is not None and min_action_std > max_action_std:
            raise ValueError(f"min_action_std {min_action_std} > max_action_std {max_action_std}")
        self._check_std_parameterization()
        self._clamp_action_std()
        self._std_log: dict[str, torch.Tensor] = {}
        self._refresh_std_log()
        # deterministic evaluation envs (last ceil(f * N) env indices)
        if not 0.0 <= deterministic_env_fraction < 1.0:
            raise ValueError(f"deterministic_env_fraction must be in [0, 1), got {deterministic_env_fraction}")
        n = storage.num_envs
        num_det = min(int(math.ceil(deterministic_env_fraction * n)), n - 1)
        self.deterministic_env_fraction = deterministic_env_fraction
        self.num_deterministic_envs = num_det
        self.deterministic_mask = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.deterministic_mask[n - num_det :] = True
        self._stochastic_env_ids = torch.nonzero(~self.deterministic_mask).squeeze(-1)
        if num_det > 0 and (actor.is_recurrent or critic.is_recurrent):
            raise ValueError("deterministic_env_fraction > 0 is implemented for feed-forward actor/critic only.")
        # reward normalization
        self.reward_normalizer: DiscountedReturnNormalizer | None = None
        if reward_norm:
            self.reward_normalizer = DiscountedReturnNormalizer(
                storage.num_envs,
                self.gamma,
                decay=reward_norm_decay,
                eps=reward_norm_eps,
                clip=reward_norm_clip,
                device=self.device,
            )
        self._reset_reward_stats()
        # L2C2
        self.l2c2_enabled = bool(l2c2_enabled)
        self.l2c2_lambda_actor = float(l2c2_lambda_actor)
        self.l2c2_lambda_critic = float(l2c2_lambda_critic)
        self.l2c2_interp_max = float(l2c2_interp_max)
        self.l2c2_num_samples = l2c2_num_samples
        if self.l2c2_enabled and (actor.is_recurrent or critic.is_recurrent):
            raise ValueError("L2C2 is implemented for feed-forward actor/critic only.")

    # ----------------------------------------------------------------------------------------------------------------
    # Construction / env binding
    # ----------------------------------------------------------------------------------------------------------------

    @staticmethod
    def construct_algorithm(obs: TensorDict, env, cfg: dict, device: str) -> GetUpPPO:
        """rsl-rl 5.0.1 construction, then publish the deterministic-env mask to the env."""
        alg = PPO.construct_algorithm(obs, env, cfg, device)
        if isinstance(alg, GetUpPPO):
            alg.bind_env(env)
        return alg

    def bind_env(self, env) -> None:
        """Write the deterministic-env mask to ``env.unwrapped.getup_state.deterministic`` (bool ``[N]``)."""
        unwrapped = getattr(env, "unwrapped", env)
        state = getattr(unwrapped, "getup_state", None)
        if state is None:
            try:
                from isaac_asimov.tasks.getup.mdp.state import ensure_state
            except Exception as err:  # not a get-up env (or Isaac Lab unavailable)
                print(f"[GetUpPPO] WARNING: env has no getup_state ({err}); deterministic mask not published.")
                return
            state = ensure_state(unwrapped)
        env_device = getattr(unwrapped, "device", self.device)
        state.deterministic = self.deterministic_mask.clone().to(env_device)
        print(
            f"[GetUpPPO] {self.num_deterministic_envs}/{self.deterministic_mask.numel()} envs act deterministically"
            " (policy mean) and are excluded from the PPO update: getup_state.deterministic"
        )

    # ----------------------------------------------------------------------------------------------------------------
    # Action std
    # ----------------------------------------------------------------------------------------------------------------

    def _std_distribution(self):
        return getattr(self.actor, "distribution", None)

    def _check_std_parameterization(self) -> None:
        dist = self._std_distribution()
        if dist is None or (self.max_action_std is None and self.min_action_std is None):
            return
        if not (hasattr(dist, "std_param") or hasattr(dist, "log_std_param")):
            raise ValueError(
                f"Action-std clamp needs a state-independent std parameter; {type(dist).__name__} has none."
                " Use GaussianDistribution or set max_action_std=min_action_std=None."
            )

    def action_std(self) -> torch.Tensor | None:
        """Current per-dim action std of the policy (None if the actor has no std parameter)."""
        dist = self._std_distribution()
        if dist is None:
            return None
        if hasattr(dist, "std_param"):
            return dist.std_param.detach()
        if hasattr(dist, "log_std_param"):
            return dist.log_std_param.detach().exp()
        return None

    @torch.no_grad()
    def _clamp_action_std(self) -> None:
        dist = self._std_distribution()
        if dist is None or (self.max_action_std is None and self.min_action_std is None):
            return
        if hasattr(dist, "std_param"):
            dist.std_param.data.clamp_(min=self.min_action_std, max=self.max_action_std)
        elif hasattr(dist, "log_std_param"):
            lo = math.log(self.min_action_std) if self.min_action_std is not None else None
            hi = math.log(self.max_action_std) if self.max_action_std is not None else None
            dist.log_std_param.data.clamp_(min=lo, max=hi)

    def _refresh_std_log(self) -> None:
        std = self.action_std()
        self._std_log = {} if std is None else {"Policy/std_mean": std.mean(), "Policy/std_max": std.max()}

    # ----------------------------------------------------------------------------------------------------------------
    # Rollout
    # ----------------------------------------------------------------------------------------------------------------

    def act(self, obs: TensorDict) -> torch.Tensor:
        actions = super().act(obs)
        if self.num_deterministic_envs > 0:
            # the stochastic forward pass left the distribution in place; its mean is the deterministic action
            det = self.deterministic_mask
            actions[det] = self.actor.output_mean[det].detach().to(actions.dtype)
        return actions

    def _reset_reward_stats(self) -> None:
        self._raw_reward_sum = 0.0
        self._norm_reward_sum = 0.0
        self._reward_steps = 0

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> None:
        if self.reward_normalizer is not None:
            raw = rewards.to(self.device)
            rewards = self.reward_normalizer(raw, dones.to(self.device))
            # tensors (no host sync per step); read once per iteration in reward_norm_stats()
            self._raw_reward_sum = self._raw_reward_sum + raw.mean()
            self._norm_reward_sum = self._norm_reward_sum + rewards.mean()
            self._reward_steps += 1
        if self._std_log:
            log = extras.setdefault("log", {})
            log.update(self._std_log)
        super().process_env_step(obs, rewards, dones, extras)

    def compute_returns(self, obs: TensorDict) -> None:
        """rsl-rl 5.0.1 GAE (ppo.py:187-209); advantages are normalized over the stochastic envs only."""
        if self.num_deterministic_envs == 0:
            return super().compute_returns(obs)
        st = self.storage
        last_values = self.critic(obs).detach()
        advantage = 0
        for step in reversed(range(st.num_transitions_per_env)):
            next_values = last_values if step == st.num_transitions_per_env - 1 else st.values[step + 1]
            next_is_not_terminal = 1.0 - st.dones[step].float()
            delta = st.rewards[step] + next_is_not_terminal * self.gamma * next_values - st.values[step]
            advantage = delta + next_is_not_terminal * self.gamma * self.lam * advantage
            st.returns[step] = advantage + st.values[step]
        st.advantages = st.returns - st.values
        if not self.normalize_advantage_per_mini_batch:
            adv = st.advantages[:, self._stochastic_env_ids]
            st.advantages = (st.advantages - adv.mean()) / (adv.std() + 1e-8)
        if self.deterministic_mask.any():
            st.advantages[:, self.deterministic_mask] = 0.0  # never used; keeps them out of any statistic

    def _mini_batch_generator(self):
        """``RolloutStorage.mini_batch_generator`` (5.0.1) restricted to the stochastic envs' transitions."""
        st = self.storage
        if self.actor.is_recurrent or self.critic.is_recurrent:
            return st.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        if self.num_deterministic_envs == 0:
            return st.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        return self._masked_generator()

    def _masked_generator(self):
        st = self.storage
        T = st.num_transitions_per_env
        valid = torch.nonzero((~self.deterministic_mask).repeat(T)).squeeze(-1)  # flat index t * N + n
        mini_batch_size = valid.numel() // self.num_mini_batches
        observations = st.observations.flatten(0, 1)
        actions = st.actions.flatten(0, 1)
        values = st.values.flatten(0, 1)
        returns = st.returns.flatten(0, 1)
        old_actions_log_prob = st.actions_log_prob.flatten(0, 1)
        advantages = st.advantages.flatten(0, 1)
        old_distribution_params = tuple(p.flatten(0, 1) for p in st.distribution_params)
        for _ in range(self.num_learning_epochs):
            indices = valid[torch.randperm(valid.numel(), device=valid.device)]
            for i in range(self.num_mini_batches):
                batch_idx = indices[i * mini_batch_size : (i + 1) * mini_batch_size]
                yield RolloutStorage.Batch(
                    observations=observations[batch_idx],
                    actions=actions[batch_idx],
                    values=values[batch_idx],
                    advantages=advantages[batch_idx],
                    returns=returns[batch_idx],
                    old_actions_log_prob=old_actions_log_prob[batch_idx],
                    old_distribution_params=tuple(p[batch_idx] for p in old_distribution_params),
                )

    # ----------------------------------------------------------------------------------------------------------------
    # L2C2
    # ----------------------------------------------------------------------------------------------------------------

    def _sample_l2c2_pairs(self, num_samples: int) -> tuple[TensorDict, TensorDict] | None:
        """Random ``(s_t, s_{t+1})`` pairs of the same env from the rollout storage, excluding reset boundaries."""
        st = self.storage
        T = st.num_transitions_per_env
        if T < 2:
            return None
        t_idx = torch.randint(0, T - 1, (num_samples,), device=self.device)
        stoch = self._stochastic_env_ids
        n_idx = stoch[torch.randint(0, stoch.numel(), (num_samples,), device=self.device)]
        valid = st.dones[t_idx, n_idx, 0] == 0
        t_idx, n_idx = t_idx[valid], n_idx[valid]
        if t_idx.numel() == 0:
            return None
        obs = TensorDict({k: v[t_idx, n_idx] for k, v in st.observations.items()}, batch_size=[t_idx.numel()])
        obs_next = TensorDict(
            {k: v[t_idx + 1, n_idx] for k, v in st.observations.items()}, batch_size=[t_idx.numel()]
        )
        return obs, obs_next

    def _l2c2_losses(self, num_samples: int) -> tuple[torch.Tensor, torch.Tensor]:
        pairs = self._sample_l2c2_pairs(num_samples)
        if pairs is None:
            zero = torch.zeros((), device=self.device)
            return zero, zero
        obs, obs_next = pairs
        u = torch.rand(obs.batch_size[0], 1, device=self.device) * self.l2c2_interp_max
        obs_interp = interpolate_obs(obs, obs_next, u)
        return l2c2_loss(self.actor, obs, obs_interp), l2c2_loss(self.critic, obs, obs_interp)

    # ----------------------------------------------------------------------------------------------------------------
    # Update: rsl-rl 5.0.1 PPO.update (ppo.py:211-416) + L2C2 + extra logging
    # ----------------------------------------------------------------------------------------------------------------

    def update(self) -> dict[str, float]:  # noqa: C901
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_rnd_loss = 0.0 if self.rnd else None
        mean_symmetry_loss = 0.0 if self.symmetry else None
        mean_l2c2_actor = 0.0 if self.l2c2_enabled else None
        mean_l2c2_critic = 0.0 if self.l2c2_enabled else None

        generator = self._mini_batch_generator()

        l2c2_samples = self.l2c2_num_samples
        if l2c2_samples is None:
            num_stochastic = self.storage.num_envs - self.num_deterministic_envs
            l2c2_samples = num_stochastic * self.storage.num_transitions_per_env // self.num_mini_batches

        for batch in generator:
            original_batch_size = batch.observations.batch_size[0]

            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)

            if self.symmetry and self.symmetry["use_data_augmentation"]:
                data_augmentation_func = self.symmetry["data_augmentation_func"]
                batch.observations, batch.actions = data_augmentation_func(
                    env=self.symmetry["_env"], obs=batch.observations, actions=batch.actions
                )
                num_aug = int(batch.observations.batch_size[0] / original_batch_size)
                batch.old_actions_log_prob = batch.old_actions_log_prob.repeat(num_aug, 1)
                batch.values = batch.values.repeat(num_aug, 1)
                batch.advantages = batch.advantages.repeat(num_aug, 1)
                batch.returns = batch.returns.repeat(num_aug, 1)

            self.actor(
                batch.observations,
                masks=batch.masks,
                hidden_state=batch.hidden_states[0],
                stochastic_output=True,
            )
            actions_log_prob = self.actor.get_output_log_prob(batch.actions)
            values = self.critic(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[1])
            distribution_params = tuple(p[:original_batch_size] for p in self.actor.output_distribution_params)
            entropy = self.actor.output_entropy[:original_batch_size]

            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = self.actor.get_kl_divergence(batch.old_distribution_params, distribution_params)
                    kl_mean = torch.mean(kl)
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))
            surrogate = -torch.squeeze(batch.advantages) * ratio
            surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            if self.use_clipped_value_loss:
                value_clipped = batch.values + (values - batch.values).clamp(-self.clip_param, self.clip_param)
                value_losses = (values - batch.returns).pow(2)
                value_losses_clipped = (value_clipped - batch.returns).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (batch.returns - values).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()

            if self.symmetry:
                if not self.symmetry["use_data_augmentation"]:
                    data_augmentation_func = self.symmetry["data_augmentation_func"]
                    batch.observations, _ = data_augmentation_func(
                        obs=batch.observations, actions=None, env=self.symmetry["_env"]
                    )
                mean_actions = self.actor(batch.observations.detach().clone())
                action_mean_orig = mean_actions[:original_batch_size]
                _, actions_mean_symm = data_augmentation_func(
                    obs=None, actions=action_mean_orig, env=self.symmetry["_env"]
                )
                symmetry_loss = nn.functional.mse_loss(
                    mean_actions[original_batch_size:], actions_mean_symm.detach()[original_batch_size:]
                )
                if self.symmetry["use_mirror_loss"]:
                    loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss
                else:
                    symmetry_loss = symmetry_loss.detach()

            if self.l2c2_enabled:
                l2c2_actor, l2c2_critic = self._l2c2_losses(l2c2_samples)
                loss = loss + self.l2c2_lambda_actor * l2c2_actor + self.l2c2_lambda_critic * l2c2_critic

            if self.rnd:
                with torch.no_grad():
                    rnd_state = self.rnd.get_rnd_state(batch.observations[:original_batch_size])
                    rnd_state = self.rnd.state_normalizer(rnd_state)
                predicted_embedding = self.rnd.predictor(rnd_state)
                target_embedding = self.rnd.target(rnd_state).detach()
                rnd_loss = nn.functional.mse_loss(predicted_embedding, target_embedding)

            self.optimizer.zero_grad()
            loss.backward()
            if self.rnd:
                self.rnd_optimizer.zero_grad()
                rnd_loss.backward()
            if self.is_multi_gpu:
                self.reduce_parameters()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
            self._clamp_action_std()
            if self.rnd_optimizer:
                self.rnd_optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy.mean().item()
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()
            if self.l2c2_enabled:
                mean_l2c2_actor += l2c2_actor.item()
                mean_l2c2_critic += l2c2_critic.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        self.storage.clear()

        loss_dict = {
            "value": mean_value_loss / num_updates,
            "surrogate": mean_surrogate_loss / num_updates,
            "entropy": mean_entropy / num_updates,
        }
        if mean_rnd_loss is not None:
            loss_dict["rnd"] = mean_rnd_loss / num_updates
        if mean_symmetry_loss is not None:
            loss_dict["symmetry"] = mean_symmetry_loss / num_updates
        if self.l2c2_enabled:
            loss_dict["l2c2_actor"] = mean_l2c2_actor / num_updates
            loss_dict["l2c2_critic"] = mean_l2c2_critic / num_updates
        loss_dict.update(self.reward_norm_stats())
        self._reset_reward_stats()
        self._refresh_std_log()
        return loss_dict

    def reward_norm_stats(self) -> dict[str, float]:
        """Reward-normalization diagnostics for the last rollout (empty if normalization is off)."""
        if self.reward_normalizer is None:
            return {}
        steps = max(self._reward_steps, 1)
        return {
            "reward_norm_return_std": float(self.reward_normalizer.std),
            "reward_raw_mean": float(self._raw_reward_sum) / steps,
            "reward_normalized_mean": float(self._norm_reward_sum) / steps,
        }

    # ----------------------------------------------------------------------------------------------------------------
    # Modes and checkpoints
    # ----------------------------------------------------------------------------------------------------------------

    def train_mode(self) -> None:
        super().train_mode()
        if self.reward_normalizer is not None:
            self.reward_normalizer.train()

    def eval_mode(self) -> None:
        super().eval_mode()
        if self.reward_normalizer is not None:
            self.reward_normalizer.eval()

    def save(self) -> dict:
        saved_dict = super().save()
        if self.reward_normalizer is not None:
            saved_dict["reward_normalizer_state_dict"] = self.reward_normalizer.state_dict()
        saved_dict["learning_rate"] = float(self.learning_rate)
        return saved_dict

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        load_iteration = super().load(loaded_dict, load_cfg, strict)
        if self.reward_normalizer is not None and "reward_normalizer_state_dict" in loaded_dict:
            if load_cfg is None or load_cfg.get("critic", False):  # the critic's value scale depends on it
                self.reward_normalizer.load_state_dict(loaded_dict["reward_normalizer_state_dict"], strict=strict)
        self._restore_learning_rate(loaded_dict, load_cfg)
        # the loaded std may lie outside this run's bounds (e.g. Stage B lowers max_action_std)
        self._clamp_action_std()
        self._refresh_std_log()
        return load_iteration

    def _restore_learning_rate(self, loaded_dict: dict, load_cfg: dict | None) -> None:
        """Set ``self.learning_rate`` (and every optimizer param group) after a load.

        * ``schedule="adaptive"`` and not ``reset_learning_rate_on_load``: resume the adapted lr, from
          ``loaded_dict["learning_rate"]`` or, for checkpoints saved before it was persisted, from the loaded
          optimizer's first param group. Otherwise keep the configured lr.
        * ``reset_learning_rate_on_load=True`` (warm-start fine-tunes) or a fixed schedule: the configured lr.

        Always re-syncs the optimizer param groups, since ``optimizer.load_state_dict`` restores the saved lr there
        while ``self.learning_rate`` is what the adaptive schedule updates from.
        """
        lr = self.initial_learning_rate
        optimizer_loaded = load_cfg is None or bool(load_cfg.get("optimizer", False))
        if self.schedule == "adaptive" and not self.reset_learning_rate_on_load:
            if "learning_rate" in loaded_dict:
                lr = float(loaded_dict["learning_rate"])
            elif optimizer_loaded and "optimizer_state_dict" in loaded_dict:
                lr = float(loaded_dict["optimizer_state_dict"]["param_groups"][0]["lr"])
        self.learning_rate = lr
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr


# ---------------------------------------------------------------------------------------------------------------------
# Isaac Lab config (only defined when isaaclab_rl is importable; the algorithm above needs only rsl-rl + torch)
# ---------------------------------------------------------------------------------------------------------------------

try:
    from isaaclab.utils import configclass
    from isaaclab_rl.rsl_rl import RslRlPpoAlgorithmCfg
except ImportError:  # pragma: no cover - CPU unit tests without Isaac Lab
    GetUpPPOAlgorithmCfg = None  # type: ignore[assignment,misc]
else:

    @configclass
    class GetUpPPOAlgorithmCfg(RslRlPpoAlgorithmCfg):
        """``GetUpPPO`` settings; override as needed."""

        class_name: str = "isaac_asimov.algorithms.getup_ppo:GetUpPPO"

        num_learning_epochs: int = 5
        num_mini_batches: int = 4
        learning_rate: float = 1.0e-3
        schedule: str = "adaptive"
        gamma: float = 0.995
        lam: float = 0.95
        entropy_coef: float = 0.005
        desired_kl: float = 0.01
        max_grad_norm: float = 1.0
        value_loss_coef: float = 1.0
        use_clipped_value_loss: bool = True
        clip_param: float = 0.2

        reward_norm: bool = True
        """Divide rewards by the running std of the discounted return."""
        reward_norm_decay: float = 0.999
        """EMA decay of the return moments (per env step; count-averaged for the first 1/(1-decay) steps)."""
        reward_norm_eps: float = 1.0e-2
        """Added to the return std before dividing."""
        reward_norm_clip: float | None = None
        """Optional clip of normalized per-step rewards (None = off)."""

        l2c2_enabled: bool = False
        """L2C2 smoothness regularization (off until Stage B)."""
        l2c2_lambda_actor: float = 1.0
        l2c2_lambda_critic: float = 0.1
        l2c2_interp_max: float = 1.0
        """Interpolation factor u ~ U(0, l2c2_interp_max) between s_t and s_{t+1}."""
        l2c2_num_samples: int | None = None
        """Consecutive-state pairs per mini-batch (None = mini-batch size)."""

        max_action_std: float | None = 1.0
        """Upper clamp of the policy's action std parameter, applied after every optimizer step (None = off)."""
        min_action_std: float | None = 0.05
        """Lower clamp of the policy's action std parameter (None = off)."""
        reset_learning_rate_on_load: bool = False
        """On checkpoint load, use ``learning_rate`` from this config instead of the checkpoint's adapted lr (for
        warm-start fine-tunes such as Stage B). By default a resume continues from the adapted lr."""
        deterministic_env_fraction: float = 0.05
        """The last ceil(f * N) envs act with the policy mean and are excluded from the PPO update; the mask is
        published as ``env.unwrapped.getup_state.deterministic`` for success metrics / curricula."""
