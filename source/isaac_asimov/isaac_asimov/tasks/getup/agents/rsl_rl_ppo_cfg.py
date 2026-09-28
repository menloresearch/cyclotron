# Copyright (c) 2026, Menlo Research.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""rsl-rl runner configuration for the Asimov 1 get-up task."""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg, RslRlSymmetryCfg

from isaac_asimov.tasks.getup.symmetry import mirror_getup

try:
    from isaac_asimov.algorithms.getup_ppo import GetUpPPOAlgorithmCfg
except ImportError:  # pragma: no cover
    GetUpPPOAlgorithmCfg = None

if GetUpPPOAlgorithmCfg is None:  # fallback: only if the GetUpPPO config class is unavailable

    @configclass
    class GetUpPPOAlgorithmCfg(RslRlPpoAlgorithmCfg):  # type: ignore[no-redef]
        class_name: str = "isaac_asimov.algorithms.getup_ppo:GetUpPPO"


# Exploration-bounding fields (std clamp, deterministic evaluation envs); only set if this GetUpPPOAlgorithmCfg has them
_V11_FIELDS = {"max_action_std": 1.0, "deterministic_env_fraction": 0.05}
_V11_KWARGS = {k: v for k, v in _V11_FIELDS.items() if k in getattr(GetUpPPOAlgorithmCfg, "__dataclass_fields__", {})}


@configclass
class Asimov1GetUpPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 250
    experiment_name = "asimov1_getup"
    run_name = "ppo"
    obs_groups = {"actor": ["policy"], "critic": ["critic"]}
    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.5, std_type="scalar"),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
    )
    algorithm = GetUpPPOAlgorithmCfg(
        class_name="isaac_asimov.algorithms.getup_ppo:GetUpPPO",
        optimizer="adam",
        normalize_advantage_per_mini_batch=False,
        rnd_cfg=None,
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=True,
            use_mirror_loss=False,
            data_augmentation_func=mirror_getup,
        ),
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.002,  # was 0.005; exploration std ran away past the +-1 clip
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.995,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        **_V11_KWARGS,
    )


@configclass
class Asimov1GetUpStageBPPORunnerCfg(Asimov1GetUpPPORunnerCfg):
    """Stage B fine-tune: warm-started from a Stage A checkpoint via ``agents/warm_start.py``.

    L2C2 on (lambda_pi 1, lambda_V 0.1), a lower initial learning rate and a tighter exploration-std clamp
    (the warm-start file resets the std to ``WARM_START_STD``; the optimizer state is not carried over)."""

    experiment_name = "asimov1_getup_stageB"
    run_name = "stageB"
    max_iterations = 6000
    save_interval = 250

    def __post_init__(self):
        if hasattr(super(), "__post_init__"):
            super().__post_init__()
        self.algorithm.learning_rate = 3.0e-4
        for k, v in {"l2c2_enabled": True, "l2c2_lambda_actor": 1.0, "l2c2_lambda_critic": 0.1,
                     "max_action_std": 0.6, "reset_learning_rate_on_load": True}.items():  # fmt: skip
            if hasattr(self.algorithm, k):
                setattr(self.algorithm, k, v)


WARM_START_STD = 0.4
"""Exploration std written into the Stage B warm-start checkpoint (<= the Stage B clamp ``max_action_std`` 0.6)."""


@configclass
class Asimov1GetUpStageB2PPORunnerCfg(Asimov1GetUpStageBPPORunnerCfg):
    """Stage B2: warm start from a Stage B checkpoint; same experiment folder as Stage B (the warm start is built
    there). lr 2e-4 reset on load, std clamp 0.5, 2000 iterations."""

    run_name = "stageB2"
    max_iterations = 2000

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.learning_rate = 2.0e-4
        if hasattr(self.algorithm, "max_action_std"):
            self.algorithm.max_action_std = 0.5


@configclass
class Asimov1GetUpStageB3PPORunnerCfg(Asimov1GetUpStageB2PPORunnerCfg):
    """Stage B3: warm start from a Stage B2 checkpoint; lr 1.5e-4 reset on load, std clamp 0.45, 2000 iterations."""

    run_name = "stageB3"

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.learning_rate = 1.5e-4
        if hasattr(self.algorithm, "max_action_std"):
            self.algorithm.max_action_std = 0.45
