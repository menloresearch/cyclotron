# Copyright (c) 2026, Menlo Research.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Asimov-1 get-up tasks."""

import gymnasium as gym

from . import agents

gym.register(
    id="Asimov1-GetUp-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.getup_env_cfg:Asimov1GetUpEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Asimov1GetUpPPORunnerCfg",
    },
)

gym.register(
    id="Asimov1-GetUp-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.getup_env_cfg:Asimov1GetUpEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Asimov1GetUpPPORunnerCfg",
    },
)

gym.register(
    id="Asimov1-GetUp-StageB-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.getup_env_cfg:Asimov1GetUpStageBEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Asimov1GetUpStageBPPORunnerCfg",
    },
)

gym.register(
    id="Asimov1-GetUp-StageB-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.getup_env_cfg:Asimov1GetUpStageBEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Asimov1GetUpStageBPPORunnerCfg",
    },
)

gym.register(
    id="Asimov1-GetUp-StageB2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.getup_env_cfg:Asimov1GetUpStageB2EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Asimov1GetUpStageB2PPORunnerCfg",
    },
)

gym.register(
    id="Asimov1-GetUp-StageB2-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.getup_env_cfg:Asimov1GetUpStageB2EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Asimov1GetUpStageB2PPORunnerCfg",
    },
)

gym.register(
    id="Asimov1-GetUp-StageB3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.getup_env_cfg:Asimov1GetUpStageB3EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Asimov1GetUpStageB3PPORunnerCfg",
    },
)

gym.register(
    id="Asimov1-GetUp-StageB3-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.getup_env_cfg:Asimov1GetUpStageB3EnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Asimov1GetUpStageB3PPORunnerCfg",
    },
)

gym.register(
    id="Asimov1-GetUp-StageB2R-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.getup_env_cfg:Asimov1GetUpStageB2REnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Asimov1GetUpStageB2PPORunnerCfg",
    },
)

gym.register(
    id="Asimov1-GetUp-StageB2R-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.getup_env_cfg:Asimov1GetUpStageB2REnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Asimov1GetUpStageB2PPORunnerCfg",
    },
)
