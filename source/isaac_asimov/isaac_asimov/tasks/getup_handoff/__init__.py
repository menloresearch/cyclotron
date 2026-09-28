# Copyright (c) 2026, Menlo Research.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Get-up <-> walking handoff demo task.

Registers the frozen gym id ``Asimov1-GetUp-Handoff-Play-v0``: evaluation-only, no training
runner. ``scripts/getup/play_combined.py`` drives it directly with two loaded checkpoints
(get-up and walking) rather than through ``rsl_rl``'s ``OnPolicyRunner``; the
``rsl_rl_cfg_entry_point`` below is included only so generic tooling that expects one on every
registered task (e.g. ``gym.make`` via ``isaaclab_tasks.utils.hydra.hydra_task_config``) still
works, and reuses the get-up task's own runner cfg since this env's action/obs shapes are a
superset of the get-up task's.

IMPORTANT: both entry points below are **strings**, resolved
lazily by gymnasium/Isaac Lab only when the env is actually constructed (``gym.make`` /
``hydra_task_config``) — this module must NOT ``from . import handoff_env_cfg`` at module scope.
``handoff_env_cfg.py`` imports ``isaac_asimov.tasks.getup.getup_env_cfg``, and
``isaac_asimov.tasks.__init__`` auto-imports every task subpackage
(``isaaclab_tasks.utils.import_packages``) as soon as *anyone* does ``import isaac_asimov.tasks`` —
which every training/eval/play script does. An eager import here would make any error in that
dependency chain break ``import isaac_asimov.tasks`` for every task (including walking). Match the
pattern already used by ``tasks/getup/__init__.py`` and ``tasks/locomotion/__init__.py``: only ``gym.register`` with
string targets at import time.
"""

import gymnasium as gym

gym.register(
    id="Asimov1-GetUp-Handoff-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.handoff_env_cfg:Asimov1GetUpHandoffPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "isaac_asimov.tasks.getup.agents.rsl_rl_ppo_cfg:Asimov1GetUpPPORunnerCfg",
    },
)
