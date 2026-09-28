# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Export the walking policy to ONNX with a firmware metadata contract.

Sibling to ``export_onnx.py`` (the get-up exporter), kept as a *separate* script rather than a shared
``--policy_kind`` flag on that one because the two action contracts are structurally different, not just
different numbers: the get-up path is ``q_meas + beta * s_j * LPF(clip(a))`` (relative, per-joint scale, a
curriculum-tuned beta/bound_scale read from ``curriculum_state_<iter>.json``) while the walking path is
plain ``default_pos + action_scale * a`` (absolute, one flat scale, no LPF, no curriculum state to read) --
see ``tasks/locomotion/velocity_env_cfg.py``'s ``ActionsCfg.joint_pos`` (a stock
``isaaclab.envs.mdp.JointPositionActionCfg``, not a custom term like the get-up task has). Sharing one script
with a branch for each would make the "one obvious code path" harder to audit than two short scripts.

Uses the *same* joint order (``ASIMOV_1_JOINT_NAMES``) as the get-up exporter -- confirmed from
``velocity_env_cfg.py``'s own ``ActionsCfg.joint_pos`` cfg (``joint_names=list(ASIMOV_1_JOINT_NAMES),
preserve_order=True``) -- so a firmware consumer that already parses the get-up ONNX's ``joint_order``
metadata key can use the identical parsing code for this one.

Usage:
    python scripts/getup/export_walk_onnx.py \\
        --checkpoint <run_dir>/model_3000.pt --output_dir ~/getup_results/deploy/<name>/onnx_walk
"""

from __future__ import annotations

import argparse
import os
import re
import sys

from isaaclab.app import AppLauncher

# cli_args lives in scripts/rsl_rl, not next to this file -- same path-insert export_onnx.py uses.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rsl_rl"))
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Export the walking policy to ONNX with firmware metadata.")
parser.add_argument("--task", type=str, default="Asimov1-Velocity-AMP-Play-v0")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--output_dir", type=str, default=None, help="Defaults to <checkpoint_dir>/exported/.")
parser.add_argument("--num_verify_samples", type=int, default=64)
parser.add_argument("--seed", type=int, default=0)
cli_args.add_rsl_rl_args(parser)  # adds --resume/--load_run/--checkpoint (required here, enforced below)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.enable_cameras = getattr(args_cli, "enable_cameras", False)
if not args_cli.checkpoint:
    parser.error("--checkpoint is required")
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- heavy imports (after AppLauncher) ---------------------------------------------------------------------------

import numpy as np
import onnx
import torch

import gymnasium as gym
from rsl_rl.runners import OnPolicyRunner

from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import (
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
    handle_deprecated_rsl_rl_cfg,
    handle_deprecated_rsl_rl_checkpoint,
)
from isaaclab_tasks.utils.hydra import hydra_task_config

import isaac_asimov.tasks  # noqa: F401
from isaac_asimov.assets.robots.asimov_1 import ASIMOV_1_ACTION_SCALE, ASIMOV_1_JOINT_NAMES, ASIMOV_1_STANDING_INIT_STATE

from importlib import metadata as _importlib_metadata

from packaging import version as _pkg_version

_INSTALLED_RSL_RL_VERSION = _importlib_metadata.version("rsl-rl-lib")


def build_metadata(raw_env) -> dict[str, str]:
    """Firmware metadata for the walking policy: absolute-position contract, one flat scale, no LPF,
    no per-joint relative-action bound (see module docstring for why this differs from the get-up path)."""
    asset = raw_env.scene["robot"]
    joint_ids, joint_names = asset.find_joints(list(ASIMOV_1_JOINT_NAMES), preserve_order=True)
    assert joint_names == list(ASIMOV_1_JOINT_NAMES), (
        f"joint order drift: asset resolved {joint_names}, expected {list(ASIMOV_1_JOINT_NAMES)}"
    )
    kp_live = asset.data.default_joint_pos.new_zeros(len(joint_ids))
    kd_live = asset.data.default_joint_pos.new_zeros(len(joint_ids))
    for act in asset.actuators.values():
        ids = act.joint_indices
        if isinstance(ids, slice):
            ids = torch.arange(asset.num_joints, device=raw_env.device)
        stiff = act.stiffness[0] if act.stiffness.dim() > 0 else act.stiffness
        damp = act.damping[0] if act.damping.dim() > 0 else act.damping
        kp_live[ids] = stiff.to(kp_live.device) if torch.is_tensor(stiff) else float(stiff)
        kd_live[ids] = damp.to(kd_live.device) if torch.is_tensor(damp) else float(damp)
    kp_live = kp_live[joint_ids]
    kd_live = kd_live[joint_ids]

    resolved_default_pos = []
    for name in ASIMOV_1_JOINT_NAMES:
        val = 0.0
        for pattern, v in ASIMOV_1_STANDING_INIT_STATE.joint_pos.items():
            if re.fullmatch(pattern, name):
                val = v
        resolved_default_pos.append(val)

    def _csv(values) -> str:
        return ",".join(f"{float(v):.8g}" for v in values)

    return {
        "joint_order": ",".join(ASIMOV_1_JOINT_NAMES),
        "joint_stiffness": _csv(kp_live.tolist()),
        "joint_damping": _csv(kd_live.tolist()),
        "default_joint_pos": _csv(resolved_default_pos),
        "action_scale": f"{float(ASIMOV_1_ACTION_SCALE):.8g}",  # flat scalar (all 23 joints), unlike the get-up per-joint s_j
        "action_mode": "absolute",  # target = default_joint_pos + action_scale * a  (NOT relative to q_meas like the get-up path)
        "action_beta": "1.0",  # no curriculum bound multiplier on this path; kept for schema parity with the get-up ONNX
        "action_lpf_alpha": "n/a",  # no low-pass filter on the walking path (confirmed: velocity_env_cfg.ActionsCfg.joint_pos is a stock JointPositionActionCfg)
        "action_relative_per_cycle": "false",
        "obs_delay_min_max_steps": "0,2",  # base_ang_vel lag 0-1, projected_gravity lag 0-2 (velocity_env_cfg.ObservationsCfg.PolicyCfg) -- see README.md in the deploy package for exact per-term lags
        "contract_source": "velocity_env_cfg.ActionsCfg (fixed, no curriculum)",
    }


def write_onnx_metadata(onnx_path: str, metadata: dict[str, str]) -> None:
    model = onnx.load(onnx_path)
    del model.metadata_props[:]
    for key, value in metadata.items():
        entry = model.metadata_props.add()
        entry.key = key
        entry.value = value
    onnx.save(model, onnx_path)


def save_verification_samples(npz_path: str, policy, obs_dim: int, num_samples: int, seed: int) -> None:
    if hasattr(policy, "to"):
        policy = policy.to("cpu")
    if hasattr(policy, "eval"):
        policy.eval()
    rng = np.random.default_rng(seed)
    samples = rng.normal(size=(num_samples, obs_dim)).astype(np.float32)
    with torch.inference_mode():
        torch_out = policy(torch.from_numpy(samples)).cpu().numpy()
    np.savez(npz_path, obs=samples, torch_actions=torch_out)
    print(f"[getup-export] Saved {num_samples} verification samples to {npz_path}")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg):
    torch.manual_seed(args_cli.seed)
    env_cfg.seed = args_cli.seed
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, _INSTALLED_RSL_RL_VERSION)

    env = gym.make(args_cli.task, cfg=env_cfg)
    raw_env = env.unwrapped
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)

    checkpoint_path = os.path.expanduser(args_cli.checkpoint)
    resume_path = retrieve_file_path(checkpoint_path)
    resume_path = handle_deprecated_rsl_rl_checkpoint(resume_path, _INSTALLED_RSL_RL_VERSION)
    print(f"[getup-export] Loading checkpoint: {resume_path}")
    runner.load(resume_path)
    output_dir = args_cli.output_dir or os.path.join(os.path.dirname(resume_path), "exported")
    os.makedirs(output_dir, exist_ok=True)

    if _pkg_version.parse(_INSTALLED_RSL_RL_VERSION) >= _pkg_version.parse("4.0.0"):
        runner.export_policy_to_jit(path=output_dir, filename="policy.pt")
        runner.export_policy_to_onnx(path=output_dir, filename="policy.onnx")
    else:
        from rsl_rl.utils import export_policy_as_jit, export_policy_as_onnx

        policy_nn = runner.alg.policy if hasattr(runner.alg, "policy") else runner.alg.actor_critic
        normalizer = getattr(policy_nn, "actor_obs_normalizer", None)
        export_policy_as_jit(policy_nn, normalizer=normalizer, path=output_dir, filename="policy.pt")
        export_policy_as_onnx(policy_nn, normalizer=normalizer, path=output_dir, filename="policy.onnx")
    onnx_path = os.path.join(output_dir, "policy.onnx")

    onnx_metadata = build_metadata(raw_env)
    write_onnx_metadata(onnx_path, onnx_metadata)
    print(f"[getup-export] Wrote {len(onnx_metadata)} ONNX metadata keys to {onnx_path}")
    for k, v in onnx_metadata.items():
        preview = v if len(v) <= 80 else v[:77] + "..."
        print(f"[getup-export]   {k} = {preview}")

    obs_group_name = next(iter(agent_cfg.obs_groups.get("actor", ["policy"])))

    class _FlatObsAdapter:
        def __init__(self, policy, group: str):
            self._policy = policy
            self._group = group

        def __call__(self, x: torch.Tensor) -> torch.Tensor:
            return self._policy({self._group: x})

    obs_dict, _ = raw_env.reset()
    obs_dim = obs_dict[obs_group_name].shape[-1]
    inference_policy = _FlatObsAdapter(runner.get_inference_policy(device="cpu"), obs_group_name)
    npz_path = os.path.join(output_dir, "verify_samples.npz")
    save_verification_samples(npz_path, inference_policy, obs_dim, args_cli.num_verify_samples, args_cli.seed)

    env.close()
    print(f"[getup-export] Done. ONNX + metadata at {onnx_path}")
    print("[getup-export] Next: python3 scripts/getup/verify_onnx_parity.py "
          f"--onnx {onnx_path} --samples {npz_path}")


if __name__ == "__main__":
    main()
    simulation_app.close()
