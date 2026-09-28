# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Measure the walking policy's smoothness/safety baselines: action jitter/rate under normal walking, and peak
non-foot impact force/impulse when it falls under escalating pushes.

One-off measurement tool (not part of the per-checkpoint eval loop) used to fill in `evaluate.py`'s
`--walking_jitter_baseline`/`--walking_impact_baseline_n` thresholds (the get-up policy's action jitter should be
<= 1.5x the walking policy's own, and its peak non-foot impact <= that of the walking policy's own falls).

Two modes, both against `Asimov1-Velocity-AMP-Play-v0`:
  --mode normal   Jitter (3rd-difference RMS) and action-rate under ordinary walking, same formulas as
                   `evaluate.py`'s generic-mode metrics. No pushes.
  --mode push      Escalating root-velocity pushes (random horizontal direction, magnitude ratcheting up
                   every `--push_interval_s`) applied to every env in lockstep, until each env's own
                   `fell_over` termination (`mdp.bad_orientation`, 70 deg tilt) fires -- the Play cfg's
                   episode length is effectively infinite (no time_out in practice), so every reset during
                   this mode is a genuine fall, not a timeout. Peak non-foot contact force and impulse are
                   accumulated over each life (reset to reset) and logged at the fall.

The Play scene only has a feet-only sensor and a self-collision sensor that is broken in this version of the
walking task, so this script adds its own
temporary all-body ContactSensorCfg to the scene before `gym.make` (mirrors the get-up task's own
`body_contact` sensor) -- added before construction, so it isn't subject to the post-construction
cfg-deep-copy trap documented in `_common.set_category_live`.

Usage:
    python scripts/getup/measure_walking_baseline.py --mode normal \\
        --checkpoint ~/isaac_asimov/logs/rsl_rl/asimov_velocity_amp/<run>/model_99.pt \\
        --num_envs 16 --num_seconds 30 --output_dir ~/getup_results/walking_baseline --headless
    python scripts/getup/measure_walking_baseline.py --mode push \\
        --checkpoint ~/isaac_asimov/logs/rsl_rl/asimov_velocity_amp/<run>/model_99.pt \\
        --num_envs 16 --num_seconds 60 --output_dir ~/getup_results/walking_baseline --headless
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from isaaclab.app import AppLauncher

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rsl_rl"))
import cli_args  # isort: skip

import _common as gc  # isort: skip

parser = argparse.ArgumentParser(description="Measure the walking policy's jitter/impact baselines.")
parser.add_argument("--task", type=str, default="Asimov1-Velocity-AMP-Play-v0")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--mode", type=str, required=True, choices=["normal", "push"])
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--num_seconds", type=float, default=30.0, help="Total sim time to run.")
parser.add_argument("--push_interval_s", type=float, default=2.0, help="[push mode] seconds between pushes per env.")
parser.add_argument("--push_base_mps", type=float, default=0.5, help="[push mode] first push's horizontal speed delta (m/s).")
parser.add_argument("--push_increment_mps", type=float, default=0.5, help="[push mode] speed added to the push each time an env survives one.")
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--seed", type=int, default=0)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if not args_cli.checkpoint:
    parser.error("--checkpoint is required")

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import importlib.metadata as metadata

import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils.assets import retrieve_file_path

import gymnasium as gym
import isaac_asimov.tasks  # noqa: F401
import isaaclab_tasks  # noqa: F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg, handle_deprecated_rsl_rl_checkpoint
from isaaclab_tasks.utils.hydra import hydra_task_config


def _load_onnx_mlp_as_torch(onnx_path: str, device) -> torch.nn.Module:
    """Parse a plain Gemm/Elu-only ONNX MLP into an equivalent `torch.nn.Sequential`, loading its exact
    weights -- no `onnxruntime` needed (only `onnx` is required). Expected layout
    (e.g. a 4-layer 78->512->256->128->23 walking MLP): `Gemm` layers with `Elu` after all but the last,
    `transB=1` on every `Gemm` (weight already stored
    `[out_features, in_features]`, i.e. exactly `torch.nn.Linear.weight`'s own layout -- no transpose needed).
    Raises rather than silently mis-loading if a future export doesn't match this shape (a different op, or
    `transB=0`).
    """
    import onnx
    from onnx import numpy_helper

    model = onnx.load(onnx_path)
    graph = model.graph
    inits = {init.name: torch.from_numpy(numpy_helper.to_array(init).copy()) for init in graph.initializer}

    layers: list[torch.nn.Module] = []
    for node in graph.node:
        if node.op_type == "Gemm":
            trans_b = next((a.i for a in node.attribute if a.name == "transB"), 0)
            if trans_b != 1:
                raise NotImplementedError(
                    f"ONNX node {node.name!r} has transB={trans_b} (expected 1 -- weight would need "
                    "transposing before it matches torch.nn.Linear's layout; not implemented)."
                )
            w = inits[node.input[1]]
            b = inits[node.input[2]] if len(node.input) > 2 else None
            out_f, in_f = w.shape
            lin = torch.nn.Linear(in_f, out_f, bias=b is not None)
            with torch.no_grad():
                lin.weight.copy_(w)
                if b is not None:
                    lin.bias.copy_(b)
            layers.append(lin)
        elif node.op_type == "Elu":
            alpha = next((a.f for a in node.attribute if a.name == "alpha"), 1.0)
            layers.append(torch.nn.ELU(alpha=alpha))
        elif node.op_type == "Identity":
            continue
        else:
            raise NotImplementedError(
                f"unsupported ONNX op {node.op_type!r} in {onnx_path!r} -- expected only Gemm/Elu/Identity "
                "for a 'plain MLP' walking policy export."
            )
    mlp = torch.nn.Sequential(*layers).to(device)
    mlp.eval()
    n_params = sum(p.numel() for p in mlp.parameters())
    print(f"[getup-eval] loaded ONNX MLP as torch: {mlp} ({n_params} params)")
    return mlp


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg):
    torch.manual_seed(args_cli.seed)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))

    # Temporary all-body contact sensor (the Play scene only ships feet_contact + a broken self_collision
    # sensor); added before gym.make so it's picked up by the real SceneCfg construction.
    env_cfg.scene.all_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=4, track_air_time=False
    )

    resume_path = retrieve_file_path(args_cli.checkpoint)
    is_onnx = resume_path.endswith(".onnx")
    print(f"[getup-eval] Loading checkpoint: {resume_path} mode={args_cli.mode} format={'onnx' if is_onnx else 'rsl_rl'}")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    raw_env: ManagerBasedRLEnv = env.unwrapped
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    if is_onnx:
        # Walking policies may be shipped as ONNX only.
        # onnxruntime may not be installed, but `onnx` is, and the graph is a plain Gemm/Elu MLP --
        # reconstructed natively in torch (see _load_onnx_mlp_as_torch) instead of adding a new dependency,
        # and this also sidesteps the exported graph's declared batch-size-1 input shape, since we run our
        # own torch forward pass rather than the ONNX graph's own (batch-size-N-native either way).
        mlp = _load_onnx_mlp_as_torch(resume_path, device=env.unwrapped.device)

        def policy(obs):
            with torch.inference_mode():
                return mlp(obs["policy"])
    else:
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        resume_path = handle_deprecated_rsl_rl_checkpoint(resume_path, metadata.version("rsl-rl-lib"))
        runner.load(resume_path)
        policy = runner.get_inference_policy(device=env.unwrapped.device)

    step_dt = env.unwrapped.step_dt
    num_steps = max(1, int(round(args_cli.num_seconds / step_dt)))
    num_envs = env.unwrapped.num_envs
    device = env.unwrapped.device

    contact_sensor = raw_env.scene.sensors["all_contact"]
    body_names = contact_sensor.body_names
    foot_mask = torch.tensor([gc.is_foot_body(n) for n in body_names], device=device)
    print(f"[getup-eval] contact bodies: {body_names}; foot mask: {foot_mask.tolist()}")

    # Same ground-only rule as evaluate.py: the "impact" metric must count only real ground contact, not
    # self-contact (e.g. pelvis_link<->hip_yaw_link shell contact during hip flexion is not a real impact).
    # Applied here too so both sides of the get-up vs walking impact comparison use the same definition of
    # "impact" (see _common.build_ground_contact_helper's docstring).
    robot = raw_env.scene["robot"]
    collision_spheres, terrain_height_fn, sensor_to_robot_idx = gc.build_ground_contact_helper(robot, contact_sensor, raw_env)

    with torch.inference_mode():
        obs, _ = env.reset()

    if args_cli.mode == "normal":
        result = _run_normal(env, policy, num_steps, num_envs, device)
    else:
        result = _run_push(
            env, raw_env, policy, contact_sensor, foot_mask, num_steps, num_envs, device, step_dt,
            push_interval_s=args_cli.push_interval_s, push_base_mps=args_cli.push_base_mps,
            push_increment_mps=args_cli.push_increment_mps,
            collision_spheres=collision_spheres, terrain_height_fn=terrain_height_fn, sensor_to_robot_idx=sensor_to_robot_idx,
        )

    result["mode"] = args_cli.mode
    result["checkpoint"] = resume_path
    result["num_envs"] = num_envs
    result["num_seconds"] = args_cli.num_seconds
    result["peak_rss_mb"] = gc.peak_rss_mb()
    result["peak_vram_mb"] = gc.peak_vram_mb()

    os.makedirs(args_cli.output_dir, exist_ok=True)
    out_path = os.path.join(args_cli.output_dir, f"walking_baseline_{args_cli.mode}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[getup-eval] Wrote {out_path}")
    print(json.dumps(result, indent=2))

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def _run_normal(env, policy, num_steps, num_envs, device) -> dict:
    from collections import deque

    action_hist = deque(maxlen=4)
    jitter_sq_sum = torch.zeros(num_envs, device=device)
    action_rate_sum = torch.zeros(num_envs, device=device)
    prev_action = None
    n_steps = 0

    with torch.inference_mode():
        obs, _ = env.reset()
        for step in range(num_steps):
            action = policy(obs)
            obs, _, dones, _ = env.step(action)
            # Compute jitter/action-rate on the clipped action actually applied (matches
            # evaluate.py's fallback for a non-getup task; the walking task has no LPF action term).
            effective_action = action.clamp(-1.0, 1.0)
            if prev_action is None:
                prev_action = torch.zeros_like(effective_action)
            action_hist.appendleft(effective_action.clone())
            delta = effective_action - prev_action
            action_rate_sum += delta.abs().mean(dim=-1)
            if len(action_hist) >= 4:
                third_diff = action_hist[0] - 3 * action_hist[1] + 3 * action_hist[2] - action_hist[3]
                jitter_sq_sum += (third_diff**2).mean(dim=-1)
            prev_action = effective_action
            n_steps += 1
            if step % 200 == 0:
                print(f"[getup-eval] normal: step {step}/{num_steps}", flush=True)

    jitter_rms = (jitter_sq_sum / max(n_steps, 1)).sqrt()
    action_rate = action_rate_sum / max(n_steps, 1)
    return {
        "action_jitter_rms_mean": float(jitter_rms.mean().item()),
        "action_jitter_rms_max": float(jitter_rms.max().item()),
        "action_rate_mean": float(action_rate.mean().item()),
        "action_rate_max": float(action_rate.max().item()),
        "n_steps": n_steps,
    }


def _run_push(env, raw_env, policy, contact_sensor, foot_mask, num_steps, num_envs, device, step_dt, *, push_interval_s, push_base_mps, push_increment_mps, collision_spheres=None, terrain_height_fn=None, sensor_to_robot_idx=None) -> dict:
    robot = raw_env.scene["robot"]
    push_interval_steps = max(1, int(round(push_interval_s / step_dt)))

    push_mag = torch.full((num_envs,), push_base_mps, device=device)
    peak_force = torch.zeros(num_envs, device=device)
    impulse = torch.zeros(num_envs, device=device)
    falls: list[dict] = []

    with torch.inference_mode():
        obs, _ = env.reset()
        for step in range(num_steps):
            action = policy(obs)
            obs, _, dones, _ = env.step(action)

            forces = gc.contact_force_norm(contact_sensor)  # [num_envs, num_bodies], substep-history max
            nonfoot = forces[:, ~foot_mask] if (~foot_mask).any() else forces
            # Ground-only, same rule as evaluate.py -- see _common.build_ground_contact_helper.
            if collision_spheres is not None:
                ground_mask = gc.ground_contact_mask_for_sensor(collision_spheres, terrain_height_fn, sensor_to_robot_idx, robot)
                ground_mask_nonfoot = ground_mask[:, ~foot_mask] if (~foot_mask).any() else ground_mask
                nonfoot = torch.where(ground_mask_nonfoot, nonfoot, torch.zeros_like(nonfoot))
            peak = nonfoot.max(dim=-1).values
            peak_force = torch.maximum(peak_force, peak)
            impulse += peak * step_dt

            done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
            if done_ids.numel() > 0:
                for i in done_ids.tolist():
                    falls.append({
                        "push_mag_mps_at_fall": float(push_mag[i].item()),
                        "peak_nonfoot_force_n": float(peak_force[i].item()),
                        "nonfoot_impulse_ns": float(impulse[i].item()),
                        "step": step,
                    })
                peak_force[done_ids] = 0.0
                impulse[done_ids] = 0.0
                push_mag[done_ids] = push_base_mps

            if step > 0 and step % push_interval_steps == 0:
                heading = torch.rand(num_envs, device=device) * 2 * torch.pi
                vel = robot.data.root_vel_w.clone()
                vel[:, 0] += push_mag * torch.cos(heading)
                vel[:, 1] += push_mag * torch.sin(heading)
                robot.write_root_velocity_to_sim(vel)
                push_mag += push_increment_mps

            if step % 200 == 0:
                print(f"[getup-eval] push: step {step}/{num_steps}, falls so far {len(falls)}", flush=True)

    if not falls:
        return {"falls": [], "n_falls": 0, "note": "no falls observed in the time budget; increase --num_seconds or --push_base_mps"}

    peak_forces = [f["peak_nonfoot_force_n"] for f in falls]
    impulses = [f["nonfoot_impulse_ns"] for f in falls]
    push_mags = [f["push_mag_mps_at_fall"] for f in falls]
    return {
        "falls": falls,
        "n_falls": len(falls),
        "peak_nonfoot_force_n_max": max(peak_forces),
        "peak_nonfoot_force_n_mean": float(np.mean(peak_forces)),
        "peak_nonfoot_force_n_median": float(np.median(peak_forces)),
        "nonfoot_impulse_ns_max": max(impulses),
        "nonfoot_impulse_ns_mean": float(np.mean(impulses)),
        "nonfoot_impulse_ns_median": float(np.median(impulses)),
        "push_mag_mps_at_fall_mean": float(np.mean(push_mags)),
        "push_mag_mps_at_fall_median": float(np.median(push_mags)),
    }


if __name__ == "__main__":
    # Same as evaluate.py/record_videos.py: guarantee os._exit() on every path, not just the success one.
    # An uncaught exception would otherwise skip os._exit(0) and can leave Kit's background threads/process
    # alive indefinitely.
    try:
        main()  # calls os._exit(0) itself on success
    except SystemExit:
        raise
    except BaseException:
        import traceback

        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
