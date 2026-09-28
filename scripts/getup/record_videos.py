# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Render per-category get-up video clips from an RSL-RL checkpoint, headless.

Loads a checkpoint the same way `scripts/rsl_rl/play.py` does, and for each requested start category
(the `reset_fallen` category keys) records one or more episodes with a tracking camera that follows the robot
root at a fixed 3/4 front-side angle (eye ~0.9m high, lookat z=0.35m, distance ~2.4m by default -- see
`_common.camera_world_pose`), relative to the robot's own EMA-smoothed heading and xy position (not a
fixed world angle: the reset event randomizes spawn yaw, so a world-fixed camera ends up behind the
robot as often as in front of it). Each clip is 1280x720 (default), with a live overlay (run id,
iteration, category, assist level, a clock, and a "standing at X.Xs" line once `getup_state.success`
latches). Frames are written to disk as they're captured (one ffmpeg writer per episode,
`writer.close()` in a `finally`), a progress line is printed every `--progress_every_frames` frames,
and the process exits via `os._exit(0)` rather than the normal `env.close()`/`simulation_app.close()`
teardown (headless Isaac Sim can hang there for many minutes while holding the GPU).

Works against any registered task, not just the get-up one (e.g. `Asimov1-Velocity-AMP-Play-v0`, which
has no `getup_state`/`reset_fallen`). When those getup-specific pieces are absent, this script still
renders a clip labeled "category: default", just without the success/assist overlay.

Usage:
    python scripts/getup/record_videos.py --task Asimov1-GetUp-Play-v0 \\
        --checkpoint ~/isaac_asimov/logs/rsl_rl/asimov1_getup/<run>/model_999.pt \\
        --categories supine,prone,mid_fall --episodes_per_category 1 \\
        --output_dir ~/getup_results/<run>/videos/iter_999 --headless --enable_cameras
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

parser = argparse.ArgumentParser(description="Record per-category get-up video clips from a checkpoint.")
parser.add_argument("--task", type=str, default=gc.GETUP_TASK_PLAY)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point (see scripts/rsl_rl/play.py).")
# NOTE: --checkpoint itself is added by cli_args.add_rsl_rl_args() below -- declaring it twice is an
# argparse conflict (see scripts/rsl_rl/play.py's --target, which sidesteps this the same way).
parser.add_argument("--num_envs", type=int, default=1, help="Kept small on purpose (RAM/VRAM must fit alongside a training run).")
parser.add_argument("--categories", type=str, default=",".join(gc.CATEGORY_KEYS))
parser.add_argument("--episodes_per_category", type=int, default=1)
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--fps", type=int, default=None, help="Output fps. Defaults to (1/step_dt)/render_every_n_steps, computed once step_dt is known.")
parser.add_argument("--width", type=int, default=1280)
parser.add_argument("--height", type=int, default=720)
parser.add_argument("--max_seconds", type=float, default=15.0, help="Hard per-clip cutoff, independent of the env's own episode length.")
parser.add_argument("--run_id", type=str, default=None, help="Defaults to the checkpoint's run directory name.")
parser.add_argument("--iteration", type=str, default=None, help="Defaults to the number parsed out of the checkpoint filename.")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument(
    "--render_every_n_steps", type=int, default=2,
    help="Capture a frame every N policy steps (default 2 -> 25 fps of sim time at the 50 Hz get-up policy rate, "
    "half the render/GPU load of capturing every step). --fps defaults to match this unless overridden.",
)
parser.add_argument("--progress_every_frames", type=int, default=25, help="Print a progress line every N frames written, so a hang is visible.")
parser.add_argument(
    "--camera_distance_m", type=float, default=3.2,
    help="3/4 camera distance from the robot root. 2.4 m cropped the head when the robot is fully standing, "
    "hence the wider default.",
)
parser.add_argument(
    "--camera_eye_height_m", type=float, default=1.0,
    help="Fixed camera eye height (not height-aware -- see _common.camera_world_pose); raised from ~0.9 m "
    "alongside the distance increase, for the same head-cropping reason.",
)
parser.add_argument(
    "--camera_lookat_height_m", type=float, default=0.5,
    help="Fixed camera look-at height; raised from ~0.35 m so the vertical frame is centered between the "
    "ground (feet) and a standing robot's head instead of biased toward the ground.",
)
parser.add_argument(
    "--camera_azimuth_deg", type=float, default=-45.0,
    help="Camera azimuth offset relative to the robot's own (smoothed) yaw. -45 shows the robot's front "
    "(visible chest logo), not its back.",
)
parser.add_argument("--camera_smoothing_alpha", type=float, default=0.15, help="EMA alpha for the camera's xy/yaw tracking (lower = smoother, more lag).")
parser.add_argument("--warmup_frames", type=int, default=3, help="Render-and-discard this many frames right after reset, before recording (skips the black warm-up frame(s)).")
parser.add_argument(
    "--effort_scale", type=str, default="trained",
    help="Actuator effort-limit scale: a float, or one of the env's presets (nominal=1.0/stage1/stageB). Default "
    "'trained' reads the checkpoint's own curriculum_state_<N>.json action_contract and "
    "applies its exact per-joint bound_scale/beta, so a Stage B (or any mid-curriculum) checkpoint is shown "
    "moving the way it actually trained, not at a mismatched (e.g. nominal) bound -- falls back to the Play "
    "cfg's own baked-in default (no override) if the checkpoint has no curriculum_state file.",
)
parser.add_argument(
    "--terrain", type=str, default="flat", choices=["flat", "rough", "mix"],
    help="Terrain toggle: 'rough'/'mix' use the env cfg's set_eval_terrain, "
    "applied before gym.make. The camera framing accounts for the local terrain height under the robot (not "
    "just world z=0), so the whole body stays in frame off flat ground too -- see _common.camera_world_pose's "
    "ground_z_m.",
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.enable_cameras = True  # offscreen rendering requires this; force it on regardless of the CLI.
if not args_cli.checkpoint:
    parser.error("--checkpoint is required")

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import importlib.metadata as metadata

import cv2
import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.assets import retrieve_file_path

import gymnasium as gym
import imageio
import isaac_asimov.tasks  # noqa: F401
import isaaclab_tasks  # noqa: F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg, handle_deprecated_rsl_rl_checkpoint
from isaaclab_tasks.utils.hydra import hydra_task_config

try:
    from isaac_asimov.tasks.getup import mdp as getup_mdp  # noqa: F401
except Exception as exc:  # noqa: BLE001
    getup_mdp = None
    print(f"[getup-eval] NOTE: getup.mdp not importable ({exc!r}); recording in generic mode.")


def _run_and_iteration(checkpoint_path: str) -> tuple[str, str]:
    run_id = os.path.basename(os.path.dirname(os.path.abspath(checkpoint_path)))
    stem = os.path.splitext(os.path.basename(checkpoint_path))[0]
    digits = "".join(ch for ch in stem if ch.isdigit())
    return run_id, (digits if digits else stem)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg):
    torch.manual_seed(args_cli.seed)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))

    has_getup = getup_mdp is not None and getattr(getattr(env_cfg, "events", None), "reset_fallen", None) is not None
    categories = [c.strip() for c in args_cli.categories.split(",") if c.strip()] if has_getup else ["default"]

    resume_path = retrieve_file_path(args_cli.checkpoint)
    run_id = args_cli.run_id or _run_and_iteration(resume_path)[0]
    iteration = args_cli.iteration or _run_and_iteration(resume_path)[1]

    # Same rule as evaluate.py: the default must replicate exactly what the
    # checkpoint was trained under -- a Stage B checkpoint shown at a mismatched (e.g. nominal) bound moves
    # differently than it actually trained. "trained" reads curriculum_state_<N>.json's action_contract
    # (exact per-joint bound_scale + beta), applied AFTER gym.make (on the constructed action-term object) --
    # NOT via env_cfg before it, since that's a per-joint list, not one of apply_play_effort_scale's
    # presets/floats. Falls back to the Play cfg's own baked-in default (no override) if missing. Explicit
    # presets/floats still work as before, set BEFORE gym.make (deep-copy timing, see evaluate.py's note).
    curriculum_state = gc.load_curriculum_state(resume_path)
    checkpoint_contract = curriculum_state.get("action_contract") if curriculum_state else None
    use_trained = args_cli.effort_scale == "trained"
    trained_contract_available = use_trained and checkpoint_contract and "bound_scale" in checkpoint_contract and "beta" in checkpoint_contract
    if use_trained and not trained_contract_available:
        print(f"[getup-eval] NOTE: --effort_scale trained requested but no usable curriculum_state_<N>.json for "
              f"{resume_path!r}; falling back to the Play cfg's own baked-in default (no override).")
    effort_scale_value = None if use_trained else gc.parse_effort_scale_arg(args_cli.effort_scale)
    if not use_trained and hasattr(env_cfg, "play_effort_scale"):
        env_cfg.play_effort_scale = effort_scale_value

    terrain_note = gc.apply_terrain_toggle(env_cfg, args_cli.terrain)

    print(f"[getup-eval] Recording from checkpoint: {resume_path} (run={run_id}, iter={iteration}, "
          f"effort_scale={args_cli.effort_scale}{' [trained contract found]' if trained_contract_available else ''}, "
          f"terrain={args_cli.terrain} [{terrain_note}])")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    raw_env: ManagerBasedRLEnv = env.unwrapped
    if trained_contract_available and getup_mdp is not None and hasattr(getup_mdp, "apply_play_effort_scale"):
        bound_scale_dict = dict(zip(checkpoint_contract["joint_names"], checkpoint_contract["bound_scale"]))
        getup_mdp.apply_play_effort_scale(raw_env, None, bound_scale_dict, None)
        try:
            raw_env.action_manager.get_term("joint_pos").beta = float(checkpoint_contract["beta"])
        except AttributeError:
            pass
    elif not use_trained and getup_mdp is not None and hasattr(getup_mdp, "apply_play_effort_scale"):
        # Redundant with the startup event term above (harmless: set_effort_scale is not cumulative) -- a
        # safety net in case play_effort_scale isn't a field on this env_cfg for some reason.
        getup_mdp.apply_play_effort_scale(raw_env, None, effort_scale_value)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    resume_path = handle_deprecated_rsl_rl_checkpoint(resume_path, metadata.version("rsl-rl-lib"))
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    step_dt = env.unwrapped.step_dt
    max_steps = max(1, int(round(args_cli.max_seconds / step_dt)))
    render_every = max(1, args_cli.render_every_n_steps)
    fps = args_cli.fps or max(1, round((1.0 / step_dt) / render_every))

    os.makedirs(args_cli.output_dir, exist_ok=True)
    clip_paths: list[tuple[str, str]] = []  # (category, path)

    for category in categories:
        expected_idx = gc.set_category_live(raw_env, category) if has_getup else None
        for ep in range(args_cli.episodes_per_category):
            path = _record_one_episode(
                env=env,
                raw_env=raw_env,
                policy=policy,
                category=category,
                expected_category_idx=expected_idx,
                episode_idx=ep,
                run_id=run_id,
                iteration=iteration,
                has_getup=has_getup,
                step_dt=step_dt,
                max_steps=max_steps,
                output_dir=args_cli.output_dir,
                fps=fps,
                width=args_cli.width,
                height=args_cli.height,
                render_every_n_steps=render_every,
                progress_every_frames=args_cli.progress_every_frames,
                camera_distance_m=args_cli.camera_distance_m,
                camera_eye_height_m=args_cli.camera_eye_height_m,
                camera_lookat_height_m=args_cli.camera_lookat_height_m,
                camera_azimuth_deg=args_cli.camera_azimuth_deg,
                camera_smoothing_alpha=args_cli.camera_smoothing_alpha,
                warmup_frames=args_cli.warmup_frames,
            )
            clip_paths.append((category, path))
            print(f"[getup-eval] done: {path}", flush=True)

    usage = {
        "peak_rss_mb": gc.peak_rss_mb(),
        "peak_vram_mb": gc.peak_vram_mb(),
        "clips": [{"category": c, "path": p} for c, p in clip_paths],
    }
    with open(os.path.join(args_cli.output_dir, "resource_usage.json"), "w") as f:
        json.dump(usage, f, indent=2)
    print(f"[getup-eval] peak RSS = {usage['peak_rss_mb']:.0f} MB, peak VRAM = {usage['peak_vram_mb']}", flush=True)

    # Skip env.close()/simulation_app.close(): Isaac Sim is known to hang in simulation_app.close() while
    # holding the GPU. All clips and resource_usage.json are already
    # flushed to disk above, so there is nothing left to lose by exiting immediately.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def _record_one_episode(
    *, env, raw_env, policy, category, expected_category_idx, episode_idx, run_id, iteration, has_getup, step_dt,
    max_steps, output_dir, fps, width, height, render_every_n_steps, progress_every_frames, camera_distance_m,
    camera_eye_height_m, camera_lookat_height_m, camera_azimuth_deg, camera_smoothing_alpha, warmup_frames,
) -> str:
    """Stream frames straight to disk (one ffmpeg writer per episode, appended to as we go) instead of buffering
    the whole episode in memory and encoding at the end. `writer.close()` in the `finally` block means a clip is
    flushed and playable even if this episode raises partway through (a hang or crash must not lose
    already-captured frames, and progress must be visible via `--progress_every_frames`).

    The camera is driven manually each rendered step via `raw_env.sim.set_camera_view(eye=, target=)`
    (`_common.camera_world_pose` + `_common.SmoothedTracker`) instead of Isaac Lab's built-in asset-root
    viewer tracking -- see `_common.py`'s camera section docstring for why (fixed 3/4 framing, EMA
    smoothing, and it drops a whole class of teardown bugs the built-in tracker's event subscription caused).
    """
    filename = f"{category}_ep{episode_idx}.mp4"
    path = os.path.join(output_dir, filename)

    getup_state = getattr(raw_env, "getup_state", None) if has_getup else None
    robot = raw_env.scene["robot"]
    tracker = gc.SmoothedTracker(alpha=camera_smoothing_alpha)

    def _update_camera():
        pos = robot.data.root_pos_w[0]
        yaw = gc.quat_yaw(*robot.data.root_quat_w[0].tolist())
        x, y, smoothed_yaw = tracker.update(float(pos[0].item()), float(pos[1].item()), yaw)
        # Keep the full body in frame on rough terrain: eye/lookat heights were tuned
        # as heights above flat ground (world z=0); on rough/mix terrain the local ground under the robot can
        # sit well off world z=0, which would crop the body. `pelvis_height` already ray-casts against the
        # terrain under the robot, so `root_z - pelvis_height` recovers the local ground height without
        # duplicating that lookup -- deliberately NOT the robot's own current z (that would make the camera
        # rise/fall with the robot as it gets up, defeating the fixed-framing-height design).
        ground_z = 0.0
        if getup_mdp is not None and hasattr(getup_mdp, "pelvis_height"):
            try:
                ground_z = float(pos[2].item() - getup_mdp.pelvis_height(raw_env)[0].item())
            except Exception:  # noqa: BLE001 - defensive: fall back to flat-ground framing rather than crash a render
                ground_z = 0.0
        eye, lookat = gc.camera_world_pose(
            x, y, smoothed_yaw,
            azimuth_offset_deg=camera_azimuth_deg, distance_m=camera_distance_m,
            eye_height_m=camera_eye_height_m, lookat_height_m=camera_lookat_height_m,
            ground_z_m=ground_z,
        )
        raw_env.sim.set_camera_view(eye=eye, target=lookat)

    writer = imageio.get_writer(path, fps=fps, codec="libx264", format="FFMPEG", macro_block_size=None, pixelformat="yuv420p")
    n_frames = 0
    latched_success = False
    latched_ttf: float | None = None
    assist_level = "n/a"
    outcome = "INCOMPLETE"
    try:
        # The whole episode (reset through the last step) runs under one inference_mode block. Isaac Lab's
        # env.step() allocates tensors that become "inference tensors"; calling env.reset() again *outside*
        # inference_mode on a later category raises "Inplace update to inference tensor outside InferenceMode"
        # because reset writes into tensors that step already touched under the mode.
        with torch.inference_mode():
            obs, _ = env.reset()

            # Verify against getup_state, don't trust the loop variable -- gc.set_category_live is what
            # actually makes `category` correct; this check catches it if that ever breaks. The overlay below uses `overlay_category` (from getup_state), not `category`.
            overlay_category = category
            if expected_category_idx is not None:
                gc.verify_category(raw_env, expected_category_idx, category)
                getup_state_for_label = getattr(raw_env, "getup_state", None)
                if getup_state_for_label is not None and hasattr(getup_state_for_label, "category"):
                    overlay_category = gc.category_name(int(getup_state_for_label.category[0].item()))

            # The first render(s) after a reset can come back black/not-yet-warmed-up. Prime
            # the camera at the actual start pose first, then render-and-discard a few frames.
            _update_camera()
            for _ in range(max(0, warmup_frames)):
                raw_env.render()

            for step in range(max_steps):
                action = policy(obs)
                obs, _, dones, _ = env.step(action)

                if getup_state is not None:
                    success_now = getattr(getup_state, "success", None)
                    ttf_now = getattr(getup_state, "time_to_stand_s", None)
                    assist_now = getattr(getup_state, "assist_enabled", None)
                    if success_now is not None and bool(success_now[0].item()) and not latched_success:
                        latched_success = True
                        latched_ttf = float(ttf_now[0].item()) if ttf_now is not None else (step + 1) * step_dt
                    if assist_now is not None:
                        assist_level = "on" if bool(assist_now[0].item()) else "off"

                if step % render_every_n_steps == 0:
                    _update_camera()
                    frame = raw_env.render()
                    if frame is not None:
                        frame = cv2.resize(np.asarray(frame)[..., :3], (width, height), interpolation=cv2.INTER_AREA)
                        elapsed = (step + 1) * step_dt
                        lines = [f"run: {run_id}", f"iter: {iteration}", f"category: {overlay_category}", f"assist: {assist_level}"]
                        lines.append(f"t: {elapsed:.1f}s")
                        if latched_success:
                            lines.append(f"standing at {latched_ttf:.1f} s")
                        writer.append_data(gc.draw_overlay(frame, lines))
                        n_frames += 1
                        if n_frames % progress_every_frames == 0:
                            print(f"[getup-eval] {filename}: {n_frames} frames (t={elapsed:.1f}s)", flush=True)

                if bool(dones[0].item()):
                    outcome = "SUCCESS" if latched_success else "TIMEOUT/FAIL"
                    break
            else:
                outcome = "SUCCESS" if latched_success else ("CLIPPED" if has_getup else "END")
    finally:
        writer.close()

    print(f"[getup-eval] {filename}: {n_frames} frames written, outcome={outcome}", flush=True)
    return path


if __name__ == "__main__":
    # Same as evaluate.py: main() only reaches its own os._exit(0) on the success path, and an exception
    # could otherwise leave Kit's background threads/process running indefinitely. Guarantee os._exit()
    # on every path.
    try:
        main()  # calls os._exit(0) itself on success; simulation_app.close() is intentionally never reached.
    except SystemExit:
        raise
    except BaseException:
        import traceback

        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
