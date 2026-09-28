#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""CLI entry point: MuJoCo sim2sim runner for an exported ONNX get-up/walking policy.

Examples
--------
Walking-layout pipeline smoke test (no checkpoint needed, zero-action policy, just checks the model/actuator/obs
pipeline holds a standing pose without exploding)::

    python -m scripts.getup.sim2sim.run_sim2sim --mjcf $MJCF --mode walking --fallback-policy zero \\
        --episode-length-s 4 --physics-hz 1000

Walking-layout validation against the exported smoke-run ONNX, 1 kHz and Isaac's own dt=0.002, with a video::

    python -m scripts.getup.sim2sim.run_sim2sim --mjcf $MJCF --mode walking \\
        --onnx .../exported/policy.onnx --physics-hz 1000 --command 0.6,0,0 \\
        --episode-length-s 6 --video-dir out/videos --metrics-json out/walking_1khz.json
    python -m scripts.getup.sim2sim.run_sim2sim --mjcf $MJCF --mode walking \\
        --onnx .../exported/policy.onnx --physics-dt 0.002 --command 0.6,0,0 \\
        --episode-length-s 6 --metrics-json out/walking_isaac_dt.json

Get-up-layout pipeline test across all fallen categories (zero-action policy just exercises the
obs/action/actuator pipeline; pass --onnx with a trained checkpoint to evaluate a policy)::

    python -m scripts.getup.sim2sim.run_sim2sim --mjcf $MJCF --mode getup --fallback-policy zero \\
        --categories supine,prone,side_left,side_right,sitting,kneeling,mid_fall,standing \\
        --episodes-per-category 2 --episode-length-s 12 --metrics-json out/getup_smoke.json
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

from . import constants as C
from .episode import EpisodeConfig, Sim2SimEpisode
from .init_states import generate_fallen_state, sample_mid_fall_limp_duration, standing_state
from .metrics import RunSummary, compute_episode_metrics
from .mj_model import build_robot_model
from .policy import OnnxPolicy, load_policy
from .video import EpisodeRecorder, TrackingCamera, torso_heading_rad

DROPPED_CATEGORIES = ("supine", "prone", "side_left", "side_right", "sitting", "kneeling")


def _default_mjcf_path() -> str | None:
    env = os.environ.get("ASIMOV_1_MODEL_DIR")
    if env:
        p = Path(env) / "xmls" / "asimov_1.xml"
        if p.is_file():
            return str(p)
    home_default = Path.home() / "isaac_asimov" / "third_party" / "asimov-1" / "sim-model" / "xmls" / "asimov_1.xml"
    return str(home_default) if home_default.is_file() else None


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mjcf", default=_default_mjcf_path(), help="Path to asimov_1.xml (default: $ASIMOV_1_MODEL_DIR or ~/isaac_asimov/third_party/asimov-1/sim-model/xmls/asimov_1.xml).")
    p.add_argument("--mode", choices=["walking", "getup"], required=True)
    p.add_argument("--onnx", default=None, help="Path to an exported policy.onnx. If omitted, uses --fallback-policy.")
    p.add_argument("--fallback-policy", choices=["zero", "random"], default="zero")
    p.add_argument("--physics-hz", type=float, default=None, help="Physics rate; 1000 or 500. Mutually exclusive with --physics-dt.")
    p.add_argument("--physics-dt", type=float, default=None, help="Explicit physics dt (e.g. 0.002 for Isaac's own dt).")
    p.add_argument("--effort-scale", type=float, default=1.0, help="Flat effort-limit scale (weak-motor check: 0.9). Ignored if --effort-scale-stage1 is set.")
    p.add_argument(
        "--effort-scale-stage1", action="store_true",
        help="Get-up Stage-1 curriculum effort scale (hips/knees/shoulders x1.2, elbow/wrist/ankle/waist x1.0), "
             "matching tasks/getup/mdp/curriculums.py::effort_beta_schedule at stage 0/1 -- use for parity with a "
             "Stage-1 checkpoint instead of a flat --effort-scale 1.2.",
    )
    p.add_argument("--armature-source", choices=["isaac", "mjcf"], default="isaac")
    p.add_argument("--no-wrist-stub", action="store_true", help="Disable the wrist-stub collision patch.")
    p.add_argument("--no-upper-arm", action="store_true", help="Disable the upper-arm collision patch.")
    p.add_argument("--disable-obs-noise", action="store_true")
    p.add_argument("--disable-action-delay", action="store_true")
    p.add_argument("--seed", type=int, default=0)

    # walking-only
    p.add_argument("--command", default="0.6,0.0,0.0", help="vx,vy,wz for walking mode's command obs term.")

    # getup-only
    p.add_argument("--categories", default=",".join(DROPPED_CATEGORIES + ("mid_fall", "standing")))
    p.add_argument("--episodes-per-category", type=int, default=1)
    p.add_argument(
        "--beta", type=float, default=None,
        help="Curriculum bound multiplier. Default: read from the ONNX metadata's action_beta when an --onnx "
             "checkpoint with a get-up action contract is given (so the trained bound is used, not assumed); "
             f"else falls back to {C.GETUP_BETA_DEFAULT}.",
    )
    p.add_argument(
        "--ignore-onnx-action-contract", action="store_true",
        help="Use the nominal scale_torque_factor*tau_max/Kp s_j and --beta instead of the ONNX metadata's "
             "action_s_j/action_beta, even if present.",
    )
    p.add_argument("--no-lpf", action="store_true")

    p.add_argument("--episode-length-s", type=float, default=None, help="Default: 6s walking, 12s get-up.")
    p.add_argument("--video-dir", default=None, help="If given, write one MP4 per episode here.")
    p.add_argument("--metrics-json", default=None)
    return p.parse_args(argv)


def _physics_dt(args) -> float:
    if args.physics_dt is not None:
        return args.physics_dt
    if args.physics_hz is not None:
        return 1.0 / args.physics_hz
    return C.MJ_PHYSICS_DT_1KHZ


def _run_one_episode(
    ep: Sim2SimEpisode,
    category: str,
    args,
    rng: np.random.Generator,
    episode_length_s: float,
    recorder: EpisodeRecorder | None,
) -> tuple[list, float]:
    """Returns (history, control_start_t)."""
    if category == "standing":
        st = standing_state()
        ep.reset(st.qpos_joints, st.root_pos, st.root_quat_wxyz, st.qvel_joints, st.root_lin_vel, st.root_ang_vel)
        control_start_t = 0.0
    elif category == "mid_fall":
        st = standing_state()
        limp_s = sample_mid_fall_limp_duration(rng)
        ep.reset(st.qpos_joints, st.root_pos, st.root_quat_wxyz, st.qvel_joints, st.root_lin_vel, st.root_ang_vel,
                  limp_until_s=limp_s)
        control_start_t = limp_s
    else:
        st = generate_fallen_state(category, ep.robot, ep.cfg.physics_dt, rng)
        if not st.accepted:
            print(f"[warn] {category}: fallen-state generator did not converge (residual speed "
                  f"{st.max_residual_speed:.3f} m/s after {st.attempts} attempts); using it anyway.", file=sys.stderr)
        ep.reset(st.qpos_joints, st.root_pos, st.root_quat_wxyz, st.qvel_joints, st.root_lin_vel, st.root_ang_vel)
        control_start_t = 0.0

    command = None
    if args.mode == "walking":
        command = np.array([float(v) for v in args.command.split(",")])

    if recorder is not None and recorder.tracking_camera is not None:
        recorder.tracking_camera.reset()

    n_ticks = int(round(episode_length_s / C.POLICY_DT))
    for _ in range(n_ticks):
        rec = ep.step(command=command)
        if recorder is not None:
            if recorder.tracking_camera is not None:
                recorder.tracking_camera.update(rec.root_pos[:2], torso_heading_rad(rec.root_quat))
            recorder.capture(ep.data)
        if not np.all(np.isfinite(ep.data.qpos)) or not np.all(np.isfinite(ep.data.qvel)):
            print(f"[warn] {category}: non-finite sim state at t={ep.t:.2f}s, stopping episode early.", file=sys.stderr)
            break
    return ep.history, control_start_t


def main(argv=None) -> int:
    args = parse_args(argv)
    if not args.mjcf:
        print("error: --mjcf not given and no default MJCF found (set --mjcf or $ASIMOV_1_MODEL_DIR)", file=sys.stderr)
        return 2

    physics_dt = _physics_dt(args)
    episode_length_s = args.episode_length_s if args.episode_length_s is not None else (6.0 if args.mode == "walking" else 12.0)

    robot = build_robot_model(
        args.mjcf, armature_source=args.armature_source,
        add_wrist_stub=not args.no_wrist_stub, add_upper_arm=not args.no_upper_arm,
    )
    if robot.xml_patch_log:
        print("[info] collision patches added:")
        for line in robot.xml_patch_log:
            print(f"  - {line}")

    effort_scale = C.stage1_effort_scale() if args.effort_scale_stage1 else args.effort_scale

    action_dim = C.NUM_JOINTS
    policy = load_policy(args.onnx, action_dim, kind=args.fallback_policy)

    beta = args.beta if args.beta is not None else C.GETUP_BETA_DEFAULT
    s_j_override = None
    if args.mode == "getup" and isinstance(policy, OnnxPolicy) and not args.ignore_onnx_action_contract:
        meta = policy.metadata
        if meta.has_action_contract():
            s_j_override = meta.action_s_j
            beta = args.beta if args.beta is not None else meta.action_beta
            print(f"[info] get-up action contract read from ONNX metadata (bound_scale already baked into "
                  f"action_s_j): action_beta={meta.action_beta} action_s_j={np.array2string(meta.action_s_j, precision=4)}")
        else:
            print("[warn] --mode getup with an --onnx checkpoint, but no action_s_j/action_beta metadata found "
                  "-- falling back to the nominal scale_torque_factor*tau_max/Kp s_j and --beta "
                  f"({beta}). Pass --ignore-onnx-action-contract to silence this if that's intended.", file=sys.stderr)

    cfg = EpisodeConfig(
        mode=args.mode, physics_dt=physics_dt, effort_scale=effort_scale,
        armature_source=args.armature_source, enable_obs_noise=not args.disable_obs_noise,
        enable_action_delay=not args.disable_action_delay, seed=args.seed,
        beta=beta, use_lpf=not args.no_lpf, s_j_override=s_j_override,
    )
    ep = Sim2SimEpisode(robot, cfg)
    ep.set_policy(policy)
    print(f"[info] mode={args.mode} physics_dt={physics_dt:g}s ({1/physics_dt:.0f} Hz), decimation={ep.decimation}, "
          f"effort_scale={effort_scale}, beta={beta}, "
          f"policy={'ONNX:'+args.onnx if args.onnx else 'fallback:'+args.fallback_policy}")

    rng = np.random.default_rng(args.seed)
    summary = RunSummary(mode=args.mode, physics_dt=physics_dt, effort_scale=effort_scale)

    categories = args.categories.split(",") if args.mode == "getup" else ["standing"]
    n_eps = args.episodes_per_category if args.mode == "getup" else 1

    if args.video_dir:
        Path(args.video_dir).mkdir(parents=True, exist_ok=True)

    for category in categories:
        for i in range(n_eps):
            recorder = None
            if args.video_dir:
                recorder = EpisodeRecorder(robot.model, width=1280, height=720, camera=TrackingCamera())
            history, control_start_t = _run_one_episode(ep, category, args, rng, episode_length_s, recorder)
            m = compute_episode_metrics(history, category, control_start_t)
            summary.add(m)
            print(f"[result] {category} ep{i}: success_1s={m.success_1s} success_g1={m.success_g1} "
                  f"time_to_stand_s={m.time_to_stand_s} peak_height={m.peak_pelvis_height_m:.3f} "
                  f"final_height={m.final_pelvis_height_m:.3f} torque_sat={m.torque_saturation_frac:.3f} "
                  f"jitter_rms={m.action_jitter_rms:.4f} peak_impact_N={m.peak_non_foot_impact_force_n:.1f}")
            if recorder is not None:
                out_path = str(Path(args.video_dir) / f"{args.mode}_{category}_ep{i}.mp4")
                recorder.save(out_path, fps=1.0 / C.POLICY_DT)
                recorder.close()
                print(f"[info] wrote {out_path}")

    if args.metrics_json:
        Path(args.metrics_json).parent.mkdir(parents=True, exist_ok=True)
        summary.to_json(args.metrics_json)
        print(f"[info] wrote {args.metrics_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
