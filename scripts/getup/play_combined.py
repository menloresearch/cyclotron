# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Combined get-up <-> walking handoff demo.

Runs ``Asimov1-GetUp-Handoff-Play-v0`` (``tasks/getup_handoff``) end to end: fall start -> get-up ->
handoff (hysteresis) -> walk N meters -> optional push -> lying detector -> back to get-up. Loads
two independent checkpoints (get-up, walking) as exported TorchScript policies and feeds both
policies' raw actions into the env's single combined action term every step (see
``tasks/getup_handoff/mdp.py``'s module docstring for why both run every step). The switching
decision itself is pure-tensor logic in ``scripts/getup/handoff_logic.py`` (no Isaac Lab import,
independently unit-tested there). The handoff's pose bound is per joint group:
legs+waist <= 0.15 rad, shoulders+elbows <= 0.30 rad, wrist yaw <= 0.50 rad -- pass
``--pose_diag`` for a reusable per-group diagnostic (periodic prints plus a final per-env summary)
useful for checking a new checkpoint against these bounds without changing them.

Checkpoints:
    --getup_checkpoint   path to an exported get-up policy (``<run>/exported/policy.pt``, produced
                          by ``scripts/rsl_rl/play.py`` or ``export_onnx.py``'s sibling .pt). If
                          omitted, a dummy all-zero policy is used instead, which is enough to test
                          the walking half and the switching logic without a get-up checkpoint. A zero
                          action makes the get-up path's target track the *measured* joint position
                          every control cycle -- closer to a passively-damped hold than an active
                          balance controller (the position-error term is ~0 by construction, so
                          torque comes almost entirely from damping) -- so it will not actually
                          stand up, and in practice does not reliably hold `is_standing`'s
                          continuous-1s bar even starting from ``--start_category standing`` (see
                          docs/getup/HANDOFF.md §5 for a run that confirms this: 0 handoffs over 20s).
                          Use ``--check_only`` to verify the env/action wiring without a checkpoint;
                          exercising the actual handoff *decision* end to end needs at least a
                          weakly-competent get-up checkpoint (even an early, undertrained one should
                          do), not the all-zero fallback.
    --walk_checkpoint     path to a raw rsl_rl checkpoint (``model_N.pt``) or its ``exported/policy.pt``.
                          If omitted, auto-discovers the best available walking policy: first any
                          released baseline under the repo (see ``_find_released_walk_checkpoint``),
                          else the newest run under ``~/isaac_asimov/logs/rsl_rl/asimov_velocity_amp/``.

Usage:
    python scripts/getup/play_combined.py \\
        --walk_checkpoint ~/isaac_asimov/logs/rsl_rl/asimov_velocity_amp/<run>/model_99.pt \\
        --num_envs 4 --start_category mid_fall --push --video --headless --enable_cameras

    # Pipeline-only check (no GPU-heavy rollout, no checkpoints needed):
    python scripts/getup/play_combined.py --check_only --headless
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

from isaaclab.app import AppLauncher

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as gc  # isort: skip  (isaaclab-free; see its own docstring)
from handoff_logic import (  # isort: skip  (isaaclab-free)
    GROUP_LEGS_WAIST,
    GROUP_SHOULDERS_ELBOWS,
    GROUP_WRIST_YAW,
    HandoffConfig,
    HandoffStateMachine,
)

# Per-joint-group deviation bound for the handoff gate (instead of a single 0.15 rad bound over
# all 23 joints -- see handoff_logic.py's module docstring for why). Regex patterns, not hardcoded indices, so this survives a joint reordering; `main()` builds
# a joint-id tensor per group after the asset exists and asserts the union covers every joint
# exactly once.
GROUP_JOINT_PATTERNS: dict[str, list[str]] = {
    GROUP_LEGS_WAIST: [
        r".*_hip_pitch_joint", r".*_hip_roll_joint", r".*_hip_yaw_joint",
        r".*_knee_joint", r".*_ankle_pitch_joint", r".*_ankle_roll_joint",
        r"waist_yaw_joint",
    ],
    GROUP_SHOULDERS_ELBOWS: [
        r".*_shoulder_pitch_joint", r".*_shoulder_roll_joint", r".*_shoulder_yaw_joint", r".*_elbow_joint",
    ],
    GROUP_WRIST_YAW: [r".*_wrist_yaw_joint"],
}

parser = argparse.ArgumentParser(description="Combined get-up <-> walking handoff demo.")
parser.add_argument("--task", type=str, default="Asimov1-GetUp-Handoff-Play-v0")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--getup_checkpoint", type=str, default=None)
parser.add_argument("--walk_checkpoint", type=str, default=None)
parser.add_argument(
    "--start_category", type=str, default="mid_fall",
    help="One get-up start category (supine, prone, side_left, side_right, sitting, kneeling, mid_fall, standing), 'mixed' for the env's default probability mix, or 'all' for "
    "round-robin (env i gets category i, one env per category -- use --num_envs 8 for exactly one each).",
)
parser.add_argument("--walk_distance", type=float, default=5.0, help="Meters (net planar displacement from the handoff point) to walk before the optional push.")
parser.add_argument("--walk_speed", type=float, default=0.6, help="Commanded forward velocity [m/s], body frame, once walking. Note: the walker's *actual* speed tracks well below this "
                     "(observed ~0.25 m/s at a 0.5 m/s command) -- see --max_seconds' note on the 45s walk-window budget this implies for --walk_distance 5.0.")
parser.add_argument("--heading_hold_kp", type=float, default=0.5, help="P-gain [1/s] of the scripted yaw-rate controller that holds heading at the robot's own yaw AT HANDOFF while walking "
                     "(so gait asymmetry does not drift the heading). Matches the walking env's own heading_control_stiffness default so the closed-loop behavior is familiar, but this is entirely scripted here -- "
                     "the twist command's heading_command flag is left OFF (handoff_env_cfg.py) and this script is the command's only yaw-rate authority during WALK mode.")
parser.add_argument("--heading_hold_max_rad_s", type=float, default=0.8, help="Clamp for the scripted heading-hold yaw-rate command (matches the walking task's own trained ang_vel_z range).")
parser.add_argument("--push", action="store_true", default=False, help="Apply a lateral velocity kick once --walk_distance is reached, to force a fall and re-trigger get-up.")
parser.add_argument("--push_speed", type=float, default=2.0, help="Lateral root velocity kick [m/s] applied by --push.")
parser.add_argument("--max_seconds", type=float, default=75.0, help="Hard wall-clock cutoff (sim time) for the whole demo (get-up + walk), independent of whether the loop above completed. "
                     "Budgets >= 45s of WALKING after handoff (the walker's actual speed is ~0.25 m/s, so 5m needs ~20s at that pace; 45s gives ~2x margin for an "
                     "imperfect straight line) *plus* whatever get-up takes before the first handoff (observed up to ~10-20s for the harder categories, e.g. mid_fall/prone) -- 75s covers both "
                     "with room. This is a single GLOBAL cutoff from t=0, not a per-env clock that restarts at each env's own handoff; a run's timeline.json records each env's own "
                     "HANDOFF_TO_WALK/WALK_DISTANCE_REACHED timestamps, so 'did env i reach 5m within 45s of ITS OWN handoff' is computed from those two timestamps after the run, not enforced live.")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--output_dir", type=str, default=None, help="Defaults to logs/getup_handoff/play_<timestamp>/ under the cwd.")
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--fps", type=int, default=30)
parser.add_argument("--width", type=int, default=1280)
parser.add_argument("--height", type=int, default=720)
# Camera defaults match record_videos.py exactly (full-body 3/4 view), so a clip from this
# script and one from record_videos.py look consistent. See `_common.camera_world_pose`.
parser.add_argument("--camera_distance_m", type=float, default=3.2)
parser.add_argument("--camera_eye_height_m", type=float, default=1.0)
parser.add_argument("--camera_lookat_height_m", type=float, default=0.5)
parser.add_argument("--camera_azimuth_deg", type=float, default=-45.0)
parser.add_argument("--camera_smoothing_alpha", type=float, default=0.15)
parser.add_argument("--check_only", action="store_true", default=False, help="Build the env, run a few zero-action steps, print obs/action shapes, and exit. No policies loaded.")
parser.add_argument(
    "--walk_sanity_check",
    action="store_true",
    default=False,
    help="Run the walking policy ALONE on the walking task (Asimov1-Velocity-Play-v0, default standing "
    "pose), commanded at --walk_sanity_speed for --walk_sanity_seconds, and report whether it falls. "
    "Exits before touching the get-up/handoff env at all. Run this first for any new "
    "walking checkpoint -- if it fails, the walking contract (obs layout / action scale / gains) is "
    "wrong and a full handoff run would just be measuring that, not the get-up/handoff pipeline.",
)
parser.add_argument("--walk_sanity_seconds", type=float, default=10.0)
parser.add_argument("--walk_sanity_speed", type=float, default=0.5)
parser.add_argument(
    "--pose_diag",
    action="store_true",
    default=False,
    help="Reusable diagnostic: print per-joint-group deviation stats "
    "periodically and a full per-env summary at the end (max stand-hold time, getup_state.success, "
    "final per-group deviation vs. each group's handoff bound, final angular velocity). Meant to be "
    "re-run against later checkpoints of a training run to check whether the handoff "
    "gate starts clearing without changing its bounds.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- heavy imports (after AppLauncher; see every other scripts/rsl_rl and scripts/getup script) ---

import numpy as np
import torch

import gymnasium as gym
from isaaclab.envs import ManagerBasedRLEnv

import isaac_asimov.tasks  # noqa: F401  (registers Asimov1-GetUp-Handoff-Play-v0, lazily)
from isaac_asimov.tasks.getup_handoff.handoff_env_cfg import Asimov1GetUpHandoffPlayEnvCfg
from isaac_asimov.tasks.getup_handoff.mdp import HandoffJointPositionAction

# ---------------------------------------------------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------------------------------------------------


def _exported_pt_for(checkpoint_or_dir: str) -> str:
    """Resolve a raw rsl_rl checkpoint or run dir to its sibling ``exported/policy.pt`` (play.py's export path)."""
    path = os.path.expanduser(checkpoint_or_dir)
    if path.endswith(".pt") and os.path.basename(os.path.dirname(path)) == "exported":
        return path
    if os.path.isdir(path):
        run_dir = path
    else:
        run_dir = os.path.dirname(path)
    exported = os.path.join(run_dir, "exported", "policy.pt")
    if not os.path.isfile(exported):
        raise FileNotFoundError(
            f"No exported policy at {exported!r}. Run `scripts/rsl_rl/play.py --task <task> "
            f"--checkpoint {checkpoint_or_dir} --headless` once first (it always exports "
            "exported/policy.pt + policy.onnx alongside the checkpoint it loads)."
        )
    return exported


def _find_released_walk_checkpoint() -> str | None:
    """Look for a released walking baseline anywhere in the repo, preferring it over the smoke-run checkpoint.

    Searches `releases/` and `checkpoints/` for a velocity/walk `.pt`, so a baseline dropped there is picked
    up without a script change. Returns None if nothing is found.
    """
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    candidates = []
    for pattern in ("releases/**/*velocity*.pt", "releases/**/*walk*.pt", "checkpoints/**/*velocity*.pt"):
        candidates += glob.glob(os.path.join(repo_root, pattern), recursive=True)
    candidates = [c for c in candidates if os.path.isfile(c)]
    if not candidates:
        return None
    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates[0]


def _find_smoke_walk_checkpoint() -> str:
    """Newest run's highest-iteration checkpoint under ``~/isaac_asimov/logs/rsl_rl/asimov_velocity_amp/``."""
    base = os.path.expanduser("~/isaac_asimov/logs/rsl_rl/asimov_velocity_amp")
    runs = sorted(glob.glob(os.path.join(base, "*")), key=os.path.getmtime, reverse=True)
    for run_dir in runs:
        ckpts = glob.glob(os.path.join(run_dir, "model_*.pt"))
        if ckpts:

            def _iter(p):
                digits = "".join(ch for ch in os.path.splitext(os.path.basename(p))[0] if ch.isdigit())
                return int(digits) if digits else -1

            ckpts.sort(key=_iter, reverse=True)
            return ckpts[0]
    raise FileNotFoundError(
        f"No checkpoint found under {base}. Pass --walk_checkpoint explicitly, or run "
        "scripts/rsl_rl/train.py --task Asimov1-Velocity-AMP-v0 first."
    )


def resolve_walk_checkpoint(explicit: str | None) -> str:
    if explicit:
        return explicit
    released = _find_released_walk_checkpoint()
    if released:
        print(f"[getup-handoff] Using released walking baseline: {released}")
        return released
    smoke = _find_smoke_walk_checkpoint()
    print(f"[getup-handoff] No released walking baseline found; using the smoke-run checkpoint: {smoke}")
    return smoke


def _quat_yaw_batch(quat_wxyz: torch.Tensor) -> torch.Tensor:
    """Vectorized yaw (rad) from a batch of (w, x, y, z) quaternions -- same formula as
    ``_common.quat_yaw`` (kept in sync there, scalar version), just batched over ``[N, 4]`` instead
    of looped per-env, since the heading-hold controller below needs it every step for every env."""
    w, x, y, z = quat_wxyz.unbind(-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class DummyGetUpPolicy(torch.nn.Module):
    """All-zero action policy: with the get-up path's formula, this holds the current joint
    positions (LPF -> 0 -> delta -> 0). Used when no get-up checkpoint exists yet."""

    def __init__(self, action_dim: int, device):
        super().__init__()
        self._zeros = torch.zeros(1, action_dim, device=device)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        n = obs.shape[0]
        return self._zeros.expand(n, -1)


def load_policy(path: str, device) -> torch.nn.Module:
    policy = torch.jit.load(path, map_location=device)
    policy.eval()
    return policy


def _curriculum_state_path_for(checkpoint_path: str) -> str | None:
    """Same lookup as ``export_onnx.py``'s ``_curriculum_state_path`` (duplicated, not imported --
    ``export_onnx.py`` parses argv at module scope, so importing it here would collide with this
    script's own ``args_cli``). Exact-iteration match first, else the closest-mtime
    ``curriculum_state_*.json`` in the same run dir (needed in practice: a `model_3999.pt` can have
    no `curriculum_state_3999.json`, only `..._3750.json`/`..._4000.json`, an off-by-one between
    the checkpoint saver and the curriculum tracker -- `model_3999.pt` and `curriculum_state_4000.json`
    share the exact same mtime on disk)."""
    run_dir = os.path.dirname(checkpoint_path)
    stem = os.path.splitext(os.path.basename(checkpoint_path))[0]
    digits = "".join(ch for ch in stem if ch.isdigit())
    if digits:
        exact = os.path.join(run_dir, f"curriculum_state_{digits}.json")
        if os.path.isfile(exact):
            return exact
    candidates = [c for c in glob.glob(os.path.join(run_dir, "curriculum_state_*.json")) if os.path.basename(c) != "curriculum_state.json"]
    if not candidates:
        return None
    try:
        ckpt_mtime = os.path.getmtime(checkpoint_path)
    except OSError:
        return sorted(candidates)[-1]
    return min(candidates, key=lambda c: abs(os.path.getmtime(c) - ckpt_mtime))


def apply_trained_action_contract(action_term, checkpoint_path: str) -> dict | None:
    """Apply a get-up checkpoint's REAL trained ``beta``/``bound_scale`` to the live combined
    action term, via its ``set_bound_scale``/``beta`` API -- without this, a Stage-B checkpoint
    (e.g. beta 0.7, bound_scale 0.9) would run the demo with the handoff env's nominal
    defaults (beta 1.0, bound_scale 1.0), commanding a target delta up to 1.6x too large (the same
    mismatch ``export_onnx.py`` guards against for the ONNX metadata path, here for the *live sim
    demo* path). Best-effort, not fail-loud like ``export_onnx.py``'s exporter: a demo missing a
    contract should still run (with a loud warning), not abort. Returns the applied contract dict,
    or ``None`` if none was found."""
    checkpoint_path = os.path.expanduser(checkpoint_path)
    path = _curriculum_state_path_for(checkpoint_path)
    contract = None
    if path:
        with open(path) as f:
            contract = json.load(f).get("action_contract")
    if contract is None:
        print(f"[getup-handoff] WARNING: no action_contract found for {checkpoint_path!r} "
              f"(checked {path!r}). Running with the handoff env's NOMINAL beta/bound_scale "
              f"(beta={action_term.beta}, bound_scale={action_term.bound_scale.tolist()[:3]}...) -- "
              "if this checkpoint is actually Stage B, the demo will run with the wrong action "
              "contract. Verify this is a nominal/Stage-A checkpoint before trusting these results.")
        return None
    action_term.beta = float(contract["beta"])
    action_term.set_bound_scale(torch.tensor(contract["bound_scale"], device=action_term.device))
    print(f"[getup-handoff] applied {path}'s trained action_contract to the live get-up path -- "
          f"beta={action_term.beta}, bound_scale[:3]={contract['bound_scale'][:3]}... "
          f"(joint_scale_live[:3]={action_term.joint_scale_live.tolist()[:3]})")
    return contract


def load_onnx_mlp_policy(path: str, device, num_verify_samples: int = 8, atol: float = 1e-5) -> torch.nn.Module:
    """Load a plain "Gemm + Elu only" MLP policy straight from an .onnx file into a torch
    ``nn.Sequential`` (for walking policies shipped as bare ONNX with no metadata and no
    normalizer -- there is no sibling ``exported/policy.pt`` to fall back on for these).

    Reads the graph generically (any number of Gemm/Elu nodes, not hardcoded to 4 layers) and
    fails loudly on any other op type, rather than silently ignoring it. Gemm nodes with ``transB=1``
    (weight stored ``[out_features, in_features]``, i.e. exactly ``torch.nn.Linear``'s own
    convention) load into ``nn.Linear`` with no transpose; ``transB=0`` is transposed.

    Verifies the reconstructed torch module against ``onnx.reference.ReferenceEvaluator`` (pure
    Python, ships with the already-installed ``onnx`` package -- no need to leave this venv, unlike
    the get-up ONNX export path elsewhere in this task, which specifically needs the *real*
    firmware runtime) on random observations, to ``atol`` (default 1e-5). Raises if it doesn't
    match; never returns an unverified reconstruction.
    """
    import onnx

    model = onnx.load(path)
    initializers = {t.name: onnx.numpy_helper.to_array(t) for t in model.graph.initializer}
    input_name = model.graph.input[0].name
    output_name = model.graph.output[0].name
    obs_dim = model.graph.input[0].type.tensor_type.shape.dim[-1].dim_value

    layers: list[torch.nn.Module] = []
    for node in model.graph.node:
        if node.op_type == "Gemm":
            attrs = {a.name: a.i for a in node.attribute if a.name in ("transA", "transB")}
            if attrs.get("transA", 0) != 0:
                raise NotImplementedError(f"{path}: Gemm node {node.name!r} has transA=1 (unsupported).")
            weight_name, bias_name = node.input[1], node.input[2]
            weight = initializers[weight_name]
            bias = initializers[bias_name]
            if attrs.get("transB", 0) != 1:
                # transB=0 means B is stored [in, out] (NOT torch.nn.Linear's convention) -- transpose it.
                weight = weight.T
            out_features, in_features = weight.shape
            linear = torch.nn.Linear(in_features, out_features)
            with torch.no_grad():
                linear.weight.copy_(torch.from_numpy(weight.copy()))
                linear.bias.copy_(torch.from_numpy(bias.copy()))
            layers.append(linear)
        elif node.op_type == "Elu":
            alpha = next((a.f for a in node.attribute if a.name == "alpha"), 1.0)
            layers.append(torch.nn.ELU(alpha=alpha))
        else:
            raise NotImplementedError(
                f"{path}: unsupported op {node.op_type!r} (node {node.name!r}) -- this loader only "
                "handles the plain 'Gemm + Elu only' MLP contract (bare-ONNX walking policies). "
                "If this file has a normalizer or a different architecture, it needs a different "
                "loader, not this one."
            )
    policy = torch.nn.Sequential(*layers).to(device).eval()

    # Verification: this reconstruction must reproduce the ONNX graph's own output, or it must not
    # be trusted at all -- see docstring.
    from onnx.reference import ReferenceEvaluator

    rng = np.random.default_rng(0)
    samples = rng.normal(size=(num_verify_samples, obs_dim)).astype(np.float32)
    sess = ReferenceEvaluator(model)
    onnx_out = np.concatenate([sess.run([output_name], {input_name: samples[i : i + 1]})[0] for i in range(num_verify_samples)], axis=0)
    with torch.inference_mode():
        torch_out = policy(torch.from_numpy(samples).to(device)).cpu().numpy()
    max_diff = float(np.max(np.abs(onnx_out - torch_out)))
    print(f"[getup-handoff] load_onnx_mlp_policy({os.path.basename(path)}): reconstructed {len(layers)} layers "
          f"({obs_dim} -> ... -> {layers[-1].out_features if hasattr(layers[-1], 'out_features') else '?'}), "
          f"verified against onnx.reference on {num_verify_samples} samples: max diff = {max_diff:.3e}")
    assert max_diff <= atol, (
        f"{path}: reconstructed torch module diverges from the ONNX graph by {max_diff:.3e} > atol={atol:.3e} "
        "-- do not trust this reconstruction; the Gemm/Elu parsing above has a bug."
    )
    return policy


def run_walk_sanity_check(device) -> None:
    """The walker ALONE, from the default standing pose, commanded forward for a fixed duration --
    no get-up, no handoff env at all. If it does not walk, the walking contract is wrong -- run
    this before trusting any full handoff result for a new walking checkpoint."""
    walk_checkpoint = resolve_walk_checkpoint(args_cli.walk_checkpoint)
    if walk_checkpoint.endswith(".onnx"):
        print(f"[getup-handoff] [sanity] Loading walking policy from ONNX: {walk_checkpoint}")
        walk_policy = load_onnx_mlp_policy(os.path.expanduser(walk_checkpoint), device)
    else:
        walk_path = _exported_pt_for(walk_checkpoint)
        print(f"[getup-handoff] [sanity] Loading walking policy from {walk_path}")
        walk_policy = load_policy(walk_path, device)

    from isaac_asimov.tasks.locomotion.velocity_env_cfg import Asimov1VelocityEnvCfg_PLAY

    env_cfg = Asimov1VelocityEnvCfg_PLAY()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env = gym.make("Asimov1-Velocity-Play-v0", cfg=env_cfg)
    raw_env: ManagerBasedRLEnv = env.unwrapped
    obs_dict, _ = raw_env.reset()
    command_term = raw_env.command_manager.get_term("twist")
    command_term.command[:, 0] = args_cli.walk_sanity_speed
    command_term.command[:, 1:] = 0.0
    asset = raw_env.scene["robot"]
    step_dt = raw_env.step_dt
    n = raw_env.num_envs
    max_steps = max(1, int(round(args_cli.walk_sanity_seconds / step_dt)))

    fell_at: list[float | None] = [None] * n
    max_height = asset.data.root_pos_w[:, 2].clone()
    start_x = asset.data.root_pos_w[:, 0].clone()
    for step in range(max_steps):
        with torch.inference_mode():
            action = walk_policy(obs_dict["policy"])
        obs_dict, reward, terminated, truncated, extras = raw_env.step(action)
        max_height = torch.maximum(max_height, asset.data.root_pos_w[:, 2])
        for i in terminated.nonzero(as_tuple=False).flatten().tolist():
            if fell_at[i] is None:
                fell_at[i] = step * step_dt
    walked = (asset.data.root_pos_w[:, 0] - start_x).tolist()
    env.close()

    print(f"[getup-handoff] [sanity] {walk_checkpoint}: commanded {args_cli.walk_sanity_speed} m/s for "
          f"{args_cli.walk_sanity_seconds}s, {n} envs.")
    print(f"[getup-handoff] [sanity]   fell_at (s, None=never) = {fell_at}")
    print(f"[getup-handoff] [sanity]   distance walked (m)     = {[round(w, 3) for w in walked]}")
    n_ok = sum(1 for f in fell_at if f is None)
    verdict = "PASS" if n_ok == n else ("PARTIAL" if n_ok > 0 else "FAIL")
    print(f"[getup-handoff] [sanity] {verdict}: {n_ok}/{n} envs walked the full {args_cli.walk_sanity_seconds}s without falling.")


# ---------------------------------------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------------------------------------


def main():
    if args_cli.walk_sanity_check:
        device = args_cli.device if args_cli.device else ("cuda:0" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(args_cli.seed)
        run_walk_sanity_check(device)
        return
    device = args_cli.device if args_cli.device else ("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args_cli.seed)

    env_cfg = Asimov1GetUpHandoffPlayEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed

    # Camera: driven manually every rendered step via `raw_env.sim.set_camera_view` (below), the
    # same pattern record_videos.py uses (`_common.camera_world_pose` +
    # `_common.SmoothedTracker`, absolute world-frame eye/lookat) instead of Isaac Lab's built-in
    # asset-root viewer tracking, which had teardown bugs -- no static `env_cfg.viewer` needed.
    camera_tracker = gc.SmoothedTracker(alpha=args_cli.camera_smoothing_alpha)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    raw_env: ManagerBasedRLEnv = env.unwrapped

    # Category selection: setting `env_cfg.events.reset_fallen.params[...]` *before* `gym.make`
    # doesn't reliably take effect (Isaac Lab managers deep-copy their cfg; see `_common.set_category_live`).
    # Set it on the *constructed* event manager's own term cfg instead, matching
    # evaluate.py/record_videos.py and the curriculum code (`getup/mdp/curriculums.category_reweighting`, which
    # reads/writes `env.event_manager.get_term_cfg("reset_fallen").params["category_probs"]` the
    # same way) -- then reset once so the very first episode already uses it.
    if args_cli.start_category != "mixed":
        try:
            reset_term_cfg = raw_env.event_manager.get_term_cfg("reset_fallen")
            if args_cli.start_category == "all":
                reset_term_cfg.params["assignment"] = "round_robin"
                print("[getup-handoff] --start_category all: round-robin, env i gets category i "
                      f"(use --num_envs {len(gc.CATEGORY_KEYS)} for exactly one env per category).")
            else:
                reset_term_cfg.params["category_probs"] = gc.one_hot_category_probs(args_cli.start_category)
        except (KeyError, ValueError) as exc:
            print(f"[getup-handoff] NOTE: could not set --start_category ({exc!r}); using the env's default category mix.")
    obs_dict, _ = raw_env.reset()

    action_term: HandoffJointPositionAction = raw_env.action_manager.get_term("joint_pos")
    command_term = raw_env.command_manager.get_term("twist")
    asset = raw_env.scene["robot"]
    step_dt = raw_env.step_dt
    n = raw_env.num_envs

    # Per-group joint ids, resolved once. Assert the three groups partition all joints exactly
    # (no joint left out, none double-counted) -- a silent gap here would make the handoff gate
    # quietly ignore whatever joint fell through the cracks, in either direction (too lax or, if a
    # joint is double-covered by two group regexes, too strict on one group unnecessarily).
    group_joint_ids: dict[str, torch.Tensor] = {}
    seen_ids: set[int] = set()
    for group, patterns in GROUP_JOINT_PATTERNS.items():
        ids, _ = asset.find_joints(patterns, preserve_order=False)
        overlap = seen_ids.intersection(ids)
        assert not overlap, f"joint id(s) {overlap} matched by more than one joint-group pattern"
        seen_ids.update(ids)
        group_joint_ids[group] = torch.as_tensor(ids, device=raw_env.device, dtype=torch.long)
    assert len(seen_ids) == asset.num_joints, (
        f"joint-group patterns cover {len(seen_ids)}/{asset.num_joints} joints -- "
        "some joint matched none of GROUP_JOINT_PATTERNS and would be silently excluded from the handoff gate."
    )

    print(f"[getup-handoff] task={args_cli.task} num_envs={n} step_dt={step_dt:.4f}s device={device}")
    print(f"[getup-handoff] obs groups: {sorted(obs_dict.keys())}")
    for k, v in obs_dict.items():
        print(f"[getup-handoff]   obs[{k!r}].shape = {tuple(v.shape)}")
    print(f"[getup-handoff] action_dim (combined term) = {action_term.action_dim} ({action_term.action_dim // 2} get-up + {action_term.action_dim // 2} walk)")

    if args_cli.check_only:
        j = action_term.action_dim // 2
        for step in range(5):
            zero_action = torch.zeros(n, 2 * j, device=raw_env.device)
            obs_dict, reward, terminated, truncated, extras = raw_env.step(zero_action)
        print(f"[getup-handoff] check_only: {step + 1} zero-action steps ran without error. reward[0]={float(reward[0]):.4f}")
        print("[getup-handoff] check_only: OK")
        env.close()
        simulation_app.close()
        return

    getup_checkpoint = args_cli.getup_checkpoint
    if getup_checkpoint:
        getup_checkpoint_expanded = os.path.expanduser(getup_checkpoint)
        getup_path = _exported_pt_for(getup_checkpoint)
        print(f"[getup-handoff] Loading get-up policy from {getup_path}")
        getup_policy = load_policy(getup_path, device)
        # Use the checkpoint's REAL trained beta/bound_scale (e.g. Stage B: 0.7/0.9),
        # not the handoff env's nominal defaults -- see apply_trained_action_contract's docstring.
        apply_trained_action_contract(action_term, getup_checkpoint_expanded)
    else:
        print("[getup-handoff] No --getup_checkpoint given: using an all-zero dummy get-up policy "
              "(holds the current pose; see module docstring). Pass --start_category standing to "
              "still exercise the walk/switch/push/fall loop meaningfully.")
        getup_policy = DummyGetUpPolicy(action_term.action_dim // 2, device)

    walk_checkpoint = resolve_walk_checkpoint(args_cli.walk_checkpoint)
    if walk_checkpoint.endswith(".onnx"):
        # Bare-ONNX walking policy: no sibling exported/policy.pt.
        print(f"[getup-handoff] Loading walking policy from ONNX: {walk_checkpoint}")
        walk_policy = load_onnx_mlp_policy(os.path.expanduser(walk_checkpoint), device)
    else:
        walk_path = _exported_pt_for(walk_checkpoint)
        print(f"[getup-handoff] Loading walking policy from {walk_path}")
        walk_policy = load_policy(walk_path, device)

    hcfg = HandoffConfig(walk_distance_m=args_cli.walk_distance)
    sm = HandoffStateMachine(n, raw_env.device, hcfg)
    command_term.command[:] = 0.0  # zero the twist command; get-up mode drives first.

    output_dir = args_cli.output_dir or os.path.join("logs", "getup_handoff", f"play_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(output_dir, exist_ok=True)
    timeline: list[dict] = []
    frames: list[np.ndarray] = []
    max_steps = max(1, int(round(args_cli.max_seconds / step_dt)))

    def log_event(step: int, env_id: int, kind: str, **extra):
        entry = {"step": step, "t_s": round(step * step_dt, 3), "env": env_id, "event": kind}
        entry.update(extra)
        timeline.append(entry)
        print(f"[getup-handoff] t={entry['t_s']:>7.2f}s env{env_id} {kind} " + " ".join(f"{k}={v}" for k, v in extra.items()))

    was_getup = action_term.mode.clone()
    push_applied = torch.zeros(n, dtype=torch.bool, device=raw_env.device)
    max_stand_hold_s = torch.zeros(n, device=raw_env.device)  # diagnostic: closest any env got to the 1.0s bar
    # While walking, hold heading at the robot's yaw at handoff. `heading_command` is already OFF
    # (handoff_env_cfg.py __post_init__), so `UniformVelocityCommand._update_command()` never touches
    # index 2 of the twist command itself -- but a pure zero-yaw-rate command still lets the walking
    # gait's own left/right asymmetry drift the heading over 5m+ of walking. This scripted P-controller
    # replaces "always command 0" with "hold whatever heading the robot had the instant it handed off",
    # which walks a much straighter line in practice than a bare zero.
    heading_anchor = _quat_yaw_batch(asset.data.root_quat_w)

    for step in range(max_steps):
        with torch.inference_mode():
            getup_raw = getup_policy(obs_dict["policy"])
            walk_raw = walk_policy(obs_dict["walk"])
        action = torch.cat([getup_raw, walk_raw], dim=-1)

        getup_state = raw_env.getup_state
        joint_dev_all = (asset.data.joint_pos - asset.data.default_joint_pos).abs()  # [N, J]
        joint_dev_by_group = {g: joint_dev_all[:, ids].amax(dim=-1) for g, ids in group_joint_ids.items()}
        ang_vel_norm = asset.data.root_ang_vel_b.norm(dim=-1)
        gravity_z = asset.data.projected_gravity_b[:, 2]
        base_x = asset.data.root_pos_w[:, 0]
        base_y = asset.data.root_pos_w[:, 1]
        max_stand_hold_s = torch.maximum(max_stand_hold_s, getup_state.stand_timer_s)

        out = sm.step(
            dt=step_dt,
            stand_hold_s=getup_state.stand_timer_s,
            joint_dev_by_group=joint_dev_by_group,
            ang_vel_norm=ang_vel_norm,
            gravity_z=gravity_z,
            base_x=base_x,
            base_y=base_y,
        )

        if args_cli.pose_diag and step % max(1, int(round(2.0 / step_dt))) == 0:
            print(
                f"[getup-handoff] t={step * step_dt:6.2f}s pose_diag "
                + " ".join(
                    f"{g}[min/mean/max]={float(d.min()):.3f}/{float(d.mean()):.3f}/{float(d.max()):.3f}"
                    for g, d in joint_dev_by_group.items()
                )
                + f" ang_vel[max]={float(ang_vel_norm.max()):.3f} stand_hold[max]={float(getup_state.stand_timer_s.max()):.2f}"
            )
        action_term.set_mode(out["mode_getup"])

        newly_walk = was_getup & ~out["mode_getup"]
        newly_getup = ~was_getup & out["mode_getup"]
        # Anchor the scripted heading-hold at the yaw the robot actually has the instant it hands
        # off, not e.g. 0 (world +x) or whatever
        # yaw it had at spawn -- a fallen robot can settle facing any direction.
        current_yaw = _quat_yaw_batch(asset.data.root_quat_w)
        heading_anchor = torch.where(newly_walk, current_yaw, heading_anchor)

        # command: forward speed while walking, zero while in get-up mode.
        # Yaw-rate (index 2): a P-controller holding `heading_anchor` while walking, exactly 0 while
        # in get-up mode (no reason to fight the get-up recovery's own rotation). heading_command is
        # OFF on the twist command term (handoff_env_cfg.py), so this is the ONLY yaw-rate authority.
        yaw_error = torch.atan2(torch.sin(heading_anchor - current_yaw), torch.cos(heading_anchor - current_yaw))
        heading_yaw_rate = torch.clamp(
            args_cli.heading_hold_kp * yaw_error, -args_cli.heading_hold_max_rad_s, args_cli.heading_hold_max_rad_s
        )
        command_term.command[:, 0] = torch.where(
            out["mode_getup"], torch.zeros(n, device=raw_env.device), torch.full((n,), args_cli.walk_speed, device=raw_env.device)
        )
        command_term.command[:, 1] = 0.0
        command_term.command[:, 2] = torch.where(out["mode_getup"], torch.zeros(n, device=raw_env.device), heading_yaw_rate)

        for i in newly_walk.nonzero(as_tuple=False).flatten().tolist():
            log_event(
                step, i, "HANDOFF_TO_WALK",
                **{f"joint_dev_{g}": round(float(d[i]), 4) for g, d in joint_dev_by_group.items()},
                ang_vel=round(float(ang_vel_norm[i]), 4),
                heading_anchor_rad=round(float(heading_anchor[i]), 4),
            )
        for i in newly_getup.nonzero(as_tuple=False).flatten().tolist():
            log_event(step, i, "FALL_DETECTED_BACK_TO_GETUP", gravity_z=round(float(gravity_z[i]), 3))
        was_getup = out["mode_getup"].clone()

        if args_cli.push and bool(out["should_push"].any()):
            push_ids = out["should_push"].nonzero(as_tuple=False).flatten()
            vel = asset.data.root_com_state_w[push_ids, 7:13].clone()
            vel[:, 1] += args_cli.push_speed  # lateral kick, world frame
            asset.write_root_velocity_to_sim(vel, env_ids=push_ids)
            push_applied[push_ids] = True
            for i in push_ids.tolist():
                log_event(step, i, "PUSH_APPLIED", net_displacement_m=round(float(out["net_displacement"][i]), 2),
                          path_length_m=round(float(out["path_length"][i]), 2), push_speed=args_cli.push_speed)
        elif bool(out["reached_distance"].any()):
            for i in out["reached_distance"].nonzero(as_tuple=False).flatten().tolist():
                log_event(step, i, "WALK_DISTANCE_REACHED", net_displacement_m=round(float(out["net_displacement"][i]), 2),
                          path_length_m=round(float(out["path_length"][i]), 2))

        obs_dict, reward, terminated, truncated, extras = raw_env.step(action)

        if args_cli.video:
            pos = asset.data.root_pos_w[0]
            yaw = gc.quat_yaw(*asset.data.root_quat_w[0].tolist())
            cx, cy, cyaw = camera_tracker.update(float(pos[0].item()), float(pos[1].item()), yaw)
            eye, lookat = gc.camera_world_pose(
                cx, cy, cyaw,
                azimuth_offset_deg=args_cli.camera_azimuth_deg, distance_m=args_cli.camera_distance_m,
                eye_height_m=args_cli.camera_eye_height_m, lookat_height_m=args_cli.camera_lookat_height_m,
            )
            raw_env.sim.set_camera_view(eye=eye, target=lookat)
            frame = raw_env.render()
            if frame is not None:
                mode_str = "GET-UP" if bool(out["mode_getup"][0]) else "WALK"
                lines = [
                    f"t: {step * step_dt:.1f}s", f"mode: {mode_str}",
                    # Both numbers are measured from the robot's position AT ITS LAST HANDOFF
                    # (handoff_logic.py's anchor), not from spawn/env origin, so they read ~0
                    # during get-up instead of the robot's raw world position.
                    f"net disp. since handoff: {float(out['net_displacement'][0]):.2f}m",
                    f"path length since handoff: {float(out['path_length'][0]):.2f}m",
                    f"handoffs: {int(sm.num_handoffs[0])}",
                    f"falls: {int(sm.num_falls[0])}",
                ]
                import cv2

                frame = cv2.resize(np.asarray(frame)[..., :3], (args_cli.width, args_cli.height), interpolation=cv2.INTER_AREA)
                frames.append(gc.draw_overlay(frame.copy(), lines))

    # --pose_diag summary: distinguishes "the get-up
    # policy never got close to is_standing" from "it got close but never satisfied the *extra*
    # per-group joint-deviation/angular-velocity handoff gate" -- the two look identical from
    # `handoffs=[0, ...]` alone. Re-run this flag against later checkpoints to see whether the
    # per-group deviations shrink toward their bounds as training progresses, without needing
    # to change the bounds themselves to find out.
    if args_cli.pose_diag:
        final_joint_dev_all = (asset.data.joint_pos - asset.data.default_joint_pos).abs()
        final_ang_vel = asset.data.root_ang_vel_b.norm(dim=-1)
        print(f"[getup-handoff] pose_diag summary (per env): max_stand_hold_s (bar={hcfg.stand_hold_s}) = "
              f"{[round(v, 2) for v in max_stand_hold_s.tolist()]}")
        print(f"[getup-handoff]   getup_state.success (base get-up success, not the handoff gate) = {raw_env.getup_state.success.tolist()}")
        for g, ids in group_joint_ids.items():
            bound = hcfg.max_joint_dev_by_group.get(g)
            final_dev = final_joint_dev_all[:, ids].amax(dim=-1)
            print(f"[getup-handoff]   final joint_dev[{g}] (bound < {bound}) = {[round(v, 3) for v in final_dev.tolist()]}")
        print(f"[getup-handoff]   final ang_vel_norm (bound < {hcfg.max_ang_vel}) = {[round(v, 3) for v in final_ang_vel.tolist()]}")

    # Handoff-gate statistics: per-env/per-category summary
    # built entirely from state already collected above (final `getup_state.success` / `sm.num_handoffs`
    # / `sm.num_falls`, plus the timeline log) -- no extra simulation state needed. The handoff gate's
    # success definition ("get up -> handoff -> walk 5m without falling") is exactly "this env has at least one
    # WALK_DISTANCE_REACHED (or, with --push, PUSH_APPLIED -- both are the *same* underlying
    # `reached_distance` event in handoff_logic.py, just logged under a different name depending on
    # --push) event": `reached_distance` requires `~mode_getup` (still walking, i.e. hasn't fallen
    # since its last handoff) by construction (handoff_logic.py's `step()`), so its presence alone
    # already encodes "reached 5m without an intervening fall" -- no extra bookkeeping needed to keep
    # the two conditions from being conflated.
    final_stood = raw_env.getup_state.success.tolist()
    final_handoffs = sm.num_handoffs.tolist()
    final_falls = sm.num_falls.tolist()
    round_robin = args_cli.start_category == "all"
    first_handoff_t = [None] * n
    reached_5m_t = [None] * n
    for ev in timeline:
        i = ev["env"]
        if ev["event"] == "HANDOFF_TO_WALK" and first_handoff_t[i] is None:
            first_handoff_t[i] = ev["t_s"]
        if ev["event"] in ("WALK_DISTANCE_REACHED", "PUSH_APPLIED") and reached_5m_t[i] is None:
            reached_5m_t[i] = ev["t_s"]

    def _pct(vals: list[bool]) -> float:
        return 100.0 * sum(1 for v in vals if v) / len(vals) if vals else 0.0

    def _summarize(idxs: list[int]) -> dict:
        stood = [final_stood[i] for i in idxs]
        handed_off = [final_handoffs[i] > 0 for i in idxs]
        reached_5m = [reached_5m_t[i] is not None for i in idxs]
        fell_after_handoff = [final_falls[i] > 0 for i in idxs]
        handoff_times = [first_handoff_t[i] for i in idxs if first_handoff_t[i] is not None]
        return {
            "n_episodes": len(idxs),
            "stood_pct": round(_pct(stood), 1),
            "handed_off_pct": round(_pct(handed_off), 1),
            "reached_5m_pct": round(_pct(reached_5m), 1),  # == handoff-gate pass rate (see comment above)
            "falls_after_handoff_pct": round(_pct(fell_after_handoff), 1),
            "falls_after_handoff_total": sum(final_falls[i] for i in idxs),
            "time_to_handoff_mean_s": round(sum(handoff_times) / len(handoff_times), 2) if handoff_times else None,
            "time_to_handoff_p90_s": round(sorted(handoff_times)[max(0, int(round(0.9 * (len(handoff_times) - 1))))], 2)
            if handoff_times else None,
        }

    g4_summary = {"overall": _summarize(list(range(n)))}
    if round_robin:
        cats = [gc.CATEGORY_KEYS[i % len(gc.CATEGORY_KEYS)] for i in range(n)]
        g4_summary["per_category"] = {
            cat: _summarize([i for i in range(n) if cats[i] == cat]) for cat in gc.CATEGORY_KEYS if cat in cats
        }
    g4_summary["g4_gate"] = {
        "criterion": "get up -> handoff -> walk >=5m net displacement, without falling (handoff gate)",
        "threshold_pct": 90.0,
        "actual_pct": g4_summary["overall"]["reached_5m_pct"],
        "passes": bool(g4_summary["overall"]["reached_5m_pct"] >= 90.0),
    }
    with open(os.path.join(output_dir, "g4_metrics.json"), "w") as f:
        json.dump(g4_summary, f, indent=2)
    print(f"[getup-handoff] handoff summary: {json.dumps(g4_summary['overall'])}")
    print(f"[getup-handoff] handoff gate: {json.dumps(g4_summary['g4_gate'])}")
    if "per_category" in g4_summary:
        for cat, s in g4_summary["per_category"].items():
            print(f"[getup-handoff]   per_category[{cat}] = {json.dumps(s)}")
    print(f"[getup-handoff] Wrote handoff summary to {output_dir}/g4_metrics.json")

    env.close()

    with open(os.path.join(output_dir, "timeline.json"), "w") as f:
        json.dump(timeline, f, indent=2)
    print(f"[getup-handoff] Wrote {len(timeline)} timeline events to {output_dir}/timeline.json")

    if args_cli.video and frames:
        video_path = os.path.join(output_dir, "handoff_demo.mp4")
        import imageio

        writer = imageio.get_writer(
            video_path, fps=args_cli.fps, codec="libx264", format="FFMPEG", macro_block_size=None, pixelformat="yuv420p"
        )
        for f in frames:
            writer.append_data(f)
        writer.close()
        print(f"[getup-handoff] Wrote {len(frames)} frames to {video_path}")
    elif args_cli.video:
        print("[getup-handoff] WARN: --video was set but no frames were captured (raw_env.render() returned None every step).")

    print(f"[getup-handoff] Done. handoffs={sm.num_handoffs.tolist()} falls={sm.num_falls.tolist()}")


if __name__ == "__main__":
    main()
    simulation_app.close()
