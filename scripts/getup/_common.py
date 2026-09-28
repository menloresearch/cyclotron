# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Shared, isaaclab-free helpers for the get-up evaluation/video scripts.

Deliberately has no ``isaaclab`` / ``isaac_asimov`` imports so it can be imported before
``AppLauncher`` starts the Kit app (see ``scripts/rsl_rl/play.py`` for why that ordering matters),
and so it is trivially unit-testable off the server.

Not a stable public interface; kept here only to avoid duplicating this logic between
``evaluate.py`` and ``record_videos.py``.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# Frozen names shared with the get-up task
# ---------------------------------------------------------------------------

# Key order is frozen: it must match the "reset_fallen" event params and the env.getup_state.category index order.
CATEGORY_KEYS: list[str] = [
    "supine",
    "prone",
    "side_left",
    "side_right",
    "sitting",
    "kneeling",
    "mid_fall",
    "standing",
]

GETUP_TASK_PLAY = "Asimov1-GetUp-Play-v0"
GETUP_TASK_TRAIN = "Asimov1-GetUp-v0"

# Task success definition (the ">= 1s hold" success; gates use a stricter ">= 5s within 6s" check,
# see `success_5s_within_6s` computed in evaluate.py).
STANDING_PELVIS_HEIGHT_M = 0.50
STANDING_TILT_RAD = 0.35
STANDING_MAX_LIN_VEL = 0.3
STANDING_HOLD_S = 1.0

# Sim-success gate: standing within 6s of control start, held >= 5s; >= 90% overall, >= 80% per category.
G1_TIME_TO_STAND_MAX_S = 6.0
G1_HOLD_S = 5.0
G1_OVERALL_MIN = 0.90
G1_PER_CATEGORY_MIN = 0.80

# Smoothness & safety gate.
G2_TAU_HAT_SAT_THRESHOLD = 0.9
G2_TAU_HAT_SAT_STEP_FRAC_MAX = 0.05
G2_ELBOW_WRIST_SAT_MAX_S = 0.2
G2_ACTION_JITTER_RATIO_MAX = 1.5  # relative to the walking policy's own jitter (external baseline, optional)
# Thermal proxy peak <= 1.0 (the rated-torque EMA at its own rated value). Certifies a short-overload indicator
# (2 s EMA of (tau/tau_rated)^2), not a winding-temperature model -- see getup_env_cfg.py's
# `thermal_proxy_penalty` reward term (threshold=0.6 for the reward penalty; the *gate* threshold here is the
# looser, single-episode-peak bound).
G2_THERMAL_PROXY_PEAK_MAX = 1.0

# Walking-policy baselines for the smoothness & safety gate (get-up's action jitter <= 1.5x the walking
# policy's own; peak non-foot impact <= the walking policy's own falls), measured via
# `scripts/getup/measure_walking_baseline.py` against `Asimov1-Velocity-AMP-Play-v0`.
#
# Measured against a 3000-iteration checkpoint of the public-recipe AMP walker (mean episode length 821/1000
# at measurement time), replacing earlier numbers from a ~100-iteration AMP smoke checkpoint.
# `--walking_baseline_provisional` defaults to False now that a real walking checkpoint backs these numbers.
#
# Method: --mode normal, 16 envs x 30s, no pushes, jitter/rate computed on the actually-applied (clipped)
# action. --mode push, 16 envs x 60s (run alone, not concurrently with another GPU job -- a
# concurrent normal+push attempt hit a CUDA OOM on this measurement), escalating root-velocity pushes (0.5
# m/s base, +0.5 m/s every 2s survived, random horizontal heading) until each env's own `fell_over`
# termination (mdp.bad_orientation, 70 deg tilt) fires; peak non-foot contact force/impulse per fall uses the
# sensor's 4-substep history max, accumulated from that env's last reset to the fall. 159 falls
# observed in 60s x 16 envs (fewer envs/seconds than the smoke-checkpoint's 765-fall sample, since a
# real-walking checkpoint survives each individual push far longer before finally falling -- push_mag at
# fall: mean 1.91 m/s, median 2.0 m/s, vs. the smoke checkpoint's median 0.5 m/s).
WALKING_BASELINE_JITTER_RMS = 0.096  # action_jitter_rms_mean, normal-walking mode (was 0.281 on the smoke checkpoint)
WALKING_BASELINE_ACTION_RATE = 0.039  # action_rate_mean, normal-walking mode (not gated, informational)
# Ground-only measurement (self-contact, e.g. an arm catching the torso mid-fall, excluded
# per _common.build_ground_contact_helper) -- these replace the earlier unfiltered numbers (mean 633.4,
# p95 2739.0, max 8944.8 N), which included self-contact and so overstated what a real ground impact looks
# like. 136 of 159 falls (85.5%) show ZERO ground-level non-foot impact once self-contact is excluded -- most
# falls are caught by feet/ankles, or the fall termination fires before any non-foot body reaches the ground.
WALKING_BASELINE_IMPACT_N = 181.1  # peak_nonfoot_force_n_mean across 159 falls, ground-only (the "typical fall" reading; was 633.4 unfiltered, 1589.0 on the smoke checkpoint)
WALKING_BASELINE_IMPACT_N_P95 = 1339.3  # peak_nonfoot_force_n, 95th percentile across 159 falls, ground-only (for like-for-like transparency reporting, not gated on)
WALKING_BASELINE_IMPACT_N_MAX = 3799.8  # peak_nonfoot_force_n_max across 159 falls, ground-only (worst observed; not used as the gate default -- see note; was 8944.8 unfiltered, 13030.0 on the smoke checkpoint)
WALKING_BASELINE_IMPULSE_NS = 5.0  # nonfoot_impulse_ns_mean across 159 falls, ground-only (informational; impulse is not gated; was 22.1 unfiltered, 54.9 on the smoke checkpoint)
WALKING_BASELINE_PROVENANCE = (
    "measured via scripts/getup/measure_walking_baseline.py against a 3000-iteration checkpoint of the "
    "public-recipe AMP walker (mean episode length 821/1000 at measurement time -- superseding the "
    "~100-iteration AMP smoke checkpoint used previously); ground-only (excludes "
    "self-contact, e.g. an arm catching the torso mid-fall -- a contact diagnostic showed the "
    "unfiltered numbers included this on the get-up side; the same filter is applied to both sides for a "
    "like-for-like comparison, see _common.build_ground_contact_helper). peak-force default is the MEAN of "
    "159 observed push-induced falls (p95=1339.3 N, max=3799.8 N; 136/159 falls show zero ground-level "
    "non-foot impact), not the max -- a 'typical fall' bar, not a worst-case one, matching the gate's "
    "'<= the walking policy's own falls' intent as a mean-vs-mean comparison (the get-up side gates on its "
    "own per-episode-peak MEAN too, not max -- see evaluate.py's peak_nonfoot_impact gate check). No longer "
    "provisional (--walking_baseline_provisional default is now False)."
)

# Robustness gate: sim-success thresholds again, but at 0.9x effort, full DR, rough terrain (sim2sim is checked separately).
G3_EFFORT_SCALE = 0.9
G3_PER_CATEGORY_MIN = 0.80

# g2_impact_vs_own_fall -- in mid_fall episodes, the peak ground
# (non-foot, ground-only) force after policy control starts must stay at or below the peak ground force
# during the limp fall phase of the same episode, in >= this fraction of mid_fall episodes. Absolute hardware
# load limits are a separate hardware sign-off item (HARDWARE_TEST_PROTOCOL.md), not checked here.
G2_IMPACT_VS_OWN_FALL_MIN_RATE = 0.90


def one_hot_category_probs(category: str) -> dict[str, float]:
    """Build a `category_probs` dict (the `reset_fallen` event param) that is 1.0 on `category`."""
    if category not in CATEGORY_KEYS:
        raise ValueError(f"Unknown get-up category {category!r}; expected one of {CATEGORY_KEYS}")
    return {key: (1.0 if key == category else 0.0) for key in CATEGORY_KEYS}


# ---------------------------------------------------------------------------
# Joint / body name classification (ASIMOV_1_JOINT_NAMES order, see assets/robots/asimov_1.py)
# ---------------------------------------------------------------------------

_ELBOW_WRIST_RE = re.compile(r"(elbow|wrist)", re.IGNORECASE)
_FOOT_BODY_RE = re.compile(r"ankle_roll_link$", re.IGNORECASE)

# Rated-torque table (Nm). Used only for the local thermal-proxy fallback in evaluate.py.
RATED_TORQUE_NM_BY_KEYWORD: list[tuple[re.Pattern, float]] = [
    (re.compile(r"hip_pitch"), 40.0),
    (re.compile(r"knee"), 25.0),
    (re.compile(r"hip_roll"), 30.0),
    (re.compile(r"(elbow|wrist)"), 12.0),
]
DEFAULT_RATED_TORQUE_NM = 30.0  # hip_yaw, ankle, waist, shoulder: no explicit rated value, conservative guess

# Local thermal-proxy fallback (evaluate.py): a leaky I^2t-style integrator of (tau/rated_torque)^2 with this
# time constant. The canonical thermal proxy lives in the env's reward terms; this fallback exists so
# evaluate.py still reports a "thermal-proxy peak" metric when that signal is unavailable.
THERMAL_TIME_CONSTANT_S = 5.0


def is_elbow_or_wrist(joint_name: str) -> bool:
    return _ELBOW_WRIST_RE.search(joint_name) is not None


def is_foot_body(body_name: str) -> bool:
    return _FOOT_BODY_RE.search(body_name) is not None


def rated_torque_nm(joint_name: str) -> float:
    for pattern, value in RATED_TORQUE_NM_BY_KEYWORD:
        if pattern.search(joint_name):
            return value
    return DEFAULT_RATED_TORQUE_NM


# ---------------------------------------------------------------------------
# Camera: fixed 3/4 front-side pose, relative to the robot's own (smoothed) heading and xy position.
#
# A height-aware eye/lookat ramp put the camera too close and too high, cutting off the feet and legs
# (e.g. supine at t=2-6s). Fixed values instead:
# eye ~0.9 m high, ~2.2-2.6 m away; lookat at a constant z=0.35 m (not root height, which is ~0.1 m lying and
# ~0.6 m standing). The azimuth is relative to the robot's current (smoothed) yaw, not a fixed world angle,
# so every clip gets a consistent 3/4 view regardless of the randomized spawn heading -- a fixed world azimuth
# can end up behind the robot. This also drops the built-in
# `ViewportCameraController` asset-root auto-tracking entirely (it only snaps exactly to the root, no
# smoothing, and its own event-subscription teardown was a separate source of bugs) in favor of computing the
# world-frame eye/lookat here and setting it directly with `env.unwrapped.sim.set_camera_view(eye=, target=)`.
# ---------------------------------------------------------------------------


def quat_yaw(w: float, x: float, y: float, z: float) -> float:
    """Yaw (rad, world frame) from a (w, x, y, z) orientation quaternion."""
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


class SmoothedTracker:
    """EMA-smoothed (x, y, yaw) world pose, so the tracking camera doesn't jitter frame to frame.

    Yaw is smoothed through its (cos, sin) components so it wraps around correctly at +-pi.
    """

    def __init__(self, alpha: float = 0.15):
        self.alpha = alpha
        self._x: float | None = None
        self._y: float | None = None
        self._cos = 1.0
        self._sin = 0.0

    def update(self, x: float, y: float, yaw: float) -> tuple[float, float, float]:
        c, s = float(np.cos(yaw)), float(np.sin(yaw))
        if self._x is None:
            self._x, self._y, self._cos, self._sin = x, y, c, s
        else:
            a = self.alpha
            self._x = a * x + (1 - a) * self._x
            self._y = a * y + (1 - a) * self._y
            self._cos = a * c + (1 - a) * self._cos
            self._sin = a * s + (1 - a) * self._sin
        return self._x, self._y, float(np.arctan2(self._sin, self._cos))


def camera_world_pose(
    root_x: float,
    root_y: float,
    root_yaw_rad: float,
    azimuth_offset_deg: float = 135.0,
    distance_m: float = 2.4,
    eye_height_m: float = 0.9,
    lookat_height_m: float = 0.35,
    ground_z_m: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """World-frame ``(eye, lookat)`` for a fixed 3/4 front-side shot, relative to the robot's own heading.

    ``azimuth_offset_deg`` is added to ``root_yaw_rad``, so the camera sits at a consistent angle relative to
    the robot's *own* forward direction (not a fixed world-frame angle) -- required because the reset event
    randomizes spawn yaw per episode. 135 degrees puts the camera diagonally in front of and to the side of a
    robot facing along its own +x; adjust per a visual check if the robot's forward axis convention differs.

    ``ground_z_m`` (rough-terrain montage): ``eye_height_m``/``lookat_height_m`` were tuned as
    heights above flat ground (world z=0). On rough/mix terrain the local ground under the robot can sit well
    above or below world z=0, so both heights are added on top of ``ground_z_m`` (the local terrain height
    under the robot's xy, NOT the robot's own current z -- using the robot's own z would make the camera rise
    and fall with the robot as it gets up, defeating the whole point of a fixed framing height). Defaults to
    0.0 (flat terrain, world z=0 ground -- unchanged behavior for every existing caller).
    """
    angle = root_yaw_rad + np.radians(azimuth_offset_deg)
    eye = np.array([root_x + distance_m * np.cos(angle), root_y + distance_m * np.sin(angle), ground_z_m + eye_height_m])
    lookat = np.array([root_x, root_y, ground_z_m + lookat_height_m])
    return eye, lookat


def apply_terrain_toggle(env_cfg, terrain: str) -> str:
    """Terrain toggle for record_videos.py (evaluate.py keeps its own equivalent copy). The env
    cfg provides rough-terrain support via `env_cfg.set_eval_terrain(kind)` ("flat" | "rough" (+-2cm) |
    "mix"), called **before** `gym.make`; it also makes `pelvis_height` measure against the terrain
    under the robot, not the flat env origin. Falls back to the old flat/generator-probe for any cfg that
    predates it (not expected to trigger, but safer than crashing if one slips through).
    """
    if hasattr(env_cfg, "set_eval_terrain"):
        env_cfg.set_eval_terrain(terrain)
        return f"applied via set_eval_terrain({terrain!r})"
    terrain_cfg = getattr(getattr(env_cfg, "scene", None), "terrain", None)
    if terrain_cfg is None:
        return "unsupported: no scene.terrain"
    if terrain == "flat":
        terrain_cfg.terrain_type = "plane"
        return "applied: terrain_type=plane"
    if getattr(terrain_cfg, "terrain_generator", None) is not None:
        terrain_cfg.terrain_type = "generator"
        return "applied: terrain_type=generator"
    return (
        f"unsupported: {terrain} requested but this cfg predates set_eval_terrain and has no "
        "terrain_generator of its own (pre-fix cfg; silently ran flat)"
    )


# ---------------------------------------------------------------------------
# Overlay text (drawn directly onto rgb frames with OpenCV before encoding)
# ---------------------------------------------------------------------------


def draw_overlay(frame_rgb: np.ndarray, lines: list[str], banner: str | None = None, banner_color: tuple[int, int, int] | None = None) -> np.ndarray:
    """Burn a translucent info box (top-left) and an optional outcome banner (bottom) into an RGB frame.

    Mutates and returns `frame_rgb` (uint8, HxWx3, RGB order). Imports cv2 lazily so this module stays
    importable without the venv (e.g. for a quick sanity check off the server).
    """
    import cv2

    frame = frame_rgb
    h, w = frame.shape[:2]
    pad = 10
    line_h = 22
    box_h = pad * 2 + line_h * len(lines)
    box_w = min(w - 20, 520)

    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (10 + box_w, 10 + box_h), (0, 0, 0), thickness=-1)
    frame[:] = cv2.addWeighted(overlay, 0.45, frame, 0.55, 0)

    for i, line in enumerate(lines):
        y = 10 + pad + line_h * i + 16
        cv2.putText(frame, line, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    if banner:
        color = banner_color or (60, 220, 60)
        (tw, th), _ = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        x = (w - tw) // 2
        y = h - 30
        cv2.rectangle(frame, (x - 14, y - th - 14), (x + tw + 14, y + 10), (0, 0, 0), thickness=-1)
        cv2.putText(frame, banner, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)

    return frame


def resolve_ffmpeg_exe() -> str:
    """Return a usable ffmpeg binary path, preferring the venv-bundled `imageio_ffmpeg` static build.

    A training server may have no system `ffmpeg` on $PATH; `imageio_ffmpeg` ships one.
    """
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        import shutil

        exe = shutil.which("ffmpeg")
        if exe is None:
            raise RuntimeError("No ffmpeg available: imageio_ffmpeg is not installed and 'ffmpeg' is not on PATH.")
        return exe


# ---------------------------------------------------------------------------
# Small running-stat helpers used by evaluate.py
# ---------------------------------------------------------------------------


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=float), p))


@dataclass
class Stopwatch:
    """Trivial wall-clock timer used to report wall time for the eval/render loops."""

    start_time: float = field(default_factory=time.time)

    def elapsed_s(self) -> float:
        return time.time() - self.start_time


def peak_rss_mb() -> float:
    """Peak resident-set size of this process (and its children) in MB, on Linux/macOS."""
    import resource
    import sys

    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss is KB on Linux, bytes on macOS.
    return usage / 1024.0 if sys.platform != "darwin" else usage / (1024.0 * 1024.0)


def peak_vram_mb() -> float | None:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Live category selection (evaluate.py / record_videos.py)
#
# Isaac Lab managers deep-copy the cfg they're constructed with (`ManagerBase.__init__`:
# `self.cfg = copy.deepcopy(cfg)`), so mutating `raw_env.cfg.events.reset_fallen` (or `env_cfg` after
# `gym.make()`) never reaches the live `EventManager` and silently leaves the Play cfg's default category mix
# in place. Go through `raw_env.event_manager.get_term_cfg(...)` instead (mirrors `update_category_probs` in
# `mdp/resets.py`), and verify the result against `getup_state.category` after every reset rather than trust
# it silently.
# ---------------------------------------------------------------------------


def set_category_live(raw_env, category: str) -> int:
    """Set the get-up start category through the live `reset_fallen` event term. Returns the category index
    (`CATEGORY_KEYS` order) for use with `verify_category`. `assignment = "round_robin"` forces a
    deterministic split across envs instead of an i.i.d. draw from the (now one-hot) `category_probs`.
    """
    term_cfg = raw_env.event_manager.get_term_cfg("reset_fallen")
    term_cfg.params["category_probs"] = one_hot_category_probs(category)
    term_cfg.params["assignment"] = "round_robin"
    return CATEGORY_KEYS.index(category)


def verify_category(raw_env, expected_idx: int, category: str, log_prefix: str = "[getup-eval]") -> None:
    """Read back `getup_state.category` right after a reset and raise loudly on a mismatch.

    Not optional: a silent mismatch here mislabels every video and eval number the run produces.
    """
    getup_state = getattr(raw_env, "getup_state", None)
    if getup_state is None or not hasattr(getup_state, "category"):
        return
    actual = getup_state.category.detach().cpu()
    print(f"{log_prefix} VERIFY category={category!r} (idx {expected_idx}): getup_state.category={actual.tolist()}", flush=True)
    if not bool((actual == expected_idx).all()):
        bad = actual[actual != expected_idx].tolist()
        raise RuntimeError(
            f"{log_prefix} category mismatch: requested {category!r} (idx {expected_idx}) but "
            f"getup_state.category has {bad} for some envs. The live category_probs/assignment did not take "
            "effect -- do not trust any category label from this run."
        )


def parse_effort_scale_arg(raw: str) -> float | str:
    """`--effort_scale` accepts a float ("0.9") or one of the env's preset names ("nominal"/"stage1"/"stageB"),
    resolved by `getup.mdp.apply_play_effort_scale`/`EFFORT_PRESETS`."""
    try:
        return float(raw)
    except ValueError:
        return raw


def contact_force_norm(sensor) -> "torch.Tensor":
    """Per-body contact-force norm `[N, B]`, taking the max over the sensor's substep history when available.

    `sensor.data.net_forces_w` alone is only the *last* physics substep; with `history_length=4` an impact
    that peaks mid-decimation (5-10 kN single-substep spikes have been observed) is missed 3 times out of 4, biasing any peak-force reading low. Mirrors
    `tasks/getup/mdp/state.py:feet_in_contact`'s own history handling.
    """
    hist = getattr(sensor.data, "net_forces_w_history", None)
    if hist is not None:
        return hist.norm(dim=-1).amax(dim=1)
    return sensor.data.net_forces_w.norm(dim=-1)


def build_ground_contact_helper(robot, contact_sensor, raw_env):
    """Separate ground contact from self-contact for the impact metric: the all-body `body_contact` sensor has no
    `filter_prim_paths_expr`, so `net_forces_w`/`net_forces_w_history` is a body's AGGREGATE contact force
    from EVERYTHING it touches -- ground and self-contact (an arm resting on the torso, the fattened
    collision shells at pelvis/hip touching during hip flexion, etc.) are indistinguishable at the sensor level.
    A contact diagnostic on a trained checkpoint showed the non-foot "impact" peaks were `pelvis_link` <->
    `*_hip_yaw_link` self-contact, both ~0.32-0.37 m above the ground -- not impacts.

    Returns `(collision_spheres, terrain_height_fn, sensor_to_robot_idx)` for `ground_contact_mask_for_sensor`
    below, reusing the reset code's `CollisionSpheres`/`TerrainHeight` (`isaac_asimov.tasks.getup.mdp.resets`) -- the
    exact machinery `mdp.rewards.arm_self_contact` uses for its own (arms-only) self-contact check,
    generalized here to every body with collision geometry (works on flat AND rough/mix terrain, since
    `TerrainHeight` ray-casts against the real terrain mesh when it isn't a plane). Returns `(None, None,
    None)` if unavailable (e.g. a non-getup task, or a robot cfg not spawned from a URDF) -- callers must
    fall back to the OLD unfiltered impact metric in that case, not crash.
    """
    try:
        import torch

        from isaac_asimov.tasks.getup.mdp.resets import _terrain_height, get_collision_spheres

        collision_spheres = get_collision_spheres(robot)
        terrain_height_fn = _terrain_height(raw_env)
        robot_body_names = list(robot.body_names)
        sensor_to_robot_idx = torch.tensor(
            [robot_body_names.index(n) for n in contact_sensor.body_names], device=robot.device, dtype=torch.long
        )
        return collision_spheres, terrain_height_fn, sensor_to_robot_idx
    except Exception as exc:  # noqa: BLE001 - best-effort; fall back to unfiltered rather than crash a whole run
        print(f"[getup-eval] NOTE: ground-contact filter unavailable ({exc!r}); the non-foot impact metric will "
              "be unfiltered (may include self-contact).")
        return None, None, None


def ground_contact_mask_for_sensor(collision_spheres, terrain_height_fn, sensor_to_robot_idx, robot, ground_margin: float = 0.03) -> "torch.Tensor":
    """`[N, num_sensor_bodies]` bool, aligned with the `ContactSensor`'s own body axis (via
    `sensor_to_robot_idx`): True where that body's lowest collision point is within `ground_margin` (default
    0.03 m, matching `mdp.rewards.arm_self_contact`'s own threshold) of the local terrain -- a plausible real
    ground contact. False (excluded) means the force is more likely self-contact."""
    sphere_centers_w = collision_spheres.centers_w(robot.data.body_link_pos_w, robot.data.body_link_quat_w)
    sphere_bottoms = sphere_centers_w[..., 2] - collision_spheres.radii
    lowest_per_body = collision_spheres.lowest_per_body(sphere_bottoms)  # [N, B], robot's own body order
    terrain_z_per_body = terrain_height_fn(robot.data.body_link_pos_w[..., :2])  # [N, B]
    near_ground = (lowest_per_body - terrain_z_per_body) <= ground_margin
    return near_ground[:, sensor_to_robot_idx]


def load_curriculum_state(checkpoint_path: str) -> dict | None:
    """Load the `curriculum_state_<N>.json` matching a checkpoint (`mdp/curriculums.py:curriculum_checkpoint`).

    Contains, among other things, `action_contract` -- the *trained* `beta`/`s_j`/bound-scale/clip -- which
    evaluate.py reads instead of guessing or hard-coding them.

    `model_<N>.pt` and `curriculum_state_<N>.json` are not always written at the same `N` --
    the state is saved on the first reset call of env iteration N, so it's
    typically one step ahead of the checkpoint (`model_3999.pt` paired with `curriculum_state_4000.json`).
    An exact-match-only lookup would silently return None on this off-by-one,
    making evaluate.py fall back to "no curriculum_state" and print the wrong (Play-cfg-default) beta
    for a checkpoint that actually had a real trained contract available. Tries, in order: (1) exact
    iteration match, (2) iteration N+1 (the known off-by-one), (3) whichever `curriculum_state_*.json` in the
    same directory has the closest file *modification time* to the checkpoint's own (handles irregular save
    intervals where neither exact nor N+1 exists). Prints which one it picked. Returns None only if the
    checkpoint filename doesn't parse or no `curriculum_state_*.json` exists in that directory at all.
    """
    import glob
    import json
    import os
    import re

    m = re.search(r"model_(\d+)\.pt$", os.path.basename(checkpoint_path))
    if not m:
        return None
    ckpt_iter = int(m.group(1))
    directory = os.path.dirname(checkpoint_path)

    def _load(path: str, note: str) -> dict:
        print(f"[getup-eval] curriculum_state: using {os.path.basename(path)} for {os.path.basename(checkpoint_path)} ({note})")
        with open(path) as f:
            return json.load(f)

    exact = os.path.join(directory, f"curriculum_state_{ckpt_iter}.json")
    if os.path.isfile(exact):
        return _load(exact, "exact iteration match")

    plus_one = os.path.join(directory, f"curriculum_state_{ckpt_iter + 1}.json")
    if os.path.isfile(plus_one):
        return _load(plus_one, "N+1, the known checkpoint/state save-order off-by-one")

    try:
        ckpt_mtime = os.path.getmtime(checkpoint_path)
    except OSError:
        return None
    candidates = []
    for path in glob.glob(os.path.join(directory, "curriculum_state_*.json")):
        if re.search(r"curriculum_state_\d+\.json$", os.path.basename(path)):
            try:
                candidates.append((abs(os.path.getmtime(path) - ckpt_mtime), path))
            except OSError:
                continue
    if not candidates:
        return None
    candidates.sort(key=lambda c: c[0])
    diff_s, chosen_path = candidates[0]
    return _load(chosen_path, f"closest by file mtime, Δt={diff_s:.1f}s (no exact or N+1 match existed)")


def category_name(idx: int) -> str:
    """Inverse of `CATEGORY_KEYS.index(...)`, for labeling output with the *actual* category
    (`getup_state.category`) rather than trusting the loop variable that requested it."""
    if 0 <= idx < len(CATEGORY_KEYS):
        return CATEGORY_KEYS[idx]
    return f"unknown({idx})"
