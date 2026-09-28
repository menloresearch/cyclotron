# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Evaluate a get-up checkpoint against the success/smoothness/robustness gates and report its metrics.

Gates reported in `metrics.json` under `gates`:
  * G1 (sim success): >= 90 % overall and >= 80 % per start category with zero assist, 1.0x effort, no DR,
    standing within 6 s of policy control and held >= 5 s.
  * G2 (smoothness & safety): torque saturation, elbow/wrist saturation streak, thermal proxy, action jitter
    and ground impact limits.
  * G3 (robustness): per-category success at 0.9x motor strength on rough terrain, split into G3a (pushes
    disabled) and G3c (pushes on).

Loads a checkpoint the same way `scripts/rsl_rl/play.py` does, runs `--num_episodes` per start category by one-hot-setting the
category through the **live** `reset_fallen` event term (`raw_env.event_manager.get_term_cfg(...)`,
NOT `env_cfg`/`raw_env.cfg`, which Isaac Lab's managers deep-copy at construction and never re-read --
see `_common.set_category_live`'s docstring), and writes a metrics JSON (+ a short markdown table) to
`--output_dir`. Every reset is verified against `getup_state.category` and raises loudly on a
mismatch (`_common.verify_category`) -- this is not optional: writing the category to the cfg instead of
the live term silently evaluates the wrong category.

Works against any registered task, not just the get-up one: every getup-specific read
(`env.getup_state`, `getup.mdp.is_standing`, `reset_fallen`) is guarded, so this script still runs
(reporting the generic, non-getup subset of metrics) against a task like
`Asimov1-Velocity-AMP-Play-v0` that has none of that.

Usage:
    python scripts/getup/evaluate.py --task Asimov1-GetUp-Play-v0 \\
        --checkpoint ~/isaac_asimov/logs/rsl_rl/asimov1_getup/<run>/model_4999.pt \\
        --num_episodes 50 --effort_scale nominal --dr off --terrain flat \\
        --output_dir ~/getup_results/<run>/eval/iter_4999 --headless --enable_cameras
"""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import os
import re
import sys


def _self_test() -> None:
    """`python evaluate.py --self_test`: pure torch, no Isaac Sim/AppLauncher needed -- runs fast, anywhere.

    Regression test for an aliasing bug: the action term's `filtered_actions` is a persistent buffer
    (`self._filtered`, mutated in place every `env.step()` -- not a fresh tensor like `policy(obs)`'s raw
    output), and without a clone action_rate read exactly 0.000 in every category: `prev_action =
    effective_action` (no clone) aliased the SAME buffer as the *next* step's `effective_action`, so
    `action - prev_action` was identically zero. This test reproduces that exact aliasing
    scenario with a mock "persistent buffer" action source and asserts the fixed accumulation logic in
    `_run_category` (the clone-once-immediately pattern) does NOT collapse to zero.
    """
    import torch

    torch.manual_seed(0)
    num_envs, num_joints = 3, 5

    class _FakePersistentBuffer:
        """Mimics FilteredRelativeJointPositionAction.filtered_actions: same tensor object every call,
        mutated in place, exactly the shape of the real bug's trigger condition."""

        def __init__(self):
            self._buf = torch.zeros(num_envs, num_joints)

        def step(self, new_value: torch.Tensor):
            self._buf[:] = new_value  # in-place mutation, like the real action term does
            return self._buf  # SAME object every call -- this is the trap

    # A realistic episode length (~700 steps, like a real get-up episode at 50 Hz for ~15s), not a handful --
    # the unfixed pattern's bug only zeroes deltas from the *second* step onward (the very first step's delta
    # is still real, since prev_action starts as a fresh zeros_like tensor, not yet aliased to the buffer).
    # With too few steps that one real term isn't diluted enough to look like the observed "exactly 0.000"
    # (a 3-decimal-place table read); over ~700 steps it rounds away, matching the real symptom.
    n_steps = 700

    def _run(clone_before_use: bool) -> float:
        src = _FakePersistentBuffer()
        prev_action = torch.zeros(num_envs, 0)
        action_rate_sum = torch.zeros(num_envs)
        for _ in range(n_steps):
            raw = torch.rand(num_envs, num_joints)  # a different value every step
            fetched = src.step(raw)
            effective_action = fetched.clone() if clone_before_use else fetched
            if prev_action.shape[-1] == 0:
                prev_action = torch.zeros_like(effective_action)
            delta = effective_action - prev_action
            action_rate_sum += delta.abs().mean(dim=-1)
            prev_action = effective_action
        return float((action_rate_sum / n_steps).mean().item())

    # This is the fixed pattern from _run_category: clone once, immediately, before any other use.
    action_rate_mean = _run(clone_before_use=True)
    assert action_rate_mean > 1e-3, (
        f"[self_test] FAILED: action_rate_mean={action_rate_mean} (expected a real, non-negligible value for "
        "random per-step actions over 700 steps). This is the action-buffer aliasing bug "
        "(action_rate reading 0.000 because prev_action wasn't cloned before aliasing a persistent buffer) "
        "-- it has regressed."
    )

    # Sanity: also confirm the UNFIXED (no-clone) pattern really does collapse toward ~0 (relative to the
    # fixed pattern), so this test is actually exercising the bug, not just checking that *some* action-rate
    # formula gives a nonzero number.
    action_rate_mean_unfixed = _run(clone_before_use=False)
    assert action_rate_mean_unfixed < action_rate_mean / 50.0, (
        f"[self_test] test harness itself is wrong: the deliberately-unfixed pattern gave "
        f"action_rate_mean={action_rate_mean_unfixed}, not small relative to the fixed pattern's "
        f"{action_rate_mean} (aliasing should make all but the first step's delta zero)."
    )

    print(f"[getup-eval] self_test OK: fixed pattern action_rate_mean={action_rate_mean:.4f} (real signal), "
          f"unfixed pattern gives {action_rate_mean_unfixed:.2e} (collapses toward 0 as expected, confirms "
          "the test reproduces the real bug).")


def _resolve_effort_targets(
    trained_contract_available: bool,
    use_trained: bool,
    motor_strength_value,
    contract_bound_scale,
    resolved_preset_bound,
    expected_beta,
) -> dict:
    """Pure decision of what to apply for the action BOUND (`joint_pos_term.set_bound_scale`), beta, and the
    actuator CLIP (`getup_mdp.set_effort_scale`) -- decoupled, see the note in `main()` above the call site.
    No Isaac Sim/torch dependency, so it's exercised directly by `--self_test` (a regression check that two
    different `--motor_strength` values yield two different clips).
    Returns `{"bound": ..., "beta": ..., "clip": ...}`; any value of `None` means "don't touch it".
    """
    if motor_strength_value is not None:
        if trained_contract_available:
            bound = contract_bound_scale
        elif not use_trained:
            bound = resolved_preset_bound
        else:
            bound = None  # fallback: leave whatever bound is already live (Play cfg's own default)
        beta = expected_beta if trained_contract_available else None
        clip = motor_strength_value  # ABSOLUTE clip, decoupled from the bound
        return {"bound": bound, "beta": beta, "clip": clip}
    if trained_contract_available:
        return {"bound": contract_bound_scale, "beta": expected_beta, "clip": None}
    if not use_trained:
        return {"bound": resolved_preset_bound, "beta": None, "clip": None}
    return {"bound": None, "beta": None, "clip": None}  # use_trained, no contract, no --motor_strength


def _self_test_motor_strength_decoupling() -> None:
    """Regression check that `--motor_strength` is applied as an ABSOLUTE clip, independent of the bound.
    Reproduces the original symptom: `--motor_strength 1.0` on a Stage-B checkpoint (bound=0.9) was
    numerically identical to the untouched default, because the old code composed
    `clip = bound * motor_strength`."""
    stageb_bound = {"j0": 0.9, "j1": 0.9}  # a Stage-B-style trained contract, uniform 0.9 bound

    # (a) Two DIFFERENT motor strengths against the SAME (Stage-B) bound must give two DIFFERENT clips --
    # running two strengths must yield different tau_hat, i.e. a different clip.
    plan_g1 = _resolve_effort_targets(True, True, 1.0, stageb_bound, None, 0.7)
    plan_default = _resolve_effort_targets(True, True, 0.9, stageb_bound, None, 0.7)
    assert plan_g1["clip"] == 1.0 and plan_default["clip"] == 0.9 and plan_g1["clip"] != plan_default["clip"], (
        f"[self_test] FAILED: --motor_strength 1.0 and 0.9 against the same bound must resolve to different "
        f"clips; got {plan_g1['clip']} and {plan_default['clip']}. This is the motor_strength composition bug "
        "(motor_strength=1.0 on a 0.9-bound checkpoint reproducing the untouched clip=0.9) -- "
        "it has regressed."
    )
    # (b) The bound itself must still be applied at its own trained value (0.9), untouched by motor_strength
    # -- "whatever the bound" -- not silently forced to 1.0 alongside the clip.
    assert plan_g1["bound"] == stageb_bound, (
        f"[self_test] FAILED: --motor_strength must not touch the trained bound; got bound={plan_g1['bound']} "
        f"(expected the untouched Stage-B contract {stageb_bound})."
    )
    # (c) Old buggy composition, for contrast: clip = bound * motor_strength gives 0.9 at motor_strength=1.0
    # on a 0.9 bound, i.e. the untouched default clip -- confirm the OLD formula really reproduces that, so
    # this test would have caught the real bug.
    old_buggy_clip_g1 = stageb_bound["j0"] * 1.0
    old_buggy_clip_default = stageb_bound["j0"] * 0.9
    assert abs(old_buggy_clip_g1 - 0.9) < 1e-9 and old_buggy_clip_g1 != old_buggy_clip_default, (
        "[self_test] test harness itself is wrong: expected the old multiplicative formula to give a "
        "misleadingly-close-to-nominal 0.9 clip at motor_strength=1.0, distinct from its own clip at 0.9."
    )
    # But the NEW clip that _resolve_effort_targets actually plans is NOT the old buggy value:
    assert plan_g1["clip"] != old_buggy_clip_g1 or plan_g1["clip"] == 1.0, (
        f"[self_test] FAILED: new clip {plan_g1['clip']} should be the absolute motor_strength (1.0), not "
        f"the old multiplicative composition ({old_buggy_clip_g1})."
    )

    # (d) No --motor_strength given at all: bound and clip still move together (unchanged legacy behavior).
    plan_no_motor = _resolve_effort_targets(True, True, None, stageb_bound, None, 0.7)
    assert plan_no_motor["clip"] is None and plan_no_motor["bound"] == stageb_bound, (
        f"[self_test] FAILED: with no --motor_strength, expected clip=None (apply_play_effort_scale's own "
        f"composed default, untouched) and bound={stageb_bound}; got {plan_no_motor}."
    )

    print("[getup-eval] self_test OK (motor_strength decoupling): "
          f"motor_strength=1.0 -> clip={plan_g1['clip']}, bound unchanged={plan_g1['bound']}; "
          f"motor_strength=0.9 -> clip={plan_default['clip']} (different from the 1.0 case, as required).")


if "--self_test" in sys.argv:
    _self_test()
    _self_test_motor_strength_decoupling()
    sys.exit(0)

from isaaclab.app import AppLauncher

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rsl_rl"))
import cli_args  # isort: skip  (scripts/rsl_rl/cli_args.py)

import _common as gc  # isort: skip  (scripts/getup/_common.py)

parser = argparse.ArgumentParser(description="Evaluate a get-up (or other) RSL-RL checkpoint against the get-up gates.")
parser.add_argument("--task", type=str, default=gc.GETUP_TASK_PLAY, help="Gym task id to evaluate.")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point (see scripts/rsl_rl/play.py).")
# NOTE: --checkpoint itself is added by cli_args.add_rsl_rl_args() below (matches scripts/rsl_rl/play.py's
# --target pattern of not re-declaring it) -- declaring it twice is an argparse conflict.
parser.add_argument("--num_envs", type=int, default=16, help="Envs run in parallel per category (default kept <=16 so this fits alongside a concurrent training run).")
parser.add_argument(
    "--num_episodes", type=int, default=20, help="Episodes to collect *per category* (rounded up to a multiple of num_envs)."
)
parser.add_argument(
    "--categories",
    type=str,
    default=",".join(gc.CATEGORY_KEYS),
    help=f"Comma-separated subset of {gc.CATEGORY_KEYS}. Ignored if the task has no `reset_fallen` event.",
)
parser.add_argument(
    "--effort_scale", type=str, default="trained",
    help="Trained-condition scale: sets BOTH the policy's action bound and the actuator clip. Default "
    "'trained' reads the checkpoint's own curriculum_state_<N>.json action_contract and "
    "applies its exact per-joint bound_scale/beta -- required for a Stage B (or any mid-curriculum) "
    "checkpoint, which would otherwise be evaluated at the wrong (nominal) bound, biasing "
    "every G2 torque metric; falls back to the Play cfg's own baked-in default (no override) if the "
    "checkpoint has no curriculum_state file. For deliberate certification at a FIXED setting regardless of "
    "what the checkpoint trained at, pass an explicit preset/float instead (nominal=1.0/stage1/stageB, or a "
    "number) -- G1 uses nominal. Do NOT use this for G3's '0.9x effort' -- that must weaken the clip only; "
    "use --motor_strength instead; G3 = --effort_scale trained "
    "--motor_strength 0.9.",
)
parser.add_argument(
    "--motor_strength", type=str, default=None,
    help="Stress-test scale for the actuator clip ONLY (a float or preset), leaving the policy's trained action "
    "bound at --effort_scale's value -- e.g. G3's weak-motor check. Maps to getup.mdp.apply_play_effort_scale's "
    "motor_strength arg / env_cfg.play_motor_strength. Omit to leave the clip == the bound (nominal use).",
)
parser.add_argument(
    "--dr", type=str, default="off", choices=["on", "narrow", "off"],
    help="Domain-randomization toggle. 'off': disabled. 'on': the WIDE DR "
    "ranges (WIDE_DR_OVERRIDES, pushes included at +-0.5 m/s) -- what Stage B/B2 actually trains under from "
    "init, and what G3 ('full DR') means. 'narrow': init-time/narrow ranges only "
    "(pushes +-0.1 m/s) -- kept for comparison, not what the G3 gate should be read against.",
)
parser.add_argument(
    "--disable_dr_term", type=str, action="append", default=None,
    help="Disable a named DR event term (repeatable, e.g. --disable_dr_term push_robot --disable_dr_term physics_material) "
    "while leaving everything else --dr turned on "
    "untouched -- for isolating one DR source's effect (e.g. a G3 push-vs-no-push breakdown). "
    "Refuses to run if that term isn't enabled.",
)
parser.add_argument("--terrain", type=str, default="flat", choices=["flat", "rough", "mix"], help="Terrain toggle. 'rough'/'mix' need the env cfg's set_eval_terrain; falls back to a flat no-op note on an older cfg.")
parser.add_argument(
    "--break_diag", type=lambda s: s.strip().lower() not in ("false", "0", "no"), default=False,
    help="Hold-break diagnostics: for every episode that reached the 1s success latch "
    "but then broke before G1's full hold, record which is_standing sub-condition (height/tilt/feet/vel) "
    "was violated at the break, and whether a push_robot DR event fired in the preceding 0.5s (detected via "
    "the event manager's own interval countdown, best-effort). Adds a 'g1_break_diag' section per category.",
)
parser.add_argument("--output_dir", type=str, required=True, help="Directory to write metrics.json / metrics.md into.")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--walking_jitter_baseline", type=float, default=gc.WALKING_BASELINE_JITTER_RMS,
    help="Walking policy's action-jitter RMS, for the G2 <=1.5x check. Defaults to the measured baseline "
    "(see _common.py for provenance/caveats); pass 0 or a negative number to skip that sub-check.",
)
parser.add_argument(
    "--walking_impact_baseline_n",
    type=float,
    default=gc.WALKING_BASELINE_IMPACT_N,
    help="Walking policy's own peak non-foot impact force (N) from its own falls, for the G2 check. Defaults "
    "to the measured baseline (see _common.py for provenance/caveats); pass 0 or negative to skip.",
)
parser.add_argument(
    "--thermal_threshold", type=float, default=None, help="Thermal-proxy peak threshold for the G2 check. Omit to use the default (_common.G2_THERMAL_PROXY_PEAK_MAX)."
)
parser.add_argument(
    "--impact_warmup_s", type=float, default=0.1,
    help="Exclude this many seconds right after each episode reset from the non-foot impact-force/impulse "
    "metrics (a 'standing' category run showed an implausible ~26 kN spike with 100%% success "
    "-- a reset/spawn artifact, not a real impact). Set to 0 to disable.",
)
parser.add_argument(
    "--walking_baseline_provisional", type=lambda s: s.strip().lower() not in ("false", "0", "no"), default=False,
    help="The walking G2 baselines (jitter/impact) are measured from a trained (public-recipe) walking "
    "checkpoint -- see _common.WALKING_BASELINE_PROVENANCE. Default False counts the action_jitter G2 check "
    "toward G2's pass/fail. Pass true to mark it provisional (computed but not counted), e.g. for a "
    "walking checkpoint you don't yet trust as a baseline.",
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.enable_cameras = getattr(args_cli, "enable_cameras", False)
if not args_cli.checkpoint:
    parser.error("--checkpoint is required")

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --------------------------------------------------------------------------------------------------
# Heavy imports (after the Kit app is up, matching scripts/rsl_rl/play.py's ordering).
# --------------------------------------------------------------------------------------------------
import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.assets import retrieve_file_path

import gymnasium as gym
import isaac_asimov.tasks  # noqa: F401
import isaaclab_tasks  # noqa: F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg, handle_deprecated_rsl_rl_checkpoint
from isaaclab_tasks.utils.hydra import hydra_task_config

try:
    from isaac_asimov.tasks.getup import mdp as getup_mdp  # noqa: F401
except Exception as exc:  # noqa: BLE001 - report, don't crash: the get-up task may not be installed.
    getup_mdp = None
    print(f"[getup-eval] NOTE: 'isaac_asimov.tasks.getup.mdp' is not importable ({exc!r}). "
          "Running in generic (non-getup) mode.")


def _has_reset_fallen(env_cfg) -> bool:
    return getattr(getattr(env_cfg, "events", None), "reset_fallen", None) is not None



def _parse_effort_scale_arg(raw: str) -> float | str:
    """`--effort_scale` accepts a float ("0.9") or one of the env's preset names ("nominal"/"stage1"/"stageB")."""
    try:
        return float(raw)
    except ValueError:
        return raw


def _apply_effort_scale(raw_env, scale: float | str | dict | None, motor_strength: float | str | None = None):
    """Route through the env's `apply_play_effort_scale`, not a direct `set_effort_scale` call: it resolves
    the named presets (nominal/stage1/stageB) and is documented as callable by evaluation scripts after
    gym.make. `motor_strength` weakens the actuator clip only, independent of `scale`'s effect on the action
    bound.
    """
    if getup_mdp is None or not hasattr(getup_mdp, "apply_play_effort_scale"):
        print(f"[getup-eval] WARN: getup.mdp.apply_play_effort_scale unavailable; --effort_scale={scale} was NOT applied.")
        return
    getup_mdp.apply_play_effort_scale(raw_env, None, scale, motor_strength)


# From source/isaac_asimov/isaac_asimov/tasks/getup/getup_env_cfg.py: the exact 8 DR event-term attribute names
# EventCfg defines, and the exact list Asimov1GetUpEnvCfg_PLAY.__post_init__ nulls out for "no DR". The regex
# below is only a forward-compatible fallback for names added later that this list doesn't know about yet.
KNOWN_GETUP_DR_EVENT_NAMES = (
    "physics_material", "link_mass", "torso_mass", "torso_com", "joint_armature", "actuator_gains",
    "torso_wrench", "push_robot",
)  # fmt: skip


def _apply_dr_toggle(env_cfg, mode: str):
    """DR toggle for the get-up Play cfg. `mode`:
      - "off": disable all DR event terms.
      - "narrow": copy DR terms verbatim from the sibling training cfg class (same module, class name minus
        `_PLAY`) -- their own class-level, init-time (narrow) params. This was the entire old "--dr on".
      - "on": the SAME copy as "narrow", then also patches in
        `WIDE_DR_OVERRIDES` (module-level dict next to the env cfg, e.g. `getup_env_cfg.py`) on top -- the
        exact params `mdp.curriculums.domain_rand_widen` applies once `getup_state.stage >= 1` mid-training.
        Play/eval envs run with 0 active curriculum terms, so that widening never fires there on its own;
        Stage B/B2 train with DR fully wide from init, and "full DR" means the widened ranges, so "--dr on"
        must reproduce THAT to be a faithful G3 read, not the narrow stage-0 default.
        Patching happens on the cfg BEFORE `gym.make()`, so (unlike `domain_rand_widen`'s own runtime
        re-application to an already-constructed term) no bucket-resample dance is needed: every term
        samples fresh from the wide ranges the first time the EventManager ever constructs/calls it.
    See `KNOWN_GETUP_DR_EVENT_NAMES` above for where the exact name list came from; the regex fallback covers
    anything added later under a name that list doesn't know about.
    """
    touched: list[str] = []
    events_cfg = getattr(env_cfg, "events", None)
    if events_cfg is None:
        return touched
    dr_name_re = re.compile(r"(friction|material|mass|com|armature|gain|motor|push|wrench|gravity|randomiz)", re.IGNORECASE)

    def _is_dr_term(name: str) -> bool:
        return name in KNOWN_GETUP_DR_EVENT_NAMES or bool(dr_name_re.search(name))

    if mode == "off":
        for name in list(vars(events_cfg).keys()):
            if name.startswith("_") or name == "reset_fallen":
                continue
            if _is_dr_term(name) and getattr(events_cfg, name, None) is not None:
                setattr(events_cfg, name, None)
                touched.append(f"disabled:{name}")
        return touched

    # "narrow" and "on" (wide) both start by copying DR terms from the sibling training cfg class.
    cls = env_cfg.__class__
    train_cls_name = cls.__name__.replace("_PLAY", "")
    train_cls = getattr(sys.modules.get(cls.__module__), train_cls_name, None)
    if train_cls is None or train_cls is cls:
        print(f"[getup-eval] WARN: --dr {mode} requested but no sibling training env cfg class was found to copy DR terms from.")
        return touched
    train_events = getattr(train_cls(), "events", None)
    if train_events is None:
        return touched
    for name, value in vars(train_events).items():
        if name.startswith("_") or name == "reset_fallen":
            continue
        if _is_dr_term(name) and value is not None and getattr(events_cfg, name, None) is None:
            setattr(events_cfg, name, value)
            touched.append(f"enabled:{name}")
    if not touched:
        print(f"[getup-eval] WARN: --dr {mode} requested but nothing was copied (Play cfg may already match training cfg, or names don't match the DR heuristic).")

    if mode == "on":  # wide
        wide_overrides = getattr(sys.modules.get(cls.__module__), "WIDE_DR_OVERRIDES", None)
        if wide_overrides is None:
            print("[getup-eval] WARN: --dr on (wide) requested but this task module has no WIDE_DR_OVERRIDES; DR stayed narrow.")
        else:
            for name, params in wide_overrides.items():
                term_cfg = getattr(events_cfg, name, None)
                if term_cfg is None:
                    continue  # not an enabled term on this cfg (e.g. disabled by name-heuristic mismatch)
                term_cfg.params.update(params)
                touched.append(f"widened:{name}={params}")
    return touched


def _disable_dr_term(env_cfg, name: str) -> bool:
    """Disable exactly one named DR event term, leaving every other term `--dr on` enabled untouched -- e.g.
    `--disable_dr_term push_robot` isolates the push-perturbation DR source for a G3 break-cause breakdown
    without also disabling `torso_wrench`/`actuator_gains`/etc. Returns False (no-op) if the
    term wasn't already enabled (nothing to disable)."""
    events_cfg = getattr(env_cfg, "events", None)
    if events_cfg is None or getattr(events_cfg, name, None) is None:
        return False
    setattr(events_cfg, name, None)
    return True


def _apply_terrain_toggle(env_cfg, terrain: str) -> str:
    """Apply the eval terrain via `env_cfg.set_eval_terrain(kind)` ("flat" | "rough" (+-2cm) | "mix"),
    called **before** `gym.make`; it also makes `pelvis_height` measure against the terrain under the robot,
    not the flat env origin. For a cfg without that method (whose `TerrainImporterCfg` is a hardcoded
    `terrain_type="plane"` with no generator), fall back to a flat/generator probe -- not expected to trigger
    with the current env cfgs, but safer than crashing.
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
        f"unsupported: {terrain} requested but this cfg has no set_eval_terrain and no "
        "terrain_generator of its own (older cfg; would silently run flat)"
    )


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg):
    torch.manual_seed(args_cli.seed)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))

    has_getup = getup_mdp is not None and _has_reset_fallen(env_cfg)
    categories = [c.strip() for c in args_cli.categories.split(",") if c.strip()] if has_getup else ["default"]
    for c in categories:
        if has_getup and c not in gc.CATEGORY_KEYS:
            raise ValueError(f"Unknown category {c!r}; expected one of {gc.CATEGORY_KEYS}")

    dr_touched = _apply_dr_toggle(env_cfg, args_cli.dr)
    for _term in (args_cli.disable_dr_term or []):
        if _disable_dr_term(env_cfg, _term):
            dr_touched.append(f"disabled:{_term} (--disable_dr_term, on top of --dr {args_cli.dr})")
        else:
            raise ValueError(f"--disable_dr_term {_term!r} is not an enabled event term; refusing to run a mislabeled ablation.")
    terrain_note = _apply_terrain_toggle(env_cfg, args_cli.terrain)

    motor_strength_value = _parse_effort_scale_arg(args_cli.motor_strength) if args_cli.motor_strength is not None else None

    # The *default* must replicate exactly what the checkpoint was trained under, not "nominal" -- a
    # Stage B policy trained at bound x0.9 / beta 0.7 evaluated at the nominal x1.0 bound sees an ~11%
    # larger command than it ever trained with, biasing every G2 torque metric. "--effort_scale trained"
    # (the default) reads the checkpoint's own curriculum_state_<N>.json -> action_contract and applies
    # its exact per-joint bound_scale and beta -- both the action bound AND (unless --motor_strength
    # overrides it) the actuator clip, via the env's apply_play_effort_scale (bound_scale used as both `scale` and, absent --motor_strength, the clip too:
    # in stage 0/1 the two are set to identical values at training time anyway -- see
    # mdp/curriculums.py:effort_beta_schedule; only Stage B's additional per-env randomized motor-strength
    # jitter isn't reproduced exactly, since it's randomized per env/per reset and can't be replayed
    # deterministically -- this uses the pre-jitter central value instead, which --motor_strength can then
    # stress on top of for a G3 read). Falls back to the Play cfg's own baked-in default (NO override at
    # all) when the checkpoint has no curriculum_state_<N>.json or no action_contract in it. Explicit
    # presets/floats (nominal/stage1/stageB/a number) remain available for deliberate experiments and use
    # a fixed bound/beta pairing.
    resume_path = retrieve_file_path(args_cli.checkpoint)
    curriculum_state = gc.load_curriculum_state(resume_path)
    checkpoint_contract = curriculum_state.get("action_contract") if curriculum_state else None
    use_trained = args_cli.effort_scale == "trained"
    trained_contract_available = use_trained and checkpoint_contract and "bound_scale" in checkpoint_contract and "beta" in checkpoint_contract

    if use_trained and not trained_contract_available:
        print(f"[getup-eval] NOTE: --effort_scale trained requested but no usable curriculum_state_<N>.json/"
              f"action_contract found for {resume_path!r}; falling back to the Play cfg's own baked-in "
              "default (no override applied).")

    effort_scale_value = None if use_trained else _parse_effort_scale_arg(args_cli.effort_scale)
    is_stageb_effort = effort_scale_value == "stageB" or (isinstance(effort_scale_value, (int, float)) and effort_scale_value < 1.0)
    # Only claim an "expected" beta for the two cases where this script is actually ABOUT to force one -- the trained-contract case (forces the checkpoint's own beta) and the
    # explicit-preset case (forces the nominal/stageB pairing). The fallback case (no contract, no override
    # applied at all) doesn't get to assume "Play's default is 1.0": e.g. Asimov1GetUpStageBEnvCfg_PLAY's
    # OWN baked-in default is 0.7, not 1.0; hard-coding 1.0 there would report a false "beta=1.0" next to
    # the real live value. `expected_beta` is left None and resolved from the live contract after gym.make.
    if trained_contract_available:
        expected_beta = float(checkpoint_contract["beta"])
    elif not use_trained:
        expected_beta = 0.7 if is_stageb_effort else 1.0
    else:
        expected_beta = None  # fallback: whatever the env_cfg's own default beta turns out to be, unforced
    effort_source = (
        f"trained:{resume_path}" if trained_contract_available
        else ("play_cfg_default (no curriculum_state)" if use_trained else f"preset:{args_cli.effort_scale}")
    )

    if not use_trained:
        # Explicit preset/float path: set the beta pairing and play_effort_scale
        # BEFORE gym.make -- see the deep-copy-timing note below.
        joint_pos_action = getattr(getattr(env_cfg, "actions", None), "joint_pos", None)
        if joint_pos_action is not None and hasattr(joint_pos_action, "beta"):
            joint_pos_action.beta = expected_beta
        # Set BEFORE gym.make, not after --
        # env_cfg.play_effort_scale is read by the Play cfg's own "play_effort" startup event term the
        # moment the env is constructed. Mutating it (or any nested cfg like events.reset_fallen.params)
        # *after* gym.make has no effect: every Isaac Lab manager deep-copies the cfg it's given
        # (`ManagerBase.__init__`: `self.cfg = copy.deepcopy(cfg)`), so a later mutation of
        # `env_cfg`/`raw_env.cfg` never reaches the live manager.
        if hasattr(env_cfg, "play_effort_scale"):
            env_cfg.play_effort_scale = effort_scale_value
        if motor_strength_value is not None and hasattr(env_cfg, "play_motor_strength"):
            env_cfg.play_motor_strength = motor_strength_value
    # "trained" mode deliberately does NOT touch env_cfg before gym.make: when a contract is available it's
    # applied exactly, after gym.make, on the constructed action-term object (see below); when it isn't,
    # the whole point is to change nothing and let the Play cfg's own defaults stand untouched.

    print(f"[getup-eval] Loading checkpoint: {resume_path}")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    raw_env: ManagerBasedRLEnv = env.unwrapped

    # Guard against the same class of bug as the category deep-copy issue: `dr_terms_touched` only proves a
    # cfg OBJECT got patched, not what the live EventManager actually reads (symptom: bit-identical
    # trajectories with push magnitudes 5x apart). Verify directly, right after gym.make: read back the LIVE term cfg via
    # `event_manager.get_term_cfg(...)` (not `env_cfg`/`raw_env.cfg`, which managers deep-copy at
    # construction and never re-read -- see `_common.set_category_live`'s docstring for the original
    # instance of this bug class) and assert it actually matches `WIDE_DR_OVERRIDES`. A **permanent**
    # assertion in `--dr on` mode, not a one-off debug print, so a regression here is loud immediately.
    if has_getup and args_cli.dr == "on":
        wide_overrides = getattr(sys.modules.get(env_cfg.__class__.__module__), "WIDE_DR_OVERRIDES", None)
        if wide_overrides:
            mismatches = []
            for name, expected_params in wide_overrides.items():
                try:
                    live_cfg = raw_env.event_manager.get_term_cfg(name)
                except (KeyError, ValueError):
                    continue  # term not active on this cfg -- nothing live to check
                print(f"[getup-eval] DR-wide live-param check: {name}.params (live, post-gym.make) = {live_cfg.params}")
                for k, expected_v in expected_params.items():
                    live_v = live_cfg.params.get(k)
                    if live_v != expected_v:
                        mismatches.append((name, k, live_v, expected_v))
            if mismatches:
                raise RuntimeError(
                    f"[getup-eval] --dr on (wide) assert failed: the LIVE event term params do not match "
                    f"WIDE_DR_OVERRIDES -- mismatches (term, param, live, expected): {mismatches}. This is "
                    "the same class of bug as the category deep-copy bug (pre-gym.make cfg mutations not "
                    "reaching the live EventManager). Do not trust this run's G3 numbers."
                )
            print(f"[getup-eval] DR-wide live-param check: PASSED -- all {len(wide_overrides)} widened terms' "
                  "live params match WIDE_DR_OVERRIDES.")

    # motor_strength semantics: `apply_play_effort_scale`'s own
    # `clip = bound * motor_strength` treats motor_strength as a MULTIPLIER on the bound -- but
    # `--motor_strength 1.0` on a Stage B checkpoint (bound 0.9) then reproduces clip=0.9*1.0=0.9, not the
    # clip=1.0 a "G1 at full motor strength, whatever the bound" test actually needs (such a run was
    # numerically identical to the untouched default). When
    # --motor_strength is given, apply it as an ABSOLUTE clip value, decoupled from the bound, via two
    # independent calls (`set_bound_scale` then `set_effort_scale`) instead of the composed helper. When it
    # isn't given, bound and clip still move together through `apply_play_effort_scale`, exactly matching
    # what training itself did in stage 0/1 (`mdp/curriculums.py:effort_beta_schedule` calls
    # `set_bound_scale`/`set_effort_scale` with the identical value there). The decision of *what* to set is
    # factored into `_resolve_effort_targets` (pure, no Isaac Sim) so it's covered by `--self_test` (a
    # regression check that two different motor strengths yield two different clips) without needing a
    # live env; this block only performs the actual (side-effecting) calls its plan says to make.
    joint_pos_term = None
    try:
        joint_pos_term = raw_env.action_manager.get_term("joint_pos")
    except AttributeError:
        pass  # non-getup task

    contract_bound_scale = (
        dict(zip(checkpoint_contract["joint_names"], checkpoint_contract["bound_scale"])) if trained_contract_available else None
    )
    resolved_preset_bound = (
        getup_mdp.resolve_effort_scale(effort_scale_value) if isinstance(effort_scale_value, str) else effort_scale_value
    )
    plan = _resolve_effort_targets(
        trained_contract_available=trained_contract_available,
        use_trained=use_trained,
        motor_strength_value=motor_strength_value,
        contract_bound_scale=contract_bound_scale,
        resolved_preset_bound=resolved_preset_bound,
        expected_beta=expected_beta,
    )
    if joint_pos_term is not None:
        if plan["bound"] is not None:
            joint_pos_term.set_bound_scale(plan["bound"])
        if plan["beta"] is not None:
            joint_pos_term.beta = plan["beta"]
    if plan["clip"] is not None:
        # `set_effort_scale(env, scale: float | dict, asset_name="robot", env_ids=None)` -- no preset-string
        # support of its own (unlike `apply_play_effort_scale`, which calls `resolve_effort_scale` internally);
        # resolve a preset name (e.g. "stageB") here at the call site, same as `resolved_preset_bound` above,
        # so `_resolve_effort_targets` itself stays pure/Isaac-Sim-free for --self_test. Also NOTE:
        # asset_name is the 3rd positional param, not env_ids -- call with keywords past `scale` to avoid
        # passing env_ids positionally by mistake.
        clip_value = getup_mdp.resolve_effort_scale(plan["clip"]) if isinstance(plan["clip"], str) else plan["clip"]
        getup_mdp.set_effort_scale(raw_env, clip_value, asset_name="robot")
    # else: use_trained with no contract and no --motor_strength -- genuinely apply nothing, as intended.

    # Assert the live action bound actually ended up at what we intended.
    live_beta = None
    live_bound_scale = None
    try:
        live_contract = raw_env.action_manager.get_term("joint_pos").contract()
        live_beta = live_contract.get("beta")
        live_bound_scale = live_contract.get("bound_scale")
        if trained_contract_available and live_beta is not None and abs(live_beta - expected_beta) > 1e-6:
            raise RuntimeError(
                f"[getup-eval] action-bound assert failed: intended beta={expected_beta} (from the checkpoint's "
                f"own trained contract) but the live env's action_contract reports beta={live_beta}. Do not "
                "trust this run's G1/G2/G3 numbers."
            )
        elif not use_trained and live_beta is not None and abs(live_beta - expected_beta) > 1e-6:
            raise RuntimeError(
                f"[getup-eval] action-bound assert failed: set actions.joint_pos.beta={expected_beta} before "
                f"gym.make, but the live env's action_contract reports beta={live_beta}. The pre-gym.make "
                "mutation didn't take effect -- do not trust this run's G1/G2/G3 numbers."
            )
        print(f"[getup-eval] applied action contract ({effort_source}): live beta={live_beta}, "
              f"bound_scale={live_bound_scale}, motor_strength={motor_strength_value}")
    except AttributeError:
        pass  # non-getup task: no "joint_pos" action term with a .contract() method
    if expected_beta is None:
        # Fallback case: nothing was forced, so "expected" just means whatever ended up live (Play cfg's own
        # default, e.g. Asimov1GetUpStageBEnvCfg_PLAY's own beta=0.7, not a guessed 1.0) -- report that
        # honestly downstream (run_params, the metrics.md header) instead of a hard-coded number.
        expected_beta = live_beta

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    resume_path = handle_deprecated_rsl_rl_checkpoint(resume_path, metadata.version("rsl-rl-lib"))
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    robot = raw_env.scene["robot"]
    joint_names = list(robot.joint_names)
    num_joints = len(joint_names)
    elbow_wrist_idx = [i for i, n in enumerate(joint_names) if gc.is_elbow_or_wrist(n)]
    rated_torque = torch.tensor([gc.rated_torque_nm(n) for n in joint_names], device=env.unwrapped.device)
    effort_limits = _actuator_effort_limits(robot)
    # Report the CLIP that actually ended up live per joint group, not
    # just the intended bound_scale/motor_strength inputs -- catches this class of bug (motor_strength
    # silently not applied) by inspection even without re-deriving tau_hat. env 0 is representative (DR, if
    # on, only perturbs a few terms, not effort_limit itself).
    applied_clip_by_group = _summarize_clip_by_group(effort_limits[0], joint_names, elbow_wrist_idx)
    print(f"[getup-eval] live applied clip (Nm) by joint group: {applied_clip_by_group}")

    contact_sensor = _find_all_body_contact_sensor(raw_env)
    foot_body_mask = None
    collision_spheres = terrain_height_fn = sensor_to_robot_idx = None
    if contact_sensor is not None:
        body_names = contact_sensor.body_names
        foot_body_mask = torch.tensor([gc.is_foot_body(n) for n in body_names], device=env.unwrapped.device)
        # The non-foot "impact" metric must count only ground contact, not self-contact (e.g.
        # pelvis_link<->*_hip_yaw_link shell contact during hip flexion can otherwise dominate the "impact").
        # See `_common.build_ground_contact_helper`'s docstring for details.
        collision_spheres, terrain_height_fn, sensor_to_robot_idx = gc.build_ground_contact_helper(robot, contact_sensor, raw_env)

    step_dt = env.unwrapped.step_dt
    num_envs = env.unwrapped.num_envs
    episode_length_steps = env.unwrapped.max_episode_length

    # Used by --break_diag (hold-break cause) and by the wide-DR Δv assertion below:
    # best-effort per-env "did push_robot fire this step" detection via the EventManager's own interval
    # countdown (no public API for this; `_interval_term_time_left` only ever DECREASES by dt each step
    # except when its term just fired and got a fresh resampled interval, so "value went UP" is an
    # unambiguous fire signal). Looked up whenever --break_diag is on OR --dr on (wide) needs the live-Δv
    # check; only used for diagnosis/assertion, never for gating -- if Isaac Lab's internals change shape
    # this degrades to "unknown"/skipped, not a crash.
    push_term_idx = None
    measure_push_dv = has_getup and (args_cli.break_diag or args_cli.dr == "on")
    if measure_push_dv:
        try:
            push_term_idx = raw_env.event_manager._mode_term_names["interval"].index("push_robot")
        except (AttributeError, KeyError, ValueError):
            push_term_idx = None
            print("[getup-eval] NOTE: could not locate the push_robot interval-mode event term "
                  "(push-timing correlation / live-Δv check will report as unknown/skipped).")

    results: dict[str, list[dict]] = {}
    max_push_dv_xy_overall = 0.0
    n_pushes_observed_overall = 0
    for category in categories:
        expected_idx = gc.set_category_live(raw_env, category) if has_getup else None
        n_target = args_cli.num_episodes
        completed, push_dv_stats = _run_category(
            env=env,
            raw_env=raw_env,
            policy=policy,
            category=category,
            expected_category_idx=expected_idx,
            n_episodes_target=n_target,
            num_envs=num_envs,
            num_joints=num_joints,
            elbow_wrist_idx=elbow_wrist_idx,
            rated_torque=rated_torque,
            effort_limits=effort_limits,
            contact_sensor=contact_sensor,
            foot_body_mask=foot_body_mask,
            collision_spheres=collision_spheres,
            terrain_height_fn=terrain_height_fn,
            sensor_to_robot_idx=sensor_to_robot_idx,
            step_dt=step_dt,
            episode_length_steps=episode_length_steps,
            has_getup=has_getup,
            impact_warmup_s=args_cli.impact_warmup_s,
            break_diag=args_cli.break_diag,
            push_term_idx=push_term_idx,
        )
        results[category] = completed
        print(f"[getup-eval] category={category}: {len(completed)} episodes collected.", flush=True)
        n_pushes_observed_overall += push_dv_stats["n_pushes_seen"]
        if push_dv_stats["max_push_dv_xy"] is not None:
            max_push_dv_xy_overall = max(max_push_dv_xy_overall, push_dv_stats["max_push_dv_xy"])

    # Permanent assertion (not just a print) for --dr on (wide) -- if the live params passed
    # the earlier post-gym.make check but the ACTUAL simulated push magnitude never gets anywhere near what
    # ±0.5 m/s implies, something is still wrong downstream of the cfg (e.g. the physics call itself, or a
    # second copy of the term). 0.15 m/s is a conservative floor (well below the ~0.3-0.5 m/s a ±0.5 m/s
    # uniform range should produce over enough pushes, but comfortably above what a ±0.1 m/s narrow range
    # alone could ever produce) -- only checked when enough pushes were actually observed to be a meaningful
    # sample (episodes/category x num_envs x ~15s at a 3-8s interval should give plenty).
    if has_getup and args_cli.dr == "on" and push_term_idx is not None:
        if n_pushes_observed_overall < 5:
            print(f"[getup-eval] NOTE: --dr on (wide) but only {n_pushes_observed_overall} push_robot firings "
                  "were observed across the whole run -- too few to meaningfully sanity-check the live Δv "
                  "magnitude; skipping that assertion (the live-param check right after gym.make already ran).")
        elif max_push_dv_xy_overall < 0.15:
            raise RuntimeError(
                f"[getup-eval] --dr on (wide) assert failed: observed {n_pushes_observed_overall} push_robot "
                f"firings but the largest root-velocity jump any of them produced was only "
                f"{max_push_dv_xy_overall:.3f} m/s -- far below what a ±0.5 m/s uniform velocity_range should "
                "produce. The live-param check passed (cfg says wide), but the actual physics doesn't show "
                "it; do not trust this run's G3 numbers until this is root-caused."
            )
        else:
            print(f"[getup-eval] DR-wide live-Δv check: PASSED -- {n_pushes_observed_overall} push_robot "
                  f"firings observed, max |Δv_xy| = {max_push_dv_xy_overall:.3f} m/s across the whole run.")

    run_params = {
        "task": args_cli.task,
        "checkpoint": resume_path,
        "effort_scale": args_cli.effort_scale,
        "effort_scale_resolved": effort_scale_value,
        "effort_contract_source": effort_source,
        "effort_bound_scale_applied": (checkpoint_contract["bound_scale"] if trained_contract_available else None),
        "motor_strength": motor_strength_value,
        "applied_clip_nm_by_group": applied_clip_by_group,
        "beta": expected_beta,
        "live_beta": live_beta,
        "live_bound_scale": live_bound_scale,
        "checkpoint_trained_beta": (checkpoint_contract.get("beta") if checkpoint_contract else None),
        "dr": args_cli.dr,
        "dr_terms_touched": dr_touched,
        "dr_push_n_observed": n_pushes_observed_overall if push_term_idx is not None else None,
        "dr_push_max_dv_xy_mps": max_push_dv_xy_overall if push_term_idx is not None else None,
        "terrain": args_cli.terrain,
        "terrain_note": terrain_note,
        "num_envs": num_envs,
        "num_episodes_per_category": args_cli.num_episodes,
        "has_getup_env": has_getup,
        "step_dt_s": step_dt,
        "impact_warmup_s": args_cli.impact_warmup_s,
    }
    summary = _summarize(results, run_params, args_cli)
    _sanity_check_action_metrics(summary)
    os.makedirs(args_cli.output_dir, exist_ok=True)
    json_path = os.path.join(args_cli.output_dir, "metrics.json")
    md_path = os.path.join(args_cli.output_dir, "metrics.md")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    with open(md_path, "w") as f:
        f.write(_to_markdown(summary))
    print(f"[getup-eval] Wrote {json_path} and {md_path}", flush=True)

    # Skip env.close()/simulation_app.close(): headless Isaac Sim is known to hang there while holding the
    # GPU. The JSON/markdown outputs are already flushed to disk above.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def _summarize_clip_by_group(effort_limits_row: torch.Tensor, joint_names: list, elbow_wrist_idx: list) -> dict:
    """Per-joint-group summary (min/max/mean, in Nm) of the LIVE actuator clip, split into elbow/wrist
    (the joints G2's saturation gate cares about) vs everything else. `effort_limits_row` is one env's
    `[num_joints]` slice of `_actuator_effort_limits()`, read AFTER the motor_strength/bound application
    above so it reflects what the policy actually ran under, not what was merely requested."""
    ew_set = set(elbow_wrist_idx)
    other_idx = [i for i in range(len(joint_names)) if i not in ew_set]

    def _stats(idx: list) -> dict | None:
        if not idx:
            return None
        vals = effort_limits_row[idx].tolist()
        return {"min": min(vals), "max": max(vals), "mean": sum(vals) / len(vals), "n_joints": len(vals)}

    return {"elbow_wrist": _stats(elbow_wrist_idx), "other": _stats(other_idx)}


def _actuator_effort_limits(robot) -> torch.Tensor:
    """Per-joint effort limit actually enforced on `applied_torque`.

    `robot.data.joint_effort_limits` is PhysX's own raw DOF max-force (`root_physx_view.get_dof_max_forces()`),
    which for an explicit software actuator (e.g. `DelayedPDActuatorCfg`, used by the walking/AMP robot cfg
    and presumably the get-up one too) can be left at PhysX's own default rather than the actuator's configured
    `effort_limit` -- observed empirically: it produced tau_hat == 0.0 for every step against a
    real checkpoint even though `applied_torque` and the derived energy metric were clearly nonzero. Each
    `ActuatorBase` instance carries the real per-env-per-joint clip in `.effort_limit`; assemble those into a
    full `[num_envs, num_joints]` tensor instead, falling back to the PhysX value for any joint not covered by
    an actuator (shouldn't happen, but safer than crashing).
    """
    limits = robot.data.joint_effort_limits.clone()
    for actuator in robot.actuators.values():
        idx = actuator.joint_indices
        if isinstance(idx, slice):
            idx = list(range(robot.num_joints))
        limits[:, idx] = actuator.effort_limit
    return limits


def _find_all_body_contact_sensor(raw_env: "ManagerBasedRLEnv"):
    """Best-effort: find the ContactSensor covering the most bodies (the get-up env has one unfiltered contact
    sensor on all bodies). The sensor's scene name isn't part of a stable interface, so this picks the sensor
    with the largest body count among `env.scene.sensors` rather than assuming a specific name.
    """
    try:
        from isaaclab.sensors import ContactSensor
    except Exception:
        return None
    best = None
    best_n = -1
    for sensor in raw_env.scene.sensors.values():
        if isinstance(sensor, ContactSensor):
            n = len(sensor.body_names)
            if n > best_n:
                best, best_n = sensor, n
    if best is None:
        print("[getup-eval] WARN: no ContactSensor found in the scene; contact-force metrics will be unavailable.")
    return best


def _run_category(
    *,
    env,
    raw_env,
    policy,
    category: str,
    expected_category_idx: int | None,
    n_episodes_target: int,
    num_envs: int,
    num_joints: int,
    elbow_wrist_idx: list[int],
    rated_torque: torch.Tensor,
    effort_limits: torch.Tensor,
    contact_sensor,
    foot_body_mask,
    step_dt: float,
    episode_length_steps: int,
    has_getup: bool,
    impact_warmup_s: float = 0.1,
    break_diag: bool = False,
    push_term_idx: int | None = None,
    collision_spheres=None,
    terrain_height_fn=None,
    sensor_to_robot_idx=None,
) -> list[dict]:
    device = env.unwrapped.device
    impact_warmup_steps = max(0, round(impact_warmup_s / step_dt))
    # Wrapped in inference_mode like every step below: env.step() marks tensors it touches as "inference
    # tensors", and a later env.reset() call *outside* inference_mode on one of those same tensors raises
    # "Inplace update to inference tensor outside InferenceMode" once a previous category's steps have run
    # (observed with the get-up env).
    with torch.inference_mode():
        obs, _ = env.reset()

    if expected_category_idx is not None:
        gc.verify_category(raw_env, expected_category_idx, category)

    # Jitter/action-rate must be computed on the action the firmware actually sees -- clipped and
    # LPF-filtered -- not the raw policy output. For get-up that's the action term's own `filtered_actions`
    # buffer (= the "actions" observation); there is no LPF
    # for a generic (non-getup) task, so fall back to clipping the raw action to [-1, 1] (what
    # RslRlVecEnvWrapper.step applies before the *underlying* env ever sees it).
    joint_pos_action_term = None
    if has_getup:
        try:
            joint_pos_action_term = raw_env.action_manager.get_term("joint_pos")
        except Exception:  # noqa: BLE001 - defensive: fall back to the generic path if the term name differs
            joint_pos_action_term = None

    acc = _new_accumulators(num_envs, num_joints, device)
    prev_action = torch.zeros(num_envs, 0, device=device)
    completed: list[dict] = []

    getup_state = getattr(raw_env, "getup_state", None) if has_getup else None
    max_stand_timer = torch.zeros(num_envs, device=device)
    latched_success = torch.zeros(num_envs, dtype=torch.bool, device=device)
    latched_ttf = torch.full((num_envs,), float("nan"), device=device)

    # `is_standing`'s `lin_vel < 0.3 m/s` condition is ill-posed under DR, because
    # `push_robot` directly SETS base velocity up to 0.5 m/s -- that condition then fails by construction
    # right after a push, regardless of anything the policy does. G3 (DR on) is rescored against a separate,
    # client-side "standing" latch that drops the velocity check (height/tilt/feet only) but is otherwise
    # structurally identical to the env's own success/stand_timer/time_to_stand bookkeeping (mdp/tracker.py);
    # G1 (no DR) is unaffected and still uses the env's own `success`/`stand_timer_s` (with velocity) below.
    # Computed unconditionally (not just under --break_diag) since it now feeds the G3 gate itself.
    g3_stand_timer = torch.zeros(num_envs, device=device)
    max_stand_timer_g3 = torch.zeros(num_envs, device=device)
    latched_success_g3 = torch.zeros(num_envs, dtype=torch.bool, device=device)
    latched_ttf_g3 = torch.full((num_envs,), float("nan"), device=device)

    # Alternative G3 criterion under pushes, `success_g3b` -- stood within 6s (the G3
    # criterion: height/tilt/feet, no velocity term, same 1s latch as success_g3 above) AND never "fell"
    # afterwards for the rest of the episode, where "fell" = height<0.50m OR tilt>0.35rad *continuously* for
    # > 0.5s (a brief stumble that recovers within 0.5s is allowed, unlike success_g3's hold timer, which
    # resets on ANY momentary violation, however brief). Reporting only -- does not feed any gate.
    fall_timer_g3b = torch.zeros(num_envs, device=device)
    fell_after_standing_g3b = torch.zeros(num_envs, dtype=torch.bool, device=device)

    # Hold-break cause diagnostics (--break_diag only): `prev_stand_timer` lets us detect
    # the exact step `is_standing` flips True->False after having held for a while (tracker.py resets
    # `stand_timer_s` to exactly 0.0 on that transition, never in between -- see mdp/tracker.py); `steps_since_push`
    # (in *steps*, converted to seconds at use) tracks how recently `push_robot` last fired per env, so a
    # break can be checked against "within 0.5s of a push"; `break_info[i]` holds the first post-1s-success
    # break's diagnosis for env i's in-flight episode (None until one is found, reset per episode).
    prev_stand_timer = torch.zeros(num_envs, device=device)
    steps_since_push = torch.full((num_envs,), 1.0e9, device=device)
    break_info: list = [None] * num_envs

    # Cap total steps so a checkpoint that never terminates can't hang the eval forever.
    max_batches = 4 * max(1, -(-n_episodes_target // num_envs))
    max_steps = episode_length_steps * max_batches + episode_length_steps

    max_push_dv_xy = torch.zeros(num_envs, device=device)
    n_pushes_seen = 0

    step = 0
    while len(completed) < n_episodes_target and step < max_steps:
        push_before = (
            raw_env.event_manager._interval_term_time_left[push_term_idx].clone()
            if push_term_idx is not None
            else None
        )
        # Verify the push is REALLY landing with the intended magnitude, not just that the
        # cfg object says so (the wide-DR live-param assert in main() covers the cfg side; this covers the
        # physics side). Captured before every step whenever we're tracking pushes at all, cheap ([N,2]).
        root_vel_xy_before = raw_env.scene["robot"].data.root_lin_vel_w[:, :2].clone() if push_before is not None else None
        with torch.inference_mode():
            action = policy(obs)
            obs, _, dones, extras = env.step(action)
        if push_before is not None:
            push_after = raw_env.event_manager._interval_term_time_left[push_term_idx]
            pushed_now = push_after > push_before  # only true the step a term fires and resamples (see above)
            steps_since_push = torch.where(pushed_now, torch.zeros_like(steps_since_push), steps_since_push + 1.0)
            root_vel_xy_after = raw_env.scene["robot"].data.root_lin_vel_w[:, :2]
            dv_xy = (root_vel_xy_after - root_vel_xy_before).norm(dim=-1)
            max_push_dv_xy = torch.where(pushed_now, torch.maximum(max_push_dv_xy, dv_xy), max_push_dv_xy)
            n_pushes_seen += int(pushed_now.sum().item())

        # `filtered_actions` returns the action term's own persistent buffer (`self._filtered`),
        # not a fresh tensor -- the SAME object every step, mutated in place inside env.step(). Assigning
        # `prev_action = effective_action` without cloning made `prev_action` and next step's
        # `effective_action` alias the same memory, so `action - prev_action` would be identically zero every
        # step and action_rate would silently read 0.000 in every category. Clone unconditionally here, once, so every downstream use (history, prev_action) owns an
        # independent snapshot regardless of whether the source was a persistent buffer or a fresh tensor.
        effective_action = (joint_pos_action_term.filtered_actions if joint_pos_action_term is not None else action.clamp(-1.0, 1.0)).clone()
        if prev_action.shape[-1] == 0:
            prev_action = torch.zeros_like(effective_action)
        acc["action_hist"].appendleft(effective_action)
        while len(acc["action_hist"]) > 4:
            acc["action_hist"].pop()

        _accumulate_step(acc, raw_env, effective_action, prev_action, elbow_wrist_idx, rated_torque, effort_limits, contact_sensor, foot_body_mask, step_dt, num_joints, getup_state, impact_warmup_steps, collision_spheres, terrain_height_fn, sensor_to_robot_idx)
        prev_action = effective_action

        if getup_state is not None:
            success_now = getattr(getup_state, "success", None)
            stand_timer_now = getattr(getup_state, "stand_timer_s", None)
            ttf_now = getattr(getup_state, "time_to_stand_s", None)
            if stand_timer_now is not None:
                max_stand_timer = torch.maximum(max_stand_timer, stand_timer_now)
            if success_now is not None:
                newly = success_now & (~latched_success)
                latched_success = latched_success | success_now
                if ttf_now is not None:
                    latched_ttf = torch.where(newly, ttf_now, latched_ttf)

            # Recompute height/tilt/feet every step (no velocity check) to drive the G3-only
            # `success_g3` latch, and reuse the same three sub-conditions for --break_diag's cause breakdown.
            height = getup_mdp.pelvis_height(raw_env)
            tilt = getup_mdp.torso_tilt(raw_env)
            feet_ok_all = getup_mdp.feet_in_contact(raw_env).all(dim=1)
            height_ok = height >= getup_mdp.STAND_PELVIS_HEIGHT
            tilt_ok = tilt <= getup_mdp.STAND_MAX_TILT
            standing_no_vel = height_ok & tilt_ok & feet_ok_all
            active = getattr(getup_state, "policy_active", None)
            if active is not None:
                standing_no_vel = standing_no_vel & active
            g3_stand_timer = torch.where(standing_no_vel, g3_stand_timer + step_dt, torch.zeros_like(g3_stand_timer))
            max_stand_timer_g3 = torch.maximum(max_stand_timer_g3, g3_stand_timer)
            newly_g3 = (g3_stand_timer >= gc.STANDING_HOLD_S - 1.0e-6) & (~latched_success_g3)
            latched_success_g3 = latched_success_g3 | newly_g3
            gs_step = getattr(getup_state, "step", None)
            gs_control_start = getattr(getup_state, "control_start_step", None)
            if gs_step is not None and gs_control_start is not None:
                t_ctrl_g3 = (gs_step - gs_control_start).clamp(min=0).float() * step_dt
                latched_ttf_g3 = torch.where(newly_g3, t_ctrl_g3 - g3_stand_timer, latched_ttf_g3)

            # success_g3b: "fell" = height<0.50m OR tilt>0.35rad (the same thresholds as
            # height_ok/tilt_ok above, just inverted), sustained continuously for > 0.5s -- a brief stumble
            # that recovers within 0.5s does NOT count, unlike success_g3's stand_timer (any momentary
            # violation resets it). Only latches "fell after standing" once the episode has already stood
            # (latched_success_g3); a fall-timer run before ever standing isn't meaningful here.
            fell_now = (~height_ok) | (~tilt_ok)
            fall_timer_g3b = torch.where(fell_now, fall_timer_g3b + step_dt, torch.zeros_like(fall_timer_g3b))
            sustained_fall_now = (fall_timer_g3b > 0.5 + 1.0e-6) & latched_success_g3
            fell_after_standing_g3b = fell_after_standing_g3b | sustained_fall_now

            # --break_diag only: a "break" is `is_standing` flipping True->False after having
            # held for a real stretch (> 0.05s, to ignore single-frame noise); only the FIRST such break
            # *after* this episode already reached the 1s success latch is recorded (that's specifically the
            # break that would turn a would-be G1 pass into a fail -- earlier breaks, before 1s was ever
            # reached, are "never stood up cleanly" rather than "hold broke").
            if break_diag and stand_timer_now is not None:
                broke = (prev_stand_timer > 0.05) & (stand_timer_now <= 1.0e-6)
                broke_ids = broke.nonzero(as_tuple=False).squeeze(-1).tolist()
                if broke_ids:
                    lin_vel = raw_env.scene["robot"].data.root_lin_vel_w.norm(dim=-1)
                    for i in broke_ids:
                        if not latched_success[i].item() or break_info[i] is not None:
                            continue  # not yet past the 1s bar, or we already recorded this episode's break
                        violated = []
                        if not bool(height_ok[i].item()):
                            violated.append("height")
                        if not bool(tilt_ok[i].item()):
                            violated.append("tilt")
                        if not bool(feet_ok_all[i].item()):
                            violated.append("feet")
                        if not (lin_vel[i].item() < getup_mdp.STAND_MAX_LIN_VEL):
                            violated.append("lin_vel")
                        push_recent_s = (
                            steps_since_push[i].item() * step_dt if push_term_idx is not None else None
                        )
                        break_info[i] = {
                            "step": step,
                            "break_time_s": step * step_dt,
                            "stand_timer_before_break_s": float(prev_stand_timer[i].item()),
                            "violated_conditions": violated or ["unknown (all 4 sub-conditions read OK at the "
                                                                  "recorded break step -- likely a one-step "
                                                                  "readback race; treat as inconclusive)"],
                            "push_time_since_s": push_recent_s,
                            "push_within_0_5s_before": (push_recent_s is not None and push_recent_s <= 0.5),
                        }
                prev_stand_timer = stand_timer_now.clone()

        done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
        if done_ids.numel() > 0:
            for i in done_ids.tolist():
                if len(completed) >= n_episodes_target:
                    break
                entry = _finalize_episode(
                    acc, i, elbow_wrist_idx, latched_success, latched_ttf, max_stand_timer, has_getup, step_dt,
                    latched_success_g3=latched_success_g3, latched_ttf_g3=latched_ttf_g3, max_stand_timer_g3=max_stand_timer_g3,
                    fell_after_standing_g3b=fell_after_standing_g3b,
                )
                if break_diag:
                    entry["g1_break_diag"] = break_info[i]
                completed.append(entry)
                _reset_episode_accumulators(acc, i, num_joints, device)
                max_stand_timer[i] = 0.0
                latched_success[i] = False
                latched_ttf[i] = float("nan")
                g3_stand_timer[i] = 0.0
                max_stand_timer_g3[i] = 0.0
                latched_success_g3[i] = False
                latched_ttf_g3[i] = float("nan")
                fall_timer_g3b[i] = 0.0
                fell_after_standing_g3b[i] = False
                break_info[i] = None
                prev_stand_timer[i] = 0.0
                steps_since_push[i] = 1.0e9

        step += 1

    if push_term_idx is not None:
        print(f"[getup-eval] category={category}: {n_pushes_seen} push_robot firings observed, "
              f"max |push Δv_xy| = {max_push_dv_xy.max().item():.3f} m/s "
              f"(sanity check: for the wide ±0.5 m/s range this should reach roughly "
              "0.3-0.5 m/s over enough pushes; a narrow-range-sized max here despite --dr on would mean the "
              "wide override isn't actually reaching the physics, matching the wide-DR live-param assert "
              "in main()).")

    return completed, {"n_pushes_seen": n_pushes_seen, "max_push_dv_xy": float(max_push_dv_xy.max().item()) if push_term_idx is not None else None}


def _new_accumulators(num_envs, num_joints, device):
    from collections import deque

    return {
        "peak_tau_hat": torch.zeros(num_envs, num_joints, device=device),
        "sat_steps": torch.zeros(num_envs, num_joints, device=device),
        "sat_any_steps": torch.zeros(num_envs, device=device),
        "n_steps": torch.zeros(num_envs, device=device),
        "ew_sat_streak": torch.zeros(num_envs, num_joints, device=device),
        "ew_sat_streak_max": torch.zeros(num_envs, device=device),
        "energy_j": torch.zeros(num_envs, device=device),
        "nonfoot_force_peak": torch.zeros(num_envs, device=device),
        "nonfoot_force_peak_step": torch.zeros(num_envs, device=device),
        "nonfoot_impulse": torch.zeros(num_envs, device=device),
        # Peak ground-only non-foot force split by whether the policy is already
        # in control (`getup_state.policy_active`) -- "limp" = the scripted pre-control fall phase, "active"
        # = after the policy takes over. Used by `g2_impact_vs_own_fall` (mid_fall only): does the policy's
        # OWN handling ever hit harder than the fall that put the robot on the ground in the first place.
        "ground_force_peak_limp": torch.zeros(num_envs, device=device),
        "ground_force_peak_active": torch.zeros(num_envs, device=device),
        "jitter_sq_sum": torch.zeros(num_envs, device=device),
        "action_rate_sum": torch.zeros(num_envs, device=device),
        "thermal_state": torch.zeros(num_envs, num_joints, device=device),
        "thermal_peak": torch.zeros(num_envs, device=device),
        "action_hist": deque(maxlen=4),
    }


def _reset_episode_accumulators(acc, i, num_joints, device):
    acc["peak_tau_hat"][i] = 0.0
    acc["sat_steps"][i] = 0.0
    acc["sat_any_steps"][i] = 0.0
    acc["n_steps"][i] = 0.0
    acc["ew_sat_streak"][i] = 0.0
    acc["ew_sat_streak_max"][i] = 0.0
    # note: "ew_sat_streak" is per-joint (shape [num_envs, num_joints]), see _accumulate_step
    acc["energy_j"][i] = 0.0
    acc["nonfoot_force_peak"][i] = 0.0
    acc["nonfoot_force_peak_step"][i] = 0.0
    acc["nonfoot_impulse"][i] = 0.0
    acc["ground_force_peak_limp"][i] = 0.0
    acc["ground_force_peak_active"][i] = 0.0
    acc["jitter_sq_sum"][i] = 0.0
    acc["action_rate_sum"][i] = 0.0
    acc["thermal_state"][i] = 0.0
    acc["thermal_peak"][i] = 0.0


def _accumulate_step(acc, raw_env, action, prev_action, elbow_wrist_idx, rated_torque, effort_limits, contact_sensor, foot_body_mask, step_dt, num_joints, getup_state=None, impact_warmup_steps=0, collision_spheres=None, terrain_height_fn=None, sensor_to_robot_idx=None):
    robot = raw_env.scene["robot"]
    torque = robot.data.applied_torque
    joint_vel = robot.data.joint_vel

    tau_hat = (torque.abs() / effort_limits.clamp_min(1e-6)).clamp(max=10.0)
    acc["peak_tau_hat"] = torch.maximum(acc["peak_tau_hat"], tau_hat)
    sat_now = tau_hat > gc.G2_TAU_HAT_SAT_THRESHOLD  # [N, J]
    acc["sat_steps"] += sat_now.float()
    # G2's "> 0.9 tau_max for < 5% of steps" is a per-STEP criterion over
    # ANY joint, not a mean over joint-steps -- `sat_steps.sum()/(n*num_joints)` lets one joint saturated
    # 100% of the time read as ~4.3% (1/23) and silently pass. Track the any-joint-per-step fraction
    # separately; `_finalize_episode`/`_aggregate` report both this and the per-joint max fraction.
    acc["sat_any_steps"] += sat_now.any(dim=-1).float()
    acc["n_steps"] += 1.0

    if elbow_wrist_idx:
        # Track each elbow/wrist joint's own consecutive-saturation streak independently and report the
        # longest any single joint ever sustained. G2's concern ("elbow and wrist never saturated for > 0.2 s")
        # is one specific actuator staying pinned; an `.any(dim=-1)` union across the 4 joints would keep the
        # streak running as long as *some* joint was saturated, which with constant policy noise can read as
        # nearly the whole episode even when no single joint stays pinned.
        ew_cols = elbow_wrist_idx
        ew_saturated = tau_hat[:, ew_cols] > 0.999  # [N, K], per joint
        acc["ew_sat_streak"][:, ew_cols] = torch.where(
            ew_saturated, acc["ew_sat_streak"][:, ew_cols] + step_dt, torch.zeros_like(acc["ew_sat_streak"][:, ew_cols])
        )
        acc["ew_sat_streak_max"] = torch.maximum(acc["ew_sat_streak_max"], acc["ew_sat_streak"][:, ew_cols].max(dim=-1).values)

    acc["energy_j"] += (torque.abs() * joint_vel.abs()).sum(dim=-1) * step_dt

    # Thermal proxy: prefer the env's real per-joint EMA (env.getup_state.thermal, the same signal the reward
    # term uses, matching its own rated-torque table) when available; otherwise fall back to a local leaky I^2t
    # integrator against a guessed rated-torque table (see _common.py for that fallback's caveats).
    thermal = getattr(getup_state, "thermal", None) if getup_state is not None else None
    if thermal is not None and thermal.shape == acc["thermal_state"].shape:
        acc["thermal_peak"] = torch.maximum(acc["thermal_peak"], thermal.max(dim=-1).values)
    else:
        tau_over_rated = (torque.abs() / rated_torque.clamp_min(1e-6)).clamp(max=10.0)
        decay = float(np.exp(-step_dt / gc.THERMAL_TIME_CONSTANT_S))
        acc["thermal_state"] = decay * acc["thermal_state"] + (1.0 - decay) * tau_over_rated**2
        acc["thermal_peak"] = torch.maximum(acc["thermal_peak"], acc["thermal_state"].max(dim=-1).values)

    if contact_sensor is not None and foot_body_mask is not None:
        forces = gc.contact_force_norm(contact_sensor)  # [num_envs, num_bodies], substep-history max
        nonfoot = forces[:, ~foot_body_mask] if (~foot_body_mask).any() else forces
        # Ground-only impact metric -- a body's force counts only if its lowest collision
        # point is within 3cm of the local terrain (else it's self-contact, e.g.
        # pelvis_link<->hip_yaw_link shell contact during hip flexion; see _common.build_ground_contact_helper).
        if collision_spheres is not None:
            ground_mask = gc.ground_contact_mask_for_sensor(collision_spheres, terrain_height_fn, sensor_to_robot_idx, robot)
            ground_mask_nonfoot = ground_mask[:, ~foot_body_mask] if (~foot_body_mask).any() else ground_mask
            nonfoot = torch.where(ground_mask_nonfoot, nonfoot, torch.zeros_like(nonfoot))
        peak = nonfoot.max(dim=-1).values
        debug_threshold = os.environ.get("W5_DEBUG_CONTACT")
        if debug_threshold:
            # Diagnose which body a large non-foot impact reading actually comes from. Checks EVERY env (a
            # spike is an occasional, per-episode event, so an env0-only check easily misses it) and takes the
            # threshold from the env var's own value (any truthy non-numeric string like "1" falls back to
            # 1000.0), so a small sample can still catch a real spike in whichever env it lands.
            try:
                threshold_n = float(debug_threshold)
            except ValueError:
                threshold_n = 1000.0
            over_ids = (peak > threshold_n).nonzero(as_tuple=False).squeeze(-1).tolist()
            nonfoot_names = [n for n, is_foot in zip(contact_sensor.body_names, foot_body_mask.tolist()) if not is_foot]
            if over_ids:
                # The all-body sensor (`body_contact`, getup_env_cfg.py) has no
                # `filter_prim_paths_expr`, so `net_forces_w`/`net_forces_w_history` is the body's AGGREGATE
                # contact force from EVERYTHING it touches -- the ground AND self-contact (an arm resting on
                # the torso, a knee against the opposite shank, etc.) are indistinguishable at the sensor
                # level. Report each flagged body's own height above the local ground alongside its force, so
                # a human can tell which it is -- same geometric-inference idea as
                # `mdp/rewards.py:arm_self_contact` (body height above ground vs. its `ground_margin=0.03`),
                # but coarser: that reward term subtracts each arm link's own collision-sphere radius from a
                # precomputed sample-point table to get the exact lowest collision point; no such table exists
                # for non-arm bodies here, so this uses the body's own ORIGIN height above the local ground
                # (flat-terrain approximation: `env_origins[:, 2]`, matching `pelvis_height`'s own flat-terrain
                # branch) -- a few cm too generous for a thick link, but sufficient to tell "at the ground"
                # from "a meter up on a standing robot" unambiguously, which is the actual question here.
                robot_body_names = list(robot.body_names)
                ground_z = raw_env.scene.env_origins[:, 2]
            for env_id in over_ids:
                vals = nonfoot[env_id].tolist()
                top = sorted(zip(nonfoot_names, vals), key=lambda kv: -kv[1])[:5]
                top_annotated = []
                for name, force in top:
                    try:
                        b_idx = robot_body_names.index(name)
                        height_above_ground = float(robot.data.body_link_pos_w[env_id, b_idx, 2].item() - ground_z[env_id].item())
                    except ValueError:
                        height_above_ground = None
                    top_annotated.append((name, round(force, 1), None if height_above_ground is None else round(height_above_ground, 3)))
                print(f"[getup-eval] DEBUG_CONTACT env{env_id} peak={peak[env_id].item():.1f}N "
                      f"top5(name, force_N, height_above_ground_m)={top_annotated}", flush=True)
        # A "standing" category run showed a 25981 N non-foot impact peak with 100% success -- implausible for
        # a policy that's already upright and barely moving: a reset/spawn artifact (e.g. transient
        # penetration resolving in the first physics step(s) after the fallen-state cache write, or the
        # mid_fall limp-to-active handoff), not a real mid-episode impact. Exclude the first
        # `impact_warmup_steps` steps (default 0.1s, ~5 steps @ 50Hz) after each reset from the non-foot
        # impact metrics, and separately record
        # *when* (which step) each episode's peak occurred (`nonfoot_force_peak_step`) so a reviewer can
        # directly check whether remaining spikes still cluster near t=0 (reported in the markdown/JSON).
        past_warmup = acc["n_steps"] >= impact_warmup_steps  # n_steps already incremented above this step
        peak_gated = torch.where(past_warmup, peak, torch.zeros_like(peak))
        newly_max = peak_gated > acc["nonfoot_force_peak"]
        acc["nonfoot_force_peak_step"] = torch.where(newly_max, acc["n_steps"], acc["nonfoot_force_peak_step"])
        acc["nonfoot_force_peak"] = torch.maximum(acc["nonfoot_force_peak"], peak_gated)
        acc["nonfoot_impulse"] += peak_gated * step_dt

        # `g2_impact_vs_own_fall` -- deliberately NOT warmup-gated (the whole
        # point is to capture the limp-phase impact, which happens right at/near reset -- excluding it would
        # defeat the comparison). Uses the same ground-only `peak` as above (self-contact already excluded).
        active_now = getattr(getup_state, "policy_active", None)
        if active_now is not None:
            acc["ground_force_peak_limp"] = torch.maximum(acc["ground_force_peak_limp"], torch.where(~active_now, peak, torch.zeros_like(peak)))
            acc["ground_force_peak_active"] = torch.maximum(acc["ground_force_peak_active"], torch.where(active_now, peak, torch.zeros_like(peak)))

    delta = action - prev_action
    acc["action_rate_sum"] += delta.abs().mean(dim=-1)
    hist = acc["action_hist"]
    if len(hist) >= 4:
        third_diff = hist[0] - 3 * hist[1] + 3 * hist[2] - hist[3]
        acc["jitter_sq_sum"] += (third_diff**2).mean(dim=-1)


def _finalize_episode(
    acc, i, elbow_wrist_idx, latched_success, latched_ttf, max_stand_timer, has_getup, step_dt: float,
    latched_success_g3=None, latched_ttf_g3=None, max_stand_timer_g3=None, fell_after_standing_g3b=None,
) -> dict:
    n = max(float(acc["n_steps"][i].item()), 1.0)
    entry = {
        "success_1s": bool(latched_success[i].item()) if has_getup else None,
        "time_to_stand_s": (None if not has_getup or torch.isnan(latched_ttf[i]) else float(latched_ttf[i].item())),
        "success_g1": (
            bool(latched_success[i].item() and not torch.isnan(latched_ttf[i]) and latched_ttf[i].item() <= gc.G1_TIME_TO_STAND_MAX_S
                 and max_stand_timer[i].item() >= gc.G1_HOLD_S)
            if has_getup
            else None
        ),
        # success_g3 is the G3-only rescoring of the same "standing within 6s, held
        # >=5s" rule, using is_standing WITHOUT the lin_vel<0.3m/s condition (push_robot sets base velocity
        # up to 0.5 m/s directly, so that condition fails by construction under DR regardless of the policy).
        # G1 (no DR) is unaffected and keeps using success_g1 above.
        "success_g3": (
            bool(latched_success_g3[i].item() and not torch.isnan(latched_ttf_g3[i])
                 and latched_ttf_g3[i].item() <= gc.G1_TIME_TO_STAND_MAX_S and max_stand_timer_g3[i].item() >= gc.G1_HOLD_S)
            if has_getup and latched_success_g3 is not None
            else None
        ),
        "time_to_stand_g3_s": (
            None if not has_getup or latched_ttf_g3 is None or torch.isnan(latched_ttf_g3[i])
            else float(latched_ttf_g3[i].item())
        ),
        # success_g3c -- "handoff-ready", the criterion for `G3c` (the push-run
        # half of the G3 gate: G3 = G3a AND G3c). The G3 standing gate (no velocity term) held >=1s within
        # 6s -- i.e. success_g3's own 1s/6s latch, WITHOUT the additional 5s-continuous-hold requirement
        # (that requirement is G3a's job, checked with pushes disabled -- see _gate_checks).
        "success_g3c": (
            bool(latched_success_g3[i].item() and not torch.isnan(latched_ttf_g3[i])
                 and latched_ttf_g3[i].item() <= gc.G1_TIME_TO_STAND_MAX_S)
            if has_getup and latched_success_g3 is not None
            else None
        ),
        # success_g3b (alternative G3 criterion): stood within 6s (G3's 1s latch, no velocity term)
        # AND never "fell" (height<0.50m OR tilt>0.35rad, sustained > 0.5s) for the rest of the episode --
        # brief stumbles that recover within 0.5s are allowed, unlike success_g3's stand_timer (any momentary
        # violation resets it to 0). Reporting only -- does not feed any gate.
        "success_g3b": (
            bool(latched_success_g3[i].item() and not torch.isnan(latched_ttf_g3[i])
                 and latched_ttf_g3[i].item() <= gc.G1_TIME_TO_STAND_MAX_S and not fell_after_standing_g3b[i].item())
            if has_getup and latched_success_g3 is not None and fell_after_standing_g3b is not None
            else None
        ),
        "peak_tau_hat": float(acc["peak_tau_hat"][i].max().item()),
        "peak_tau_hat_per_joint": acc["peak_tau_hat"][i].tolist(),
        # G2's "> 0.9 tau_max for < 5% of steps" is a per-step, any-joint criterion, not a mean over
        # joint-steps (a mean would let one joint saturated 100% of the time read as ~1/23 = 4.3% and pass). Report both: the any-joint fraction (what the gate uses)
        # and the worst single joint's own fraction (for diagnosis).
        "pct_steps_tau_hat_gt_0_9_any_joint": float((acc["sat_any_steps"][i] / n).item()),
        "pct_steps_tau_hat_gt_0_9_per_joint_max": float((acc["sat_steps"][i] / n).max().item()),
        "elbow_wrist_sat_s_max": (float(acc["ew_sat_streak_max"][i].item()) if elbow_wrist_idx else None),
        "energy_j": float(acc["energy_j"][i].item()),
        "thermal_proxy_peak": float(acc["thermal_peak"][i].item()),
        "peak_nonfoot_force_n": float(acc["nonfoot_force_peak"][i].item()),
        "peak_nonfoot_force_time_s": float(acc["nonfoot_force_peak_step"][i].item()) * step_dt,
        "nonfoot_impulse_ns": float(acc["nonfoot_impulse"][i].item()),
        # Raw ground-only peaks split by limp (pre-control) vs active (post-
        # control) phase, for `g2_impact_vs_own_fall` (gated on `mid_fall` episodes only in `_gate_checks`;
        # reported informationally for every other category).
        "ground_force_peak_limp_n": float(acc["ground_force_peak_limp"][i].item()),
        "ground_force_peak_active_n": float(acc["ground_force_peak_active"][i].item()),
        "action_jitter_rms": float((acc["jitter_sq_sum"][i] / n).sqrt().item()),
        "action_rate_mean": float((acc["action_rate_sum"][i] / n).item()),
        "n_steps": int(n),
    }
    return entry


def _summarize(results: dict[str, list[dict]], run_params: dict, args_cli) -> dict:
    per_category = {}
    all_entries: list[dict] = []
    for category, entries in results.items():
        all_entries.extend(entries)
        per_category[category] = _aggregate(entries, run_params["has_getup_env"])

    overall = _aggregate(all_entries, run_params["has_getup_env"])

    gates = _gate_checks(per_category, overall, run_params, args_cli)

    return {
        "run_params": run_params,
        "overall": overall,
        "per_category": per_category,
        "gates": gates,
    }


def _sanity_check_action_metrics(summary: dict) -> None:
    """Catch a known failure signature: action_rate reading ~0 while jitter is clearly nonzero is a strong
    signal of an aliasing/frozen-buffer bug in the action-history accumulation (see the clone note in
    `_run_category`), not a real "the policy never moves" result -- jitter (a 3rd difference)
    can't be meaningfully nonzero if consecutive actions genuinely weren't changing. Runs on every eval so a
    regression of this class is loud immediately, not found later by a human eyeballing a table.
    """
    suspects = []
    for label, s in {**summary.get("per_category", {}), "overall": summary.get("overall", {})}.items():
        rate = s.get("action_rate_mean")
        jitter = s.get("action_jitter_rms_mean")
        if rate is not None and jitter is not None and rate < 1e-6 and jitter > 1e-4:
            suspects.append((label, rate, jitter))
    if suspects:
        msg = (
            "[getup-eval] SANITY CHECK FAILED: action_rate_mean is ~0 while action_jitter_rms_mean is clearly "
            f"nonzero for: {suspects}. This is the exact signature of an aliased/frozen action buffer (see "
            "the clone note in _run_category) -- action_rate should not be able to read ~0 if actions "
            "are visibly changing enough to jitter. Treat this run's action-rate numbers as untrustworthy."
        )
        print(msg, flush=True)
        summary["action_metrics_sanity_check"] = {"status": "FAILED", "suspects": suspects, "message": msg}
    else:
        summary["action_metrics_sanity_check"] = {"status": "ok"}


def _aggregate_break_diag(entries: list[dict]) -> dict:
    """Hold-break cause diagnostics: summarize `--break_diag`'s per-episode records
    across a category. Only meaningful for episodes that reached the 1s success latch but failed G1 -- that's
    specifically the "stood up, then the hold broke before G1's 5s bar" population this was built to explain.
    """
    fail_g1_after_1s = [e for e in entries if e.get("success_1s") and not e.get("success_g1")]
    with_break = [e for e in fail_g1_after_1s if e.get("g1_break_diag") is not None]
    no_break_recorded = len(fail_g1_after_1s) - len(with_break)
    violated_counts: dict[str, int] = {}
    push_within_0_5s = 0
    push_unknown = 0
    for e in with_break:
        d = e["g1_break_diag"]
        for cond in d["violated_conditions"]:
            violated_counts[cond] = violated_counts.get(cond, 0) + 1
        if d["push_time_since_s"] is None:
            push_unknown += 1
        elif d["push_within_0_5s_before"]:
            push_within_0_5s += 1
    return {
        "n_success_1s_fail_g1": len(fail_g1_after_1s),
        "n_with_recorded_break": len(with_break),
        # success_1s True + success_g1 False + no break ever recorded means stand_timer_s kept RE-CROSSING >0
        # without a hard reset-to-0 (e.g. still climbing when the episode ended, or hovering near the 1s mark
        # without ever fully re-falling) -- "ran out of time / wobbled", not "broke", genuinely different from
        # a hard break and worth telling apart.
        "n_no_break_recorded_ran_out_of_time_or_wobbled": no_break_recorded,
        "violated_condition_counts": violated_counts,
        "n_push_within_0_5s_before_break": push_within_0_5s,
        "n_push_unknown": push_unknown,
        "n_push_not_within_0_5s": len(with_break) - push_within_0_5s - push_unknown,
        "break_details": [
            {
                "break_time_s": e["g1_break_diag"]["break_time_s"],
                "stand_timer_before_break_s": e["g1_break_diag"]["stand_timer_before_break_s"],
                "violated_conditions": e["g1_break_diag"]["violated_conditions"],
                "push_time_since_s": e["g1_break_diag"]["push_time_since_s"],
            }
            for e in with_break
        ],
    }


def _aggregate(entries: list[dict], has_getup: bool) -> dict:
    if not entries:
        return {"n_episodes": 0}
    n = len(entries)
    out = {"n_episodes": n}
    if has_getup:
        successes = [e["success_1s"] for e in entries if e["success_1s"] is not None]
        g1_successes = [e["success_g1"] for e in entries if e["success_g1"] is not None]
        ttfs = [e["time_to_stand_s"] for e in entries if e["time_to_stand_s"] is not None]
        out["success_rate_1s"] = float(np.mean(successes)) if successes else None
        out["success_rate_g1"] = float(np.mean(g1_successes)) if g1_successes else None
        out["time_to_stand_p50_s"] = gc.percentile(ttfs, 50)
        out["time_to_stand_p90_s"] = gc.percentile(ttfs, 90)
        # G3's own rescoring, dropping the lin_vel condition (see _finalize_episode).
        g3_successes = [e["success_g3"] for e in entries if e.get("success_g3") is not None]
        ttfs_g3 = [e["time_to_stand_g3_s"] for e in entries if e.get("time_to_stand_g3_s") is not None]
        out["success_rate_g3"] = float(np.mean(g3_successes)) if g3_successes else None
        out["time_to_stand_g3_p50_s"] = gc.percentile(ttfs_g3, 50)
        # success_g3b (alternative G3 criterion) -- reporting only, does not feed any gate.
        g3b_successes = [e["success_g3b"] for e in entries if e.get("success_g3b") is not None]
        out["success_rate_g3b"] = float(np.mean(g3b_successes)) if g3b_successes else None
        # success_g3c -- feeds the G3c gate (push runs) in _gate_checks.
        g3c_successes = [e["success_g3c"] for e in entries if e.get("success_g3c") is not None]
        out["success_rate_g3c"] = float(np.mean(g3c_successes)) if g3c_successes else None
        if any("g1_break_diag" in e for e in entries):
            out["g1_break_diag"] = _aggregate_break_diag(entries)
    out["peak_tau_hat_max"] = max(e["peak_tau_hat"] for e in entries)
    # The any-joint fraction is what the G2 gate uses; the per-joint max fraction is diagnostic only.
    out["pct_steps_tau_hat_gt_0_9_any_joint_mean"] = float(np.mean([e["pct_steps_tau_hat_gt_0_9_any_joint"] for e in entries]))
    out["pct_steps_tau_hat_gt_0_9_per_joint_max"] = max(e["pct_steps_tau_hat_gt_0_9_per_joint_max"] for e in entries)
    ew_vals = [e["elbow_wrist_sat_s_max"] for e in entries if e["elbow_wrist_sat_s_max"] is not None]
    out["elbow_wrist_sat_s_max"] = max(ew_vals) if ew_vals else None
    out["energy_j_mean"] = float(np.mean([e["energy_j"] for e in entries]))
    out["thermal_proxy_peak_max"] = max(e["thermal_proxy_peak"] for e in entries)
    _nonfoot_peaks = [e["peak_nonfoot_force_n"] for e in entries]
    out["peak_nonfoot_force_n_max"] = max(_nonfoot_peaks)
    # The walking baseline's own number is the MEAN of per-fall peaks (not the max), and the comparison
    # "<= the walking policy's own falls" is mean-vs-mean, not mean-vs-max. Report mean AND p95 alongside the existing max, so all three are always
    # available for either the gate (mean) or a transparency check (p95/max) -- see `_gate_checks`.
    out["peak_nonfoot_force_n_mean"] = float(np.mean(_nonfoot_peaks))
    out["peak_nonfoot_force_n_p95"] = gc.percentile(_nonfoot_peaks, 95)
    # Report *when* the worst peak happened, so a reviewer can directly check whether it's a
    # near-reset artifact even after the impact_warmup_s exclusion (e.g. a spike right at the warmup boundary).
    _worst = max(entries, key=lambda e: e["peak_nonfoot_force_n"])
    out["peak_nonfoot_force_time_s_at_max"] = _worst["peak_nonfoot_force_time_s"]
    peak_times = [e["peak_nonfoot_force_time_s"] for e in entries if e["peak_nonfoot_force_n"] > 0]
    out["peak_nonfoot_force_time_s_median"] = gc.percentile(peak_times, 50) if peak_times else None
    out["nonfoot_impulse_ns_mean"] = float(np.mean([e["nonfoot_impulse_ns"] for e in entries]))
    out["action_jitter_rms_mean"] = float(np.mean([e["action_jitter_rms"] for e in entries]))
    out["action_rate_mean"] = float(np.mean([e["action_rate_mean"] for e in entries]))
    # g2_impact_vs_own_fall -- per episode, does the policy's own peak ground
    # impact (after control starts) stay at or below the peak ground impact of the limp fall that put it on
    # the ground in the first place. Gated on `mid_fall` only (in `_gate_checks`, by category key); computed
    # and reported for every category here, informationally for the rest (limp-phase impact is only really
    # meaningful for `mid_fall`, where there's a real scripted fall before control starts).
    limp_peaks = [e["ground_force_peak_limp_n"] for e in entries]
    active_peaks = [e["ground_force_peak_active_n"] for e in entries]
    out["ground_force_peak_limp_n_mean"] = float(np.mean(limp_peaks))
    out["ground_force_peak_active_n_mean"] = float(np.mean(active_peaks))
    per_episode_pass = [a <= l for a, l in zip(active_peaks, limp_peaks)]
    out["g2_impact_vs_own_fall_pass_rate"] = float(np.mean(per_episode_pass)) if per_episode_pass else None
    return out


def _is_g1_motor_strength_ok(run_params: dict) -> bool:
    """G1's "1.0x effort" regime: G1 keys off the actuator
    CLIP (motor strength) being 1.0 -- "no assist, no DR, whatever the [action] bound". Since `main()`
    applies `--motor_strength` as an ABSOLUTE clip decoupled from the trained bound,
    when it was passed explicitly we check IT directly and deliberately ignore beta/bound (a checkpoint can
    be at a Stage-B bound of 0.9 and still pass G1 at full clip). When `--motor_strength` wasn't passed at
    all, there's no explicit clip to check, so fall back to a nominal-bound heuristic (robust to
    `--effort_scale trained`): the checkpoint's own resolved beta/bound, or the resolved
    preset string, must be nominal, since that's the only other way the live clip ends up at 1.0.
    """
    motor_strength = run_params.get("motor_strength")
    if motor_strength is not None:
        resolved = getup_mdp.resolve_effort_scale(motor_strength) if isinstance(motor_strength, str) else motor_strength
        if isinstance(resolved, dict):
            return all(abs(v - 1.0) < 1e-6 for v in resolved.values())
        return abs(float(resolved) - 1.0) < 1e-6
    if abs((run_params.get("beta") or 0.0) - 1.0) > 1e-6:
        return False
    bound = run_params.get("effort_bound_scale_applied")
    if bound is not None:
        return all(abs(v - 1.0) < 1e-6 for v in bound)
    # No explicit trained-contract bound recorded: either an explicit preset (checked by the resolved value
    # below) or the Play-cfg-default fallback, which getup_env_cfg.py fixes at nominal (1.0).
    resolved = run_params.get("effort_scale_resolved")
    return resolved in (None, "nominal", 1.0)


def _gate_checks(per_category: dict, overall: dict, run_params: dict, args_cli) -> dict:
    has_getup = run_params["has_getup_env"]
    gates: dict = {}

    is_g1_regime = _is_g1_motor_strength_ok(run_params) and run_params["dr"] == "off"
    if not has_getup:
        gates["G1"] = {"status": "not_applicable", "reason": "no getup env / no success signal available"}
    elif not is_g1_regime:
        gates["G1"] = {"status": "not_applicable", "reason": "G1 requires zero-assist, 1.0x effort, no DR; rerun with those settings"}
    else:
        overall_ok = (overall.get("success_rate_g1") or 0.0) >= gc.G1_OVERALL_MIN
        per_cat_ok = all((s.get("success_rate_g1") or 0.0) >= gc.G1_PER_CATEGORY_MIN for s in per_category.values())
        gates["G1"] = {
            "status": "pass" if (overall_ok and per_cat_ok) else "fail",
            "overall_success_rate_g1": overall.get("success_rate_g1"),
            "overall_threshold": gc.G1_OVERALL_MIN,
            "per_category_min": min((s.get("success_rate_g1") or 0.0) for s in per_category.values()) if per_category else None,
            "per_category_threshold": gc.G1_PER_CATEGORY_MIN,
        }

    # The gate uses the any-joint-per-step fraction, not a mean over joint-steps.
    sat_frac = overall.get("pct_steps_tau_hat_gt_0_9_any_joint_mean")
    sat_frac_per_joint_max = overall.get("pct_steps_tau_hat_gt_0_9_per_joint_max")
    ew_sat = overall.get("elbow_wrist_sat_s_max")
    g2_checks = {
        "tau_hat_sat_frac": {
            "value": sat_frac, "threshold": gc.G2_TAU_HAT_SAT_STEP_FRAC_MAX,
            "pass": (sat_frac is not None and sat_frac < gc.G2_TAU_HAT_SAT_STEP_FRAC_MAX),
            "per_joint_max_frac": sat_frac_per_joint_max,
        },
        "elbow_wrist_sat_s": {"value": ew_sat, "threshold": gc.G2_ELBOW_WRIST_SAT_MAX_S, "pass": (ew_sat is not None and ew_sat <= gc.G2_ELBOW_WRIST_SAT_MAX_S)},
    }
    thermal_peak = overall.get("thermal_proxy_peak_max")
    # Default threshold 1.0 (the rated-torque EMA at its own rated value); overridable. Note whether the
    # value came from the env's real signal or the local fallback (see _accumulate_step) -- material to how much to trust a borderline reading.
    thermal_threshold = args_cli.thermal_threshold if args_cli.thermal_threshold is not None else gc.G2_THERMAL_PROXY_PEAK_MAX
    g2_checks["thermal_proxy"] = {
        "value": thermal_peak,
        "threshold": thermal_threshold,
        "pass": (thermal_peak is not None and thermal_peak <= thermal_threshold),
        "note": "certifies a short-overload EMA against rated torque, not a winding-temperature model; "
        "value is the env's getup_state.thermal when available, else a local approximation (see _common.py)",
    }
    # With --walking_baseline_provisional, the walking-baseline jitter check is still computed and reported,
    # but marked "provisional" and excluded from what decides G2's overall pass/fail (see hard_checks below).
    provisional = args_cli.walking_baseline_provisional
    if args_cli.walking_jitter_baseline is not None and args_cli.walking_jitter_baseline > 0:
        jitter = overall.get("action_jitter_rms_mean")
        limit = args_cli.walking_jitter_baseline * gc.G2_ACTION_JITTER_RATIO_MAX
        g2_checks["action_jitter"] = {
            "value": jitter, "threshold": limit, "pass": (jitter is not None and jitter <= limit),
            "walking_baseline": args_cli.walking_jitter_baseline, "walking_baseline_provenance": gc.WALKING_BASELINE_PROVENANCE,
            "provisional": provisional,
        }
    else:
        g2_checks["action_jitter"] = {"status": "skipped", "reason": "--walking_jitter_baseline <= 0"}
    # The walking-baseline "<= the walking policy's own falls" comparison is kept ONLY as an informational
    # line -- G2's pass/fail uses the like-for-like `g2_impact_vs_own_fall` below instead. Absolute hardware
    # load limits are a separate hardware sign-off item (HARDWARE_TEST_PROTOCOL), not checked by this script.
    if args_cli.walking_impact_baseline_n is not None and args_cli.walking_impact_baseline_n > 0:
        peak_force_mean = overall.get("peak_nonfoot_force_n_mean")
        g2_checks["peak_nonfoot_impact_vs_walking_baseline"] = {
            "value": peak_force_mean, "threshold": args_cli.walking_impact_baseline_n,
            "pass": (peak_force_mean is not None and peak_force_mean <= args_cli.walking_impact_baseline_n),
            "value_p95": overall.get("peak_nonfoot_force_n_p95"),
            "value_max": overall.get("peak_nonfoot_force_n_max"),
            "walking_baseline_provenance": gc.WALKING_BASELINE_PROVENANCE,
            "informational": True,
        }
    else:
        g2_checks["peak_nonfoot_impact_vs_walking_baseline"] = {"status": "skipped", "reason": "--walking_impact_baseline_n <= 0"}
    # The counted ground-impact check -- in mid_fall episodes, does the policy's own peak ground
    # impact (after control starts) stay at or below the peak ground impact of the limp fall that put the
    # robot on the ground in the first place, per episode, in >= 90% of mid_fall episodes. Non-mid_fall
    # categories' ground-impact numbers (`ground_force_peak_limp_n_mean`/`_active_n_mean`, in `per_category`)
    # are informational only -- the limp-phase comparison is only really meaningful for `mid_fall`, where
    # there's a real scripted fall before control starts.
    mid_fall_stats = per_category.get("mid_fall")
    rate = mid_fall_stats.get("g2_impact_vs_own_fall_pass_rate") if mid_fall_stats else None
    if rate is not None:
        g2_checks["g2_impact_vs_own_fall"] = {
            "value": rate, "threshold": gc.G2_IMPACT_VS_OWN_FALL_MIN_RATE,
            "pass": rate >= gc.G2_IMPACT_VS_OWN_FALL_MIN_RATE,
            "n_mid_fall_episodes": mid_fall_stats.get("n_episodes"),
            "ground_force_peak_limp_n_mean": mid_fall_stats.get("ground_force_peak_limp_n_mean"),
            "ground_force_peak_active_n_mean": mid_fall_stats.get("ground_force_peak_active_n_mean"),
            "note": "per mid_fall episode, peak ground (non-foot, ground-only) "
            "force after policy control starts <= peak ground force during the limp fall phase of the same "
            "episode. Absolute hardware load limits are a hardware sign-off item "
            "(HARDWARE_TEST_PROTOCOL), not checked here.",
        }
    else:
        g2_checks["g2_impact_vs_own_fall"] = {"status": "skipped", "reason": "no mid_fall category data in this run (needs --categories including mid_fall)"}
    # Provisional/informational checks are computed and shown but don't decide G2's overall pass/fail.
    hard_checks = [v for v in g2_checks.values() if "pass" in v and not v.get("provisional", False) and not v.get("informational", False)]
    gates["G2"] = {
        "status": "pass" if hard_checks and all(v["pass"] for v in hard_checks) else ("fail" if any(not v.get("pass", True) for v in hard_checks) else "incomplete"),
        "checks": g2_checks,
    }

    # G3's "weak motors" condition is --motor_strength (the actuator clip only), not
    # --effort_scale (which would also -- wrongly -- shrink the trained action bound). The recommended
    # invocation is `--effort_scale trained --motor_strength 0.9` (or "stageB"): whatever bound the
    # checkpoint actually trained under, stress-tested with a weakened clip on top.
    motor_strength_resolved = run_params.get("motor_strength")
    is_weak_motor = motor_strength_resolved == "stageB" or (
        isinstance(motor_strength_resolved, (int, float)) and motor_strength_resolved <= gc.G3_EFFORT_SCALE + 1e-6
    )
    # G3 = G3a AND G3c.
    #   G3a (pushes disabled, --dr off): success_g3 (strict, continuous 5s hold within 6s, no velocity term)
    #     >= 0.80 per category.
    #   G3c (pushes on, --dr on): "handoff-ready" = success_g3c (held >= 1s within 6s, no velocity term, no
    #     continuous-hold requirement) >= 0.80 per category.
    # `--dr` no longer needs to be "on" for this gate to apply at all -- motor + terrain alone put a run in
    # "G3 regime"; which SIDE (G3a or G3c) gets assessed depends on whether that specific run had pushes.
    # A single eval run only has pushes on or off, so only one side can be assessed per run -- combine across
    # a no-push run and a push run for the full G3 verdict. success_g3b (alternative criterion -- tolerates
    # brief recoverable stumbles) is reported informationally, not gating.
    is_g3_motor_terrain = is_weak_motor and run_params["terrain"] == "rough"
    if not has_getup:
        gates["G3"] = {"status": "not_applicable", "reason": "no getup env / no success signal available"}
    elif not is_g3_motor_terrain:
        gates["G3"] = {"status": "not_applicable", "reason": "G3 requires --motor_strength <= 0.9 (or 'stageB') and terrain=rough; rerun with those settings, e.g. --effort_scale trained --motor_strength 0.9 --terrain rough, with --dr off for a G3a read or --dr on for a G3c read (sim2sim is checked separately, not here)"}
    else:
        def _per_category_check(key: str, threshold: float) -> dict:
            per_cat_ok = all((s.get(key) or 0.0) >= threshold for s in per_category.values())
            return {
                "status": "pass" if per_cat_ok else "fail",
                "per_category_min": min((s.get(key) or 0.0) for s in per_category.values()) if per_category else None,
                "per_category_threshold": threshold,
            }

        if run_params["dr"] == "off":
            g3a = _per_category_check("success_rate_g3", gc.G3_PER_CATEGORY_MIN)
            g3a["note"] = "G3a: success_g3 (strict, continuous 5s hold within 6s, no velocity term), pushes disabled."
            g3c = {"status": "not_applicable", "reason": "this run has --dr off; G3c requires pushes (--dr on)"}
        elif run_params["dr"] == "on":
            g3a = {"status": "not_applicable", "reason": "this run has --dr on (pushes); G3a requires pushes disabled (--dr off)"}
            g3c = _per_category_check("success_rate_g3c", gc.G3_PER_CATEGORY_MIN)
            g3c["note"] = "G3c: handoff-ready (held >=1s within 6s, no velocity term, no continuous-hold requirement), with pushes."
        else:
            g3a = g3c = {"status": "not_applicable", "reason": f"unexpected --dr value {run_params['dr']!r}"}

        if g3a.get("status") == "fail" or g3c.get("status") == "fail":
            g3_status = "fail"  # AND short-circuits: the applicable side already failed, regardless of the other
        elif g3a.get("status") == "pass" and g3c.get("status") == "pass":
            g3_status = "pass"  # both sides known (not possible from a single run today, kept for completeness)
        else:
            g3_status = "incomplete"  # only one side known and it passed -- needs the other run to conclude G3

        gates["G3"] = {
            "status": g3_status,
            "G3a": g3a,
            "G3c": g3c,
            "success_g3b_per_category_min_informational": (
                min((s.get("success_rate_g3b") or 0.0) for s in per_category.values()) if per_category else None
            ),
            "success_g1_per_category_min_informational": (
                min((s.get("success_rate_g1") or 0.0) for s in per_category.values()) if per_category else None
            ),
            "note": "G3 = G3a AND G3c. G3a (pushes disabled): "
            "success_g3 (strict, continuous 5s hold within 6s, no velocity term) >= 0.80/category. G3c "
            "(pushes on): handoff-ready = success_g3c (held >=1s within 6s, no velocity term) >= "
            "0.80/category. A single eval run only has pushes on or off, so only one of G3a/G3c is assessed "
            "per run -- combine across a no-push run and a push run for the full G3 verdict; this gate "
            "short-circuits to 'fail' if the applicable side already fails. success_g3b (tolerates brief "
            "recoverable stumbles) and success_g1 are kept informational, not gating. sim2sim (MuJoCo 1kHz "
            "/ Isaac dt 0.002) is checked separately, not by this script.",
        }
    return gates


def _fmt_bound(bound_scale) -> str:
    """Compact summary of a per-joint bound_scale list for the markdown header (23 numbers inline isn't
    readable; min/max/mean is enough to see whether it's uniform-nominal or a real Stage-B-style split)."""
    if not bound_scale:
        return "n/a"
    return f"min={min(bound_scale):.3f} max={max(bound_scale):.3f} mean={sum(bound_scale) / len(bound_scale):.3f}"


def _fmt_clip_group_stats(stats: dict | None) -> str:
    if not stats:
        return "n/a"
    return f"min={stats['min']:.1f} max={stats['max']:.1f} mean={stats['mean']:.1f} Nm (n={stats['n_joints']})"


def _fmt_clip_by_group(by_group: dict | None) -> str:
    """Show the LIVE applied clip per joint group in the header, so a
    motor_strength-not-applied bug (or any other clip mismatch) is visible by inspection, not just by
    diffing metrics.json across runs."""
    if not by_group:
        return "n/a"
    return (
        f"elbow/wrist: {_fmt_clip_group_stats(by_group.get('elbow_wrist'))}; "
        f"other: {_fmt_clip_group_stats(by_group.get('other'))}"
    )


def _to_markdown(summary: dict) -> str:
    rp = summary["run_params"]
    lines = [
        f"# Get-up eval — `{rp['task']}`",
        "",
        f"Checkpoint: `{rp['checkpoint']}`  ",
        f"effort_scale={rp['effort_scale']} (resolved: {rp.get('effort_contract_source')}) "
        f"motor_strength={rp.get('motor_strength')} dr={rp['dr']} terrain={rp['terrain']} num_envs={rp['num_envs']} "
        f"episodes/category={rp['num_episodes_per_category']} has_getup_env={rp['has_getup_env']}  ",
        f"DR terms touched (--dr {rp['dr']}): {rp.get('dr_terms_touched') or '(none)'}  ",
        f"DR-wide live-Δv check: {rp.get('dr_push_n_observed')} push_robot firings observed, "
        f"max |Δv_xy| = {rp.get('dr_push_max_dv_xy_mps')} m/s (verifies the wide push range "
        "actually reaches the physics, not just the cfg -- n/a unless --dr on)  ",
        f"applied action contract: beta={rp.get('beta')} (live={rp.get('live_beta')}) "
        f"bound_scale={_fmt_bound(rp.get('live_bound_scale'))} "
        f"(checkpoint's own trained beta was {rp.get('checkpoint_trained_beta')})  ",
        f"live applied clip by joint group: {_fmt_clip_by_group(rp.get('applied_clip_nm_by_group'))}  ",
        f"non-foot impact metrics exclude the first {rp.get('impact_warmup_s', 0.1):.2f}s after each reset "
        "(guards against a reset/spawn artifact, not a real impact -- `F_nonfoot@t` below is "
        "when the reported peak actually happened, so a reviewer can check it isn't still right at the "
        "warmup boundary)",
        "G3-family metrics (`success G3`/`G3b`/`G3c`) are scored without `is_standing`'s "
        "`lin_vel < 0.3 m/s` condition, unlike `success G1` -- that condition is ill-posed under DR, since "
        "`push_robot` SETS base velocity directly (up to 0.5 m/s), so it fails by construction after a push "
        "regardless of the policy. `success G3` = standing within 6s and held continuously >=5s using "
        "height/tilt/feet only (no velocity check) -- see below for how this feeds the gate. "
        "`success G1` (with velocity) is still reported alongside for comparison and is what the G1 (no-DR) "
        "gate itself uses, unchanged.",
        "Alternative G3 criterion: `success G3b` = stood within 6s (G3's own criterion) AND never "
        "\"fell\" for the rest of the episode, where fell = height<0.50m OR tilt>0.35rad *sustained* for "
        "> 0.5s (a brief stumble that recovers within 0.5s is allowed, unlike `success G3`'s hold timer, "
        "which resets on any momentary violation). Informational, alongside `success G3c` ("
        "`success G3` = held continuously >=5s within 6s -> gates G3a with pushes disabled; "
        "`success G3c` = held only >=1s within 6s, no continuous-hold requirement -> gates G3c with pushes "
        "on; G3 = G3a AND G3c, see the Gate checks section below).",
        "",
        "| category | n | success 1s | success G1 | success G3 (G3a) | success G3b (info) | success G3c (G3c) | ttf p50 | ttf G3 p50 | ttf p90 | peak τ̂ | %τ̂>0.9 (any/max joint) | e/w sat s | thermal peak | energy J | peak F_nonfoot N | F_nonfoot@t(s) | impulse N·s | jitter RMS | action rate |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    def row(name, s):
        def f(v, fmt="{:.3f}"):
            return fmt.format(v) if isinstance(v, (int, float)) else "-"

        sat_any = f(s.get("pct_steps_tau_hat_gt_0_9_any_joint_mean"))
        sat_max = f(s.get("pct_steps_tau_hat_gt_0_9_per_joint_max"))
        return (
            f"| {name} | {s.get('n_episodes', 0)} | {f(s.get('success_rate_1s'))} | {f(s.get('success_rate_g1'))} | "
            f"{f(s.get('success_rate_g3'))} | {f(s.get('success_rate_g3b'))} | {f(s.get('success_rate_g3c'))} | "
            f"{f(s.get('time_to_stand_p50_s'))} | {f(s.get('time_to_stand_g3_p50_s'))} | {f(s.get('time_to_stand_p90_s'))} | {f(s.get('peak_tau_hat_max'))} | "
            f"{sat_any}/{sat_max} | {f(s.get('elbow_wrist_sat_s_max'))} | {f(s.get('thermal_proxy_peak_max'))} | "
            f"{f(s.get('energy_j_mean'))} | {f(s.get('peak_nonfoot_force_n_max'))} | {f(s.get('peak_nonfoot_force_time_s_at_max'))} | "
            f"{f(s.get('nonfoot_impulse_ns_mean'))} | {f(s.get('action_jitter_rms_mean'))} | {f(s.get('action_rate_mean'))} |"
        )

    for category, s in summary["per_category"].items():
        lines.append(row(category, s))
    lines.append(row("**overall**", summary["overall"]))
    lines.append("")
    lines.append("## Gate checks")
    lines.append(
        "Note: absolute hardware load limits are a separate hardware sign-off item "
        "(HARDWARE_TEST_PROTOCOL) -- not checked by any gate below."
    )
    for gate, info in summary["gates"].items():
        lines.append(f"- **{gate}**: {info.get('status')}" + (f" — {info['reason']}" if "reason" in info else ""))
        for check_name, check in info.get("checks", {}).items():
            if check.get("provisional"):
                lines.append(
                    f"  - *(provisional, not counted toward {gate}'s pass/fail -- "
                    f"--walking_baseline_provisional is set)* "
                    f"`{check_name}`: value={check.get('value')}, threshold={check.get('threshold')}, "
                    f"would-{'pass' if check.get('pass') else 'fail'}"
                )
            elif check.get("informational"):
                lines.append(
                    f"  - *(informational, not counted toward {gate}'s pass/fail)* "
                    f"`{check_name}`: value={check.get('value')}, threshold={check.get('threshold')}, "
                    f"would-{'pass' if check.get('pass') else 'fail'}"
                )
            elif "pass" in check:
                lines.append(
                    f"  - `{check_name}`: value={check.get('value')}, threshold={check.get('threshold')}, "
                    f"{'pass' if check.get('pass') else 'fail'}"
                )
        # G3's structure is bespoke (G3a/G3c), not the generic "checks" dict the other gates use.
        for sub_name in ("G3a", "G3c"):
            if sub_name in info:
                sub = info[sub_name]
                lines.append(
                    f"  - **{sub_name}**: {sub.get('status')}"
                    + (f" — {sub['reason']}" if "reason" in sub else "")
                    + (f" (per_category_min={sub.get('per_category_min')}, threshold={sub.get('per_category_threshold')})" if "per_category_min" in sub else "")
                    + (f" — {sub['note']}" if "note" in sub else "")
                )
    sanity = summary.get("action_metrics_sanity_check", {})
    if sanity.get("status") == "FAILED":
        lines.append("")
        lines.append(f"**SANITY CHECK FAILED**: {sanity.get('message')}")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    # main() only reaches its own `os._exit(0)` on the success path; any exception raised inside it (an
    # Isaac Lab internal error, or one of this script's own asserts -- category mismatch, action-bound
    # mismatch) would skip that call. Kit's own background threads don't necessarily terminate just because
    # Python's main thread raised, and a wrapping shell `timeout` doesn't reliably forward SIGTERM down to a
    # grandchild process either -- so the underlying Kit process could keep running indefinitely. Guarantee
    # os._exit() runs on EVERY path, not just the happy one.
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
