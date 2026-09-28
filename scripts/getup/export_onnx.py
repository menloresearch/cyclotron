# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Export the get-up policy to ONNX with the firmware metadata contract.

Loads an rsl_rl checkpoint the same way ``scripts/rsl_rl/play.py`` does, exports it with Isaac Lab's
own ``export_policy_as_onnx``/``export_policy_as_jit`` (single input ``obs``, single output
``actions``, opset 18 -- unmodified, so the graph itself is identical to what ``play.py`` would
produce), then **adds ONNX metadata** a robot-side GETUP mode needs (see
``docs/getup/FIRMWARE_SPEC.md`` for how each key is used):

    joint_stiffness, joint_damping   comma-separated Kp/Kd per joint, ASIMOV_1_JOINT_NAMES order
                                      (the existing walking-policy metadata key names)
    default_joint_pos                comma-separated q_default (ASIMOV_1_STANDING_INIT_STATE), the
                                      walking target pose -- also what firmware's own handoff gate
                                      needs for "||q - q_default||_inf < 0.15 rad"
    action_scale                     comma-separated s_j (existing key name, repurposed: for the
                                      get-up contract this is the *relative* per-joint scale, not a
                                      flat 0.25 -- action_mode below is what tells firmware which
                                      interpretation applies)
    action_mode                      "relative" (vs. the walking policy's "absolute")
    action_s_j                       identical values to action_scale, under an unambiguous name
                                      -- read this one if a consumer already
                                      hardcodes a different meaning for action_scale
    action_beta                      the trained curriculum bound (float)
    action_lpf_alpha                 the trained LPF coefficient (float) -- the authoritative
                                      alpha value (see the deploy spec): firmware must read
                                      this value, never hardcode 0.24 or 0.557
    action_relative_per_cycle        "true"/"false": whether q_meas must be re-read
                                      every 200 Hz control cycle (true) or only every policy tick
                                      (false) when computing the relative target
    obs_history_length                5 (policy observation group history length)
    joint_order                       comma-separated ASIMOV_1_JOINT_NAMES (disambiguates every
                                      comma-separated array above)

Verification is a **separate step, in a separate venv, on purpose** (keeping ``onnxruntime`` out of
the Isaac Sim training venv). This script
(run in the Isaac venv, which has ``torch``+``isaaclab`` but no ``onnxruntime``) only exports the
ONNX/JIT files, writes the metadata, and saves a small ``verify_samples.npz`` (random observations
plus this exact in-memory torch model's outputs on them) next to the ONNX file. The actual
ONNX-vs-torch numeric parity check runs separately, via ``scripts/getup/verify_onnx_parity.py`` (no
``isaaclab``/``torch`` import at all -- just ``onnxruntime``+``numpy``, so it runs in a plain venv
that already has those, e.g. ``~/venvs/sim2sim``) against that saved ``.npz``.

Usage:
    # Step 1 (Isaac venv): export + write metadata + save verification samples.
    python scripts/getup/export_onnx.py \\
        --task Asimov1-GetUp-Play-v0 --checkpoint <run>/model_999.pt --headless

    # Step 2 (a plain venv with onnxruntime+numpy, e.g. ~/venvs/sim2sim -- NOT the Isaac venv):
    source ~/venvs/sim2sim/bin/activate && python3 scripts/getup/verify_onnx_parity.py \\
        --onnx <run>/exported/policy.onnx --samples <run>/exported/verify_samples.npz

    # No trained checkpoint yet: export the untrained network anyway, to prove the metadata and
    # export pipeline itself is correct:
    python scripts/getup/export_onnx.py \\
        --task Asimov1-GetUp-Play-v0 --random_init --headless
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

from isaaclab.app import AppLauncher

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rsl_rl"))
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Export the get-up policy to ONNX with the firmware metadata contract.")
parser.add_argument("--task", type=str, default="Asimov1-GetUp-Play-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument(
    "--random_init",
    action="store_true",
    default=False,
    help="Skip loading a checkpoint; export the freshly-initialized (untrained) network instead, "
    "to test the export/metadata/verification pipeline before a trained checkpoint exists.",
)
parser.add_argument("--output_dir", type=str, default=None, help="Defaults to <checkpoint_dir>/exported/, or ./exported/ with --random_init.")
parser.add_argument("--num_verify_samples", type=int, default=64, help="Random observation vectors saved for the separate-venv parity check.")
parser.add_argument(
    "--allow_nominal_contract",
    action="store_true",
    default=False,
    help="If the checkpoint's curriculum_state_<iter>.json has no 'action_contract' field, export "
    "with the fresh Play cfg's NOMINAL beta/s_j instead of failing (loud warning either way). Only pass this "
    "for a checkpoint you've confirmed is still Stage 0 (nominal effort).",
)
parser.add_argument("--seed", type=int, default=0)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- heavy imports ---------------------------------------------------------------------------------------------------

import importlib.metadata as metadata

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
from isaac_asimov.assets.robots.asimov_1 import ASIMOV_1_JOINT_NAMES, ASIMOV_1_STANDING_INIT_STATE

_INSTALLED_RSL_RL_VERSION = metadata.version("rsl-rl-lib")


# ---------------------------------------------------------------------------------------------------------------------
# Trained action contract: the checkpoint's ACTUAL trained beta/s_j, not the fresh Play cfg's defaults.
#
# The Play env this script builds (`Asimov1-GetUp-Play-v0`) always disables curricula
# (`Asimov1GetUpEnvCfg_PLAY.__post_init__` sets `self.curriculum = None`), so its action term starts at whatever
# `beta`/`scale_torque_factor` the Cfg *class* defaults to (currently beta=1.0, nominal effort) -- regardless of
# what stage a *training* run's curriculum (`getup.mdp.curriculums.effort_beta_schedule`) had actually reached by
# the iteration a given checkpoint was saved at. For a Stage-B checkpoint (beta 0.7, effort bound scaled x0.9),
# exporting with the Play defaults instead bakes in a firmware target delta 1/(0.7*0.9) = ~1.59x too large.
#
# So this script reads the checkpoint's own `curriculum_state_<iter>.json`, whose top-level `"action_contract"` key
# (written by `getup.mdp.curriculums.action_contract`) holds the trained beta/s_j, and uses ITS values rather than
# anything recomputed from the fresh Play env. Checkpoints saved before that key existed lack it; in that case this
# script fails loudly by default (`RuntimeError`), since silently falling back to nominal values would ship the
# wrong action scale. Pass `--allow_nominal_contract` to export anyway (loud warning, not silent), which is the
# right call for a checkpoint you know is still Stage-A/nominal (e.g. its `curriculum_state_<iter>.json` shows
# `"effort": {"stage": 1, ...}` with no sign that Stage B/beta decay has started).
#
# Field names below match `getup.mdp.actions.FilteredRelativeJointPositionAction.contract()` verbatim:
# `joint_names`, `s_j` (= joint_scale_live, i.e. already beta-independent nominal_s_j * bound_scale), `beta`,
# `bound_scale` (per-joint list), `scale_torque_factor`, `tau_max_nominal`, `kp_nominal`, `action_clip`, `use_lpf`,
# `lpf_alpha`, `lpf_per_substep`, `relative_per_substep`.
# ---------------------------------------------------------------------------------------------------------------------

_ACTION_CONTRACT_REQUIRED_FIELDS = ("joint_names", "s_j", "beta", "bound_scale", "scale_torque_factor", "tau_max_nominal", "kp_nominal")


def _curriculum_state_path(checkpoint_path: str) -> str | None:
    """``<run_dir>/curriculum_state_<iter>.json`` next to ``model_<iter>.pt`` -- falls back to the
    curriculum_state file with the closest mtime if there's no exact iteration-number match.

    The fallback is needed because the checkpoint saver's iteration count and the curriculum
    tracker's can be off by one: a run's final checkpoint may be `model_3999.pt` with no
    `curriculum_state_3999.json` (only `..._3750.json` and `..._4000.json`). The matching pair is
    written at the same moment, so mtime, not the number in the filename, identifies it.
    """
    run_dir = os.path.dirname(checkpoint_path)
    stem = os.path.splitext(os.path.basename(checkpoint_path))[0]
    digits = "".join(ch for ch in stem if ch.isdigit())
    if digits:
        exact = os.path.join(run_dir, f"curriculum_state_{digits}.json")
        if os.path.isfile(exact):
            return exact
    candidates = [
        c for c in glob.glob(os.path.join(run_dir, "curriculum_state_*.json"))
        if os.path.basename(c) != "curriculum_state.json"
    ]
    if not candidates:
        return None
    try:
        ckpt_mtime = os.path.getmtime(checkpoint_path)
    except OSError:
        return sorted(candidates)[-1]
    best = min(candidates, key=lambda c: abs(os.path.getmtime(c) - ckpt_mtime))
    diff_s = abs(os.path.getmtime(best) - ckpt_mtime)
    print(f"[getup-export] No exact curriculum_state_{digits}.json for {os.path.basename(checkpoint_path)!r}; "
          f"using the closest-mtime match instead: {os.path.basename(best)} (|mtime diff| = {diff_s:.1f}s).")
    return best


def load_action_contract(checkpoint_path: str, allow_nominal: bool) -> dict | None:
    """The checkpoint's actual trained action contract, or ``None`` if unavailable and ``allow_nominal`` was
    passed (loud warning either way -- see the module-level comment above for the full rationale)."""
    path = _curriculum_state_path(checkpoint_path)
    contract, reason = None, None
    if path is None:
        reason = f"no curriculum_state_<iter>.json next to {checkpoint_path!r}"
    else:
        with open(path) as f:
            state = json.load(f)
        contract = state.get("action_contract")
        if contract is None:
            reason = f"{path!r} exists but has no 'action_contract' key (see the comment above)"
        else:
            missing = [k for k in _ACTION_CONTRACT_REQUIRED_FIELDS if k not in contract]
            if missing:
                reason = f"{path!r}'s action_contract is missing field(s) {missing}"
                contract = None
    if contract is None:
        msg = f"Cannot determine {checkpoint_path!r}'s ACTUAL trained action contract: {reason}."
        if allow_nominal:
            print(f"[getup-export] WARNING: {msg} Falling back to the Play cfg's NOMINAL contract "
                  "(beta=1.0, unscaled effort) via --allow_nominal_contract. This metadata's action_beta/"
                  "action_s_j will NOT match this checkpoint's actual training if it is past Stage 0 "
                  "(up to ~1.6x off for a Stage-B checkpoint). Only pass this flag for a "
                  "checkpoint you've confirmed is still nominal (Stage 0, e.g. check its own "
                  "curriculum_state_<iter>.json 'effort' term).")
            return None
        raise RuntimeError(
            f"{msg} Exporting with the Play cfg's nominal contract instead would silently bake in the wrong "
            "beta/s_j. Pass --allow_nominal_contract to export anyway with a loud warning "
            "(only if you've confirmed this checkpoint is still Stage 0 / nominal), or re-save the "
            "checkpoint from a training run that writes the action_contract field."
        )
    contract_joint_names = list(contract["joint_names"])
    assert contract_joint_names == list(ASIMOV_1_JOINT_NAMES), (
        f"action_contract joint order {contract_joint_names} != ASIMOV_1_JOINT_NAMES {list(ASIMOV_1_JOINT_NAMES)}"
    )
    # File-level self-consistency: recompute s_j from the contract's own raw
    # ingredients and confirm it matches the contract's own declared s_j, using the one formula this whole
    # codebase agrees on (getup.mdp.actions.FilteredRelativeJointPositionAction): this catches a bug in
    # whatever wrote the file, or in this script's own reading of it, independent of any live-env application.
    tau_max_nominal = torch.tensor(contract["tau_max_nominal"], dtype=torch.float64)
    kp_nominal = torch.tensor(contract["kp_nominal"], dtype=torch.float64)
    bound_scale = torch.as_tensor(contract["bound_scale"], dtype=torch.float64).expand_as(tau_max_nominal)
    recomputed_s_j = float(contract["scale_torque_factor"]) * bound_scale * tau_max_nominal / kp_nominal.clamp(min=1e-9)
    declared_s_j = torch.tensor(contract["s_j"], dtype=torch.float64)
    max_diff = (recomputed_s_j - declared_s_j).abs().max().item()
    assert max_diff < 1e-4, (
        f"Parity check failed: action_contract's own declared s_j disagrees with "
        f"scale_torque_factor*bound_scale*tau_max_nominal/kp_nominal by {max_diff:.3e} -- do not ship this "
        f"metadata; the contract file ({path}) or this script's formula has drifted from the action term's definition."
    )
    print(f"[getup-export] Parity check passed: {path}'s declared s_j matches "
          f"scale_torque_factor*bound_scale*tau_max_nominal/kp_nominal to {max_diff:.1e}.")
    return contract


def _apply_and_cross_check_live(raw_env, action_term, contract: dict) -> None:
    """Live cross-check (the exported bound must equal the env's live bound): apply the contract's
    own ``beta`` and ``bound_scale`` to the fresh Play env's *live* action term via the exact same public API
    ``getup.mdp.curriculums.effort_beta_schedule`` uses during training (``action_term.beta = ...`` and
    ``action_term.set_bound_scale(...)``, see `getup/mdp/actions.py`'s
    `FilteredRelativeJointPositionAction`), then read back the term's own
    ``joint_scale_live`` and assert it equals the contract's declared ``s_j`` exactly. This exercises the real
    action-term code path (tau_max_nominal/kp_nominal come from the *live* env's own asset/actuators, via
    `joint_tables`), so it also catches asset drift between what this Play env has and what training used --
    something the file-only self-consistency check in :func:`load_action_contract` cannot see."""
    action_term.beta = float(contract["beta"])
    action_term.set_bound_scale(torch.tensor(contract["bound_scale"], device=action_term.device))
    live_s_j = action_term.joint_scale_live.reshape(-1).to("cpu", dtype=torch.float64)
    declared_s_j = torch.tensor(contract["s_j"], dtype=torch.float64)
    max_diff = (live_s_j - declared_s_j).abs().max().item()
    print(f"[getup-export] Live cross-check: applied the contract's beta={action_term.beta:.4g} and "
          f"bound_scale to the fresh Play env's own action term; its joint_scale_live vs. the contract's "
          f"declared s_j -- max diff {max_diff:.3e}.")
    assert max_diff < 1e-4, (
        f"Live parity check failed: applying the contract's beta/bound_scale to this Play env's own "
        f"action term gives joint_scale_live differing from the contract's declared s_j by {max_diff:.3e} -- "
        "either this Play env's asset/actuators (tau_max/Kp) have drifted from what training used, or the "
        "contract file is stale/wrong. Do not ship this metadata; investigate first."
    )


# ---------------------------------------------------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------------------------------------------------


def _csv(values) -> str:
    return ",".join(f"{float(v):.8g}" for v in values)


def build_metadata(env_cfg, raw_env, action_contract: dict | None) -> dict[str, str]:
    """Everything the firmware GETUP-mode contract needs. See module docstring for each key.

    ``action_contract``: the checkpoint's real trained beta/s_j, or ``None`` to fall back to the fresh
    Play env's nominal defaults (only reached via ``--allow_nominal_contract``, with its own loud warning
    already printed by :func:`load_action_contract`).
    """
    asset = raw_env.scene["robot"]
    joint_ids, joint_names = asset.find_joints(list(ASIMOV_1_JOINT_NAMES), preserve_order=True)
    assert joint_names == list(ASIMOV_1_JOINT_NAMES), (
        f"joint order drift: asset resolved {joint_names}, expected {list(ASIMOV_1_JOINT_NAMES)} -- "
        "the firmware indexes every comma-separated array below by this exact order."
    )

    from isaac_asimov.tasks.getup.mdp.state import joint_tables

    tab = joint_tables(raw_env)
    tau_max = tab.tau_max[joint_ids]
    kp_nominal = tab.kp[joint_ids]
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

    action_term = raw_env.action_manager.get_term("joint_pos")

    if action_contract is not None:
        _apply_and_cross_check_live(raw_env, action_term, action_contract)
        s_j = [float(v) for v in action_contract["s_j"]]
        beta = float(action_contract["beta"])
        lpf_alpha = float(action_contract.get("lpf_alpha", getattr(action_term.cfg, "lpf_alpha", 0.557)))
        relative_per_cycle = bool(action_contract.get("relative_per_substep", getattr(action_term.cfg, "relative_per_substep", True)))
        contract_source = "curriculum_state"
    else:
        scale_factor = getattr(action_term.cfg, "scale_torque_factor", 1.1)
        beta = float(getattr(action_term, "beta", getattr(action_term.cfg, "beta", 1.0)))
        lpf_alpha = float(getattr(action_term.cfg, "lpf_alpha", 0.557))
        relative_per_cycle = bool(getattr(action_term.cfg, "relative_per_substep", True))
        s_j = (scale_factor * tau_max / kp_nominal.clamp(min=1e-6)).tolist()
        contract_source = "nominal_play_cfg"

    default_pos = [ASIMOV_1_STANDING_INIT_STATE.joint_pos.get(name, 0.0) for name in ASIMOV_1_JOINT_NAMES]
    # ASIMOV_1_STANDING_INIT_STATE uses regex keys (e.g. ".*_hip_roll_joint": 0.0); resolve them properly instead
    # of a plain dict lookup, which would silently default every regex-keyed joint to 0.0.
    import re

    resolved_default_pos = []
    for name in ASIMOV_1_JOINT_NAMES:
        val = 0.0
        for pattern, v in ASIMOV_1_STANDING_INIT_STATE.joint_pos.items():
            if re.fullmatch(pattern, name):
                val = v
        resolved_default_pos.append(val)

    return {
        "joint_order": ",".join(ASIMOV_1_JOINT_NAMES),
        "joint_stiffness": _csv(kp_live.tolist()),
        "joint_damping": _csv(kd_live.tolist()),
        "default_joint_pos": _csv(resolved_default_pos),
        "action_scale": _csv(s_j),  # existing key name, repurposed as the relative scale (see action_mode)
        "action_s_j": _csv(s_j),  # unambiguous alias of action_scale
        "action_mode": "relative",
        "action_beta": f"{beta:.8g}",
        "action_lpf_alpha": f"{lpf_alpha:.8g}",
        "action_relative_per_cycle": "true" if relative_per_cycle else "false",
        "obs_history_length": str(int(getattr(env_cfg.observations.policy, "history_length", 1) or 1)),
        "contract_source": contract_source,  # "curriculum_state" (trained values) or "nominal_play_cfg" (--allow_nominal_contract)
    }


def write_onnx_metadata(onnx_path: str, metadata: dict[str, str]) -> None:
    model = onnx.load(onnx_path)
    del model.metadata_props[:]
    for key, value in metadata.items():
        entry = model.metadata_props.add()
        entry.key = key
        entry.value = value
    onnx.save(model, onnx_path)


# ---------------------------------------------------------------------------------------------------------------------
# Verification samples (the actual numeric check happens later, in a separate venv -- see module docstring)
# ---------------------------------------------------------------------------------------------------------------------


def save_verification_samples(npz_path: str, policy, obs_dim: int, num_samples: int, seed: int) -> None:
    """Save random obs + this exact in-memory policy's outputs on them, for
    ``verify_onnx_parity.py`` to replay against the exported ONNX graph in a different venv.

    ``policy`` is whatever ``runner.get_inference_policy(device="cpu")`` returns -- a plain
    ``callable(obs_tensor) -> action_tensor`` in this rsl-rl-lib version, not necessarily a
    ``torch.nn.Module`` with its own ``.to()``/``.eval()`` (those are called only if present, so
    this also still works if a caller passes an actual ``nn.Module``)."""
    if hasattr(policy, "to"):
        policy = policy.to("cpu")
    if hasattr(policy, "eval"):
        policy.eval()
    rng = np.random.default_rng(seed)
    samples = rng.normal(size=(num_samples, obs_dim)).astype(np.float32)
    with torch.inference_mode():
        torch_out = policy(torch.from_numpy(samples)).cpu().numpy()
    np.savez(npz_path, obs=samples, torch_actions=torch_out)
    print(f"[getup-export] Saved {num_samples} verification samples to {npz_path} (obs {samples.shape} -> actions {torch_out.shape})")


# ---------------------------------------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------------------------------------


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    torch.manual_seed(args_cli.seed)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    # Required for this pinned rsl-rl-lib version, exactly like scripts/rsl_rl/play.py does: without
    # it, OnPolicyRunner's internal MLPModel construction raises
    # `TypeError: MLPModel.__init__() got an unexpected keyword argument 'stochastic'`.
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, _INSTALLED_RSL_RL_VERSION)

    env = gym.make(args_cli.task, cfg=env_cfg)
    raw_env = env.unwrapped
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)

    action_contract = None
    if args_cli.random_init:
        print("[getup-export] --random_init: exporting the freshly-initialized (UNTRAINED) network. "
              "This validates the export/metadata/verification pipeline only, not a real policy.")
        resume_path = None
        output_dir = args_cli.output_dir or "exported"
    else:
        if not args_cli.checkpoint:
            parser.error("--checkpoint is required unless --random_init is set")
        # Locate the action contract from the *original* checkpoint path (before any deprecated-checkpoint
        # reformatting below, which may write to a different temp location) -- curriculum_state_<iter>.json
        # lives next to the checkpoint the training run actually saved, not next to a converted copy.
        original_resume_path = retrieve_file_path(args_cli.checkpoint)
        action_contract = load_action_contract(original_resume_path, args_cli.allow_nominal_contract)
        resume_path = handle_deprecated_rsl_rl_checkpoint(original_resume_path, _INSTALLED_RSL_RL_VERSION)
        print(f"[getup-export] Loading checkpoint: {resume_path}")
        runner.load(resume_path)
        output_dir = args_cli.output_dir or os.path.join(os.path.dirname(resume_path), "exported")

    os.makedirs(output_dir, exist_ok=True)

    # Export exactly like scripts/rsl_rl/play.py does: rsl-rl-lib >= 4.0 exposes export methods on
    # the *runner* itself (this pinned version does). Reaching into `runner.alg.policy`/`.actor_critic`
    # directly, an older-API assumption, raises `AttributeError: 'GetUpPPO' object has no attribute
    # 'actor_critic'`, since `GetUpPPO` (the get-up PPO algorithm subclass) doesn't carry a monolithic
    # actor-critic object in this version. Falls back to the older manual
    # policy/normalizer extraction only if the installed version is actually < 4.0.
    from packaging import version as _pkg_version

    if _pkg_version.parse(_INSTALLED_RSL_RL_VERSION) >= _pkg_version.parse("4.0.0"):
        runner.export_policy_to_jit(path=output_dir, filename="policy.pt")
        runner.export_policy_to_onnx(path=output_dir, filename="policy.onnx")
    else:
        policy_nn = runner.alg.policy if hasattr(runner.alg, "policy") else runner.alg.actor_critic
        normalizer = getattr(policy_nn, "actor_obs_normalizer", None)
        export_policy_as_jit(policy_nn, normalizer=normalizer, path=output_dir, filename="policy.pt")
        export_policy_as_onnx(policy_nn, normalizer=normalizer, path=output_dir, filename="policy.onnx")
    onnx_path = os.path.join(output_dir, "policy.onnx")
    print(f"[getup-export] Exported (no metadata yet): {onnx_path}")

    onnx_metadata = build_metadata(env_cfg, raw_env, action_contract)
    write_onnx_metadata(onnx_path, onnx_metadata)
    print(f"[getup-export] Wrote {len(onnx_metadata)} ONNX metadata keys:")
    for k, v in onnx_metadata.items():
        preview = v if len(v) <= 80 else v[:77] + "..."
        print(f"[getup-export]   {k} = {preview}")

    # `get_inference_policy` is the version-proof way to get a callable, exactly what play.py
    # itself uses to drive the env -- avoids depending on any internal attribute layout of
    # `runner.alg`. It expects the *multi-group dict* shape (`{"policy": tensor}`, matching
    # `obs_groups={"actor": ["policy"]}`) since that's what `env.step()`/`get_observations()`
    # hand it in the training loop; passing a bare tensor
    # raises `IndexError: too many indices for tensor of dimension 2` from `MLPModel.get_latent`'s
    # `obs[obs_group]` lookup. The *exported ONNX graph*, in contrast, takes a single flat tensor
    # (matching every other export in this codebase, e.g. the walking policy's own [N,78] input) --
    # `runner.export_policy_to_onnx` wraps that dict-unpacking internally. This tiny adapter
    # restores the flat-tensor-in/flat-tensor-out contract `save_verification_samples` (and the
    # ONNX graph) both expect, so the samples we save are directly comparable to the ONNX output.
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
    print(f"[getup-export] Next: in a plain venv with onnxruntime+numpy (NOT this Isaac venv), run:")
    print(f"[getup-export]   python3 scripts/getup/verify_onnx_parity.py --onnx {onnx_path} --samples {npz_path}")


if __name__ == "__main__":
    main()
    simulation_app.close()
