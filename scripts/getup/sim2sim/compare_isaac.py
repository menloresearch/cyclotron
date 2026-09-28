#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Check this package's (isaaclab-free) walking obs pipeline against a captured Isaac rollout
(``capture_isaac_obs.py``'s ``.npz``), term by term.

Pure numpy -- no ``isaaclab``/``torch``/``mujoco`` needed, so it runs anywhere, including off the server.

``base_ang_vel``/``projected_gravity`` go through Isaac's ``delayed_obs`` (a per-episode random 0-1 / 0-2 tick
lag) -- since that lag isn't recorded in the capture and this script's own reconstruction primes its delay buffer
one call later than Isaac's internal reset-time obs computation does (see module docstring in ``filters.py``), it
brute-forces every candidate lag and reports the best match plus its residual, skipping the first few ticks where
that one-call priming offset could still matter. Every other term has no randomness once Isaac's obs noise is off
(``Asimov1VelocityEnvCfg_PLAY``) and should match near-exactly.
"""

from __future__ import annotations

import argparse

import numpy as np

from . import constants as C


def _term_slices() -> dict[str, slice]:
    dims = {
        "base_ang_vel": 3, "projected_gravity": 3, "command": 3,
        "joint_pos_slot01": 9, "joint_pos_slot23": 8, "joint_pos_slot45": 6,
        "joint_vel_slot01": 9, "joint_vel_slot23": 8, "joint_vel_slot45": 6,
        "actions": C.NUM_JOINTS,
    }
    out, start = {}, 0
    for name in C.WALKING_OBS_TERM_ORDER:
        out[name] = slice(start, start + dims[name])
        start += dims[name]
    assert start == 78, f"expected 78-dim walking obs, term dims sum to {start}"
    return out


def _reorder_to_asimov_order(arr: np.ndarray, captured_names: list[str]) -> np.ndarray:
    if list(captured_names) == C.ASIMOV_1_JOINT_NAMES:
        return arr
    idx = [captured_names.index(n) for n in C.ASIMOV_1_JOINT_NAMES]
    return arr[..., idx]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--capture", required=True, help="Path to capture_isaac_obs.py's .npz output.")
    p.add_argument("--skip-ticks", type=int, default=3, help="Ticks to skip at the start (delay-buffer priming transient).")
    p.add_argument("--tol", type=float, default=1e-3, help="Max abs error to call a term PASS.")
    args = p.parse_args(argv)

    data = np.load(args.capture, allow_pickle=True)
    obs_isaac = data["obs_isaac"].astype(float)  # (T, 78)
    T = obs_isaac.shape[0]
    joint_names = list(data["joint_names"])

    qpos = _reorder_to_asimov_order(data["qpos"].astype(float), joint_names)
    qvel = _reorder_to_asimov_order(data["qvel"].astype(float), joint_names)
    default_qpos = _reorder_to_asimov_order(data["default_qpos"].astype(float), joint_names)
    action = _reorder_to_asimov_order(data["action"].astype(float), joint_names)
    base_ang_vel_raw = data["base_ang_vel_raw"].astype(float)
    proj_grav_raw = data["proj_grav_raw"].astype(float)
    command = data["command"].astype(float)

    slices = _term_slices()
    ok = True

    def report(name: str, recon: np.ndarray, skip: int = 0):
        nonlocal ok
        actual = obs_isaac[:, slices[name]]
        err = np.max(np.abs(recon[skip:] - actual[skip:]))
        passed = err <= args.tol
        ok = ok and passed
        print(f"  {name:<20s} max_abs_err={err:.6g}  {'PASS' if passed else 'FAIL'}")

    print(f"Loaded {args.capture}: T={T} ticks, joint_names order matches ASIMOV_1_JOINT_NAMES: "
          f"{list(joint_names) == C.ASIMOV_1_JOINT_NAMES}")

    qpos_rel = qpos - default_qpos
    idx01 = np.array([C.JOINT_INDEX[n] for n in C.SLOT_0_1])
    idx23 = np.array([C.JOINT_INDEX[n] for n in C.SLOT_2_3])
    idx45 = np.array([C.JOINT_INDEX[n] for n in C.SLOT_4_5])

    print("Non-delayed terms (exact, no randomness with obs noise off):")
    report("command", command)
    report("joint_pos_slot01", qpos_rel[:, idx01])
    report("joint_pos_slot23", qpos_rel[:, idx23])
    report("joint_pos_slot45", qpos_rel[:, idx45])
    report("joint_vel_slot01", 0.1 * qvel[:, idx01])
    report("joint_vel_slot23", 0.1 * qvel[:, idx23])
    report("joint_vel_slot45", 0.1 * qvel[:, idx45])
    report("actions", action)

    print("Delayed terms (brute-force best-lag search):")
    for name, raw, scale, max_lag in (
        ("base_ang_vel", base_ang_vel_raw, 0.25, 1),
        ("projected_gravity", proj_grav_raw, 1.0, 2),
    ):
        best_err, best_k, best_recon = None, None, None
        for k in range(max_lag + 1):
            delayed = np.stack([raw[max(0, t - k)] for t in range(T)])
            recon = delayed * scale
            err = np.max(np.abs(recon[args.skip_ticks :] - obs_isaac[args.skip_ticks :, slices[name]]))
            if best_err is None or err < best_err:
                best_err, best_k, best_recon = err, k, recon
        passed = best_err <= args.tol
        ok = ok and passed
        print(f"  {name:<20s} best_lag={best_k} max_abs_err={best_err:.6g}  {'PASS' if passed else 'FAIL'}")

    print(f"\nOverall: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
