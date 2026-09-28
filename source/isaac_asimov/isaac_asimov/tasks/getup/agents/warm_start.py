# Copyright (c) 2026, Menlo Research.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Build a Stage B warm-start checkpoint from a Stage A checkpoint. Torch only; run it as a file.

``train.py --resume`` loads everything in a checkpoint (actor, critic, optimizer, iteration). For Stage B we want the
actor and critic weights **with their empirical obs normalizers** (they are part of the model state dicts in rsl-rl
5.0.1) and the reward normalizer (its scale matches the critic), but **not** the Adam moments or the Stage A
exploration std, and the iteration counter restarted at 0 (the Stage B schedules count from 0). This script writes
such a file where ``train.py``'s checkpoint lookup finds it (``logs/rsl_rl/<experiment>/<run_dir>/<file>``):

    python source/isaac_asimov/isaac_asimov/tasks/getup/agents/warm_start.py \\
        --src <Stage A run dir>/model_4000.pt --run_dir warmstart_stageA_4000

    python scripts/rsl_rl/train.py --task Asimov1-GetUp-StageB-v0 --num_envs 4096 --headless \\
        --resume --load_run warmstart_stageA_4000 --checkpoint model_warmstart.pt

What is kept / changed:
- ``actor_state_dict``, ``critic_state_dict`` (incl. obs normalizers), ``reward_normalizer_state_dict``: kept.
- action std (``*std_param`` / ``*log_std_param`` in the actor): set to ``--std`` (default 0.4, Stage B clamp 0.6).
- ``optimizer_state_dict``: param groups kept (shapes), per-parameter state (Adam moments) dropped, lr set to ``--lr``.
- ``iter``: 0. ``infos``: records the source checkpoint.
The source file is only read.
"""

from __future__ import annotations

import argparse
import math
import os

import torch


def build(src: str, dst: str, std: float, lr: float) -> dict:
    ck = torch.load(src, map_location="cpu", weights_only=False)
    out = {}
    actor = dict(ck["actor_state_dict"])
    changed = []
    for k, v in actor.items():
        if k.endswith("log_std_param"):
            actor[k] = torch.full_like(v, math.log(std))
            changed.append(k)
        elif k.endswith("std_param"):
            actor[k] = torch.full_like(v, std)
            changed.append(k)
    if not changed:
        raise KeyError(f"no std parameter found in the actor state dict (keys: {list(actor)[:10]}...)")
    out["actor_state_dict"] = actor
    out["critic_state_dict"] = ck["critic_state_dict"]
    for k in ("reward_normalizer_state_dict",):
        if k in ck:
            out[k] = ck[k]
    opt = ck["optimizer_state_dict"]
    groups = [dict(g, lr=lr) for g in opt["param_groups"]]
    out["optimizer_state_dict"] = {"state": {}, "param_groups": groups}
    out["iter"] = 0
    out["infos"] = {"warm_start_from": os.path.abspath(src), "source_iter": ck.get("iter"), "std": std, "lr": lr}
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    torch.save(out, dst)
    return {"std_keys": changed, "source_iter": ck.get("iter"), "dst": os.path.abspath(dst), "keys": sorted(out)}


# Stage A effort schedule defaults (getup_env_cfg.CurriculumCfg.effort) and the Stage B start values.
_A = {"stage1_scale": 1.2, "final_scale": 1.0, "beta_start": 1.0, "beta_end": 0.8, "transition_iters": 1000}
_B_START = {"strong": 1.0, "beta": 0.8}


def source_contract(src: str) -> dict | None:
    """Action bound (strong-joint scale) and beta the source checkpoint was trained with, from the
    ``curriculum_state_<iter>.json`` next to it (``action_contract`` if present, else recomputed from the effort stage)."""
    import json

    ck_iter = int(torch.load(src, map_location="cpu", weights_only=False).get("iter", -1))
    path = os.path.join(os.path.dirname(src), f"curriculum_state_{ck_iter}.json")
    if not os.path.exists(path):
        path = os.path.join(os.path.dirname(src), "curriculum_state.json")
    if not os.path.exists(path):
        return None
    d = json.load(open(path))
    c = d.get("action_contract")
    if c:
        names = c["joint_names"]
        strong = c["bound_scale"][names.index("left_hip_pitch_joint")]
        return {"strong": float(strong), "beta": float(c["beta"]), "file": path}
    e = d["terms"]["effort"]
    it = int(d["iteration"])
    if e["stage"] == 0:
        u = 0.0
    elif e["stage"] == 1:
        u = min(1.0, (it - e["stage_start_iter"]) / _A["transition_iters"])
    else:
        u = 1.0
    strong = _A["stage1_scale"] + u * (_A["final_scale"] - _A["stage1_scale"])
    beta = _A["beta_start"] + u * (_A["beta_end"] - _A["beta_start"])
    return {"strong": strong, "beta": beta, "file": path, "stage": e["stage"], "iteration": it}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", required=True, help="Stage A checkpoint (read-only)")
    p.add_argument("--run_dir", required=True, help="run folder name to create under logs/rsl_rl/<experiment>/")
    p.add_argument("--experiment", default="asimov1_getup_stageB", help="Stage B runner experiment_name")
    p.add_argument("--file", default="model_warmstart.pt")
    p.add_argument("--std", type=float, default=0.4)
    p.add_argument("--lr", type=float, default=3.0e-4)
    a = p.parse_args()
    dst = os.path.join("logs", "rsl_rl", a.experiment, a.run_dir, a.file)
    print("[warm_start]", build(a.src, dst, a.std, a.lr))
    c = source_contract(a.src)
    if c is None:
        print("[warm_start] WARNING: no curriculum_state JSON next to the source; cannot check the action contract")
        return
    print(f"[warm_start] source contract: strong-joint bound x{c['strong']:.3f}, beta {c['beta']:.3f} ({c['file']})")
    if abs(c["strong"] - _B_START["strong"]) > 1e-3 or abs(c["beta"] - _B_START["beta"]) > 1e-3:
        print(
            "[warm_start] WARNING: Stage B starts at bound x1.0 / beta 0.8. Start it from the source contract instead"
            " (a mismatch shrinks the P-torque range the policy learned) with these train.py overrides:\n"
            f"    env.curriculum.effort.params.stage1_scale={c['strong']:.4f}"
            f" env.curriculum.effort.params.beta_start={c['beta']:.4f} env.actions.joint_pos.beta={c['beta']:.4f}"
        )


if __name__ == "__main__":
    main()
