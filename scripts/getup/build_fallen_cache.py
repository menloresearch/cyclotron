# Copyright (c) 2026, Menlo Research.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of the collection procedure (spawn limp robots above the ground with random orientation, joints and
# velocities, simulate the fall with zero joint torques, capture the settled state, reject exploded states) are adapted
# from NVIDIA WBC-AGILE `agile/rl_env/mdp/events/fallen_state_dataset.py`:
#
#   SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#   SPDX-License-Identifier: Apache-2.0
#   Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0). Distributed on an
#   "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
#
# Changes vs AGILE: category-targeted spawn recipes (supine/prone/side/sitting/kneeling seeds), limp actuators
# (Kp 0, Kd 0.5) instead of zero torque, 0.3-0.8 m drops, a collision-sphere penetration check, and a category labeler.
"""Build the get-up fallen-state cache (the cached starting states for training resets).

Runs headless, <= 1024 envs, flat plane:

    python scripts/getup/build_fallen_cache.py --headless \
        --num_envs 1024 --target_states 10000 --out ~/getup_cache/fallen_v0.pt

Each round spawns every env from a spawn recipe ("seed type"), lets the limp robot fall and settle for ``--settle_s``,
then keeps the states that are finite, penetration-free and at rest (root speed < 0.05 m/s) and that the labeler
(`tasks/getup/mdp/resets.py::label_fallen_states`) puts into one of the six cached categories.
``--append`` grows an existing cache file across several <= 15 min jobs.
"""

from __future__ import annotations

import argparse
import math
import os
import socket
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Build the get-up fallen-state cache.")
parser.add_argument("--num_envs", type=int, default=1024, help="Parallel drops per round (<= 1024 on the shared server).")
parser.add_argument("--target_states", type=int, default=10000, help="Stop once this many states are accepted.")
parser.add_argument("--max_rounds", type=int, default=1000, help="Hard cap on rounds.")
parser.add_argument("--max_minutes", type=float, default=13.0, help="Stop (and save) after this wall time.")
parser.add_argument("--settle_s", type=float, default=2.0, help="Limp settle time per drop [s].")
parser.add_argument("--out", type=str, default="~/getup_cache/fallen_v0.pt", help="Output cache file.")
parser.add_argument("--append", action="store_true", help="Append to --out if it exists (same joint order required).")
parser.add_argument("--asset", choices=("getup", "walking"), default="getup", help="Robot asset config.")
parser.add_argument("--seed", type=int, default=0, help="Torch RNG seed.")
parser.add_argument(
    "--seed_mix",
    type=str,
    default="",
    help="Spawn-recipe weights, e.g. 'supine=1,prone=1,side=1,sitting=1.5,kneel_up=1,kneel_down=1,generic=0.5'.",
)
parser.add_argument("--save_every", type=int, default=10, help="Checkpoint the output every N rounds.")
parser.add_argument("--trace", type=int, default=0, help="Debug: print the settle trajectory of the first N envs.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- everything below needs the app ----------------------------------------------------------------------------------

import types  # noqa: E402

import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
import isaaclab.utils.math as math_utils  # noqa: E402
from isaaclab.assets import ArticulationCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402
from isaaclab.terrains import TerrainImporterCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

from isaac_asimov.tasks.getup.mdp.limp import set_limp  # noqa: E402
from isaac_asimov.tasks.getup.mdp.resets import (  # noqa: E402
    CACHE_FORMAT_VERSION,
    CACHED_CATEGORIES,
    GETUP_CATEGORIES,
    compute_label_inputs,
    get_collision_spheres,
    joint_mirror_perm,
    label_fallen_states,
)

if args_cli.asset == "getup":
    # hard fail on import errors: a cache must never be built silently on the walking asset
    from isaac_asimov.assets.robots.asimov_1 import ASIMOV_1_GETUP_CFG as ROBOT_CFG

    ASSET_NAME = "ASIMOV_1_GETUP_CFG"
else:
    from isaac_asimov.assets.robots.asimov_1 import ASIMOV_1_DELAYED_CFG as ROBOT_CFG

    ASSET_NAME = "ASIMOV_1_DELAYED_CFG"

PHYSICS_DT = 0.005
GROUND_MATERIAL = sim_utils.RigidBodyMaterialCfg(
    friction_combine_mode="multiply", restitution_combine_mode="multiply", static_friction=1.0, dynamic_friction=1.0
)

# Acceptance thresholds
MAX_ROOT_SPEED = 0.05  # m/s
MAX_ROOT_ANG_SPEED = 0.3  # rad/s
MAX_JOINT_SPEED = 1.0  # rad/s
MAX_PENETRATION = 0.01  # m, deepest collision sphere below the ground


@configclass
class CacheSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground", terrain_type="plane", collision_group=-1, physics_material=GROUND_MATERIAL
    )
    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


# ---------------------------------------------------------------------------------------------------------------------
# Spawn recipes ("seed types"). Angles: roll/pitch/yaw of the pelvis (x forward, y left, z up); pitch > 0 = nose down.
# Joint values are given for the LEFT joint; the right joint gets the mirrored value (-q) from an independent draw.
# Left-joint conventions (URDF): hip pitch < 0 = flexion (to -2.09), knee > 0 = flexion (to 1.5), hip roll > 0 =
# abduction, shoulder pitch < 0 = arm forward.
# ---------------------------------------------------------------------------------------------------------------------

SEED_TYPES = (
    "supine", "prone", "side", "sitting", "kneel_up", "kneel_down", "generic", "side_prop", "sit_fold", "kneel_hands",
)
DEFAULT_SEED_MIX = {  # tuned from the seed x label yields of the body-shell collision geometry
    "supine": 0.06,
    "prone": 0.06,
    "side": 0.0,
    "sitting": 0.06,
    "kneel_up": 0.01,
    "kneel_down": 0.02,
    "generic": 0.04,
    "side_prop": 0.45,
    "sit_fold": 0.08,
    "kneel_hands": 0.22,
}
# drop height of the lowest collision point [m], velocity scale, joint-randomization blend (alpha) range, posture hold
# (gain_scale of the PD toward the spawn pose during the settle; 0 = limp, see DelayedPDLimpableActuator)
SEED_PARAMS = {
    "supine": dict(drop=(0.3, 0.8), vel=1.0, alpha=(0.0, 1.0)),
    "prone": dict(drop=(0.3, 0.8), vel=1.0, alpha=(0.0, 1.0)),
    "side": dict(drop=(0.3, 0.8), vel=1.0, alpha=(0.0, 1.0)),
    "sitting": dict(drop=(0.0, 0.08), vel=0.2, alpha=(0.0, 0.6), hold=(0.3, 0.6)),
    "kneel_up": dict(drop=(0.0, 0.06), vel=0.2, alpha=(0.0, 0.5)),
    "kneel_down": dict(drop=(0.0, 0.06), vel=0.2, alpha=(0.0, 0.5)),
    "generic": dict(drop=(0.3, 0.8), vel=1.0, alpha=(1.0, 1.0)),
    "side_prop": dict(drop=(0.0, 0.10), vel=0.3, alpha=(0.0, 0.3)),
    "sit_fold": dict(drop=(0.0, 0.04), vel=0.2, alpha=(0.0, 0.5)),
    "kneel_hands": dict(drop=(0.0, 0.04), vel=0.2, alpha=(0.0, 0.3), hold=(0.3, 0.6)),
}


def uni(lo, hi, n, device):
    return lo + (hi - lo) * torch.rand(n, device=device)


def sample_spawn(robot, seed_w: torch.Tensor, band_lo, band_hi, soft_lo, soft_hi, jidx: dict[str, int], perm):
    """Sample a spawn for every env. Returns (seed_type, quat, joint_pos, joint_vel, root_vel, drop, hold)."""
    n, dev = robot.num_instances, robot.device
    seed = torch.multinomial(seed_w, n, replacement=True)
    default = robot.data.default_joint_pos.clone()
    num_j = default.shape[1]

    # base joints: blend default -> uniform within 0.9 of the joint range
    u = band_lo + (band_hi - band_lo) * torch.rand(n, num_j, device=dev)
    alpha = torch.zeros(n, 1, device=dev)
    for s, name in enumerate(SEED_TYPES):
        rows = seed == s
        lo, hi = SEED_PARAMS[name]["alpha"]
        alpha[rows, 0] = uni(lo, hi, int(rows.sum()), dev)
    q = default + alpha * (u - default)
    q = torch.maximum(torch.minimum(q, band_hi), band_lo)

    def set_lr(rows, joint, lo, hi):
        k = int(rows.sum())
        if k == 0:
            return
        for side, sgn in (("left", 1.0), ("right", -1.0)):
            q[rows, jidx[f"{side}_{joint}_joint"]] = sgn * uni(lo, hi, k, dev)

    roll = torch.zeros(n, device=dev)
    pitch = torch.zeros(n, device=dev)
    yaw = uni(-math.pi, math.pi, n, dev)

    def rows_of(name):
        return seed == SEED_TYPES.index(name)

    r = rows_of("supine")
    roll[r], pitch[r] = uni(-0.4, 0.4, int(r.sum()), dev), -math.pi / 2 + uni(-0.4, 0.4, int(r.sum()), dev)
    r = rows_of("prone")
    roll[r], pitch[r] = uni(-0.4, 0.4, int(r.sum()), dev), math.pi / 2 + uni(-0.4, 0.4, int(r.sum()), dev)
    r = rows_of("side")
    k = int(r.sum())
    sgn = torch.where(torch.rand(k, device=dev) < 0.5, -1.0, 1.0)
    roll[r], pitch[r] = sgn * (math.pi / 2 + uni(-0.35, 0.35, k, dev)), uni(-0.4, 0.4, k, dev)

    # sitting: pelvis near-upright (slightly reclined .. folded forward), hips flexed ~ 75-115 deg, legs forward
    r = rows_of("sitting")
    k = int(r.sum())
    roll[r], pitch[r] = uni(-0.15, 0.15, k, dev), uni(-0.5, 0.45, k, dev)
    set_lr(r, "hip_pitch", -2.0, -1.3)
    set_lr(r, "knee", 0.0, 1.3)
    set_lr(r, "hip_roll", -0.1, 0.5)
    set_lr(r, "hip_yaw", -0.4, 0.4)

    # upright kneeling: pelvis upright/reclined, thighs ~vertical, knees near the 1.5 rad flexion limit (shins flat)
    r = rows_of("kneel_up")
    k = int(r.sum())
    roll[r], pitch[r] = uni(-0.1, 0.1, k, dev), uni(-0.45, 0.15, k, dev)
    set_lr(r, "knee", 1.25, 1.47)
    set_lr(r, "hip_pitch", -0.5, 0.6)
    set_lr(r, "hip_roll", -0.1, 0.3)
    set_lr(r, "hip_yaw", -0.3, 0.3)

    # torso-down kneeling (hands-and-knees key state): pelvis pitched forward p, hips flexed ~p (thighs ~vertical),
    # knees near the limit, shoulders flexed ~p (arms toward the floor), elbows near straight
    r = rows_of("kneel_down")
    k = int(r.sum())
    if k:
        p = uni(0.9, 1.4, k, dev)
        roll[r], pitch[r] = uni(-0.1, 0.1, k, dev), p
        for side, sg in (("left", 1.0), ("right", -1.0)):
            q[r, jidx[f"{side}_hip_pitch_joint"]] = sg * -(p + uni(-0.3, 0.4, k, dev))
            q[r, jidx[f"{side}_shoulder_pitch_joint"]] = sg * -(p + uni(-0.3, 0.3, k, dev))
        set_lr(r, "knee", 1.2, 1.47)
        set_lr(r, "elbow", 0.0, 0.5)
        set_lr(r, "hip_roll", -0.1, 0.3)

    # Limp-stable recipes (a limp robot only keeps a posture that rests on joint limits or on the ground):
    # side_prop: recovery position, generated LEFT side down, half of them mirrored to right side down. Top leg flexed
    # forward with the knee on the ground, top arm in front, bottom arm forward/overhead: props against rolling.
    r = rows_of("side_prop")
    k = int(r.sum())
    if k:
        rid = r.nonzero().flatten()
        roll[rid], pitch[rid] = -math.pi / 2 + uni(-0.2, 0.2, k, dev), uni(-0.2, 0.4, k, dev)
        q[rid, jidx["right_hip_pitch_joint"]] = uni(1.0, 1.8, k, dev)
        q[rid, jidx["right_knee_joint"]] = -uni(0.8, 1.45, k, dev)
        q[rid, jidx["left_hip_pitch_joint"]] = uni(-0.6, 0.2, k, dev)
        q[rid, jidx["left_knee_joint"]] = uni(0.1, 0.8, k, dev)
        q[rid, jidx["right_shoulder_pitch_joint"]] = uni(0.8, 1.6, k, dev)
        q[rid, jidx["right_elbow_joint"]] = -uni(0.6, 1.8, k, dev)
        q[rid, jidx["left_shoulder_pitch_joint"]] = -uni(1.4, 2.8, k, dev)
        q[rid, jidx["left_elbow_joint"]] = uni(0.0, 1.0, k, dev)
        mir = rid[torch.rand(k, device=dev) < 0.5]
        q[mir] = -q[mir][:, perm]
        roll[mir] = -roll[mir]

    # sit_fold: long sitting folded forward onto the hip-flexion limit (torso ~30 deg forward), thighs flat
    r = rows_of("sit_fold")
    k = int(r.sum())
    if k:
        h = uni(1.85, 2.05, k, dev)
        roll[r], pitch[r] = uni(-0.1, 0.1, k, dev), h - math.pi / 2 + uni(-0.15, 0.1, k, dev)
        q[r, jidx["left_hip_pitch_joint"]] = -h
        q[r, jidx["right_hip_pitch_joint"]] = h + uni(-0.1, 0.05, k, dev)
        set_lr(r, "knee", 0.0, 0.7)
        set_lr(r, "hip_roll", 0.0, 0.4)
        set_lr(r, "hip_yaw", -0.3, 0.3)

    # kneel_hands (posture-held, see SEED_PARAMS hold): hands-and-knees. Torso pitched forward p, thighs ~vertical with
    # the knees 0-0.5 rad behind the hips, knees near the flexion limit, arms ~vertical toward the floor, elbows ~straight.
    # Asimov cannot kneel limp: with the knee limit at 1.5 rad and the toes tucked (ankle +-0.35) the hips sit ahead of
    # the knees, so a limp robot always pitches forward to prone (verified with --trace).
    r = rows_of("kneel_hands")
    k = int(r.sum())
    if k:
        p = uni(0.9, 1.5, k, dev)
        roll[r], pitch[r] = uni(-0.1, 0.1, k, dev), p
        for side, sg in (("left", 1.0), ("right", -1.0)):
            q[r, jidx[f"{side}_hip_pitch_joint"]] = sg * -(p - uni(0.0, 0.5, k, dev))
            q[r, jidx[f"{side}_shoulder_pitch_joint"]] = sg * -(p + uni(-0.2, 0.2, k, dev))
        set_lr(r, "knee", 1.2, 1.48)
        set_lr(r, "elbow", 0.0, 0.4)
        set_lr(r, "hip_roll", -0.05, 0.2)

    quat = math_utils.quat_from_euler_xyz(roll, pitch, yaw)
    # generic: uniform SO(3)
    r = rows_of("generic")
    k = int(r.sum())
    if k:
        g = torch.randn(k, 4, device=dev)
        quat[r] = g / g.norm(dim=1, keepdim=True)

    q = torch.maximum(torch.minimum(q, soft_hi), soft_lo)

    vel_scale = torch.zeros(n, 1, device=dev)
    drop = torch.zeros(n, device=dev)
    for s, name in enumerate(SEED_TYPES):
        rows = seed == s
        vel_scale[rows, 0] = SEED_PARAMS[name]["vel"]
        drop[rows] = uni(*SEED_PARAMS[name]["drop"], int(rows.sum()), dev)
    root_vel = torch.cat(
        [(2 * torch.rand(n, 3, device=dev) - 1) * 0.5, (2 * torch.rand(n, 3, device=dev) - 1) * 1.0], dim=1
    ) * vel_scale
    joint_vel = (2 * torch.rand(n, num_j, device=dev) - 1) * 0.5 * vel_scale
    hold = torch.zeros(n, device=dev)
    for s, name in enumerate(SEED_TYPES):
        rows = seed == s
        if "hold" in SEED_PARAMS[name]:
            hold[rows] = uni(*SEED_PARAMS[name]["hold"], int(rows.sum()), dev)
    return seed, quat, q, joint_vel, root_vel, drop, hold


def save_cache(path: str, store: dict, meta: dict, joint_names: list[str]):
    data = {
        "version": CACHE_FORMAT_VERSION,
        "joint_names": list(joint_names),
        "categories": GETUP_CATEGORIES,
        "root_height": torch.cat(store["root_height"]),
        "root_quat": torch.cat(store["root_quat"]),
        "joint_pos": torch.cat(store["joint_pos"]),
        "joint_vel": torch.cat(store["joint_vel"]),
        "label": torch.cat(store["label"]).to(torch.int8),
        "seed_type": torch.cat(store["seed_type"]).to(torch.int8),
        "features": {k: torch.cat(v) for k, v in store["features"].items()},
        "meta": meta,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    torch.save(data, tmp)
    os.replace(tmp, path)
    return data


def main():
    torch.manual_seed(args_cli.seed)
    out_path = os.path.abspath(os.path.expanduser(args_cli.out))
    t_start = time.time()

    sim = SimulationContext(
        sim_utils.SimulationCfg(
            dt=PHYSICS_DT,
            device=args_cli.device,
            physics_material=GROUND_MATERIAL,
            physx=sim_utils.PhysxCfg(gpu_max_rigid_patch_count=10 * 2**15),
        )
    )
    scene = InteractiveScene(CacheSceneCfg(num_envs=args_cli.num_envs, env_spacing=3.0))
    sim.reset()
    robot = scene["robot"]
    dev = robot.device
    n = robot.num_instances
    shim = types.SimpleNamespace(scene=scene, num_envs=n, device=dev)  # minimal env for set_limp
    spheres = get_collision_spheres(robot)
    jidx = {name: i for i, name in enumerate(robot.joint_names)}
    print(f"[cache] asset={ASSET_NAME} urdf={spheres.source} bodies={robot.num_bodies} joints={robot.num_joints} "
          f"spheres={spheres.num_spheres} envs={n}", flush=True)

    lim = robot.data.joint_pos_limits
    center, half = lim.mean(dim=-1), 0.5 * (lim[..., 1] - lim[..., 0])
    band_lo, band_hi = center - 0.9 * half, center + 0.9 * half  # random joints within 90 % of the range
    soft_lo, soft_hi = robot.data.soft_joint_pos_limits[..., 0], robot.data.soft_joint_pos_limits[..., 1]  # recipes
    perm = torch.tensor(joint_mirror_perm(robot.joint_names), device=dev)

    mix = dict(DEFAULT_SEED_MIX)
    if args_cli.seed_mix:
        for kv in args_cli.seed_mix.split(","):
            key, val = kv.split("=")
            mix[key.strip()] = float(val)
    seed_w = torch.tensor([mix[s] for s in SEED_TYPES], device=dev, dtype=torch.float)

    # storage (optionally continue an existing file)
    store = {k: [] for k in ("root_height", "root_quat", "joint_pos", "joint_vel", "label", "seed_type")}
    store["features"] = {}
    prev_meta = {}
    if args_cli.append and os.path.isfile(out_path):
        old = torch.load(out_path, map_location="cpu", weights_only=False)
        assert list(old["joint_names"]) == list(robot.joint_names), "joint order differs from the existing cache"
        for k in ("root_height", "root_quat", "joint_pos", "joint_vel", "label", "seed_type"):
            store[k].append(old[k])
        for k, v in old["features"].items():
            store["features"][k] = [v]
        prev_meta = old.get("meta", {})
        print(f"[cache] appending to {out_path} ({old['label'].numel()} states)", flush=True)

    num_labels = len(CACHED_CATEGORIES) + 1  # + other
    confusion = torch.zeros(len(SEED_TYPES), num_labels, dtype=torch.long)
    reject = {"nonfinite": 0, "penetration": 0, "moving": 0, "spinning": 0, "joint_moving": 0, "other_label": 0}
    spawned = 0
    accepted_total = sum(int(t.numel()) for t in store["label"])
    accepted_new = 0
    settle_steps = int(round(args_cli.settle_s / PHYSICS_DT))
    all_ids = torch.arange(n, device=dev)
    rnd = 0

    def build_meta():
        labels = torch.cat(store["label"]) if store["label"] else torch.zeros(0)
        counts = {c: int((labels == i).sum()) for i, c in enumerate(CACHED_CATEGORIES)}
        return {
            "builder": "scripts/getup/build_fallen_cache.py",
            "asset": ASSET_NAME,
            "urdf": str(robot.cfg.spawn.asset_path),
            "args": vars(args_cli) | {"seed_mix_resolved": mix},
            "seed_types": SEED_TYPES,
            "body_names": list(robot.body_names),
            "counts": counts,
            "this_run": {
                "spawned": spawned,
                "accepted": accepted_new,
                "rejections": dict(reject),
                "confusion_seed_x_label": confusion.tolist(),
                "confusion_cols": list(CACHED_CATEGORIES) + ["other"],
                "rounds": rnd,
                "wall_s": time.time() - t_start,
            },
            "previous": prev_meta,
            "thresholds": {
                "max_root_speed": MAX_ROOT_SPEED,
                "max_root_ang_speed": MAX_ROOT_ANG_SPEED,
                "max_joint_speed": MAX_JOINT_SPEED,
                "max_penetration": MAX_PENETRATION,
            },
            "settle_s": args_cli.settle_s,
            "physics_dt": PHYSICS_DT,
            "host": socket.gethostname(),
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

    while rnd < args_cli.max_rounds and accepted_total < args_cli.target_states:
        if (time.time() - t_start) / 60.0 > args_cli.max_minutes:
            print("[cache] time budget reached", flush=True)
            break
        rnd += 1
        t0 = time.time()
        seed, quat, q, qd, root_vel, drop, hold = sample_spawn(robot, seed_w, band_lo, band_hi, soft_lo, soft_hi, jidx, perm)

        # write the pose high above the ground, then place the lowest collision point at `drop`
        scene.reset()
        pos = scene.env_origins.clone()
        pos[:, 2] += 2.0
        robot.write_root_pose_to_sim(torch.cat([pos, quat], dim=-1))
        robot.write_joint_state_to_sim(q, qd)
        centers = spheres.centers_w(robot.data.body_link_pos_w, robot.data.body_link_quat_w)
        lowest = (centers[..., 2] - spheres.radii).amin(dim=1)
        pos[:, 2] += drop - lowest
        robot.write_root_pose_to_sim(torch.cat([pos, quat], dim=-1))
        robot.write_root_velocity_to_sim(root_vel)
        robot.set_joint_position_target(q)
        robot.set_joint_velocity_target(torch.zeros_like(qd))
        set_limp(shim, all_ids, True)
        if bool((hold > 0).any()):  # posture hold toward the spawn pose (limpable actuators only)
            held = False
            for act in robot.actuators.values():
                if isinstance(getattr(act, "gain_scale", None), torch.Tensor):
                    act.gain_scale[:] = hold
                    held = True
            if not held:
                hold.zero_()

        for step_i in range(settle_steps):
            if args_cli.trace and step_i % 50 == 0:
                tq, ph, _, kh, _, _ = compute_label_inputs(robot, spheres, None)
                ex = torch.tensor([1.0, 0.0, 0.0], device=dev).expand(tq.shape[0], 3)
                ez = torch.tensor([0.0, 0.0, 1.0], device=dev).expand(tq.shape[0], 3)
                fz, uz = math_utils.quat_apply(tq, ex)[:, 2], math_utils.quat_apply(tq, ez)[:, 2]
                jp = robot.data.joint_pos
                for e in range(args_cli.trace):
                    print(f"[trace] t={step_i * PHYSICS_DT:.2f} env{e} seed={SEED_TYPES[int(seed[e])]} pelvis_h={ph[e]:.3f} "
                          f"up_z={uz[e]:+.2f} chest_z={fz[e]:+.2f} knee_cap_h=({kh[e, 0]:.3f},{kh[e, 1]:.3f}) "
                          f"hipL={jp[e, jidx['left_hip_pitch_joint']]:+.2f} kneeL={jp[e, jidx['left_knee_joint']]:+.2f} "
                          f"ankL={jp[e, jidx['left_ankle_pitch_joint']]:+.2f}", flush=True)
            scene.write_data_to_sim()
            sim.step(render=False)
            scene.update(PHYSICS_DT)

        # capture
        root_pos = robot.data.root_pos_w - scene.env_origins
        root_quat = robot.data.root_quat_w
        lin = robot.data.root_lin_vel_w.norm(dim=-1)
        ang = robot.data.root_ang_vel_w.norm(dim=-1)
        jpos, jvel = robot.data.joint_pos, robot.data.joint_vel
        finite = torch.isfinite(torch.cat([root_pos, root_quat, jpos, jvel], dim=-1)).all(dim=-1)
        torso_q, pelvis_h, body_low, knee_h, body_index, min_bottom = compute_label_inputs(robot, spheres, None)
        labels, feats = label_fallen_states(torso_q, pelvis_h, body_low, knee_h, body_index)

        bad_pen = finite & (min_bottom < -MAX_PENETRATION)
        bad_mov = finite & ~bad_pen & (lin > MAX_ROOT_SPEED)
        bad_spin = finite & ~bad_pen & ~bad_mov & (ang > MAX_ROOT_ANG_SPEED)
        bad_jnt = finite & ~bad_pen & ~bad_mov & ~bad_spin & (jvel.abs().amax(dim=-1) > MAX_JOINT_SPEED)
        physical_ok = finite & ~bad_pen & ~bad_mov & ~bad_spin & ~bad_jnt
        other = physical_ok & (labels < 0)
        ok = physical_ok & (labels >= 0)
        reject["nonfinite"] += int((~finite).sum())
        reject["penetration"] += int(bad_pen.sum())
        reject["moving"] += int(bad_mov.sum())
        reject["spinning"] += int(bad_spin.sum())
        reject["joint_moving"] += int(bad_jnt.sum())
        reject["other_label"] += int(other.sum())
        col = torch.where(labels < 0, torch.full_like(labels, num_labels - 1), labels)
        for s in range(len(SEED_TYPES)):
            sel = physical_ok & (seed == s)
            confusion[s] += torch.bincount(col[sel].cpu(), minlength=num_labels)
        spawned += n

        k = ok.nonzero().flatten()
        store["root_height"].append(root_pos[k, 2].cpu())
        store["root_quat"].append(root_quat[k].cpu())
        store["joint_pos"].append(jpos[k].cpu())
        store["joint_vel"].append(jvel[k].cpu())
        store["label"].append(labels[k].to(torch.int8).cpu())
        store["seed_type"].append(seed[k].to(torch.int8).cpu())
        feats = dict(feats) | {"min_bottom": min_bottom, "root_speed": lin, "hold_scale": hold}
        n_before = sum(int(t.numel()) for t in store["label"][:-1])
        for name, v in feats.items():
            if name not in store["features"] and n_before > 0:  # feature missing in an appended older cache
                store["features"][name] = [torch.zeros(n_before, dtype=v.dtype)]
            store["features"].setdefault(name, []).append(v[k].cpu())
        accepted_new += int(k.numel())
        accepted_total += int(k.numel())
        lab_counts = torch.bincount(labels[k].cpu(), minlength=len(CACHED_CATEGORIES)).tolist()
        print(
            f"[cache] round {rnd}: accepted {k.numel()}/{n} (total {accepted_total}) per-cat {lab_counts} "
            f"| {time.time() - t0:.1f}s",
            flush=True,
        )
        if rnd % args_cli.save_every == 0:
            save_cache(out_path, store, build_meta(), robot.joint_names)

    data = save_cache(out_path, store, build_meta(), robot.joint_names)
    meta = data["meta"]
    print("=" * 100)
    print(f"[cache] saved {data['label'].numel()} states -> {out_path}")
    print(f"[cache] per-category counts: {meta['counts']}")
    tr = meta["this_run"]
    rej_total = tr["spawned"] - tr["accepted"]
    print(f"[cache] this run: spawned {tr['spawned']}, accepted {tr['accepted']} "
          f"({100.0 * tr['accepted'] / max(tr['spawned'], 1):.1f} %), rejected {rej_total} "
          f"({100.0 * rej_total / max(tr['spawned'], 1):.1f} %): {tr['rejections']}")
    print("[cache] seed type x label (physically valid states):")
    print("  " + " ".join(f"{c:>10s}" for c in ["seed"] + tr["confusion_cols"]))
    for s, row in zip(SEED_TYPES, tr["confusion_seed_x_label"]):
        print("  " + f"{s:>10s} " + " ".join(f"{v:>10d}" for v in row))
    f = data["features"]
    for i, c in enumerate(CACHED_CATEGORIES):
        sel = data["label"].long() == i
        if sel.any():
            print(f"[cache] {c:>10s}: n={int(sel.sum()):6d} pelvis h mean {f['pelvis_height'][sel].mean():.3f} "
                  f"[{f['pelvis_height'][sel].min():.3f}, {f['pelvis_height'][sel].max():.3f}] "
                  f"torso up_z mean {f['torso_up_z'][sel].mean():+.2f} min_bottom mean {f['min_bottom'][sel].mean() * 1000:+.1f} mm")
    print(f"[cache] wall time {time.time() - t_start:.0f} s")


if __name__ == "__main__":
    # Exit without simulation_app.close(): it can spin for > 10 min on a headless server while holding the GPU.
    # All outputs are written and flushed before this point.
    exit_code = 0
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        exit_code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
