# Copyright (c) 2026, Menlo Research.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Simulation test for the get-up reset event.

Builds a minimal ManagerBasedRLEnv (plane, get-up asset, `reset_fallen` event with round-robin categories), resets
64 envs (8 per category), then steps 2 s with zero actions:

* cached categories are kept limp (they must stay at rest: no popping, no penetration),
* ``mid_fall`` follows its limp schedule (released by ``update_limp_schedule`` at ``control_start_step``),
* ``standing`` holds the default pose with zero actions.

Prints per-category evidence (label agreement, penetration, heights, tilts, speeds, limp timing) and, with
``--enable_cameras``, saves PNG frames + a montage to ``--frames_dir``.

    python scripts/getup/test_resets.py --headless --enable_cameras \
        --cache ~/getup_cache/fallen_v0.pt
"""

from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Test the get-up reset event.")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--cache", type=str, default="~/getup_cache/fallen_v0.pt")
parser.add_argument("--duration_s", type=float, default=2.0)
parser.add_argument("--frames_dir", type=str, default="~/getup_results/reset_frames")
parser.add_argument("--asset", choices=("getup", "walking"), default="getup")
parser.add_argument("--seed", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402

import isaaclab.envs.mdp as mdp  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import ArticulationCfg, AssetBaseCfg  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg  # noqa: E402
from isaaclab.managers import EventTermCfg as EventTerm  # noqa: E402
from isaaclab.managers import ObservationGroupCfg as ObsGroup  # noqa: E402
from isaaclab.managers import ObservationTermCfg as ObsTerm  # noqa: E402
from isaaclab.managers import TerminationTermCfg as DoneTerm  # noqa: E402
from isaaclab.scene import InteractiveSceneCfg  # noqa: E402
from isaaclab.terrains import TerrainImporterCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

from isaac_asimov.tasks.getup.mdp.limp import is_limp, set_limp, update_limp_schedule  # noqa: E402
from isaac_asimov.tasks.getup.mdp.resets import (  # noqa: E402
    CACHED_CATEGORIES,
    GETUP_CATEGORIES,
    MID_FALL,
    STANDING,
    _terrain_height,
    compute_label_inputs,
    get_collision_spheres,
    get_state,
    ground_penetration,
    label_fallen_states,
    reset_fallen_state,
    reweight_category_probs,
)

if args_cli.asset == "getup":
    from isaac_asimov.assets.robots.asimov_1 import ASIMOV_1_GETUP_CFG as ROBOT_CFG
else:
    from isaac_asimov.assets.robots.asimov_1 import ASIMOV_1_DELAYED_CFG as ROBOT_CFG

GROUND = sim_utils.RigidBodyMaterialCfg(
    friction_combine_mode="multiply", restitution_combine_mode="multiply", static_friction=1.0, dynamic_friction=1.0
)


@configclass
class SceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(prim_path="/World/ground", terrain_type="plane", collision_group=-1, physics_material=GROUND)
    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight", spawn=sim_utils.DomeLightCfg(intensity=750.0, color=(0.9, 0.9, 0.9))
    )


@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True)


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventsCfg:
    reset_fallen = EventTerm(
        func=reset_fallen_state,
        mode="reset",
        params={
            "category_probs": {c: 1.0 / len(GETUP_CATEGORIES) for c in GETUP_CATEGORIES},
            "cache_path": "~/getup_cache/fallen_v0.pt",
            "assignment": "round_robin",
        },
    )


@configclass
class RewardsCfg:
    pass


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class TestEnvCfg(ManagerBasedRLEnvCfg):
    scene: SceneCfg = SceneCfg(num_envs=64, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventsCfg = EventsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 1000.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = GROUND
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        self.viewer.origin_type = "env"
        self.viewer.eye = (1.25, 1.25, 0.85)
        self.viewer.lookat = (0.0, 0.0, 0.2)
        self.viewer.resolution = (480, 360)


def tilt_of(quat):
    return torch.acos((1 - 2 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)).clamp(-1, 1))


def main():
    torch.manual_seed(args_cli.seed)
    cfg = TestEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.events.reset_fallen.params["cache_path"] = args_cli.cache
    render = bool(getattr(args_cli, "enable_cameras", False))
    env = ManagerBasedRLEnv(cfg, render_mode="rgb_array" if render else None)
    robot = env.scene["robot"]
    dev = env.device
    spheres = get_collision_spheres(robot)
    terrain = _terrain_height(env)

    env.reset()
    st = get_state(env)
    cat = st.category.clone()
    all_ids = torch.arange(env.num_envs, device=dev)
    print("=" * 100)
    print(f"[test] asset={'ASIMOV_1_GETUP_CFG' if args_cli.asset == 'getup' else 'walking'} envs={env.num_envs} "
          f"step_dt={env.step_dt} cache={args_cli.cache}")
    print(f"[test] category per env: {cat.tolist()}")

    # ---- t = 0 evidence -------------------------------------------------------------------------------------------
    pen0 = ground_penetration(robot, all_ids, spheres, terrain)
    torso_q, pelvis_h, body_low, knee_h, body_index, _ = compute_label_inputs(robot, spheres, terrain)
    labels, _ = label_fallen_states(torso_q, pelvis_h, body_low, knee_h, body_index)
    tilt0 = tilt_of(torso_q)
    pos0 = robot.data.root_pos_w.clone()
    csteps = st.control_start_step.clone()
    print(f"[test] mid_fall control_start_step (steps): {csteps[cat == MID_FALL].tolist()} "
          f"-> limp {csteps[cat == MID_FALL].min().item() * env.step_dt:.2f}..{csteps[cat == MID_FALL].max().item() * env.step_dt:.2f} s")
    print(f"[test] limp at t0 per category: " + ", ".join(
        f"{c}={int(is_limp(env)[cat == i].sum())}/{int((cat == i).sum())}" for i, c in enumerate(GETUP_CATEGORIES)))
    gs = [a.gain_scale for a in robot.actuators.values() if hasattr(a, "gain_scale")]
    if gs:
        print(f"[test] gain_scale at t0: mid_fall {gs[0][cat == MID_FALL].tolist()} | others max "
              f"{gs[0][cat != MID_FALL].max().item()} min {gs[0][cat != MID_FALL].min().item()}")
    else:
        print("[test] actuators have no gain_scale (fallback limp path via stiffness/damping)")

    # keep cached categories limp for the rest check (the reset turned them to policy control)
    # (control_start_step is pushed out of reach so update_limp_schedule does not release them)
    cached_mask = cat < len(CACHED_CATEGORIES)
    set_limp(env, all_ids[cached_mask], True)
    st.control_start_step[cached_mask] = 10**9

    # ---- frames --------------------------------------------------------------------------------------------------
    frames = {}
    show_env = {i: int((cat == i).nonzero()[0]) for i in range(len(GETUP_CATEGORIES)) if (cat == i).any()}

    warm = {"done": False}

    def grab(tag):
        if not render:
            return
        if not warm["done"]:  # the first RTX frames (and the first annotator read) come out black
            for _ in range(30):
                env.sim.render()
            env.render()
            warm["done"] = True
        for i, e in show_env.items():
            env.viewport_camera_controller.set_view_env_index(e)
            env.viewport_camera_controller.update_view_location()
            for _ in range(4):
                env.sim.render()
            img = env.render()
            if img is not None:
                frames[(tag, i)] = img.copy()

    grab("t0.0")

    # ---- step ----------------------------------------------------------------------------------------------------
    steps = int(round(args_cli.duration_s / env.step_dt))
    zero = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=dev)
    max_speed_early = torch.zeros(env.num_envs, device=dev)
    released_at = torch.full((env.num_envs,), -1, dtype=torch.long, device=dev)
    capture_at = {int(round(0.5 / env.step_dt)): "t0.5", steps: f"t{args_cli.duration_s:.1f}"}
    for k in range(1, steps + 1):
        was_limp = is_limp(env).clone()
        active = update_limp_schedule(env)
        st.policy_active[:] = active
        newly = was_limp & ~is_limp(env)
        released_at[newly] = st.step[newly]
        env.step(zero)
        st.step += 1
        if k <= 10:
            max_speed_early = torch.maximum(max_speed_early, robot.data.root_lin_vel_w.norm(dim=-1))
        if k in capture_at:
            grab(capture_at[k])

    # ---- t = end evidence -----------------------------------------------------------------------------------------
    pos1 = robot.data.root_pos_w
    drift = (pos1 - pos0).norm(dim=-1)
    pen1 = ground_penetration(robot, all_ids, spheres, terrain)
    tq1, ph1, bl1, kh1, bi1, _ = compute_label_inputs(robot, spheres, terrain)
    labels1, _ = label_fallen_states(tq1, ph1, bl1, kh1, bi1)
    tilt1 = tilt_of(tq1)
    speed1 = robot.data.root_lin_vel_w.norm(dim=-1)

    print("-" * 100)
    hdr = (f"{'category':>10s} {'n':>3s} {'label==cat':>10s} {'pen0 mm':>8s} {'pelvis h0':>9s} {'tilt0':>6s} "
           f"{'vmax(0-0.2s)':>12s} {'drift m':>8s} {'pelvis h2s':>10s} {'tilt2s':>6s} {'v2s':>6s} {'label2s==cat':>12s}")
    print(hdr)
    for i, c in enumerate(GETUP_CATEGORIES):
        sel = cat == i
        if not sel.any():
            continue
        agree = (labels[sel] == i).float().mean().item() if i < len(CACHED_CATEGORIES) else float("nan")
        agree1 = (labels1[sel] == i).float().mean().item() if i < len(CACHED_CATEGORIES) else float("nan")
        print(f"{c:>10s} {int(sel.sum()):>3d} {agree:>10.2f} {pen0[sel].max().item() * 1000:>8.1f} "
              f"{pelvis_h[sel].mean().item():>9.3f} {tilt0[sel].mean().item():>6.2f} "
              f"{max_speed_early[sel].max().item():>12.3f} {drift[sel].max().item():>8.3f} "
              f"{ph1[sel].mean().item():>10.3f} {tilt1[sel].mean().item():>6.2f} {speed1[sel].max().item():>6.3f} "
              f"{agree1:>12.2f}")
    mid = cat == MID_FALL
    ok_timing = bool((released_at[mid] == csteps[mid]).all())
    print(f"[test] mid_fall limp released at step == control_start_step for all: {ok_timing} "
          f"(released {released_at[mid].tolist()} vs start {csteps[mid].tolist()})")
    print(f"[test] mid_fall pelvis h at 2 s: {[round(v, 3) for v in ph1[mid].tolist()]}")
    print(f"[test] standing pelvis h at 2 s: {[round(v, 3) for v in ph1[cat == STANDING].tolist()]}")
    print(f"[test] max penetration at t0 over all envs: {pen0.max().item() * 1000:.2f} mm, at 2 s: {pen1.max().item() * 1000:.2f} mm")

    # ---- reweighting hook sanity ----------------------------------------------------------------------------------
    succ = {"supine": 0.9, "prone": 0.5, "side_left": 0.2, "side_right": 0.2, "sitting": 1.0, "kneeling": 0.0,
            "mid_fall": 0.7, "standing": 1.0}
    new = reweight_category_probs(dict(cfg.events.reset_fallen.params["category_probs"]), succ)
    print(f"[test] reweight example (success {succ}) -> {{{', '.join(f'{k}: {v:.3f}' for k, v in new.items())}}} "
          f"sum={sum(new.values()):.3f}")

    from isaac_asimov.tasks.getup.mdp.resets import load_fallen_cache

    for inc in (True, False):
        c = load_fallen_cache(args_cli.cache, robot.joint_names, dev, include_held=inc)
        print(f"[test] cache include_held={inc}: {c.num_states} states "
              f"{ {GETUP_CATEGORIES[i]: int(c.count[i]) for i in range(len(CACHED_CATEGORIES))} }")

    # ---- save frames ----------------------------------------------------------------------------------------------
    if frames:
        from PIL import Image, ImageDraw

        out_dir = os.path.expanduser(args_cli.frames_dir)
        os.makedirs(out_dir, exist_ok=True)
        tags = sorted({t for t, _ in frames}, key=lambda s: float(s[1:]))
        cats = sorted(show_env)
        h, w = next(iter(frames.values())).shape[:2]
        sw, sh = w // 2, h // 2  # 240 x 180 thumbnails
        mont = Image.new("RGB", (sw * len(cats), sh * len(tags)), (255, 255, 255))
        for r, t in enumerate(tags):
            for col, i in enumerate(cats):
                if (t, i) not in frames:
                    continue
                im = Image.fromarray(frames[(t, i)][..., :3])
                im.save(os.path.join(out_dir, f"{GETUP_CATEGORIES[i]}_{t}.png"))
                small = im.resize((sw, sh))
                ImageDraw.Draw(small).text((4, 4), f"{GETUP_CATEGORIES[i]} {t}s", fill=(0, 0, 0))
                mont.paste(small, (col * sw, r * sh))
        mont.convert("P", palette=Image.ADAPTIVE, colors=128).save(os.path.join(out_dir, "montage.png"), optimize=True)
        print(f"[test] saved {len(frames)} frames + montage.png to {out_dir}")
    # env.close() skipped: see the exit note below


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
