# Copyright (c) 2026, Menlo Research.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Validate the Asimov 1 get-up asset.

Checks, in order:
  1. collision report: every collision prim per body, confirming the patched wrist-stub and upper-arm geoms;
  2. joint position / soft / velocity limits as seen by Isaac Lab and PhysX (get-up and walking assets);
  3. actuator probe: gain_scale (nominal / half / limp / 1.5x) and set_effort_scale produce the expected torques and
     those torques reach PhysX;
  4. Stage B probe: torque-speed curve and ankle-differential motor clip;
  5. joint reach: hip pitch 2.09 and knee 1.5 reachable (fixed base);
  6. velocity limits: are the 3.98 rad/s URDF limits enforced by PhysX (get-up and walking assets);
  7. limp drop test from 0.6 m with random orientation and joints: settles, no explosion; then responds to gains.

Usage:
    python scripts/getup/validate_asset.py --headless --num_envs 32
    python scripts/getup/validate_asset.py --meshes_only
"""

import argparse
import os
import sys

parser = argparse.ArgumentParser(description="Validate the Asimov 1 get-up asset.")
parser.add_argument("--num_envs", type=int, default=32, help="Number of envs (<= 64).")
parser.add_argument("--drop_time", type=float, default=3.0, help="Limp drop duration [s].")
parser.add_argument("--pos_iters", type=int, default=None, help="Override solver position iterations.")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--only_drop", action="store_true", help="Skip everything but the drop test.")
parser.add_argument("--meshes_only", action="store_true", help="Only measure the STL meshes (no Isaac Sim).")


def measure_meshes() -> None:
    import numpy as np
    import trimesh

    d = os.path.join(os.environ["ASIMOV_1_MODEL_DIR"], "assets", "meshes")
    ax = np.array([0.7660444431, 0.0, -0.6427876097])
    for side in ("LEFT", "RIGHT"):
        v = trimesh.load(os.path.join(d, f"{side}_WRIST_YAW.STL")).vertices
        s = v @ ax
        r = np.linalg.norm(v - np.outer(s, ax), axis=1)
        print(f"[mesh] {side}_WRIST_YAW bbox ext {np.ptp(v, 0).round(4)} | along wrist axis s=[{s.min():.4f},"
              f" {s.max():.4f}] r_max={r.max():.4f}")
        v = trimesh.load(os.path.join(d, f"{side}_SHOULDER_YAW.STL")).vertices
        print(f"[mesh] {side}_SHOULDER_YAW bounds {v.min(0).round(4)} .. {v.max(0).round(4)};"
              f" r_max(xy)={np.linalg.norm(v[:, :2], axis=1).max():.4f}")
    v = trimesh.load(os.path.join(d, "WAIST_YAW.STL")).vertices
    print(f"[mesh] WAIST_YAW (torso) bounds {v.min(0).round(4)} .. {v.max(0).round(4)}")
    coverage_audit(d)


def _rpy(r, p, y):
    import numpy as np

    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def coverage_audit(mesh_dir: str) -> None:
    """Distance of every visual-mesh vertex to the union of the link's collision primitives (stock vs patched)."""
    import importlib.util
    import xml.etree.ElementTree as ET

    import numpy as np
    import trimesh

    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location(
        "getup_urdf", os.path.join(here, "../../source/isaac_asimov/isaac_asimov/assets/robots/getup_urdf.py")
    )
    gu = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gu)
    src = os.path.join(os.environ["ASIMOV_1_MODEL_DIR"], "urdf", "asimov_1.urdf")
    stock = ET.parse(src).getroot()
    patched = ET.parse(gu.build_getup_urdf(src)).getroot()

    def dist(link, pts):
        d = np.full(len(pts), np.inf)
        for c in link.findall("collision"):
            o = c.find("origin")
            xyz = np.array([float(x) for x in o.get("xyz", "0 0 0").split()]) if o is not None else np.zeros(3)
            rpy = [float(x) for x in o.get("rpy", "0 0 0").split()] if o is not None else [0, 0, 0]
            loc = (pts - xyz) @ _rpy(*rpy)
            g = c.find("geometry")[0]
            if g.tag == "sphere":
                dd = np.maximum(np.linalg.norm(loc, axis=1) - float(g.get("radius")), 0)
            elif g.tag == "cylinder":
                r = np.linalg.norm(loc[:, :2], axis=1)
                dd = np.hypot(np.maximum(r - float(g.get("radius")), 0),
                              np.maximum(np.abs(loc[:, 2]) - 0.5 * float(g.get("length")), 0))
            elif g.tag == "box":
                h = 0.5 * np.array([float(x) for x in g.get("size").split()])
                dd = np.linalg.norm(np.maximum(np.abs(loc) - h, 0), axis=1)
            else:
                continue
            d = np.minimum(d, dd)
        return d

    print("[audit] visual-mesh vertices vs collision primitives (link frame); frac>2cm = share of vertices more than")
    print("[audit] 2 cm outside every collision shape; max = worst gap [m]. 'inf' = link has no collision.")
    print(f"[audit] {'link':26s} {'stock frac>2cm':>14s} {'stock max':>9s} {'patched frac>2cm':>16s} {'patched max':>11s}")
    for lk in stock.findall("link"):
        vis = lk.find("visual")
        if vis is None:
            continue
        mesh = vis.find("geometry/mesh")
        if mesh is None:
            continue
        v = trimesh.load(os.path.join(mesh_dir, os.path.basename(mesh.get("filename")))).vertices
        o = vis.find("origin")
        if o is not None:
            v = v @ _rpy(*[float(x) for x in o.get("rpy", "0 0 0").split()]).T + np.array(
                [float(x) for x in o.get("xyz", "0 0 0").split()])
        name = lk.get("name")
        ds = dist(lk, v)
        dp = dist(patched.find(f"link[@name='{name}']"), v)
        print(f"[audit] {name:26s} {np.mean(ds > 0.02):14.1%} {ds.max():9.3f} {np.mean(dp > 0.02):16.1%} {dp.max():11.3f}")


def profile_meshes(names: list[str], nz: int = 14) -> None:
    """Print z-slices (link frame) of each mesh: x and y extents per slice, to size collision primitives."""
    import numpy as np
    import trimesh

    d = os.path.join(os.environ["ASIMOV_1_MODEL_DIR"], "assets", "meshes")
    for n in names:
        m = trimesh.load(os.path.join(d, f"{n}.STL"))
        # sample the surface densely so large flat faces are represented, not only vertices
        v = np.concatenate([m.vertices, m.sample(60000)])
        print(f"[profile] {n}: bounds {v.min(0).round(4)} .. {v.max(0).round(4)}")
        z = v[:, 2]
        edges = np.linspace(z.min(), z.max(), nz + 1)
        for i in range(nz):
            k = (z >= edges[i]) & (z <= edges[i + 1])
            if k.sum() < 5:
                continue
            x, y = v[k, 0], v[k, 1]
            px = np.percentile(x, [1, 99]).round(3)
            py = np.percentile(y, [1, 99]).round(3)
            print(f"[profile]   z {edges[i]:+.3f}..{edges[i + 1]:+.3f}  x[{x.min():+.3f},{x.max():+.3f}] (p1-99 {px})"
                  f"  y[{y.min():+.3f},{y.max():+.3f}] (p1-99 {py})")


if "--profile" in sys.argv:
    profile_meshes(sys.argv[sys.argv.index("--profile") + 1].split(","))
    sys.exit(0)

if "--meshes_only" in sys.argv:
    measure_meshes()
    sys.exit(0)

from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.num_envs = min(args_cli.num_envs, 64)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import re  # noqa: E402
import xml.etree.ElementTree as ET  # noqa: E402

import torch  # noqa: E402
from pxr import Usd, UsdGeom, UsdPhysics  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import ArticulationCfg, AssetBaseCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sensors import ContactSensorCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

from isaac_asimov.assets.robots import asimov_1 as A  # noqa: E402
from isaac_asimov.assets.robots import getup_actuators as GA  # noqa: E402
from isaac_asimov.assets.robots.getup_urdf import GEOM_TAG  # noqa: E402

torch.manual_seed(args_cli.seed)
DT = 0.005


def _fixed(cfg: ArticulationCfg, y: float, zero_delay: bool = True) -> ArticulationCfg:
    acts = cfg.actuators
    if zero_delay:
        acts = {k: v.replace(min_delay=0, max_delay=0) for k, v in acts.items()}
    return cfg.replace(
        spawn=cfg.spawn.replace(fix_base=True),
        init_state=cfg.init_state.replace(pos=(0.0, y, 1.3)),
        actuators=acts,
    )


GETUP = A.ASIMOV_1_GETUP_CFG
if args_cli.pos_iters is not None:
    GETUP = GETUP.replace(
        spawn=GETUP.spawn.replace(
            articulation_props=GETUP.spawn.articulation_props.replace(solver_position_iteration_count=args_cli.pos_iters)
        )
    )


@configclass
class SceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0)
        ),
    )
    light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=2000.0))
    robot: ArticulationCfg = GETUP.replace(prim_path="{ENV_REGEX_NS}/Robot")
    fix: ArticulationCfg = _fixed(GETUP, 1.5).replace(prim_path="{ENV_REGEX_NS}/Fix")
    dcfix: ArticulationCfg = _fixed(A.ASIMOV_1_GETUP_DC_CFG, -1.5).replace(prim_path="{ENV_REGEX_NS}/DcFix")
    walk: ArticulationCfg = _fixed(A.ASIMOV_1_DELAYED_CFG, 3.0, zero_delay=False).replace(prim_path="{ENV_REGEX_NS}/Walk")
    contacts = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=1)
    # Fix hangs in the air, so any contact it reports is self-collision
    fix_contacts = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Fix/.*", history_length=1)


def hdr(title: str) -> None:
    print("\n" + "=" * 100 + f"\n== {title}\n" + "=" * 100, flush=True)


def step(sim, scene, n: int = 1):
    for _ in range(n):
        scene.write_data_to_sim()
        sim.step(render=False)
        scene.update(DT)


# ---------------------------------------------------------------------------------------------------------------------
# 1. collision report
# ---------------------------------------------------------------------------------------------------------------------


def collision_report(stage, robot_path: str, urdf_path: str) -> None:
    hdr(f"1. Collision report  {robot_path}  (urdf {urdf_path})")
    root = stage.GetPrimAtPath(robot_path)
    by_link: dict[str, list[str]] = {}
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        # owning rigid body
        p = prim
        while p and not p.HasAPI(UsdPhysics.RigidBodyAPI):
            p = p.GetParent()
        link = p.GetName() if p else "?"
        t = prim.GetTypeName()
        desc = t
        if t == "Cylinder":
            g = UsdGeom.Cylinder(prim)
            desc += f"(r={g.GetRadiusAttr().Get():.4f}, h={g.GetHeightAttr().Get():.4f}, axis={g.GetAxisAttr().Get()})"
        elif t == "Sphere":
            desc += f"(r={UsdGeom.Sphere(prim).GetRadiusAttr().Get():.4f})"
        elif t == "Capsule":
            g = UsdGeom.Capsule(prim)
            desc += f"(r={g.GetRadiusAttr().Get():.4f}, h={g.GetHeightAttr().Get():.4f})"
        xf = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        pw = UsdGeom.Xformable(p).ComputeLocalToWorldTransform(Usd.TimeCode.Default()) if p else xf
        rel = (xf * pw.GetInverse()).ExtractTranslation()
        desc += f" @link({rel[0]:+.4f},{rel[1]:+.4f},{rel[2]:+.4f}) [{prim.GetName()}]"
        by_link.setdefault(link, []).append(desc)
    # stock counts from the stock URDF
    stock = ET.parse(A.ASIMOV_1_URDF_PATH).getroot()
    stock_n = {lk.get("name"): len(lk.findall("collision")) for lk in stock.findall("link")}
    patched = ET.parse(urdf_path).getroot()
    new_geoms = {
        lk.get("name"): [c.get("name") for c in lk.findall("collision") if (c.get("name") or "").startswith(GEOM_TAG)]
        for lk in patched.findall("link")
    }
    total = 0
    for link in sorted(by_link):
        n = len(by_link[link])
        total += n
        tag = ""
        if new_geoms.get(link):
            tag = f"   <-- NEW {new_geoms[link]} (stock {stock_n.get(link, 0)})"
        print(f"  {link:28s} {n:2d} shapes{tag}")
        for d in by_link[link]:
            print(f"      {d}")
    print(f"  total collision prims: {total}; stock URDF collisions: {sum(stock_n.values())};"
          f" patched URDF collisions: {sum(len(lk.findall('collision')) for lk in patched.findall('link'))}")
    for link in ("left_wrist_yaw_link", "right_wrist_yaw_link", "left_shoulder_yaw_link", "right_shoulder_yaw_link"):
        ok = len(by_link.get(link, [])) >= 1
        print(f"  CHECK {link} has collision: {'PASS' if ok else 'FAIL'}")


# ---------------------------------------------------------------------------------------------------------------------
# 2. limits
# ---------------------------------------------------------------------------------------------------------------------


def limits_report(name: str, art) -> None:
    hdr(f"2. Joint limits: {name}")
    lim = art.data.joint_pos_limits[0]
    soft = art.data.soft_joint_pos_limits[0]
    vlim = art.data.joint_vel_limits[0]
    try:
        physx_v = art.root_physx_view.get_dof_max_velocities()[0]
    except Exception as e:  # noqa: BLE001
        physx_v = None
        print(f"  (get_dof_max_velocities unavailable: {e})")
    eff_sim = art.data.joint_effort_limits[0]
    act_eff = GA.get_effort_limits(art)[0] if name != "walk" else None
    print(f"  {'joint':28s} {'lower':>8s} {'upper':>8s} {'soft_lo':>8s} {'soft_hi':>8s} {'vel_lim':>8s}"
          f" {'physx_v':>8s} {'eff_sim':>9s} {'act_eff':>8s}")
    for j, n in enumerate(art.joint_names):
        pv = f"{physx_v[j].item():8.3f}" if physx_v is not None else "     n/a"
        ae = f"{act_eff[j].item():8.2f}" if act_eff is not None else "     n/a"
        print(f"  {n:28s} {lim[j, 0].item():8.4f} {lim[j, 1].item():8.4f} {soft[j, 0].item():8.4f}"
              f" {soft[j, 1].item():8.4f} {vlim[j].item():8.3f} {pv} {eff_sim[j].item():9.2e} {ae}")
    if name == "getup":
        for n in ("left_hip_pitch_joint", "right_hip_pitch_joint", "left_knee_joint", "right_knee_joint"):
            j = art.joint_names.index(n)
            print(f"  {n}: hard [{lim[j, 0]:.4f}, {lim[j, 1]:.4f}] soft [{soft[j, 0]:.4f}, {soft[j, 1]:.4f}]")


# ---------------------------------------------------------------------------------------------------------------------
# 3. actuator probe (no stepping: compute torques for a known state and read them back from PhysX)
# ---------------------------------------------------------------------------------------------------------------------


def _probe(art, dq: float, qd: float):
    n = art.num_instances
    q0 = art.data.default_joint_pos.clone()
    lo, hi = art.data.joint_pos_limits[..., 0], art.data.joint_pos_limits[..., 1]
    q0 = torch.clamp(q0, lo + 0.2, hi - 0.2)
    art.write_joint_state_to_sim(q0, torch.full_like(q0, qd))
    art.set_joint_position_target(q0 + dq)
    art.set_joint_velocity_target(torch.zeros_like(q0))
    art.set_joint_effort_target(torch.zeros_like(q0))
    art.write_data_to_sim()
    physx = art.root_physx_view.get_dof_actuation_forces().clone()
    assert physx.shape[0] == n
    return art.data.computed_torque.clone(), art.data.applied_torque.clone(), physx


def actuator_probe(art) -> None:
    hdr("3a. gain_scale probe (Fix, zero delay): state q0, qd=+1 rad/s, target q0+0.1, qd_des=0")
    n = art.num_instances
    scales = torch.tensor([1.0, 0.5, 0.0, 1.5], device=art.device).repeat((n + 3) // 4)[:n]
    GA.set_gain_scale(art, None, scales)
    comp, appl, physx = _probe(art, 0.1, 1.0)
    kp = torch.zeros_like(comp)
    kd = torch.zeros_like(comp)
    lim = torch.zeros_like(comp)
    for act in art.actuators.values():
        kp[:, act.joint_indices] = act.stiffness
        kd[:, act.joint_indices] = act.damping
        lim[:, act.joint_indices] = act.effort_limit
    s = scales.unsqueeze(1)
    exp = s * kp * 0.1 + (s * kd + torch.clamp(1 - s, min=0) * 0.5) * (0.0 - 1.0)
    exp_c = torch.maximum(torch.minimum(exp, lim), -lim)
    for k, sv in enumerate([1.0, 0.5, 0.0, 1.5]):
        e = k
        err = (comp[e] - exp[e]).abs().max().item()
        perr = (physx[e] - exp_c[e]).abs().max().item()
        j = art.joint_names.index("left_hip_pitch_joint")
        print(f"  gain_scale={sv:3.1f}: computed max|err|={err:.2e}  physx(applied) max|err|={perr:.2e}  |"
              f" left_hip_pitch: computed {comp[e, j]:+.3f} expected {exp[e, j]:+.3f} physx {physx[e, j]:+.3f}")
    limp_env = 2
    print(f"  limp env torques (expect -0.5*qd = -0.5 on every joint): min {comp[limp_env].min():+.4f}"
          f" max {comp[limp_env].max():+.4f}")
    GA.set_gain_scale(art, None, 1.0)

    hdr("3b. set_effort_scale probe (Fix): target error 3 rad -> saturated torque = base_limit * scale")
    base = torch.zeros_like(comp)
    for act in art.actuators.values():
        base[:, act.joint_indices] = act.base_effort_limit
    GA.set_effort_scale(art, {".*_hip_.*": 1.2, ".*_knee_joint": 1.2, ".*_shoulder_.*": 1.2})
    odd = torch.arange(1, n, 2, device=art.device)
    GA.set_effort_scale(art, 0.9, env_ids=odd)
    comp, appl, physx = _probe(art, 3.0, 0.0)
    exp_scale = torch.ones_like(comp)
    for j, nm in enumerate(art.joint_names):
        if re.fullmatch(r".*_hip_.*|.*_knee_joint|.*_shoulder_.*", nm):
            exp_scale[:, j] = 1.2
    exp_scale[odd] = 0.9
    ratio = physx.abs() / base
    err = (ratio - exp_scale).abs()
    # joints whose unsaturated demand is below the limit are excluded (Kp*3 < limit never happens here)
    print(f"  even env (dict 1.2 hips/knees/shoulders else 1.0): max|ratio-exp|={err[0].max():.2e}")
    print(f"  odd env  (float 0.9, per-env override):          max|ratio-exp|={err[1].max():.2e}")
    for nm in ("left_hip_pitch_joint", "left_knee_joint", "left_elbow_joint", "left_ankle_roll_joint"):
        j = art.joint_names.index(nm)
        print(f"    {nm:24s} base {base[0, j]:5.1f}  env0 physx {physx[0, j]:+7.2f}  env1 physx {physx[1, j]:+7.2f}")
    try:
        GA.set_effort_scale(art, {"not_a_joint": 1.0})
        print("  typo key: NOT rejected (FAIL)")
    except ValueError as e:
        print(f"  typo key rejected: PASS ({e})")
    GA.set_effort_scale(art, 1.0)


def dc_probe(art) -> None:
    hdr("4. Stage B probe (DcFix, DelayedDCMotorLimpable, zero delay)")
    n = art.num_instances
    # (a) torque-speed curve on non-ankle joints: qd = f * no_load, big positive error
    fr = torch.tensor([0.0, 0.3, 0.6, 0.9], device=art.device).repeat((n + 3) // 4)[:n]
    q0 = torch.clamp(art.data.default_joint_pos.clone(), art.data.joint_pos_limits[..., 0] + 0.2,
                     art.data.joint_pos_limits[..., 1] - 0.2)
    v0 = torch.zeros_like(q0)
    sat = torch.zeros_like(q0)
    lim = torch.zeros_like(q0)
    curve = torch.zeros(q0.shape[1], dtype=torch.bool, device=art.device)
    for act in art.actuators.values():
        v0[:, act.joint_indices] = act.velocity_limit
        sat[:, act.joint_indices] = act.saturation_effort
        lim[:, act.joint_indices] = act.effort_limit
        curve[act.joint_indices] = act._joint_curve_mask
    qd = fr.unsqueeze(1) * v0
    art.write_joint_state_to_sim(q0, qd)
    art.set_joint_position_target(q0 + 10.0)
    art.set_joint_velocity_target(torch.zeros_like(q0))
    art.write_data_to_sim()
    physx = art.root_physx_view.get_dof_actuation_forces().clone()
    exp = torch.minimum(lim, sat * (1 - qd / v0))
    err = (physx - exp).abs()[:, curve]
    print(f"  torque-speed (non-ankle joints, qd = f*no_load, f in 0/.3/.6/.9): max|applied-expected| = {err.max():.2e}")
    for nm in ("left_hip_roll_joint", "left_hip_pitch_joint", "left_knee_joint"):
        j = art.joint_names.index(nm)
        row = ", ".join(f"f={fr[e]:.1f}:{physx[e, j]:.1f}" for e in range(min(4, n)))
        print(f"    {nm:22s} stall {sat[0, j]:.0f} Nm, no-load {v0[0, j]:.2f} rad/s, flat {lim[0, j]:.0f}: {row}")
    # (b) ankle differential: saturated pitch/roll demands of both signs, qd = 0
    art.write_joint_state_to_sim(q0, torch.zeros_like(q0))
    sp = torch.tensor([1.0, 1.0, -1.0, -1.0], device=art.device).repeat((n + 3) // 4)[:n]
    sr = torch.tensor([1.0, -1.0, 1.0, -1.0], device=art.device).repeat((n + 3) // 4)[:n]
    tgt = q0.clone()
    ids = {nm: art.joint_names.index(nm) for nm in art.joint_names if "ankle" in nm}
    for nm, j in ids.items():
        tgt[:, j] = q0[:, j] + (sp if "pitch" in nm else sr) * 1.0
    art.set_joint_position_target(tgt)
    art.write_data_to_sim()
    physx = art.root_physx_view.get_dof_actuation_forces().clone()
    kp_, kr_ = GA.ANKLE_K_PITCH, GA.ANKLE_K_ROLL
    for side in ("left", "right"):
        jp, jr = ids[f"{side}_ankle_pitch_joint"], ids[f"{side}_ankle_roll_joint"]
        for e in range(min(4, n)):
            tp, tr = physx[e, jp].item(), physx[e, jr].item()
            ta, tb = 0.5 * (tp / kp_ - tr / kr_), 0.5 * (-tp / kp_ - tr / kr_)
            # independent expectation: joint clip (40 / 17) then clamp motors to 12 and map back
            dp, dr = max(min(110.0 * sp[e].item(), 40.0), -40.0), max(min(110.0 * sr[e].item(), 17.0), -17.0)
            ea, eb = 0.5 * (dp / kp_ - dr / kr_), 0.5 * (-dp / kp_ - dr / kr_)
            ea, eb = max(min(ea, 12.0), -12.0), max(min(eb, 12.0), -12.0)
            ep, er = kp_ * (ea - eb), -kr_ * (ea + eb)
            print(f"    {side} env{e} demand(p={dp:+.0f}, r={dr:+.0f}) -> applied p={tp:+7.3f} r={tr:+7.3f}"
                  f" | motors A={ta:+7.3f} B={tb:+7.3f} | expected p={ep:+7.3f} r={er:+7.3f}")
    art.set_joint_position_target(q0)


# ---------------------------------------------------------------------------------------------------------------------
# 5/6. reach and velocity limits
# ---------------------------------------------------------------------------------------------------------------------


def reset_fixed(art):
    q = art.data.default_joint_pos.clone()
    art.write_joint_state_to_sim(q, torch.zeros_like(q))
    art.set_joint_position_target(q)
    art.reset()


def self_contact_report(sim, scene, art, sensor, label: str, n_steps: int = 100) -> None:
    """Hold the current targets for n_steps and report bodies with contact force (Fix is in the air: self-collision)."""
    peak = torch.zeros(len(sensor.body_names), device=art.device)
    qd_rms = []
    for _ in range(n_steps):
        step(sim, scene)
        peak = torch.maximum(peak, sensor.data.net_forces_w.norm(dim=-1).max(dim=0).values)
        qd_rms.append(art.data.joint_vel.pow(2).mean().sqrt())
    hits = [(sensor.body_names[b], peak[b].item()) for b in range(len(peak)) if peak[b] > 0.5]
    print(f"  [{label}] self-contact bodies (peak force > 0.5 N over {n_steps * DT:.1f} s): "
          f"{hits if hits else 'NONE'}; joint-vel RMS last step {qd_rms[-1]:.4f} rad/s")


def default_pose_self_collision(sim, scene, art, sensor) -> None:
    hdr("5a. Self-collision at the default standing pose (Fix, in the air, nominal gains)")
    reset_fixed(art)
    self_contact_report(sim, scene, art, sensor, "default pose", 200)


def reach_test(sim, scene, art, sensor=None) -> None:
    hdr("5. Joint reach (Fix): command 0.4 rad beyond the crouch limits for 2 s")
    reset_fixed(art)
    step(sim, scene, 20)
    tgt = art.data.default_joint_pos.clone()
    lim = art.data.joint_pos_limits
    # flexion (crouch) directions: left hip pitch -, right hip pitch +, left knee +, right knee -
    cases = {"left_hip_pitch_joint": 0, "right_hip_pitch_joint": 1, "left_knee_joint": 1, "right_knee_joint": 0}
    for nm, side in cases.items():
        j = art.joint_names.index(nm)
        tgt[:, j] = lim[:, j, side] + (0.4 if side == 1 else -0.4)
    art.set_joint_position_target(tgt)
    step(sim, scene, 400)
    for nm, side in cases.items():
        j = art.joint_names.index(nm)
        q = art.data.joint_pos[:, j]
        print(f"  {nm:22s} hard limit {lim[0, j, side]:+.4f}  soft {art.data.soft_joint_pos_limits[0, j, side]:+.4f}"
              f"  reached mean {q.mean():+.4f} (min {q.min():+.4f}, max {q.max():+.4f})"
              f"  torque {art.data.applied_torque[:, j].mean():+.1f} Nm")
    if sensor is not None:
        self_contact_report(sim, scene, art, sensor, "deep crouch (hip 2.09, knee 1.5)", 20)
    reset_fixed(art)


def vel_limit_test(sim, scene, art, name: str) -> None:
    hdr(f"6. Velocity-limit test ({name}): large position steps, peak |qd| over 1 s")
    reset_fixed(art)
    step(sim, scene, 40)
    tgt = art.data.default_joint_pos.clone()
    steps = {
        "right_shoulder_pitch_joint": +2.0, "left_shoulder_pitch_joint": -2.0,
        "left_hip_roll_joint": +0.7, "right_hip_roll_joint": -0.7,
        "left_hip_yaw_joint": +0.7, "right_hip_yaw_joint": -0.7,
        "left_elbow_joint": +1.5, "right_elbow_joint": -1.5,
    }
    ids = {nm: art.joint_names.index(nm) for nm in steps}
    for nm, d in steps.items():
        tgt[:, ids[nm]] += d
    art.set_joint_position_target(tgt)
    peak = torch.zeros(art.num_instances, len(steps), device=art.device)
    for _ in range(200):
        step(sim, scene)
        peak = torch.maximum(peak, art.data.joint_vel[:, list(ids.values())].abs())
    vl = art.data.joint_vel_limits[0]
    for k, nm in enumerate(steps):
        print(f"  {nm:28s} vel limit {vl[ids[nm]]:6.2f}  peak |qd| mean {peak[:, k].mean():6.3f} max {peak[:, k].max():6.3f}"
              f"  -> {'CAPPED' if peak[:, k].max() <= vl[ids[nm]] * 1.02 else 'NOT capped'}")
    reset_fixed(art)


# ---------------------------------------------------------------------------------------------------------------------
# 7. limp drop test
# ---------------------------------------------------------------------------------------------------------------------


def random_quat(n, device):
    q = torch.randn(n, 4, device=device)
    return q / q.norm(dim=1, keepdim=True)


def drop_test(sim, scene, robot, contacts) -> None:
    hdr(f"7. Limp drop test: root at 0.6 m (+lift if needed), random orientation/joints, {args_cli.drop_time} s limp")
    n = robot.num_instances
    dev = robot.device
    lo, hi = robot.data.soft_joint_pos_limits[..., 0], robot.data.soft_joint_pos_limits[..., 1]
    mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo)
    q = mid + 0.9 * half * (2 * torch.rand_like(mid) - 1)
    root = torch.zeros(n, 13, device=dev)
    root[:, :3] = scene.env_origins
    root[:, 2] += 0.6
    root[:, 3:7] = random_quat(n, dev)
    root[:, 7:10] = (2 * torch.rand(n, 3, device=dev) - 1) * 0.5
    root[:, 10:13] = (2 * torch.rand(n, 3, device=dev) - 1) * 1.0
    robot.write_root_pose_to_sim(root[:, :7])
    robot.write_joint_state_to_sim(q, torch.zeros_like(q))
    zmin = robot.data.body_link_pose_w[:, :, 2].min(dim=1).values
    lift = torch.clamp(0.10 - zmin, min=0.0)
    root[:, 2] += lift
    robot.write_root_pose_to_sim(root[:, :7])
    robot.write_root_velocity_to_sim(root[:, 7:])
    robot.set_joint_position_target(q)
    scene.reset()  # resets actuators (gain_scale -> 1) and sensors
    GA.set_gain_scale(robot, None, 0.0)  # limp: Kp 0, Kd 0.5
    print(f"  lift applied to keep lowest body origin >= 0.10 m: mean {lift.mean():.3f} max {lift.max():.3f} m;"
          f" start root z mean {root[:, 2].mean() - scene.env_origins[:, 2].mean():.3f}")
    vl = robot.data.joint_vel_limits
    nsteps = int(args_cli.drop_time / DT)
    max_body_v = torch.zeros(n, device=dev)
    at_vlim = torch.zeros(robot.num_joints, device=dev)
    max_force = torch.zeros(len(contacts.body_names), device=dev)
    ever_contact = torch.zeros(len(contacts.body_names), dtype=torch.bool, device=dev)
    nan = False
    settled_at = torch.full((n,), float("nan"), device=dev)
    for k in range(nsteps):
        step(sim, scene)
        bv = robot.data.body_lin_vel_w.norm(dim=-1).max(dim=1).values
        qd = robot.data.joint_vel
        if not torch.isfinite(bv).all() or not torch.isfinite(qd).all():
            nan = True
        max_body_v = torch.maximum(max_body_v, bv)
        at_vlim += (qd.abs() >= 0.98 * vl).float().sum(dim=0)
        f = contacts.data.net_forces_w.norm(dim=-1)  # (n, bodies)
        max_force = torch.maximum(max_force, f.max(dim=0).values)
        ever_contact |= (f > 1.0).any(dim=0)
        quiet = (bv < 0.05) & (qd.abs().max(dim=1).values < 0.2)
        settled_at = torch.where(torch.isnan(settled_at) & quiet, torch.full_like(settled_at, (k + 1) * DT), settled_at)
        settled_at = torch.where(~quiet, torch.full_like(settled_at, float("nan")), settled_at)
        if (k + 1) % int(1.0 / DT) == 0 or k + 1 == nsteps:
            print(f"  t={(k + 1) * DT:4.1f}s  settled (max body speed<0.05 m/s, |qd|<0.2) {quiet.float().mean():5.1%}"
                  f"  body speed median {bv.median():.4f} max {bv.max():.4f} m/s")
    # diagnose envs that are still moving: fastest body, fastest joint, contact bodies
    bvs = robot.data.body_lin_vel_w.norm(dim=-1)
    moving = torch.nonzero(bvs.max(dim=1).values >= 0.05).flatten().tolist()
    for e in moving[:8]:
        b = int(bvs[e].argmax())
        j = int(robot.data.joint_vel[e].abs().argmax())
        f = contacts.data.net_forces_w[e].norm(dim=-1)
        touching = [contacts.body_names[i] for i in range(len(f)) if f[i] > 1.0]
        print(f"  still moving env {e}: fastest body {robot.body_names[b]} {bvs[e, b]:.3f} m/s; fastest joint"
              f" {robot.joint_names[j]} {robot.data.joint_vel[e, j]:+.3f} rad/s; root ang vel"
              f" {robot.data.root_ang_vel_w[e].norm():.3f} rad/s; touching {touching}")
    h = robot.data.root_pos_w[:, 2] - scene.env_origins[:, 2]
    gz = robot.data.projected_gravity_b
    print(f"  NaN/inf: {nan}   max body speed during drop: median {max_body_v.median():.2f} max {max_body_v.max():.2f} m/s"
          f"  -> {'EXPLOSION' if max_body_v.max() > 15 else 'no explosion'} (threshold 15 m/s)")
    print(f"  final pelvis height: min {h.min():.3f} median {h.median():.3f} max {h.max():.3f} m;"
          f" final min body-origin z {robot.data.body_link_pose_w[:, :, 2].min() - scene.env_origins[:, 2].max():.3f} m")
    print(f"  final projected gravity (base frame) x: forward-down<0? mean {gz[:, 0].mean():+.2f}; z mean {gz[:, 2].mean():+.2f}"
          f" (supine/prone split by sign of gx: {(gz[:, 0] > 0.5).sum().item()} gx>0.5, {(gz[:, 0] < -0.5).sum().item()} gx<-0.5,"
          f" {((gz[:, 0].abs() <= 0.5)).sum().item()} other)")
    print("  steps with |qd| >= 0.98 vel limit (summed over envs), top joints:")
    for j in torch.argsort(at_vlim, descending=True)[:6].tolist():
        if at_vlim[j] > 0:
            print(f"    {robot.joint_names[j]:28s} {int(at_vlim[j].item()):6d} env-steps (limit {vl[0, j]:.2f} rad/s)")
    print("  bodies that touched something (>1 N) during the drop, and peak contact force:")
    for b, nm in enumerate(contacts.body_names):
        print(f"    {nm:28s} contact={'Y' if ever_contact[b] else '-'}  peak {max_force[b]:9.1f} N")
    # jitter at rest
    rms = []
    for _ in range(40):
        step(sim, scene)
        rms.append(robot.data.joint_vel.pow(2).mean(dim=1))
    rms = torch.stack(rms).mean(dim=0).sqrt()
    print(f"  rest jitter (RMS joint vel over 0.2 s after drop): median {rms.median():.4f} max {rms.max():.4f} rad/s")
    # responds to actuation after settling (sleep threshold 0)
    q_before = robot.data.joint_pos.clone()
    GA.set_gain_scale(robot, None, 1.0)
    robot.set_joint_position_target(robot.data.default_joint_pos.clone())
    step(sim, scene, 100)
    dq = (robot.data.joint_pos - q_before).abs().max(dim=1).values
    print(f"  un-limp to default pose for 0.5 s: envs with max|dq| > 0.05 rad: {(dq > 0.05).sum().item()}/{n}"
          f" (median max|dq| {dq.median():.3f} rad)")


def armature_note() -> None:
    hdr("Armature: ASIMOV_1_ACTUATORS vs rotor inertia x gear^2 (datasheet)")
    rotor = {"EC-A6416-P2-25": (104.395e-6, 25), "EC-A5013-H17-100": (10e-6, 100), "EC-A3814-H14-107": (3e-6, 107),
             "EC-A4315-P2-36": (25.5e-6, 36), "EC-A4310-P2-36": (18.2e-6, 36)}
    group_motor = {"hip_pitch": "EC-A6416-P2-25", "waist": "EC-A6416-P2-25", "hip_roll": "EC-A5013-H17-100",
                   "shoulder_pitch": "EC-A5013-H17-100", "hip_yaw": "EC-A3814-H14-107",
                   "shoulder_yaw": "EC-A3814-H14-107", "knee": "EC-A4315-P2-36", "shoulder_roll": "EC-A4315-P2-36",
                   "elbow_wrist": "EC-A4310-P2-36", "ankle_pitch": "EC-A4310-P2-36", "ankle_roll": "EC-A4310-P2-36"}
    for g, cfg in A.ASIMOV_1_ACTUATORS.items():
        j, r = rotor[group_motor[g]]
        refl = j * r * r
        extra = ""
        if g == "ankle_pitch":
            refl = 2 * GA.ANKLE_K_PITCH**2 * j * r * r
            extra = " (2*Kp^2*Jm through differential)"
        elif g == "ankle_roll":
            refl = 2 * GA.ANKLE_K_ROLL**2 * j * r * r
            extra = " (2*Kr^2*Jm through differential)"
        print(f"  {g:15s} cfg {cfg.armature:.4f}  datasheet {refl:.4f}  ratio {cfg.armature / refl:5.2f}{extra}")


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=DT, device=args_cli.device)
    sim_cfg.physx.gpu_max_rigid_patch_count = 10 * 2**15
    sim = sim_utils.SimulationContext(sim_cfg)
    scene = InteractiveScene(SceneCfg(num_envs=args_cli.num_envs, env_spacing=6.0))
    sim.reset()
    robot, fix, dcfix, walk = scene["robot"], scene["fix"], scene["dcfix"], scene["walk"]
    contacts = scene["contacts"]
    print(f"[getup-asset] envs={scene.num_envs} dt={DT} pos_iters={GETUP.spawn.articulation_props.solver_position_iteration_count}"
          f" getup urdf={A.ASIMOV_1_GETUP_URDF_PATH}")
    print(f"[getup-asset] actuator classes: robot={sorted({type(a).__name__ for a in robot.actuators.values()})}"
          f" dcfix={sorted({type(a).__name__ for a in dcfix.actuators.values()})}"
          f" walk={sorted({type(a).__name__ for a in walk.actuators.values()})}")
    print(f"[getup-asset] ASIMOV_1_RATED_TORQUE={A.ASIMOV_1_RATED_TORQUE}")
    if args_cli.only_drop:
        drop_test(sim, scene, robot, contacts)
        print("\n[getup-asset] validate_asset.py finished", flush=True)
        return
    armature_note()
    stage = sim_utils.get_current_stage() if hasattr(sim_utils, "get_current_stage") else sim.stage
    collision_report(stage, "/World/envs/env_0/Robot", A.ASIMOV_1_GETUP_URDF_PATH)
    try:
        print(f"  PhysX max_shapes: getup={robot.root_physx_view.max_shapes} walk={walk.root_physx_view.max_shapes}")
    except Exception as e:  # noqa: BLE001
        print(f"  (max_shapes unavailable: {e})")
    step(sim, scene, 2)
    limits_report("getup", robot)
    limits_report("walk", walk)
    actuator_probe(fix)
    dc_probe(dcfix)
    reset_fixed(dcfix)
    default_pose_self_collision(sim, scene, fix, scene["fix_contacts"])
    reach_test(sim, scene, fix, scene["fix_contacts"])
    vel_limit_test(sim, scene, fix, "Fix = getup asset, zero delay")
    vel_limit_test(sim, scene, walk, "Walk = ASIMOV_1_DELAYED_CFG")
    drop_test(sim, scene, robot, contacts)
    print("\n[getup-asset] validate_asset.py finished", flush=True)


if __name__ == "__main__":
    main()
    # simulation_app.close() hung for >10 min on the server (headless, Isaac Sim 5.1) while holding the shared GPU
    # lock; nothing is left to save, so exit hard.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
