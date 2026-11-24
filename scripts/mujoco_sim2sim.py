"""MuJoCo sim2sim of an exported cyclotron policy, set up like the Isaac Lab environment it was trained in.

Everything the policy sees and every actuator is rebuilt from the run's ``env.yaml``, so a policy that walks in Isaac
Lab and not here points at a physics (contact, solver, model) difference rather than a config mismatch:

* timing: ``sim.dt`` physics steps, the policy every ``decimation`` steps,
* observations: the ``observations.policy`` terms in order (base angular velocity and projected gravity in the root
  frame, with their per-episode observation lag; command; joint positions/velocities relative to the default pose,
  per slot; last raw action), with their scales,
* actions: ``raw * scale + default`` with the configured clip, applied in the action's joint order,
* actuators, per ``scene.robot.actuators`` group: ``DelayedPDActuator`` (PD on the delayed target, joint velocity
  target 0, torque clipped to ``actuator_effort_limit``; the delay in physics steps drawn per group at reset), and the
  ``ParallelAnkleActuator``: a PD on the joint-space errors whose gains follow the nonlinear linkage
  (``tau_j = J^T clip(K_m J e + D_m J de)``, i.e. joint gains ``J^T K_m J``), so the joint impedance changes with the
  pose every physics step, exactly as cyclotron/assets/robots/asimov_1_ankle_actuator.py computes it,
* joint armature and Coulomb friction (``frictionloss``) from the actuator config, which overrides the model's in
  Isaac Lab too.

Usage:
    python scripts/mujoco_sim2sim.py <run folder | exported/model_N folder | policy.onnx>
    python scripts/mujoco_sim2sim.py <run> --cmd 0.7 0 0             # one fixed command
    python scripts/mujoco_sim2sim.py <run> --headless --video out.mp4  # demo sequence, rendered offscreen

Without ``--cmd`` the robot runs a demo sequence (stand, forward, backward, left, right, turn left, turn right) and
prints, per phase, the measured velocity next to the command. In the viewer, the keys W/S, A/D, Q/E change the
command (vx, vy, wz), Space zeroes it, and R resets.
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import re
import time

import numpy as np
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_XML = os.path.join(REPO, "third_party/asimov-1/sim-model/xmls/asimov_1.xml")
KINEMATICS = os.path.join(REPO, "source/cyclotron/cyclotron/assets/robots/asimov_1_ankle_kinematics.py")
SIDES = ("left", "right")

DEMO = [  # (seconds, vx, vy, wz, name)
    (2.0, 0.0, 0.0, 0.0, "stand"),
    (5.0, 0.7, 0.0, 0.0, "forward"),
    (5.0, -0.5, 0.0, 0.0, "backward"),
    (5.0, 0.0, 0.4, 0.0, "left"),
    (5.0, 0.0, -0.4, 0.0, "right"),
    (5.0, 0.0, 0.0, 0.6, "turn left"),
    (5.0, 0.0, 0.0, -0.6, "turn right"),
    (2.0, 0.0, 0.0, 0.0, "stand"),
]


# ----------------------------------------------------------------------------------------------------------- config
class _Loader(yaml.SafeLoader):
    """env.yaml holds python tags (tuples, slices); read tuples as lists and anything else as plain data."""


_Loader.add_constructor("tag:yaml.org,2002:python/tuple", lambda loader, node: loader.construct_sequence(node))
_Loader.add_multi_constructor(
    "tag:yaml.org,2002:python/",
    lambda loader, suffix, node: (
        loader.construct_sequence(node)
        if isinstance(node, yaml.SequenceNode)
        else loader.construct_mapping(node)
        if isinstance(node, yaml.MappingNode)
        else loader.construct_scalar(node)
    ),
)


def resolve_policy(path: str) -> tuple[str, str]:
    """policy.onnx and env.yaml from a run folder (latest export), an export folder, or an .onnx file."""
    path = os.path.abspath(path)
    if path.endswith(".onnx"):
        folder = os.path.dirname(path)
        onnx_path = path
    else:
        folder = path
        if not os.path.exists(os.path.join(folder, "policy.onnx")):
            exports = glob.glob(os.path.join(folder, "exported", "model_*", "policy.onnx"))
            if exports:
                folder = os.path.dirname(max(exports, key=lambda p: int(re.findall(r"model_(\d+)", p)[-1])))
            elif os.path.exists(os.path.join(folder, "exported", "policy.onnx")):  # older layout: exported/policy.onnx
                folder = os.path.join(folder, "exported")
            else:
                raise SystemExit(f"No policy.onnx in {path} or its exported/ folder; run ./cyclotron.sh --export first.")
        onnx_path = os.path.join(folder, "policy.onnx")
    env_yaml = os.path.join(folder, "env.yaml")
    if not os.path.exists(env_yaml):  # older layout: the run's params/env.yaml
        env_yaml = os.path.join(os.path.dirname(folder), "params", "env.yaml")
    if not os.path.exists(env_yaml):
        raise SystemExit(f"No env.yaml next to {onnx_path}: the policy's gains and inputs can't be rebuilt without it.")
    return onnx_path, env_yaml


def per_joint(value, names: list[str], what: str) -> np.ndarray:
    """A float, or a {regex: value} dict as Isaac Lab resolves it, for each joint in ``names``."""
    if value is None:
        return np.full(len(names), np.nan)
    if not isinstance(value, dict):
        return np.full(len(names), float(value))
    out = np.full(len(names), np.nan)
    for i, name in enumerate(names):
        for expr, v in value.items():
            if re.fullmatch(expr, name):
                out[i] = float(v)
                break
        else:
            raise SystemExit(f"{what}: no entry matches joint {name}")
    return out


def match_joints(exprs: list[str], names: list[str]) -> list[str]:
    return [n for n in names if any(re.fullmatch(e, n) for e in exprs)]


# -------------------------------------------------------------------------------------------------------- actuators
class DelayBuffer:
    """Isaac Lab's DelayBuffer: returns the value pushed ``lag`` calls ago (0 = this call's)."""

    def __init__(self, max_lag: int, lag: int, size: int):
        self.lag = lag
        self.hist = np.zeros((max_lag + 1, size))
        self.n = 0

    def compute(self, x: np.ndarray) -> np.ndarray:
        self.hist = np.roll(self.hist, 1, axis=0)
        self.hist[0] = x
        if self.n == 0:  # Isaac Lab fills the history with the first value after a reset
            self.hist[:] = x
        self.n += 1
        return self.hist[min(self.lag, self.n - 1)].copy()


class DelayedPD:
    def __init__(self, cfg: dict, idx: np.ndarray, names: list[str], rng: np.random.Generator, fixed_delay):
        self.idx = idx
        self.kp = per_joint(cfg["stiffness"], names, "stiffness")
        self.kd = per_joint(cfg["damping"], names, "damping")
        limit = cfg.get("actuator_effort_limit")
        if limit is None:  # Isaac Lab 2.x name
            limit = cfg.get("effort_limit")
        self.limit = per_joint(limit if limit is not None else 1e9, names, "actuator_effort_limit")
        self.min_delay, self.max_delay = int(cfg.get("min_delay") or 0), int(cfg.get("max_delay") or 0)
        self.rng, self.fixed_delay = rng, fixed_delay
        self.reset()

    def reset(self):
        lag = self.fixed_delay if self.fixed_delay is not None else self.rng.integers(self.min_delay, self.max_delay + 1)
        self.lag = int(min(lag, self.max_delay))
        self.buf = DelayBuffer(self.max_delay, self.lag, len(self.idx))

    def torque(self, target: np.ndarray, q: np.ndarray, dq: np.ndarray) -> np.ndarray:
        t = self.buf.compute(target[self.idx])
        tau = self.kp * (t - q[self.idx]) + self.kd * (0.0 - dq[self.idx])
        return np.clip(tau, -self.limit, self.limit)


class ParallelAnkle(DelayedPD):
    """cyclotron's ParallelAnkleActuator: joint-space PD with pose-dependent gains J^T K_m J."""

    def __init__(self, cfg, idx, names, rng, fixed_delay, kin):
        super().__init__(cfg, idx, names, rng, fixed_delay)
        import torch

        self.torch, self.kin = torch, kin
        self.pi = np.array([names.index(f"{s}_ankle_pitch_joint") for s in SIDES])
        self.ri = np.array([names.index(f"{s}_ankle_roll_joint") for s in SIDES])
        self.right = torch.tensor([False, True])
        self.motor_limit = float(cfg["motor_effort_limit"])
        self.pitch_lim = float(cfg.get("pitch_target_limit", 0.35))
        self.roll_lim = float(cfg.get("roll_target_limit", 0.09))
        self.last_joint_kp = None  # (2 legs, 2, 2) J^T K_m J of the last step, for logging

    def torque(self, target, q, dq):
        torch, kin = self.torch, self.kin
        t = self.buf.compute(target[self.idx])
        qa, dqa = q[self.idx], dq[self.idx]
        f = lambda a: torch.tensor(a, dtype=torch.float64)  # noqa: E731
        pitch, roll = f(qa[self.pi]), f(qa[self.ri])
        dq2 = f(np.stack((dqa[self.pi], dqa[self.ri]), axis=-1))
        pitch_t = torch.clamp(f(t[self.pi]), -self.pitch_lim, self.pitch_lim)
        roll_t = torch.clamp(f(t[self.ri]), -self.roll_lim, self.roll_lim)
        _, jac = kin.motor_angles_and_jacobian(pitch, roll, self.right)
        err = torch.stack((pitch_t - pitch, roll_t - roll), dim=-1)
        derr = -dq2  # position-only action: joint velocity targets are 0
        err_m = (jac @ err.unsqueeze(-1)).squeeze(-1)
        derr_m = (jac @ derr.unsqueeze(-1)).squeeze(-1)
        kp = f(np.stack((self.kp[self.pi], self.kp[self.ri]), axis=-1))
        kd = f(np.stack((self.kd[self.pi], self.kd[self.ri]), axis=-1))
        tau_m = torch.clamp(kp * err_m + kd * derr_m, -self.motor_limit, self.motor_limit)
        tau_j = (jac.transpose(-1, -2) @ tau_m.unsqueeze(-1)).squeeze(-1).numpy()
        self.last_joint_kp = (jac.transpose(-1, -2) @ torch.diag_embed(kp) @ jac).numpy()
        out = np.zeros(len(self.idx))
        out[self.pi], out[self.ri] = tau_j[:, 0], tau_j[:, 1]
        return out


# The Asimov-1 firmware's ankle control: each motor's own PD (27.5 / 1.25, 12 Nm) behind the linear target map, i.e.
# a joint-space PD with the pose-dependent gains J^T K_m J.
FIRMWARE_ANKLE = {
    "class_type": "cyclotron.assets.robots.asimov_1_ankle_actuator:ParallelAnkleActuator",
    "joint_names_expr": [".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
    "stiffness": 27.5, "damping": 1.25, "motor_effort_limit": 12.0,
    "pitch_target_limit": 0.35, "roll_target_limit": 0.09,
}


def override_ankles(groups: dict, mode: str) -> dict:
    """The run's actuator groups, with the ankles replaced by the firmware's ankle control when ``mode`` is 'firmware'."""
    if mode == "trained":
        return groups
    is_ankle = lambda c: any("ankle" in e for e in c["joint_names_expr"])  # noqa: E731
    ankle = [c for c in groups.values() if is_ankle(c)]
    keep = {k: c for k, c in groups.items() if not is_ankle(c)}
    shared = {k: ankle[0].get(k) for k in ("min_delay", "max_delay", "friction")}
    # armature: the run's per-joint value, which the firmware doesn't change
    joints = [f"{s}_ankle_{j}_joint" for s in SIDES for j in ("pitch", "roll")]
    arm = {}
    for c in ankle:
        names = match_joints(c["joint_names_expr"], joints)
        vals = per_joint(c.get("armature"), names, "armature")
        arm.update({re.escape(n): float(v) for n, v in zip(names, vals)})
    keep["ankle"] = {**FIRMWARE_ANKLE, **shared, "armature": arm}
    print("[INFO] ankles: the firmware's ankle control instead of the run's actuator config")
    return keep


# ---------------------------------------------------------------------------------------------------------- sim2sim
def quat_rotate_inverse(q_wxyz: np.ndarray, v: np.ndarray) -> np.ndarray:
    w, xyz = q_wxyz[0], q_wxyz[1:]
    return v * (2.0 * w * w - 1.0) - np.cross(xyz, v) * w * 2.0 + xyz * np.dot(xyz, v) * 2.0


class Sim2Sim:
    def __init__(self, args):
        import mujoco
        import onnxruntime as ort

        self.mj = mujoco
        self.args = args
        onnx_path, env_yaml = resolve_policy(args.policy)
        print(f"[INFO] policy:   {onnx_path}\n[INFO] env.yaml: {env_yaml}\n[INFO] model:    {args.xml}")
        with open(env_yaml) as fh:
            env = yaml.load(fh, Loader=_Loader)
        self.rng = np.random.default_rng(args.seed)

        # --- policy and its metadata
        self.sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.in_name = self.sess.get_inputs()[0].name
        meta = self.sess.get_modelmeta().custom_metadata_map

        # --- model
        self.m = mujoco.MjModel.from_xml_path(args.xml)
        self.d = mujoco.MjData(self.m)
        self.sim_dt = float(env["sim"]["dt"])
        self.decimation = int(env["decimation"])
        self.m.opt.timestep = self.sim_dt
        g = env["sim"].get("gravity")
        if g:
            self.m.opt.gravity[:] = g

        # --- action: joint order, scale, clip, default pose
        act = env["actions"]["joint_pos"]
        if not (act.get("use_default_offset") and act.get("preserve_order")):
            raise SystemExit("Only JointPositionAction with use_default_offset and preserve_order is supported.")
        self.joints = list(act["joint_names"])
        if "joint_names" in meta and json.loads(meta["joint_names"]) != self.joints:
            raise SystemExit("policy.onnx joint_names differ from env.yaml actions.joint_pos.joint_names")
        n = len(self.joints)
        self.qadr = np.array([self.m.jnt_qposadr[self.m.joint(j).id] for j in self.joints])
        self.vadr = np.array([self.m.jnt_dofadr[self.m.joint(j).id] for j in self.joints])
        robot = env["scene"]["robot"]
        self.default_q = np.zeros(n)
        for i, j in enumerate(self.joints):
            for expr, v in robot["init_state"]["joint_pos"].items():
                if re.fullmatch(expr, j):
                    self.default_q[i] = float(v)
                    break
        self.scale = per_joint(act["scale"], self.joints, "action scale")
        self.clip_lo, self.clip_hi = np.full(n, -np.inf), np.full(n, np.inf)
        for expr, (lo, hi) in (act.get("clip") or {}).items():
            for i, j in enumerate(self.joints):
                if re.fullmatch(expr, j):
                    self.clip_lo[i], self.clip_hi[i] = lo, hi
        self.init_pos = np.array(robot["init_state"]["pos"], dtype=float)
        rot = [float(x) for x in robot["init_state"]["rot"]]
        # Isaac Lab 3 (sim.physics set) stores (x, y, z, w); Isaac Lab 2.x stored (w, x, y, z).
        isaaclab3 = (env["sim"].get("physics") or {}).get("class_type") is not None
        self.init_quat = np.array([rot[3], *rot[:3]] if isaaclab3 else rot, dtype=float)
        print(f"[INFO] env.yaml layout: Isaac Lab {'3' if isaaclab3 else '2.x'}; initial base quat (w, x, y, z) {self.init_quat.tolist()}")

        # --- actuators
        kin = None
        self.actuators, covered = [], set()
        groups = override_ankles(robot["actuators"], args.ankle)
        for group, cfg in groups.items():
            names = match_joints(cfg["joint_names_expr"], self.joints)
            idx = np.array([self.joints.index(x) for x in names])
            cls = cfg["class_type"].split(":")[-1]
            if cls == "ParallelAnkleActuator":
                if kin is None:
                    spec = importlib.util.spec_from_file_location("asimov_1_ankle_kinematics", KINEMATICS)
                    kin = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(kin)
                a = ParallelAnkle(cfg, idx, names, self.rng, args.act_delay, kin)
            elif cls in ("DelayedPDActuator", "IdealPDActuator"):
                a = DelayedPD(cfg, idx, names, self.rng, args.act_delay)
            else:
                raise SystemExit(f"Actuator group {group}: {cls} is not supported by this script.")
            a.name, a.cls = group, cls
            self.actuators.append(a)
            covered.update(names)
            # Isaac Lab writes the group's armature and Coulomb friction to the joints, over the model's values.
            arm = per_joint(cfg.get("armature"), names, "armature")
            fric = per_joint(cfg.get("friction"), names, "friction")
            for k, j in enumerate(names):
                dof = self.m.jnt_dofadr[self.m.joint(j).id]
                if not np.isnan(arm[k]):
                    self.m.dof_armature[dof] = arm[k]
                if not np.isnan(fric[k]) and not args.no_joint_friction:
                    self.m.dof_frictionloss[dof] = fric[k]
                self.m.dof_damping[dof] = 0.0
        if covered != set(self.joints):
            raise SystemExit(f"Joints without an actuator: {sorted(set(self.joints) - covered)}")

        # --- observation terms
        self.obs_terms = []
        for name, term in env["observations"]["policy"].items():
            if not isinstance(term, dict) or "func" not in term:
                continue
            if term.get("clip") is not None or term.get("history_length"):
                raise SystemExit(f"Observation term {name}: clip/history are not supported by this script.")
            func = term["func"].split(":")[-1]
            params = term.get("params") or {}
            scale = 1.0 if term.get("scale") is None else term["scale"]
            spec = {"name": name, "func": func, "scale": scale}
            if func == "delayed_obs":
                spec["quantity"] = params["quantity"]
                spec["min_lag"], spec["max_lag"] = int(params["min_lag"]), int(params["max_lag"])
            elif func in ("joint_pos_rel", "joint_vel_rel"):
                jn = params["asset_cfg"]["joint_names"]
                spec["idx"] = np.array([self.joints.index(j) for j in jn])
            elif func not in ("generated_commands", "last_action", "base_ang_vel", "projected_gravity"):
                raise SystemExit(f"Observation term {name}: {func} is not supported by this script.")
            self.obs_terms.append(spec)
        self.root_id = self.m.joint(0).id  # the free joint
        self.root_body = self.m.jnt_bodyid[self.root_id]
        self.cmd = np.zeros(3)
        self.reset()
        obs_dim = self.observe().shape[0]
        expected = self.sess.get_inputs()[0].shape[-1]
        if isinstance(expected, int) and expected != obs_dim:
            raise SystemExit(f"Observation size {obs_dim} != policy input {expected}")
        self.describe()

    # ----------------------------------------------------------------------------------------------------- helpers
    def describe(self):
        print(f"[INFO] physics dt {self.sim_dt} s, decimation {self.decimation} -> policy {1 / (self.sim_dt * self.decimation):.0f} Hz")
        for a in self.actuators:
            joint_names = [self.joints[i] for i in a.idx]
            print(
                f"[INFO] actuator {a.name:15s} {a.cls:22s} delay {a.lag} phys steps"
                f" | kp {np.unique(a.kp)} kd {np.unique(a.kd)} | {', '.join(joint_names)}"
            )
            if isinstance(a, ParallelAnkle):
                print(
                    f"       joint PD with gains J^T K_m J (motor kp {a.kp[0]}, kd {a.kd[0]}, {a.motor_limit} Nm),"
                    f" target clamp pitch ±{a.pitch_lim} roll ±{a.roll_lim}"
                )
        lags = {t["name"]: t["lag"] for t in self.obs_terms if "lag" in t}
        print(f"[INFO] observation lags (policy steps): {lags}")

    def reset(self):
        self.mj.mj_resetData(self.m, self.d)
        self.d.qpos[0:3] = self.init_pos
        self.d.qpos[3:7] = self.init_quat
        self.d.qpos[self.qadr] = self.default_q
        self.d.qvel[:] = 0.0
        self.mj.mj_forward(self.m, self.d)
        self.last_action = np.zeros(len(self.joints))
        self.target = self.default_q.copy()
        for a in self.actuators:
            a.reset()
        for t in self.obs_terms:
            if t["func"] == "delayed_obs":
                lag = self.args.obs_lag if self.args.obs_lag is not None else self.rng.integers(t["min_lag"], t["max_lag"] + 1)
                t["lag"] = int(min(lag, t["max_lag"]))
                t["buf"] = None

    def quantity(self, name: str) -> np.ndarray:
        quat = self.d.qpos[3:7]
        if name == "base_ang_vel":
            return self.d.qvel[3:6].copy()  # free joint angular velocity is in the body (root) frame
        if name == "projected_gravity":
            return quat_rotate_inverse(quat, np.array([0.0, 0.0, -1.0]))
        raise KeyError(name)

    def observe(self) -> np.ndarray:
        q = self.d.qpos[self.qadr]
        dq = self.d.qvel[self.vadr]
        parts = []
        for t in self.obs_terms:
            f = t["func"]
            if f == "delayed_obs":
                v = self.quantity(t["quantity"])
                if t["buf"] is None:
                    t["buf"] = np.repeat(v[None], t["max_lag"] + 1, axis=0)
                else:
                    t["buf"] = np.roll(t["buf"], 1, axis=0)
                    t["buf"][0] = v
                v = t["buf"][t["lag"]]
            elif f in ("base_ang_vel", "projected_gravity"):
                v = self.quantity(f)
            elif f == "generated_commands":
                v = self.cmd
            elif f == "joint_pos_rel":
                v = (q - self.default_q)[t["idx"]]
            elif f == "joint_vel_rel":
                v = dq[t["idx"]]
            elif f == "last_action":
                v = self.last_action
            parts.append(np.asarray(v, dtype=np.float64) * t["scale"])
        return np.concatenate(parts).astype(np.float32)

    def policy_step(self):
        obs = self.observe()
        action = self.sess.run(None, {self.in_name: obs[None]})[0][0].astype(np.float64)
        self.last_action = action
        self.target = np.clip(action * self.scale + self.default_q, self.clip_lo, self.clip_hi)
        for _ in range(self.decimation):
            q = self.d.qpos[self.qadr]
            dq = self.d.qvel[self.vadr]
            tau = np.zeros(len(self.joints))
            for a in self.actuators:
                tau[a.idx] = a.torque(self.target, q, dq)
            self.d.qfrc_applied[:] = 0.0
            self.d.qfrc_applied[self.vadr] = tau
            self.mj.mj_step(self.m, self.d)

    def base_vel_yaw(self) -> np.ndarray:
        """(vx, vy, wz) of the root in its heading frame."""
        quat = self.d.qpos[3:7]
        w, x, y, z = quat
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        v = self.d.qvel[0:3]  # world frame
        c, s = np.cos(yaw), np.sin(yaw)
        return np.array([c * v[0] + s * v[1], -s * v[0] + c * v[1], self.world_ang_vel()[2]])

    def world_ang_vel(self) -> np.ndarray:
        rot = np.zeros(9)
        self.mj.mju_quat2Mat(rot, self.d.qpos[3:7])
        return rot.reshape(3, 3) @ self.d.qvel[3:6]

    def fallen(self) -> bool:
        return self.d.qpos[2] < 0.3 or self.quantity("projected_gravity")[2] > -0.5


# ------------------------------------------------------------------------------------------------------------- main
def schedule(args):
    if args.cmd is not None:
        return [(args.duration, *args.cmd, "cmd")]
    return DEMO


def run(args):
    import mujoco

    sim = Sim2Sim(args)
    policy_dt = sim.sim_dt * sim.decimation
    phases = schedule(args)
    total = sum(p[0] for p in phases)

    renderer = writer = None
    if args.video:
        import imageio

        renderer = mujoco.Renderer(sim.m, args.height, args.width)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        cam.trackbodyid = sim.root_body
        cam.distance, cam.elevation, cam.azimuth = 2.6, -15.0, 135.0
        fps = round(1.0 / policy_dt)
        writer = imageio.get_writer(args.video, fps=fps, codec="libx264", quality=8)

    viewer = None
    if not args.headless:
        import mujoco.viewer

        def key(code):
            k = chr(code) if 0 < code < 256 else ""
            step = {"W": (0, 0.1), "S": (0, -0.1), "A": (1, 0.1), "D": (1, -0.1), "Q": (2, 0.1), "E": (2, -0.1)}
            if k in step:
                sim.manual = True
                i, dv = step[k]
                sim.cmd[i] = round(sim.cmd[i] + dv, 2)
                print(f"[KEY] command (vx, vy, wz) = {sim.cmd.tolist()}")
            elif k == " ":
                sim.manual = True
                sim.cmd[:] = 0.0
            elif k == "R":
                sim.reset()

        sim.manual = False
        viewer = mujoco.viewer.launch_passive(sim.m, sim.d, key_callback=key)
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = sim.root_body
        viewer.cam.distance, viewer.cam.elevation = 2.6, -15.0

    print(f"\n{'phase':12s} {'command (vx vy wz)':>22s} {'measured (vx vy wz)':>24s}   ankle pitch kp L/R")
    t, step, fell = 0.0, 0, False
    try:
        for dur, vx, vy, wz, name in phases:
            if not getattr(sim, "manual", False):
                sim.cmd[:] = (vx, vy, wz)
            steps = int(round(dur / policy_dt))
            meas, kps = [], []
            for k in range(steps):
                wall = time.time()
                sim.policy_step()
                step += 1
                t += policy_dt
                if k >= steps // 3:  # skip the transition into the phase
                    meas.append(sim.base_vel_yaw())
                    ank = next((a for a in sim.actuators if isinstance(a, ParallelAnkle)), None)
                    if ank is not None and ank.last_joint_kp is not None:
                        kps.append(ank.last_joint_kp[:, 0, 0])
                if sim.fallen() and not fell:
                    fell = True
                    print(f"[WARN] fell at t = {t:.2f} s ({name})")
                if writer is not None:
                    renderer.update_scene(sim.d, camera=cam)
                    writer.append_data(renderer.render())
                if viewer is not None:
                    if not viewer.is_running():
                        raise KeyboardInterrupt
                    viewer.sync()
                    sleep = policy_dt - (time.time() - wall)
                    if sleep > 0:
                        time.sleep(sleep)
            mv = np.mean(meas, axis=0) if meas else np.zeros(3)
            kp = np.mean(kps, axis=0) if kps else None
            kp_s = f"{kp[0]:6.1f} / {kp[1]:6.1f}" if kp is not None else "-"
            print(
                f"{name:12s} {sim.cmd[0]:+6.2f} {sim.cmd[1]:+6.2f} {sim.cmd[2]:+6.2f}"
                f"     {mv[0]:+6.2f} {mv[1]:+6.2f} {mv[2]:+6.2f}      {kp_s}"
            )
        if viewer is not None and args.cmd is None:
            print("[INFO] demo done; the viewer stays open with the last command (W/S A/D Q/E, Space, R). Close it to exit.")
            while viewer.is_running():
                wall = time.time()
                sim.policy_step()
                viewer.sync()
                time.sleep(max(0.0, policy_dt - (time.time() - wall)))
    except KeyboardInterrupt:
        pass
    finally:
        if writer is not None:
            writer.close()
            print(f"[INFO] video: {os.path.abspath(args.video)} ({total:.0f} s)")
        if viewer is not None:
            viewer.close()
    print(f"[INFO] {'FELL' if fell else 'stayed up'}; final base height {sim.d.qpos[2]:.3f} m")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("policy", help="run folder, exported/model_N folder, or policy.onnx (with env.yaml next to it)")
    p.add_argument("--xml", default=DEFAULT_XML, help="MuJoCo model (default: the asimov-1 submodule's MJCF)")
    p.add_argument("--cmd", type=float, nargs=3, metavar=("VX", "VY", "WZ"), help="fixed command instead of the demo")
    p.add_argument("--duration", type=float, default=10.0, help="seconds to run a --cmd (default 10)")
    p.add_argument("--act_delay", type=int, default=None, help="actuator delay in physics steps (default: random per group within the trained min/max, like Isaac Lab)")
    p.add_argument("--obs_lag", type=int, default=None, help="observation lag in policy steps (default: random within the trained range)")
    p.add_argument(
        "--ankle",
        choices=("trained", "firmware"),
        default="trained",
        help="ankle actuation: as the run was trained (default), or 'firmware': the robot's motor-side PD 27.5/1.25"
        " (joint-space PD with gains J^T K_m J), e.g. to test a constant-gain run against the real ankle behaviour",
    )
    p.add_argument("--no_joint_friction", action="store_true", help="don't apply the actuator config's Coulomb joint friction")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--headless", action="store_true", help="no viewer window")
    p.add_argument("--video", type=str, default=None, help="write an mp4 rendered offscreen (set MUJOCO_GL=egl when headless)")
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--height", type=int, default=540)
    args = p.parse_args()
    if args.headless and args.video and "MUJOCO_GL" not in os.environ:
        os.environ["MUJOCO_GL"] = "egl"
    run(args)


if __name__ == "__main__":
    main()
