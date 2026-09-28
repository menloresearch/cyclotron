# Copyright (c) 2026, Menlo Research.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""CPU tests for tasks/getup/symmetry.py (no Isaac Lab needed).

The module is loaded by file path because importing ``isaac_asimov.tasks`` pulls in ``isaaclab_tasks``.
The joint mirror table is re-derived independently from the URDF.
"""

from __future__ import annotations

import ast
import importlib.util
import math
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from tensordict import TensorDict

REPO = Path(__file__).resolve().parents[2]
PKG = REPO / "source" / "isaac_asimov" / "isaac_asimov"


def _load_symmetry():
    name = "getup_symmetry_under_test"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, PKG / "tasks" / "getup" / "symmetry.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


sym = _load_symmetry()
J = list(sym.ASIMOV_1_JOINT_NAMES)
M = np.diag([1.0, -1.0, 1.0])


# ---------------------------------------------------------------------------------------------------------------------
# Source-of-truth extraction (AST, no imports of Isaac Lab modules)
# ---------------------------------------------------------------------------------------------------------------------


def _module_ast(path: Path) -> ast.Module:
    return ast.parse(path.read_text())


def _assigned_literal(path: Path, name: str):
    for node in _module_ast(path).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise KeyError(name)


def _standing_joint_pos() -> dict[str, float]:
    """Resolve ASIMOV_1_STANDING_INIT_STATE.joint_pos (regex keys) to all 23 joints."""
    import re

    for node in _module_ast(PKG / "assets" / "robots" / "asimov_1.py").body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "ASIMOV_1_STANDING_INIT_STATE" for t in node.targets
        ):
            for kw in node.value.keywords:
                if kw.arg == "joint_pos":
                    patterns = ast.literal_eval(kw.value)
    out = {}
    for j in J:
        matches = [v for k, v in patterns.items() if re.fullmatch(k, j)]
        assert len(matches) == 1, (j, matches)
        out[j] = float(matches[0])
    return out


def _urdf_path() -> Path | None:
    candidates = [
        Path(os.environ["ASIMOV_1_MODEL_DIR"]).expanduser() / "urdf" / "asimov_1.urdf"
        if "ASIMOV_1_MODEL_DIR" in os.environ
        else None,
        REPO / "third_party" / "asimov-1" / "sim-model" / "urdf" / "asimov_1.urdf",
        Path.home() / "isaac_asimov" / "third_party" / "asimov-1" / "sim-model" / "urdf" / "asimov_1.urdf",
    ]
    for c in candidates:
        if c is not None and c.is_file():
            return c
    return None


def _rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def _urdf_joints(path: Path) -> dict[str, dict]:
    """World-frame (pelvis) joint axes/positions at the zero configuration + limits."""
    root = ET.parse(path).getroot()
    joints = {}
    child_of = {}
    for j in root.findall("joint"):
        o = j.find("origin")
        xyz = np.array([float(v) for v in (o.get("xyz") if o is not None else "0 0 0").split()])
        rpy = [float(v) for v in (o.get("rpy") if o is not None else "0 0 0").split()] if o is not None else [0, 0, 0]
        ax = j.find("axis")
        axis = np.array([float(v) for v in ax.get("xyz").split()]) if ax is not None else None
        lim = j.find("limit")
        joints[j.get("name")] = dict(
            type=j.get("type"),
            parent=j.find("parent").get("link"),
            child=j.find("child").get("link"),
            xyz=xyz,
            R=_rpy(*rpy),
            axis=axis,
            limit=(float(lim.get("lower")), float(lim.get("upper"))) if lim is not None else None,
        )
        child_of[j.find("child").get("link")] = j.get("name")

    def link_pose(link):
        if link not in child_of:
            return np.eye(3), np.zeros(3)
        jn = joints[child_of[link]]
        R_p, p_p = link_pose(jn["parent"])
        return R_p @ jn["R"], p_p + R_p @ jn["xyz"]

    for jn in joints.values():
        R, p = link_pose(jn["child"])  # joint frame == child frame at q = 0
        jn["axis_w"] = R @ jn["axis"] if jn["axis"] is not None else None
        jn["pos_w"] = p
    return joints


# ---------------------------------------------------------------------------------------------------------------------
# Constants vs repo sources
# ---------------------------------------------------------------------------------------------------------------------


def test_constants_match_repo_sources():
    assert J == _assigned_literal(PKG / "assets" / "robots" / "asimov_1.py", "ASIMOV_1_JOINT_NAMES")
    vel_cfg = PKG / "tasks" / "locomotion" / "velocity_env_cfg.py"
    assert sym.SLOT_0_1 == _assigned_literal(vel_cfg, "SLOT_0_1")
    assert sym.SLOT_2_3 == _assigned_literal(vel_cfg, "SLOT_2_3")
    assert sym.SLOT_4_5 == _assigned_literal(vel_cfg, "SLOT_4_5")
    assert sorted(sym.SLOT_0_1 + sym.SLOT_2_3 + sym.SLOT_4_5) == sorted(J)


def test_known_permutations():
    """Tables from research/isaaclab_implementation.md §6.2."""
    perm, sign = sym.action_map()
    assert perm == [6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 12, 18, 19, 20, 21, 22, 13, 14, 15, 16, 17]
    assert sign == [-1.0] * 23
    assert sym.lr_swap_perm(sym.SLOT_0_1) == [2, 3, 0, 1, 4, 7, 8, 5, 6]
    assert sym.lr_swap_perm(sym.SLOT_2_3) == [2, 3, 0, 1, 6, 7, 4, 5]
    assert sym.lr_swap_perm(sym.SLOT_4_5) == [2, 3, 0, 1, 5, 4]


# ---------------------------------------------------------------------------------------------------------------------
# URDF-derived joint mirror (independent of the hand-written rule)
# ---------------------------------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def urdf():
    path = _urdf_path()
    if path is None:
        pytest.skip("asimov_1.urdf not found (set ASIMOV_1_MODEL_DIR)")
    return _urdf_joints(path)


def test_urdf_mirror_signs_and_geometry(urdf):
    """For each joint j with partner p: M a_j = s * a_p  =>  q_p' = -s * q_j. Every Asimov 1 joint must give -1."""
    perm, sign = sym.action_map()
    for i, j in enumerate(J):
        p = J[perm[i]]
        assert p == sym.mirror_name(j)
        a_j, a_p = urdf[j]["axis_w"], urdf[p]["axis_w"]
        ma = M @ a_j
        if np.allclose(ma, a_p, atol=1e-6):
            derived = -1.0
        elif np.allclose(ma, -a_p, atol=1e-6):
            derived = +1.0
        else:
            raise AssertionError(f"{j}: reflected axis {ma} is not +-{a_p}")
        assert derived == sign[i], f"{j}: URDF says mirror sign {derived}, table says {sign[i]}"
        # joint origins mirror (<= 1 mm; the hip-yaw/knee origins differ by ~0.6 mm in the URDF)
        assert np.allclose(M @ urdf[j]["pos_w"], urdf[p]["pos_w"], atol=1e-3), j
        # limits mirror: [lo_p, hi_p] = sign * [hi_j, lo_j]
        lo, hi = urdf[j]["limit"]
        expected = sorted([derived * lo, derived * hi])
        assert np.allclose(expected, urdf[p]["limit"], atol=1e-5), (j, urdf[j]["limit"], p, urdf[p]["limit"])


def test_urdf_non_mirror_joints_are_fixed(urdf):
    movable = {n for n, j in urdf.items() if j["type"] != "fixed"}
    assert movable == set(J)


# ---------------------------------------------------------------------------------------------------------------------
# Layouts
# ---------------------------------------------------------------------------------------------------------------------

H = 5


def _policy_ctxs(history=H):
    """Get-up policy group: walking layout minus `command`."""
    terms = [
        ("base_ang_vel", 3, None),
        ("projected_gravity", 3, None),
        ("joint_pos_slot01", 9, sym.SLOT_0_1),
        ("joint_pos_slot23", 8, sym.SLOT_2_3),
        ("joint_pos_slot45", 6, sym.SLOT_4_5),
        ("joint_vel_slot01", 9, sym.SLOT_0_1),
        ("joint_vel_slot23", 8, sym.SLOT_2_3),
        ("joint_vel_slot45", 6, sym.SLOT_4_5),
        ("actions", 23, None),
    ]
    return [sym.TermContext(n, d * history, history, joint_names=jn, action_joint_names=tuple(J)) for n, d, jn in terms]


CONTACT_BODIES = (
    "left_ankle_roll_link", "right_ankle_roll_link", "left_knee_link", "right_knee_link",
    "left_elbow_link", "right_elbow_link", "left_wrist_yaw_link", "right_wrist_yaw_link",
    "waist_yaw_link", "pelvis_link",
)  # fmt: skip


def _critic_ctxs(history=H):
    extra = [
        sym.TermContext("pelvis_height", 1 * history, history),
        sym.TermContext("base_lin_vel", 3 * history, history),
        sym.TermContext("root_quat", 4 * history, history),
        sym.TermContext("body_contact_norms", len(CONTACT_BODIES) * history, history, body_names=CONTACT_BODIES),
        sym.TermContext("effort_saturation", 23 * history, history),
        sym.TermContext("thermal_proxy", 23 * history, history),
        sym.TermContext("assist_force", 3 * history, history),
    ]
    return _policy_ctxs(history) + extra


def _tensors(fm):
    perm, sign = fm
    return torch.tensor(perm), torch.tensor(sign)


def test_policy_layout_dims():
    ctxs = _policy_ctxs()
    assert sum(c.dim for c in ctxs) == 75 * H
    assert sum(c.dim for c in _policy_ctxs(1)) == 75


@pytest.mark.parametrize("ctx_fn", [_policy_ctxs, _critic_ctxs])
@pytest.mark.parametrize("history", [1, H])
def test_mirror_twice_is_identity_obs(ctx_fn, history):
    ctxs = ctx_fn(history)
    perm, sign = _tensors(sym.build_group_map(ctxs))
    x = torch.randn(64, sum(c.dim for c in ctxs))
    y = sym.apply_map(x, perm, sign)
    assert not torch.allclose(x, y)
    assert torch.equal(sym.apply_map(y, perm, sign), x)


def test_mirror_twice_is_identity_actions():
    perm, sign = _tensors(sym.action_map())
    a = torch.randn(128, 23)
    assert torch.equal(sym.apply_map(sym.apply_map(a, perm, sign), perm, sign), a)


def test_history_frames_preserved():
    """Each history frame is mirrored in place (oldest first), never shuffled across time."""
    ctxs = _critic_ctxs(H)
    perm, sign = _tensors(sym.build_group_map(ctxs))
    frame_perm, frame_sign = _tensors(sym.build_group_map(_critic_ctxs(1)))
    x = torch.randn(16, sum(c.dim for c in ctxs))
    y = sym.apply_map(x, perm, sign)
    off = 0
    frame_off = 0
    for c1 in _critic_ctxs(1):
        d = c1.dim
        for t in range(H):
            # frame t of this term, mirrored with the single-frame map, must equal frame t of the mirrored output
            frame_x = torch.zeros(16, frame_perm.numel())
            frame_x[:, frame_off : frame_off + d] = x[:, off + t * d : off + (t + 1) * d]
            frame_y = sym.apply_map(frame_x, frame_perm, frame_sign)[:, frame_off : frame_off + d]
            assert torch.equal(frame_y, y[:, off + t * d : off + (t + 1) * d]), (c1.name, t)
        off += d * H
        frame_off += d


def test_history_divisibility_error():
    with pytest.raises(ValueError):
        sym.build_term_map(sym.TermContext("base_ang_vel", 14, 5))
    with pytest.raises(ValueError):
        sym.build_term_map(sym.TermContext("base_ang_vel", 4 * 5, 5))  # 4 != 3


# ---------------------------------------------------------------------------------------------------------------------
# Physical poses
# ---------------------------------------------------------------------------------------------------------------------


def _q_vec(d: dict[str, float]) -> torch.Tensor:
    return torch.tensor([[d[j] for j in J]])


def test_standing_pose_is_mirror_invariant():
    q0 = _q_vec(_standing_joint_pos())
    perm, sign = _tensors(sym.action_map())
    assert torch.allclose(sym.apply_map(q0, perm, sign), q0)


def test_standing_obs_is_mirror_invariant():
    """Standing still, default pose: rel. joint pos 0, gravity (0,0,-1), zero velocities/actions -> invariant."""
    ctxs = _critic_ctxs(H)
    perm, sign = _tensors(sym.build_group_map(ctxs))
    frame = {
        "projected_gravity": [0.0, 0.0, -1.0],
        "pelvis_height": [0.639],
        "root_quat": [1.0, 0.0, 0.0, 0.0],
        "body_contact_norms": [150.0, 150.0] + [0.0] * 8,
        "thermal_proxy": [0.3] * 23,
        "assist_force": [0.0, 0.0, 80.0],
    }
    parts = []
    for c in ctxs:
        d = c.dim // c.history
        parts.append(torch.tensor(frame.get(c.name, [0.0] * d)).repeat(c.history))
    x = torch.cat(parts).unsqueeze(0)
    assert torch.allclose(sym.apply_map(x, perm, sign), x)


def test_left_leg_bent_mirrors_to_right_leg_bent(urdf):
    q0 = _standing_joint_pos()
    q = dict(q0)
    # left hip flexion is negative pitch (default -0.15), left knee flexion is positive (URDF limit [0, 1.5])
    q["left_hip_pitch_joint"] = -1.2
    q["left_knee_joint"] = 1.3
    q["left_ankle_pitch_joint"] = -0.2
    q["left_hip_roll_joint"] = 0.3  # abduction
    q["left_shoulder_roll_joint"] = -0.8
    q["left_elbow_joint"] = 1.5
    perm, sign = _tensors(sym.action_map())
    qm = sym.apply_map(_q_vec(q), perm, sign)[0]
    got = dict(zip(J, qm.tolist()))
    expected = dict(q0)
    expected.update(
        right_hip_pitch_joint=1.2,
        right_knee_joint=-1.3,
        right_ankle_pitch_joint=0.2,
        right_hip_roll_joint=-0.3,
        right_shoulder_roll_joint=0.8,
        right_elbow_joint=-1.5,
    )
    for j in J:
        assert got[j] == pytest.approx(expected[j], abs=1e-6), j
        lo, hi = urdf[j]["limit"]
        assert lo - 1e-6 <= got[j] <= hi + 1e-6, f"mirrored {j}={got[j]} outside URDF limits {lo, hi}"

    # the same pose through the slot observations (relative to default)
    rel = {j: q[j] - q0[j] for j in J}
    ctxs = _policy_ctxs(1)
    x = []
    for c in ctxs:
        if c.name.startswith("joint_pos_slot"):
            x.extend(rel[j] for j in c.joint_names)
        else:
            x.extend([0.0] * c.dim)
    x = torch.tensor([x])
    y = sym.apply_map(x, *_tensors(sym.build_group_map(ctxs)))
    slot01 = y[0, 6:15].tolist()  # left_hip_pitch, left_hip_roll, right_hip_pitch, right_hip_roll, ...
    assert slot01[0] == pytest.approx(0.0) and slot01[2] == pytest.approx(-rel["left_hip_pitch_joint"])
    assert slot01[3] == pytest.approx(-0.3)
    slot23 = y[0, 15:23].tolist()  # left_hip_yaw, left_knee, right_hip_yaw, right_knee, ...
    assert slot23[1] == pytest.approx(0.0) and slot23[3] == pytest.approx(-rel["left_knee_joint"])


def _quat_to_R(q):
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def test_base_frame_vectors_and_quaternion():
    ctx1 = lambda n, d: sym.TermContext(n, d, 1)  # noqa: E731
    # lying on the LEFT side: base +y points down -> projected gravity (0, +1, 0); mirrored = lying on the right side
    perm, sign = _tensors(sym.build_term_map(ctx1("projected_gravity", 3)))
    assert sym.apply_map(torch.tensor([0.0, 1.0, 0.0]), perm, sign).tolist() == [0.0, -1.0, 0.0]
    # rolling left (+wx) mirrors to rolling right; pitching (+wy) stays; yawing flips
    perm, sign = _tensors(sym.build_term_map(ctx1("base_ang_vel", 3)))
    assert sym.apply_map(torch.tensor([1.0, 2.0, 3.0]), perm, sign).tolist() == [-1.0, 2.0, -3.0]
    # quaternion: R(q') == M R(q) M for random rotations
    perm, sign = _tensors(sym.build_term_map(ctx1("root_quat", 4)))
    rng = np.random.default_rng(0)
    for _ in range(20):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        qm = sym.apply_map(torch.tensor(q), perm, sign).numpy()
        assert np.allclose(_quat_to_R(qm), M @ _quat_to_R(q) @ M, atol=1e-6)


def test_body_swap_and_category():
    ctx = sym.TermContext("body_contact_norms", 4, 1, body_names=("pelvis_link", "left_knee_link", "right_knee_link", "waist_yaw_link"))
    perm, sign = _tensors(sym.build_term_map(ctx))
    assert sym.apply_map(torch.tensor([1.0, 2.0, 3.0, 4.0]), perm, sign).tolist() == [1.0, 3.0, 2.0, 4.0]
    # vec3 per body (polar): swap feet and flip y
    ctx = sym.TermContext("foot_contact_forces", 6, 1)
    perm, sign = _tensors(sym.build_term_map(ctx))
    assert sym.apply_map(torch.arange(6.0), perm, sign).tolist() == [3.0, -4.0, 5.0, 0.0, -1.0, 2.0]
    # category one-hot: side_left <-> side_right
    ctx = sym.TermContext("category_onehot", 8, 1)
    perm, sign = _tensors(sym.build_term_map(ctx))
    onehot = torch.eye(8)
    mirrored = sym.apply_map(onehot, perm, sign)
    names = sym.GETUP_CATEGORIES
    for i, n in enumerate(names):
        target = {"side_left": "side_right", "side_right": "side_left"}.get(n, n)
        assert mirrored[i].argmax().item() == names.index(target)


# ---------------------------------------------------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------------------------------------------------


def test_unknown_term_fails_loudly():
    with pytest.raises(KeyError, match="no_such_term"):
        sym.build_group_map([sym.TermContext("no_such_term", 3, 1)])


def test_register_rule():
    sym.register_mirror_rule("w3_test_vec", sym.SignedPerm(sign=(1.0, -1.0)))
    sym.register_mirror_rule("w3_test_vec", sym.SignedPerm(sign=(1.0, -1.0)))  # identical: no-op
    with pytest.raises(KeyError):
        sym.register_mirror_rule("w3_test_vec", sym.Invariant())
    sym.register_mirror_rule("w3_test_vec", sym.Invariant(), override=True)
    assert sym.get_mirror_rule("w3_test_vec") == sym.Invariant()
    with pytest.raises(ValueError):
        sym.build_group_map([sym.TermContext("w3_test_asym", 2, 1)], rules={"w3_test_asym": sym.SignedPerm((1.0, 1.0), (1, 1))})


# ---------------------------------------------------------------------------------------------------------------------
# rsl-rl entry point with a duck-typed Isaac Lab env
# ---------------------------------------------------------------------------------------------------------------------

# PhysX-style articulation order (differs from ASIMOV_1_JOINT_NAMES on purpose)
ARTICULATION_ORDER = tuple(sorted(J))


class _Scene(dict):
    def __init__(self):
        super().__init__(robot=SimpleNamespace(joint_names=list(ARTICULATION_ORDER), body_names=list(CONTACT_BODIES)))
        self.sensors = {"contact": SimpleNamespace(body_names=list(CONTACT_BODIES))}


def _slot_cfg(names):
    return SimpleNamespace(name="robot", joint_ids=[ARTICULATION_ORDER.index(n) for n in names], body_ids=slice(None))


def _fake_env(history=H, with_full_joint_pos=True):
    policy = [
        ("base_ang_vel", 3, {}),
        ("projected_gravity", 3, {}),
        ("joint_pos_slot01", 9, {"asset_cfg": _slot_cfg(sym.SLOT_0_1)}),
        ("joint_pos_slot23", 8, {"asset_cfg": _slot_cfg(sym.SLOT_2_3)}),
        ("joint_pos_slot45", 6, {"asset_cfg": _slot_cfg(sym.SLOT_4_5)}),
        ("joint_vel_slot01", 9, {"asset_cfg": _slot_cfg(sym.SLOT_0_1)}),
        ("joint_vel_slot23", 8, {"asset_cfg": _slot_cfg(sym.SLOT_2_3)}),
        ("joint_vel_slot45", 6, {"asset_cfg": _slot_cfg(sym.SLOT_4_5)}),
        ("actions", 23, {}),
    ]
    critic = policy + [
        ("pelvis_height", 1, {}),
        ("base_lin_vel", 3, {}),
        ("root_quat", 4, {}),
        (
            "body_contact_norms",
            len(CONTACT_BODIES),
            {"sensor_cfg": SimpleNamespace(name="contact", body_ids=list(range(len(CONTACT_BODIES))), joint_ids=slice(None))},
        ),
        ("effort_saturation", 23, {"asset_cfg": SimpleNamespace(name="robot", joint_ids=slice(None), body_ids=slice(None))}),
        ("thermal_proxy", 23, {}),
        ("assist_force", 3, {}),
    ]
    if with_full_joint_pos:
        critic.append(("joint_pos", 23, {}))  # no asset_cfg in params -> articulation order
    groups = {"policy": policy, "critic": critic}
    om = SimpleNamespace(
        active_terms={g: [t[0] for t in ts] for g, ts in groups.items()},
        group_obs_term_dim={g: [(t[1] * history,) for t in ts] for g, ts in groups.items()},
        group_obs_concatenate={g: True for g in groups},
        _group_obs_concatenate_dim={g: -1 for g in groups},
        _group_obs_term_cfgs={
            g: [SimpleNamespace(history_length=history if history > 1 else 0, flatten_history_dim=True, params=t[2]) for t in ts]
            for g, ts in groups.items()
        },
    )
    action_term = SimpleNamespace(_joint_names=list(J))
    am = SimpleNamespace(active_terms=["joint_pos"], get_term=lambda name: action_term)
    unwrapped = SimpleNamespace(observation_manager=om, action_manager=am, scene=_Scene())
    return SimpleNamespace(unwrapped=unwrapped), groups


def test_mirror_getup_end_to_end():
    env, groups = _fake_env()
    B = 32
    obs = TensorDict(
        {g: torch.randn(B, sum(t[1] for t in ts) * H) for g, ts in groups.items()}, batch_size=[B]
    )
    actions = torch.randn(B, 23)
    obs_aug, act_aug = sym.mirror_getup(env=env, obs=obs, actions=actions)
    assert obs_aug.batch_size[0] == 2 * B and act_aug.shape == (2 * B, 23)
    for g in groups:
        assert torch.equal(obs_aug[g][:B], obs[g])
        assert not torch.allclose(obs_aug[g][B:], obs[g])
    assert torch.equal(act_aug[:B], actions)
    # mirror twice = identity through the public function
    obs_aug2, act_aug2 = sym.mirror_getup(env=env, obs=obs_aug[B:], actions=act_aug[B:])
    for g in groups:
        assert torch.equal(obs_aug2[g][B:], obs[g])
    assert torch.equal(act_aug2[B:], actions)
    # None passthrough (mirror-loss call path)
    o, a = sym.mirror_getup(env=env, obs=None, actions=actions)
    assert o is None and a.shape == (2 * B, 23)
    o, a = sym.mirror_getup(env=env, obs=obs, actions=None)
    assert a is None and o.batch_size[0] == 2 * B


def test_env_binding_uses_resolved_joint_order():
    """Terms without asset_cfg use the articulation order; slot terms use their resolved asset_cfg order."""
    env, groups = _fake_env(history=1)
    D = sum(t[1] for t in groups["critic"])
    x = torch.zeros(1, D)
    off = D - 23  # trailing full joint_pos term, in ARTICULATION_ORDER
    x[0, off + ARTICULATION_ORDER.index("left_knee_joint")] = 0.7
    y = sym.mirror_obs_group(env, "critic", x)
    assert y[0, off + ARTICULATION_ORDER.index("right_knee_joint")].item() == pytest.approx(-0.7)
    assert y[0, off + ARTICULATION_ORDER.index("left_knee_joint")].item() == pytest.approx(0.0)


def test_unknown_obs_term_in_env_fails_loudly():
    env, groups = _fake_env(history=1)
    env.unwrapped.observation_manager.active_terms["critic"][-1] = "mystery_term"
    obs = TensorDict({g: torch.randn(2, sum(t[1] for t in ts)) for g, ts in groups.items()}, batch_size=[2])
    with pytest.raises(KeyError, match="mystery_term"):
        sym.mirror_getup(env=env, obs=obs, actions=None)


def _multi_term_action_manager(terms):
    """terms: list of (name, action_dim, joint_names | None) -> duck-typed Isaac Lab ActionManager."""
    objs = {n: SimpleNamespace(action_dim=d, _joint_names=list(j) if j is not None else None) for n, d, j in terms}
    return SimpleNamespace(active_terms=[t[0] for t in terms], get_term=lambda name: objs[name])


# a non-ASIMOV action order, so resolving it (vs. the ASIMOV fallback) is observable
PERMUTED_ACTION_ORDER = tuple(reversed(J))


@pytest.mark.parametrize(
    "terms",
    [
        [("joint_pos", 23, PERMUTED_ACTION_ORDER), ("assist", 0, None)],  # the real get-up env
        [("assist", 0, None), ("joint_pos", 23, PERMUTED_ACTION_ORDER)],
        [("assist", 0, None), ("legs", 23, PERMUTED_ACTION_ORDER)],  # no "joint_pos": first term with dim > 0
    ],
)
def test_action_order_resolved_with_multiple_action_terms(terms, capsys):
    env, _ = _fake_env(history=1)
    env.unwrapped.action_manager = _multi_term_action_manager(terms)
    assert sym._action_joint_names(env.unwrapped) == PERMUTED_ACTION_ORDER
    a = torch.zeros(1, 23)
    a[0, PERMUTED_ACTION_ORDER.index("left_knee_joint")] = 0.9
    _, a_aug = sym.mirror_getup(env=env, obs=None, actions=a)
    assert a_aug[1, PERMUTED_ACTION_ORDER.index("right_knee_joint")].item() == pytest.approx(-0.9)
    assert a_aug[1, PERMUTED_ACTION_ORDER.index("left_knee_joint")].item() == pytest.approx(0.0)
    assert "WARNING" not in capsys.readouterr().out


def test_action_order_fallback_kept(capsys):
    env, _ = _fake_env(history=1)
    env.unwrapped.action_manager = _multi_term_action_manager([("assist", 0, None), ("other", 0, None)])
    assert sym._action_joint_names(env.unwrapped) is None
    a = torch.randn(3, 23)
    _, a_aug = sym.mirror_getup(env=env, obs=None, actions=a)
    perm, sign = _tensors(sym.action_map())  # ASIMOV_1_JOINT_NAMES fallback
    assert torch.equal(a_aug[3:], sym.apply_map(a, perm, sign))
    assert "WARNING" in capsys.readouterr().out
