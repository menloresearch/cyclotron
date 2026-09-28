# Copyright (c) 2026, Menlo Research.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Independent end-to-end check of the joint mirror map via forward kinematics.

For random joint configurations q, the mirrored configuration q' = mirror(q) (map from tasks/getup/symmetry.py)
must place every link at the reflection of its L/R partner link:  p'(link) = M p(partner(link)),
R'(link) = M R(partner(link)) M  (M = diag(1,-1,1), pelvis/base frame). This checks signs, permutation, joint
origins and axes together, without relying on the "axis flips" argument. Pure numpy + URDF; no Isaac Lab.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
PKG = REPO / "source" / "isaac_asimov" / "isaac_asimov"
M = np.diag([1.0, -1.0, 1.0])


def _load(name, path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


sym = _load("v_getup_symmetry", PKG / "tasks" / "getup" / "symmetry.py")


def _urdf_path() -> Path:
    base = os.environ.get("ASIMOV_1_MODEL_DIR")
    cands = [Path(base) / "urdf" / "asimov_1.urdf"] if base else []
    cands.append(REPO / "third_party" / "asimov-1" / "sim-model" / "urdf" / "asimov_1.urdf")
    for c in cands:
        if c.exists():
            return c
    pytest.skip("URDF not found")


def _rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _axang(a, q):
    a = np.asarray(a, float) / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(q) * K + (1 - np.cos(q)) * K @ K


class Urdf:
    def __init__(self, path):
        root = ET.parse(path).getroot()
        self.joints = {}
        self.child_of = {}
        for j in root.findall("joint"):
            o = j.find("origin")
            xyz = np.array([float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()])
            rpy = [float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()]
            ax = j.find("axis")
            axis = np.array([float(v) for v in ax.get("xyz").split()]) if ax is not None else np.zeros(3)
            lim = j.find("limit")
            self.joints[j.get("name")] = dict(
                type=j.get("type"), parent=j.find("parent").get("link"), child=j.find("child").get("link"),
                xyz=xyz, R=_rpy(*rpy), axis=axis,
                lower=float(lim.get("lower")) if lim is not None and lim.get("lower") else None,
                upper=float(lim.get("upper")) if lim is not None and lim.get("upper") else None,
            )
            self.child_of[j.find("child").get("link")] = j.get("name")
        links = {l.get("name") for l in root.findall("link")}
        self.root = (links - set(self.child_of)).pop()

    def fk(self, q: dict):
        out = {self.root: (np.zeros(3), np.eye(3))}

        def pose(link):
            if link in out:
                return out[link]
            j = self.joints[self.child_of[link]]
            pp, pR = pose(j["parent"])
            R = pR @ j["R"]
            if j["type"] in ("revolute", "continuous"):
                R = R @ _axang(j["axis"], q.get(self.child_of[link], 0.0))
            out[link] = (pp + pR @ j["xyz"], R)
            return out[link]

        for j in self.joints.values():
            pose(j["child"])
        return out


@pytest.fixture(scope="module")
def urdf():
    return Urdf(_urdf_path())


def test_every_actuated_joint_is_in_the_action_order(urdf):
    act = sorted(n for n, j in urdf.joints.items() if j["type"] in ("revolute", "continuous", "prismatic"))
    assert sorted(sym.ASIMOV_1_JOINT_NAMES) == act


def test_mirror_map_is_a_geometric_reflection(urdf):
    names = list(sym.ASIMOV_1_JOINT_NAMES)
    perm, sign = sym.action_map(names)
    rng = np.random.default_rng(0)
    for _ in range(50):
        q = np.array([rng.uniform(urdf.joints[n]["lower"], urdf.joints[n]["upper"]) for n in names])
        qm = np.array([sign[i] * q[perm[i]] for i in range(len(names))])
        fk0 = urdf.fk(dict(zip(names, q)))
        fk1 = urdf.fk(dict(zip(names, qm)))
        for link, (p1, R1) in fk1.items():
            p0, R0 = fk0[sym.mirror_name(link)]
            assert np.allclose(p1, M @ p0, atol=2e-3), (link, p1, M @ p0)
            assert np.allclose(R1, M @ R0 @ M, atol=1e-6), link


def test_mirrored_configuration_stays_inside_urdf_limits(urdf):
    names = list(sym.ASIMOV_1_JOINT_NAMES)
    perm, sign = sym.action_map(names)
    for i, n in enumerate(names):
        m = names[perm[i]]
        lo, hi = urdf.joints[m]["lower"], urdf.joints[m]["upper"]
        # mirrored interval of the partner must equal this joint's interval
        mlo, mhi = sorted((sign[i] * lo, sign[i] * hi))
        assert np.isclose(mlo, urdf.joints[n]["lower"]) and np.isclose(mhi, urdf.joints[n]["upper"]), n


def test_slot_maps_equal_the_action_map_restricted(urdf):
    for slot in (sym.SLOT_0_1, sym.SLOT_2_3, sym.SLOT_4_5):
        rule = sym.JointMirror(-1.0, tuple(slot))
        ctx = sym.TermContext(name="x", dim=len(slot))
        perm, sign = rule.frame_map(ctx, len(slot))
        for i, n in enumerate(slot):
            assert slot[perm[i]] == sym.mirror_name(n) and sign[i] == -1.0
