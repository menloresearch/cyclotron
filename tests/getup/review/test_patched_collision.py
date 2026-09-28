# Copyright (c) 2026, Menlo Research.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Self-collision check of the body-shell patched URDF at the poses the get-up needs.

PhysX (self-collisions on) filters only parent/child link pairs. The shell boxes added by
``assets/robots/getup_urdf.py`` must not overlap non-adjacent links' shapes at poses the task must reach (deep
crouch hip pitch -2.09 / knee 1.5, sitting, kneeling), or the crouch analysis (which used the stock URDF) is void.

Shapes are sampled as interior points on a 5 mm grid; an overlap is reported when a point of one shape lies inside
another shape of a non-adjacent link (fixed joints merged as the importer does). Pure numpy.
"""

from __future__ import annotations

import importlib.util
import itertools
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
ROB = REPO / "source" / "isaac_asimov" / "isaac_asimov" / "assets" / "robots"
URDF = REPO / "third_party" / "asimov-1" / "sim-model" / "urdf" / "asimov_1.urdf"
sys.path.insert(0, str(Path(__file__).parent))
from test_mirror_fk import Urdf, _rpy  # noqa: E402


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _patched(tmp):
    gu = _load("v_getup_urdf", ROB / "getup_urdf.py")
    return Path(gu.build_getup_urdf(str(URDF), out_root=tmp))


def _shapes(path):
    """link -> list of (kind, T_link_shape (4x4), dims)."""
    root = ET.parse(path).getroot()
    out = {}
    for link in root.findall("link"):
        for col in link.findall("collision"):
            o = col.find("origin")
            xyz = np.array([float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()])
            rpy = [float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()]
            T = np.eye(4)
            T[:3, :3], T[:3, 3] = _rpy(*rpy), xyz
            g = col.find("geometry")[0]
            if g.tag == "sphere":
                dims = (float(g.get("radius")),)
            elif g.tag == "cylinder":
                dims = (float(g.get("radius")), float(g.get("length")))
            elif g.tag == "box":
                dims = tuple(float(v) for v in g.get("size").split())
            else:
                continue
            out.setdefault(link.get("name"), []).append((g.tag, T, dims, col.get("name") or ""))
    return out


def _inside(kind, dims, p):  # p in shape frame, [K,3]
    if kind == "sphere":
        return np.linalg.norm(p, axis=1) < dims[0] - 1e-3
    if kind == "cylinder":
        return (np.linalg.norm(p[:, :2], axis=1) < dims[0] - 1e-3) & (np.abs(p[:, 2]) < dims[1] / 2 - 1e-3)
    h = np.array(dims) / 2 - 1e-3
    return np.all(np.abs(p) < h, axis=1)


def _samples(kind, dims, step=0.005):
    if kind == "sphere":
        r = dims[0]
        g = np.arange(-r, r + 1e-9, step)
        p = np.array(list(itertools.product(g, g, g)))
        return p[np.linalg.norm(p, axis=1) <= r]
    if kind == "cylinder":
        r, L = dims
        g = np.arange(-r, r + 1e-9, step)
        z = np.arange(-L / 2, L / 2 + 1e-9, step)
        p = np.array(list(itertools.product(g, g, z)))
        return p[np.linalg.norm(p[:, :2], axis=1) <= r]
    axes = [np.arange(-d / 2, d / 2 + 1e-9, step) for d in dims]
    return np.array(list(itertools.product(*axes)))


def _body_of(urdf, link):
    while True:
        jn = urdf.child_of.get(link)
        if jn is None or urdf.joints[jn]["type"] != "fixed":
            return link
        link = urdf.joints[jn]["parent"]


def overlaps(urdf, shapes, q):
    fk = urdf.fk(q)
    world = []
    for link, lst in shapes.items():
        p, R = fk[link]
        Tl = np.eye(4)
        Tl[:3, :3], Tl[:3, 3] = R, p
        for kind, T, dims, name in lst:
            world.append((_body_of(urdf, link), Tl @ T, kind, dims, name))
    adjacent = set()
    for j in urdf.joints.values():
        a, b = _body_of(urdf, j["parent"]), _body_of(urdf, j["child"])
        adjacent |= {(a, b), (b, a)}
    hits = []
    for (ba, Ta, ka, da, na), (bb, Tb, kb, db, nb) in itertools.combinations(world, 2):
        if ba == bb or (ba, bb) in adjacent:
            continue
        pa = _samples(ka, da) @ Ta[:3, :3].T + Ta[:3, 3]
        inv = np.linalg.inv(Tb)
        pb = pa @ inv[:3, :3].T + inv[:3, 3]
        n = int(_inside(kb, db, pb).sum())
        if n:
            hits.append((ba, na or ka, bb, nb or kb, n))
    return hits


def pose(**kw):
    """Symmetric pose from left-joint values (right = mirrored, -q)."""
    q = {}
    for k, v in kw.items():
        q[f"left_{k}_joint"] = v
        q[f"right_{k}_joint"] = -v
    return q


POSES = {
    "deep_crouch": pose(hip_pitch=-2.09, knee=1.5, ankle_pitch=-0.35),
    "crouch_1.8": pose(hip_pitch=-1.8, knee=1.5, ankle_pitch=-0.35),
    "sitting_90": pose(hip_pitch=-1.57, knee=1.2),
    "kneel_up": pose(hip_pitch=0.0, knee=1.5, ankle_pitch=0.35),
    "kneel_down": pose(hip_pitch=-1.2, knee=1.4, shoulder_pitch=-1.2),
    "arms_push_back": pose(shoulder_pitch=1.0, elbow=0.4),
    "arms_down_side": pose(shoulder_roll=0.0, elbow=0.0),
}


@pytest.fixture(scope="module")
def model():
    with tempfile.TemporaryDirectory() as tmp:
        path = _patched(tmp)
        yield Urdf(path), _shapes(path)


@pytest.mark.parametrize("name", list(POSES))
def test_no_patch_self_collision(model, name):
    urdf, shapes = model
    hits = overlaps(urdf, shapes, POSES[name])
    patch_hits = [h for h in hits if "getup_patch" in h[1] or "getup_patch" in h[3]]
    assert not patch_hits, f"{name}: {patch_hits}"


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        path = _patched(tmp)
        urdf, shapes = Urdf(path), _shapes(path)
        stock = _shapes(URDF)
        for name, q in POSES.items():
            print(name, "patched:", overlaps(urdf, shapes, q))
            print(name, "stock  :", overlaps(urdf, stock, q))
