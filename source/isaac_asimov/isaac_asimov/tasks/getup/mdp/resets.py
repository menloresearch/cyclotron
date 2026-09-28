# Copyright (c) 2026, Menlo Research.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# The fallen-state cache design (drop limp robots, keep settled states, sample them at reset) follows NVIDIA WBC-AGILE
# `agile/rl_env/mdp/events/fallen_state_dataset.py` (Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES,
# Apache-2.0, https://github.com/nvidia-isaac/WBC-AGILE). No code is copied verbatim; see
# `scripts/getup/build_fallen_cache.py` for the builder.
"""Get-up start states: the reset event and the fallen-state cache.

Public API (re-exported by ``tasks/getup/mdp/__init__.py``):

* :data:`GETUP_CATEGORIES` - fixed category key order. ``env.getup_state.category`` stores an index into it.
* :func:`reset_fallen_state` - reset event (term name ``reset_fallen``).
* :func:`update_category_probs` - adaptive category reweighting hook (p_c ∝ (1 - success_c) + 0.1, floor 0.03).
* :func:`label_fallen_states` - the category labeler used by the cache builder (and usable for evaluation).
* :class:`CollisionSpheres` / :func:`get_collision_spheres` - sphere model of the robot's collision shapes (every
  Asimov 1 collision shape is a capsule or a sphere), used for exact ground clearance and contact labels.
* :func:`load_fallen_cache` - load a cache file produced by ``scripts/getup/build_fallen_cache.py``.

Cache file format (``torch.save`` dict, version 1)::

    version:     1
    joint_names: list[str]            joint order of ``joint_pos`` / ``joint_vel``
    categories:  tuple[str]            == GETUP_CATEGORIES
    root_height: float32 [M]           root (pelvis link) z above the flat ground the state settled on
    root_quat:   float32 [M, 4]        (w, x, y, z), world frame, yaw as settled (re-randomized at reset)
    joint_pos:   float32 [M, J]
    joint_vel:   float32 [M, J]        residual settled velocities (written as zero by default)
    label:       int8    [M]           index into GETUP_CATEGORIES (only 0..5 are stored)
    seed_type:   int8    [M]           index into ``meta["seed_types"]`` (spawn recipe that produced the state)
    features:    dict[str, Tensor]     labeler features (pelvis height, torso up-z, contact flags, ...)
    meta:        dict                  builder args, asset, URDF path, per-category counts, rejection stats
"""

from __future__ import annotations

import math
import os
import warnings
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils

from .limp import set_limp_mask

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedEnv

__all__ = [
    "GETUP_CATEGORIES",
    "CATEGORY_INDEX",
    "CACHED_CATEGORIES",
    "DEFAULT_CATEGORY_PROBS",
    "MIRROR_CATEGORY",
    "OTHER_LABEL",
    "CollisionSpheres",
    "FallenStateCache",
    "TerrainHeight",
    "build_collision_spheres",
    "check_cache_asset",
    "compute_label_inputs",
    "get_collision_spheres",
    "ground_penetration",
    "joint_mirror_perm",
    "label_fallen_states",
    "load_fallen_cache",
    "mirror_root_quat",
    "reset_fallen_state",
    "reweight_category_probs",
    "update_category_probs",
]

# ---------------------------------------------------------------------------------------------------------------------
# Categories
# ---------------------------------------------------------------------------------------------------------------------

GETUP_CATEGORIES: tuple[str, ...] = (
    "supine",
    "prone",
    "side_left",
    "side_right",
    "sitting",
    "kneeling",
    "mid_fall",
    "standing",
)
"""Fixed category key order. ``env.getup_state.category`` stores the index into this tuple."""

CATEGORY_INDEX: dict[str, int] = {c: i for i, c in enumerate(GETUP_CATEGORIES)}

CACHED_CATEGORIES: tuple[str, ...] = GETUP_CATEGORIES[:6]
"""Categories sampled from the fallen-state cache (settled, limp-dropped states)."""

MID_FALL = CATEGORY_INDEX["mid_fall"]
STANDING = CATEGORY_INDEX["standing"]
OTHER_LABEL = -1
"""Label for settled states that fit no category (inverted, crouching on the feet, ...). Never stored in the cache."""

MIRROR_CATEGORY: dict[str, str] = {c: c for c in GETUP_CATEGORIES} | {"side_left": "side_right", "side_right": "side_left"}
"""Category of the left/right mirror image of a state."""

DEFAULT_CATEGORY_PROBS: dict[str, float] = {
    "supine": 0.22,
    "prone": 0.22,
    "side_left": 0.08,
    "side_right": 0.08,
    "sitting": 0.10,
    "kneeling": 0.08,
    "mid_fall": 0.14,
    "standing": 0.08,
}
"""Initial category mix."""

CACHE_FORMAT_VERSION = 1

# Body names of Asimov 1 used by the labeler.
PELVIS_BODY = "pelvis_link"
TORSO_BODY = "waist_yaw_link"
KNEE_BODIES = ("left_knee_link", "right_knee_link")
FOOT_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link")
THIGH_BODIES = ("left_hip_pitch_link", "right_hip_pitch_link", "left_hip_yaw_link", "right_hip_yaw_link")


# ---------------------------------------------------------------------------------------------------------------------
# Shared state access
# ---------------------------------------------------------------------------------------------------------------------


class _FallbackState:
    """Minimal stand-in for ``GetUpState`` when ``mdp/state.py`` is unavailable (standalone tests only)."""

    def __init__(self, env: ManagerBasedEnv):
        n, dev = env.num_envs, env.device
        self.category = torch.zeros(n, dtype=torch.long, device=dev)
        self.control_start_step = torch.zeros(n, dtype=torch.long, device=dev)
        self.policy_active = torch.ones(n, dtype=torch.bool, device=dev)
        self.step = torch.zeros(n, dtype=torch.long, device=dev)


def get_state(env: ManagerBasedEnv):
    """``env.getup_state`` via :func:`~.state.ensure_state` (fallback: a minimal local state object)."""
    try:
        from .state import ensure_state
    except ImportError:
        ensure_state = None
    if ensure_state is not None:
        return ensure_state(env)
    st = getattr(env, "getup_state", None)
    if st is None:
        st = _FallbackState(env)
        env.getup_state = st
    return st


# ---------------------------------------------------------------------------------------------------------------------
# Collision-sphere model (exact for Asimov 1: every collision shape is a sphere or a capsule = cylinder + 2 spheres)
# ---------------------------------------------------------------------------------------------------------------------


def _rpy_to_matrix(rpy: tuple[float, float, float]) -> torch.Tensor:
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    rx = torch.tensor([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=torch.float64)
    ry = torch.tensor([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=torch.float64)
    rz = torch.tensor([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=torch.float64)
    return rz @ ry @ rx


def _origin_to_tf(elem) -> torch.Tensor:
    tf = torch.eye(4, dtype=torch.float64)
    if elem is None:
        return tf
    xyz = [float(v) for v in elem.get("xyz", "0 0 0").split()]
    rpy = [float(v) for v in elem.get("rpy", "0 0 0").split()]
    tf[:3, :3] = _rpy_to_matrix(tuple(rpy))
    tf[:3, 3] = torch.tensor(xyz, dtype=torch.float64)
    return tf


@dataclass
class CollisionSpheres:
    """Swept-sphere model of an articulation's collision shapes, in articulation body frames.

    A capsule (cylinder + 2 end spheres, the URDF style of Asimov 1) is represented exactly by its two end spheres for
    "lowest point" queries. A bare cylinder is approximated by spheres at its cap centres (conservative), a box by its
    8 corners (radius 0). Meshes are skipped with a warning.
    """

    body_names: list[str]
    body_ids: torch.Tensor  # [S] long, index into the articulation's bodies
    offsets: torch.Tensor  # [S, 3] sphere centre in the body frame
    radii: torch.Tensor  # [S]
    num_bodies: int
    source: str = ""

    @property
    def num_spheres(self) -> int:
        return int(self.body_ids.numel())

    def centers_w(self, body_pos_w: torch.Tensor, body_quat_w: torch.Tensor) -> torch.Tensor:
        """World sphere centres [N, S, 3] from body link poses [N, B, 3] / [N, B, 4]."""
        pos = body_pos_w[:, self.body_ids]
        quat = body_quat_w[:, self.body_ids]
        n, s = pos.shape[:2]
        off = self.offsets.unsqueeze(0).expand(n, s, 3)
        return pos + math_utils.quat_apply(quat.reshape(-1, 4), off.reshape(-1, 3)).view(n, s, 3)

    def lowest_per_body(self, sphere_bottom: torch.Tensor) -> torch.Tensor:
        """[N, S] sphere bottoms -> [N, B] lowest point per body (+inf for bodies without collision shapes)."""
        n = sphere_bottom.shape[0]
        out = torch.full((n, self.num_bodies), float("inf"), device=sphere_bottom.device, dtype=sphere_bottom.dtype)
        return out.scatter_reduce(1, self.body_ids.unsqueeze(0).expand(n, -1), sphere_bottom, reduce="amin")


def build_collision_spheres(urdf_path: str, body_names: list[str], device: str | torch.device) -> CollisionSpheres:
    """Parse the URDF collision shapes into a :class:`CollisionSpheres` model for ``body_names``.

    Links that are not articulation bodies (merged by ``merge_fixed_joints``) are attached to their nearest body
    ancestor through the fixed-joint chain, as the URDF importer does.
    """
    root = ET.parse(os.path.expanduser(urdf_path)).getroot()
    parent_joint = {}
    for j in root.findall("joint"):
        parent_joint[j.find("child").get("link")] = (j.find("parent").get("link"), j.get("type"), _origin_to_tf(j.find("origin")))
    body_index = {n: i for i, n in enumerate(body_names)}
    ids, offs, rads, skipped = [], [], [], []
    for link in root.findall("link"):
        name = link.get("name")
        cols = link.findall("collision")
        if not cols:
            continue
        # walk up fixed joints to the owning articulation body
        tf_body_link = torch.eye(4, dtype=torch.float64)
        cur = name
        while cur not in body_index:
            if cur not in parent_joint or parent_joint[cur][1] != "fixed":
                cur = None
                break
            par, _, tf = parent_joint[cur]
            tf_body_link = tf @ tf_body_link
            cur = par
        if cur is None:
            skipped.append(name)
            continue
        for col in cols:
            tf = tf_body_link @ _origin_to_tf(col.find("origin"))
            geom = col.find("geometry")[0]
            pts, r = [], 0.0
            if geom.tag == "sphere":
                r = float(geom.get("radius"))
                pts = [(0.0, 0.0, 0.0)]
            elif geom.tag == "cylinder":
                r = float(geom.get("radius"))
                half = 0.5 * float(geom.get("length"))
                pts = [(0.0, 0.0, half), (0.0, 0.0, -half)]
            elif geom.tag == "capsule":
                r = float(geom.get("radius"))
                half = 0.5 * float(geom.get("length"))
                pts = [(0.0, 0.0, half), (0.0, 0.0, -half)]
            elif geom.tag == "box":
                sx, sy, sz = (0.5 * float(v) for v in geom.get("size").split())
                pts = [(a * sx, b * sy, c * sz) for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)]
            else:
                skipped.append(f"{name}:{geom.tag}")
                continue
            for p in pts:
                pb = tf[:3, :3] @ torch.tensor(p, dtype=torch.float64) + tf[:3, 3]
                ids.append(body_index[cur])
                offs.append(pb)
                rads.append(r)
    if skipped:
        warnings.warn(f"[getup.resets] collision shapes not modelled (mesh or no body): {skipped}")
    if not ids:
        raise RuntimeError(f"No collision spheres parsed from {urdf_path}")
    return CollisionSpheres(
        body_names=list(body_names),
        body_ids=torch.tensor(ids, dtype=torch.long, device=device),
        offsets=torch.stack(offs).to(device=device, dtype=torch.float32),
        radii=torch.tensor(rads, dtype=torch.float32, device=device),
        num_bodies=len(body_names),
        source=str(urdf_path),
    )


def get_collision_spheres(robot: Articulation) -> CollisionSpheres:
    """Cached :class:`CollisionSpheres` for ``robot``, parsed from its spawn URDF.

    Fallback when the asset is not a URDF: one 6 cm sphere per body origin (coarse, warns).
    """
    model = getattr(robot, "_getup_collision_spheres", None)
    if model is not None:
        return model
    path = getattr(robot.cfg.spawn, "asset_path", None)
    if path is not None and str(path).endswith(".urdf") and os.path.isfile(os.path.expanduser(str(path))):
        model = build_collision_spheres(str(path), robot.body_names, robot.device)
    else:
        warnings.warn(f"[getup.resets] asset is not a readable URDF ({path}); using a coarse sphere-per-body model")
        b = robot.num_bodies
        model = CollisionSpheres(
            body_names=list(robot.body_names),
            body_ids=torch.arange(b, device=robot.device),
            offsets=torch.zeros(b, 3, device=robot.device),
            radii=torch.full((b,), 0.06, device=robot.device),
            num_bodies=b,
            source="coarse",
        )
    robot._getup_collision_spheres = model
    return model


# ---------------------------------------------------------------------------------------------------------------------
# Terrain height queries (plane: 0; generator / usd terrain: ray cast against the terrain mesh)
# ---------------------------------------------------------------------------------------------------------------------


class TerrainHeight:
    """Ground height below world xy points. Exact for a plane; ray cast against the terrain mesh otherwise."""

    def __init__(self, env: ManagerBasedEnv):
        terrain = getattr(env.scene, "terrain", None)
        self.device = env.device
        self.mesh = None
        self.flat = terrain is None or terrain.cfg.terrain_type == "plane"
        if not self.flat:
            self.mesh = self._read_mesh(terrain.cfg.prim_path)

    def _read_mesh(self, prim_path: str):
        import numpy as np
        import omni.usd
        from pxr import UsdGeom

        import isaaclab.sim as sim_utils
        from isaaclab.utils.warp import convert_to_warp_mesh

        prim = sim_utils.get_first_matching_child_prim(prim_path, lambda p: p.GetTypeName() == "Mesh")
        if prim is None or not prim.IsValid():
            warnings.warn(f"[getup.resets] no terrain mesh under {prim_path}; assuming flat ground at z=0")
            self.flat = True
            return None
        mesh = UsdGeom.Mesh(prim)
        points = np.asarray(mesh.GetPointsAttr().Get())
        tf = np.array(omni.usd.get_world_transform_matrix(prim)).T
        points = points @ tf[:3, :3].T + tf[:3, 3]
        indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
        return convert_to_warp_mesh(points, indices, device=str(self.device))

    def __call__(self, xy: torch.Tensor) -> torch.Tensor:
        """[..., 2] world xy -> [...] ground z."""
        if self.flat:
            return torch.zeros(xy.shape[:-1], device=xy.device, dtype=xy.dtype)
        from isaaclab.utils.warp import raycast_mesh

        flat_xy = xy.reshape(-1, 2)
        starts = torch.cat([flat_xy, torch.full_like(flat_xy[:, :1], 100.0)], dim=-1)
        dirs = torch.zeros_like(starts)
        dirs[:, 2] = -1.0
        hits = raycast_mesh(starts, dirs, self.mesh)[0]
        z = torch.nan_to_num(hits[:, 2], nan=0.0, posinf=0.0, neginf=0.0)
        return z.view(xy.shape[:-1]).to(xy.dtype)


def _terrain_height(env: ManagerBasedEnv) -> TerrainHeight:
    th = env.__dict__.get("_getup_terrain_height")
    if th is None:
        th = TerrainHeight(env)
        env.__dict__["_getup_terrain_height"] = th
    return th


def ground_penetration(
    robot: Articulation, env_ids: torch.Tensor, spheres: CollisionSpheres, terrain: TerrainHeight
) -> torch.Tensor:
    """Max penetration depth of any collision sphere into the terrain [n] (<= 0 means no penetration)."""
    pos = robot.data.body_link_pos_w[env_ids]
    quat = robot.data.body_link_quat_w[env_ids]
    centers = spheres.centers_w(pos, quat)
    ground = terrain(centers[..., :2])
    return (ground + spheres.radii - centers[..., 2]).amax(dim=1)


# ---------------------------------------------------------------------------------------------------------------------
# Labeler
# ---------------------------------------------------------------------------------------------------------------------

# Thresholds (metres / cosines); the rules are described in :func:`label_fallen_states`.
LABEL_CONTACT_TOL = 0.02  # a body touches the ground if its lowest collision point is < 2 cm above it
LABEL_INVERTED_UPZ = -0.5  # torso z axis pointing > 120 deg from world up -> other
LABEL_UPRIGHT_UPZ = 0.5  # torso tilt < 60 deg counts as "upright" for sitting
LABEL_KNEEL_MIN_PELVIS = 0.18  # kneeling: pelvis origin at least this high
LABEL_KNEEL_KNEE_BELOW_PELVIS = 0.10  # kneeling: knee cap at least this far below the pelvis origin
LABEL_SIT_MAX_PELVIS = 0.20  # sitting: pelvis origin at most this high


def label_fallen_states(
    torso_quat_w: torch.Tensor,
    pelvis_height: torch.Tensor,
    body_lowest: torch.Tensor,
    knee_cap_height: torch.Tensor,
    body_index: dict[str, int],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Label settled states by torso orientation and ground contacts.

    Args:
        torso_quat_w: [N, 4] torso (``waist_yaw_link``) orientation (w, x, y, z).
        pelvis_height: [N] pelvis link origin height above the ground.
        body_lowest: [N, B] lowest collision point of each body above the ground (+inf if the body has no shapes).
        knee_cap_height: [N, 2] lowest point of the left/right knee region (upper shin) above the ground.
        body_index: body name -> column of ``body_lowest``.

    Returns:
        (labels [N] long in {-1, 0..5}, features dict).

    "Knee cap" below = the knee region: collision points of ``*_knee_link`` at link z >= -0.10 m (upper shin).

    Rules, evaluated in this order (u = torso z axis, f = torso x axis (chest), l = torso y axis (left), in world):

    1. ``other``   if u_z < -0.5 (inverted, e.g. head-stand).
    2. ``kneeling`` if a knee cap touches the ground, the pelvis does not, the pelvis origin is >= 0.18 m high and the
       knee cap is >= 0.10 m below the pelvis origin. Covers upright kneeling (held by the knee flexion limit) and
       torso-down kneeling (knees under the hips, chest/arms/head on the ground: the hands-and-knees key state).
    3. ``sitting``  if u_z >= 0.5 (torso tilt <= 60 deg), the pelvis origin is <= 0.20 m high, the pelvis or a thigh
       touches the ground, and the torso does not.
    4. ``other``    if u_z >= 0.5 (upright but neither kneeling nor sitting, e.g. crouched on the feet).
    5. lying (|u_z| < 0.5): argmax of (f_z, -f_z, -l_z, l_z) -> supine (chest up), prone (chest down),
       side_left (left side down), side_right (right side down).
    """
    n = torso_quat_w.shape[0]
    dev = torso_quat_w.device
    ex = torch.tensor([1.0, 0.0, 0.0], device=dev).expand(n, 3)
    ey = torch.tensor([0.0, 1.0, 0.0], device=dev).expand(n, 3)
    ez = torch.tensor([0.0, 0.0, 1.0], device=dev).expand(n, 3)
    f_z = math_utils.quat_apply(torso_quat_w, ex)[:, 2]
    l_z = math_utils.quat_apply(torso_quat_w, ey)[:, 2]
    u_z = math_utils.quat_apply(torso_quat_w, ez)[:, 2]

    def touching(names) -> torch.Tensor:
        cols = [body_index[b] for b in names if b in body_index]
        if not cols:
            return torch.zeros(n, dtype=torch.bool, device=dev)
        return (body_lowest[:, cols] < LABEL_CONTACT_TOL).any(dim=1)

    pelvis_c = touching([PELVIS_BODY])
    torso_c = touching([TORSO_BODY])
    thigh_c = touching(THIGH_BODIES)
    feet_c = touching(FOOT_BODIES)
    knee_c = knee_cap_height < LABEL_CONTACT_TOL  # [N, 2]
    knee_any = knee_c.any(dim=1)
    knee_low = knee_cap_height.min(dim=1).values
    knee_below = knee_low < pelvis_height - LABEL_KNEEL_KNEE_BELOW_PELVIS

    labels = torch.full((n,), OTHER_LABEL, dtype=torch.long, device=dev)
    undecided = u_z >= LABEL_INVERTED_UPZ

    kneel = undecided & knee_any & ~pelvis_c & (pelvis_height >= LABEL_KNEEL_MIN_PELVIS) & knee_below
    labels[kneel] = CATEGORY_INDEX["kneeling"]
    undecided &= ~kneel

    upright = u_z >= LABEL_UPRIGHT_UPZ
    sit = undecided & upright & (pelvis_height <= LABEL_SIT_MAX_PELVIS) & (pelvis_c | thigh_c) & ~torso_c
    labels[sit] = CATEGORY_INDEX["sitting"]
    undecided &= ~sit & ~upright

    scores = torch.stack([f_z, -f_z, -l_z, l_z], dim=1)
    lying = scores.argmax(dim=1)  # 0 supine, 1 prone, 2 side_left, 3 side_right == category indices 0..3
    labels[undecided] = lying[undecided]

    feats = {
        "pelvis_height": pelvis_height,
        "torso_up_z": u_z,
        "torso_fwd_z": f_z,
        "torso_left_z": l_z,
        "contact_pelvis": pelvis_c,
        "contact_torso": torso_c,
        "contact_thigh": thigh_c,
        "contact_feet": feet_c,
        "contact_knee_l": knee_c[:, 0],
        "contact_knee_r": knee_c[:, 1],
    }
    return labels, feats


KNEE_REGION_Z = -0.10
"""Collision points of ``*_knee_link`` with link-frame z >= this (the upper shin, around the knee joint) form the knee."""


def _knee_region_spheres(spheres: CollisionSpheres, body_index: dict[str, int]) -> list[torch.Tensor]:
    """Per knee link: indices of its collision points near the knee joint (upper shin, link z >= ``KNEE_REGION_Z``).

    Derived from the collision model, so it follows geometry patches (stock URDF: the r = 0.04 knee sphere at
    z = -0.04; shell-matched collision box: its upper corners).
    """
    out = []
    for name in KNEE_BODIES:
        sel = (spheres.body_ids == body_index[name]).nonzero().flatten()
        sel = sel[spheres.offsets[sel, 2] >= KNEE_REGION_Z]
        if sel.numel() == 0:
            raise RuntimeError(f"no collision points near the knee joint on {name}")
        out.append(sel)
    return out


def compute_label_inputs(robot: Articulation, spheres: CollisionSpheres, terrain: TerrainHeight | None = None):
    """Labeler inputs for all envs of ``robot`` from the current sim state (flat ground at z=0 when ``terrain`` is None).

    Returns (torso_quat_w, pelvis_height, body_lowest, knee_cap_height, body_index, min_penetration_free_height).
    """
    pos = robot.data.body_link_pos_w
    quat = robot.data.body_link_quat_w
    centers = spheres.centers_w(pos, quat)
    ground = terrain(centers[..., :2]) if terrain is not None else torch.zeros_like(centers[..., 2])
    bottoms = centers[..., 2] - spheres.radii - ground
    body_lowest = spheres.lowest_per_body(bottoms)
    body_index = {b: i for i, b in enumerate(robot.body_names)}
    torso_quat = quat[:, body_index[TORSO_BODY]]
    pelvis_id = body_index[PELVIS_BODY]
    pelvis_ground = terrain(pos[:, pelvis_id, :2]) if terrain is not None else 0.0
    pelvis_h = pos[:, pelvis_id, 2] - pelvis_ground
    knee_cap_h = torch.stack([bottoms[:, sel].amin(dim=1) for sel in _knee_region_spheres(spheres, body_index)], dim=1)
    return torso_quat, pelvis_h, body_lowest, knee_cap_h, body_index, bottoms.amin(dim=1)


# ---------------------------------------------------------------------------------------------------------------------
# Mirror helpers (Asimov 1: every L/R joint pair mirrors with sign -1, waist yaw flips sign)
# ---------------------------------------------------------------------------------------------------------------------


def joint_mirror_perm(joint_names: list[str]) -> list[int]:
    """perm such that mirrored q = -q[perm] (joint order of ``joint_names``)."""
    idx = {n: i for i, n in enumerate(joint_names)}
    perm = []
    for n in joint_names:
        if n.startswith("left_"):
            m = "right_" + n[len("left_"):]
        elif n.startswith("right_"):
            m = "left_" + n[len("right_"):]
        else:
            m = n
        perm.append(idx[m])
    return perm


def mirror_root_quat(quat: torch.Tensor) -> torch.Tensor:
    """Reflect a (w, x, y, z) orientation through the world x-z plane (body y axis mirrored)."""
    return quat * torch.tensor([1.0, -1.0, 1.0, -1.0], device=quat.device, dtype=quat.dtype)


# ---------------------------------------------------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------------------------------------------------


@dataclass
class FallenStateCache:
    """Device-resident cache, states sorted by label, joints in the articulation's joint order."""

    path: str
    root_height: torch.Tensor  # [M]
    root_quat: torch.Tensor  # [M, 4]
    joint_pos: torch.Tensor  # [M, J]
    joint_vel: torch.Tensor  # [M, J]
    label: torch.Tensor  # [M] long
    start: torch.Tensor  # [C] first index of each category (C = len(GETUP_CATEGORIES))
    count: torch.Tensor  # [C] number of states per category (0 for mid_fall / standing)
    count_list: list[int]  # host copy of ``count`` (no GPU sync in the reset path)
    meta: dict

    @property
    def num_states(self) -> int:
        return int(self.label.numel())


def load_fallen_cache(
    path: str, joint_names: list[str], device: str | torch.device, include_held: bool = True
) -> FallenStateCache:
    """Load a cache file and remap joints to ``joint_names`` order.

    ``include_held=False`` drops posture-held states (``features["hold_scale"] > 0``: settled under a weak PD hold toward
    the spawn pose instead of limp; all hands-and-knees kneeling states and some sitting/prone states).
    """
    path = os.path.expanduser(path)
    data = torch.load(path, map_location="cpu", weights_only=False)
    if data.get("version") != CACHE_FORMAT_VERSION:
        raise ValueError(f"{path}: unsupported cache version {data.get('version')}")
    if tuple(data["categories"]) != GETUP_CATEGORIES:
        raise ValueError(f"{path}: category order {data['categories']} != {GETUP_CATEGORIES}")
    hold = data.get("features", {}).get("hold_scale")
    if not include_held and hold is not None:
        keep = hold <= 0
        for key in ("root_height", "root_quat", "joint_pos", "joint_vel", "label"):
            data[key] = data[key][keep]
    src_names = list(data["joint_names"])
    missing = [n for n in joint_names if n not in src_names]
    if missing:
        raise ValueError(f"{path}: cache lacks joints {missing}")
    cols = torch.tensor([src_names.index(n) for n in joint_names], dtype=torch.long)
    label = data["label"].long()
    order = torch.argsort(label, stable=True)
    label = label[order]
    num_c = len(GETUP_CATEGORIES)
    count = torch.bincount(label.clamp(min=0), minlength=num_c)[:num_c]
    start = torch.cumsum(count, 0) - count
    return FallenStateCache(
        path=path,
        root_height=data["root_height"][order].float().to(device),
        root_quat=data["root_quat"][order].float().to(device),
        joint_pos=data["joint_pos"][order][:, cols].float().to(device),
        joint_vel=data["joint_vel"][order][:, cols].float().to(device),
        label=label.to(device),
        start=start.to(device),
        count=count.to(device),
        count_list=[int(v) for v in count.tolist()],
        meta=data.get("meta", {}),
    )


def _urdf_id(path: str | None) -> str | None:
    """Identity of a spawn URDF: its real path (patched URDFs live in a directory named by their content hash)."""
    if not path:
        return None
    return os.path.realpath(os.path.expanduser(str(path)))


def check_cache_asset(cache: FallenStateCache, robot: Articulation) -> str | None:
    """Return a mismatch message if the cache was built on a different spawn URDF than ``robot``'s, else None."""
    built = _urdf_id(cache.meta.get("urdf"))
    live = _urdf_id(getattr(robot.cfg.spawn, "asset_path", None))
    if built is None or live is None:
        return f"cannot verify the cache asset (cache urdf={built}, live urdf={live})"
    if built != live:
        return (f"cache {cache.path} was built on URDF {built} (asset {cache.meta.get('asset')}) but the robot spawns "
                f"{live}: collision geometry may differ, states can start inside the new shapes")
    return None


def _get_cache(
    env: ManagerBasedEnv,
    robot: Articulation,
    cache_path: str,
    include_held: bool = True,
    allow_asset_mismatch: bool = False,
) -> FallenStateCache:
    caches = env.__dict__.setdefault("_getup_fallen_caches", {})
    path = os.path.expanduser(cache_path)
    key = (path, bool(include_held))
    if key not in caches:
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"[getup.reset_fallen_state] fallen-state cache not found: {path}. Build it with "
                "scripts/getup/build_fallen_cache.py, "
                "or set category_probs of the cached categories to 0."
            )
        c = load_fallen_cache(path, robot.joint_names, env.device, include_held=include_held)
        problem = check_cache_asset(c, robot)
        if problem is not None:
            if not allow_asset_mismatch:
                raise RuntimeError(f"[getup.reset_fallen_state] {problem}. Rebuild the cache for this asset or pass "
                                   "allow_asset_mismatch=True.")
            msg = f"[getup.reset_fallen_state] WARNING: {problem} (allow_asset_mismatch=True)"
            warnings.warn(msg)
            print("!" * 100 + "\n" + msg + "\n" + "!" * 100, flush=True)
        caches[key] = c
        counts = {GETUP_CATEGORIES[i]: c.count_list[i] for i in range(len(CACHED_CATEGORIES))}
        print(f"[getup.reset_fallen_state] loaded {c.num_states} cached states from {path} "
              f"(include_held={include_held}, urdf={c.meta.get('urdf')}): {counts}", flush=True)
    return caches[key]


# ---------------------------------------------------------------------------------------------------------------------
# Reset event
# ---------------------------------------------------------------------------------------------------------------------


def _probs_list(category_probs: dict[str, float]) -> list[float]:
    unknown = set(category_probs) - set(GETUP_CATEGORIES)
    if unknown:
        raise KeyError(f"unknown get-up categories {unknown}; valid: {GETUP_CATEGORIES}")
    return [max(float(category_probs.get(c, 0.0)), 0.0) for c in GETUP_CATEGORIES]


_PARTNER = [CATEGORY_INDEX[MIRROR_CATEGORY[c]] for c in GETUP_CATEGORIES]


def _uniform(lo: float, hi: float, shape, device) -> torch.Tensor:
    return lo + (hi - lo) * torch.rand(shape, device=device)


def reset_fallen_state(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    category_probs: dict[str, float],
    cache_path: str,
    assignment: str = "random",
    mirror_prob: float = 0.5,
    xy_range: float = 0.25,
    zero_joint_vel: bool = True,
    limp_time_range: tuple[float, float] = (0.04, 1.0),
    mid_fall_joint_noise: float = 0.1,
    mid_fall_tilt: float = 0.05,
    mid_fall_lin_vel_range: tuple[float, float] = (0.3, 1.2),
    mid_fall_ang_vel: float = 0.5,
    include_held: bool = True,
    allow_asset_mismatch: bool = False,
    asset_name: str = "robot",
) -> None:
    """Reset ``env_ids`` into a get-up start state (event term ``reset_fallen``, mode ``"reset"``).

    Per env a category is drawn from ``category_probs`` (keys = :data:`GETUP_CATEGORIES`; missing keys = 0; read on
    every call, so :func:`update_category_probs` and evaluation one-hots take effect at the next reset).

    * Cached categories (supine .. kneeling): a settled state from the cache at ``cache_path``, mirrored left/right
      with probability ``mirror_prob`` (a mirrored ``side_left`` state is a ``side_right`` state), random yaw, random
      xy offset in ``±xy_range``, zero velocity. Policy control from step 0 (no settle).
    * ``mid_fall``: default standing pose + U(±``mid_fall_joint_noise``) joint noise, U(±``mid_fall_tilt``) roll/pitch,
      random yaw, a push (horizontal speed U(``mid_fall_lin_vel_range``) in a random direction, angular velocity
      U(±``mid_fall_ang_vel``)). Actuators limp (Kp 0, Kd 0.5) for U(``limp_time_range``) s, then policy control.
    * ``standing``: default standing pose, random yaw, zero velocity, policy control from step 0.

    Every state is lifted so that no collision sphere penetrates the terrain at its xy; ``standing`` and ``mid_fall``
    are also lowered so the lowest collision point touches the terrain.

    Writes ``env.getup_state.category`` (index into :data:`GETUP_CATEGORIES`), ``control_start_step`` (env steps after
    the reset at which the policy takes control; 0 = immediately) and ``policy_active`` (= control_start_step == 0).
    The action term calls :func:`~.limp.update_limp_schedule` every step to release the limp phase.

    The whole path is free of GPU->host syncs (category logic on host floats, masks + ``torch.where``).

    Args:
        assignment: ``"random"`` (i.i.d. draws from ``category_probs``) or ``"round_robin"`` (env i gets the
            (i mod K)-th category with non-zero probability; deterministic equal split for evaluation).
        include_held: also sample posture-held cache states (see :func:`load_fallen_cache`). With False the
            kneeling category is empty in the v1 cache (Asimov cannot kneel limp) and its probability is dropped.
        allow_asset_mismatch: load a cache built on a different spawn URDF (warns loudly) instead of raising.
    """
    robot: Articulation = env.scene[asset_name]
    dev = env.device
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=dev)
    elif not isinstance(env_ids, torch.Tensor):
        env_ids = torch.tensor(env_ids, device=dev, dtype=torch.long)
    n = env_ids.shape[0]
    if n == 0:
        return
    st = get_state(env)
    num_cached = len(CACHED_CATEGORIES)

    # ---- category draw (host-side probability logic) -----------------------------------------------------------
    probs = _probs_list(category_probs)
    cache = None
    if sum(probs[:num_cached]) > 0:
        cache = _get_cache(env, robot, cache_path, include_held, allow_asset_mismatch)
        cnt = cache.count_list
        dropped = []
        for i in range(num_cached):
            avail = cnt[i] > 0 or (mirror_prob > 0.0 and cnt[_PARTNER[i]] > 0)
            if probs[i] > 0 and not avail:
                dropped.append(GETUP_CATEGORIES[i])
                probs[i] = 0.0
        if dropped:
            key = ("_getup_warned_empty", tuple(dropped))
            if not env.__dict__.get(key):
                warnings.warn(f"[getup.reset_fallen_state] no cached states for {dropped}; their probability is dropped")
                env.__dict__[key] = True
    total = sum(probs)
    if total <= 0:
        raise ValueError(f"category_probs has no available category: {category_probs}")
    if assignment == "random":
        cat = torch.multinomial(torch.tensor(probs, device=dev) / total, n, replacement=True)
    elif assignment == "round_robin":
        active = torch.tensor([i for i, p in enumerate(probs) if p > 0], device=dev)
        cat = active[env_ids % active.shape[0]]
    else:
        raise ValueError(f"unknown assignment '{assignment}'")
    is_cached = cat < num_cached
    is_mid = cat == MID_FALL
    snap = is_mid | (cat == STANDING)

    # ---- defaults (standing) ------------------------------------------------------------------------------------
    num_j = robot.num_joints
    joint_pos = robot.data.default_joint_pos[env_ids].clone()
    joint_vel = torch.zeros(n, num_j, device=dev)
    root_quat = torch.zeros(n, 4, device=dev)
    root_quat[:, 0] = 1.0
    root_height = robot.data.default_root_state[env_ids, 2].clone()

    # ---- cached categories (computed for every env, selected by mask) --------------------------------------------
    if cache is not None:
        partner = torch.tensor(_PARTNER, device=dev)
        c = cat.clamp(max=num_cached - 1)
        pc = partner[c]
        m = torch.rand(n, device=dev) < mirror_prob
        m = m & (cache.count[pc] > 0)
        m = m | (cache.count[c] == 0)
        src = torch.where(m, pc, c)
        cnt_src = cache.count[src].clamp(min=1)
        idx = cache.start[src] + (torch.rand(n, device=dev) * cnt_src).long().clamp(max=cnt_src - 1)
        idx = idx.clamp(max=cache.num_states - 1)
        perm = env.__dict__.get("_getup_mirror_perm")
        if perm is None:
            perm = torch.tensor(joint_mirror_perm(robot.joint_names), device=dev)
            env.__dict__["_getup_mirror_perm"] = perm
        q = cache.joint_pos[idx]
        q = torch.where(m.unsqueeze(1), -q[:, perm], q)
        quat = cache.root_quat[idx]
        quat = torch.where(m.unsqueeze(1), mirror_root_quat(quat), quat)
        sel = is_cached.unsqueeze(1)
        joint_pos = torch.where(sel, q, joint_pos)
        root_quat = torch.where(sel, quat, root_quat)
        root_height = torch.where(is_cached, cache.root_height[idx], root_height)
        if not zero_joint_vel:
            jv = cache.joint_vel[idx]
            jv = torch.where(m.unsqueeze(1), -jv[:, perm], jv)
            joint_vel = torch.where(sel, jv, joint_vel)

    # ---- mid_fall (computed for every env, selected by mask) ----------------------------------------------------
    mid = is_mid.unsqueeze(1)
    lim = robot.data.soft_joint_pos_limits[env_ids]
    q_mid = joint_pos + _uniform(-mid_fall_joint_noise, mid_fall_joint_noise, (n, num_j), dev)
    q_mid = torch.maximum(torch.minimum(q_mid, lim[..., 1]), lim[..., 0])
    joint_pos = torch.where(mid, q_mid, joint_pos)
    rp = _uniform(-mid_fall_tilt, mid_fall_tilt, (n, 2), dev)
    quat_mid = math_utils.quat_from_euler_xyz(rp[:, 0], rp[:, 1], torch.zeros(n, device=dev))
    root_quat = torch.where(mid, quat_mid, root_quat)
    heading = 2 * math.pi * torch.rand(n, device=dev)
    speed = _uniform(mid_fall_lin_vel_range[0], mid_fall_lin_vel_range[1], n, dev)
    root_vel = torch.zeros(n, 6, device=dev)
    root_vel[:, 0] = speed * torch.cos(heading)
    root_vel[:, 1] = speed * torch.sin(heading)
    root_vel[:, 3:6] = _uniform(-mid_fall_ang_vel, mid_fall_ang_vel, (n, 3), dev)
    root_vel = root_vel * mid.float()
    t_limp = _uniform(limp_time_range[0], limp_time_range[1], n, dev)
    limp_steps = torch.round(t_limp / env.step_dt).long().clamp(min=1)
    control_start = torch.where(is_mid, limp_steps, torch.zeros_like(limp_steps))

    # ---- yaw + xy randomization, terrain height, write ----------------------------------------------------------
    yaw = _uniform(-math.pi, math.pi, n, dev)
    zeros = torch.zeros(n, device=dev)
    root_quat = math_utils.quat_mul(math_utils.quat_from_euler_xyz(zeros, zeros, yaw), root_quat)
    terrain = _terrain_height(env)
    pos = env.scene.env_origins[env_ids].clone()
    pos[:, :2] += _uniform(-xy_range, xy_range, (n, 2), dev)
    pos[:, 2] = terrain(pos[:, :2]) + root_height

    robot.write_root_pose_to_sim(torch.cat([pos, root_quat], dim=-1), env_ids=env_ids)
    robot.write_root_velocity_to_sim(root_vel, env_ids=env_ids)
    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    robot.set_joint_position_target(joint_pos, env_ids=env_ids)
    robot.set_joint_velocity_target(torch.zeros_like(joint_vel), env_ids=env_ids)

    # lift out of any terrain penetration (FK is refreshed lazily after the writes above) and snap standing /
    # mid_fall starts down onto the ground (the default root height leaves the feet ~2.5 cm up)
    pen = ground_penetration(robot, env_ids, get_collision_spheres(robot), terrain)
    pos[:, 2] += torch.where(snap, pen, pen.clamp(min=0.0))
    robot.write_root_pose_to_sim(torch.cat([pos, root_quat], dim=-1), env_ids=env_ids)

    # ---- limp + shared state ------------------------------------------------------------------------------------
    set_limp_mask(env, env_ids, is_mid, asset_name=asset_name)
    st.category[env_ids] = cat
    st.control_start_step[env_ids] = control_start
    st.policy_active[env_ids] = control_start == 0


# ---------------------------------------------------------------------------------------------------------------------
# Adaptive category reweighting
# ---------------------------------------------------------------------------------------------------------------------


def reweight_category_probs(
    current: dict[str, float],
    success_by_category: dict[str, float] | torch.Tensor,
    floor: float = 0.03,
    offset: float = 0.1,
) -> dict[str, float]:
    """Pure function behind :func:`update_category_probs`.

    p_c ∝ (1 - success_c) + ``offset`` for every category with current probability > 0 (disabled categories stay 0),
    then a ``floor`` on every enabled category and renormalization (exact: floored categories are pinned at ``floor``
    and the rest share the remaining mass proportionally). A category without a success estimate (missing or NaN) is
    treated as success 0.
    """
    if isinstance(success_by_category, torch.Tensor):
        vals = success_by_category.detach().float().cpu().tolist()
        success = {c: vals[i] for i, c in enumerate(GETUP_CATEGORIES) if i < len(vals)}
    else:
        success = dict(success_by_category)
    enabled = [c for c in GETUP_CATEGORIES if float(current.get(c, 0.0)) > 0.0]
    if not enabled:
        return {c: float(current.get(c, 0.0)) for c in GETUP_CATEGORIES}
    w = {}
    for c in enabled:
        s = success.get(c, float("nan"))
        s = 0.0 if s is None or math.isnan(float(s)) else min(max(float(s), 0.0), 1.0)
        w[c] = (1.0 - s) + offset
    floor = min(floor, 1.0 / len(enabled))
    pinned: set[str] = set()
    while True:
        free = [c for c in enabled if c not in pinned]
        mass = 1.0 - floor * len(pinned)
        tot = sum(w[c] for c in free)
        p = {c: mass * w[c] / tot for c in free}
        low = [c for c in free if p[c] < floor]
        if not low:
            break
        pinned.update(low)
    out = {c: 0.0 for c in GETUP_CATEGORIES}
    out.update(p)
    for c in pinned:
        out[c] = floor
    return out


def update_category_probs(
    env: ManagerBasedEnv,
    success_by_category: dict[str, float] | torch.Tensor,
    floor: float = 0.03,
    offset: float = 0.1,
    term_name: str = "reset_fallen",
) -> dict[str, float]:
    """Adaptive reweighting hook (called by the ``category_reweighting`` curriculum every 200 iterations by default).

    Computes :func:`reweight_category_probs` from the current ``category_probs`` of the ``term_name`` event term and
    writes the result back into that term's params in place, so the next resets use it. Returns the new probabilities.

    Args:
        success_by_category: success rate per category, as a dict keyed by :data:`GETUP_CATEGORIES` names or a
            tensor in that order.
    """
    cfg = env.event_manager.get_term_cfg(term_name)
    current = cfg.params["category_probs"]
    new = reweight_category_probs(current, success_by_category, floor=floor, offset=offset)
    current.clear()
    current.update(new)
    return new
