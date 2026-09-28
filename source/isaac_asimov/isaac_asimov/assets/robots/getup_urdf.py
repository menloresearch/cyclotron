# Copyright (c) 2026, Menlo Research.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Patched Asimov 1 URDF for the get-up task.

The stock URDF (``third_party/asimov-1/sim-model/urdf/asimov_1.urdf``) models collisions with primitives only and
leaves two gaps that matter when the robot is lying on the ground and pushing itself up:

1. ``*_wrist_yaw_link`` (the wrist stub at the end of the forearm) has no collision at all.
2. ``*_shoulder_yaw_link`` (the upper arm) has no collision, so between the end of the shoulder-roll capsule and the
   elbow sphere there is an ~5 cm uncovered band (~11.5 cm between the two capsule centres) through which the upper
   arm can pass into the floor or the torso.

This module writes a patched copy of the URDF with these collisions added. It never touches ``third_party/``.

Geometry (measured from the stock meshes):

* **Wrist stub.** ``{LEFT,RIGHT}_WRIST_YAW.STL`` is a flat disc coaxial with the wrist-yaw axis
  ``a = (0.766, 0, -0.643)`` (link frame): it spans ``s = a.p`` in [-0.0040, 0.0150] m with a radial extent of
  0.0188 m. Its axis-aligned bounding box (38.1 x 37.6 x 40.4 mm) is the tilted disc, not a box-shaped part. We add a
  **cylinder** r = 0.0188 m, length = 0.0190 m, centred at ``0.0055 * a`` with its z-axis along ``a``. A cylinder is
  exact and invariant to the wrist-yaw rotation; an axis-aligned 38x38x40 mm box would overstate the stub by up to
  ~40 % radially at its corners.
* **Upper arm.** ``*_SHOULDER_YAW.STL`` spans z in [-0.130, +0.005] m of the link frame with |x|,|y| <= 0.035 m around
  the z axis (the elbow joint sits at z = -0.0966 / -0.0963 m). We add a **cylinder** r = 0.033 m from z = 0 to the
  elbow joint centre. It overlaps the shoulder-roll capsule end-sphere (r 0.035 at z = +0.019) and the elbow sphere
  (r 0.03 at the elbow joint), which closes the gap without changing any existing geometry.

* **Body shell (on by default).** The stock torso is one r = 0.06 m capsule while the shell is ~22 x 25 x 38 cm;
  shank, thigh, pelvis and head are similarly thin. :data:`SHELL_BOXES` adds boxes sized to the mesh profiles (see the
  table there). ``body_shell=False`` gives the v1 patch (wrist + upper arm only).

Adding collisions does not change mass or inertia: every link has an ``<inertial>`` tag, and the importer only
derives mass from geometry for links without one.

Mesh paths in the URDF are relative (``../assets/meshes/X.STL``). The patched file is written to
``<cache>/<key>/urdf/asimov_1_getup.urdf`` next to a symlink ``<cache>/<key>/assets -> <model_dir>/assets``, so the
relative paths still resolve. ``<key>`` hashes the source URDF bytes, the resolved model directory and the patch
version, so the file is regenerated whenever the source URDF (or this patch) changes and is reused otherwise. Writes are
atomic (temp file + ``os.replace``), so concurrent processes are safe.
"""

from __future__ import annotations

import hashlib
import math
import os
import xml.etree.ElementTree as ET
from pathlib import Path

PATCH_VERSION = "getup-v2"
"""Bump when the patch content changes so the cache key changes."""

GETUP_URDF_FILENAME = "asimov_1_getup.urdf"

DEFAULT_CACHE_ROOT = Path(os.environ.get("ASIMOV_GETUP_URDF_DIR", Path.home() / ".cache" / "isaac_asimov" / "asimov_1_getup"))

# Wrist-yaw axis in the wrist link frame (identical for both sides, from the URDF joint axis).
WRIST_AXIS = (0.7660444431, 0.0, -0.6427876097)
WRIST_STUB_RADIUS = 0.0188
WRIST_STUB_S_MIN = -0.0040
WRIST_STUB_S_MAX = 0.0150

UPPER_ARM_RADIUS = 0.033
UPPER_ARM_Z_TOP = 0.0
# the bottom end is the elbow joint centre, read from the URDF per side

# Shell boxes: (link, name, centre xyz, full size xyz) in the link frame, sized from z-slice profiles of
# the visual meshes (``validate_asset.py --profile``). Left-leg values; the right leg is mirrored in y. Every box is kept
# clear of the collision shapes of links that PhysX does NOT filter against it (non parent/child) at the default pose and
# over the joint ranges we checked (e.g. the thigh box stops at z = -0.04 to stay off the hip-pitch motor spheres, and
# the pelvis box stops at z = -0.085 above the thigh). Existing primitives are kept; the boxes are added to them.
SHELL_BOXES: list[tuple[str, str, tuple[float, float, float], tuple[float, float, float]]] = [
    # torso (waist_yaw_link): mesh x -0.117..0.102, y +-0.125, z 0..0.378; shoulders at y +-0.162 need clearance
    ("waist_yaw_link", "torso_abdomen", (0.0, 0.0, 0.075), (0.16, 0.19, 0.09)),  # z 0.03..0.12
    ("waist_yaw_link", "torso_chest", (-0.0035, 0.0, 0.195), (0.193, 0.24, 0.15)),  # z 0.12..0.27
    ("waist_yaw_link", "torso_upper_back", (-0.03, 0.0, 0.3075), (0.15, 0.24, 0.075)),  # z 0.27..0.345
    # head (neck_pitch_link, merged into waist_yaw_link by the importer): mesh x -0.072..0.097, y +-0.066, z -0.05..0.133
    ("neck_pitch_link", "head", (0.013, 0.0, 0.0425), (0.162, 0.126, 0.125)),  # z -0.02..0.105
    # pelvis (pelvis_link, mesh IMU_ORIGIN): x -0.124..0.008, y +-0.067, z -0.104..0.08
    ("pelvis_link", "pelvis", (-0.0575, 0.0, -0.005), (0.125, 0.12, 0.16)),  # z -0.085..0.075
    # thigh (hip_yaw_link): mesh x -0.056..0.067, y +-0.055, z -0.232..0.077 (knee joint at z -0.218)
    ("{side}_hip_yaw_link", "{side}_thigh", (0.006, 0.0, -0.095), (0.112, 0.10, 0.11)),  # z -0.15..-0.04
    # shank (knee_link): mesh x -0.067..0.062, y +-0.049, z -0.308..0.062 (ankle joint at z -0.272)
    ("{side}_knee_link", "{side}_shank", (-0.005, 0.0, -0.085), (0.114, 0.094, 0.23)),  # z -0.20..0.03
]

GEOM_TAG = "getup_patch"
"""Name prefix of every added ``<collision>`` element (used by the validation script)."""


def _fmt(x: float) -> str:
    return f"{x:.6g}"


def _add_cylinder(link: ET.Element, name: str, xyz, rpy, radius: float, length: float) -> None:
    col = ET.SubElement(link, "collision", {"name": name})
    ET.SubElement(col, "origin", {"xyz": " ".join(_fmt(v) for v in xyz), "rpy": " ".join(_fmt(v) for v in rpy)})
    geom = ET.SubElement(col, "geometry")
    ET.SubElement(geom, "cylinder", {"radius": _fmt(radius), "length": _fmt(length)})


def _add_box(link: ET.Element, name: str, xyz, size) -> None:
    col = ET.SubElement(link, "collision", {"name": name})
    ET.SubElement(col, "origin", {"xyz": " ".join(_fmt(v) for v in xyz), "rpy": "0 0 0"})
    geom = ET.SubElement(col, "geometry")
    ET.SubElement(geom, "box", {"size": " ".join(_fmt(v) for v in size)})


def _find(root: ET.Element, tag: str, name: str) -> ET.Element:
    el = root.find(f"{tag}[@name='{name}']")
    if el is None:
        raise ValueError(f"URDF has no <{tag} name='{name}'>")
    return el


def patch_urdf_tree(root: ET.Element, body_shell: bool = True) -> list[str]:
    """Add the get-up collisions to a parsed URDF ``<robot>`` element in place.

    Args:
        root: The ``<robot>`` element.
        body_shell: Also add the :data:`SHELL_BOXES` (torso, head, pelvis, thighs, shanks). Default True.

    Returns:
        Descriptions of the added geoms (for logging).
    """
    added = []
    ax = WRIST_AXIS
    # rotation about y taking the cylinder z-axis onto the wrist axis: (sin t, 0, cos t) = ax
    wrist_pitch = math.atan2(ax[0], ax[2])
    s_mid = 0.5 * (WRIST_STUB_S_MIN + WRIST_STUB_S_MAX)
    s_len = WRIST_STUB_S_MAX - WRIST_STUB_S_MIN
    for side in ("left", "right"):
        # -- wrist stub
        link = _find(root, "link", f"{side}_wrist_yaw_link")
        if link.find("collision") is not None:
            raise ValueError(f"{side}_wrist_yaw_link already has a collision; upstream URDF changed, review the patch")
        _add_cylinder(
            link,
            f"{GEOM_TAG}_{side}_wrist_stub",
            xyz=tuple(s_mid * a for a in ax),
            rpy=(0.0, wrist_pitch, 0.0),
            radius=WRIST_STUB_RADIUS,
            length=s_len,
        )
        added.append(f"{side}_wrist_yaw_link: cylinder r={WRIST_STUB_RADIUS} L={s_len:.4f} along wrist axis")
        # -- upper arm
        link = _find(root, "link", f"{side}_shoulder_yaw_link")
        if link.find("collision") is not None:
            raise ValueError(f"{side}_shoulder_yaw_link already has a collision; upstream URDF changed, review the patch")
        elbow = _find(root, "joint", f"{side}_elbow_joint")
        if elbow.find("parent").get("link") != f"{side}_shoulder_yaw_link":
            raise ValueError(f"{side}_elbow_joint parent is not {side}_shoulder_yaw_link")
        ex, ey, ez = (float(v) for v in elbow.find("origin").get("xyz").split())
        z_bot = ez
        length = UPPER_ARM_Z_TOP - z_bot
        _add_cylinder(
            link,
            f"{GEOM_TAG}_{side}_upper_arm",
            xyz=(ex, ey, 0.5 * (UPPER_ARM_Z_TOP + z_bot)),
            rpy=(0.0, 0.0, 0.0),
            radius=UPPER_ARM_RADIUS,
            length=length,
        )
        added.append(f"{side}_shoulder_yaw_link: cylinder r={UPPER_ARM_RADIUS} z=[{z_bot:.4f}, {UPPER_ARM_Z_TOP}]")
    if body_shell:
        for link_t, name_t, xyz, size in SHELL_BOXES:
            sides = ("left", "right") if "{side}" in link_t else (None,)
            for side in sides:
                ys = -1.0 if side == "right" else 1.0
                link_name = link_t.format(side=side)
                c = (xyz[0], ys * xyz[1] + 0.0, xyz[2])
                _add_box(_find(root, "link", link_name), f"{GEOM_TAG}_{name_t.format(side=side)}", c, size)
                added.append(f"{link_name}: box size={size} at {c}")
    return added


def _cache_key(src: Path, src_bytes: bytes, body_shell: bool) -> str:
    h = hashlib.sha256()
    h.update(PATCH_VERSION.encode())
    h.update(b"shell" if body_shell else b"noshell")
    h.update(str(src.resolve().parent.parent).encode())
    h.update(src_bytes)
    return h.hexdigest()[:16]


def build_getup_urdf(src: str, out_root: str | os.PathLike | None = None, body_shell: bool = True) -> str:
    """Return the path of the patched get-up URDF, generating it if needed.

    Args:
        src: Path of the stock ``asimov_1.urdf`` (``<model_dir>/urdf/asimov_1.urdf``); meshes are expected at
            ``<model_dir>/assets``.
        out_root: Cache root. Defaults to ``$ASIMOV_GETUP_URDF_DIR`` or ``~/.cache/isaac_asimov/asimov_1_getup``.
        body_shell: Add the torso/head/pelvis/thigh/shank shell boxes. Default True.

    Returns:
        Absolute path of the patched URDF.
    """
    src_path = Path(src).expanduser()
    src_bytes = src_path.read_bytes()
    root_dir = Path(out_root).expanduser() if out_root is not None else DEFAULT_CACHE_ROOT
    out_dir = root_dir / _cache_key(src_path, src_bytes, body_shell)
    out_urdf = out_dir / "urdf" / GETUP_URDF_FILENAME
    if out_urdf.is_file():
        return str(out_urdf)

    (out_dir / "urdf").mkdir(parents=True, exist_ok=True)
    # keep "../assets/meshes/X.STL" resolvable
    assets_link = out_dir / "assets"
    if not assets_link.exists() and not assets_link.is_symlink():
        try:
            assets_link.symlink_to(src_path.resolve().parent.parent / "assets", target_is_directory=True)
        except FileExistsError:
            pass  # another process won the race

    root = ET.fromstring(src_bytes)
    patch_urdf_tree(root, body_shell=body_shell)
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tmp = out_urdf.with_name(f".{GETUP_URDF_FILENAME}.{os.getpid()}.tmp")
    tree.write(tmp, encoding="utf-8", xml_declaration=True)
    os.replace(tmp, out_urdf)
    return str(out_urdf)
