# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Build a MuJoCo model from the shared ``asimov-1`` MJCF, patched in-memory (never touches ``third_party/``).

Patches applied, all driven by :mod:`constants`:

1. Per-joint ``armature`` override (Isaac's values by default; the MJCF's own baked-in values differ -- see
   :data:`constants.MJCF_ARMATURE_GROUPS`) and an explicit ``frictionloss`` (the MJCF ships ``frictionloss="0"`` and relies on the training
   sim's actuator model for friction, same as here).
2. A ``<motor>`` actuator per joint (the MJCF defines none -- "the training sim sets these in Python" per its own
   comment). ``ctrl`` on these is the already-clipped torque this package's own PD computes each physics step;
   ``ctrllimited="false"`` because clipping happens in Python (:mod:`actuator`), not in MuJoCo.
3. Wrist-stub and upper-arm collision cylinders on ``*_wrist_yaw_link`` / ``*_shoulder_yaw_link``, with the exact
   dimensions added to the get-up URDF (``assets/robots/getup_urdf.py``), so both simulators see the same collision
   geometry. Expressed with MuJoCo's ``fromto`` (two endpoints along the relevant axis), which needs no quaternion
   math and is exact for a coaxial cylinder.
"""

from __future__ import annotations

import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from . import constants as C
from .constants import JointArrays


def _find_body(root: ET.Element, name: str) -> ET.Element:
    el = root.find(f".//body[@name='{name}']")
    if el is None:
        raise ValueError(f"MJCF has no <body name='{name}'>")
    return el


def _find_joint(root: ET.Element, name: str) -> ET.Element:
    el = root.find(f".//joint[@name='{name}']")
    if el is None:
        raise ValueError(f"MJCF has no <joint name='{name}'>")
    return el


def _fmt(*vals: float) -> str:
    return " ".join(f"{v:.6g}" for v in vals)


def add_wrist_and_upper_arm_collisions(root: ET.Element, add_wrist: bool = True, add_upper_arm: bool = True) -> list[str]:
    """Add the wrist-stub and/or upper-arm collision cylinders (the get-up URDF's exact dimensions) in place. Returns log lines."""
    added = []
    ax = np.asarray(C.WRIST_AXIS, dtype=float)
    ax = ax / np.linalg.norm(ax)
    p0 = C.WRIST_STUB_S_MIN * ax
    p1 = C.WRIST_STUB_S_MAX * ax
    for side in ("left", "right"):
        if add_wrist:
            wrist_body = _find_body(root, f"{side}_wrist_yaw_link")
            if wrist_body.find("./geom[@class='collision']") is not None:
                raise ValueError(f"{side}_wrist_yaw_link already has a collision geom; MJCF changed, review this patch")
            ET.SubElement(
                wrist_body, "geom",
                {
                    "name": f"{side}_wrist_stub_collision",
                    "class": "collision",
                    "type": "cylinder",
                    "fromto": _fmt(*p0, *p1),
                    "size": f"{C.WRIST_STUB_RADIUS:.6g}",
                },
            )
            added.append(f"{side}_wrist_yaw_link: cylinder r={C.WRIST_STUB_RADIUS} fromto={p0}->{p1}")

        if add_upper_arm:
            shoulder_body = _find_body(root, f"{side}_shoulder_yaw_link")
            if shoulder_body.find("./geom[@class='collision']") is not None:
                raise ValueError(f"{side}_shoulder_yaw_link already has a collision geom; MJCF changed, review this patch")
            elbow_body = _find_body(root, f"{side}_elbow_link")
            ex, ey, ez = (float(v) for v in elbow_body.get("pos").split())
            ET.SubElement(
                shoulder_body, "geom",
                {
                    "name": f"{side}_upper_arm_collision",
                    "class": "collision",
                    "type": "cylinder",
                    "fromto": _fmt(ex, ey, C.UPPER_ARM_Z_TOP, ex, ey, ez),
                    "size": f"{C.UPPER_ARM_RADIUS:.6g}",
                },
            )
            added.append(f"{side}_shoulder_yaw_link: cylinder r={C.UPPER_ARM_RADIUS} z=[{ez:.4f}, {C.UPPER_ARM_Z_TOP}]")
    return added


def add_shell_boxes(root: ET.Element) -> list[str]:
    """Add the get-up URDF's body-shell collision boxes (torso, head, pelvis, thigh, shank), on top of each
    body's existing collision primitives (additive, matching ``getup_urdf.py::SHELL_BOXES`` -- see the exact
    ``(link, name, center, full_size)`` tuples in :data:`constants.SHELL_BOXES`)."""
    added = []
    for link_t, name_t, xyz, full_size in C.SHELL_BOXES:
        sides = ("left", "right") if "{side}" in link_t else (None,)
        for side in sides:
            ys = -1.0 if side == "right" else 1.0
            link_name = link_t.format(side=side) if side else link_t
            name = name_t.format(side=side) if side else name_t
            center = (xyz[0], ys * xyz[1], xyz[2])
            half_size = tuple(s / 2.0 for s in full_size)  # URDF box size is full; MuJoCo box size is half-extent
            body = _find_body(root, link_name)
            ET.SubElement(
                body, "geom",
                {"name": f"{name}_shell_collision", "class": "collision", "type": "box",
                 "pos": _fmt(*center), "size": _fmt(*half_size)},
            )
            added.append(f"{link_name}: box size={full_size} at {center}")
    return added


def override_joint_dynamics(root: ET.Element, joints: JointArrays) -> None:
    """Set ``armature``/``frictionloss`` on each of the 23 named joints (damping stays 0: PD is applied in Python)."""
    for i, name in enumerate(C.ASIMOV_1_JOINT_NAMES):
        el = _find_joint(root, name)
        el.set("armature", f"{joints.armature[i]:.8g}")
        el.set("frictionloss", f"{joints.friction[i]:.8g}")
        el.set("damping", "0")


def add_motor_actuators(root: ET.Element) -> None:
    """Add one unlimited ``<motor>`` per joint (this package clips torque itself, see :mod:`actuator`)."""
    actuator_el = root.find("actuator")
    if actuator_el is None:
        actuator_el = ET.SubElement(root, "actuator")
    for name in C.ASIMOV_1_JOINT_NAMES:
        ET.SubElement(actuator_el, "motor", {"name": f"act_{name}", "joint": name, "ctrllimited": "false"})


@dataclass
class RobotModel:
    model: "mujoco.MjModel"
    data: "mujoco.MjData"
    joints: "JointArrays"
    joint_qpos_adr: np.ndarray  # (23,) int, index into data.qpos
    joint_dof_adr: np.ndarray  # (23,) int, index into data.qvel / qfrc_applied
    actuator_id: np.ndarray  # (23,) int, index into data.ctrl, same order as ASIMOV_1_JOINT_NAMES
    free_joint_qpos_adr: int  # 0 if the floating base is qpos[0:7]
    free_joint_dof_adr: int  # 0 if qvel[0:6]
    foot_body_id: tuple[int, int]
    torso_body_id: int
    pelvis_body_id: int
    imu_site_id: int
    gyro_sensor_adr: int
    vel_sensor_adr: int
    quat_sensor_adr: int
    xml_patch_log: list[str]


def build_robot_model(
    mjcf_path: str | Path,
    armature_source: str = "isaac",
    add_wrist_stub: bool = True,
    add_upper_arm: bool = True,
    add_body_shell: bool = True,
) -> RobotModel:
    """Parse the shared MJCF, apply the patches above in memory, compile, and return a :class:`RobotModel`.

    ``add_body_shell`` (on by default): torso/head/pelvis/thigh/shank collision boxes,
    copied from ``getup_urdf.py::SHELL_BOXES`` (PATCH_VERSION "getup-v2") -- see :mod:`constants`.
    """
    mjcf_path = Path(mjcf_path).expanduser().resolve()
    xml_text = mjcf_path.read_text()
    root = ET.fromstring(xml_text)

    # resolve the (relative) compiler meshdir to an absolute path so we can compile from an arbitrary temp file
    compiler = root.find("compiler")
    meshdir = compiler.get("meshdir", ".")
    abs_meshdir = str((mjcf_path.parent / meshdir).resolve())
    compiler.set("meshdir", abs_meshdir)

    joints = C.build_joint_arrays(armature_source=armature_source)
    override_joint_dynamics(root, joints)
    add_motor_actuators(root)
    log: list[str] = []
    if add_wrist_stub or add_upper_arm:
        added = add_wrist_and_upper_arm_collisions(root, add_wrist=add_wrist_stub, add_upper_arm=add_upper_arm)
        log.extend(added)
    if add_body_shell:
        log.extend(add_shell_boxes(root))

    xml_out = ET.tostring(root, encoding="unicode")
    with tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False) as f:
        f.write(xml_out)
        tmp_path = f.name
    try:
        model = mujoco.MjModel.from_xml_path(tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)
    data = mujoco.MjData(model)

    joint_qpos_adr = np.array(
        [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in C.ASIMOV_1_JOINT_NAMES],
        dtype=int,
    )
    joint_dof_adr = np.array(
        [model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in C.ASIMOV_1_JOINT_NAMES],
        dtype=int,
    )
    actuator_id = np.array(
        [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"act_{n}") for n in C.ASIMOV_1_JOINT_NAMES],
        dtype=int,
    )
    free_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "floating_base")
    foot_ids = tuple(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b) for b in C.FEET_BODIES)
    torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, C.TORSO_BODY)
    pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, C.PELVIS_BODY)
    imu_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, C.IMU_SITE)

    def _sensor_adr(name: str) -> int:
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        if sid < 0:
            raise ValueError(f"MJCF has no <sensor name='{name}'>")
        return int(model.sensor_adr[sid])

    return RobotModel(
        model=model,
        data=data,
        joints=joints,
        joint_qpos_adr=joint_qpos_adr,
        joint_dof_adr=joint_dof_adr,
        actuator_id=actuator_id,
        free_joint_qpos_adr=int(model.jnt_qposadr[free_jid]),
        free_joint_dof_adr=int(model.jnt_dofadr[free_jid]),
        foot_body_id=foot_ids,
        torso_body_id=torso_id,
        pelvis_body_id=pelvis_id,
        imu_site_id=imu_site_id,
        gyro_sensor_adr=_sensor_adr("imu_ang_vel"),
        vel_sensor_adr=_sensor_adr("imu_lin_vel"),
        quat_sensor_adr=_sensor_adr("imu_quat"),
        xml_patch_log=log,
    )

