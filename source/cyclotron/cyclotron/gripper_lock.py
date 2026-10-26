"""Import a gripper-equipped URDF with its gripper joints locked, without editing the third-party file.

Plain Python except ``spawn_gripper_locked_urdf``, which imports Isaac Lab when called.
"""

from pathlib import Path
from tempfile import TemporaryDirectory
import xml.etree.ElementTree as ET


def write_gripper_locked_urdf(source: str | Path, destination: str | Path) -> list[str]:
    """Lock grippers at their zero pose, retaining all links and inertias.

    Mesh references become absolute because the derived file lives in a
    temporary directory. The source URDF and meshes remain unchanged.
    """
    source = Path(source).resolve()
    tree = ET.parse(source)
    locked = []
    for joint in tree.getroot().findall("joint"):
        if "_gripper_" not in joint.attrib["name"] or joint.attrib["type"] == "fixed":
            continue
        joint.set("type", "fixed")
        for tag in ("mimic", "limit", "axis", "dynamics"):
            for element in joint.findall(tag):
                joint.remove(element)
        locked.append(joint.attrib["name"])
    for mesh in tree.getroot().iter("mesh"):
        filename = mesh.attrib["filename"]
        if "://" not in filename:
            mesh.set("filename", str((source.parent / filename).resolve()))
    tree.write(destination, encoding="utf-8", xml_declaration=True)
    return locked


def spawn_gripper_locked_urdf(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """Use Isaac Lab's importer with gripper joints fixed before USD conversion."""
    import isaaclab.sim as sim_utils

    with TemporaryDirectory(prefix="asimov_locomotion_") as directory:
        derived = Path(directory) / Path(cfg.asset_path).name
        locked = write_gripper_locked_urdf(cfg.asset_path, derived)
        print(f"[INFO] Locomotion import: locked {len(locked)} gripper joints from {cfg.asset_path}")
        # Record the source path in the task config; only the importer's copy
        # points to the derived URDF. USD conversion finishes before cleanup.
        import_cfg = cfg.replace(asset_path=str(derived))
        return sim_utils.spawn_from_urdf(
            prim_path, import_cfg, translation=translation, orientation=orientation, **kwargs
        )
