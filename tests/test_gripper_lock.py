from pathlib import Path
import xml.etree.ElementTree as ET

from cyclotron.gripper_lock import write_gripper_locked_urdf


def test_lock_grippers_preserves_controlled_joints_mass_geometry_and_source(tmp_path):
    source = tmp_path / "model" / "urdf" / "robot.urdf"
    source.parent.mkdir(parents=True)
    source.write_text('''<robot name="test">
      <link name="hand"><inertial><mass value="0.2"/></inertial>
        <visual><geometry><mesh filename="../meshes/hand.stl"/></geometry></visual></link>
      <joint name="right_6DOF_joint" type="revolute"><limit lower="-1.5" upper="1.5"/></joint>
      <joint name="right_gripper_driver_joint" type="revolute">
        <parent link="hand"/><child link="finger"/><origin xyz="0 0 1"/>
        <axis xyz="1 0 0"/><limit lower="0" upper="1"/><dynamics damping="0.1"/>
      </joint>
      <joint name="right_gripper_finger_joint" type="revolute">
        <mimic joint="right_gripper_driver_joint" multiplier="-1"/>
      </joint></robot>''')
    original = source.read_bytes()
    destination = tmp_path / "derived.urdf"
    locked = write_gripper_locked_urdf(source, destination)
    assert locked == ["right_gripper_driver_joint", "right_gripper_finger_joint"]
    assert source.read_bytes() == original
    root = ET.parse(destination).getroot()
    joints = {j.get("name"): j for j in root.findall("joint")}
    assert joints["right_6DOF_joint"].get("type") == "revolute"
    assert joints["right_6DOF_joint"].find("limit").get("upper") == "1.5"
    for name in locked:
        assert joints[name].get("type") == "fixed"
        assert all(joints[name].find(tag) is None for tag in ("axis", "limit", "mimic", "dynamics"))
    assert joints[locked[0]].find("origin").get("xyz") == "0 0 1"
    assert root.find(".//mass").get("value") == "0.2"
    assert Path(root.find(".//mesh").get("filename")) == source.parent.parent / "meshes/hand.stl"
