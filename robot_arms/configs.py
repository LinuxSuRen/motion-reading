import math
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class JointConfig:
    min_deg: float
    max_deg: float
    label: str = ""


@dataclass
class RobotConfig:
    manufacturer: str
    model: str
    dof: int
    joints: Dict[str, JointConfig]
    description: str = ""

    @property
    def display_name(self) -> str:
        return f"{self.manufacturer} {self.model}"

    @property
    def sort_key(self) -> str:
        return f"{self.manufacturer}_{self.model}"


ROBOT_CONFIGS: Dict[str, RobotConfig] = {
    "nero7": RobotConfig(
        manufacturer="AgileX",
        model="NERO",
        dof=7,
        description="7-axis collaborative arm, S-R-S configuration",
        joints={
            "j0": JointConfig(-170, 170, "Base"),
            "j1": JointConfig(-140, 140, "Shoulder Pitch"),
            "j2": JointConfig(-140, 140, "Shoulder Roll"),
            "j3": JointConfig(-130, 130, "Elbow"),
            "j4": JointConfig(-160, 160, "Forearm Roll"),
            "j5": JointConfig(-150, 150, "Wrist Pitch"),
            "j6": JointConfig(-43, 58, "Wrist Roll"),
        },
    ),
    "franka_panda": RobotConfig(
        manufacturer="Franka Emika",
        model="Panda",
        dof=7,
        description="7-axis torque-controlled research arm",
        joints={
            "j0": JointConfig(-166, 166, "Base"),
            "j1": JointConfig(-101, 101, "Shoulder Pitch"),
            "j2": JointConfig(-166, 166, "Shoulder Roll"),
            "j3": JointConfig(-176, -4, "Elbow"),
            "j4": JointConfig(-166, 166, "Forearm Roll"),
            "j5": JointConfig(-1, 215, "Wrist Pitch"),
            "j6": JointConfig(-166, 166, "Wrist Roll"),
        },
    ),
    "kuka_iiwa7": RobotConfig(
        manufacturer="KUKA",
        model="LBR iiwa 7",
        dof=7,
        description="7-axis lightweight robot, ±170° all joints",
        joints={
            "j0": JointConfig(-170, 170, "Base"),
            "j1": JointConfig(-120, 120, "Shoulder Pitch"),
            "j2": JointConfig(-170, 170, "Shoulder Roll"),
            "j3": JointConfig(-120, 120, "Elbow"),
            "j4": JointConfig(-170, 170, "Forearm Roll"),
            "j5": JointConfig(-120, 120, "Wrist Pitch"),
            "j6": JointConfig(-175, 175, "Wrist Roll"),
        },
    ),
    "kinova_gen3": RobotConfig(
        manufacturer="Kinova",
        model="Gen3 7-DOF",
        dof=7,
        description="7-axis ultra-lightweight arm for research",
        joints={
            "j0": JointConfig(-180, 180, "Base"),
            "j1": JointConfig(-128.9, 128.9, "Shoulder Pitch"),
            "j2": JointConfig(-180, 180, "Shoulder Roll"),
            "j3": JointConfig(-147.8, 147.8, "Elbow"),
            "j4": JointConfig(-180, 180, "Forearm Roll"),
            "j5": JointConfig(-115, 115, "Wrist Pitch"),
            "j6": JointConfig(-180, 180, "Wrist Roll"),
        },
    ),
    "ur5e": RobotConfig(
        manufacturer="Universal Robots",
        model="UR5e",
        dof=6,
        description="6-axis collaborative industrial arm",
        joints={
            "j0": JointConfig(-360, 360, "Base"),
            "j1": JointConfig(-360, 360, "Shoulder"),
            "j2": JointConfig(-360, 360, "Elbow"),
            "j3": JointConfig(-360, 360, "Wrist 1"),
            "j4": JointConfig(-360, 360, "Wrist 2"),
            "j5": JointConfig(-360, 360, "Wrist 3"),
        },
    ),
}


def list_robot_ids() -> List[str]:
    return sorted(ROBOT_CONFIGS.keys(), key=lambda k: ROBOT_CONFIGS[k].sort_key)


def get_config(robot_id: str) -> RobotConfig:
    cfg = ROBOT_CONFIGS.get(robot_id)
    if cfg is None:
        available = ", ".join(ROBOT_CONFIGS.keys())
        raise ValueError(f"Unknown robot '{robot_id}'. Available: {available}")
    return cfg


def load_robot_from_urdf(urdf_path: str, robot_id: str = "urdf_robot") -> RobotConfig:
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    name = root.get("name", robot_id)

    joints: Dict[str, JointConfig] = {}
    dof = 0

    for joint in root.findall("joint"):
        jtype = joint.get("type", "")
        if jtype == "fixed":
            continue
        jname = joint.get("name", "")
        limit = joint.find("limit")
        if limit is None:
            continue
        lower = float(limit.get("lower", 0)) if limit.get("lower") is not None else 0
        upper = float(limit.get("upper", 0)) if limit.get("upper") is not None else 0
        jkey = f"j{dof}"
        joints[jkey] = JointConfig(
            min_deg=round(math.degrees(lower), 1),
            max_deg=round(math.degrees(upper), 1),
            label=jname,
        )
        dof += 1

    cfg = RobotConfig(
        manufacturer="URDF",
        model=name,
        dof=dof,
        joints=joints,
        description=f"Loaded from {os.path.basename(urdf_path)}",
    )
    ROBOT_CONFIGS[robot_id] = cfg
    return cfg


def load_urdf_kinematics(urdf_path: str) -> List[dict]:
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    links = {}
    for link in root.findall("link"):
        links[link.get("name")] = link

    chain = []
    for joint in root.findall("joint"):
        jtype = joint.get("type", "")
        if jtype == "fixed":
            continue
        jname = joint.get("name", "")
        parent_el = joint.find("parent")
        child_el = joint.find("child")
        parent = parent_el.get("link", "") if parent_el is not None else ""
        child = child_el.get("link", "") if child_el is not None else ""
        origin = joint.find("origin")
        xyz = [0.0, 0.0, 0.0]
        rpy = [0.0, 0.0, 0.0]
        if origin is not None:
            if origin.get("xyz"):
                xyz = [float(v) for v in origin.get("xyz").split()]
            if origin.get("rpy"):
                rpy = [float(v) for v in origin.get("rpy").split()]
        axis = joint.find("axis")
        ax = [0.0, 0.0, 1.0]
        if axis is not None and axis.get("xyz"):
            ax = [float(v) for v in axis.get("xyz").split()]
        limit = joint.find("limit")
        lower = 0.0
        upper = 0.0
        if limit is not None:
            lower = float(limit.get("lower", 0)) if limit.get("lower") else 0
            upper = float(limit.get("upper", 0)) if limit.get("upper") else 0

        chain.append({
            "name": jname,
            "parent": parent,
            "child": child,
            "origin_xyz": xyz,
            "origin_rpy": rpy,
            "axis_xyz": ax,
            "limit_lower": lower,
            "limit_upper": upper,
        })

    return chain
