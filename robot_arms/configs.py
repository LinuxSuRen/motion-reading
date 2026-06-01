from dataclasses import dataclass, field
from typing import Dict, List


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
