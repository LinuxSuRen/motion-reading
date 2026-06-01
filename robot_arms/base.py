import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class JointLimit:
    min_deg: float
    max_deg: float
    default_deg: float = 0.0

    def clamp(self, angle_deg: float) -> float:
        return max(self.min_deg, min(self.max_deg, angle_deg))


@dataclass
class ArmVectors:
    upper_arm: np.ndarray
    lower_arm: np.ndarray
    hand_normal: Optional[np.ndarray] = None
    hand_direction: Optional[np.ndarray] = None


@dataclass
class RobotJointState:
    joints: Dict[str, float] = field(default_factory=dict)
    gripper: float = 0.0


class RobotArmBase(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def joint_names(self) -> List[str]:
        pass

    @property
    @abstractmethod
    def joint_limits(self) -> Dict[str, JointLimit]:
        pass

    @abstractmethod
    def map_human_to_robot(
        self,
        human_angles: dict,
        arm_vectors: ArmVectors,
        hand_data: Optional[dict],
        side: str,
    ) -> RobotJointState:
        pass

    def _compute_default_vectors(
        self,
        landmarks: list,
        shoulder_idx: int,
        elbow_idx: int,
        wrist_idx: int,
    ) -> ArmVectors:
        sh = np.array(landmarks[shoulder_idx])
        el = np.array(landmarks[elbow_idx])
        wr = np.array(landmarks[wrist_idx])

        upper = el - sh
        upper_norm = np.linalg.norm(upper)
        lower = wr - el
        lower_norm = np.linalg.norm(lower)

        upper_arm = upper / upper_norm if upper_norm > 1e-6 else np.array([0, 0, 1])
        lower_arm = lower / lower_norm if lower_norm > 1e-6 else np.array([0, 0, 1])

        return ArmVectors(upper_arm=upper_arm, lower_arm=lower_arm)

    def _compute_gripper(self, hand_data: Optional[dict]) -> float:
        if not hand_data:
            return 0.0
        extended = sum(
            1
            for f in ["thumb", "index", "middle", "ring", "pinky"]
            if hand_data.get(f, {}).get("extended", False)
        )
        return 1.0 - (extended / 5.0)

    @staticmethod
    def _map_range(
        value: float,
        in_min: float,
        in_max: float,
        out_min: float,
        out_max: float,
    ) -> float:
        ratio = (value - in_min) / (in_max - in_min + 1e-8)
        ratio = max(0.0, min(1.0, ratio))
        return out_min + ratio * (out_max - out_min)


class ConfigurableArm(RobotArmBase):
    def __init__(self, robot_id: str, config, joint_names: List[str]):
        self._id = robot_id
        self._config = config
        self._joint_names = joint_names
        self._limits: Dict[str, JointLimit] = {}
        for name, jc in config.joints.items():
            self._limits[name] = JointLimit(jc.min_deg, jc.max_deg, 0.0)

    @property
    def name(self) -> str:
        return self._config.display_name

    @property
    def robot_id(self) -> str:
        return self._id

    @property
    def joint_names(self) -> List[str]:
        return self._joint_names

    @property
    def joint_limits(self) -> Dict[str, JointLimit]:
        return self._limits

    @property
    def dof(self) -> int:
        return self._config.dof

    def map_human_to_robot(
        self,
        human_angles: dict,
        arm_vectors: ArmVectors,
        hand_data: dict | None,
        side: str,
    ) -> RobotJointState:
        v_ua = arm_vectors.upper_arm
        v_la = arm_vectors.lower_arm
        arm = human_angles or {}

        j0, j1, j2 = self._solve_shoulder(v_ua)
        j3 = self._solve_elbow(v_ua, v_la, arm)
        j4, j5, j6 = self._solve_wrist(v_la, arm_vectors, hand_data, arm, side)

        gripper = self._compute_gripper(hand_data)

        raw = {"j0": j0, "j1": j1, "j2": j2, "j3": j3, "j4": j4, "j5": j5, "j6": j6}
        joints = {}
        for name, limit in self._limits.items():
            val = raw.get(name, 0.0)
            joints[name] = round(limit.clamp(val), 1)

        return RobotJointState(joints=joints, gripper=round(gripper, 2))

    def _solve_shoulder(self, v_ua: np.ndarray) -> tuple:
        elevation = np.degrees(np.arccos(np.clip(v_ua[1], -1.0, 1.0)))
        azimuth = np.degrees(np.arctan2(v_ua[0], v_ua[2]))
        return azimuth, max(0.0, elevation), 0.0

    def _solve_elbow(self, v_ua: np.ndarray, v_la: np.ndarray, arm: dict) -> float:
        elbow = arm.get("elbow")
        if elbow is not None:
            return elbow
        dot = np.clip(np.dot(v_ua, v_la), -1.0, 1.0)
        return np.degrees(np.arccos(dot))

    def _solve_wrist(
        self,
        v_la: np.ndarray,
        vectors: ArmVectors,
        hand_data: dict | None,
        arm: dict,
        side: str,
    ) -> tuple:
        j4 = 0.0
        j5 = 0.0
        j6 = 0.0

        wrist_angle = arm.get("wrist")
        if wrist_angle is not None:
            j5 = wrist_angle - 180.0

        if hand_data:
            pip = hand_data.get("index", {}).get("pip_angle")
            if pip is not None and pip < 130:
                j5 += (130 - pip) * 0.5

        if vectors.hand_normal is not None:
            ref = np.array([1, 0, 0]) if side == "right" else np.array([-1, 0, 0])
            dot = np.dot(vectors.hand_normal[:2], ref[:2])
            j6 = np.degrees(np.arccos(np.clip(dot, -1.0, 1.0))) - 90.0
            j6 = max(-90, min(90, j6))

        return j4, j5, j6
