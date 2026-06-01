import numpy as np
from .base import RobotArmBase, JointLimit, ArmVectors, RobotJointState


class Nero7Axis(RobotArmBase):
    JOINT_NAMES = ["j0", "j1", "j2", "j3", "j4", "j5", "j6"]
    JOINT_LIMITS = {
        "j0": JointLimit(-170, 170, 0),
        "j1": JointLimit(-140, 140, 0),
        "j2": JointLimit(-140, 140, 0),
        "j3": JointLimit(-130, 130, 0),
        "j4": JointLimit(-160, 160, 0),
        "j5": JointLimit(-150, 150, 0),
        "j6": JointLimit(-43, 58, 0),
    }

    @property
    def name(self) -> str:
        return "AgileX NERO 7-Axis"

    @property
    def joint_names(self):
        return self.JOINT_NAMES

    @property
    def joint_limits(self):
        return self.JOINT_LIMITS

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

        joints = {"j0": j0, "j1": j1, "j2": j2, "j3": j3, "j4": j4, "j5": j5, "j6": j6}
        for name, limit in self.JOINT_LIMITS.items():
            joints[name] = round(limit.clamp(joints[name]), 1)

        return RobotJointState(joints=joints, gripper=round(gripper, 2))

    def _solve_shoulder(self, v_ua: np.ndarray) -> tuple:
        elevation = np.degrees(np.arccos(np.clip(v_ua[1], -1.0, 1.0)))
        azimuth = np.degrees(np.arctan2(v_ua[0], v_ua[2]))

        j0 = azimuth
        j1 = max(0.0, elevation)
        j2 = 0.0

        return j0, j1, j2

    def _solve_elbow(
        self, v_ua: np.ndarray, v_la: np.ndarray, arm: dict
    ) -> float:
        elbow_angle = arm.get("elbow", None)
        if elbow_angle is not None:
            return elbow_angle

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

        wrist_angle = arm.get("wrist", None)
        if wrist_angle is not None:
            wrist_flex = wrist_angle - 180.0
            j5 = wrist_flex

        if hand_data:
            index_data = hand_data.get("index", {})
            pip_angle = index_data.get("pip_angle")
            if pip_angle is not None and pip_angle < 130:
                j5 += (130 - pip_angle) * 0.5

        if vectors.hand_normal is not None:
            ref_right = np.array([1, 0, 0]) if side == "right" else np.array([-1, 0, 0])
            roll_dot = np.dot(vectors.hand_normal[:2], ref_right[:2])
            j6 = np.degrees(np.arccos(np.clip(roll_dot, -1.0, 1.0))) - 90.0
            j6 = max(-90, min(90, j6))

        return j4, j5, j6
