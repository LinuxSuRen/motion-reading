import logging
import time
from datetime import datetime
from typing import Dict

from .base import ControllerState, RobotControllerBase

logger = logging.getLogger(__name__)


class DummyController(RobotControllerBase):
    def __init__(self, robot_id: str = "dummy", robot_name: str = "Dummy"):
        super().__init__(robot_id, robot_name)
        self._last_command: Dict[str, float] = {}
        self._last_gripper: float = 0.0
        self._last_time: str = ""

    def connect(self) -> bool:
        logger.info("Dummy controller connected (simulation)")
        self._set_status(ControllerState.CONNECTED, "simulation mode")
        self._last_time = datetime.now().isoformat()
        return True

    def disconnect(self):
        logger.info("Dummy controller disconnected")
        self._set_status(ControllerState.DISCONNECTED, "disconnected")

    def send_joints(self, joints: Dict[str, float], gripper: float) -> bool:
        if not self.enabled:
            return False
        self._last_command = dict(joints)
        self._last_gripper = gripper
        self._last_time = datetime.now().isoformat()
        return True

    @property
    def last_command(self) -> dict:
        return {
            "joints": self._last_command,
            "gripper": self._last_gripper,
            "time": self._last_time,
        }
