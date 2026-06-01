import logging
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class ControllerState(Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


@dataclass
class ControllerStatus:
    state: ControllerState = ControllerState.DISCONNECTED
    message: str = ""
    robot_id: str = ""
    robot_name: str = ""

    def to_dict(self) -> dict:
        return {
            "state": self.state.value,
            "message": self.message,
            "robot_id": self.robot_id,
            "robot_name": self.robot_name,
        }


class RobotControllerBase(ABC):
    def __init__(self, robot_id: str, robot_name: str = ""):
        self._robot_id = robot_id
        self._robot_name = robot_name
        self._enabled = False
        self._lock = threading.Lock()
        self._status = ControllerStatus(
            robot_id=robot_id,
            robot_name=robot_name,
        )

    @property
    def robot_id(self) -> str:
        return self._robot_id

    @property
    def robot_name(self) -> str:
        return self._robot_name

    @property
    def enabled(self) -> bool:
        with self._lock:
            return self._enabled

    @enabled.setter
    def enabled(self, value: bool):
        with self._lock:
            self._enabled = value

    @property
    def status(self) -> ControllerStatus:
        with self._lock:
            return ControllerStatus(
                state=self._status.state,
                message=self._status.message,
                robot_id=self._status.robot_id,
                robot_name=self._status.robot_name,
            )

    def _set_status(self, state: ControllerState, message: str = ""):
        with self._lock:
            self._status.state = state
            self._status.message = message

    @abstractmethod
    def connect(self) -> bool:
        pass

    @abstractmethod
    def disconnect(self):
        pass

    @abstractmethod
    def send_joints(self, joints: Dict[str, float], gripper: float) -> bool:
        pass
