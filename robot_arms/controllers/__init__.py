import logging
import os
from typing import Dict, Optional, Type

from .base import ControllerState, ControllerStatus, RobotControllerBase
from .dummy import DummyController
from .tcp import TCPController

logger = logging.getLogger(__name__)

_controller_registry: Dict[str, Type[RobotControllerBase]] = {
    "dummy": DummyController,
    "tcp": TCPController,
}


def register_controller(name: str, cls: Type[RobotControllerBase]):
    _controller_registry[name] = cls


def create_controller(
    controller_type: str = "dummy",
    robot_id: str = "",
    robot_name: str = "",
    **kwargs,
) -> RobotControllerBase:
    cls = _controller_registry.get(controller_type)
    if cls is None:
        logger.warning(
            "Unknown controller type '%s', falling back to dummy. Available: %s",
            controller_type,
            list(_controller_registry.keys()),
        )
        cls = DummyController

    if controller_type == "tcp":
        kwargs.setdefault("host", os.getenv("ROBOT_HOST", "127.0.0.1"))
        kwargs.setdefault("port", int(os.getenv("ROBOT_PORT", "30002")))

    return cls(robot_id=robot_id, robot_name=robot_name, **kwargs)


def get_available_controllers() -> list:
    return sorted(_controller_registry.keys())
