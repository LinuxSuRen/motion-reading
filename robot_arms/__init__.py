import os

from .base import ConfigurableArm
from .configs import ROBOT_CONFIGS, get_config as _get_config, load_robot_from_urdf
from .nero7 import Nero7Axis
from .controllers import create_controller, get_available_controllers

DEFAULT_ROBOT = "nero7"
_URDF_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "nero", "urdf", "nero_description.urdf")

if os.path.exists(_URDF_PATH):
    load_robot_from_urdf(_URDF_PATH, robot_id="nero_urdf")
    DEFAULT_ROBOT = "nero_urdf"


def get_robot(robot_id: str | None = None):
    rid = robot_id or DEFAULT_ROBOT
    cfg = _get_config(rid)
    joint_names = list(cfg.joints.keys())
    return ConfigurableArm(rid, cfg, joint_names)


def get_available_robots():
    return [
        {
            "id": rid,
            "manufacturer": cfg.manufacturer,
            "model": cfg.model,
            "dof": cfg.dof,
            "description": cfg.description,
        }
        for rid, cfg in sorted(
            ROBOT_CONFIGS.items(), key=lambda kv: kv[1].sort_key
        )
    ]
