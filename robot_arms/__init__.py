from .base import ConfigurableArm
from .configs import ROBOT_CONFIGS, get_config as _get_config
from .nero7 import Nero7Axis
from .controllers import create_controller, get_available_controllers

DEFAULT_ROBOT = "nero7"


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
