import json
import logging
import math
from typing import Dict, Optional
from urllib import request as urllib_request
from urllib.error import URLError

from .base import ControllerState, RobotControllerBase
from .smoother import JointSmoother

logger = logging.getLogger(__name__)

_DEFAULT_JOINT_MAPPING = {f"j{i}": f"joint{i+1}" for i in range(7)}


class HTTPController(RobotControllerBase):
    def __init__(
        self,
        robot_id: str = "",
        robot_name: str = "",
        url: str = "http://192.168.1.65:5173/v1/viewer/joints",
        joint_mapping: Optional[Dict[str, str]] = None,
        timeout: float = 1.0,
        smoother_alpha: float = 0.3,
        smoother_threshold: float = 0.02,
        smoother_interval: float = 0.05,
        **kwargs,
    ):
        super().__init__(robot_id, robot_name)
        self._url = url
        self._timeout = timeout
        self._joint_mapping = {**_DEFAULT_JOINT_MAPPING, **(joint_mapping or {})}
        self._smoother = JointSmoother(
            alpha=smoother_alpha,
            threshold=smoother_threshold,
            min_interval=smoother_interval,
        )

    def connect(self) -> bool:
        logger.info("HTTP controller connected to %s", self._url)
        self._set_status(ControllerState.CONNECTED, f"http endpoint: {self._url}")
        self._smoother.reset()
        return True

    def disconnect(self):
        logger.info("HTTP controller disconnected")
        self._set_status(ControllerState.DISCONNECTED, "disconnected")
        self._smoother.reset()

    def send_joints(self, joints: Dict[str, float], gripper: float) -> bool:
        if not self.enabled:
            return False

        should_send, smoothed = self._smoother.filter(joints)
        if not should_send:
            return True

        mapped = {}
        for k, v in smoothed.items():
            mapped_key = self._joint_mapping.get(k, k)
            mapped[mapped_key] = round(math.radians(float(v)), 4)
        mapped["gripper"] = round(float(gripper), 4)

        try:
            data = json.dumps(mapped).encode("utf-8")
            req = urllib_request.Request(
                self._url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib_request.urlopen(req, timeout=self._timeout) as resp:
                resp.read()

            self._smoother.mark_sent(smoothed)
            logger.debug("Sent joints to %s: %s", self._url, mapped)
            return True
        except URLError as e:
            logger.warning("HTTP request failed: %s", e)
            self._set_status(ControllerState.ERROR, str(e))
            return False
        except Exception as e:
            logger.warning("HTTP request error: %s", e)
            self._set_status(ControllerState.ERROR, str(e))
            return False

    @property
    def url(self) -> str:
        return self._url
