import json
import logging
import socket
import time
from typing import Dict, Optional

from .base import ControllerState, RobotControllerBase

logger = logging.getLogger(__name__)


class TCPController(RobotControllerBase):
    def __init__(
        self,
        robot_id: str,
        robot_name: str = "",
        host: str = "127.0.0.1",
        port: int = 30002,
        timeout: float = 2.0,
        auto_reconnect: bool = True,
    ):
        super().__init__(robot_id, robot_name)
        self._host = host
        self._port = port
        self._timeout = timeout
        self._auto_reconnect = auto_reconnect
        self._socket: Optional[socket.socket] = None
        self._reconnect_interval = 3.0
        self._last_attempt = 0.0

    def connect(self) -> bool:
        self._set_status(ControllerState.CONNECTING, f"connecting to {self._host}:{self._port}")
        logger.info("TCP controller connecting to %s:%d", self._host, self._port)

        try:
            if self._socket:
                self._try_close_socket()

            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._socket.settimeout(self._timeout)
            self._socket.connect((self._host, self._port))
            self._socket.settimeout(None)
            self._set_status(ControllerState.CONNECTED, f"{self._host}:{self._port}")
            logger.info("TCP controller connected to %s:%d", self._host, self._port)
            return True
        except (socket.timeout, ConnectionRefusedError, OSError) as e:
            logger.warning("TCP controller failed to connect: %s", e)
            self._set_status(ControllerState.ERROR, str(e))
            self._try_close_socket()
            self._last_attempt = time.time()
            return False

    def disconnect(self):
        logger.info("TCP controller disconnecting from %s:%d", self._host, self._port)
        self._try_close_socket()
        self._set_status(ControllerState.DISCONNECTED, "disconnected")

    def send_joints(self, joints: Dict[str, float], gripper: float) -> bool:
        if not self.enabled or not self._socket:
            if self._auto_reconnect and (
                not self._socket or time.time() - self._last_attempt > self._reconnect_interval
            ):
                self.connect()
            return False

        try:
            message = json.dumps({
                "joints": {k: round(v, 3) for k, v in joints.items()},
                "gripper": round(gripper, 3),
            })
            raw = (message + "\n").encode("utf-8")
            self._socket.sendall(raw)
            return True
        except (BrokenPipeError, ConnectionResetError, OSError) as e:
            logger.warning("TCP send failed: %s", e)
            self._set_status(ControllerState.ERROR, str(e))
            self._try_close_socket()
            self._last_attempt = time.time()
            return False

    @property
    def host(self) -> str:
        return self._host

    @property
    def port(self) -> int:
        return self._port

    def _try_close_socket(self):
        if self._socket:
            try:
                self._socket.close()
            except OSError:
                pass
            self._socket = None
