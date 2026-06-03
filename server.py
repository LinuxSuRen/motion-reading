import asyncio
import base64
import json
import logging
import os
import time
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from pose_engine import PoseEngine
from robot_arms.base import ArmVectors
from robot_arms import get_robot, get_available_robots
from robot_arms.configs import load_urdf_kinematics
from robot_arms.controllers import (
    ControllerState,
    create_controller,
    get_available_controllers,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _compute_robot_angles(angles: dict | None, robot) -> dict | None:
    if not angles:
        return None

    result = {}
    for side_key in ("left", "right"):
        arm_data = angles.get(f"{side_key}_arm", {})
        hand_data = angles.get(f"{side_key}_hand", {})
        vectors_data = angles.get(f"{side_key}_vectors", {})

        sh = vectors_data.get("shoulder", [0, 0, 0])
        el = vectors_data.get("elbow", [0, 0, 0])
        wr = vectors_data.get("wrist", [0, 0, 0])

        ua = np.array(el) - np.array(sh)
        la = np.array(wr) - np.array(el)
        ua_norm = np.linalg.norm(ua)
        la_norm = np.linalg.norm(la)

        arm_vectors = ArmVectors(
            upper_arm=ua / ua_norm if ua_norm > 1e-6 else np.array([0, 0, 1]),
            lower_arm=la / la_norm if la_norm > 1e-6 else np.array([0, 0, 1]),
        )

        robot_state = robot.map_human_to_robot(arm_data, arm_vectors, hand_data, side_key)
        result[side_key] = {
            "joints": robot_state.joints,
            "gripper": robot_state.gripper,
        }

    return result


def _safe_json(obj):
    if obj is None:
        return None
    if isinstance(obj, dict):
        return {k: _safe_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_safe_json(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        val = float(obj)
        return None if np.isnan(val) or np.isinf(val) else val
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return _safe_json(obj.tolist())
    return obj


def _init_controller(robot):
    controller_type = os.getenv("ROBOT_CONTROLLER", "dummy")
    logger.info(
        "Initializing controller type=%s robot=%s (%s)",
        controller_type, robot.robot_id, robot.name,
    )
    ctrl = create_controller(
        controller_type=controller_type,
        robot_id=robot.robot_id,
        robot_name=robot.name,
    )
    connected = ctrl.connect()
    if connected:
        logger.info("Controller connected: %s", ctrl.status.to_dict())
        ctrl.enabled = True
    else:
        logger.warning(
            "Controller NOT connected (type=%s, robot=%s). "
            "Joint data will NOT be sent to hardware.",
            controller_type, robot.robot_id,
        )
    return ctrl


@asynccontextmanager
async def lifespan(app: FastAPI):
    engine = PoseEngine(camera_id=int(os.getenv("CAMERA_ID", "0")))
    app.state.engine = engine
    robot = get_robot()
    app.state.robot = robot
    app.state.controller = _init_controller(robot)
    engine.start()
    yield
    app.state.controller.disconnect()
    engine.stop()


app = FastAPI(lifespan=lifespan)

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def root():
    return FileResponse("static/index.html")


@app.get("/api/robots")
async def list_robots():
    return JSONResponse(get_available_robots())


@app.get("/api/controllers")
async def list_controllers():
    return JSONResponse(get_available_controllers())


@app.get("/api/urdf/kinematics")
async def urdf_kinematics():
    urdf_path = os.path.join(os.path.dirname(__file__), "models", "nero", "urdf", "nero_description.urdf")
    try:
        if os.path.exists(urdf_path):
            chain = load_urdf_kinematics(urdf_path)
            return JSONResponse(chain)
        return JSONResponse({"error": "URDF not found", "path": urdf_path}, status_code=404)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    last_send = 0.0

    robot = websocket.app.state.robot
    controller = websocket.app.state.controller
    arm_side = os.getenv("CONTROL_ARM_SIDE", "right")

    def _send_to_controller(robot_angles: dict | None):
        if not robot_angles or not controller.enabled:
            return
        sides = ("left", "right") if arm_side == "both" else (arm_side,)
        for s in sides:
            side_data = robot_angles.get(s)
            if side_data and side_data.get("joints"):
                controller.send_joints(side_data["joints"], side_data.get("gripper", 0))

    async def handle_incoming():
        nonlocal robot, controller
        async for raw in websocket.iter_text():
            try:
                msg = json.loads(raw)
                action = msg.get("action", "")

                if action == "set_robot":
                    robot_id = msg.get("robot", "")
                    robot = get_robot(robot_id)
                    old_ctrl = controller
                    controller = _init_controller(robot)
                    websocket.app.state.robot = robot
                    websocket.app.state.controller = controller
                    old_ctrl.disconnect()
                    await websocket.send_text(json.dumps({
                        "type": "robot_changed",
                        "robot": {"id": robot.robot_id, "name": robot.name},
                        "control": controller.status.to_dict(),
                        "control_enabled": controller.enabled,
                    }))

                elif action == "toggle_face":
                    enabled = msg.get("enabled", False)
                    websocket.app.state.engine.face_enabled = enabled
                    await websocket.send_text(json.dumps({
                        "type": "face_toggled",
                        "enabled": enabled,
                    }))

                elif action == "control_enable":
                    enabled = msg.get("enabled", False)
                    controller.enabled = enabled
                    if enabled and controller.status.state != ControllerState.CONNECTED:
                        logger.info("Reconnecting controller (re-enable requested)")
                        controller.connect()
                        controller.enabled = True
                    await websocket.send_text(json.dumps({
                        "type": "control_status",
                        "control": controller.status.to_dict(),
                        "enabled": controller.enabled,
                    }))

                elif action == "control_reconnect":
                    logger.info("Manual reconnect requested")
                    controller.disconnect()
                    connected = controller.connect()
                    if connected:
                        controller.enabled = True
                    await websocket.send_text(json.dumps({
                        "type": "control_status",
                        "control": controller.status.to_dict(),
                        "enabled": controller.enabled,
                    }))

                elif action == "control_configure":
                    ctrl_type = msg.get("type", "dummy")
                    arm_side = msg.get("arm", "right")
                    host = msg.get("host", "127.0.0.1")
                    port = int(msg.get("port", 30002))
                    url = msg.get("url", "")
                    logger.info(
                        "Reconfiguring controller: type=%s host=%s port=%d arm=%s",
                        ctrl_type, host, port, arm_side,
                    )
                    controller.disconnect()
                    kwargs = {}
                    if ctrl_type == "tcp":
                        kwargs.update(host=host, port=port)
                    elif ctrl_type == "http":
                        kwargs["url"] = url or os.getenv("ROBOT_HTTP_URL", "http://192.168.1.65:5173/v1/viewer/joints")
                    controller = create_controller(
                        controller_type=ctrl_type,
                        robot_id=robot.robot_id,
                        robot_name=robot.name,
                        **kwargs,
                    )
                    websocket.app.state.controller = controller
                    connected = controller.connect()
                    if connected:
                        controller.enabled = True
                    await websocket.send_text(json.dumps({
                        "type": "control_status",
                        "control": controller.status.to_dict(),
                        "enabled": controller.enabled,
                    }))

            except (json.JSONDecodeError, ValueError):
                pass

    recv_task = asyncio.create_task(handle_incoming())

    await websocket.send_text(json.dumps({
        "control": controller.status.to_dict(),
        "control_enabled": controller.enabled,
        "robot_status": True,
    }))

    try:
        while True:
            frame, angles = websocket.app.state.engine.get_latest()

            if frame is None:
                await asyncio.sleep(0.01)
                continue

            now = time.time()
            interval = 1.0 / 15.0 if angles else 1.0 / 2.0
            if now - last_send < interval:
                await asyncio.sleep(0.001)
                continue
            last_send = now

            success, buffer = cv2.imencode(
                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 60]
            )
            if not success:
                continue

            image_b64 = base64.b64encode(buffer).decode("utf-8")
            robot_angles = _compute_robot_angles(angles, robot)

            _send_to_controller(robot_angles)

            message = _safe_json({
                "image": image_b64,
                "angles": angles,
                "robot": robot_angles,
                "control": controller.status.to_dict(),
                "control_enabled": controller.enabled,
            })

            try:
                await websocket.send_text(json.dumps(message))
            except RuntimeError:
                break

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected normally")
    except Exception as e:
        logger.error("WebSocket error: %s", e, exc_info=True)
    finally:
        recv_task.cancel()
        try:
            await recv_task
        except asyncio.CancelledError:
            pass


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
