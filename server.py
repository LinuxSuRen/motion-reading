import asyncio
import base64
import json
import logging
import time
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
import uvicorn

from pose_engine import PoseEngine
from robot_arms.base import ArmVectors
from robot_arms import get_robot, get_available_robots

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


@asynccontextmanager
async def lifespan(app: FastAPI):
    engine = PoseEngine()
    app.state.engine = engine
    app.state.robot = get_robot()
    engine.start()
    yield
    engine.stop()


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def root():
    return FileResponse("static/index.html")


@app.get("/api/robots")
async def list_robots():
    return JSONResponse(get_available_robots())


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    last_send = 0.0

    robot = websocket.app.state.robot

    async def handle_incoming():
        nonlocal robot
        async for raw in websocket.iter_text():
            try:
                msg = json.loads(raw)
                if msg.get("action") == "set_robot":
                    robot_id = msg.get("robot", "")
                    robot = get_robot(robot_id)
                    await websocket.send_text(json.dumps({
                        "type": "robot_changed",
                        "robot": {"id": robot.robot_id, "name": robot.name},
                    }))
                elif msg.get("action") == "toggle_face":
                    enabled = msg.get("enabled", False)
                    websocket.app.state.engine.face_enabled = enabled
                    await websocket.send_text(json.dumps({
                        "type": "face_toggled",
                        "enabled": enabled,
                    }))
            except (json.JSONDecodeError, ValueError):
                pass

    recv_task = asyncio.create_task(handle_incoming())

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

            message = _safe_json({
                "image": image_b64,
                "angles": angles,
                "robot": robot_angles,
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
