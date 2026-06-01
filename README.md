# Motion Reading

Real-time human arm & hand pose estimation from webcam, with facial expression detection and multi-robot teleoperation mapping.

## Features

- **Real-time Pose Estimation** — MediaPipe Pose (33 landmarks) + Hands (21 landmarks per hand)
- **Joint Angle Calculation** — Elbow, shoulder, wrist, and finger PIP angles
- **Facial Expression Detection** — Mouth open, smile, eye blink, eyebrow raise, head pose (toggleable)
- **Robot Arm Mapping** — Human arm → robot joint angles for multiple robots:
  - AgileX NERO 7-Axis
  - Franka Emika Panda
  - KUKA LBR iiwa 7
  - Kinova Gen3 7-DOF
  - Universal Robots UR5e
- **Gripper Control** — Hand open/close → gripper percentage
- **Web Dashboard** — Real-time video feed + angle gauges + robot joint display
- **Extensible** — Add new robot arms via config only (`robot_arms/configs.py`)

## Quick Start

```bash
pip install -r requirements.txt
python server.py
```

Open **http://localhost:8000**

## Architecture

```
camera → PoseEngine (MediaPipe) → WebSocket → Browser Dashboard
                                      ↓
                              Robot Mapping (config-driven)
```

```
motion-reading/
├── pose_engine.py        # MediaPipe pose/hands/face processing + angle calc
├── server.py             # FastAPI + WebSocket server
├── requirements.txt      # mediapipe, opencv, fastapi, uvicorn, numpy
├── robot_arms/
│   ├── __init__.py       # Registry: get_robot("nero7")
│   ├── base.py           # RobotArmBase, ConfigurableArm, ArmVectors
│   ├── configs.py        # Robot configurations (joint limits, names)
│   └── nero7.py          # Legacy Nero7 implementation
└── static/
    └── index.html        # Frontend dashboard (vanilla JS, no deps)
```

## Adding a New Robot

Add a config entry in `robot_arms/configs.py`:

```python
"my_robot": RobotConfig(
    manufacturer="MyBrand",
    model="MyModel",
    dof=7,
    joints={
        "j0": JointConfig(-180, 180, "Base"),
        "j1": JointConfig(-120, 120, "Shoulder"),
        # ... j2-j6
    },
)
```

The robot selector will auto-populate with the new entry. No mapping code needed.

## License

MIT
