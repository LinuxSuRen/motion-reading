import os
import threading
import time
import numpy as np
import cv2
import mediapipe as mp

vision = mp.tasks.vision


def calc_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def _model_path(name):
    return os.path.join(os.path.dirname(__file__), name)


class PoseEngine:

    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_HIP = 23
    RIGHT_HIP = 24

    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_MCP = 5
    INDEX_PIP = 6
    INDEX_DIP = 7
    INDEX_TIP = 8
    MIDDLE_MCP = 9
    MIDDLE_PIP = 10
    MIDDLE_DIP = 11
    MIDDLE_TIP = 12
    RING_MCP = 13
    RING_PIP = 14
    RING_DIP = 15
    RING_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20

    def __init__(self, camera_id=0, max_width=640):
        self.camera_id = camera_id
        self.max_width = max_width

        pose_options = vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=_model_path("pose_landmarker_lite.task")
            ),
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.7,
            min_tracking_confidence=0.6,
        )
        self.pose = vision.PoseLandmarker.create_from_options(pose_options)

        hand_options = vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=_model_path("hand_landmarker.task")
            ),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.hands = vision.HandLandmarker.create_from_options(hand_options)

        face_options = vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=_model_path("face_landmarker.task")
            ),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            min_face_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.face_mesh = vision.FaceLandmarker.create_from_options(face_options)

        self._lock = threading.Lock()
        self._running = False
        self._frame = None
        self._angles = None
        self._thread = None
        self.face_enabled = False

        self._pose_style = vision.drawing_styles.get_default_pose_landmarks_style()
        self._hand_style = vision.drawing_styles.get_default_hand_landmarks_style()
        self._hand_connections = vision.drawing_styles.get_default_hand_connections_style()

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def _capture_loop(self):
        cap = cv2.VideoCapture(self.camera_id)
        if not cap.isOpened():
            self._running = False
            return

        try:
            while self._running:
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.01)
                    continue

                h, w = frame.shape[:2]
                if w > self.max_width:
                    scale = self.max_width / w
                    frame = cv2.resize(frame, (self.max_width, int(h * scale)))

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

                pose_result = self.pose.detect(mp_image)
                hands_result = self.hands.detect(mp_image)
                face_result = self.face_mesh.detect(mp_image) if self.face_enabled else None

                angles = self._compute_angles_from_results(
                    frame, pose_result, hands_result, face_result
                )
                if angles:
                    self._draw_overlays(frame, angles, pose_result, hands_result, face_result)

                with self._lock:
                    self._frame = frame
                    self._angles = angles
        finally:
            cap.release()

    def process_frame(self, frame: np.ndarray) -> dict | None:
        if frame is None or frame.size == 0:
            return None

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        pose_result = self.pose.detect(mp_image)
        hands_result = self.hands.detect(mp_image)
        face_result = self.face_mesh.detect(mp_image) if self.face_enabled else None

        return self._compute_angles_from_results(frame, pose_result, hands_result, face_result)

    def _compute_angles_from_results(
        self, frame: np.ndarray, pose_result, hands_result, face_result
    ) -> dict | None:
        if not pose_result.pose_landmarks:
            return None

        pose_lms = pose_result.pose_landmarks[0]
        pose_world_lms = pose_result.pose_world_landmarks[0] if pose_result.pose_world_landmarks else None

        if not self._torso_visible(pose_lms):
            return None

        use_world = pose_world_lms is not None
        landmarks = pose_world_lms if use_world else pose_lms

        left_arm = self._compute_arm_angles(landmarks, pose_lms, "left")
        right_arm = self._compute_arm_angles(landmarks, pose_lms, "right")

        if not left_arm and not right_arm:
            return None

        left_arm = left_arm or {}
        right_arm = right_arm or {}

        person_left = None
        person_right = None
        left_world = None
        right_world = None

        if hands_result.hand_landmarks and hands_result.handedness:
            for i, handedness in enumerate(hands_result.handedness):
                label = handedness[0].category_name
                if label == "Left":
                    person_left = hands_result.hand_landmarks[i]
                    if hands_result.hand_world_landmarks:
                        left_world = hands_result.hand_world_landmarks[i]
                elif label == "Right":
                    person_right = hands_result.hand_landmarks[i]
                    if hands_result.hand_world_landmarks:
                        right_world = hands_result.hand_world_landmarks[i]

        if person_left is not None:
            right_arm["wrist"] = self._compute_wrist_angle(
                pose_lms, person_left, "right"
            )
        if person_right is not None:
            left_arm["wrist"] = self._compute_wrist_angle(
                pose_lms, person_right, "left"
            )

        right_hand = (
            self._compute_hand_fingers(person_left, left_world, "Right")
            if person_left
            else None
        )
        left_hand = (
            self._compute_hand_fingers(person_right, right_world, "Left")
            if person_right
            else None
        )

        left_vectors = self._compute_arm_vectors(landmarks, "left")
        right_vectors = self._compute_arm_vectors(landmarks, "right")

        face = self._compute_face_expressions(face_result)

        result = {}
        if left_arm:
            result["left_arm"] = left_arm
        if right_arm:
            result["right_arm"] = right_arm
        if left_hand:
            result["left_hand"] = left_hand
        if right_hand:
            result["right_hand"] = right_hand
        result["left_vectors"] = left_vectors
        result["right_vectors"] = right_vectors
        result["face"] = face
        return result

    def _draw_overlays(self, frame, angles, pose_result, hands_result, face_result=None):
        h, w = frame.shape[:2]

        if pose_result.pose_landmarks:
            vision.drawing_utils.draw_landmarks(
                frame,
                pose_result.pose_landmarks[0],
                vision.PoseLandmarksConnections.POSE_LANDMARKS,
                landmark_drawing_spec=self._pose_style,
            )

        if hands_result and hands_result.hand_landmarks:
            for hand_lm in hands_result.hand_landmarks:
                vision.drawing_utils.draw_landmarks(
                    frame,
                    hand_lm,
                    vision.HandLandmarksConnections.HAND_CONNECTIONS,
                    landmark_drawing_spec=self._hand_style,
                    connection_drawing_spec=self._hand_connections,
                )

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        thickness = 1

        def _draw_label(x, y, text, color=(0, 255, 200)):
            (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
            cv2.rectangle(frame, (x - 4, y - th - 6), (x + tw + 4, y + 2), (0, 0, 0), -1)
            cv2.putText(frame, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)

        def _px(lm, margin=0):
            return int(lm.x * w), int(lm.y * h)

        def _angle_color(angle, low, mid):
            if angle is None:
                return (128, 128, 128)
            if angle < low:
                return (0, 0, 255)
            if angle < mid:
                return (0, 200, 255)
            return (0, 255, 0)

        if pose_result.pose_landmarks and angles:
            pl = pose_result.pose_landmarks[0]
            for side, shoulder_idx, elbow_idx, wrist_idx in [
                ("L", self.LEFT_SHOULDER, self.LEFT_ELBOW, self.LEFT_WRIST),
                ("R", self.RIGHT_SHOULDER, self.RIGHT_ELBOW, self.RIGHT_WRIST),
            ]:
                arm_data = angles.get(f"{'left' if side == 'L' else 'right'}_arm", {})
                elbow_angle = arm_data.get("elbow")
                shoulder_angle = arm_data.get("shoulder")
                wrist_angle = arm_data.get("wrist")

                if elbow_angle is not None:
                    ex, ey = _px(pl[elbow_idx])
                    _draw_label(ex + 14, ey, f"{side}E:{elbow_angle:.0f}",
                                _angle_color(elbow_angle, 30, 60))

                if shoulder_angle is not None:
                    sx, sy = _px(pl[shoulder_idx])
                    _draw_label(sx - 70, sy, f"{side}S:{shoulder_angle:.0f}",
                                _angle_color(shoulder_angle, 30, 80))

                if wrist_angle is not None:
                    wx, wy = _px(pl[wrist_idx])
                    _draw_label(wx + 14, wy, f"{side}W:{wrist_angle:.0f}",
                                _angle_color(wrist_angle, 130, 150))

        if hands_result and hands_result.hand_landmarks and hands_result.handedness and angles:
            finger_names = ["thumb", "index", "middle", "ring", "pinky"]
            finger_tips = [self.THUMB_TIP, self.INDEX_TIP, self.MIDDLE_TIP,
                           self.RING_TIP, self.PINKY_TIP]
            finger_pips = [None, self.INDEX_PIP, self.MIDDLE_PIP,
                           self.RING_PIP, self.PINKY_PIP]

            for i, handedness in enumerate(hands_result.handedness):
                hand_lm = hands_result.hand_landmarks[i]
                side_label = "L" if handedness[0].category_name == "Left" else "R"
                hand_key = f"{'right' if side_label == 'L' else 'left'}_hand"
                hand_data = angles.get(hand_key) or {}

                for fi, (name, tip_idx, pip_idx) in enumerate(zip(finger_names, finger_tips, finger_pips)):
                    finger_data = hand_data.get(name, {})
                    extended = finger_data.get("extended")
                    pip_angle = finger_data.get("pip_angle")

                    tx, ty = _px(hand_lm[tip_idx])
                    dot_color = (0, 255, 0) if extended else (80, 80, 80)
                    cv2.circle(frame, (tx, ty), 5, dot_color, -1)
                    cv2.circle(frame, (tx, ty), 6, (255, 255, 255), 1)

                    if pip_idx is not None and pip_angle is not None:
                        px, py = _px(hand_lm[pip_idx])
                        _draw_label(px + 10, py - 4, f"{pip_angle:.0f}",
                                    _angle_color(pip_angle, 100, 140))

        if face_result and face_result.face_landmarks:
            face_lm = face_result.face_landmarks[0]
            face_style = vision.drawing_utils.DrawingSpec(color=(200, 200, 255), thickness=1, circle_radius=1)
            vision.drawing_utils.draw_landmarks(
                frame, face_lm, vision.FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS,
                landmark_drawing_spec=face_style, connection_drawing_spec=face_style,
            )

            flm = face_lm
            h, w = frame.shape[:2]
            cx, cy = int(flm[1].x * w), int(flm[1].y * h)
            _draw_label(cx + 10, cy, "FACE", (200, 200, 255))

    def _torso_visible(self, pose_landmarks) -> bool:
        key_points = [
            self.LEFT_SHOULDER, self.RIGHT_SHOULDER,
            self.LEFT_ELBOW, self.RIGHT_ELBOW,
        ]
        visible_count = sum(
            1 for idx in key_points
            if hasattr(pose_landmarks[idx], 'visibility') and pose_landmarks[idx].visibility >= 0.5
        )
        return visible_count >= 2

    def _compute_face_expressions(self, face_result) -> dict | None:
        if face_result is None or not face_result.face_landmarks:
            return None

        lm = face_result.face_landmarks[0]

        def _d(i, j):
            a, b = lm[i], lm[j]
            return np.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)

        mouth_w = _d(61, 291)
        mouth_h = _d(13, 14)
        mouth_open = round(min(mouth_h / (mouth_w + 1e-6), 1.0), 2)
        smile = round(max(0.0, (mouth_w / (_d(61, 291) + 1e-6) - 1.0) * 10 + 0.5), 2)
        smile = max(0.0, min(1.0, smile))

        def ear(eye_top, eye_bot, eye_l, eye_r):
            return (_d(eye_top, eye_bot) + 1e-6) / (_d(eye_l, eye_r) * 2 + 1e-6)

        left_ear = round(ear(159, 145, 33, 133), 2)
        right_ear = round(ear(386, 374, 362, 263), 2)
        blink_left = left_ear < 0.15
        blink_right = right_ear < 0.15

        left_brow = round(_d(105, 159) / (_d(33, 133) + 1e-6), 2)
        right_brow = round(_d(334, 386) / (_d(362, 263) + 1e-6), 2)

        nose_x = lm[1].x
        face_center_x = (lm[234].x + lm[454].x) / 2.0
        yaw = round((nose_x - face_center_x) * 100, 1)
        pitch = round((lm[10].y - lm[152].y) * 100, 1)
        roll = round(np.degrees(np.arctan2(
            lm[234].y - lm[454].y,
            lm[234].x - lm[454].x + 1e-6
        )), 1)

        return {
            "mouth_open": mouth_open,
            "smile": smile,
            "left_ear": left_ear,
            "right_ear": right_ear,
            "blink_left": blink_left,
            "blink_right": blink_right,
            "left_brow": left_brow,
            "right_brow": right_brow,
            "head_yaw": yaw,
            "head_pitch": pitch,
            "head_roll": roll,
            "detected": True,
        }

    def _compute_arm_vectors(self, landmarks, side):
        if side == "left":
            sh, el, wr = self.LEFT_SHOULDER, self.LEFT_ELBOW, self.LEFT_WRIST
        else:
            sh, el, wr = self.RIGHT_SHOULDER, self.RIGHT_ELBOW, self.RIGHT_WRIST

        return {
            "shoulder": list(self._xyz(landmarks[sh])),
            "elbow": list(self._xyz(landmarks[el])),
            "wrist": list(self._xyz(landmarks[wr])),
        }

    def _compute_arm_angles(self, landmarks, image_landmarks, side):
        if side == "left":
            shoulder, elbow, wrist = self.LEFT_SHOULDER, self.LEFT_ELBOW, self.LEFT_WRIST
        else:
            shoulder, elbow, wrist = self.RIGHT_SHOULDER, self.RIGHT_ELBOW, self.RIGHT_WRIST

        if hasattr(image_landmarks[shoulder], 'visibility'):
            key_vis = [
                image_landmarks[shoulder].visibility,
                image_landmarks[elbow].visibility,
                image_landmarks[wrist].visibility,
            ]
            if key_vis[0] < 0.5:
                return None
            if sum(1 for v in key_vis[1:] if v > 0) == 0:
                return None

        elbow_angle = calc_angle(
            self._xyz(landmarks[shoulder]),
            self._xyz(landmarks[elbow]),
            self._xyz(landmarks[wrist]),
        )
        shoulder_angle = calc_angle(
            self._xyz(landmarks[self.LEFT_HIP if side == "left" else self.RIGHT_HIP]),
            self._xyz(landmarks[shoulder]),
            self._xyz(landmarks[elbow]),
        )
        return {
            "elbow": round(elbow_angle, 1),
            "shoulder": round(shoulder_angle, 1),
        }

    def _compute_wrist_angle(self, pose_landmarks, hand_landmarks, side):
        if side == "left":
            elbow = pose_landmarks[self.LEFT_ELBOW]
            wrist = pose_landmarks[self.LEFT_WRIST]
        else:
            elbow = pose_landmarks[self.RIGHT_ELBOW]
            wrist = pose_landmarks[self.RIGHT_WRIST]

        index_mcp = hand_landmarks[self.INDEX_MCP]

        return round(
            calc_angle(
                self._xyz(elbow),
                self._xyz(wrist),
                self._xyz(index_mcp),
            ),
            1,
        )

    def _compute_hand_fingers(self, hand_landmarks, hand_world_landmarks, side):
        lm = hand_world_landmarks if hand_world_landmarks else hand_landmarks
        image_lm = hand_landmarks

        result = {}

        thumb_tip = image_lm[self.THUMB_TIP]
        thumb_ip = image_lm[self.THUMB_IP]
        if side == "Left":
            thumb_extended = (thumb_tip.x - thumb_ip.x) > 0.04
        else:
            thumb_extended = (thumb_ip.x - thumb_tip.x) > 0.04

        result["thumb"] = {"extended": thumb_extended}

        finger_configs = [
            ("index", self.INDEX_MCP, self.INDEX_PIP, self.INDEX_DIP, self.INDEX_TIP),
            ("middle", self.MIDDLE_MCP, self.MIDDLE_PIP, self.MIDDLE_DIP, self.MIDDLE_TIP),
            ("ring", self.RING_MCP, self.RING_PIP, self.RING_DIP, self.RING_TIP),
            ("pinky", self.PINKY_MCP, self.PINKY_PIP, self.PINKY_DIP, self.PINKY_TIP),
        ]

        for name, mcp_idx, pip_idx, dip_idx, tip_idx in finger_configs:
            extended = image_lm[tip_idx].y < image_lm[pip_idx].y

            pip_angle = calc_angle(
                self._xyz(lm[mcp_idx]),
                self._xyz(lm[pip_idx]),
                self._xyz(lm[dip_idx]),
            )

            result[name] = {
                "extended": extended,
                "pip_angle": round(pip_angle, 1),
            }

        return result

    @staticmethod
    def _xyz(landmark):
        return (landmark.x, landmark.y, landmark.z)

    def get_latest(self):
        with self._lock:
            return self._frame, self._angles

    def stop(self):
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        self.pose.close()
        self.hands.close()
        self.face_mesh.close()
