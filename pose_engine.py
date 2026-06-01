import threading
import time
import numpy as np
import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


def calc_angle(a, b, c):
    """Angle at point b, formed by a-b-c. Returns degrees."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


class PoseEngine:
    """Real-time pose and hand angle computation using MediaPipe."""

    # Pose landmark indices
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_HIP = 23
    RIGHT_HIP = 24

    # Hand landmark indices
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

        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.6,
        )
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self._lock = threading.Lock()
        self._running = False
        self._frame = None
        self._angles = None
        self._thread = None
        self.face_enabled = False

        self._pose_style = mp_drawing_styles.get_default_pose_landmarks_style()
        self._hand_style = mp_drawing_styles.get_default_hand_landmarks_style()
        self._hand_connections = mp_drawing_styles.get_default_hand_connections_style()

    def start(self):
        """Start background camera capture and processing thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def _capture_loop(self):
        """Continuously capture frames from camera and process through MediaPipe."""
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

                # Resize to max width while maintaining aspect ratio
                h, w = frame.shape[:2]
                if w > self.max_width:
                    scale = self.max_width / w
                    frame = cv2.resize(frame, (self.max_width, int(h * scale)))

                # Process through MediaPipe pipeline
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                pose_result = self.pose.process(rgb)
                hands_result = self.hands.process(rgb)
                face_result = self.face_mesh.process(rgb) if self.face_enabled else None
                rgb.flags.writeable = True

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
        rgb.flags.writeable = False
        pose_result = self.pose.process(rgb)
        hands_result = self.hands.process(rgb)
        face_result = self.face_mesh.process(rgb) if self.face_enabled else None
        rgb.flags.writeable = True

        return self._compute_angles_from_results(frame, pose_result, hands_result, face_result)

    def _compute_angles_from_results(
        self, frame: np.ndarray, pose_result, hands_result, face_result
    ) -> dict | None:
        if pose_result.pose_landmarks is None:
            return None

        if not self._torso_visible(pose_result.pose_landmarks):
            return None

        # Prefer 3D world landmarks when available, fall back to 2D image landmarks
        use_world = pose_result.pose_world_landmarks is not None
        landmarks = pose_result.pose_world_landmarks if use_world else pose_result.pose_landmarks

        # Compute arm angles with per-arm visibility check
        left_arm = self._compute_arm_angles(landmarks, pose_result.pose_landmarks, "left")
        right_arm = self._compute_arm_angles(landmarks, pose_result.pose_landmarks, "right")

        if not left_arm and not right_arm:
            return None

        left_arm = left_arm or {}
        right_arm = right_arm or {}

        person_left = None
        person_right = None
        left_world = None
        right_world = None

        if hands_result.multi_hand_landmarks and hands_result.multi_handedness:
            for i, handedness in enumerate(hands_result.multi_handedness):
                label = handedness.classification[0].label
                if label == "Left":
                    person_left = hands_result.multi_hand_landmarks[i]
                    if hands_result.multi_hand_world_landmarks:
                        left_world = hands_result.multi_hand_world_landmarks[i]
                elif label == "Right":
                    person_right = hands_result.multi_hand_landmarks[i]
                    if hands_result.multi_hand_world_landmarks:
                        right_world = hands_result.multi_hand_world_landmarks[i]

        if person_left is not None:
            right_arm["wrist"] = self._compute_wrist_angle(
                pose_result.pose_landmarks, person_left, "right"
            )
        if person_right is not None:
            left_arm["wrist"] = self._compute_wrist_angle(
                pose_result.pose_landmarks, person_right, "left"
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

        return {
            "left_arm": left_arm,
            "right_arm": right_arm,
            "left_hand": left_hand,
            "right_hand": right_hand,
            "left_vectors": left_vectors,
            "right_vectors": right_vectors,
            "face": face,
        }

    def _draw_overlays(self, frame, angles, pose_result, hands_result, face_result=None):
        h, w = frame.shape[:2]

        # ---- Draw pose skeleton ----
        if pose_result.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                pose_result.pose_landmarks,
                mp.solutions.pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self._pose_style,
            )

        # ---- Draw hand skeletons ----
        if hands_result and hands_result.multi_hand_landmarks:
            for hand_lm in hands_result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_lm,
                    mp.solutions.hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=self._hand_style,
                    connection_drawing_spec=self._hand_connections,
                )

        # ---- Draw angle text at joints ----
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

        # Arm angle labels
        if pose_result.pose_landmarks and angles:
            pl = pose_result.pose_landmarks.landmark
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

        # ---- Hand finger labels ----
        if hands_result and hands_result.multi_hand_landmarks and hands_result.multi_handedness and angles:
            finger_names = ["thumb", "index", "middle", "ring", "pinky"]
            finger_tips = [self.THUMB_TIP, self.INDEX_TIP, self.MIDDLE_TIP,
                           self.RING_TIP, self.PINKY_TIP]
            finger_pips = [None, self.INDEX_PIP, self.MIDDLE_PIP,
                           self.RING_PIP, self.PINKY_PIP]

            for i, handedness in enumerate(hands_result.multi_handedness):
                hand_lm = hands_result.multi_hand_landmarks[i]
                side_label = "L" if handedness.classification[0].label == "Left" else "R"
                hand_key = f"{'right' if side_label == 'L' else 'left'}_hand"
                hand_data = angles.get(hand_key) or {}

                for fi, (name, tip_idx, pip_idx) in enumerate(zip(finger_names, finger_tips, finger_pips)):
                    finger_data = hand_data.get(name, {})
                    extended = finger_data.get("extended")
                    pip_angle = finger_data.get("pip_angle")

                    tx, ty = _px(hand_lm.landmark[tip_idx])
                    dot_color = (0, 255, 0) if extended else (80, 80, 80)
                    cv2.circle(frame, (tx, ty), 5, dot_color, -1)
                    cv2.circle(frame, (tx, ty), 6, (255, 255, 255), 1)

                    if pip_idx is not None and pip_angle is not None:
                        px, py = _px(hand_lm.landmark[pip_idx])
                        _draw_label(px + 10, py - 4, f"{pip_angle:.0f}",
                                    _angle_color(pip_angle, 100, 140))

        # ---- Draw face mesh (lightweight, contours only) ----
        if face_result and face_result.multi_face_landmarks:
            face_lm = face_result.multi_face_landmarks[0]
            face_style = mp_drawing.DrawingSpec(color=(200, 200, 255), thickness=1, circle_radius=1)
            mp_drawing.draw_landmarks(
                frame, face_lm, mp.solutions.face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=face_style, connection_drawing_spec=face_style,
            )

            flm = face_lm.landmark
            h, w = frame.shape[:2]
            cx, cy = int(flm[1].x * w), int(flm[1].y * h)
            _draw_label(cx + 10, cy, "FACE", (200, 200, 255))

    def _torso_visible(self, pose_landmarks) -> bool:
        landmarks = pose_landmarks.landmark
        key_points = [
            self.LEFT_SHOULDER, self.RIGHT_SHOULDER,
            self.LEFT_ELBOW, self.RIGHT_ELBOW,
        ]
        visible_count = sum(
            1 for idx in key_points
            if landmarks[idx].visibility >= 0.5
        )
        return visible_count >= 2

    def _compute_face_expressions(self, face_result) -> dict | None:
        if face_result is None or face_result.multi_face_landmarks is None:
            return None

        lm = face_result.multi_face_landmarks[0].landmark

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
            "shoulder": list(self._xyz(landmarks.landmark[sh])),
            "elbow": list(self._xyz(landmarks.landmark[el])),
            "wrist": list(self._xyz(landmarks.landmark[wr])),
        }

    def _compute_arm_angles(self, landmarks, image_landmarks, side):
        if side == "left":
            shoulder, elbow, wrist = self.LEFT_SHOULDER, self.LEFT_ELBOW, self.LEFT_WRIST
        else:
            shoulder, elbow, wrist = self.RIGHT_SHOULDER, self.RIGHT_ELBOW, self.RIGHT_WRIST

        if image_landmarks:
            key_vis = [
                image_landmarks.landmark[shoulder].visibility,
                image_landmarks.landmark[elbow].visibility,
                image_landmarks.landmark[wrist].visibility,
            ]
            if key_vis[0] < 0.5:
                return None
            if sum(1 for v in key_vis[1:] if v > 0) == 0:
                return None

        elbow_angle = calc_angle(
            self._xyz(landmarks.landmark[shoulder]),
            self._xyz(landmarks.landmark[elbow]),
            self._xyz(landmarks.landmark[wrist]),
        )
        shoulder_angle = calc_angle(
            self._xyz(landmarks.landmark[self.LEFT_HIP if side == "left" else self.RIGHT_HIP]),
            self._xyz(landmarks.landmark[shoulder]),
            self._xyz(landmarks.landmark[elbow]),
        )
        return {
            "elbow": round(elbow_angle, 1),
            "shoulder": round(shoulder_angle, 1),
        }

    def _compute_wrist_angle(self, pose_landmarks, hand_landmarks, side):
        """Compute wrist angle from pose elbow/wrist + hand index MCP in image space."""
        if side == "left":
            elbow = pose_landmarks.landmark[self.LEFT_ELBOW]
            wrist = pose_landmarks.landmark[self.LEFT_WRIST]
        else:
            elbow = pose_landmarks.landmark[self.RIGHT_ELBOW]
            wrist = pose_landmarks.landmark[self.RIGHT_WRIST]

        index_mcp = hand_landmarks.landmark[self.INDEX_MCP]

        return round(
            calc_angle(
                self._xyz(elbow),
                self._xyz(wrist),
                self._xyz(index_mcp),
            ),
            1,
        )

    def _compute_hand_fingers(self, hand_landmarks, hand_world_landmarks, side):
        """Compute finger extension states and PIP joint angles for one hand.
        
        Uses 3D world landmarks for angles when available, 2D image landmarks as fallback.
        Extension heuristic: for index/middle/ring/pinky, tip.y < pip.y (tip above PIP).
        For thumb, compare x coordinates based on hand side.
        """
        lm = hand_world_landmarks if hand_world_landmarks else hand_landmarks
        image_lm = hand_landmarks  # used for extension heuristic (needs image coords)

        result = {}

        # Thumb: no PIP joint; extension uses x-axis comparison
        thumb_tip = image_lm.landmark[self.THUMB_TIP]
        thumb_ip = image_lm.landmark[self.THUMB_IP]
        if side == "Left":
            thumb_extended = (thumb_tip.x - thumb_ip.x) > 0.04
        else:
            thumb_extended = (thumb_ip.x - thumb_tip.x) > 0.04

        result["thumb"] = {"extended": thumb_extended}

        # Index, Middle, Ring, Pinky: PIP angle + extension
        finger_configs = [
            ("index", self.INDEX_MCP, self.INDEX_PIP, self.INDEX_DIP, self.INDEX_TIP),
            ("middle", self.MIDDLE_MCP, self.MIDDLE_PIP, self.MIDDLE_DIP, self.MIDDLE_TIP),
            ("ring", self.RING_MCP, self.RING_PIP, self.RING_DIP, self.RING_TIP),
            ("pinky", self.PINKY_MCP, self.PINKY_PIP, self.PINKY_DIP, self.PINKY_TIP),
        ]

        for name, mcp_idx, pip_idx, dip_idx, tip_idx in finger_configs:
            # Extension: fingertip is above PIP joint (smaller y in image coordinates)
            extended = image_lm.landmark[tip_idx].y < image_lm.landmark[pip_idx].y

            # PIP angle: angle at PIP joint formed by MCP-PIP-DIP
            pip_angle = calc_angle(
                self._xyz(lm.landmark[mcp_idx]),
                self._xyz(lm.landmark[pip_idx]),
                self._xyz(lm.landmark[dip_idx]),
            )

            result[name] = {
                "extended": extended,
                "pip_angle": round(pip_angle, 1),
            }

        return result

    @staticmethod
    def _xyz(landmark):
        """Extract (x, y, z) tuple from a MediaPipe landmark."""
        return (landmark.x, landmark.y, landmark.z)

    def get_latest(self):
        """Thread-safe access to latest frame and angles.
        
        Returns:
            Tuple of (frame: np.ndarray | None, angles: dict | None)
        """
        with self._lock:
            return self._frame, self._angles

    def stop(self):
        """Stop capture thread and release MediaPipe resources."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        self.pose.close()
        self.hands.close()
        self.face_mesh.close()
