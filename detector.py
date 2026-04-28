"""
detector.py — Core MediaPipe detection engine
Handles pose estimation, eye tracking, angle calculation,
and feature extraction for the ML classifier.
"""

import numpy as np
import mediapipe as mp
import cv2
from config import (
    LEFT_EYE_IDX, RIGHT_EYE_IDX,
    EAR_THRESHOLD, EYE_DIST_MIN, EYE_DIST_MAX,
    SPINE_ANGLE_THRESHOLD, NECK_TILT_THRESHOLD,
)


class PostureDetector:
    """
    Wraps MediaPipe Pose + FaceMesh for real-time posture analysis.

    Usage
    -----
    detector = PostureDetector()
    result   = detector.process(frame)
    # result is a DetectionResult namedtuple
    """

    def __init__(self):
        self._pose = mp.solutions.pose.Pose(
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        self._face = mp.solutions.face_mesh.FaceMesh(
            refine_landmarks=True,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        self._draw      = mp.solutions.drawing_utils
        self._draw_style = mp.solutions.drawing_styles
        self._pose_conn  = mp.solutions.pose.POSE_CONNECTIONS

    # ── Public API ──────────────────────────────────────────────────────────

    def process(self, frame: np.ndarray) -> dict:
        """
        Run full detection on one BGR frame.

        Returns
        -------
        dict with keys:
            pose_landmarks   : raw MediaPipe pose result (or None)
            face_landmarks   : raw MediaPipe face result (or None)
            keypoint_row     : list[float] — 132 features for ML model
            spine_angle      : float (degrees)
            neck_tilt        : float (degrees, left is negative)
            ear              : float — Eye Aspect Ratio
            eye_dist         : float — inter-eye pixel distance
            issues           : list[str] — active issue keys
            frame_annotated  : np.ndarray — frame with skeleton drawn
        """
        h, w     = frame.shape[:2]
        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        out      = frame.copy()

        pose_res = self._pose.process(rgb)
        face_res = self._face.process(rgb)

        result = {
            "pose_landmarks":  pose_res.pose_landmarks if pose_res else None,
            "face_landmarks":  face_res.multi_face_landmarks[0] if (face_res and face_res.multi_face_landmarks) else None,
            "keypoint_row":    [],
            "spine_angle":     180.0,
            "neck_tilt":       0.0,
            "ear":             1.0,
            "eye_dist":        100.0,
            "issues":          [],
            "frame_annotated": out,
        }

        # ── Pose ───────────────────────────────────────────────────────────
        if pose_res and pose_res.pose_landmarks:
            lm = pose_res.pose_landmarks.landmark

            # Draw skeleton
            self._draw.draw_landmarks(
                out,
                pose_res.pose_landmarks,
                self._pose_conn,
                self._draw_style.get_default_pose_landmarks_style(),
            )

            # Feature vector (132 values)
            row = []
            for point in lm:
                row += [point.x, point.y, point.z, point.visibility]
            result["keypoint_row"] = row

            # Angles
            result["spine_angle"] = self._spine_angle(lm, w, h)
            result["neck_tilt"]   = self._neck_tilt(lm, w, h)

            # Rule-based issue detection
            if result["spine_angle"] < SPINE_ANGLE_THRESHOLD:
                result["issues"].append("slouch")
            if abs(result["neck_tilt"]) > NECK_TILT_THRESHOLD:
                side = "left" if result["neck_tilt"] < 0 else "right"
                result["issues"].append(f"neck_tilt_{side}")

        # ── Eyes ───────────────────────────────────────────────────────────
        if face_res and face_res.multi_face_landmarks:
            lm_f = face_res.multi_face_landmarks[0].landmark
            result["ear"]      = self._avg_ear(lm_f, w, h)
            result["eye_dist"] = self._inter_eye_dist(lm_f, w, h)

            if result["ear"] < EAR_THRESHOLD:
                result["issues"].append("eye_closing")
            if result["eye_dist"] < EYE_DIST_MIN:
                result["issues"].append("too_close")
            elif result["eye_dist"] > EYE_DIST_MAX:
                result["issues"].append("too_far")

        result["frame_annotated"] = out
        return result

    def extract_features_from_image(self, image_path: str):
        """
        Extract keypoint row from a static image file.
        Returns list[float] or None if detection failed.
        """
        img = cv2.imread(image_path)
        if img is None:
            return None
        result = self.process(img)
        return result["keypoint_row"] if result["keypoint_row"] else None

    def release(self):
        """Free MediaPipe resources."""
        self._pose.close()
        self._face.close()

    # ── Geometry helpers ────────────────────────────────────────────────────

    @staticmethod
    def _angle(a, b, c) -> float:
        """Angle in degrees at point b, formed by a-b-c."""
        a, b, c  = np.array(a), np.array(b), np.array(c)
        ba, bc   = a - b, c - b
        cos_val  = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
        return float(np.degrees(np.arccos(np.clip(cos_val, -1.0, 1.0))))

    @staticmethod
    def _pt(lm, idx, w, h):
        return (lm[idx].x * w, lm[idx].y * h)

    def _spine_angle(self, lm, w, h) -> float:
        ls  = self._pt(lm, 11, w, h)
        rs  = self._pt(lm, 12, w, h)
        lhip = self._pt(lm, 23, w, h)
        rhip = self._pt(lm, 24, w, h)
        mid_s   = ((ls[0]+rs[0])/2,   (ls[1]+rs[1])/2)
        mid_h   = ((lhip[0]+rhip[0])/2, (lhip[1]+rhip[1])/2)
        vert    = (mid_h[0], mid_h[1] - 100)
        return self._angle(vert, mid_h, mid_s)

    def _neck_tilt(self, lm, w, h) -> float:
        ls = self._pt(lm, 11, w, h)
        rs = self._pt(lm, 12, w, h)
        nose = self._pt(lm, 0, w, h)
        mid_s = ((ls[0]+rs[0])/2, (ls[1]+rs[1])/2)
        dx = nose[0] - mid_s[0]
        dy = mid_s[1] - nose[1]
        return float(np.degrees(np.arctan2(dx, dy + 1e-9)))

    def _ear_one(self, lm, indices, w, h) -> float:
        pts = [(int(lm[i].x * w), int(lm[i].y * h)) for i in indices]
        A = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
        B = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
        C = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
        return float((A + B) / (2.0 * C + 1e-9))

    def _avg_ear(self, lm, w, h) -> float:
        return (self._ear_one(lm, LEFT_EYE_IDX, w, h) +
                self._ear_one(lm, RIGHT_EYE_IDX, w, h)) / 2.0

    def _inter_eye_dist(self, lm, w, h) -> float:
        lx = lm[LEFT_EYE_IDX[0]].x * w
        rx = lm[RIGHT_EYE_IDX[0]].x * w
        return float(abs(lx - rx))
