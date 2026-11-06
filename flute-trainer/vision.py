"""Camera + MediaPipe wrapper (MediaPipe Tasks API: pose and hands).

Frames are mirrored (selfie view) BEFORE detection, so MediaPipe's "Left"/"Right" labels match
the player's real left/right hand.
"""
import cv2
import numpy as np

import config

HAND_LINKS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10),
              (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18),
              (18, 19), (19, 20), (0, 17)]
POSE_LINKS = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24),
              (0, 7), (0, 8)]


class VisionTracker:
    def __init__(self, use_hands=True, use_pose=True):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
        needed = ([config.POSE_MODEL] if use_pose else []) + ([config.HAND_MODEL] if use_hands else [])
        for p in needed:
            if not p.exists():
                raise FileNotFoundError(f"{p.name} not found. Run: python download_models.py")
        self.mp = mp
        run_mode = mp_vision.RunningMode.VIDEO
        self.pose = None
        if use_pose:
            self.pose = mp_vision.PoseLandmarker.create_from_options(
                mp_vision.PoseLandmarkerOptions(
                    base_options=mp_python.BaseOptions(model_asset_path=str(config.POSE_MODEL)),
                    running_mode=run_mode, num_poses=1))
        self.hands = None
        if use_hands:
            self.hands = mp_vision.HandLandmarker.create_from_options(
                mp_vision.HandLandmarkerOptions(
                    base_options=mp_python.BaseOptions(model_asset_path=str(config.HAND_MODEL)),
                    running_mode=run_mode, num_hands=2))
        self._last_ts = -1

    def process(self, bgr_frame, ts_ms):
        """Return (pose_landmarks or None, {"Left": (21,3)|None, "Right": ...})."""
        ts_ms = int(max(ts_ms, self._last_ts + 1))      # MediaPipe needs strictly rising times
        self._last_ts = ts_ms
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        pose = None
        if self.pose is not None:
            pose_res = self.pose.detect_for_video(image, ts_ms)
            pose = pose_res.pose_landmarks[0] if pose_res.pose_landmarks else None
        hands = {"Left": None, "Right": None}
        if self.hands is not None:
            res = self.hands.detect_for_video(image, ts_ms)
            for lms, handed in zip(res.hand_landmarks, res.handedness):
                side = handed[0].category_name
                hands[side] = np.array([[p.x, p.y, p.z] for p in lms])
        return pose, hands

    def close(self):
        if self.pose is not None:
            self.pose.close()
        if self.hands is not None:
            self.hands.close()


def draw_pose(frame, pose, issues_codes=()):
    h, w = frame.shape[:2]
    for a, b in POSE_LINKS:
        pa, pb = pose[a], pose[b]
        cv2.line(frame, (int(pa.x * w), int(pa.y * h)), (int(pb.x * w), int(pb.y * h)),
                 (255, 200, 0), 2)
    for i in (0, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24):
        cv2.circle(frame, (int(pose[i].x * w), int(pose[i].y * h)), 4, (0, 255, 255), -1)


def draw_hands(frame, hands):
    h, w = frame.shape[:2]
    for side, pts in hands.items():
        if pts is None:
            continue
        color = (0, 200, 0) if side == "Left" else (0, 140, 255)
        for a, b in HAND_LINKS:
            cv2.line(frame, (int(pts[a][0] * w), int(pts[a][1] * h)),
                     (int(pts[b][0] * w), int(pts[b][1] * h)), color, 2)
        for p in pts:
            cv2.circle(frame, (int(p[0] * w), int(p[1] * h)), 3, (255, 255, 255), -1)
