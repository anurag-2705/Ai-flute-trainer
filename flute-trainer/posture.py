"""Posture rules from MediaPipe pose landmarks. Pure geometry, so it can be tested without a camera.

Coordinates: pixel space, x to the right, y DOWN. The frame is mirrored (selfie view), so
"left"/"right" below are the player's own left/right.
"""
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

import config

# MediaPipe pose landmark indices
IDX = {"nose": 0, "left_ear": 7, "right_ear": 8,
       "left_shoulder": 11, "right_shoulder": 12,
       "left_elbow": 13, "right_elbow": 14,
       "left_wrist": 15, "right_wrist": 16,
       "left_hip": 23, "right_hip": 24}


@dataclass
class Issue:
    code: str
    message: str


def angle_deg(a, b, c):
    """Angle at point b formed by a-b-c, in degrees (0..180)."""
    a, b, c = (np.asarray(p, dtype=float)[:2] for p in (a, b, c))
    v1, v2 = a - b, c - b
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    if n == 0:
        return 0.0
    return float(np.degrees(np.arccos(np.clip(np.dot(v1, v2) / n, -1.0, 1.0))))


def line_tilt_deg(p1, p2):
    """Tilt of the line p1->p2 from horizontal, in (-90, 90]. Positive = p2 is lower than p1."""
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    return float(np.degrees(np.arctan2(dy, abs(dx) if dx != 0 else 1e-9)))


def from_mediapipe(landmarks, width, height) -> Dict[str, Tuple[float, float, float]]:
    """Convert MediaPipe normalised landmarks to {name: (x_px, y_px, visibility)}."""
    out = {}
    for name, i in IDX.items():
        lm = landmarks[i]
        out[name] = (lm.x * width, lm.y * height, getattr(lm, "visibility", 1.0))
    return out


class PostureChecker:
    def __init__(self, flute_side="right", limits=None):
        self.far = "right" if flute_side == "right" else "left"
        self.near = "left" if self.far == "right" else "right"
        self.lim = dict(config.POSTURE, **(limits or {}))

    def _visible(self, pts, names):
        return all(pts[n][2] >= self.lim["min_visibility"] for n in names)

    def check(self, pts) -> List[Issue]:
        L = self.lim
        issues: List[Issue] = []
        arms = ["left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                "left_wrist", "right_wrist"]
        if not self._visible(pts, arms):
            return [Issue("not_visible", "Move back so both arms and shoulders are in view.")]

        # 1. flute angle: line between the two wrists (near hand -> far hand)
        tilt = line_tilt_deg(pts[f"{self.near}_wrist"], pts[f"{self.far}_wrist"])
        if tilt > L["flute_droop_max"]:
            issues.append(Issue("flute_droop",
                                f"Flute is drooping ({tilt:.0f} deg). Lift the far end a little."))
        elif tilt < -L["flute_rise_max"]:
            issues.append(Issue("flute_rise",
                                f"Flute is pointing up ({-tilt:.0f} deg). Lower the far end."))

        # 2. shoulders level
        s_tilt = abs(line_tilt_deg(pts["left_shoulder"], pts["right_shoulder"]))
        if s_tilt > L["shoulder_tilt_max"]:
            issues.append(Issue("shoulders", "Keep both shoulders level and relaxed."))

        # 3. head upright
        if self._visible(pts, ["left_ear", "right_ear"]):
            h_tilt = abs(line_tilt_deg(pts["left_ear"], pts["right_ear"]))
            if h_tilt > L["head_tilt_max"]:
                issues.append(Issue("head", "Keep your head upright, do not tilt it sideways."))

        # 4. elbows: not folded too tight, not locked straight, not hiked up
        sw = abs(pts["left_shoulder"][0] - pts["right_shoulder"][0]) or 1.0
        for side in ("left", "right"):
            ang = angle_deg(pts[f"{side}_shoulder"], pts[f"{side}_elbow"], pts[f"{side}_wrist"])
            lo = L["near_elbow_angle_min"] if side == self.near else L["elbow_angle_min"]
            if ang < lo:
                issues.append(Issue(f"{side}_elbow_tight",
                                    f"Open your {side} elbow a little, the arm is folded too tight."))
            elif ang > L["elbow_angle_max"]:
                issues.append(Issue(f"{side}_elbow_straight",
                                    f"Bend your {side} elbow slightly, the arm is too straight."))
            rise = (pts[f"{side}_shoulder"][1] - pts[f"{side}_elbow"][1]) / sw
            if rise > L["elbow_over_shoulder"]:
                issues.append(Issue(f"{side}_elbow_high",
                                    f"Lower your {side} elbow, it is above your shoulder."))

        # 5. torso upright (only if hips are visible)
        if self._visible(pts, ["left_hip", "right_hip"]):
            sm = np.mean([pts["left_shoulder"][:2], pts["right_shoulder"][:2]], axis=0)
            hm = np.mean([pts["left_hip"][:2], pts["right_hip"][:2]], axis=0)
            lean = abs(np.degrees(np.arctan2(sm[0] - hm[0], hm[1] - sm[1])))
            if lean > L["torso_lean_max"]:
                issues.append(Issue("lean", "Stand or sit straight, you are leaning."))
        return issues
