"""Finger/hand-shape features and the classifier that recognises which note-shape a player makes.

Idea: a teacher (or a good player) records each note's hand shape with collect_data.py. The model
learns "this hand shape = note X". During a lesson we compare the predicted shape with the target.
We do NOT hard-code finger charts because they differ between bansuri, Western flute and
individual flutes; recording real examples is more reliable.
"""
import numpy as np

import config

FINGERTIPS = [8, 12, 16, 20]          # index, middle, ring, pinky tips
PIPS = [6, 10, 14, 18]                # their middle joints
MCPS = [5, 9, 13, 17]                 # their knuckles
PER_HAND = 63 + 4                     # 21 points x (x,y,z) + 4 curl angles
FEATURE_COUNT = PER_HAND * 2 + 2      # two hands + two "hand present" flags


def _angle(a, b, c):
    v1, v2 = a - b, c - b
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    return 0.0 if n == 0 else float(np.arccos(np.clip(np.dot(v1, v2) / n, -1, 1)))


def hand_features(pts):
    """pts: (21, 3) array of normalised landmarks for ONE hand -> 67 numbers.

    Coordinates are moved so the wrist is the origin and scaled by palm size, so the result does
    not depend on how far the hand is from the camera or where it is in the frame."""
    pts = np.asarray(pts, dtype=float)
    rel = pts - pts[0]
    scale = np.linalg.norm(rel[9]) or 1.0
    rel = rel / scale
    curls = [_angle(pts[m], pts[p], pts[t]) / np.pi for m, p, t in zip(MCPS, PIPS, FINGERTIPS)]
    return np.concatenate([rel.ravel(), curls])


def two_hand_vector(hands):
    """hands: {"Left": (21,3) array or None, "Right": ...} -> FEATURE_COUNT numbers."""
    vec = []
    flags = []
    for side in ("Left", "Right"):
        h = hands.get(side)
        if h is None:
            vec.append(np.zeros(PER_HAND))
            flags.append(0.0)
        else:
            vec.append(hand_features(h))
            flags.append(1.0)
    return np.concatenate(vec + [np.array(flags)])


class FingerModel:
    def __init__(self, path=config.FINGER_MODEL):
        import joblib
        data = joblib.load(path)
        self.clf = data["model"]
        self.classes = list(self.clf.classes_)

    def predict(self, hands):
        """Return (label, probability). Label is the note-shape name used when recording."""
        x = two_hand_vector(hands).reshape(1, -1)
        proba = self.clf.predict_proba(x)[0]
        i = int(np.argmax(proba))
        return self.clf.classes_[i], float(proba[i])

    def probability_of(self, hands, label):
        x = two_hand_vector(hands).reshape(1, -1)
        proba = self.clf.predict_proba(x)[0]
        if label not in self.classes:
            return 0.0
        return float(proba[self.classes.index(label)])


def finger_hint(target_label, predicted_label, prob, min_prob=0.6):
    """Turn a prediction into a message for the player."""
    if predicted_label is None:
        return "Hands not detected. Bring both hands into view."
    if prob < min_prob:
        return "Finger position unclear. Hold your hands steady."
    if predicted_label == target_label:
        return "Finger position looks right."
    return f"Your fingers look like '{predicted_label}', but the target is '{target_label}'."


def finger_feedback(model, hands, target_label):
    """Full finger message for the lesson. Honest when the model was never shown the target note."""
    if target_label not in model.classes:
        return (f"No finger recordings for {target_label} yet, so fingers are not checked "
                f"(record it with collect_data.py).")
    if not any(v is not None for v in hands.values()):
        return finger_hint(target_label, None, 0)
    label, prob = model.predict(hands)
    return finger_hint(target_label, label, prob)
