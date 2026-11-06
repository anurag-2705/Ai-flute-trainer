import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from fingers import FEATURE_COUNT, FingerModel, finger_hint, hand_features, two_hand_vector
from synth import fake_hand
from train import train

# three fake "notes", each a different pattern of curled fingers on the left hand
NOTES = {"Sa": (True, True, True, True), "Re": (False, True, True, True),
         "Ga": (False, False, True, True), "Ma": (False, False, False, True)}


def make_csv(path, sessions=4, per_session=60, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(sessions):
        rot, sc = rng.uniform(-0.4, 0.4), rng.uniform(0.7, 1.4)
        shift = rng.uniform(0.3, 0.7, 2)
        for label, curled in NOTES.items():
            for _ in range(per_session):
                hands = {"Left": fake_hand(curled, rng, rot, sc, shift), "Right": None}
                rows.append(list(two_hand_vector(hands)) + [label, f"s{s}"])
    cols = [f"f{i}" for i in range(FEATURE_COUNT)] + ["label", "session"]
    pd.DataFrame(rows, columns=cols).to_csv(path, index=False)


def test_features_ignore_position_and_size():
    rng = np.random.default_rng(0)
    a = hand_features(fake_hand(NOTES["Re"], rng, 0, 1.0, (0.3, 0.3), noise=0))
    b = hand_features(fake_hand(NOTES["Re"], rng, 0, 1.8, (0.7, 0.6), noise=0))
    assert np.allclose(a, b, atol=1e-6)


def test_feature_vector_length_and_presence_flags():
    rng = np.random.default_rng(0)
    v = two_hand_vector({"Left": fake_hand(NOTES["Sa"], rng), "Right": None})
    assert len(v) == FEATURE_COUNT and list(v[-2:]) == [1.0, 0.0]


def test_train_and_predict_on_unseen_session(tmp_path):
    csv, model = tmp_path / "f.csv", tmp_path / "m.joblib"
    make_csv(csv)
    acc = train(csv, model, verbose=False)
    assert acc is not None and acc > 0.95
    fm = FingerModel(model)
    rng = np.random.default_rng(99)                       # brand-new "session"
    ok = 0
    for label, curled in NOTES.items():
        for _ in range(20):
            hands = {"Left": fake_hand(curled, rng, 0.2, 1.1, (0.4, 0.5)), "Right": None}
            ok += fm.predict(hands)[0] == label
    assert ok / (20 * len(NOTES)) > 0.95
    assert fm.probability_of({"Left": fake_hand(NOTES["Sa"], rng), "Right": None}, "Sa") > 0.5


def test_hint_messages():
    assert "right" in finger_hint("Sa", "Sa", 0.9)
    assert "'Re'" in finger_hint("Sa", "Re", 0.9) and "'Sa'" in finger_hint("Sa", "Re", 0.9)
    assert "unclear" in finger_hint("Sa", "Re", 0.3)
    assert "not detected" in finger_hint("Sa", None, 0)
