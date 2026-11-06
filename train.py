"""Train the finger-shape classifier from data/fingers.csv.

    python train.py
"""
import argparse

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GroupKFold, cross_val_predict

import config


def train(csv_path=config.FINGER_CSV, out_path=config.FINGER_MODEL, verbose=True):
    df = pd.read_csv(csv_path)
    feats = [c for c in df.columns if c.startswith("f")]
    X, y, groups = df[feats].to_numpy(), df["label"].astype(str).to_numpy(), df["session"].astype(str)
    labels = sorted(set(y))
    if len(labels) < 2:
        raise SystemExit("Need recordings of at least 2 different notes to train.")

    new_clf = lambda: RandomForestClassifier(n_estimators=200, min_samples_leaf=2,
                                             class_weight="balanced", random_state=0, n_jobs=-1)
    n_sessions = groups.nunique()
    score = None
    if n_sessions >= 3:
        # hold out whole recording sessions: consecutive frames look alike, so a random split
        # would give a falsely high score
        folds = min(5, n_sessions)
        pred = cross_val_predict(new_clf(), X, y, groups=groups, cv=GroupKFold(n_splits=folds))
        score = float(np.mean(pred == y))
        if verbose:
            print(f"Cross-session accuracy ({folds} folds): {score:.1%}")
            print(classification_report(y, pred, zero_division=0))
    elif verbose:
        print("WARNING: fewer than 3 recording sessions, so accuracy cannot be measured honestly.\n"
              "Record each note in at least 3 separate runs of collect_data.py.")

    clf = new_clf().fit(X, y)
    config.MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump({"model": clf, "labels": labels, "cv_accuracy": score}, out_path)
    if verbose:
        print(f"Trained on {len(df)} frames, notes: {labels}\nSaved model to {out_path}")
    return score


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    train()
