"""Two-camera mode, tested with fake cameras and fake trackers (no hardware or models needed)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import app
import cv2
import vision
from synth import write_wav

CALLS = []


class FakeCap:
    """Camera 0 gives dark frames (value 10), camera 1 gives bright frames (value 200)."""
    def __init__(self, src):
        self.src = src
        self.ok = src != 9          # camera 9 = "does not exist"
        self.released = False

    def isOpened(self):
        return self.ok

    def read(self):
        return True, np.full((240, 320, 3), 10 if self.src == 0 else 200, dtype=np.uint8)

    def release(self):
        self.released = True


class FakeTracker:
    def __init__(self, use_hands=True, use_pose=True):
        self.use_hands, self.use_pose = use_hands, use_pose

    def process(self, frame, ts):
        CALLS.append((self.use_pose, self.use_hands, int(frame.mean())))
        return None, {"Left": None, "Right": None}

    def close(self):
        pass


@pytest.fixture
def silent_wav(tmp_path):
    p = tmp_path / "silence.wav"
    write_wav(p, np.zeros(int(44100 * 0.8), dtype=np.float32))
    return p


def run_app(monkeypatch, argv, tmp_path):
    CALLS.clear()
    monkeypatch.setattr(cv2, "VideoCapture", FakeCap)
    monkeypatch.setattr(vision, "VisionTracker", FakeTracker)
    monkeypatch.setattr("config.DATA_DIR", tmp_path)
    monkeypatch.setattr(sys, "argv", ["app.py", "--headless", "--sa", "C5"] + argv)
    app.main()


def test_each_camera_feeds_the_right_tracker(monkeypatch, tmp_path, silent_wav):
    run_app(monkeypatch, ["--audio-file", str(silent_wav), "--camera", "0", "--camera2", "1"],
            tmp_path)
    assert CALLS, "trackers never ran"
    pose_calls = {c for c in CALLS if c[0]}
    hand_calls = {c for c in CALLS if c[1]}
    assert pose_calls == {(True, False, 10)}        # posture tracker only sees camera 0
    assert hand_calls == {(False, True, 200)}       # hand tracker only sees camera 1


def test_single_camera_mode_unchanged(monkeypatch, tmp_path, silent_wav):
    run_app(monkeypatch, ["--audio-file", str(silent_wav), "--camera", "0"], tmp_path)
    assert {c[:2] for c in CALLS} == {(True, True)}


def test_same_camera_twice_is_rejected(monkeypatch, tmp_path, silent_wav):
    with pytest.raises(SystemExit, match="different"):
        run_app(monkeypatch, ["--audio-file", str(silent_wav), "--camera", "0", "--camera2", "0"],
                tmp_path)


def test_missing_second_camera_gives_clear_error(monkeypatch, tmp_path, silent_wav):
    with pytest.raises(SystemExit, match="camera2"):
        run_app(monkeypatch, ["--audio-file", str(silent_wav), "--camera", "0", "--camera2", "9"],
                tmp_path)


def test_inset_is_pasted_bottom_right_and_does_not_resize_main():
    main = np.zeros((480, 640, 3), dtype=np.uint8)
    small = np.full((720, 1280, 3), 200, dtype=np.uint8)       # phone-style 16:9 frame
    out = app.compose_inset(main, small)
    assert out.shape == (480, 640, 3)
    assert out[470 - 5, 630 - 5].tolist() == [200, 200, 200]   # inside the inset
    assert out[20, 20].tolist() == [0, 0, 0]                   # top-left untouched


def test_inset_handles_tall_portrait_phone_frames():
    main = np.zeros((480, 640, 3), dtype=np.uint8)
    small = np.full((1280, 720, 3), 120, dtype=np.uint8)       # phone held upright
    out = app.compose_inset(main, small)
    assert out.shape == (480, 640, 3)
