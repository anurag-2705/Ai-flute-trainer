"""Open a camera robustly. Phone-as-webcam apps often fail with OpenCV's default backend on Windows,
so we retry with the other backends before giving up."""
import sys

import cv2


def _backends():
    if sys.platform.startswith("win"):
        return [None, getattr(cv2, "CAP_DSHOW", None), getattr(cv2, "CAP_MSMF", None)]
    if sys.platform == "darwin":
        return [None, getattr(cv2, "CAP_AVFOUNDATION", None)]
    return [None, getattr(cv2, "CAP_V4L2", None)]


def open_camera(src):
    """src: int index or file path. Returns an opened cv2.VideoCapture that really delivers
    frames, or None."""
    is_index = isinstance(src, int)
    for backend in (_backends() if is_index else [None]):
        try:
            cap = cv2.VideoCapture(src) if backend is None else cv2.VideoCapture(src, backend)
        except Exception:
            continue
        if cap is not None and cap.isOpened():
            ok, _ = cap.read()           # some virtual cameras open but never give a picture
            if ok:
                return cap
        if cap is not None:
            cap.release()
    return None


def parse_source(text):
    return int(text) if str(text).isdigit() else text
