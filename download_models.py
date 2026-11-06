"""Download the two MediaPipe model files (about 15 MB total). Run once: python download_models.py"""
import sys
import urllib.request

import config

BASE = "https://storage.googleapis.com/mediapipe-models/"
FILES = {
    config.POSE_MODEL: BASE + "pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    config.HAND_MODEL: BASE + "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
}


def main():
    config.MODEL_DIR.mkdir(exist_ok=True)
    failed = False
    for path, url in FILES.items():
        if path.exists() and path.stat().st_size > 1_000_000:
            print(f"already have {path.name}")
            continue
        print(f"downloading {path.name} ...")
        try:
            urllib.request.urlretrieve(url, path)
            print(f"  saved {path.stat().st_size / 1e6:.1f} MB")
        except Exception as e:
            failed = True
            print(f"  FAILED: {e}\n  Download it manually from:\n  {url}\n  and save it as {path}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
