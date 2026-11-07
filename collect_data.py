"""Record hand-shape examples for one note.

Example (teacher holds the Sa fingering for 20 seconds):
    python collect_data.py --label Sa --seconds 20

Run it several times per note (at least 3 separate runs, slightly moving the camera/hands each
time). Separate runs let train.py measure honestly how well the model works on new sessions.
"""
import argparse
import csv
import time

import cv2

import config
from fingers import FEATURE_COUNT, two_hand_vector
from vision import VisionTracker, draw_hands


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, help="note-shape name, e.g. Sa, Re, Ga")
    ap.add_argument("--seconds", type=float, default=20)
    ap.add_argument("--countdown", type=float, default=5)
    ap.add_argument("--camera", default="0")
    args = ap.parse_args()

    from camera import open_camera, parse_source
    cap = open_camera(parse_source(args.camera))
    if cap is None:
        raise SystemExit("Cannot open camera. Run: python list_cameras.py to see which numbers "
                         "work, and close other apps using the camera.")
    tracker = VisionTracker(use_hands=True)
    session = time.strftime("%Y%m%d-%H%M%S")
    config.DATA_DIR.mkdir(exist_ok=True)
    new_file = not config.FINGER_CSV.exists()
    t0 = time.time()
    rows = 0

    with open(config.FINGER_CSV, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow([f"f{i}" for i in range(FEATURE_COUNT)] + ["label", "session"])
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            now = time.time() - t0
            _, hands = tracker.process(frame, now * 1000)
            draw_hands(frame, hands)
            if now < args.countdown:
                msg = f"Get ready: '{args.label}' in {args.countdown - now:.0f}"
                color = (0, 200, 255)
            elif now < args.countdown + args.seconds:
                if any(v is not None for v in hands.values()):
                    w.writerow(list(two_hand_vector(hands)) + [args.label, session])
                    rows += 1
                msg = f"RECORDING '{args.label}'  frames: {rows}"
                color = (0, 0, 255)
            else:
                break
            cv2.putText(frame, msg, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.imshow("Collect finger data (q to stop)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cap.release()
    cv2.destroyAllWindows()
    tracker.close()
    print(f"Saved {rows} frames for '{args.label}' to {config.FINGER_CSV}")


if __name__ == "__main__":
    main()
