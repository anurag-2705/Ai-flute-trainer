"""Flute trainer: live camera + microphone feedback.

    python app.py --lesson sa_re_ga --sa C5

Keys in the window:  q = quit   n = skip note   r = restart lesson
"""
import argparse
import time

import cv2
import numpy as np

import config
from audio import FileSource, MicSource
from fingers import FingerModel, finger_hint
from lesson import LessonSession
from pitch import PitchTracker, parse_note, sargam_label, swara_of
from posture import PostureChecker, from_mediapipe
from progress import save_results

COLORS = {"ok": (0, 200, 0), "warn": (0, 140, 255), "info": (255, 255, 255)}


def put(frame, text, org, scale=0.6, color=(255, 255, 255), thick=2):
    cv2.putText(frame, text, (org[0] + 1, org[1] + 1), cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0), thick + 2)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick)


def draw_meter(frame, cents_off, x, y, w=300):
    """Horizontal tuning meter: centre = perfect, +-50 cents at the ends."""
    cv2.rectangle(frame, (x, y), (x + w, y + 18), (60, 60, 60), -1)
    tol = int(w / 2 * config.TOLERANCE_CENTS / 50)
    cv2.rectangle(frame, (x + w // 2 - tol, y), (x + w // 2 + tol, y + 18), (0, 120, 0), -1)
    cv2.line(frame, (x + w // 2, y - 4), (x + w // 2, y + 22), (255, 255, 255), 1)
    if cents_off is not None:
        c = max(-50, min(50, cents_off))
        px = int(x + w / 2 + c / 50 * w / 2)
        cv2.circle(frame, (px, y + 9), 8, (0, 255, 255), -1)
    put(frame, "flat", (x - 40, y + 14), 0.45)
    put(frame, "sharp", (x + w + 6, y + 14), 0.45)


def compose_inset(main_frame, small, frac=0.34, margin=10):
    """Paste `small` (second camera) into the bottom-right corner of `main_frame`, with a border."""
    h, w = main_frame.shape[:2]
    iw = int(w * frac)
    ih = int(small.shape[0] * iw / small.shape[1])
    ih = min(ih, h - 2 * margin)
    iw = int(small.shape[1] * ih / small.shape[0]) if ih == h - 2 * margin else iw
    thumb = cv2.resize(small, (iw, ih))
    x0, y0 = w - iw - margin, h - ih - margin
    main_frame[y0:y0 + ih, x0:x0 + iw] = thumb
    cv2.rectangle(main_frame, (x0, y0), (x0 + iw, y0 + ih), (255, 255, 255), 2)
    return main_frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lesson", default="sa_re_ga", choices=list(config.LESSONS))
    ap.add_argument("--sa", default="C5", help="note you play as Sa / first note, e.g. C5")
    ap.add_argument("--camera", default="0", help="camera index or video file")
    ap.add_argument("--camera2", default=None,
                    help="optional second camera (e.g. phone) used for HANDS/FINGERS; "
                         "--camera is then used for posture only")
    ap.add_argument("--flute-side", default="right", choices=["right", "left"])
    ap.add_argument("--fmin", type=float, default=config.FMIN)
    ap.add_argument("--device", type=int, default=None, help="microphone device number (see check_mic.py)")
    ap.add_argument("--no-camera", action="store_true", help="sound feedback only")
    ap.add_argument("--no-fingers", action="store_true", help="skip finger model")
    ap.add_argument("--audio-file", help="play a WAV file instead of using the microphone")
    ap.add_argument("--headless", action="store_true", help="no window, print to console")
    ap.add_argument("--no-calibrate", action="store_true",
                    help="skip the 1 second quiet-room calibration at start")
    ap.add_argument("--gate", type=float, default=None,
                    help="set the 'sounds above' level by hand (e.g. 0.004). Skips calibration")
    ap.add_argument("--noise-level", type=float, default=None,
                    help="set the room noise level by hand (e.g. 0.001); the 'sounds above' level "
                         "becomes 2.5x this. Skips calibration")
    ap.add_argument("--tune-cents", type=float, default=0.0,
                    help="your flute's own tuning, e.g. 20 if it plays 20 cents sharp, -25 if flat "
                         "(the app suggests a value at the end of a lesson)")
    ap.add_argument("--log", default=None,
                    help="write a detailed per-moment log (csv) to this file, to see what happened")
    args = ap.parse_args()

    sa_midi = parse_note(args.sa)
    session = LessonSession(config.LESSONS[args.lesson], sa_midi, tune_cents=args.tune_cents)
    tracker = PitchTracker(fmin=args.fmin)
    audio = FileSource(args.audio_file) if args.audio_file else MicSource(device=args.device)
    posture = PostureChecker(args.flute_side)
    manual_levels = args.gate is not None or args.noise_level is not None
    if args.gate is not None:
        tracker.gate = args.gate
        print(f"Manual setting: sounds above {tracker.gate:.4f} count as playing.")
    elif args.noise_level is not None:
        tracker.noise_floor = args.noise_level
        tracker.gate = max(config.SILENCE_RMS, config.NOISE_MARGIN * args.noise_level)
        print(f"Manual setting: room noise {args.noise_level:.4f}; "
              f"sounds above {tracker.gate:.4f} count as playing.")

    cap = vision = finger_model = cap2 = vision2 = None
    if not args.no_camera:
        from camera import open_camera, parse_source
        src = parse_source(args.camera)
        cap = open_camera(src)
        if cap is None:
            raise SystemExit("Cannot open --camera. Run: python list_cameras.py to see which "
                             "numbers work, or use --no-camera.")
        from vision import VisionTracker
        if args.camera2 is not None:
            src2 = parse_source(args.camera2)
            if str(src2) == str(src):
                raise SystemExit("--camera and --camera2 must be different cameras.")
            cap2 = open_camera(src2)
            if cap2 is None:
                cap.release()
                raise SystemExit("Cannot open --camera2. Make sure the phone app is connected "
                                 "(Start pressed on the laptop app), then run: "
                                 "python list_cameras.py")
            vision = VisionTracker(use_pose=True, use_hands=False)    # camera 1: posture
            vision2 = VisionTracker(use_pose=False, use_hands=True)   # camera 2: fingers
        else:
            vision = VisionTracker(use_hands=True)
        if not args.no_fingers and config.FINGER_MODEL.exists():
            finger_model = FingerModel()
        elif not args.no_fingers:
            print("No finger model yet (run collect_data.py then train.py). Continuing without it.")

    log_file = log_writer = None
    if args.log:
        import csv
        log_file = open(args.log, "w", newline="")
        log_writer = csv.writer(log_file)
        log_writer.writerow(["time_s", "level", "clarity", "note_heard", "cents_from_note",
                             "midi", "target", "cents_off_target", "hold_fraction", "feedback"])
    recent = []                      # (time, midi) of the last 0.2 s, for outlier smoothing
    t0 = time.time()
    last_print = 0.0
    calibrating = isinstance(audio, MicSource) and not args.no_calibrate and not manual_levels
    fb = session.update(None, 0.0)
    pose_issues, finger_msg, frame, pr, frame2 = [], "", None, None, None
    try:
        while True:
            now = time.time() - t0
            # ---- sound ----
            chunk = audio.latest()
            if calibrating:
                # listen to the quiet room first so room noise is not mistaken for the flute
                tracker.calibrate(chunk)
                if now >= config.CALIBRATION_SECONDS:
                    gate = tracker.finish_calibration()
                    calibrating = False
                    print(f"Room noise level {tracker.noise_floor:.4f}; "
                          f"sounds above {gate:.4f} count as playing.")
                pr = None
            else:
                pr = tracker.process(chunk) if chunk is not None else None
                midi_now = pr.midi if pr else None
                if midi_now is not None:
                    recent.append((now, midi_now))
                recent[:] = [(t, m) for t, m in recent if now - t <= 0.2]
                if midi_now is not None and len(recent) >= 3:
                    # one wild reading (octave jump, click) must not break your hold
                    midi_now = float(np.median([m for _, m in recent]))
                fb = session.update(midi_now, now)
                if log_writer and pr is not None:
                    log_writer.writerow([f"{now:.3f}", f"{pr.rms:.5f}", f"{pr.confidence:.2f}",
                                         pr.name or "", "" if pr.cents is None else f"{pr.cents:.1f}",
                                         "" if midi_now is None else f"{midi_now:.2f}",
                                         session.target_label() if not session.done else "",
                                         "" if fb.cents_off is None else f"{fb.cents_off:.1f}",
                                         f"{fb.hold_fraction:.2f}", fb.text])

            # ---- camera ----
            if cap is not None:
                ok, frame = cap.read()
                if not ok:
                    break
                frame = cv2.flip(frame, 1)
                pose, hands = vision.process(frame, now * 1000)
                h, w = frame.shape[:2]
                if pose is not None:
                    pose_issues = posture.check(from_mediapipe(pose, w, h))
                    from vision import draw_pose
                    draw_pose(frame, pose)
                else:
                    pose_issues = []
                from vision import draw_hands
                if vision2 is not None:
                    ok2, frame2 = cap2.read()
                    if ok2:
                        frame2 = cv2.flip(frame2, 1)
                        _, hands = vision2.process(frame2, now * 1000)
                        draw_hands(frame2, hands)
                    else:
                        frame2 = None
                        hands = {"Left": None, "Right": None}
                else:
                    draw_hands(frame, hands)
                if finger_model is not None and not session.done:
                    if any(v is not None for v in hands.values()):
                        label, prob = finger_model.predict(hands)
                        target = sargam_label(session.steps[session.index][0])
                        finger_msg = finger_hint(target, label, prob)
                    else:
                        finger_msg = finger_hint(None, None, 0)
            elif frame is None:
                time.sleep(0.02)

            # ---- draw / print ----
            if args.headless:
                if now - last_print > 0.5:
                    last_print = now
                    print(f"[{now:5.1f}s] {fb.text}  (hold {fb.hold_fraction:.0%})")
            else:
                if frame is None:
                    canvas = np.zeros((480, 640, 3), dtype=np.uint8)
                else:
                    canvas = frame
                    if frame2 is not None:
                        compose_inset(canvas, frame2)
                if not session.done:
                    put(canvas, f"Play: {session.target_label()}", (20, 35), 1.0, (255, 255, 0), 2)
                if not calibrating:
                    put(canvas, fb.text, (20, 70), 0.7, COLORS[fb.level])
                draw_meter(canvas, fb.cents_off, 60, 100)
                # live "what I hear" readout so you can see whether sound is arriving
                vol = pr.rms if pr else 0.0
                if calibrating:
                    heard = "calibrating - stay quiet..."
                    put(canvas, "Calibrating: stay quiet for a second...", (20, 70), 0.8,
                        (0, 255, 255))
                elif pr and pr.name:
                    sw, sw_c = swara_of(pr.midi - args.tune_cents / 100.0, sa_midi)
                    heard = f"{sw} ({pr.name}) {sw_c:+.0f}c"
                elif vol < tracker.gate:
                    heard = f"too quiet ({vol:.4f} < {tracker.gate:.4f}): raise mic volume"
                else:
                    heard = f"sound heard, pitch unclear (clarity {pr.confidence:.2f})"
                bar = min(200, int((vol ** 0.5) * 600))
                cv2.rectangle(canvas, (60, 156), (60 + bar, 172), (255, 200, 0), -1)
                gx = 60 + min(200, int((tracker.gate ** 0.5) * 600))
                cv2.line(canvas, (gx, 152), (gx, 176), (0, 0, 255), 2)     # red line = minimum level
                cv2.rectangle(canvas, (60, 156), (260, 172), (200, 200, 200), 1)
                put(canvas, "mic: " + heard, (270, 169), 0.5)
                bar_w = int(300 * fb.hold_fraction)
                cv2.rectangle(canvas, (60, 130), (60 + bar_w, 142), (0, 200, 0), -1)
                cv2.rectangle(canvas, (60, 130), (360, 142), (200, 200, 200), 1)
                y = 210
                for issue in pose_issues[:4]:
                    put(canvas, "! " + issue.message, (20, y), 0.55, (0, 140, 255))
                    y += 26
                if finger_msg:
                    put(canvas, finger_msg, (20, canvas.shape[0] - 20), 0.6, (255, 255, 255))
                cv2.imshow("Flute Trainer (q quit, n skip, r restart)", canvas)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("n"):
                    session.skip(now)
                if key == ord("r"):
                    session = LessonSession(config.LESSONS[args.lesson], sa_midi)

            if session.done:
                break
            if args.audio_file and audio.finished:
                break
            time.sleep(0.01)
    finally:
        if session.results:
            save_results(args.lesson, session.results)
            print("\nResults (time counted from your first sound on each note):")
            for r in session.results:
                print(f"  {r.label:12s} {r.seconds_to_succeed:5.1f}s   avg error {r.mean_abs_cents:4.1f} cents")
            offs = [r.median_offset_cents for r in session.results if r.median_offset_cents is not None]
            if len(offs) >= 3 and abs(float(np.median(offs))) >= 10:
                m = float(np.median(offs))
                print(f"\nTip: your flute plays about {abs(m):.0f} cents {'sharp' if m > 0 else 'flat'} "
                      f"of standard tuning. Add --tune-cents {m:.0f} so correct playing is not marked wrong.")
        audio.close()
        if log_file:
            log_file.close()
            print(f"Log saved to {args.log}")
        if cap is not None:
            cap.release()
            vision.close()
        if cap2 is not None:
            cap2.release()
            vision2.close()
        if not args.headless:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
