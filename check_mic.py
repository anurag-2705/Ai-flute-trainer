"""Microphone check and recorder.

    python check_mic.py                    lists devices, listens 6 seconds, tells you what is wrong
    python check_mic.py --device 2         use a specific input device number
    python check_mic.py --save flute.wav   ALSO saves what the microphone heard (send it to me!)

Stay quiet for the first 1.5 seconds (it measures room noise), then play a steady flute note.
"""
import argparse
import time
import wave

import numpy as np
import sounddevice as sd

import config
from pitch import PitchTracker


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=None)
    ap.add_argument("--seconds", type=float, default=6)
    ap.add_argument("--save", default=None, help="save the recording to this .wav file")
    args = ap.parse_args()

    print("Input devices (number: name):")
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            mark = "  <- default" if i == sd.default.device[0] else ""
            print(f"  {i}: {d['name']}{mark}")

    tracker = PitchTracker()
    buf = []

    def cb(indata, frames, t, status):
        buf.extend(indata[:, 0])

    print(f"\nStay QUIET for 1.5 s, then play a steady note. Total {args.seconds:.0f} s...\n")
    t_start = time.time()
    calibrated = False
    heard_note = False
    peak_level = 0.0
    with sd.InputStream(samplerate=config.SAMPLE_RATE, channels=1, dtype="float32",
                        device=args.device, callback=cb):
        while time.time() - t_start < args.seconds:
            time.sleep(0.3)
            if len(buf) < config.FRAME_SIZE:
                continue
            frame = np.array(buf[-config.FRAME_SIZE:], dtype=np.float32)
            elapsed = time.time() - t_start
            if elapsed < 1.5:
                tracker.calibrate(frame)
                print("calibrating (stay quiet)...")
                continue
            if not calibrated:
                gate = tracker.finish_calibration()
                calibrated = True
                print(f"room noise {tracker.noise_floor:.4f} -> sounds above {gate:.4f} count as playing\n")
            r = tracker.process(frame)
            peak_level = max(peak_level, r.rms)
            heard_note = heard_note or r.name is not None
            bars = "#" * min(40, int((r.rms ** 0.5) * 120))
            heard = f"{r.name} ({r.cents:+.0f} cents)" if r.name else f"no note (clarity {r.confidence:.2f})"
            print(f"level {r.rms:.4f} |{bars:<40}| {heard}")

    if args.save and buf:
        data = (np.clip(np.array(buf), -1, 1) * 32767).astype(np.int16)
        with wave.open(args.save, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(config.SAMPLE_RATE)
            wf.writeframes(data.tobytes())
        print(f"\nSaved recording to {args.save}")

    print("\nDiagnosis:")
    if not buf:
        print("  No audio arrived. Wrong device, or microphone permission is blocked.")
    elif np.sqrt(np.mean(np.square(buf))) < 0.0003:
        print("  Almost completely silent: the mic is muted/blocked, or this is the wrong device.")
        print("  Windows: Settings > System > Sound > Input, pick the right mic and raise volume.")
    elif heard_note:
        print("  Works: the flute note was recognised. If the app still ignores you, run it again")
        print("  and watch the 'mic:' line in its window.")
    elif peak_level < tracker.gate * 1.5:
        print("  Sound is too quiet compared with the room noise. Move closer to the laptop, raise")
        print("  the mic volume, and turn OFF noise suppression / 'voice focus' / audio enhancements")
        print("  (they treat a flute as noise and remove it).")
    else:
        print("  Sound arrives but no steady pitch was found. Play one long steady note, closer to")
        print("  the mic, and avoid very breathy playing. If the note is below about 180 Hz use")
        print("  --fmin 120 in app.py. Try --save flute.wav and send me the file.")


if __name__ == "__main__":
    main()
