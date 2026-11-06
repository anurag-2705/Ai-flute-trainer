"""Microphone check. Run:  python check_mic.py            (lists devices, then listens 5 seconds)
                          python check_mic.py --device 2  (use a specific input device number)
Play a steady flute note while it listens."""
import argparse
import time

import numpy as np
import sounddevice as sd

import config
from pitch import PitchTracker


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=None)
    ap.add_argument("--seconds", type=float, default=5)
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
    print(f"\nListening for {args.seconds:.0f} s. Play a steady note now...\n")
    with sd.InputStream(samplerate=config.SAMPLE_RATE, channels=1, dtype="float32",
                        device=args.device, callback=cb):
        end = time.time() + args.seconds
        while time.time() < end:
            time.sleep(0.4)
            if len(buf) >= config.FRAME_SIZE:
                frame = np.array(buf[-config.FRAME_SIZE:], dtype=np.float32)
                r = tracker.process(frame)
                bars = "#" * min(40, int(r.rms * 400))
                heard = f"{r.name} ({r.cents:+.0f} cents)" if r.name else "no clear note"
                print(f"volume {r.rms:.3f} |{bars:<40}| {heard}")
    peak = np.sqrt(np.mean(np.square(buf))) if buf else 0
    print("\nDiagnosis:")
    if not buf:
        print("  No audio arrived. Wrong device, or microphone permission is blocked.")
    elif peak < 0.002:
        print("  Almost silent. Unmute the mic, raise input volume, or pick another --device.")
    elif peak < config.SILENCE_RMS:
        print(f"  Too quiet (need above {config.SILENCE_RMS}). Raise mic volume or play closer.")
    else:
        print("  Microphone works. If notes still show 'no clear note', the pitch may be outside")
        print("  the range: try --fmin 120 in app.py for low notes, and play a steadier note.")


if __name__ == "__main__":
    main()
