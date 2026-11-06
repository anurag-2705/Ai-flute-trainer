"""Synthetic data for tests: fake flute audio and fake hand landmarks."""
import wave

import numpy as np


def flute_tone(freq, seconds, sr=44100, vibrato_cents=0.0, noise=0.01, amp=0.3, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * sr)) / sr
    f = freq * 2 ** ((vibrato_cents / 1200) * np.sin(2 * np.pi * 5 * t))
    phase = 2 * np.pi * np.cumsum(f) / sr
    sig = np.sin(phase) + 0.3 * np.sin(2 * phase) + 0.1 * np.sin(3 * phase)
    sig = amp * sig / 1.4 + noise * rng.standard_normal(len(t))
    return sig.astype(np.float32)


def write_wav(path, samples, sr=44100):
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes((np.clip(samples, -1, 1) * 32767).astype(np.int16).tobytes())


def fake_hand(curled, rng, rotate=0.0, scale=1.0, shift=(0.5, 0.5), noise=0.004):
    """21 landmarks. `curled` = 4-tuple of bools for index, middle, ring, little finger."""
    pts = np.zeros((21, 3))
    pts[0] = (0, 0, 0)
    # thumb
    for i, p in enumerate([(0.3, -0.2), (0.5, -0.4), (0.65, -0.55), (0.8, -0.65)], start=1):
        pts[i] = (*p, 0)
    xs = [0.25, 0.0, -0.22, -0.42]
    for f, (x, c) in enumerate(zip(xs, curled)):
        base = 5 + 4 * f
        pts[base] = (x, -1.0, 0)
        if not c:   # straight finger
            for k in range(1, 4):
                pts[base + k] = (x, -1.0 - 0.45 * k, 0)
        else:       # curled finger: folds back toward the palm
            pts[base + 1] = (x, -1.35, 0.1)
            pts[base + 2] = (x, -1.2, 0.3)
            pts[base + 3] = (x, -0.95, 0.35)
    pts += rng.normal(0, noise, pts.shape)
    c, s = np.cos(rotate), np.sin(rotate)
    rot = pts.copy()
    rot[:, 0] = pts[:, 0] * c - pts[:, 1] * s
    rot[:, 1] = pts[:, 0] * s + pts[:, 1] * c
    return rot * scale * 0.2 + np.array([*shift, 0.0])
