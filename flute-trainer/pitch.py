"""Pitch detection (YIN algorithm) and note naming. Pure numpy/scipy, no heavy dependencies."""
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.signal import fftconvolve

import config


def yin(x, sr, fmin=config.FMIN, fmax=config.FMAX, threshold=0.15):
    """Return (frequency_hz or None, confidence 0..1) for one audio frame."""
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    tau_max = int(sr / fmin)
    tau_min = max(2, int(sr / fmax))
    w = len(x) - tau_max
    if w < tau_max:
        raise ValueError("Frame too short for this fmin; use a longer frame or higher fmin")

    # difference function d(tau) computed with an FFT
    cs = np.concatenate(([0.0], np.cumsum(x * x)))
    taus = np.arange(tau_max + 1)
    energy_shifted = cs[taus + w] - cs[taus]
    cross = fftconvolve(x, x[:w][::-1], mode="valid")        # length tau_max + 1
    d = cs[w] + energy_shifted - 2.0 * cross
    d[0] = 0.0

    # cumulative mean normalised difference
    cmnd = np.ones_like(d)
    running = np.cumsum(d[1:])
    cmnd[1:] = d[1:] * np.arange(1, tau_max + 1) / np.maximum(running, 1e-12)

    tau = None
    t = tau_min
    while t < tau_max:
        if cmnd[t] < threshold:
            while t + 1 < tau_max and cmnd[t + 1] < cmnd[t]:
                t += 1
            tau = t
            break
        t += 1
    if tau is None:
        return None, 0.0

    a, b, c = cmnd[tau - 1], cmnd[tau], cmnd[tau + 1]
    denom = a - 2 * b + c
    shift = 0.5 * (a - c) / denom if denom != 0 else 0.0
    return sr / (tau + shift), float(1.0 - cmnd[tau])


# ---------- note helpers ----------
def freq_to_midi(freq):
    return 69.0 + 12.0 * np.log2(freq / 440.0)


def midi_to_freq(midi):
    return 440.0 * 2.0 ** ((midi - 69.0) / 12.0)


def midi_to_name(midi):
    m = int(round(midi))
    return f"{config.NOTE_NAMES[m % 12]}{m // 12 - 1}"


def parse_note(name):
    """'C5' -> 72, 'F#4' -> 66, 'Bb3' -> 58"""
    name = name.strip()
    letter = name[0].upper()
    base = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}[letter]
    rest = name[1:]
    if rest.startswith("#"):
        base += 1
        rest = rest[1:]
    elif rest.startswith("b"):
        base -= 1
        rest = rest[1:]
    return (int(rest) + 1) * 12 + base


def sargam_label(offset):
    return config.SARGAM[offset % 12] + ("'" if offset >= 12 else "")


@dataclass
class PitchResult:
    freq: Optional[float]      # Hz, None if no clear note
    confidence: float
    rms: float
    midi: Optional[float]      # fractional MIDI number
    name: Optional[str]        # nearest note name e.g. "D5"
    cents: Optional[float]     # distance from nearest note, -50..+50


class PitchTracker:
    def __init__(self, sr=config.SAMPLE_RATE, fmin=config.FMIN, fmax=config.FMAX):
        self.sr, self.fmin, self.fmax = sr, fmin, fmax

    def process(self, frame) -> PitchResult:
        rms = float(np.sqrt(np.mean(np.square(frame)))) if len(frame) else 0.0
        if rms < config.SILENCE_RMS:
            return PitchResult(None, 0.0, rms, None, None, None)
        freq, conf = yin(frame, self.sr, self.fmin, self.fmax)
        if freq is None or conf < config.MIN_CONFIDENCE:
            return PitchResult(None, conf, rms, None, None, None)
        midi = float(freq_to_midi(freq))
        return PitchResult(freq, conf, rms, midi, midi_to_name(midi),
                           (midi - round(midi)) * 100.0)
