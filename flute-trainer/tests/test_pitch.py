import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pitch import PitchTracker, freq_to_midi, midi_to_name, parse_note, sargam_label, yin
from synth import flute_tone


@pytest.mark.parametrize("note", ["C4", "D4", "G4", "A4", "C5", "E5", "A5", "C6", "G6"])
def test_yin_accuracy_across_flute_range(note):
    from pitch import midi_to_freq
    f = midi_to_freq(parse_note(note))
    frame = flute_tone(f, 0.2)[:4096]
    est, conf = yin(frame, 44100)
    assert est is not None
    cents = 1200 * np.log2(est / f)
    assert abs(cents) < 5, f"{note}: off by {cents:.1f} cents"
    assert conf > 0.9


def test_detects_flat_and_sharp_notes():
    for off in (-30, -15, 20, 40):
        f = 523.25 * 2 ** (off / 1200)
        est, _ = yin(flute_tone(f, 0.2)[:4096], 44100)
        assert abs(1200 * np.log2(est / f)) < 5


def test_tracker_names_note_and_cents():
    tr = PitchTracker()
    r = tr.process(flute_tone(587.33 * 2 ** (25 / 1200), 0.2)[:4096])   # D5, 25 cents sharp
    assert r.name == "D5"
    assert 20 < r.cents < 30


def test_silence_and_noise_give_no_note():
    tr = PitchTracker()
    assert tr.process(np.zeros(4096, dtype=np.float32)).freq is None
    rng = np.random.default_rng(1)
    noise = (0.2 * rng.standard_normal(4096)).astype(np.float32)
    assert tr.process(noise).freq is None


def test_vibrato_still_reads_close_to_centre():
    f = 659.25
    est, _ = yin(flute_tone(f, 0.2, vibrato_cents=15)[:4096], 44100)
    assert abs(1200 * np.log2(est / f)) < 25


def test_note_helpers():
    assert parse_note("C5") == 72 and parse_note("A4") == 69 and parse_note("F#4") == 66
    assert midi_to_name(72) == "C5"
    assert round(float(freq_to_midi(440.0))) == 69
    assert sargam_label(0) == "Sa" and sargam_label(12) == "Sa'" and sargam_label(7) == "Pa"
