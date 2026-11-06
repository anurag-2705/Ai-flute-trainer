import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lesson import LessonSession

SA = 72  # C5


def run(session, midi, start, seconds, dt=0.05):
    fb, t = None, start
    while t < start + seconds:
        fb = session.update(midi, t)
        t += dt
    return fb, t


def test_holding_correct_note_advances():
    s = LessonSession([(0, 1.0), (2, 1.0)], SA)
    fb, t = run(s, 72.0, 0, 1.3)
    assert s.index == 1 and len(s.results) == 1
    fb, t = run(s, 74.05, t, 1.3)
    assert s.done and fb.done


def test_wrong_note_never_advances_and_explains():
    s = LessonSession([(0, 1.0)], SA)
    fb, _ = run(s, 74.0, 0, 3)
    assert s.index == 0
    assert "D5" in fb.text and "Sa" in fb.text


def test_flat_and_sharp_messages():
    s = LessonSession([(0, 1.0)], SA)
    fb, _ = run(s, 72.6, 0, 0.5)
    assert "sharp" in fb.text.lower()
    s2 = LessonSession([(0, 1.0)], SA)
    fb, _ = run(s2, 71.5, 0, 0.5)
    assert "flat" in fb.text.lower()


def test_octave_error_detected():
    s = LessonSession([(0, 1.0)], SA)
    fb, _ = run(s, 84.0, 0, 0.5)
    assert "octave" in fb.text.lower() and "too high" in fb.text.lower()
    fb, _ = run(LessonSession([(0, 1.0)], SA), 60.0, 0, 0.5)
    assert "too low" in fb.text.lower()


def test_short_breath_gap_is_forgiven_but_long_gap_resets():
    s = LessonSession([(0, 2.0)], SA)
    _, t = run(s, 72.0, 0, 1.0)
    _, t = run(s, None, t, 0.15)           # quick breath
    fb, t = run(s, 72.0, t, 1.2)
    assert s.done
    s = LessonSession([(0, 2.0)], SA)
    _, t = run(s, 72.0, 0, 1.0)
    _, t = run(s, None, t, 0.6)            # long gap
    fb, t = run(s, 72.0, t, 1.0)
    assert not s.done


def test_wobble_warning():
    s = LessonSession([(0, 5.0)], SA)
    t, fb = 0.0, None
    for i in range(30):
        fb = s.update(72.0 + (0.3 if i % 2 else -0.3), t)
        t += 0.05
    assert "wobbl" in fb.text.lower()


def test_skip_and_results_recorded():
    s = LessonSession([(0, 0.5), (4, 0.5)], SA)
    run(s, 72.0, 0, 0.8)
    s.skip(1.0)
    assert s.done and s.results[0].label.startswith("Sa")
