"""Lesson logic: decides if the played note matches the target and gives spoken-style feedback.
No camera or audio code in here, so it is easy to test."""
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

import config
from pitch import midi_to_name, sargam_label, swara_of


@dataclass
class Feedback:
    text: str
    level: str                    # "ok", "warn", "info"
    cents_off: Optional[float]    # signed distance from target, None if silent
    hold_fraction: float          # 0..1 progress of the current hold
    step_index: int
    done: bool = False


@dataclass
class StepResult:
    label: str
    seconds_to_succeed: float          # from your FIRST sound on this note, not from when it appeared
    mean_abs_cents: float
    median_offset_cents: Optional[float] = None   # typical signed error vs the nominal note


class LessonSession:
    def __init__(self, steps, sa_midi, tol_cents=config.TOLERANCE_CENTS, tune_cents=0.0):
        self.steps = list(steps)
        self.sa_midi = sa_midi
        self.tol = tol_cents
        self.tune = tune_cents     # your flute's own tuning: +20 means it plays 20 cents sharp
        self.index = 0
        self.results: List[StepResult] = []
        self._reset_step(None)

    # ----- helpers -----
    def _reset_step(self, now):
        self._hold_start = None
        self._last_good = None
        self._step_start = None          # set at the first sound, so waiting time is not counted
        self._errors: List[float] = []
        self._near: List[float] = []     # signed errors of near-miss frames (for tuning advice)
        self._recent = deque()          # (time, midi) for steadiness check
        # just after a success the audio buffer still holds the old note, so ignore it briefly
        self._settle_until = (now + config.SETTLE_SECONDS) if now is not None and self.index else 0.0

    @property
    def done(self):
        return self.index >= len(self.steps)

    def target_midi(self):
        return self.sa_midi + self.steps[self.index][0]

    def target_label(self):
        off = self.steps[self.index][0]
        return f"{sargam_label(off)} ({midi_to_name(self.target_midi())})"

    def skip(self, now=None):
        if not self.done:
            self.index += 1
            self._reset_step(now)

    # ----- main update -----
    def update(self, midi, now) -> Feedback:
        """midi = fractional MIDI number of the sound being played, or None for silence."""
        if self.done:
            return Feedback("Lesson complete. Well done!", "ok", None, 1.0, self.index, True)

        offset, hold = self.steps[self.index]
        target = self.target_midi() + self.tune / 100.0

        if now < self._settle_until:
            return Feedback("Good! Next note.", "ok", None, 0.0, self.index)

        if midi is None:
            # allow a short gap (breath) without losing the hold
            if self._hold_start is not None and now - self._last_good > config.DROPOUT_GRACE:
                self._hold_start = None
            frac = self._fraction(now, hold)
            return Feedback(f"Play {self.target_label()}", "info", None, frac, self.index)

        if self._step_start is None:
            self._step_start = now
        cents_off = (midi - target) * 100.0
        raw_off = (midi - self.target_midi()) * 100.0
        if abs(raw_off) <= 60:
            self._near.append(raw_off)
        self._recent.append((now, midi))
        while self._recent and now - self._recent[0][0] > 0.5:
            self._recent.popleft()

        if abs(cents_off) <= self.tol:
            if self._hold_start is None:
                self._hold_start = now
            self._last_good = now
            self._errors.append(abs(cents_off))
            frac = self._fraction(now, hold)
            if now - self._hold_start >= hold:
                self.results.append(StepResult(
                    self.target_label(), now - self._step_start,
                    float(np.mean(self._errors)) if self._errors else 0.0,
                    float(np.median(self._near)) if self._near else None))
                self.index += 1
                self._reset_step(now)
                if self.done:
                    return Feedback("Lesson complete. Well done!", "ok", cents_off, 1.0,
                                    self.index, True)
                return Feedback("Good! Next note.", "ok", cents_off, 0.0, self.index)
            if self._wobble() > config.STEADY_STD_CENTS:
                return Feedback("In tune, but wobbling. Keep your breath steady.", "warn",
                                cents_off, frac, self.index)
            return Feedback("In tune. Hold it!", "ok", cents_off, frac, self.index)

        # not in tune: reset the hold and explain what is wrong
        self._hold_start = None
        return Feedback(self._explain(midi, target, cents_off), "warn", cents_off, 0.0, self.index)

    def _fraction(self, now, hold):
        if self._hold_start is None:
            return 0.0
        return min(1.0, (now - self._hold_start) / hold)

    def _wobble(self):
        if len(self._recent) < 4:
            return 0.0
        return float(np.std([m * 100.0 for _, m in self._recent]))

    def _explain(self, midi, target, cents_off):
        semis = midi - target
        near_octave = round(semis / 12.0)
        if near_octave != 0 and abs(semis - 12 * near_octave) * 100 <= self.tol:
            return "Right note, wrong octave: " + ("too high, use less air speed."
                                                    if near_octave > 0 else
                                                    "too low, blow a little faster.")
        if abs(cents_off) <= 100:
            if cents_off > 0:
                return f"Too sharp by {abs(cents_off):.0f} cents. Roll the flute slightly outward."
            return f"Too flat by {abs(cents_off):.0f} cents. Roll the flute slightly inward."
        played = swara_of(midi - self.tune / 100.0, self.sa_midi)[0]
        return (f"You played {played} ({midi_to_name(midi)}). Target is {self.target_label()}. "
                f"Check your fingers.")
