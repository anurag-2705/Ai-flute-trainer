"""Shared settings for the flute trainer. Change numbers here, not all over the code."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"

# ---- audio ----
SAMPLE_RATE = 44100
FRAME_SIZE = 4096            # samples analysed at once (~93 ms)
FMIN = 180.0                 # lowest note we expect (Hz). Lower it for a low bansuri.
FMAX = 2300.0                # highest note we expect (Hz)
SILENCE_RMS = 0.01           # below this volume we treat it as "not playing"
MIN_CONFIDENCE = 0.80        # YIN confidence needed to trust a pitch

# ---- notes ----
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
# Sargam names for semitone offsets from Sa (lowercase = komal/flat, MA = tivra/sharp)
SARGAM = ["Sa", "re", "Re", "ga", "Ga", "Ma", "MA", "Pa", "dha", "Dha", "ni", "Ni"]

# ---- lessons: list of (semitone offset from Sa, seconds to hold) ----
LESSONS = {
    "long_tone": [(0, 4.0), (0, 4.0), (0, 4.0)],
    "sa_re_ga": [(0, 1.5), (2, 1.5), (4, 1.5), (2, 1.5), (0, 1.5)],
    "sargam_basic": [(0, 1.5), (2, 1.5), (4, 1.5), (5, 1.5),
                     (7, 1.5), (9, 1.5), (11, 1.5), (12, 1.5)],
}

# ---- lesson judging ----
TOLERANCE_CENTS = 35         # how close counts as "in tune" (100 cents = 1 semitone)
DROPOUT_GRACE = 0.25         # seconds of breathing/gap allowed without losing progress
SETTLE_SECONDS = 0.35        # ignore sound briefly after a note is completed
STEADY_STD_CENTS = 20       # wobble above this gets a "hold steady" tip

# ---- posture limits (degrees unless noted) ----
POSTURE = {
    "flute_droop_max": 20,       # far end of flute lower than near end by more than this
    "flute_rise_max": 10,        # far end higher than near end by more than this
    "shoulder_tilt_max": 8,
    "head_tilt_max": 10,
    "elbow_angle_min": 40,       # far arm folded too tightly
    "near_elbow_angle_min": 15,  # arm nearest the mouth is naturally folded (2D view makes it look tighter)
    "elbow_angle_max": 165,      # arm almost straight
    "elbow_over_shoulder": 0.25, # elbow this many shoulder-widths above shoulder
    "torso_lean_max": 12,
    "min_visibility": 0.5,
}

# ---- model files (downloaded by download_models.py) ----
POSE_MODEL = MODEL_DIR / "pose_landmarker_lite.task"
HAND_MODEL = MODEL_DIR / "hand_landmarker.task"
FINGER_MODEL = MODEL_DIR / "finger_model.joblib"
FINGER_CSV = DATA_DIR / "fingers.csv"
PROGRESS_CSV = DATA_DIR / "progress.csv"
