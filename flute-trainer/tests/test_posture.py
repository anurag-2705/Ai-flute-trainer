import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from posture import PostureChecker, angle_deg, line_tilt_deg


def good_pose():
    """Front view, mirrored. Player's left wrist near mouth, right wrist further out; all visible."""
    return {
        "nose": (320, 120, 1), "left_ear": (290, 118, 1), "right_ear": (350, 118, 1),
        "left_shoulder": (260, 200, 1), "right_shoulder": (380, 200, 1),
        "left_elbow": (270, 290, 1), "right_elbow": (420, 280, 1),
        "left_wrist": (330, 160, 1), "right_wrist": (500, 170, 1),
        "left_hip": (270, 420, 1), "right_hip": (370, 420, 1),
    }


def codes(pts, **kw):
    return {i.code for i in PostureChecker(**kw).check(pts)}


def test_angle_and_tilt_math():
    assert abs(angle_deg((1, 0), (0, 0), (0, 1)) - 90) < 1e-6
    assert abs(angle_deg((1, 0), (0, 0), (-1, 0)) - 180) < 1e-6
    assert line_tilt_deg((0, 0), (100, 0)) == 0
    assert abs(line_tilt_deg((0, 0), (100, 100)) - 45) < 1e-6
    assert abs(line_tilt_deg((0, 0), (-100, 100)) - 45) < 1e-6      # direction independent
    assert abs(line_tilt_deg((0, 0), (100, -100)) + 45) < 1e-6


def test_good_pose_has_no_issues():
    assert codes(good_pose()) == set()


def test_flute_drooping_and_rising():
    p = good_pose()
    p["right_wrist"] = (500, 260, 1)
    assert "flute_droop" in codes(p)
    p["right_wrist"] = (500, 60, 1)
    assert "flute_rise" in codes(p)


def test_left_handed_flute_side_flips_which_wrist_is_far():
    p = good_pose()
    p["right_wrist"], p["left_wrist"] = (330, 160, 1), (160, 260, 1)   # flute to the player's left
    assert "flute_droop" in codes(p, flute_side="left")


def test_shoulders_and_head_tilt():
    p = good_pose()
    p["right_shoulder"] = (380, 245, 1)
    assert "shoulders" in codes(p)
    p = good_pose()
    p["right_ear"] = (350, 150, 1)
    assert "head" in codes(p)


def test_elbows():
    p = good_pose()
    p["right_elbow"] = (410, 120, 1)               # hiked above the shoulder
    assert "right_elbow_high" in codes(p)
    p = good_pose()
    p["left_wrist"] = (262, 205, 1)                # folded tight
    assert "left_elbow_tight" in codes(p)
    p = good_pose()
    p["right_elbow"] = (440, 200, 1)               # arm almost straight out
    p["right_wrist"] = (500, 200, 1)
    assert "right_elbow_straight" in codes(p)


def test_leaning_torso():
    p = good_pose()
    p["left_hip"], p["right_hip"] = (160, 420, 1), (260, 420, 1)
    assert "lean" in codes(p)


def test_low_visibility_asks_to_move_back_and_hips_are_optional():
    p = good_pose()
    p["right_wrist"] = (500, 170, 0.1)
    assert codes(p) == {"not_visible"}
    p = good_pose()
    p["left_hip"], p["right_hip"] = (0, 0, 0.1), (0, 0, 0.1)           # sitting at a desk
    assert codes(p) == set()
