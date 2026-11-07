# Ai Flute Trainer

A real-time flute coach. The **microphone** checks pitch, steadiness and octave. The **camera** checks
posture (flute angle, shoulders, head, elbows, torso) and, once trained, finger positions.

## 1. Install (once)

Python 3.10 to 3.13. In this folder:

```
python -m venv venv
venv\Scripts\activate            (Windows)     source venv/bin/activate   (Mac/Linux)
pip install -r requirements.txt
python download_models.py        (downloads two small MediaPipe model files, needs internet)
```

Check everything works without any hardware:

```
python -m pytest -q
```

## 2. Sound-only lesson (works immediately)

```
python app.py --no-camera --lesson sa_re_ga --sa C5
```

`--sa` is the note you play as Sa (or your first note). Western flute: `C5`. Bansuri: use the note
your bansuri actually plays with all holes closed. Lowest note below about 180 Hz (low bansuri):
add `--fmin 120`.

Lessons: `long_tone`, `sa_re_ga`, `sargam_basic` (edit them in `config.py`).
Keys: `q` quit, `n` skip note, `r` restart.

## 3. Camera + posture

```
python app.py --lesson sa_re_ga --sa C5
```

Place the laptop so the camera sees your head, both arms and shoulders (about 1.5 to 2 m away,
plain background, light in front of you). If the flute is held to your left, add `--flute-side left`.
Camera not opening? Try `--camera 1` and close Zoom/Meet.

## 4. Teach it the finger positions (the machine-learning part)

Ask a teacher (or a good player) to hold each note's fingering in front of the camera:

```
python collect_data.py --label Sa  --seconds 20
python collect_data.py --label Re  --seconds 20
python collect_data.py --label Ga  --seconds 20
```

Do **at least 3 separate runs per note**, moving slightly each time. Then:

```
python train.py
```

It prints accuracy measured on whole recording sessions it did not train on, which is the honest
number. The labels must match the lesson names (`Sa`, `Re`, `Ga`, `Ma`, `Pa`, `Dha`, `Ni`, `Sa'`).
When `models/finger_model.joblib` exists, `app.py` uses it automatically.

## 5. Progress

Each finished lesson is appended to `data/progress.csv` (date, note, seconds taken, average error in
cents).

## Files

| File | Job |
|---|---|
| `pitch.py` | YIN pitch detection, note names |
| `lesson.py` | decides if the note is right, gives feedback |
| `posture.py` | posture rules from pose landmarks |
| `vision.py` | camera + MediaPipe pose/hands |
| `fingers.py`, `collect_data.py`, `train.py` | finger-shape classifier |
| `app.py` | the live trainer |
| `config.py` | all thresholds and lessons |

## Honest limits

- Posture thresholds in `config.py` are sensible starting values, not expert truth. Have a flute
  teacher watch a session and adjust them.
- The finger model recognises hand shapes it was shown. It will not know a fingering nobody recorded.
- Sound is not fully tested on a real flute yet: tests use synthetic tones. Real breath noise can
  need small changes to `SILENCE_RMS` and `MIN_CONFIDENCE` in `config.py`.
- A camera cannot judge embouchure (lip shape) or breath control. Pitch steadiness is the proxy.

## Two cameras at once (laptop + phone)

Use the laptop camera for **posture** and a phone (as a webcam, with DroidCam / Iriun / Continuity
Camera) for **hands and fingers**. The phone view appears as a small picture in the bottom-right.

```
python app.py --sa C5 --camera 0 --camera2 1
```

`--camera` = posture camera, `--camera2` = finger camera. Try other numbers (1, 2) if the phone is not
found. Record finger data with the phone camera, in the same position you will play in:

```
python collect_data.py --label Sa --seconds 20 --camera 1
```

Placement: phone about 1 m from your hands, slightly to the side, so both hands and all fingers are
visible. Laptop 1.5 to 2 m away, seeing head, shoulders and both arms.

### Camera not found?

```
python list_cameras.py
```

It tries camera numbers 0 to 9 and shows each one. Use the number whose picture is your phone as
`--camera2`. If only one camera is listed, the phone is not connected to the laptop yet: open the phone
app on both devices and press Start/Connect first.

## The app does not react to my flute

1. Start the app and **stay quiet for the first second**. It measures room noise ("Calibrating...").
2. Watch the **mic:** line and the blue level bar under the hold bar. The red line is the minimum
   level that counts as playing. The blue bar must go past it when you play.
   - `too quiet`: raise the microphone volume, move closer, and turn OFF noise suppression /
     "voice focus" / audio enhancements in your sound settings (they remove flute sound).
   - `sound heard, pitch unclear`: play one long steady note; for low notes add `--fmin 120`.
   - a note name appears but the lesson does not move: you are playing a different note than the
     target. Change `--sa` to the note your flute plays (for example `--sa G4`).
3. Still stuck? Run `python check_mic.py --save flute.wav`, play a long note, and send the wav file.

## Making the lesson fairer for your flute

- Times are counted from your **first sound** on each note, so resting between notes does not count.
- If your flute is not at standard pitch (common with bansuri), correct playing is marked "too sharp"
  or "too flat". At the end of a lesson the app tells you, e.g. `Add --tune-cents -25`. Then run:
  ```
  python app.py --no-camera --sa C5 --tune-cents -25
  ```
- To see exactly what the app heard, moment by moment, add `--log session.csv`.

## Setting the noise levels by hand

The app normally measures room noise itself in the first second. To set it manually:

```
python app.py --no-camera --sa C5 --gate 0.004          (sounds above 0.004 count as playing)
python app.py --no-camera --sa C5 --noise-level 0.001   (room noise 0.001; gate becomes 2.5x = 0.0025)
```

Either option skips the automatic calibration. Pick the numbers from the level bar in the window or
from `python check_mic.py`: the gate should be above your quiet-room level and below your quietest
flute note.

## Swara names and a C bansuri

The app names what you play as a swara relative to Sa (`Sa Re Ga Ma Pa Dha Ni`, `Sa'` = upper Sa,
a comma after the name = lower octave, small letters = komal/flat such as `ga`, `ni`).

For a C-natural bansuri whose lower Sa is C4 (261.63 Hz), use `--sa C4`. If you play the upper Sa
(523.25 Hz) as your first note, use `--sa C5`. The equal-tempered table
Sa 261.63, Re 293.66, Ga 329.63, Ma 349.23, Pa 392.00, Dha 440.00, Ni 493.88, Sa' 523.25
is exactly what the app uses with `--sa C4`.
