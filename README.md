# Optxt

Reads another person's gestures and facial expressions from a webcam and says
them out loud, so a blind or low-vision user can pick up the social cues that
sighted people read automatically.

**Live: https://optxt.vercel.app** — runs entirely in the browser. No video,
landmarks or audio ever leave the device.

## What it recognises

| | |
|---|---|
| **Gestures** | thumbs up, thumbs down, pointing, middle finger, waving, neutral |
| **Emotions** | happy, sad, angry, shocked, neutral |

It speaks only when a reading has held steady for several frames, and stays
quiet when it isn't confident.

## Accuracy

Measured with **grouped cross-validation** — whole recordings are held out, so
no take ever appears in both training and test. This matters: an earlier version
split randomly by frame, which put near-identical frames on both sides and
reported a meaningless 100%.

| | Gesture | Emotion |
|---|---|---|
| Grouped 3-fold CV | **97.7%** (±1.3) | **93.0%** (±3.4) |
| Held-out recordings | 99.0% | 97.6% |
| Classes | 6 | 5 |
| Training data | 3,263 frames / 47 recordings | 2,759 frames / 39 recordings |

## How it works

```
webcam frame
  → MediaPipe (face mesh, pose, hands) extracts landmarks
  → normalized geometric features
  → ExtraTrees / RandomForest classifier
  → temporal smoothing, then speech
```

Features are **normalized so they describe what a body is doing, not where it
is**. Gestures are measured from the shoulder midpoint in shoulder-width units;
expressions from between the eyes in interocular units, rotated to the eye line.
That makes them invariant to position, camera distance and head tilt. Raw pixel
coordinates only work from the exact spot the data was recorded in.

No deep learning: with a few hundred samples per class a neural net overfits,
while a tree ensemble trains in seconds and runs at 30+ FPS on a laptop.

## Running it

**Desktop (Python):**

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt      # plus requirements-windows.txt on Windows
python main.py                       # q quit · m mute · s toggle wireframe
```

**Web:**

```bash
cd web && python -m http.server 8777
```

**Collecting data and retraining:**

```bash
python training/collect.py     # guided capture; move between rounds
python training/train.py       # grouped validation, picks the best model
python training/export_onnx.py # re-export for the web build
```

## Structure

```
main.py  landmarks.py  features.py  gestures.py  emotions.py
state_tracker.py  speech.py  overlay.py  predict.py
training/    collect.py, train.py, export_onnx.py, verify_*.py, models/, data/
web/         client-side build: MediaPipe JS + ONNX, deployed to Vercel
CONTEXT.md   full technical write-up
```

The web build duplicates three Python modules in JavaScript, so three tests
check they haven't drifted — ONNX against scikit-learn, `features.js` against
`features.py`, and the whole pipeline end to end. Currently **2150/2150 frames
identical**. See `web/README.md`.

## Limitations

- All training recordings are of one person; generalisation to other faces and
  body types is untested and is the main open risk.
- `waving` and `sad` have 3 recordings each against 9 for other classes.
- Expressions need to be reasonably pronounced.
- Single-frame only: anything defined by movement is out of scope by design.

## Team

Built at a hackathon by [Max Deng](https://github.com/Denger878),
Noor Ahmar and [Spencer Krafczek](https://github.com/spencerkrafczek),
then substantially rebuilt afterwards.
