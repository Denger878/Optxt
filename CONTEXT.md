# Optxt — Project Context

> Handoff document. Written so an assistant with no prior exposure to this
> codebase can help with it, and so Max can describe it accurately on a resume
> and in interviews.

## What it is

A real-time desktop application that watches a webcam, recognises another
person's hand gestures and facial expressions, and narrates them out loud. The
intended user is blind or low-vision: social cues that sighted people read
automatically ("they're smiling", "they're pointing at you") are announced as
speech.

Built at a hackathon by a team of three, then substantially rebuilt afterwards.

- Repo: https://github.com/spencerkrafczek/Optxt (owned by a teammate; Max has
  direct push access)
- Platform: macOS desktop, Python 3.11
- 53 commits: Max Deng 34, Noor Ahmar 12, Spencer Krafczek 12

## Tech stack

| Layer | Choice | Why |
|---|---|---|
| Landmark extraction | **MediaPipe** (FaceMesh, Pose, Hands) | Three graphs run concurrently per frame, ~19ms |
| Video I/O + display | **OpenCV** | Capture, colour conversion, window |
| Feature engineering | **NumPy** | Hand-built geometric features |
| Classification | **scikit-learn** (ExtraTrees, RandomForest) | Small tabular dataset; trains in seconds |
| Model persistence | **joblib** | `.pkl` files, self-contained at inference |
| Speech | **pyttsx3** + macOS `say` | Offline, no API key, no network in the loop |
| HUD rendering | **Pillow** | Real system typefaces; OpenCV only has Hershey fonts |

Deliberately **no deep learning and no cloud APIs.** With a few hundred samples
per class a neural net would overfit badly, and a tree ensemble trains in
seconds. An earlier version used ElevenLabs for speech; it was removed because
an accessibility tool must not stop working when the network does.

## Repo structure

```
main.py             100  app loop: capture → landmarks → predict → speak → draw
landmarks.py        170  MediaPipe wrapper; raw video frame → landmark dict
features.py         311  landmark dict → model feature vector  (the core)
gestures.py          55  gesture model load + predict, with confidence gating
emotions.py          54  emotion model load + predict, with confidence gating
state_tracker.py     85  temporal smoothing + phrasing of spoken sentences
speech.py            84  threaded text-to-speech
overlay.py          271  HUD and wireframe rendering
predict.py           44  display-free entry point (landmarks in, JSON out)

training/
  collect.py        171  guided webcam data-collection tool
  train.py          217  training, grouped validation, model selection
  models/*.pkl           trained classifiers
  data/{gesture,emotion}/<label>_<timestamp>.json
```

**The label of every recording is its filename prefix,** and the timestamp makes
each take a distinct group. Both facts matter to the validation strategy below.

## Data flow

```
webcam frame (BGR)
   └─ landmarks.extract_landmarks()          MediaPipe ×3
        → {face: {17 named points}, pose: {6 joints}, hands: [{6 points}]}
   └─ features.gesture_features() / emotion_features()
        → 45-float vector / 27-float vector
   └─ ExtraTrees / RandomForest .predict_proba()
        → label + confidence, or "unsure" below 0.55
   └─ StateTracker: 5 consecutive agreeing frames before it counts
   └─ say_interaction() on a worker thread + HUD transcript
```

Runs at **32–40 FPS**; MediaPipe is ~19ms of the ~25ms frame budget.

## Feature engineering — the substantive part

Raw landmark coordinates are **normalized** so the features describe what a body
is doing rather than where it happened to be:

- **Gestures** (45 features): origin at the shoulder midpoint, distances in
  shoulder-width units. Hand *shape* is measured in the hand's own frame scaled
  by hand span, so "which fingers are extended" is independent of hand size.
  Includes per-finger extension ratios and **finger-spread** between adjacent
  fingertips.
- **Emotions** (27 features): origin between the eyes, x-axis along the eye
  line, distances in interocular units — invariant to head tilt and camera
  distance. Interocular distance is the scale because, unlike face height, it
  doesn't change when the jaw drops.

Emotion features are **pure geometry — no raw coordinates.** This was measured,
not assumed (see bug 3).

MediaPipe detects far more than is used: 468 face points reduced to 17 curated
ones, 33 pose points to 6, 21 hand points to 6 plus derived spread.

## Validation methodology

Frames within one 5-second recording are near-identical. A random train/test
split therefore puts near-duplicates on both sides and measures memorisation.

`train.py` splits by **recording**, using `StratifiedGroupKFold` so whole takes
land on one side and every class is still represented. It also compares
RandomForest / ExtraTrees / HistGradientBoosting by grouped CV and keeps the
winner, and refuses to print an accuracy at all if any class has fewer than two
recordings.

## Current results

| | Gesture | Emotion |
|---|---|---|
| Model | ExtraTreesClassifier | RandomForestClassifier |
| Classes | 6: middlefinger, neutral, pointing, thumbsdown, thumbsup, waving | 5: angry, happy, neutral, sad, shocked |
| Features | 45 | 27 |
| Data | 3,263 frames / 47 recordings | 2,759 frames / 39 recordings |
| **Grouped 3-fold CV** | **97.7% (±1.3)** | **93.0% (±3.4)** |
| Held-out (whole recordings) | 99.0% | 97.6% |

Quote the **CV** figure — it's the more conservative and stable estimate.

## The four defects that were found and fixed

This is the technically interesting history.

**1. Data leakage in evaluation.** The original code split train/test randomly
*by frame*. Measured: every test sample's nearest training neighbour sat at
distance 0.007 versus ~0.5 for genuinely different samples of the same class.
The reported 100% gesture / 94.9% emotion accuracy was meaningless. Fixed with
recording-level grouped splits. Honest baseline after the fix: gesture 83.4%
held-out, emotion 62.1%.

**2. Eyebrow landmarks named inside-out.** MediaPipe's brow chains run lateral →
medial, so indices 70/300 are the *outer* ends and 107/336 the inner. They were
labelled the opposite way, so the "brow furrow" feature was measuring the gap
between the outer brow ends — effectively face width, which barely moves when
you scowl. Anger scored **0.00 precision**, misclassified as "shocked" 72 times
out of 75.

**3. Raw coordinates were actively harmful.** Ablation on the emotion features:

```
raw landmark coordinates only    51–63%
hand-built geometry only         85.2%
both together                    77–82%
```

The raw coordinates dragged the combined set *below* geometry alone — on a few
hundred samples the model keys on where a face sat rather than what it was
doing. They were removed entirely.

**4. No smile-direction feature.** Mouth width and openness existed, but nothing
encoded whether the corners point up or down — and a wide-open mouth is a grin
or a snarl depending entirely on that. 59 of 139 happy frames were classified as
angry. Adding corner-lift, brow slope and lip compression took emotion from 71%
to 90.5% on the *same data*.

**Separately:** "shrug" was retired as a class. Held at realistic intensity it is
nearly indistinguishable from neutral in a single frame, and it accounted for
almost all remaining gesture error while every hand gesture sat at 97–99%.
Dropping it moved gesture CV from 81.6% → 97.7%.

**Waving** is recognised as a *static open palm* — all five fingers extended and
spread — rather than by motion. A motion-window implementation (velocity,
direction reversals) was built and then deliberately reverted for a simpler
single-frame architecture. `spread_index_middle` is now the single
highest-importance gesture feature.

## Ownership

Max owns the recognition system and the web deployment end to end.

- **Data collection.** Designed and ran the capture protocol: which classes to
  record, how to pose them, and multiple takes per class from different
  positions and distances, which is what makes an honest held-out score
  possible. 86 recordings, ~6,000 frames.
- **Class design.** Chose which gestures and emotions to recognise, added
  `waving` and `sad`, and retired `shrug` after the confusion matrix showed it
  was indistinguishable from neutral in a single frame.
- **Feature engineering** (`features.py`). The normalized geometric features
  for both models, including smile direction, brow slope, finger spread, and
  the decision to drop raw coordinates from the emotion model after ablation.
- **Training and validation** (`training/`). Recording-level grouped
  validation, model selection, and the iterative loop that took the models
  from a leaky 100% to an honest 62–83% and up to 93–97.7%.
- **Landmark extraction** (`landmarks.py`). Which MediaPipe landmarks to use,
  and the fix for the inside-out eyebrow indices.
- **Web deployment** (`web/`). The client-side port, ONNX export, the
  parity tests between the Python and JavaScript builds, and the Vercel
  deployment.
- Majority of commits (34 of 53).

Teammates Noor Ahmar and Spencer Krafczek owned **text-to-speech**
(`speech.py`, and an earlier ElevenLabs integration) and other pieces of the
original hackathon build inlcuding setting up working environment and main.py that tied the whole project together


## Known limitations

- **Single subject.** All recordings are of one person. Generalisation to other
  faces, skin tones and body types is untested and is the biggest open risk.
- `waving` and `sad` have only 3 recordings each versus 9 for other classes,
  which caps cross-validation at 3 folds.
- Emotion CV variance is ±3.4% — a few takes are much harder than others.
- Expressions must be exaggerated; subtle real-world expressions are unproven.
- Single-frame only. Anything defined by movement is out of scope by design.

## Running it

```bash
cd Optxt
source venv/bin/activate
python main.py                 # q quit · m mute · s toggle wireframe
python training/collect.py     # guided capture, 3 rounds, change position between them
python training/train.py       # choose G / E / B
```

Requires macOS camera permission for the launching terminal.

## Web build

`web/` is a fully client-side port, deployable to Vercel as static files. There
is no backend: MediaPipe runs in WebAssembly, and the classifiers were exported
to ONNX and run in the page with onnxruntime-web. Putting a server in the loop
would add a network round trip to every frame of a 30fps loop and would break
the offline property an accessibility tool needs.

The port duplicates three Python modules in JavaScript — `landmarks.py`,
`features.py`, `state_tracker.py` — which is a real risk: a silent numerical
drift means the web build disagrees with the models it is running. Three
verification scripts exist for exactly that, and all currently pass:

- `training/verify_onnx.py` — ONNX vs scikit-learn (100% label agreement,
  probabilities within 3e-7)
- `training/verify_features_js.py` — `features.js` vs `features.py` (max
  difference 5.6e-17)
- `web/verify_pipeline.mjs` + `training/verify_pipeline.py` — end to end
  (**2150/2150 frames identical**)

Run all three after changing any model, feature or landmark index. See
`web/README.md`.
