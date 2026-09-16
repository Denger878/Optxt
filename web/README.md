# Optxt — web build

The browser version of the desktop app. **Everything runs client-side**: no
backend, no API keys, and no video, landmarks or audio ever leave the device.

```
index.html            page shell + camera permission gate
app.js                capture loop, inference, HUD, speech
landmarks.js          MediaPipe output  ->  named landmark dict   (mirrors landmarks.py)
features.js           landmark dict     ->  feature vector        (mirrors features.py)
tracker.js            temporal smoothing + phrasing               (mirrors state_tracker.py)
models/*.onnx         classifiers exported from the .pkl files
models/*_labels.json  class names, in the model's own class order
verify_pipeline.mjs   runs this pipeline in Node so it can be diffed against Python
```

## Why there is no server

Sending every frame to a backend would put a network round trip inside a 30fps
loop, and an accessibility tool should keep working when the connection drops.
MediaPipe runs in WebAssembly and the models are ~6MB of ONNX, so the whole
thing fits in the page.

## Keeping the two builds honest

Three things have to stay in lockstep with the Python code, and each has a test:

| Risk | Test |
|---|---|
| ONNX export drifts from the `.pkl` | `python training/verify_onnx.py` |
| `features.js` drifts from `features.py` | `python training/verify_features_js.py` |
| End-to-end disagreement | `node web/verify_pipeline.mjs /tmp/p.json && python training/verify_pipeline.py /tmp/p.json` |

Current state: **2150/2150 frames identical**, ONNX probabilities within 3e-7 of
scikit-learn, and feature vectors matching to 5.6e-17.

Run all three after changing any model, any feature, or `landmarks.py`.

## Rebuilding the models

```bash
python training/train.py         # retrain from recordings
python training/export_onnx.py   # re-export to web/models/
python training/verify_onnx.py   # confirm the export
```

## Local preview

```bash
cd web && python -m http.server 8777
```

Then open http://localhost:8777. `getUserMedia` needs a secure context —
`localhost` counts as one, so this works without HTTPS.

## Deploying

Vercel, with **Root Directory set to `web`** and framework preset **Other**.
There is no build step; it is static files.
