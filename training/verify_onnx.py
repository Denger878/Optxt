# training/verify_onnx.py
"""
Check the exported ONNX models against scikit-learn on real recordings.

A silent numerical drift between the two would mean the web build quietly
disagrees with the desktop app, so this compares predicted labels and raw
probabilities across every frame of every recording.
"""
import glob
import json
import os
import sys

import joblib
import numpy as np
import onnxruntime as ort

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features import extract

HERE = os.path.dirname(os.path.abspath(__file__))
WEB_MODELS = os.path.join(os.path.dirname(HERE), "web", "models")


def check(data_type):
    model = joblib.load(os.path.join(HERE, "models", f"{data_type}_model.pkl"))
    session = ort.InferenceSession(os.path.join(WEB_MODELS, f"{data_type}.onnx"))
    labels = json.load(open(os.path.join(WEB_MODELS, f"{data_type}_labels.json")))

    X = []
    for path in sorted(glob.glob(os.path.join(HERE, "data", data_type, "*.json"))):
        for frame in json.load(open(path)):
            feats = extract(frame["landmarks"], data_type)
            if feats is not None:
                X.append(feats)
    X = np.array(X, dtype=np.float32)

    sk_proba = model.predict_proba(X)
    sk_labels = model.predict(X)

    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: X})
    onnx_labels = np.array([str(v) for v in outputs[0]])
    onnx_proba = np.array(outputs[1])

    label_match = float((onnx_labels == sk_labels.astype(str)).mean())
    max_prob_diff = float(np.abs(sk_proba - onnx_proba).max())

    print(f"{data_type}: {len(X)} frames")
    print(f"  class order identical:      {labels == [str(c) for c in model.classes_]}")
    print(f"  labels agree:               {label_match:.4%}")
    print(f"  max probability difference: {max_prob_diff:.2e}")
    ok = label_match == 1.0 and max_prob_diff < 1e-5
    print(f"  -> {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    results = [check(dt) for dt in ("gesture", "emotion")]
    sys.exit(0 if all(results) else 1)
