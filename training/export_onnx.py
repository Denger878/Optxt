# training/export_onnx.py
"""
Export the trained classifiers to ONNX for in-browser inference.

The web build runs the models client-side with onnxruntime-web, so there is no
server in the prediction path at all. That keeps the offline property the
desktop app has and avoids an HTTP round trip per video frame, which at 30fps
would never have worked.

Run after training:  python training/export_onnx.py
"""
import os
import sys

import joblib
import numpy as np
from skl2onnx import to_onnx
from skl2onnx.common.data_types import FloatTensorType

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features import FEATURE_NAMES

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(os.path.dirname(HERE), "web", "models")


def export(data_type):
    model_path = os.path.join(HERE, "models", f"{data_type}_model.pkl")
    model = joblib.load(model_path)
    n_features = len(FEATURE_NAMES[data_type])

    onx = to_onnx(
        model,
        initial_types=[("input", FloatTensorType([None, n_features]))],
        # zipmap wraps probabilities in a dict, which onnxruntime-web reports as
        # an unsupported sequence type. A plain tensor is what the browser wants.
        options={id(model): {"zipmap": False}},
        target_opset=15,
    )

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"{data_type}.onnx")
    with open(out_path, "wb") as f:
        f.write(onx.SerializeToString())

    size_mb = os.path.getsize(out_path) / 1e6
    print(f"{data_type}: {n_features} features, {len(model.classes_)} classes "
          f"-> {out_path} ({size_mb:.1f} MB)")

    # The browser gets class names from a separate JSON; ONNX stores them as
    # ints or bytes depending on the converter.
    labels_path = os.path.join(OUT_DIR, f"{data_type}_labels.json")
    import json
    with open(labels_path, "w") as f:
        json.dump([str(c) for c in model.classes_], f)

    return model, onx, n_features


if __name__ == "__main__":
    for dt in ("gesture", "emotion"):
        export(dt)
    print("\nDone. Verify with: python training/verify_onnx.py")
