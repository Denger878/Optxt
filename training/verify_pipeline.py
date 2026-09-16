# training/verify_pipeline.py
"""
Diff the browser pipeline's predictions against the Python app's.

Takes the JSON written by web/verify_pipeline.mjs and compares it frame by
frame. Anything short of 100% means the web build and the desktop build would
disagree about the same person.

  node web/verify_pipeline.mjs /tmp/js_preds.json
  python training/verify_pipeline.py /tmp/js_preds.json
"""
import glob
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gestures import detect_gesture
from emotions import detect_emotion

HERE = os.path.dirname(os.path.abspath(__file__))
FN = {"gesture": detect_gesture, "emotion": detect_emotion}
LIMIT = 25      # frames per recording; must match verify_pipeline.mjs


def main(js_path):
    js = json.load(open(js_path))
    total = agree = 0

    for data_type in ("gesture", "emotion"):
        py = []
        for path in sorted(glob.glob(os.path.join(HERE, "data", data_type, "*.json"))):
            for frame in json.load(open(path))[:LIMIT]:
                py.append(FN[data_type](frame["landmarks"]))

        other = js[data_type]
        if len(py) != len(other):
            print(f"{data_type}: FAIL - {len(py)} python frames vs {len(other)} js")
            return 1

        n = sum(a == b for a, b in zip(py, other))
        print(f"{data_type}: {n}/{len(py)} agree ({n / len(py):.4%})")
        for a, b in zip(py, other):
            if a != b:
                print(f"   first disagreement: python={a}  js={b}")
                break
        total += len(py)
        agree += n

    print(f"\nTOTAL: {agree}/{total} identical")
    return 0 if agree == total else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/js_preds.json"))
