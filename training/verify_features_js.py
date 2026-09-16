# training/verify_features_js.py
"""
Compare web/features.js against features.py on every recorded frame.

The browser build computes its own feature vectors in JavaScript. If those drift
from the Python ones by even a small amount, the models receive inputs they were
never trained on and the web app quietly disagrees with the desktop app. This
runs both implementations over the real dataset and diffs them element by
element.

Run after touching either file:  python training/verify_features_js.py
"""
import glob
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features import extract, FEATURE_NAMES

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

RUNNER = """
import { extract } from '%s/web/features.js';
import fs from 'fs';
const payload = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const out = payload.frames.map(f => extract(f, payload.type));
fs.writeFileSync(process.argv[3], JSON.stringify(out));
""" % ROOT


def check(data_type, limit_per_file=25):
    frames = []
    for path in sorted(glob.glob(os.path.join(HERE, "data", data_type, "*.json"))):
        for frame in json.load(open(path))[:limit_per_file]:
            frames.append(frame["landmarks"])

    py = [extract(f, data_type) for f in frames]

    with tempfile.TemporaryDirectory() as tmp:
        runner = os.path.join(tmp, "run.mjs")
        inp = os.path.join(tmp, "in.json")
        outp = os.path.join(tmp, "out.json")
        open(runner, "w").write(RUNNER)
        json.dump({"type": data_type, "frames": frames}, open(inp, "w"))
        proc = subprocess.run(["node", runner, inp, outp],
                              capture_output=True, text=True)
        if proc.returncode != 0:
            print(proc.stderr.strip())
            raise SystemExit(f"node failed for {data_type}")
        js = json.load(open(outp))

    assert len(py) == len(js), "frame count mismatch"

    mismatched_none = sum((p is None) != (j is None) for p, j in zip(py, js))
    pairs = [(p, j) for p, j in zip(py, js) if p is not None and j is not None]

    lengths_ok = all(len(p) == len(j) == len(FEATURE_NAMES[data_type]) for p, j in pairs)
    diffs = np.abs(np.array([p for p, _ in pairs]) - np.array([j for _, j in pairs]))

    print(f"{data_type}: {len(frames)} frames, {len(pairs)} with features")
    print(f"  vector length {len(FEATURE_NAMES[data_type])} on both sides: {lengths_ok}")
    print(f"  None/null disagreements:  {mismatched_none}")
    print(f"  max absolute difference:  {diffs.max():.3e}")

    worst = int(np.argmax(diffs.max(axis=0)))
    print(f"  largest-drift feature:    {FEATURE_NAMES[data_type][worst]} "
          f"({diffs.max(axis=0)[worst]:.3e})")

    ok = lengths_ok and mismatched_none == 0 and diffs.max() < 1e-9
    print(f"  -> {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    results = [check(dt) for dt in ("gesture", "emotion")]
    sys.exit(0 if all(results) else 1)
