// verify_pipeline.mjs
//
// Runs the browser pipeline in Node - features.js -> ONNX -> label - over every
// recording, so its predictions can be diffed against the Python app's. If the
// JavaScript feature port ever drifts from features.py, or the ONNX export
// drifts from the .pkl, this is what catches it.
//
//   npm install onnxruntime-node
//   node web/verify_pipeline.mjs /tmp/js_preds.json
//   python training/verify_pipeline.py /tmp/js_preds.json
//
import ort from 'onnxruntime-node';
import fs from 'fs';
import path from 'path';
import { gestureFeatures, emotionFeatures } from './features.js';

const ROOT = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..');
const EXTRACT = { gesture: gestureFeatures, emotion: emotionFeatures };
const THRESHOLD = 0.55;

const out = {};
for (const type of ['gesture', 'emotion']) {
  const session = await ort.InferenceSession.create(`${ROOT}/web/models/${type}.onnx`);
  const labels = JSON.parse(fs.readFileSync(`${ROOT}/web/models/${type}_labels.json`, 'utf8'));
  const dir = `${ROOT}/training/data/${type}`;
  const rows = [];

  for (const file of fs.readdirSync(dir).filter(f => f.endsWith('.json')).sort()) {
    const frames = JSON.parse(fs.readFileSync(path.join(dir, file), 'utf8')).slice(0, 25);
    for (const fr of frames) {
      const feats = EXTRACT[type](fr.landmarks);
      if (!feats) { rows.push(type === 'emotion' ? 'no_face' : 'no_data'); continue; }
      const t = new ort.Tensor('float32', Float32Array.from(feats), [1, feats.length]);
      const res = await session.run({ [session.inputNames[0]]: t });
      let probs = null;
      for (const k of session.outputNames) {
        const d = res[k];
        if (d && d.data && d.data.length === labels.length) probs = d.data;
      }
      let best = 0;
      for (let i = 1; i < probs.length; i++) if (probs[i] > probs[best]) best = i;
      rows.push(probs[best] < THRESHOLD ? 'unsure' : labels[best]);
    }
  }
  out[type] = rows;
}
fs.writeFileSync(process.argv[2], JSON.stringify(out));
