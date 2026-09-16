// app.js
//
// Browser build of Optxt. Everything runs client-side: MediaPipe extracts
// landmarks, features.js turns them into the same vectors the Python code
// produces, and the exported ONNX models classify them here in the page.
//
// There is no backend on purpose. Sending a frame to a server and back would
// add a network round trip to every frame of a 30fps loop, and an accessibility
// tool should not stop working when the connection does.

import { FilesetResolver, FaceLandmarker, PoseLandmarker, HandLandmarker }
  from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/vision_bundle.mjs';
import { buildLandmarks } from './landmarks.js';
import { gestureFeatures, emotionFeatures } from './features.js';
import { StateTracker } from './tracker.js';

const WASM_BASE = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm';
const ORT_WASM = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/';
const MODEL_BASE = 'https://storage.googleapis.com/mediapipe-models';

// Matches CONFIDENCE_THRESHOLD in gestures.py / emotions.py.
const CONFIDENCE_THRESHOLD = 0.55;
const ANNOUNCEMENT_COOLDOWN = 4000;   // ms, matches main.py

const video = document.getElementById('video');
const canvas = document.getElementById('overlay');
const ctx = canvas.getContext('2d');
const gate = document.getElementById('gate');
const startBtn = document.getElementById('start');
const errBox = document.getElementById('err');

let face, pose, hands;
let sessions = {};
let labels = {};

// ---------------------------------------------------------------- inference

async function loadModel(name) {
  const [session, classNames] = await Promise.all([
    ort.InferenceSession.create(`./models/${name}.onnx`,
      { executionProviders: ['wasm'] }),
    fetch(`./models/${name}_labels.json`).then((r) => r.json()),
  ]);
  sessions[name] = session;
  labels[name] = classNames;
}

/**
 * Run one feature vector through a model.
 *
 * The probability tensor is located by shape rather than by name - the exported
 * graph's output names differ between the two estimators, and argmax over
 * probabilities is more robust than parsing the label tensor, which comes back
 * as ints or bytes depending on the converter.
 */
async function classify(name, features) {
  const session = sessions[name];
  const classNames = labels[name];
  const input = new ort.Tensor('float32', Float32Array.from(features), [1, features.length]);
  const results = await session.run({ [session.inputNames[0]]: input });

  let probs = null;
  for (const key of session.outputNames) {
    const t = results[key];
    if (t && t.data && t.data.length === classNames.length) probs = t.data;
  }
  if (!probs) return { label: 'no_model', confidence: 0 };

  let best = 0;
  for (let i = 1; i < probs.length; i++) if (probs[i] > probs[best]) best = i;
  const confidence = probs[best];

  return confidence < CONFIDENCE_THRESHOLD
    ? { label: 'unsure', confidence }
    : { label: classNames[best], confidence };
}

// ---------------------------------------------------------------- speech

let muted = false;
function say(text) {
  if (muted || !('speechSynthesis' in window)) return;
  const utter = new SpeechSynthesisUtterance(text);
  utter.rate = 1.05;
  speechSynthesis.cancel();      // never queue a backlog of stale announcements
  speechSynthesis.speak(utter);
}

// ---------------------------------------------------------------- drawing

const FACE_WIRE = '#00ff00';
const POSE_WIRE = '#ff0000';
const HAND_WIRE = '#0000ff';

const POSE_BONES = [[11, 12], [11, 13], [13, 15], [12, 14], [14, 16]];

function drawWire(points, connections, colour, w, h, s) {
  if (!points) return;
  ctx.strokeStyle = colour;
  ctx.fillStyle = colour;
  ctx.lineWidth = Math.max(2, Math.round(s * 1.5));
  ctx.beginPath();
  for (const c of connections) {
    const a = points[c.start ?? c[0]];
    const b = points[c.end ?? c[1]];
    if (!a || !b) continue;
    ctx.moveTo(a.x * w, a.y * h);
    ctx.lineTo(b.x * w, b.y * h);
  }
  ctx.stroke();

  const r = Math.max(2, Math.round(s * 1.5));
  const seen = new Set();
  for (const c of connections) {
    for (const i of [c.start ?? c[0], c.end ?? c[1]]) {
      if (seen.has(i) || !points[i]) continue;
      seen.add(i);
      ctx.fillRect(points[i].x * w - r, points[i].y * h - r, r * 2, r * 2);
    }
  }
}

function plate(x, y, w, h, outline, s) {
  ctx.fillStyle = 'rgba(0,0,0,0.70)';
  ctx.fillRect(x, y, w, h);
  ctx.strokeStyle = outline;
  ctx.lineWidth = Math.max(2, Math.round(2 * s));
  ctx.strokeRect(x, y, w, h);
}

function drawHud(gesture, gConf, emotion, eConf, transcript) {
  const w = canvas.width, h = canvas.height;
  const s = h / 480;
  const sz = (v) => Math.max(1, Math.round(v * s));
  const placeholder = new Set(['no_data', 'no_model', 'unsure', 'no_face']);

  // reading plate
  const pad = sz(13);
  const px = sz(20), py = sz(20);
  const pw = Math.min(sz(214), w * 0.42);
  const inner = pw - pad * 2;
  const rowH = sz(10) + sz(28) + sz(10);
  const ph = pad * 2 + rowH * 2 - sz(8);

  plate(px, py, pw, ph, '#ff0000', s);

  let y = py + pad;
  for (const [title, label, conf] of [['GESTURE', gesture, gConf],
                                      ['EMOTION', emotion, eConf]]) {
    const strong = !placeholder.has(label);
    ctx.textBaseline = 'top';
    ctx.font = `bold ${sz(9)}px "Courier New", monospace`;
    ctx.fillStyle = 'rgba(255,255,255,0.6)';
    ctx.fillText(title, px + pad, y);

    const pct = `${Math.round(conf * 100)}%`;
    ctx.font = `bold ${sz(10)}px "Courier New", monospace`;
    ctx.fillText(pct, px + pw - pad - ctx.measureText(pct).width, y);

    ctx.font = `bold ${sz(18)}px "Courier New", monospace`;
    ctx.fillStyle = strong ? '#fff' : 'rgba(255,255,255,0.6)';
    ctx.fillText(String(label).replace(/_/g, ' ').toUpperCase(), px + pad, y + sz(9));

    const barY = y + sz(9) + sz(28), barH = Math.max(4, sz(5));
    ctx.fillStyle = 'rgba(38,38,42,0.94)';
    ctx.fillRect(px + pad, barY, inner, barH);
    ctx.fillStyle = strong ? '#fff' : 'rgba(255,255,255,0.6)';
    ctx.fillRect(px + pad, barY, inner * Math.max(0, Math.min(1, conf)), barH);
    y += rowH;
  }

  // transcript plate
  const lines = transcript.slice(-3);
  const lineH = sz(15), tpad = sz(11);
  const th = tpad + sz(11) + sz(7) + lineH * Math.max(lines.length, 1) + tpad - sz(4);
  const tx = sz(20), tw = w - sz(40), ty = h - sz(20) - th;

  plate(tx, ty, tw, th, '#0000ff', s);

  let cy = ty + tpad;
  ctx.font = `bold ${sz(9)}px "Courier New", monospace`;
  ctx.fillStyle = 'rgba(255,255,255,0.6)';
  ctx.fillText('TRANSCRIPT', tx + tpad, cy);
  const hint = muted ? '// MUTED //   M UNMUTE' : 'M MUTE   S WIRE';
  ctx.fillText(hint, tx + tw - tpad - ctx.measureText(hint).width, cy);

  cy += sz(14);
  ctx.strokeStyle = 'rgba(255,255,255,0.85)';
  ctx.lineWidth = Math.max(2, sz(2));
  ctx.beginPath();
  ctx.moveTo(tx + tpad, cy); ctx.lineTo(tx + tw - tpad, cy); ctx.stroke();
  cy += sz(7);

  ctx.font = `bold ${sz(11)}px "Courier New", monospace`;
  if (!lines.length) {
    ctx.fillStyle = 'rgba(255,255,255,0.6)';
    ctx.fillText('> AWAITING SIGNAL', tx + tpad, cy);
  } else {
    lines.forEach((entry, i) => {
      const newest = i === lines.length - 1;
      ctx.fillStyle = newest ? '#fff' : 'rgba(255,255,255,0.6)';
      const mm = String(Math.floor(entry.t / 60)).padStart(2, '0');
      const ss = String(Math.floor(entry.t % 60)).padStart(2, '0');
      ctx.fillText(`${mm}:${ss}`, tx + tpad, cy);
      ctx.fillText(entry.text, tx + tpad + ctx.measureText('00:00  ').width, cy);
      cy += lineH;
    });
  }
}

// ---------------------------------------------------------------- loop

const tracker = new StateTracker(5);
const transcript = [];
let showWire = true;
let lastAnnouncement = 0;
let sessionStart = 0;
let busy = false;

async function frame() {
  if (video.readyState >= 2 && !busy) {
    busy = true;
    const now = performance.now();

    const faceRes = face.detectForVideo(video, now);
    const poseRes = pose.detectForVideo(video, now);
    const handRes = hands.detectForVideo(video, now);
    const lm = buildLandmarks(faceRes, poseRes, handRes);

    let gesture = { label: 'no_data', confidence: 0 };
    let emotion = { label: 'no_data', confidence: 0 };

    if (lm) {
      // Placeholder labels mirror gestures.py / emotions.py exactly, so the
      // two builds describe a missing detection the same way.
      const gf = gestureFeatures(lm);
      const ef = emotionFeatures(lm);
      gesture = gf ? await classify('gesture', gf) : { label: 'no_data', confidence: 0 };
      emotion = ef ? await classify('emotion', ef) : { label: 'no_face', confidence: 0 };

      const result = tracker.update(gesture.label, emotion.label);
      if (result.changed && now - lastAnnouncement >= ANNOUNCEMENT_COOLDOWN) {
        transcript.push({ t: (now - sessionStart) / 1000, text: result.message });
        if (transcript.length > 12) transcript.shift();
        say(result.message);
        lastAnnouncement = now;
      }
    }

    // Size the canvas to the element so the HUD is never stretched.
    const rect = canvas.getBoundingClientRect();
    if (canvas.width !== Math.round(rect.width) || canvas.height !== Math.round(rect.height)) {
      canvas.width = Math.round(rect.width);
      canvas.height = Math.round(rect.height);
    }
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (showWire) {
      // The video is mirrored in CSS; mirror the wireframe to match.
      ctx.save();
      ctx.translate(canvas.width, 0);
      ctx.scale(-1, 1);
      const s = canvas.height / 480;
      drawWire(faceRes?.faceLandmarks?.[0], FaceLandmarker.FACE_LANDMARKS_CONTOURS,
               FACE_WIRE, canvas.width, canvas.height, s);
      drawWire(poseRes?.landmarks?.[0], POSE_BONES,
               POSE_WIRE, canvas.width, canvas.height, s);
      for (const hand of handRes?.landmarks ?? []) {
        drawWire(hand, HandLandmarker.HAND_CONNECTIONS,
                 HAND_WIRE, canvas.width, canvas.height, s);
      }
      ctx.restore();
    }

    drawHud(gesture.label, gesture.confidence, emotion.label, emotion.confidence, transcript);
    busy = false;
  }
  requestAnimationFrame(frame);
}

// ---------------------------------------------------------------- boot

async function init() {
  try {
    // onnxruntime fetches its own .wasm binaries; point them at the same CDN
    // build as the loader, or it looks for them beside the page and 404s.
    ort.env.wasm.wasmPaths = ORT_WASM;

    const vision = await FilesetResolver.forVisionTasks(WASM_BASE);
    [face, pose, hands] = await Promise.all([
      FaceLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: `${MODEL_BASE}/face_landmarker/face_landmarker/float16/1/face_landmarker.task` },
        runningMode: 'VIDEO', numFaces: 1, outputFaceBlendshapes: false,
      }),
      PoseLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: `${MODEL_BASE}/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task` },
        runningMode: 'VIDEO', numPoses: 1,
      }),
      HandLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: `${MODEL_BASE}/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task` },
        runningMode: 'VIDEO', numHands: 2,
      }),
    ]);

    await Promise.all([loadModel('gesture'), loadModel('emotion')]);

    startBtn.disabled = false;
    startBtn.textContent = 'START CAMERA';
  } catch (e) {
    errBox.textContent = `Failed to load: ${e.message}`;
  }
}

startBtn.addEventListener('click', async () => {
  startBtn.disabled = true;
  startBtn.textContent = 'REQUESTING CAMERA...';
  try {
    video.srcObject = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' },
      audio: false,
    });
    await video.play();
    gate.style.display = 'none';
    sessionStart = performance.now();
    requestAnimationFrame(frame);
  } catch (e) {
    startBtn.disabled = false;
    startBtn.textContent = 'START CAMERA';
    errBox.textContent = e.name === 'NotAllowedError'
      ? 'Camera permission denied. Allow it in your browser settings and try again.'
      : `Camera error: ${e.message}`;
  }
});

window.addEventListener('keydown', (e) => {
  const k = e.key.toLowerCase();
  if (k === 'm') { muted = !muted; if (muted) speechSynthesis.cancel(); }
  if (k === 's') showWire = !showWire;
});

init();
