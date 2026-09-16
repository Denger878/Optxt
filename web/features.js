// features.js
//
// Exact JavaScript port of the Python features.py.
//
// Both must produce byte-identical vectors or the browser build silently
// disagrees with the model it is running. training/verify_features_js.py
// compares the two across every recorded frame; run it after any change here.

export const FACE_POINTS = [
  'mouth_left', 'mouth_right', 'mouth_top', 'mouth_bottom',
  'upper_lip', 'lower_lip',
  'left_eyebrow_inner', 'left_eyebrow_outer',
  'right_eyebrow_inner', 'right_eyebrow_outer',
  'left_eye_top', 'left_eye_bottom',
  'right_eye_top', 'right_eye_bottom',
  'nose_tip', 'chin', 'forehead',
];

export const FINGER_TIPS = ['thumb_tip', 'index_tip', 'middle_tip', 'ring_tip', 'pinky_tip'];

const EPS = 1e-6;

const xy = (p) => [p[0], p[1]];
const sub = (a, b) => [a[0] - b[0], a[1] - b[1]];
const mid = (a, b) => [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
const norm = (v) => Math.sqrt(v[0] * v[0] + v[1] * v[1]);
const dist = (a, b) => norm(sub(a, b));
const dot = (a, b) => a[0] * b[0] + a[1] * b[1];

// MediaPipe's hand list order is not stable between frames, so hands[0] can
// flip from left to right mid-gesture. The raised hand is the one gesturing.
function pickHand(hands) {
  if (!hands || hands.length === 0) return null;
  let best = hands[0];
  for (const h of hands) if (h.wrist[1] < best.wrist[1]) best = h;
  return best;
}

export function gestureFeatures(landmarks) {
  if (!landmarks || !landmarks.pose) return null;

  const pose = landmarks.pose;
  const lSh = xy(pose.left_shoulder);
  const rSh = xy(pose.right_shoulder);

  const origin = mid(lSh, rSh);
  const scale = dist(lSh, rSh);
  if (scale < EPS) return null;

  const rel = (p) => {
    const v = sub(xy(p), origin);
    return [v[0] / scale, v[1] / scale];
  };

  const f = [];

  // Shoulder tilt
  f.push((lSh[0] - rSh[0]) / scale, (lSh[1] - rSh[1]) / scale);

  // Arms in the body frame
  for (const k of ['left_elbow', 'right_elbow', 'left_wrist', 'right_wrist']) {
    f.push(...rel(pose[k]));
  }

  // Head in the body frame
  if (landmarks.face) {
    const face = landmarks.face;
    for (const k of ['nose_tip', 'chin', 'forehead']) f.push(...rel(face[k]));

    const faceHeight = dist(xy(face.forehead), xy(face.chin));
    if (faceHeight > EPS) {
      f.push((origin[1] - xy(face.chin)[1]) / faceHeight);
      f.push((origin[1] - xy(face.nose_tip)[1]) / faceHeight);
      f.push(dist(origin, xy(face.chin)) / faceHeight);
      f.push(scale / faceHeight);
    } else {
      f.push(0, 0, 0, 0);
    }
    f.push(dist(xy(face.chin), origin) / scale);
    f.push(1.0);
  } else {
    f.push(0, 0, 0, 0, 0, 0);
    f.push(0, 0, 0, 0);
    f.push(0);
    f.push(0);
  }

  // Hand
  const hand = pickHand(landmarks.hands);
  if (hand) {
    const hWrist = xy(hand.wrist);
    f.push((hWrist[0] - origin[0]) / scale, (hWrist[1] - origin[1]) / scale);

    const tips = FINGER_TIPS.map((t) => xy(hand[t]));
    let span = 0;
    for (const t of tips) span = Math.max(span, dist(hWrist, t));
    if (span < EPS) span = 1.0;

    for (const t of tips) f.push((t[0] - hWrist[0]) / span, (t[1] - hWrist[1]) / span);
    for (const t of tips) f.push(dist(hWrist, t) / span);

    // Finger spread - what separates an open waving palm from one raised finger
    const spreads = [];
    for (let i = 0; i < tips.length - 1; i++) spreads.push(dist(tips[i], tips[i + 1]) / span);
    f.push(...spreads);
    f.push(spreads.reduce((a, b) => a + b, 0) / spreads.length);

    f.push(1.0);
  } else {
    for (let i = 0; i < 2; i++) f.push(0);
    for (let i = 0; i < 10; i++) f.push(0);
    for (let i = 0; i < 5; i++) f.push(0);
    for (let i = 0; i < 5; i++) f.push(0);
    f.push(0);
  }

  return f;
}

export function emotionFeatures(landmarks) {
  if (!landmarks || !landmarks.face) return null;
  const face = landmarks.face;
  for (const k of FACE_POINTS) if (!(k in face)) return null;

  const p = (k) => xy(face[k]);

  const eyeL = mid(p('left_eye_top'), p('left_eye_bottom'));
  const eyeR = mid(p('right_eye_top'), p('right_eye_bottom'));

  const origin = mid(eyeL, eyeR);
  const axis = sub(eyeR, eyeL);
  const scale = norm(axis);
  if (scale < EPS) return null;

  const axisX = [axis[0] / scale, axis[1] / scale];
  const axisY = [-axisX[1], axisX[0]];
  const rel = (pt) => {
    const v = sub(pt, origin);
    return [dot(v, axisX), dot(v, axisY)];
  };

  const f = [];

  const mouthW = dist(p('mouth_left'), p('mouth_right'));
  const mouthH = dist(p('upper_lip'), p('lower_lip'));
  f.push(mouthH / scale, mouthW / scale, mouthH / (mouthW + EPS));

  // Smile direction - corners above the lip centre is a smile, below a frown
  const cornerY = (rel(p('mouth_left'))[1] + rel(p('mouth_right'))[1]) / 2;
  const centreY = (rel(p('upper_lip'))[1] + rel(p('lower_lip'))[1]) / 2;
  f.push(centreY - cornerY);
  f.push(Math.abs(rel(p('mouth_left'))[1] - rel(p('mouth_right'))[1]));

  const leftOpen = dist(p('left_eye_top'), p('left_eye_bottom'));
  const rightOpen = dist(p('right_eye_top'), p('right_eye_bottom'));
  f.push(leftOpen / scale, rightOpen / scale,
         (leftOpen + rightOpen) / (2 * scale), Math.abs(leftOpen - rightOpen) / scale);

  for (const [side, eye] of [['left', eyeL], ['right', eyeR]]) {
    for (const end of ['inner', 'outer']) {
      f.push(rel(eye)[1] - rel(p(`${side}_eyebrow_${end}`))[1]);
    }
  }
  for (const side of ['left', 'right']) {
    f.push(rel(p(`${side}_eyebrow_inner`))[1] - rel(p(`${side}_eyebrow_outer`))[1]);
  }
  f.push(dist(p('left_eyebrow_inner'), p('right_eyebrow_inner')) / scale);

  f.push(dist(p('chin'), p('nose_tip')) / scale);
  const faceHeight = dist(p('forehead'), p('chin'));
  f.push(faceHeight / scale);

  f.push((dist(p('mouth_top'), p('upper_lip')) + dist(p('mouth_bottom'), p('lower_lip'))) / scale);
  f.push(dist(p('mouth_left'), eyeL) / scale);
  f.push(dist(p('mouth_right'), eyeR) / scale);
  f.push(dist(p('nose_tip'), p('upper_lip')) / scale);

  if (faceHeight > EPS) {
    f.push((rel(eyeL)[1] - rel(p('left_eyebrow_inner'))[1]) * scale / faceHeight);
    f.push((rel(eyeR)[1] - rel(p('right_eyebrow_inner'))[1]) * scale / faceHeight);
    f.push(mouthH / faceHeight, mouthW / faceHeight);
  } else {
    f.push(0, 0, 0, 0);
  }

  f.push(rel(mid(p('mouth_left'), p('mouth_right')))[0]);

  return f;
}

export const EXTRACTORS = { gesture: gestureFeatures, emotion: emotionFeatures };
export const extract = (landmarks, type) => EXTRACTORS[type](landmarks);
