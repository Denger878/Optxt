// landmarks.js
//
// Mirrors the Python landmarks.py: turns MediaPipe's raw output into the same
// named dictionary the feature code expects. The indices below must stay in
// lockstep with landmarks.py - they are what ties the browser to the models.

export const FACE_INDICES = {
  // mouth
  mouth_left: 61, mouth_right: 291, mouth_top: 0, mouth_bottom: 17,
  upper_lip: 13, lower_lip: 14,
  // brows. MediaPipe's chains run lateral -> medial, so 70/300 are the OUTER
  // ends and 107/336 the inner ones. Naming these the wrong way round was a
  // real bug once: it made the brow-furrow feature measure face width and
  // anger was never detected.
  left_eyebrow_outer: 70, left_eyebrow_inner: 107,
  right_eyebrow_outer: 300, right_eyebrow_inner: 336,
  // eyes
  left_eye_top: 159, left_eye_bottom: 145,
  right_eye_top: 386, right_eye_bottom: 374,
  // reference points
  nose_tip: 4, chin: 152, forehead: 10,
};

export const POSE_INDICES = {
  left_shoulder: 11, right_shoulder: 12,
  left_elbow: 13, right_elbow: 14,
  left_wrist: 15, right_wrist: 16,
};

export const HAND_INDICES = {
  wrist: 0, thumb_tip: 4, index_tip: 8,
  middle_tip: 12, ring_tip: 16, pinky_tip: 20,
};

const pt = (lm) => [lm.x, lm.y, lm.z ?? 0];

function mapPoints(source, indices) {
  const out = {};
  for (const [name, i] of Object.entries(indices)) {
    if (!source[i]) return null;
    out[name] = pt(source[i]);
  }
  return out;
}

/**
 * Build the landmark dictionary from the three MediaPipe results.
 * Returns null when nothing at all was detected.
 */
export function buildLandmarks(faceResult, poseResult, handResult) {
  const landmarks = {};

  const face = faceResult?.faceLandmarks?.[0];
  if (face) {
    const mapped = mapPoints(face, FACE_INDICES);
    if (mapped) landmarks.face = mapped;
  }

  const pose = poseResult?.landmarks?.[0];
  if (pose) {
    const mapped = mapPoints(pose, POSE_INDICES);
    if (mapped) landmarks.pose = mapped;
  }

  const hands = handResult?.landmarks;
  if (hands && hands.length) {
    const mapped = hands.map((h) => mapPoints(h, HAND_INDICES)).filter(Boolean);
    if (mapped.length) landmarks.hands = mapped;
  }

  return Object.keys(landmarks).length ? landmarks : null;
}
