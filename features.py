# features.py
"""
Single source of truth for turning landmarks into model features.

Both training (training/train.py) and live inference (gestures.py, emotions.py)
import from here, so the two can never drift apart.

Everything below is normalized to be invariant to:
  - where you are standing in the frame (translation)
  - how far you are from the camera (scale)
  - head tilt, for emotions (rotation)

Raw pixel coordinates have none of those properties, which is why a model
trained on them only works from the exact spot the data was recorded in.
"""
import numpy as np

FACE_POINTS = [
    'mouth_left', 'mouth_right', 'mouth_top', 'mouth_bottom',
    'upper_lip', 'lower_lip',
    'left_eyebrow_inner', 'left_eyebrow_outer',
    'right_eyebrow_inner', 'right_eyebrow_outer',
    'left_eye_top', 'left_eye_bottom',
    'right_eye_top', 'right_eye_bottom',
    'nose_tip', 'chin', 'forehead',
]

FINGER_TIPS = ['thumb_tip', 'index_tip', 'middle_tip', 'ring_tip', 'pinky_tip']

EPS = 1e-6


def _xy(point):
    """Landmarks are (x, y, z) tuples; we only use x and y."""
    return np.array([point[0], point[1]], dtype=np.float64)


def _dist(a, b):
    return float(np.linalg.norm(a - b))


def _pick_hand(hands):
    """
    Pick which hand to describe.

    MediaPipe's list order is not stable between frames, so hands[0] can flip
    from left to right mid-gesture and scramble the features. The raised hand
    is the one making the gesture, so pick the highest one (smallest y).
    """
    if not hands:
        return None
    return min(hands, key=lambda h: h['wrist'][1])


# ---------------------------------------------------------------- gestures

def gesture_features(landmarks):
    """
    Build the gesture feature vector.

    Reference frame: origin at the midpoint of the shoulders, distances in
    units of shoulder width. Returns None if there is no pose.
    """
    if landmarks is None or 'pose' not in landmarks:
        return None

    pose = landmarks['pose']
    l_sh, r_sh = _xy(pose['left_shoulder']), _xy(pose['right_shoulder'])

    origin = (l_sh + r_sh) / 2.0
    scale = _dist(l_sh, r_sh)
    if scale < EPS:
        return None

    def rel(p):
        """Position relative to shoulder centre, in shoulder-width units."""
        return (_xy(p) - origin) / scale

    feats = []

    # Shoulder tilt: which way the body is rotated / leaning.
    feats.extend(((l_sh - r_sh) / scale).tolist())

    # Arms in the body frame.
    for key in ['left_elbow', 'right_elbow', 'left_wrist', 'right_wrist']:
        feats.extend(rel(pose[key]).tolist())

    # ---- head, and the shrug signal ----
    if 'face' in landmarks:
        face = landmarks['face']
        for key in ['nose_tip', 'chin', 'forehead']:
            feats.extend(rel(face[key]).tolist())

        # A shrug raises the shoulders toward the head. Measuring that against
        # FACE HEIGHT rather than shoulder width matters: shoulder width itself
        # narrows slightly when you shrug, so using it as the yardstick cancels
        # out part of the very signal we're trying to detect.
        face_height = _dist(_xy(face['forehead']), _xy(face['chin']))
        if face_height > EPS:
            feats.append((origin[1] - _xy(face['chin'])[1]) / face_height)
            feats.append((origin[1] - _xy(face['nose_tip'])[1]) / face_height)
            feats.append(_dist(origin, _xy(face['chin'])) / face_height)
            # Shoulder width in head units - a second, independent shrug cue.
            feats.append(scale / face_height)
        else:
            feats.extend([0.0] * 4)

        feats.append(_dist(_xy(face['chin']), origin) / scale)
        feats.append(1.0)                       # face present
    else:
        feats.extend([0.0] * 6)                 # head positions
        feats.extend([0.0] * 4)                 # shrug metrics
        feats.append(0.0)
        feats.append(0.0)                       # face absent

    # ---- hand ----
    hand = _pick_hand(landmarks.get('hands'))
    if hand is not None:
        h_wrist = _xy(hand['wrist'])
        feats.extend(((h_wrist - origin) / scale).tolist())

        # Hand SHAPE, in the hand's own frame. Scale is the hand's own span, so
        # this describes which fingers are extended independent of hand size —
        # that's what separates thumbs-up from a middle finger.
        tips = [_xy(hand[t]) for t in FINGER_TIPS]
        span = max(_dist(h_wrist, t) for t in tips)
        if span < EPS:
            span = 1.0
        for tip in tips:
            feats.extend(((tip - h_wrist) / span).tolist())
        for tip in tips:
            feats.append(_dist(h_wrist, tip) / span)

        # Spread between neighbouring fingertips. An open waving palm has all
        # five fingers extended AND splayed; every other gesture here raises
        # one finger from a closed fist. Extension alone doesn't separate those
        # as cleanly as extension plus spread does.
        for a, b in zip(tips, tips[1:]):
            feats.append(_dist(a, b) / span)
        feats.append(float(np.mean([_dist(a, b) / span for a, b in zip(tips, tips[1:])])))

        feats.append(1.0)                       # hand present
    else:
        feats.extend([0.0] * 2)                 # hand position
        feats.extend([0.0] * 10)                # tip offsets
        feats.extend([0.0] * 5)                 # extension ratios
        feats.extend([0.0] * 5)                 # spread
        feats.append(0.0)                       # hand absent

    return feats


GESTURE_FEATURE_NAMES = (
    ['shoulder_tilt_x', 'shoulder_tilt_y']
    + [f'{k}_{a}' for k in ['left_elbow', 'right_elbow', 'left_wrist', 'right_wrist']
       for a in ('x', 'y')]
    + [f'{k}_{a}' for k in ['nose_tip', 'chin', 'forehead'] for a in ('x', 'y')]
    + ['shoulder_above_chin', 'shoulder_above_nose', 'chin_to_shoulder_headunits',
       'shoulder_width_headunits']
    + ['chin_to_shoulder_dist', 'face_present']
    + ['hand_pos_x', 'hand_pos_y']
    + [f'{t}_{a}' for t in FINGER_TIPS for a in ('x', 'y')]
    + [f'{t}_extension' for t in FINGER_TIPS]
    + ['spread_thumb_index', 'spread_index_middle', 'spread_middle_ring',
       'spread_ring_pinky', 'mean_finger_spread']
    + ['hand_present']
)


# ---------------------------------------------------------------- emotions

def emotion_features(landmarks):
    """
    Build the emotion feature vector.

    Reference frame: origin between the eyes, x-axis along the eye line,
    distances in units of interocular distance. That makes every measurement
    invariant to head tilt and camera distance. Interocular distance is the
    scale because, unlike face height, it doesn't change when the jaw drops.

    This returns GEOMETRY ONLY - no raw landmark coordinates. Feeding the raw
    coordinates in was measurably harmful: on their own they scored 51-63%
    against 85% for the geometry, and mixing them in dragged the combined set
    down. With a few hundred samples the model latches onto where a face
    happened to sit rather than what it was doing. Explicit distances, ratios
    and angles don't offer it that shortcut.

    Returns None if there is no face.
    """
    if landmarks is None or 'face' not in landmarks:
        return None

    face = landmarks['face']
    if not all(k in face for k in FACE_POINTS):
        return None

    def p(key):
        return _xy(face[key])

    eye_l = (p('left_eye_top') + p('left_eye_bottom')) / 2.0
    eye_r = (p('right_eye_top') + p('right_eye_bottom')) / 2.0

    origin = (eye_l + eye_r) / 2.0
    axis = eye_r - eye_l
    scale = float(np.linalg.norm(axis))
    if scale < EPS:
        return None

    axis_x = axis / scale
    axis_y = np.array([-axis_x[1], axis_x[0]])   # perpendicular, 90 deg rotation

    def rel(point):
        """Project into the eye-aligned frame, in interocular units."""
        v = point - origin
        return np.array([float(v @ axis_x), float(v @ axis_y)])

    feats = []

    # ---- mouth shape ----
    mouth_w = _dist(p('mouth_left'), p('mouth_right'))
    mouth_h = _dist(p('upper_lip'), p('lower_lip'))
    feats.append(mouth_h / scale)                       # jaw drop
    feats.append(mouth_w / scale)                       # mouth width
    feats.append(mouth_h / (mouth_w + EPS))             # aspect ratio

    # SMILE DIRECTION - corners above the lip centre is a smile, below a frown.
    # Without this the model had no way to tell happy from angry and confused
    # them constantly; it is the single most valuable feature here.
    corner_y = (rel(p('mouth_left'))[1] + rel(p('mouth_right'))[1]) / 2.0
    centre_y = (rel(p('upper_lip'))[1] + rel(p('lower_lip'))[1]) / 2.0
    feats.append(float(centre_y - corner_y))
    feats.append(abs(float(rel(p('mouth_left'))[1] - rel(p('mouth_right'))[1])))

    # ---- eyes ----
    left_open = _dist(p('left_eye_top'), p('left_eye_bottom'))
    right_open = _dist(p('right_eye_top'), p('right_eye_bottom'))
    feats.append(left_open / scale)
    feats.append(right_open / scale)
    feats.append((left_open + right_open) / (2 * scale))
    feats.append(abs(left_open - right_open) / scale)   # squint/wink asymmetry

    # ---- brows ----
    # Height above the eye, inner and outer separately. Surprise lifts the whole
    # brow; anger drives the inner end down while the outer end stays put.
    for side, eye in (('left', eye_l), ('right', eye_r)):
        for end in ('inner', 'outer'):
            feats.append(float(rel(eye)[1] - rel(p(f'{side}_eyebrow_{end}'))[1]))

    # Inner-vs-outer tilt: negative for anger, positive for the raised-inner
    # brow that characterises sadness.
    for side in ('left', 'right'):
        feats.append(float(rel(p(f'{side}_eyebrow_inner'))[1]
                           - rel(p(f'{side}_eyebrow_outer'))[1]))

    # Gap between the inner brow ends, which anger squeezes shut.
    feats.append(_dist(p('left_eyebrow_inner'), p('right_eyebrow_inner')) / scale)

    # ---- proportions ----
    feats.append(_dist(p('chin'), p('nose_tip')) / scale)
    face_height = _dist(p('forehead'), p('chin'))
    feats.append(face_height / scale)

    # Lip compression - anger presses the lips thin, so the gap between the
    # outer lip edge and the inner lip line shrinks.
    feats.append((_dist(p('mouth_top'), p('upper_lip'))
                  + _dist(p('mouth_bottom'), p('lower_lip'))) / scale)

    # Corners pulled up toward the eyes: a genuine smile raises the cheeks.
    feats.append(_dist(p('mouth_left'), eye_l) / scale)
    feats.append(_dist(p('mouth_right'), eye_r) / scale)

    # Nose to upper lip - shortens when the upper lip lifts (disgust, anger).
    feats.append(_dist(p('nose_tip'), p('upper_lip')) / scale)

    # A few measures repeated against FACE HEIGHT. A second independent scale
    # helps when the eye line is foreshortened by head rotation.
    if face_height > EPS:
        feats.append(float(rel(eye_l)[1] - rel(p('left_eyebrow_inner'))[1]) * scale / face_height)
        feats.append(float(rel(eye_r)[1] - rel(p('right_eyebrow_inner'))[1]) * scale / face_height)
        feats.append(mouth_h / face_height)
        feats.append(mouth_w / face_height)
    else:
        feats.extend([0.0] * 4)

    # Mouth centre offset from the face midline (one-sided expressions).
    feats.append(float(rel((p('mouth_left') + p('mouth_right')) / 2.0)[0]))

    return feats


EMOTION_FEATURE_NAMES = (
    ['mouth_open', 'mouth_width', 'mouth_aspect',
     'mouth_corner_lift', 'mouth_asymmetry',
     'left_eye_open', 'right_eye_open', 'mean_eye_open', 'eye_asymmetry']
    + [f'{s}_brow_{e}_raise' for s in ('left', 'right') for e in ('inner', 'outer')]
    + ['left_brow_slope', 'right_brow_slope', 'brow_furrow',
       'chin_to_nose', 'face_height',
       'lip_compression', 'left_corner_to_eye', 'right_corner_to_eye',
       'nose_to_lip',
       'left_brow_raise_headunits', 'right_brow_raise_headunits',
       'mouth_open_headunits', 'mouth_width_headunits',
       'mouth_centre_offset']
)


EXTRACTORS = {'gesture': gesture_features, 'emotion': emotion_features}
FEATURE_NAMES = {'gesture': GESTURE_FEATURE_NAMES, 'emotion': EMOTION_FEATURE_NAMES}


def extract(landmarks, data_type):
    return EXTRACTORS[data_type](landmarks)
