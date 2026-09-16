# overlay.py
"""
Drawing layer for the live view.

Kept separate from main.py so the app loop stays readable, and separate from
landmarks.py so nothing about how it looks can affect what the models see.

Text is rendered with Pillow using a real system typeface. OpenCV's built-in
putText only has the Hershey vector fonts, which are single-weight, badly
spaced and the reason the old HUD looked like a school science project.
"""
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image, ImageDraw, ImageFont

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands

# ---- palette (RGB). Deliberately plain: white text, grey labels, one accent.
INK = (255, 255, 255)
MUTED = (178, 178, 184)
FAINT = (138, 138, 144)
ACCENT = (90, 175, 255)
PANEL_FILL = (14, 14, 16)

# ---- skeleton colours (BGR; these go through OpenCV)
MESH_LINE = (150, 120, 85)
POSE_LINE = (150, 190, 150)
HAND_LINE = (200, 150, 235)

FONTS = "/System/Library/Fonts/HelveticaNeue.ttc"
_WEIGHTS = {"regular": 0, "bold": 1}
_font_cache = {}


def _font(size, weight="regular"):
    key = (size, weight)
    if key not in _font_cache:
        try:
            _font_cache[key] = ImageFont.truetype(FONTS, size, index=_WEIGHTS[weight])
        except Exception:
            _font_cache[key] = ImageFont.load_default()
    return _font_cache[key]


POSE_BONES = [
    ('left_shoulder', 'right_shoulder'),
    ('left_shoulder', 'left_elbow'), ('left_elbow', 'left_wrist'),
    ('right_shoulder', 'right_elbow'), ('right_elbow', 'right_wrist'),
]


def _px(point, w, h):
    return int(point[0] * w), int(point[1] * h)


def draw_skeleton(frame, landmarks, raw):
    """
    Draw a restrained wireframe: face contours, pose, hands.

    Deliberately NOT the full face tesselation. That draws all 468 points and
    every triangle between them, which covers the face like a mask and buries
    the person underneath it. Contours trace the features that actually carry
    expression - brows, eyes, lips, jawline - and leave the face visible.
    """
    h, w = frame.shape[:2]
    face_results, pose_results, hand_results = raw
    layer = np.zeros_like(frame)

    if face_results.multi_face_landmarks:
        face = face_results.multi_face_landmarks[0]
        mp_drawing.draw_landmarks(
            image=layer,
            landmark_list=face,
            connections=mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing.DrawingSpec(color=MESH_LINE, thickness=1),
        )

    if landmarks and 'pose' in landmarks:
        pose = landmarks['pose']
        for a, b in POSE_BONES:
            cv2.line(layer, _px(pose[a], w, h), _px(pose[b], w, h),
                     POSE_LINE, 2, cv2.LINE_AA)
        for key in pose:
            cv2.circle(layer, _px(pose[key], w, h), 3, POSE_LINE, -1, cv2.LINE_AA)

    if hand_results.multi_hand_landmarks:
        for hand in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image=layer,
                landmark_list=hand,
                connections=mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(
                    color=HAND_LINE, thickness=1, circle_radius=2),
                connection_drawing_spec=mp_drawing.DrawingSpec(
                    color=HAND_LINE, thickness=1),
            )

    # Blend rather than paste, so the wireframe sits over the video at partial
    # strength instead of stamping hard lines on top of it.
    return cv2.addWeighted(frame, 1.0, layer, 0.55, 0)


def draw_hud(frame, gesture, gesture_conf, emotion, emotion_conf, fps, last_spoken):
    """
    Status panel: what's detected, how sure the models are, what was last said.

    Every size and position is derived from the frame dimensions rather than
    hard-coded, so the layout holds together at any resolution or aspect ratio.
    A fixed-pixel HUD laid out for 640x480 turns into unreadable specks on a
    1080p or portrait-phone frame.
    """
    h, w = frame.shape[:2]
    s = h / 480.0                      # scale factor; 480p is the reference

    def sz(value):
        return max(1, int(round(value * s)))

    layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)

    pad = sz(22)
    px, py = sz(24), sz(24)
    pw = min(sz(330), int(w * 0.55))
    label_f = _font(sz(15), "bold")
    value_f = _font(sz(30), "regular")
    small_f = _font(sz(15), "regular")

    row_h = sz(30) + sz(15) + sz(30)   # label + value + bar block
    ph = pad * 2 + row_h * 2 + sz(18)

    draw.rounded_rectangle([px, py, px + pw, py + ph], radius=sz(12),
                           fill=PANEL_FILL + (205,))

    placeholder = {"no_data", "no_model", "unsure", "no_face"}
    rows = [("GESTURE", gesture, gesture_conf),
            ("EMOTION", emotion, emotion_conf)]

    x = px + pad
    y = py + pad
    inner = pw - pad * 2

    for title, label, conf in rows:
        text = str(label).replace("_", " ")
        colour = INK if label not in placeholder else FAINT

        draw.text((x, y), title, font=label_f, fill=MUTED + (255,))

        pct = f"{conf * 100:.0f}%"
        draw.text((x + inner - draw.textlength(pct, font=small_f), y + sz(1)),
                  pct, font=small_f, fill=MUTED + (255,))

        draw.text((x, y + sz(21)), text, font=value_f, fill=colour + (255,))

        bar_y = y + sz(21) + sz(36)
        bar_h = sz(5)
        draw.rounded_rectangle([x, bar_y, x + inner, bar_y + bar_h],
                               radius=bar_h // 2, fill=(52, 52, 58, 255))
        filled = int(inner * max(0.0, min(1.0, conf)))
        if filled > bar_h:
            draw.rounded_rectangle([x, bar_y, x + filled, bar_y + bar_h],
                                   radius=bar_h // 2,
                                   fill=(ACCENT if colour is INK else FAINT) + (255,))

        y += row_h + sz(18)

    # FPS, bottom-right corner of the panel, small and out of the way.
    fps_text = f"{fps:.0f} FPS"
    draw.text((px + pw - pad - draw.textlength(fps_text, font=small_f),
               py + ph - pad - sz(6)),
              fps_text, font=small_f, fill=FAINT + (255,))

    # Key hints - large enough to actually read. These were 10px grey before,
    # which made them effectively invisible.
    def pill(x, y, text, font, colour):
        tw = draw.textlength(text, font=font)
        draw.rounded_rectangle([x - sz(14), y - sz(9), x + tw + sz(14), y + sz(27)],
                               radius=sz(10), fill=PANEL_FILL + (200,))
        draw.text((x, y), text, font=font, fill=colour + (255,))
        return tw

    hint_f = _font(sz(17), "regular")
    pill(px + sz(14), h - sz(46), "Q  quit       M  mute       S  wireframe",
         hint_f, MUTED)

    # Caption sits on its own line ABOVE the hints so the two can never collide.
    if last_spoken:
        cap_f = _font(sz(19), "regular")
        text = last_spoken if len(last_spoken) < 70 else last_spoken[:67] + "..."
        tw = draw.textlength(text, font=cap_f)
        pill(max(sz(38), (w - tw) / 2), h - sz(100), text, cap_f, INK)

    rgba = np.array(layer)
    alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
    rgb = rgba[:, :, :3][:, :, ::-1].astype(np.float32)      # RGB -> BGR
    return (frame.astype(np.float32) * (1 - alpha) + rgb * alpha).astype(np.uint8)
