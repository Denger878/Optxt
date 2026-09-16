# overlay.py
"""
Drawing layer for the live view.

Kept separate from main.py so the app loop stays readable, and separate from
landmarks.py so nothing about how it looks can affect what the models see.

The look is editorial rather than sci-fi: warm off-white cards, hairline rules,
a serif for the readings and a monospace for the running transcript. Georgia and
Courier New stand in for Crimson Text and Courier Prime - the same fallbacks the
web version declares, and the only ones guaranteed to exist locally.

Text is rendered with Pillow. OpenCV's putText only has the Hershey vector
fonts, which are single-weight and badly spaced.
"""
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image, ImageDraw, ImageFont

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands

# ---- palette (RGB)
OFF_WHITE = (247, 246, 243)
INK = (26, 26, 26)
INK_BODY = (46, 46, 46)
INK_LIGHT = (102, 102, 102)
BORDER = (226, 221, 216)
RED = (218, 41, 28)

# ---- skeleton colours (BGR; these go through OpenCV)
MESH_LINE = (205, 205, 200)
POSE_LINE = (225, 225, 220)
HAND_LINE = (28, 41, 218)          # the accent red

SERIF = "/System/Library/Fonts/Supplemental/Georgia.ttf"
MONO = "/System/Library/Fonts/Supplemental/Courier New.ttf"
MONO_BOLD = "/System/Library/Fonts/Supplemental/Courier New Bold.ttf"

_FACES = {"serif": SERIF, "mono": MONO, "mono_bold": MONO_BOLD}
_font_cache = {}


def _font(size, face="serif"):
    key = (size, face)
    if key not in _font_cache:
        try:
            _font_cache[key] = ImageFont.truetype(_FACES[face], size)
        except Exception:
            _font_cache[key] = ImageFont.load_default()
    return _font_cache[key]


def _tracked(draw, xy, text, font, fill, tracking):
    """Letter-spaced text. Pillow has no tracking control, so draw per glyph."""
    x, y = xy
    for char in text:
        draw.text((x, y), char, font=font, fill=fill)
        x += draw.textlength(char, font=font) + tracking
    return x - xy[0]


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
        mp_drawing.draw_landmarks(
            image=layer,
            landmark_list=face_results.multi_face_landmarks[0],
            connections=mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing.DrawingSpec(color=MESH_LINE, thickness=1),
        )

    if landmarks and 'pose' in landmarks:
        pose = landmarks['pose']
        for a, b in POSE_BONES:
            cv2.line(layer, _px(pose[a], w, h), _px(pose[b], w, h),
                     POSE_LINE, 1, cv2.LINE_AA)
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

    return cv2.addWeighted(frame, 1.0, layer, 0.5, 0)


def _card(draw, box, radius, opacity=238):
    """Warm off-white card with a hairline border."""
    draw.rounded_rectangle(box, radius=radius, fill=OFF_WHITE + (opacity,))
    draw.rounded_rectangle(box, radius=radius, outline=BORDER + (255,), width=1)


def draw_hud(frame, gesture, gesture_conf, emotion, emotion_conf,
             transcript, muted=False):
    """
    Two cards: the current reading, and a running transcript of what was said.

    `transcript` is a sequence of (elapsed_seconds, sentence), oldest first.
    The transcript exists because this tool narrates out loud - showing the same
    lines on screen makes it legible to someone watching over your shoulder, and
    makes a demo recording self-explanatory.

    Every size derives from the frame height rather than being hard-coded, so
    the layout holds at any resolution or aspect ratio.
    """
    h, w = frame.shape[:2]
    s = h / 480.0

    def sz(v):
        return max(1, int(round(v * s)))

    layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)

    # ── reading card ───────────────────────────────────────────────
    # Just the two readings. No wordmark, no FPS counter - the app only needs
    # to answer "what is this person doing", and anything else is noise the
    # viewer has to look past.
    pad = sz(13)
    px, py = sz(20), sz(20)
    pw = min(sz(206), int(w * 0.40))
    inner = pw - pad * 2
    x = px + pad

    micro_f = _font(sz(8), "mono")
    value_f = _font(sz(20), "serif")
    meta_f = _font(sz(9), "mono")

    rows = [("GESTURE", gesture, gesture_conf), ("EMOTION", emotion, emotion_conf)]
    row_h = sz(10) + sz(30) + sz(10)
    ph = pad * 2 + row_h * len(rows) - sz(8)

    _card(draw, [px, py, px + pw, py + ph], sz(9))

    placeholder = {"no_data", "no_model", "unsure", "no_face"}
    y = py + pad

    for title, label, conf in rows:
        text = str(label).replace("_", " ")
        strong = label not in placeholder
        colour = INK if strong else INK_LIGHT

        _tracked(draw, (x, y), title, micro_f, INK_LIGHT + (255,), sz(1.1))

        pct = f"{conf * 100:.0f}%"
        draw.text((px + pw - pad - draw.textlength(pct, font=meta_f), y - sz(1)),
                  pct, font=meta_f, fill=INK_LIGHT + (255,))

        draw.text((x, y + sz(9)), text, font=value_f, fill=colour + (255,))

        bar_y = y + sz(9) + sz(31)
        bar_h = max(2, sz(2))
        draw.rounded_rectangle([x, bar_y, x + inner, bar_y + bar_h],
                               radius=bar_h // 2, fill=BORDER + (255,))
        filled = int(inner * max(0.0, min(1.0, conf)))
        if filled > bar_h:
            draw.rounded_rectangle([x, bar_y, x + filled, bar_y + bar_h],
                                   radius=bar_h // 2,
                                   fill=(RED if strong else INK_LIGHT) + (255,))
        y += row_h

    # ── transcript card ────────────────────────────────────────────
    line_f = _font(sz(10), "mono")
    time_f = _font(sz(10), "mono_bold")
    lines = list(transcript)[-3:]
    line_h = sz(14)

    tpad = sz(11)
    th = tpad + sz(10) + sz(6) + line_h * max(len(lines), 1) + tpad - sz(4)
    tx0, tx1 = sz(20), w - sz(20)
    ty0 = h - sz(20) - th

    _card(draw, [tx0, ty0, tx1, ty0 + th], sz(9))

    ty = ty0 + tpad - sz(2)
    _tracked(draw, (tx0 + tpad, ty), "TRANSCRIPT", micro_f, INK_LIGHT + (255,), sz(1.1))

    hint = "Q quit   M mute   S wireframe"
    if muted:
        hint = "MUTED   " + hint
    draw.text((tx1 - tpad - draw.textlength(hint, font=micro_f), ty),
              hint, font=micro_f, fill=INK_LIGHT + (255,))

    ty += sz(13)
    draw.line([tx0 + tpad, ty, tx1 - tpad, ty], fill=BORDER + (255,), width=1)
    ty += sz(6)

    if not lines:
        draw.text((tx0 + tpad, ty), "listening...", font=line_f,
                  fill=INK_LIGHT + (255,))
    else:
        # Older lines fade back so the newest reads first.
        for i, (elapsed, sentence) in enumerate(lines):
            newest = i == len(lines) - 1
            colour = INK if newest else INK_LIGHT
            stamp = f"{int(elapsed) // 60:02d}:{int(elapsed) % 60:02d}"
            draw.text((tx0 + tpad, ty), stamp, font=time_f,
                      fill=(RED if newest else INK_LIGHT) + (255,))
            offset = draw.textlength("00:00  ", font=time_f)
            avail = (tx1 - tpad) - (tx0 + tpad + offset)
            text = sentence
            while draw.textlength(text, font=line_f) > avail and len(text) > 4:
                text = text[:-2]
            draw.text((tx0 + tpad + offset, ty), text, font=line_f,
                      fill=colour + (255,))
            ty += line_h

    rgba = np.array(layer)
    alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
    rgb = rgba[:, :, :3][:, :, ::-1].astype(np.float32)      # RGB -> BGR
    return (frame.astype(np.float32) * (1 - alpha) + rgb * alpha).astype(np.uint8)
