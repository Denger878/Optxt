# overlay.py
"""
Drawing layer for the live view.

Kept separate from main.py so the app loop stays readable, and separate from
landmarks.py so nothing about how it looks can affect what the models see.

The look is instrument panel rather than print: black translucent plates, a
monospace face throughout, square corners and hard 1px rules. Nothing is
anti-aliased except the text - the slight aliasing on the wireframe is what
keeps it reading as a readout instead of an illustration.
"""
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image, ImageDraw, ImageFont

mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands

# ---- palette (RGB)
TEXT = (226, 238, 241)
DIM = (124, 148, 156)
ACCENT = (94, 206, 226)
WARN = (255, 176, 74)
PLATE = (0, 0, 0)

# ---- wireframe colours (BGR; these go through OpenCV)
FACE_LINE = (168, 150, 66)
FACE_NODE = (226, 212, 140)
POSE_LINE = (188, 168, 74)
POSE_NODE = (236, 224, 158)
HAND_LINE = (74, 176, 255)
HAND_NODE = (150, 214, 255)

MONO = "/System/Library/Fonts/Supplemental/Courier New.ttf"
MONO_BOLD = "/System/Library/Fonts/Supplemental/Courier New Bold.ttf"

_FACES = {"mono": MONO, "bold": MONO_BOLD}
_font_cache = {}

# Unique landmark indices touched by the contour set, so each vertex gets a node
# drawn exactly once instead of once per edge.
_FACE_EDGES = sorted(mp_face_mesh.FACEMESH_CONTOURS)
_FACE_NODES = sorted({i for edge in _FACE_EDGES for i in edge})
_HAND_EDGES = sorted(mp_hands.HAND_CONNECTIONS)


def _font(size, face="mono"):
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


def _node(layer, centre, colour, r):
    """Square vertex marker. Squares read as sampled points; circles read as decoration."""
    x, y = centre
    cv2.rectangle(layer, (x - r, y - r), (x + r, y + r), colour, -1)


def draw_skeleton(frame, landmarks, raw):
    """
    Wireframe: hard lines with a visible node at every vertex.

    Drawn without anti-aliasing and with square nodes on purpose. Smooth
    contours look like a drawing of a face; stepped lines with marked vertices
    look like something measuring one, which is what is actually happening.

    Still contours only, not the full tesselation - that draws all 468 points
    and every triangle between them, which covers the face like a mask.
    """
    h, w = frame.shape[:2]
    face_results, pose_results, hand_results = raw
    layer = np.zeros_like(frame)
    r = max(1, int(round(h / 480.0)))

    if face_results.multi_face_landmarks:
        lms = face_results.multi_face_landmarks[0].landmark
        for a, b in _FACE_EDGES:
            cv2.line(layer,
                     (int(lms[a].x * w), int(lms[a].y * h)),
                     (int(lms[b].x * w), int(lms[b].y * h)),
                     FACE_LINE, 1)
        for i in _FACE_NODES:
            _node(layer, (int(lms[i].x * w), int(lms[i].y * h)), FACE_NODE, r)

    if landmarks and 'pose' in landmarks:
        pose = landmarks['pose']
        for a, b in POSE_BONES:
            cv2.line(layer, _px(pose[a], w, h), _px(pose[b], w, h), POSE_LINE, 1)
        for key in pose:
            _node(layer, _px(pose[key], w, h), POSE_NODE, r + 1)

    if hand_results.multi_hand_landmarks:
        for hand in hand_results.multi_hand_landmarks:
            lms = hand.landmark
            for a, b in _HAND_EDGES:
                cv2.line(layer,
                         (int(lms[a].x * w), int(lms[a].y * h)),
                         (int(lms[b].x * w), int(lms[b].y * h)),
                         HAND_LINE, 1)
            for lm in lms:
                _node(layer, (int(lm.x * w), int(lm.y * h)), HAND_NODE, r)

    return cv2.addWeighted(frame, 1.0, layer, 0.85, 0)


def _plate(draw, box, accent=ACCENT, opacity=178, ticks=True):
    """
    Black translucent plate: square corners, a hairline frame, corner ticks.

    Rounded corners and soft fills were the thing that read as 'website'. Square
    corners with brackets read as an instrument.
    """
    x0, y0, x1, y1 = box
    draw.rectangle(box, fill=PLATE + (opacity,))
    draw.rectangle(box, outline=accent + (70,), width=1)

    if not ticks:
        return
    t = max(4, int((x1 - x0) * 0.035))
    for cx, cy, dx, dy in ((x0, y0, 1, 1), (x1, y0, -1, 1),
                           (x0, y1, 1, -1), (x1, y1, -1, -1)):
        draw.line([cx, cy, cx + t * dx, cy], fill=accent + (210,), width=1)
        draw.line([cx, cy, cx, cy + t * dy], fill=accent + (210,), width=1)


def draw_hud(frame, gesture, gesture_conf, emotion, emotion_conf,
             transcript, muted=False):
    """
    Two plates: the current reading, and a running transcript of what was said.

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

    # ── reading plate ──────────────────────────────────────────────
    pad = sz(13)
    px, py = sz(20), sz(20)
    pw = min(sz(214), int(w * 0.42))
    inner = pw - pad * 2
    x = px + pad

    micro_f = _font(sz(8), "mono")
    value_f = _font(sz(18), "bold")
    meta_f = _font(sz(9), "mono")

    rows = [("GESTURE", gesture, gesture_conf), ("EMOTION", emotion, emotion_conf)]
    row_h = sz(10) + sz(28) + sz(10)
    ph = pad * 2 + row_h * len(rows) - sz(8)

    _plate(draw, [px, py, px + pw, py + ph])

    placeholder = {"no_data", "no_model", "unsure", "no_face"}
    y = py + pad

    for title, label, conf in rows:
        text = str(label).replace("_", " ").upper()
        strong = label not in placeholder
        colour = TEXT if strong else DIM

        _tracked(draw, (x, y), title, micro_f, DIM + (255,), sz(1.1))

        pct = f"{conf * 100:.0f}%"
        draw.text((px + pw - pad - draw.textlength(pct, font=meta_f), y - sz(1)),
                  pct, font=meta_f, fill=DIM + (255,))

        draw.text((x, y + sz(9)), text, font=value_f, fill=colour + (255,))

        # Segmented meter - discrete cells rather than a continuous bar, so the
        # confidence reads as a quantity being counted, not a progress bar.
        bar_y = y + sz(9) + sz(28)
        bar_h = max(3, sz(4))
        cells = 20
        gap = max(1, sz(1))
        cw = (inner - gap * (cells - 1)) / cells
        lit = int(round(cells * max(0.0, min(1.0, conf))))
        for c in range(cells):
            cx0 = x + c * (cw + gap)
            on = c < lit
            draw.rectangle([cx0, bar_y, cx0 + cw, bar_y + bar_h],
                           fill=((ACCENT if strong else DIM) + (235,)) if on
                           else (46, 56, 60, 232))
        y += row_h

    # ── transcript plate ───────────────────────────────────────────
    line_f = _font(sz(10), "mono")
    time_f = _font(sz(10), "bold")
    lines = list(transcript)[-3:]
    line_h = sz(14)

    tpad = sz(11)
    th = tpad + sz(10) + sz(6) + line_h * max(len(lines), 1) + tpad - sz(4)
    tx0, tx1 = sz(20), w - sz(20)
    ty0 = h - sz(20) - th

    _plate(draw, [tx0, ty0, tx1, ty0 + th])

    ty = ty0 + tpad - sz(2)
    _tracked(draw, (tx0 + tpad, ty), "TRANSCRIPT", micro_f, DIM + (255,), sz(1.1))

    hint = "Q QUIT   M MUTE   S WIRE"
    hint_col = DIM
    if muted:
        hint = "// MUTED //   " + hint
        hint_col = WARN
    draw.text((tx1 - tpad - draw.textlength(hint, font=micro_f), ty),
              hint, font=micro_f, fill=hint_col + (255,))

    ty += sz(13)
    draw.line([tx0 + tpad, ty, tx1 - tpad, ty], fill=ACCENT + (60,), width=1)
    ty += sz(6)

    if not lines:
        draw.text((tx0 + tpad, ty), "> awaiting signal", font=line_f,
                  fill=DIM + (255,))
    else:
        # Older lines fade back so the newest reads first.
        for i, (elapsed, sentence) in enumerate(lines):
            newest = i == len(lines) - 1
            colour = TEXT if newest else DIM
            stamp = f"{int(elapsed) // 60:02d}:{int(elapsed) % 60:02d}"
            draw.text((tx0 + tpad, ty), stamp, font=time_f,
                      fill=(ACCENT if newest else DIM) + (255,))
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
