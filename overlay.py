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
# Three colours only. White carries every piece of text - secondary text is the
# same white at lower opacity rather than a grey, so nothing is a near-miss of
# anything else. Pure blue carries every structural element. Amber appears for
# one thing, muting, and is deliberately nothing like either.
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
AMBER = (255, 170, 40)

# ---- wireframe colours (BGR; these go through OpenCV)
WIRE_LINE = (255, 0, 0)            # the same pure blue
WIRE_NODE = (255, 255, 255)        # white, so vertices read against the lines

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
    r = max(2, int(round(h / 480.0)) + 1)
    t = max(2, int(round(h / 640.0)) + 1)

    if face_results.multi_face_landmarks:
        lms = face_results.multi_face_landmarks[0].landmark
        for a, b in _FACE_EDGES:
            cv2.line(layer,
                     (int(lms[a].x * w), int(lms[a].y * h)),
                     (int(lms[b].x * w), int(lms[b].y * h)),
                     WIRE_LINE, t)
        for i in _FACE_NODES:
            _node(layer, (int(lms[i].x * w), int(lms[i].y * h)), WIRE_NODE, r)

    if landmarks and 'pose' in landmarks:
        pose = landmarks['pose']
        for a, b in POSE_BONES:
            cv2.line(layer, _px(pose[a], w, h), _px(pose[b], w, h), WIRE_LINE, t)
        for key in pose:
            _node(layer, _px(pose[key], w, h), WIRE_NODE, r + 1)

    if hand_results.multi_hand_landmarks:
        for hand in hand_results.multi_hand_landmarks:
            lms = hand.landmark
            for a, b in _HAND_EDGES:
                cv2.line(layer,
                         (int(lms[a].x * w), int(lms[a].y * h)),
                         (int(lms[b].x * w), int(lms[b].y * h)),
                         WIRE_LINE, t)
            for lm in lms:
                _node(layer, (int(lm.x * w), int(lm.y * h)), WIRE_NODE, r)

    return cv2.addWeighted(frame, 1.0, layer, 0.85, 0)


def _plate(draw, box, opacity=178, width=2):
    """
    Black translucent plate with a single uniform border.

    No corner brackets - they framed the panel like a photo instead of reading
    as part of one instrument. The border is 2px because a hairline disappears
    against a moving video feed.
    """
    draw.rectangle(box, fill=(0, 0, 0, opacity))
    draw.rectangle(box, outline=BLUE + (255,), width=width)


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
    # Secondary text is white at reduced opacity, not a grey - same colour,
    # less of it.
    STRONG = WHITE + (255,)
    SOFT = WHITE + (150,)
    RULE = BLUE + (255,)

    pad = sz(13)
    px, py = sz(20), sz(20)
    pw = min(sz(214), int(w * 0.42))
    inner = pw - pad * 2
    x = px + pad

    # Every small label is bold. At these sizes the regular weight rendered as
    # a grey haze against moving video.
    micro_f = _font(sz(9), "bold")
    value_f = _font(sz(18), "bold")
    meta_f = _font(sz(10), "bold")

    rows = [("GESTURE", gesture, gesture_conf), ("EMOTION", emotion, emotion_conf)]
    row_h = sz(10) + sz(28) + sz(10)
    ph = pad * 2 + row_h * len(rows) - sz(8)

    _plate(draw, [px, py, px + pw, py + ph])

    placeholder = {"no_data", "no_model", "unsure", "no_face"}
    y = py + pad

    for title, label, conf in rows:
        text = str(label).replace("_", " ").upper()
        strong = label not in placeholder

        _tracked(draw, (x, y), title, micro_f, SOFT, sz(1.1))

        pct = f"{conf * 100:.0f}%"
        draw.text((px + pw - pad - draw.textlength(pct, font=meta_f), y - sz(1)),
                  pct, font=meta_f, fill=SOFT)

        draw.text((x, y + sz(9)), text, font=value_f,
                  fill=STRONG if strong else SOFT)

        # Segmented meter - discrete cells rather than a continuous bar, so the
        # confidence reads as a quantity being counted, not a progress bar.
        # Unlit cells are a dark version of the same blue; as translucent white
        # they tinted with whatever was behind the frame.
        bar_y = y + sz(9) + sz(28)
        bar_h = max(4, sz(5))
        cells = 20
        gap = max(2, sz(2))
        lit = int(round(cells * max(0.0, min(1.0, conf))))
        for c in range(cells):
            # Snap both edges to whole pixels, or the gaps come out ragged.
            cx0 = x + round(c * (inner + gap) / cells)
            cx1 = x + round((c + 1) * (inner + gap) / cells) - gap
            if c < lit:
                fill = BLUE + (255,) if strong else SOFT
            else:
                fill = (0, 0, 86, 240)
            draw.rectangle([cx0, bar_y, cx1, bar_y + bar_h], fill=fill)
        y += row_h

    # ── transcript plate ───────────────────────────────────────────
    line_f = _font(sz(11), "bold")
    time_f = _font(sz(11), "bold")
    lines = list(transcript)[-3:]
    line_h = sz(15)

    tpad = sz(11)
    th = tpad + sz(11) + sz(7) + line_h * max(len(lines), 1) + tpad - sz(4)
    tx0, tx1 = sz(20), w - sz(20)
    ty0 = h - sz(20) - th

    _plate(draw, [tx0, ty0, tx1, ty0 + th])

    ty = ty0 + tpad - sz(2)
    _tracked(draw, (tx0 + tpad, ty), "TRANSCRIPT", micro_f, SOFT, sz(1.1))

    hint = "Q QUIT   M MUTE   S WIRE"
    hint_col = SOFT
    if muted:
        hint = "// MUTED //   " + hint
        hint_col = AMBER + (255,)
    draw.text((tx1 - tpad - draw.textlength(hint, font=micro_f), ty),
              hint, font=micro_f, fill=hint_col)

    ty += sz(14)
    draw.line([tx0 + tpad, ty, tx1 - tpad, ty], fill=RULE, width=max(2, sz(2)))
    ty += sz(7)

    if not lines:
        draw.text((tx0 + tpad, ty), "> AWAITING SIGNAL", font=line_f, fill=SOFT)
    else:
        # Older lines fade back so the newest reads first.
        for i, (elapsed, sentence) in enumerate(lines):
            newest = i == len(lines) - 1
            stamp = f"{int(elapsed) // 60:02d}:{int(elapsed) % 60:02d}"
            draw.text((tx0 + tpad, ty), stamp, font=time_f,
                      fill=STRONG if newest else SOFT)
            offset = draw.textlength("00:00  ", font=time_f)
            avail = (tx1 - tpad) - (tx0 + tpad + offset)
            text = sentence
            while draw.textlength(text, font=line_f) > avail and len(text) > 4:
                text = text[:-2]
            draw.text((tx0 + tpad + offset, ty), text, font=line_f,
                      fill=STRONG if newest else SOFT)
            ty += line_h

    rgba = np.array(layer)
    alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
    rgb = rgba[:, :, :3][:, :, ::-1].astype(np.float32)      # RGB -> BGR
    return (frame.astype(np.float32) * (1 - alpha) + rgb * alpha).astype(np.uint8)
