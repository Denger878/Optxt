# main.py
import time
from collections import deque

import cv2

from landmarks import extract_landmarks
from gestures import detect_gesture
from emotions import detect_emotion
from speech import say_interaction, toggle_mute, is_muted
from state_tracker import StateTracker
from overlay import draw_skeleton, draw_hud

ANNOUNCEMENT_COOLDOWN = 4.0   # seconds between spoken updates


def main():
    print("=" * 50)
    print("OPTXT")
    print("=" * 50)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Camera not found")
        return

    print("✅ Webcam ready")

    tracker = StateTracker(stability=5)

    print("🎬 Running... 'q' quit, 'm' mute, 's' toggle skeleton")
    print("=" * 50)

    last_announcement_time = 0.0
    show_skeleton = True

    # Everything the app has said this session, stamped from launch. The HUD
    # shows the tail of it as a transcript.
    session_start = time.time()
    transcript = deque(maxlen=12)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read frame")
            break

        # Mirror the view so moving right on screen matches moving right in life.
        frame = cv2.flip(frame, 1)

        landmarks, raw = extract_landmarks(frame, return_raw=True)

        gesture, gesture_conf = "no_data", 0.0
        emotion, emotion_conf = "no_data", 0.0

        if landmarks:
            gesture, gesture_conf = detect_gesture(landmarks, with_confidence=True)
            emotion, emotion_conf = detect_emotion(landmarks, with_confidence=True)

            now = time.time()
            if (now - last_announcement_time) >= ANNOUNCEMENT_COOLDOWN:
                changed, message = tracker.update(gesture, emotion)
                if changed:
                    print(f">>> {message}")
                    say_interaction(message)
                    transcript.append((now - session_start, message))
                    last_announcement_time = now
            else:
                # Keep feeding the stability window even during the cooldown,
                # so the next eligible frame reflects a settled reading.
                tracker.update(gesture, emotion)

        display_frame = frame
        if show_skeleton:
            display_frame = draw_skeleton(display_frame, landmarks, raw)

        display_frame = draw_hud(
            display_frame,
            gesture, gesture_conf,
            emotion, emotion_conf,
            transcript,
            muted=is_muted(),
        )

        cv2.imshow("OPTXT", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('m'):
            print("🔇 Muted" if toggle_mute() else "🔊 Unmuted")
        elif key == ord('s'):
            show_skeleton = not show_skeleton

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
