# predict.py
"""
Display-free prediction entry point.

main.py is built around OpenCV windows and a webcam, neither of which exists on
a server. This module takes landmarks in and returns plain JSON-serializable
data, so the same models can sit behind an HTTP handler later without dragging
the UI along.

Nothing here imports cv2 or mediapipe - only the feature code and the models.
"""
from gestures import detect_gesture
from emotions import detect_emotion
from state_tracker import StateTracker


def describe(landmarks):
    """
    Read one frame of landmarks and report what's there.

    `landmarks` is the dict produced by landmarks.extract_landmarks - which is
    also exactly the shape MediaPipe's JavaScript build produces in a browser,
    so a web client can post these straight up.
    """
    gesture, gesture_conf = detect_gesture(landmarks, with_confidence=True)
    emotion, emotion_conf = detect_emotion(landmarks, with_confidence=True)

    return {
        "gesture": gesture,
        "gesture_confidence": round(float(gesture_conf), 4),
        "emotion": emotion,
        "emotion_confidence": round(float(emotion_conf), 4),
        "sentence": StateTracker.describe(gesture, emotion),
    }


if __name__ == "__main__":
    import json
    import glob

    sample = glob.glob("training/data/gesture/thumbsup_*.json")
    if sample:
        frames = json.load(open(sample[0]))
        print(json.dumps(describe(frames[len(frames) // 2]["landmarks"]), indent=2))
