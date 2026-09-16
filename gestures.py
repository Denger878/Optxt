# gestures.py
import os

import numpy as np
import joblib

from features import gesture_features

MODEL_PATH = os.path.join(os.path.dirname(__file__), "training/models/gesture_model.pkl")

# Below this much confidence we say "unsure" rather than announce a guess.
# The app speaks out loud to someone who can't check the screen, so a confident
# wrong answer is worse than admitting uncertainty.
CONFIDENCE_THRESHOLD = 0.55

try:
    gesture_model = joblib.load(MODEL_PATH)
    print("✅ Gesture model loaded successfully")
except FileNotFoundError:
    gesture_model = None
    print("⚠️  Gesture model not found - train it first!")


def detect_gesture(landmarks, with_confidence=False):
    """
    Predict a gesture from one frame of landmarks.

    Returns the label, or (label, confidence) when with_confidence is set.
    """
    def result(label, conf):
        return (label, conf) if with_confidence else label

    if gesture_model is None:
        return result("no_model", 0.0)

    features = gesture_features(landmarks)
    if features is None:
        return result("no_data", 0.0)

    features_array = np.array(features).reshape(1, -1)

    probabilities = gesture_model.predict_proba(features_array)[0]
    best = int(np.argmax(probabilities))
    confidence = float(probabilities[best])
    label = gesture_model.classes_[best]

    if confidence < CONFIDENCE_THRESHOLD:
        return result("unsure", confidence)

    return result(label, confidence)


if __name__ == "__main__":
    print("Testing gesture detection...")
    print(f"Model loaded: {gesture_model is not None}")
