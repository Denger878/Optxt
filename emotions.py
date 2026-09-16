# emotions.py
import os

import numpy as np
import joblib

from features import emotion_features

MODEL_PATH = os.path.join(os.path.dirname(__file__), "training/models/emotion_model.pkl")

# See gestures.py - same reasoning for refusing to guess out loud.
CONFIDENCE_THRESHOLD = 0.55

try:
    emotion_model = joblib.load(MODEL_PATH)
    print("✅ Emotion model loaded successfully")
except FileNotFoundError:
    emotion_model = None
    print("⚠️  Emotion model not found - train it first!")
    print(f"   Looking for: {MODEL_PATH}")


def detect_emotion(landmarks, with_confidence=False):
    """
    Predict an emotion from one frame of landmarks.

    Returns the label, or (label, confidence) when with_confidence is set.
    """
    def result(label, conf):
        return (label, conf) if with_confidence else label

    if emotion_model is None:
        return result("no_model", 0.0)

    features = emotion_features(landmarks)
    if features is None:
        return result("no_face", 0.0)

    features_array = np.array(features).reshape(1, -1)

    probabilities = emotion_model.predict_proba(features_array)[0]
    best = int(np.argmax(probabilities))
    confidence = float(probabilities[best])
    label = emotion_model.classes_[best]

    if confidence < CONFIDENCE_THRESHOLD:
        return result("unsure", confidence)

    return result(label, confidence)


if __name__ == "__main__":
    print("Testing emotion detection...")
    print(f"Model loaded: {emotion_model is not None}")
