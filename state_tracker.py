# state_tracker.py
from collections import deque


class StateTracker:
    """
    Decides when a detection is worth saying out loud.

    Single-frame predictions flicker, and every flicker used to trigger an
    announcement. A label now has to hold steady for `stability` consecutive
    frames before it counts as the current state, and only a real change to
    that stable state produces a message.
    """

    # Predictions that mean "nothing useful here" rather than a real detection.
    IGNORED = {"no_data", "no_model", "no_face", "unsure"}

    def __init__(self, stability=5):
        self.stability = stability
        self.gesture_history = deque(maxlen=stability)
        self.emotion_history = deque(maxlen=stability)
        self.current_gesture = None
        self.current_emotion = None

    def _stable(self, history):
        """The label if the whole window agrees on it, otherwise None."""
        if len(history) < self.stability:
            return None
        first = history[0]
        return first if all(item == first for item in history) else None

    def update(self, new_gesture, new_emotion):
        self.gesture_history.append(new_gesture)
        self.emotion_history.append(new_emotion)

        stable_gesture = self._stable(self.gesture_history)
        stable_emotion = self._stable(self.emotion_history)

        changed = False

        if stable_gesture is not None and stable_gesture != self.current_gesture:
            self.current_gesture = stable_gesture
            changed = True

        if stable_emotion is not None and stable_emotion != self.current_emotion:
            self.current_emotion = stable_emotion
            changed = True

        if not changed:
            return False, ""

        # Describe only the parts we actually have a reading for.
        gesture_ok = self.current_gesture not in self.IGNORED and self.current_gesture
        emotion_ok = self.current_emotion not in self.IGNORED and self.current_emotion

        message = self.describe(self.current_gesture, self.current_emotion)
        if not message:
            return False, ""

        return True, message

    @classmethod
    def describe(cls, gesture, emotion):
        """Phrase a single reading, with no tracking state involved."""
        gesture_ok = gesture and gesture not in cls.IGNORED
        emotion_ok = emotion and emotion not in cls.IGNORED
        if gesture_ok and emotion_ok:
            return f"They are {cls._phrase(gesture)} and look {emotion}"
        if gesture_ok:
            return f"They are {cls._phrase(gesture)}"
        if emotion_ok:
            return f"They look {emotion}"
        return ""

    @staticmethod
    def _phrase(gesture):
        """Turn a label into something that sounds like English when spoken."""
        return {
            "thumbsup": "giving a thumbs up",
            "thumbsdown": "giving a thumbs down",
            "pointingatyou": "pointing at you",
            "middlefinger": "making a rude gesture",
            "waving": "waving at you",
            "neutral": "standing neutrally",
        }.get(gesture, gesture)
