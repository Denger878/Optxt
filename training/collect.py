# training/collect.py
import cv2
import sys
import os
import time
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from landmarks import extract_landmarks
from features import extract
from overlay import draw_skeleton

GESTURES = ["thumbsup", "thumbsdown", "pointing", "middlefinger",
            "waving", "neutral"]
EMOTIONS = ["shocked", "angry", "happy", "sad", "neutral"]

GESTURE_HELP = {
    "thumbsup": "Fist with thumb pointing UP",
    "thumbsdown": "Fist with thumb pointing DOWN",
    "pointing": "Point index finger at the camera",
    "middlefinger": "Middle finger up",
    "waving": "Open palm up at head height, fingers SPREAD WIDE - hold it",
    "neutral": "Sit still, hands DOWN in your lap, shoulders RELAXED and low",
}

EMOTION_HELP = {
    "shocked": "Eyes SUPER WIDE, eyebrows HIGH, mouth in a big O",
    "angry": "Furrow eyebrows hard, frown, press lips thin, tighten jaw",
    "sad": "Mouth corners DOWN, inner eyebrows UP, eyes lowered",
    "happy": "Biggest smile possible, show teeth, raise cheeks",
    "neutral": "Completely relaxed resting face",
}

# Varying these between takes is what teaches the model the gesture itself
# rather than the exact spot you happened to be sitting in.
VARIATIONS = [
    "sitting at your normal distance from the camera",
    "further back, so your whole torso is visible",
    "closer to the camera, and slightly off to one side",
    "with your head tilted a little, and different lighting if you can",
]


def collect_data(label, duration=5, data_type="gesture", note=""):
    """Record one take of one label and save it as its own file."""
    save_dir = os.path.join(os.path.dirname(__file__), f"data/{data_type}")
    os.makedirs(save_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Camera not found")
        return 0

    frames_data = []
    usable = 0

    print(f"\n=== Recording: {label} ===")
    if note:
        print(f"    {note}")
    print("Get ready...")
    time.sleep(2)
    print(f"GO! HOLD THE POSE FOR {duration} SECONDS!")

    start_time = time.time()

    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        landmarks, raw = extract_landmarks(frame, return_raw=True)

        if landmarks:
            frames_data.append({
                'timestamp': time.time() - start_time,
                'landmarks': landmarks
            })
            # Check the frame would actually survive feature extraction, so a
            # take that produces nothing usable is caught now, not at training.
            if extract(landmarks, data_type) is not None:
                usable += 1

        frame = draw_skeleton(frame, landmarks, raw)

        remaining = duration - (time.time() - start_time)
        cv2.putText(frame, f"{label}: {remaining:.1f}s", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (120, 255, 120), 2, cv2.LINE_AA)
        cv2.putText(frame, "HOLD STEADY!", (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 190, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"usable frames: {usable}", (20, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow('Recording', frame)
        cv2.waitKey(1)

    cap.release()
    cv2.destroyAllWindows()

    if usable == 0:
        print("⚠️  No usable frames - the required landmarks were never detected.")
        print("    Check you're fully in frame and lit, then redo this take.")
        return 0

    # The timestamp makes each take a separate file, which is what lets
    # train.py keep whole takes out of the test set.
    filename = f"{save_dir}/{label}_{int(time.time())}.json"
    with open(filename, 'w') as f:
        json.dump(frames_data, f)

    print(f"✓ Saved {len(frames_data)} frames ({usable} usable) to {os.path.basename(filename)}")
    return usable


def existing_takes(data_type):
    """Count how many takes already exist per label."""
    data_dir = os.path.join(os.path.dirname(__file__), f"data/{data_type}")
    counts = {}
    if os.path.isdir(data_dir):
        for filename in os.listdir(data_dir):
            if filename.endswith('.json'):
                counts[filename.split('_')[0]] = counts.get(filename.split('_')[0], 0) + 1
    return counts


def run_session(data_type, labels, helptext, takes):
    counts = existing_takes(data_type)
    if counts:
        print("\nTakes already recorded:")
        for label in labels:
            print(f"  {label:15s} {counts.get(label, 0)}")

    for take in range(takes):
        variation = VARIATIONS[min(take, len(VARIATIONS) - 1)]
        print("\n" + "=" * 58)
        print(f"ROUND {take + 1} of {takes}  -  {variation}")
        print("=" * 58)
        print("Change your position BEFORE this round, and keep it for the")
        print("whole round. Each round becomes a separate test group.")

        for label in labels:
            print(f"\n  {label}: {helptext[label]}")
            choice = input(f"Press Enter to record '{label}' (or 's' to skip): ").strip().lower()
            if choice == 's':
                continue
            collect_data(label, duration=5, data_type=data_type, note=variation)


if __name__ == "__main__":
    print("=" * 58)
    print("SOCIAL CUE DATA COLLECTION")
    print("=" * 58)
    print("\nRecord each pose several times in DIFFERENT positions.")
    print("Multiple takes per class are what make an honest accuracy")
    print("score possible - with one take, the model can only be tested")
    print("on near-copies of what it already memorized.")

    choice = input("\nCollect [G]estures or [E]motions? ").strip().upper()

    raw_takes = input("How many rounds? (3 recommended, Enter for 3): ").strip()
    takes = int(raw_takes) if raw_takes.isdigit() and int(raw_takes) > 0 else 3

    if choice == 'G':
        run_session("gesture", GESTURES, GESTURE_HELP, takes)
    elif choice == 'E':
        run_session("emotion", EMOTIONS, EMOTION_HELP, takes)
    else:
        print("Invalid choice. Run again and type G or E.")
        sys.exit(1)

    print("\n=== Done! Now run:  python training/train.py  ===")
