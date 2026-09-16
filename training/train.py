# training/train.py
import json
import os
import sys
from collections import Counter

import numpy as np
import joblib
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              HistGradientBoostingClassifier)
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.base import clone

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features import extract, FEATURE_NAMES


# Labels kept on disk but left out of training. Shrug was retired: held at a
# realistic intensity it is nearly identical to neutral in a single frame, and
# it dominated the error rate while every hand gesture sat at 97-99%.
EXCLUDED_LABELS = {"shrug"}


def load_data(data_type="gesture"):
    """
    Load every recording and turn it into features.

    Returns X, y, and groups. `groups` is the recording each sample came from —
    frames within one 5-second hold are near-identical, so they must never be
    split across train and test or the score is measuring memorization.
    """
    data_dir = os.path.join(os.path.dirname(__file__), f"data/{data_type}")

    X, y, groups = [], [], []

    print(f"\n📂 Loading {data_type} data from: {data_dir}")

    files = sorted(f for f in os.listdir(data_dir) if f.endswith('.json'))
    print(f"Found {len(files)} recordings")

    skipped = 0
    for filename in files:
        # "thumbsup_1768124724.json" -> label "thumbsup", recording id = filename
        label = filename.split('_')[0]
        if label in EXCLUDED_LABELS:
            continue

        with open(os.path.join(data_dir, filename), 'r') as f:
            frames = json.load(f)

        kept = 0
        for frame in frames:
            feats = extract(frame['landmarks'], data_type)
            if feats is None:
                skipped += 1
                continue
            X.append(feats)
            y.append(label)
            groups.append(filename)
            kept += 1

        print(f"  {label:15s} {kept:4d} usable frames  ({filename})")

    if skipped:
        print(f"  ({skipped} frames skipped - required landmarks not detected)")

    return np.array(X), np.array(y), np.array(groups)


def _report_grouping(y, groups):
    """
    Check whether an honest held-out test is even possible.

    It needs at least 2 separate recordings per class. With one recording per
    class, any split leaves near-duplicate frames on both sides and the accuracy
    is meaningless no matter how it's computed.
    """
    per_class = {}
    for label, group in zip(y, groups):
        per_class.setdefault(label, set()).add(group)

    print("\n📈 Recordings per class:")
    for label in sorted(per_class):
        n = len(per_class[label])
        flag = "" if n >= 2 else "   ⚠️  only 1 - cannot validate honestly"
        print(f"  {label:15s} {n} recording(s){flag}")

    return min(len(v) for v in per_class.values())


def train_model(data_type="gesture"):
    """Train, validate honestly, and save the classifier."""
    print("=" * 62)
    print(f"🧠 TRAINING {data_type.upper()} MODEL")
    print("=" * 62)

    X, y, groups = load_data(data_type)
    if len(X) == 0:
        print("❌ No usable samples. Collect data first.")
        return None

    print(f"\n✅ {len(X)} samples, {X.shape[1]} features each")
    print(f"📊 Classes: {sorted(set(y))}")
    for label, count in sorted(Counter(y).items()):
        print(f"  {label:15s} {count:4d} samples")

    min_groups = _report_grouping(y, groups)

    candidates = {
        'RandomForest': RandomForestClassifier(
            n_estimators=300, max_depth=12, random_state=42,
            n_jobs=-1, class_weight='balanced'),
        'ExtraTrees': ExtraTreesClassifier(
            n_estimators=300, max_depth=12, random_state=42,
            n_jobs=-1, class_weight='balanced'),
        'HistGradientBoosting': HistGradientBoostingClassifier(random_state=42),
    }

    if min_groups < 2:
        print("\n" + "!" * 62)
        print("⚠️  CANNOT REPORT A TRUSTWORTHY ACCURACY")
        print("!" * 62)
        print("Every class has only one recording. Splitting it would put")
        print("near-identical frames in both train and test, so the score")
        print("would just measure memorization (this is why the old code")
        print("reported 100%).")
        print("\nFix: record each class 3+ times - different sessions,")
        print("distances from the camera, lighting, and ideally people.")
        print("Then re-run this. Training on everything for now.\n")
        test_score = None
        best_name = 'RandomForest'
    else:
        # Compare models by grouped cross-validation and keep the best. Which
        # one wins is not obvious in advance - boosting tends to beat forests
        # on these hand-built geometric features, but not always.
        cv_splits = min(5, min_groups)
        cv = StratifiedGroupKFold(n_splits=cv_splits, shuffle=True, random_state=42)

        print(f"\n🔬 Comparing models ({cv_splits}-fold grouped CV):")
        cv_results = {}
        for name, candidate in candidates.items():
            scores = cross_val_score(candidate, X, y, groups=groups, cv=cv)
            cv_results[name] = scores
            print(f"  {name:22s} {scores.mean():6.1%}  (+/- {scores.std():.1%})")

        best_name = max(cv_results, key=lambda k: cv_results[k].mean())
        best_scores = cv_results[best_name]
        print(f"  -> using {best_name}")

        # Held-out test: whole recordings go to one side or the other.
        # StratifiedGroupKFold rather than GroupShuffleSplit, because a plain
        # group split can hand the test set only some of the classes - which it
        # did, leaving two classes with zero test samples and a meaningless score.
        n_splits = min(4, min_groups)
        holdout = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
        train_idx, test_idx = next(holdout.split(X, y, groups))
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        print(f"\n🔀 Grouped split (no recording appears on both sides):")
        print(f"  Training: {len(X_train)} samples from {len(set(groups[train_idx]))} recordings")
        print(f"  Testing:  {len(X_test)} samples from {len(set(groups[test_idx]))} recordings")

        model = clone(candidates[best_name])
        model.fit(X_train, y_train)

        test_score = model.score(X_test, y_test)
        print(f"\n📊 Held-out accuracy: {test_score:.1%}   <- the honest number")
        print(f"🔁 Grouped {cv_splits}-fold CV: "
              f"{best_scores.mean():.1%} (+/- {best_scores.std():.1%})   <- quote this one")

        predictions = model.predict(X_test)
        print(f"\n📋 Classification report:")
        print(classification_report(y_test, predictions, zero_division=0))

        # Include predicted-but-absent labels, or the matrix loses a column.
        labels = sorted(set(y_test) | set(predictions))
        print("🔢 Confusion matrix (rows = actual, cols = predicted):")
        print(" " * 15 + " ".join(f"{l[:7]:>7s}" for l in labels))
        for label, row in zip(labels, confusion_matrix(y_test, predictions, labels=labels)):
            print(f"{label[:14]:14s} " + " ".join(f"{v:7d}" for v in row))

    # Final model trains on everything available.
    model = clone(candidates[best_name])
    model.fit(X, y)

    names = FEATURE_NAMES[data_type]
    if hasattr(model, 'feature_importances_'):
        print(f"\n🔍 Most useful features:")
        for i in np.argsort(model.feature_importances_)[::-1][:8]:
            print(f"  {names[i]:24s} {model.feature_importances_[i]:.3f}")

    model_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{data_type}_model.pkl")
    joblib.dump(model, model_path)

    print(f"\n✅ Saved to: {model_path}")
    print("=" * 62)

    return test_score


if __name__ == "__main__":
    print("\n🤖 MODEL TRAINING")
    print("=" * 62)

    choice = input("\nTrain [G]esture or [E]motion model? (or [B]oth): ").strip().upper()

    if choice == 'G':
        train_model("gesture")
    elif choice == 'E':
        train_model("emotion")
    elif choice == 'B':
        gesture_acc = train_model("gesture")
        emotion_acc = train_model("emotion")
        print("\n" + "=" * 62)
        print("🎉 TRAINING COMPLETE")
        print("=" * 62)
        for name, acc in [("Gesture", gesture_acc), ("Emotion", emotion_acc)]:
            print(f"{name} held-out accuracy: "
                  + (f"{acc:.1%}" if acc is not None else "not measurable (1 recording per class)"))
    else:
        print("❌ Invalid choice. Run again and type G, E, or B.")
