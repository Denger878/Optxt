// tracker.js
//
// Port of the Python state_tracker.py. Single-frame predictions flicker, so a
// label has to hold steady for `stability` consecutive frames before it counts
// as the current state, and only a real change produces a new announcement.

const IGNORED = new Set(['no_data', 'no_model', 'no_face', 'unsure']);

const PHRASES = {
  thumbsup: 'giving a thumbs up',
  thumbsdown: 'giving a thumbs down',
  pointing: 'pointing at you',
  middlefinger: 'making a rude gesture',
  waving: 'waving at you',
  neutral: 'standing neutrally',
};

export function phrase(gesture) {
  return PHRASES[gesture] ?? gesture;
}

export function describe(gesture, emotion) {
  const g = gesture && !IGNORED.has(gesture);
  const e = emotion && !IGNORED.has(emotion);
  if (g && e) return `They are ${phrase(gesture)} and look ${emotion}`;
  if (g) return `They are ${phrase(gesture)}`;
  if (e) return `They look ${emotion}`;
  return '';
}

export class StateTracker {
  constructor(stability = 5) {
    this.stability = stability;
    this.gestureHistory = [];
    this.emotionHistory = [];
    this.currentGesture = null;
    this.currentEmotion = null;
  }

  #stable(history) {
    if (history.length < this.stability) return null;
    const first = history[0];
    return history.every((v) => v === first) ? first : null;
  }

  #push(history, value) {
    history.push(value);
    if (history.length > this.stability) history.shift();
  }

  update(gesture, emotion) {
    this.#push(this.gestureHistory, gesture);
    this.#push(this.emotionHistory, emotion);

    const stableGesture = this.#stable(this.gestureHistory);
    const stableEmotion = this.#stable(this.emotionHistory);

    let changed = false;
    if (stableGesture !== null && stableGesture !== this.currentGesture) {
      this.currentGesture = stableGesture;
      changed = true;
    }
    if (stableEmotion !== null && stableEmotion !== this.currentEmotion) {
      this.currentEmotion = stableEmotion;
      changed = true;
    }
    if (!changed) return { changed: false, message: '' };

    const message = describe(this.currentGesture, this.currentEmotion);
    return message ? { changed: true, message } : { changed: false, message: '' };
  }
}
