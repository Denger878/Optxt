# speech.py
"""
Text-to-speech on a background thread.

Speaking is blocking: pyttsx3's runAndWait() and `say` both hold the caller
until the sentence finishes. Calling either from the capture loop froze the
video for the length of every announcement, so speech runs on its own worker
thread and the loop just drops a line on the queue.
"""
import platform
import queue
import subprocess
import threading

import pyttsx3

_IS_MAC = platform.system() == "Darwin"

try:
    _engine = pyttsx3.init()
    _USE_PYTTSX3 = True
except Exception:
    _engine = None
    _USE_PYTTSX3 = False
    print("⚠️  pyttsx3 failed, using system TTS")

_queue = queue.Queue(maxsize=4)
_muted = threading.Event()


def _speak_blocking(text):
    if _USE_PYTTSX3:
        try:
            _engine.say(text)
            _engine.runAndWait()
            return
        except Exception as e:
            print(f"TTS error: {e}")
    if _IS_MAC:
        subprocess.run(["say", text])


def _worker():
    while True:
        text = _queue.get()
        if text is None:
            break
        _speak_blocking(text)
        _queue.task_done()


_thread = threading.Thread(target=_worker, daemon=True)
_thread.start()


def say_interaction(text):
    """Queue a line to be spoken. Returns immediately."""
    print(f"🔊 Optxt says: {text}")
    if _muted.is_set():
        return
    try:
        _queue.put_nowait(text)
    except queue.Full:
        # Speech is falling behind the detections; skip rather than build a
        # backlog of announcements about things that already stopped happening.
        pass


def toggle_mute():
    """Flip mute. Returns True if now muted."""
    if _muted.is_set():
        _muted.clear()
        return False
    _muted.set()
    return True


def is_muted():
    return _muted.is_set()


if __name__ == "__main__":
    say_interaction("Voice system is online.")
    _queue.join()
