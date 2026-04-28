"""
alert_system.py — Timed multi-channel alert engine
Fires voice, OS notification, and visual alerts only after
an issue persists past the threshold. Resets when user corrects.
"""

import time
import threading
import pyttsx3
from plyer import notification
from config import ALERT_THRESHOLD_SECONDS, ALERT_COOLDOWN_SECONDS


class AlertSystem:
    def __init__(self, threshold: int = ALERT_THRESHOLD_SECONDS):
        self.threshold    = threshold
        self.bad_start    = {}   # key -> timestamp issue started
        self.alerted_at   = {}   # key -> timestamp last alert fired
        self._tts_lock    = threading.Lock()

        try:
            self._tts = pyttsx3.init()
            self._tts.setProperty("rate", 155)
            self._tts.setProperty("volume", 0.9)
            self._tts_ready = True
        except Exception:
            self._tts_ready = False

    def update(self, key: str, is_bad: bool, msg: str, severity: str = "warning") -> bool:
        """
        Call every frame for each tracked issue.

        Parameters
        ----------
        key      : unique identifier for the issue (e.g. 'slouch')
        is_bad   : True if the issue is currently active
        msg      : human-readable message to speak / show
        severity : 'info' | 'warning' | 'critical'

        Returns
        -------
        bool : True if an alert was fired this call
        """
        now = time.time()

        if is_bad:
            if key not in self.bad_start:
                self.bad_start[key] = now

            elapsed        = now - self.bad_start[key]
            last_alerted   = self.alerted_at.get(key, 0)
            cooldown_ok    = (now - last_alerted) >= ALERT_COOLDOWN_SECONDS

            if elapsed >= self.threshold and cooldown_ok:
                self._fire(key, msg, severity)
                self.alerted_at[key] = now
                return True
        else:
            self.bad_start.pop(key, None)

        return False

    def seconds_bad(self, key: str) -> float:
        """How long (in seconds) the issue has been active. 0 if not active."""
        if key not in self.bad_start:
            return 0.0
        return time.time() - self.bad_start[key]

    def reset(self, key: str = None):
        """Reset a specific issue or all issues."""
        if key:
            self.bad_start.pop(key, None)
            self.alerted_at.pop(key, None)
        else:
            self.bad_start.clear()
            self.alerted_at.clear()

    def _fire(self, key: str, msg: str, severity: str):
        # 1. OS toast notification
        try:
            notification.notify(
                title="Posture Alert",
                message=msg,
                app_name="AI Posture Detection",
                timeout=5,
            )
        except Exception:
            pass

        # 2. TTS voice alert (non-blocking thread)
        if self._tts_ready:
            threading.Thread(
                target=self._speak,
                args=(msg,),
                daemon=True,
            ).start()

        print(f"[ALERT][{severity.upper()}] {key}: {msg}")

    def _speak(self, msg: str):
        with self._tts_lock:
            try:
                self._tts.say(msg)
                self._tts.runAndWait()
            except Exception:
                pass
