"""
main.py — AI Posture Detection Software
Real-time posture + eye detection with alerts and session logging.

Usage
-----
    python main.py            # run with defaults
    python main.py --no-audio # disable voice alerts
    python main.py --debug    # show angle values on screen
"""

import cv2
import pickle
import numpy as np
import time
import argparse
import sys
import os
import warnings
from contextlib import contextmanager

from alert_system import AlertSystem
from session_logger import init_db, log_session, log_issue_event
from config import (
    MODEL_PATH, ENCODER_PATH,
    ALERT_MESSAGES, SUGGESTIONS,
    POSTURE_CLASSES,
    SPINE_ANGLE_THRESHOLD, NECK_TILT_THRESHOLD,
    EAR_THRESHOLD, EYE_DIST_MIN, EYE_DIST_MAX,
)


# ── Argument parser ────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="AI Posture Detection")
    p.add_argument("--no-audio",    action="store_true", help="Disable TTS alerts")
    p.add_argument("--debug",       action="store_true", help="Show angle values")
    p.add_argument("--threshold",   type=int, default=5,  help="Alert threshold seconds")
    p.add_argument("--camera",      default="auto",       help="Camera index, or 'auto' to scan")
    p.add_argument("--max-camera",  type=int, default=5,  help="Highest camera index to scan in auto mode")
    return p.parse_args()


# ── Model loading ──────────────────────────────────────────────────────────

def load_model():
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model not found: {MODEL_PATH}")
        print("        Run  python train_model.py  first.")
        sys.exit(1)

    try:
        from sklearn.exceptions import InconsistentVersionWarning
    except Exception:
        InconsistentVersionWarning = None

    version_warning_seen = False
    with warnings.catch_warnings(record=True) as caught:
        if InconsistentVersionWarning is not None:
            warnings.simplefilter("always", InconsistentVersionWarning)

        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        with open(ENCODER_PATH, "rb") as f:
            le = pickle.load(f)

    for warning in caught:
        is_version_warning = (
            InconsistentVersionWarning is not None
            and issubclass(warning.category, InconsistentVersionWarning)
        )
        if is_version_warning:
            version_warning_seen = True
        else:
            warnings.warn(warning.message, warning.category, stacklevel=2)

    if version_warning_seen:
        print("[WARN] Saved model was trained with another scikit-learn version.")
        print("       If predictions look wrong, refresh it with: python train_model.py")

    return model, le


def _camera_indexes(camera_arg: str, max_camera: int):
    if str(camera_arg).lower() == "auto":
        return range(max(0, max_camera) + 1)

    try:
        camera_index = int(camera_arg)
    except ValueError:
        print(f"[ERROR] Invalid camera value: {camera_arg}")
        print("        Use --camera auto or a number, for example --camera 1")
        sys.exit(2)

    if camera_index < 0:
        print("[ERROR] Camera index must be 0 or greater.")
        sys.exit(2)

    return [camera_index]


def _can_read_frame(cap) -> bool:
    for _ in range(10):
        ok, _ = cap.read()
        if ok:
            return True
        time.sleep(0.05)
    return False


@contextmanager
def _suppress_native_stderr():
    """Temporarily hide noisy native-library stderr output."""
    stderr_copy = None
    devnull = None
    try:
        stderr_copy = os.dup(2)
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 2)
        yield
    finally:
        if stderr_copy is not None:
            os.dup2(stderr_copy, 2)
            os.close(stderr_copy)
        if devnull is not None:
            os.close(devnull)


def open_camera(camera_arg: str, max_camera: int = 5):
    """Open a webcam, trying the most reliable OpenCV backends for this OS."""
    backend_order = []
    if sys.platform.startswith("win"):
        backend_order.extend([
            ("DirectShow", cv2.CAP_DSHOW),
            ("Media Foundation", cv2.CAP_MSMF),
        ])
    backend_order.append(("OpenCV default", cv2.CAP_ANY))

    previous_log_level = cv2.getLogLevel() if hasattr(cv2, "getLogLevel") else None
    if previous_log_level is not None and hasattr(cv2, "setLogLevel"):
        cv2.setLogLevel(0)

    try:
        for camera_index in _camera_indexes(camera_arg, max_camera):
            for backend_name, backend in backend_order:
                cap = None
                with _suppress_native_stderr():
                    cap = cv2.VideoCapture(camera_index, backend)
                    can_read = cap.isOpened() and _can_read_frame(cap)

                if can_read:
                    print(f"  Camera {camera_index} opened with {backend_name}.")
                    return cap, camera_index
                if cap is not None:
                    cap.release()
    finally:
        if previous_log_level is not None and hasattr(cv2, "setLogLevel"):
            cv2.setLogLevel(previous_log_level)

    return None, None


# ── UI overlay helpers ─────────────────────────────────────────────────────

def draw_status_bar(frame, label, confidence, issues):
    """Top status bar showing current posture label."""
    h, w = frame.shape[:2]
    bar_h = 50
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    color = (60, 210, 60) if label == "upright" else (50, 90, 240)
    label_text = label.replace("_", " ").upper()
    cv2.putText(frame, label_text,
                (12, 33), cv2.FONT_HERSHEY_DUPLEX, 0.9, color, 2)
    cv2.putText(frame, f"{confidence:.0%}",
                (230, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 180), 1)

    if issues:
        issue_str = " | ".join(i.replace("_", " ") for i in issues)
        cv2.putText(frame, issue_str,
                    (w - 10, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 160, 255), 1,
                    cv2.LINE_AA)
        # right-align
        (tw, _), _ = cv2.getTextSize(issue_str, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.putText(frame, issue_str,
                    (w - tw - 10, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 160, 255), 1)


def draw_suggestion_panel(frame, active_issue):
    """Right-side semi-transparent panel with correction tip."""
    if not active_issue:
        return
    tips = SUGGESTIONS.get(active_issue, [])
    if not tips:
        return

    h, w  = frame.shape[:2]
    px    = w - 300
    overlay = frame.copy()
    cv2.rectangle(overlay, (px, 50), (w, h), (12, 12, 12), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    cv2.putText(frame, "How to fix:", (px + 10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 220, 80), 1)

    label = active_issue.replace("_", " ").title()
    cv2.putText(frame, label, (px + 10, 104),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (100, 200, 255), 1)

    y = 130
    for tip in tips[:3]:
        words, line = tip.split(), ""
        for word in words:
            if len(line + word) > 33:
                cv2.putText(frame, line, (px + 10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.36, (200, 200, 200), 1)
                y += 17
                line = word + " "
            else:
                line += word + " "
        if line:
            cv2.putText(frame, line.strip(), (px + 10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, (200, 200, 200), 1)
        y += 22


def draw_debug_info(frame, result):
    """Bottom-left debug overlay with raw metric values."""
    lines = [
        f"Spine : {result['spine_angle']:.1f}°",
        f"Tilt  : {result['neck_tilt']:.1f}°",
        f"EAR   : {result['ear']:.3f}",
        f"EyeD  : {result['eye_dist']:.0f}px",
    ]
    h = frame.shape[0]
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (10, h - 90 + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)


def draw_posture_score(frame, good_frames, total_frames):
    """Bottom-right posture quality score."""
    if total_frames == 0:
        return
    pct  = good_frames / total_frames * 100
    h, w = frame.shape[:2]
    color = (60, 200, 60) if pct >= 70 else (50, 140, 255) if pct >= 50 else (60, 60, 240)
    cv2.putText(frame, f"Score: {pct:.0f}%",
                (w - 140, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


# ── Main detection loop ────────────────────────────────────────────────────

def _clamp(value, low=0.0, high=1.0):
    return max(low, min(high, value))


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def _score_color(score: float):
    if score >= 0.72:
        return (70, 210, 95)
    if score >= 0.42:
        return (55, 165, 245)
    return (60, 80, 245)


def _posture_color(label: str, issues: list):
    if label == "upright" and not issues:
        return (70, 215, 105)
    if label == "unknown":
        return (180, 180, 180)
    if any(issue in ("slouch", "lean_forward") for issue in issues):
        return (70, 90, 245)
    return (70, 165, 245)


def _draw_panel(frame, x1, y1, x2, y2, color=(12, 14, 18), alpha=0.68,
                border=(56, 64, 74)):
    h, w = frame.shape[:2]
    x1 = int(_clamp(x1, 0, w - 1))
    y1 = int(_clamp(y1, 0, h - 1))
    x2 = int(_clamp(x2, 0, w - 1))
    y2 = int(_clamp(y2, 0, h - 1))
    if x2 <= x1 or y2 <= y1:
        return

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), border, 1)


def _draw_progress_bar(frame, x, y, width, pct, color, height=8):
    pct = _clamp(pct)
    cv2.rectangle(frame, (x, y), (x + width, y + height), (55, 58, 64), -1)
    filled = int(width * pct)
    if filled > 0:
        cv2.rectangle(frame, (x, y), (x + filled, y + height), color, -1)
    cv2.rectangle(frame, (x, y), (x + width, y + height), (95, 100, 108), 1)


def _wrap_lines(text: str, max_chars: int, max_lines: int = 3):
    words = text.split()
    lines = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) > max_chars and current:
            lines.append(current)
            current = word
            if len(lines) == max_lines:
                return lines
        else:
            current = candidate
    if current and len(lines) < max_lines:
        lines.append(current)
    return lines


def _draw_metric(frame, x, y, label, value, score, width):
    color = _score_color(score)
    cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.38, (160, 166, 174), 1, cv2.LINE_AA)
    (tw, _), _ = cv2.getTextSize(value, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
    cv2.putText(frame, value, (x + width - tw, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.42, (224, 228, 232), 1, cv2.LINE_AA)
    _draw_progress_bar(frame, x, y + 8, width, score, color, height=7)


def draw_status_bar(frame, label, confidence, issues, stats):
    """Top HUD with posture, confidence, elapsed time, FPS, and camera."""
    h, w = frame.shape[:2]
    _draw_panel(frame, 0, 0, w, 64, color=(10, 12, 16), alpha=0.74,
                border=(46, 54, 64))

    color = _posture_color(label, issues)
    label_text = "NO PERSON" if label == "unknown" else label.replace("_", " ").upper()
    cv2.putText(frame, label_text, (14, 38), cv2.FONT_HERSHEY_DUPLEX,
                0.86, color, 2, cv2.LINE_AA)

    conf_text = f"{confidence:.0%}" if confidence else "--"
    cv2.putText(frame, f"conf {conf_text}", (14, 57), cv2.FONT_HERSHEY_SIMPLEX,
                0.38, (164, 170, 178), 1, cv2.LINE_AA)

    issue_text = "clear" if not issues else ", ".join(i.replace("_", " ") for i in issues[:2])
    if len(issues) > 2:
        issue_text += f" +{len(issues) - 2}"
    cv2.putText(frame, issue_text[:42], (130, 57), cv2.FONT_HERSHEY_SIMPLEX,
                0.38, (188, 194, 202), 1, cv2.LINE_AA)

    items = [
        ("TIME", _format_duration(stats["elapsed"])),
        ("FPS", f"{stats['fps']:.0f}"),
        ("CAM", f"#{stats['camera_index']}"),
    ]
    start_x = max(300, w - 305)
    for i, (name, value) in enumerate(items):
        x = start_x + i * 100
        cv2.putText(frame, name, (x, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.34, (135, 142, 150), 1, cv2.LINE_AA)
        cv2.putText(frame, value, (x, 45), cv2.FONT_HERSHEY_DUPLEX,
                    0.56, (232, 236, 240), 1, cv2.LINE_AA)


def draw_session_dashboard(frame, result, label, active_issues, stats, alerts):
    """Right-side live dashboard with score, metrics, alert countdown, and tips."""
    h, w = frame.shape[:2]
    panel_w = min(330, max(265, w // 3))
    x = w - panel_w
    y = 72
    bottom = h - 42
    _draw_panel(frame, x, y, w - 8, bottom, color=(11, 13, 18), alpha=0.72,
                border=(56, 64, 74))

    pad = 14
    inner_x = x + pad
    inner_w = panel_w - pad * 2 - 8
    cursor = y + 28

    cv2.putText(frame, "Live Session", (inner_x, cursor),
                cv2.FONT_HERSHEY_DUPLEX, 0.55, (232, 236, 240), 1, cv2.LINE_AA)
    cursor += 26

    good_pct = stats["good_pct"]
    score_color = _score_color(good_pct / 100)
    cv2.putText(frame, f"{good_pct:.0f}% good posture", (inner_x, cursor),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, score_color, 1, cv2.LINE_AA)
    _draw_progress_bar(frame, inner_x, cursor + 10, inner_w, good_pct / 100,
                       score_color, height=9)
    cursor += 38

    stat_lines = [
        ("Elapsed", _format_duration(stats["elapsed"])),
        ("Frames", str(stats["total_frames"])),
        ("Alerts", str(sum(stats["issue_counts"].values()))),
    ]
    for i, (name, value) in enumerate(stat_lines):
        col_x = inner_x + i * max(75, inner_w // 3)
        cv2.putText(frame, name, (col_x, cursor), cv2.FONT_HERSHEY_SIMPLEX,
                    0.33, (136, 144, 152), 1, cv2.LINE_AA)
        cv2.putText(frame, value, (col_x, cursor + 20), cv2.FONT_HERSHEY_DUPLEX,
                    0.47, (224, 228, 232), 1, cv2.LINE_AA)
    cursor += 54

    spine_score = _clamp((result["spine_angle"] - SPINE_ANGLE_THRESHOLD) /
                         max(1, 180 - SPINE_ANGLE_THRESHOLD))
    neck_score = _clamp(1 - abs(result["neck_tilt"]) /
                        max(1, NECK_TILT_THRESHOLD * 2))
    ear_score = _clamp((result["ear"] - EAR_THRESHOLD) / 0.16)
    eye_mid = (EYE_DIST_MIN + EYE_DIST_MAX) / 2
    eye_span = max(1, (EYE_DIST_MAX - EYE_DIST_MIN) / 2)
    eye_score = _clamp(1 - abs(result["eye_dist"] - eye_mid) / eye_span)

    _draw_metric(frame, inner_x, cursor, "Spine", f"{result['spine_angle']:.0f} deg",
                 spine_score, inner_w)
    cursor += 33
    _draw_metric(frame, inner_x, cursor, "Neck tilt", f"{result['neck_tilt']:.0f} deg",
                 neck_score, inner_w)
    cursor += 33
    _draw_metric(frame, inner_x, cursor, "Eyes open", f"{result['ear']:.2f}",
                 ear_score, inner_w)
    cursor += 33
    _draw_metric(frame, inner_x, cursor, "Distance", f"{result['eye_dist']:.0f}px",
                 eye_score, inner_w)
    cursor += 39

    cv2.putText(frame, "Active Issues", (inner_x, cursor),
                cv2.FONT_HERSHEY_DUPLEX, 0.47, (232, 236, 240), 1, cv2.LINE_AA)
    cursor += 24

    if not active_issues:
        cv2.putText(frame, "No active posture issues", (inner_x, cursor),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 210, 150), 1, cv2.LINE_AA)
        cursor += 24
    else:
        for issue in active_issues[:3]:
            text = issue.replace("_", " ")
            cv2.putText(frame, f"- {text}", (inner_x, cursor),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (92, 178, 245), 1, cv2.LINE_AA)
            cursor += 19
        if len(active_issues) > 3:
            cv2.putText(frame, f"+ {len(active_issues) - 3} more", (inner_x, cursor),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, (150, 156, 164), 1, cv2.LINE_AA)
            cursor += 18

    primary_issue = active_issues[0] if active_issues else None
    if primary_issue:
        seconds_bad = alerts.seconds_bad(primary_issue)
        remaining = max(0, stats["alert_threshold"] - seconds_bad)
        label_text = "Alert ready" if remaining <= 0 else f"Alert in {remaining:.0f}s"
        cursor += 4
        cv2.putText(frame, label_text, (inner_x, cursor),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (78, 174, 245), 1, cv2.LINE_AA)
        _draw_progress_bar(frame, inner_x, cursor + 9, inner_w,
                           seconds_bad / max(1, stats["alert_threshold"]),
                           (60, 118, 245), height=7)
        cursor += 34

    tips = SUGGESTIONS.get(primary_issue, []) if primary_issue else []
    if tips and cursor < bottom - 78:
        cv2.putText(frame, "Quick Fix", (inner_x, cursor),
                    cv2.FONT_HERSHEY_DUPLEX, 0.45, (232, 236, 240), 1, cv2.LINE_AA)
        cursor += 20
        max_chars = max(24, inner_w // 8)
        for tip in tips[:2]:
            for line in _wrap_lines(tip, max_chars=max_chars, max_lines=2):
                if cursor > bottom - 18:
                    break
                cv2.putText(frame, line, (inner_x, cursor),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.34, (188, 194, 202),
                            1, cv2.LINE_AA)
                cursor += 16
            cursor += 6


def draw_debug_info(frame, result):
    """Bottom-left debug overlay with raw metric values."""
    h = frame.shape[0]
    y = max(74, h - 138)
    _draw_panel(frame, 8, y, 190, h - 48, color=(10, 12, 16), alpha=0.64,
                border=(52, 60, 68))
    lines = [
        f"Spine : {result['spine_angle']:.1f} deg",
        f"Tilt  : {result['neck_tilt']:.1f} deg",
        f"EAR   : {result['ear']:.3f}",
        f"EyeD  : {result['eye_dist']:.0f}px",
    ]
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (18, y + 24 + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.43, (178, 184, 192), 1,
                    cv2.LINE_AA)


def draw_controls_bar(frame, debug_on, paused):
    h, w = frame.shape[:2]
    _draw_panel(frame, 0, h - 34, w, h, color=(10, 12, 16), alpha=0.74,
                border=(46, 54, 64))
    controls = "Q quit | P pause | D debug | S snapshot | R reset | H help"
    cv2.putText(frame, controls, (12, h - 12), cv2.FONT_HERSHEY_SIMPLEX,
                0.42, (194, 200, 208), 1, cv2.LINE_AA)
    state = []
    if paused:
        state.append("PAUSED")
    if debug_on:
        state.append("DEBUG")
    if state:
        text = " / ".join(state)
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        cv2.putText(frame, text, (w - tw - 12, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (80, 190, 245), 1,
                    cv2.LINE_AA)


def draw_help_overlay(frame):
    h, w = frame.shape[:2]
    box_w = min(520, w - 50)
    box_h = min(265, h - 90)
    x = (w - box_w) // 2
    y = (h - box_h) // 2
    _draw_panel(frame, x, y, x + box_w, y + box_h, color=(12, 14, 18),
                alpha=0.86, border=(72, 84, 96))

    lines = [
        ("PostureGuard Controls", 0.58, (236, 240, 244)),
        ("P  pause or resume live detection", 0.43, (204, 210, 218)),
        ("D  toggle raw metrics overlay", 0.43, (204, 210, 218)),
        ("S  save a snapshot to data/screenshots", 0.43, (204, 210, 218)),
        ("R  reset timer, score, and alert counters", 0.43, (204, 210, 218)),
        ("Q  quit and save the session", 0.43, (204, 210, 218)),
    ]
    cursor = y + 34
    for text, scale, color in lines:
        cv2.putText(frame, text, (x + 22, cursor), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, 1, cv2.LINE_AA)
        cursor += 34


def draw_pause_overlay(frame):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (5, 7, 10), -1)
    cv2.addWeighted(overlay, 0.48, frame, 0.52, 0, frame)
    (tw, _), _ = cv2.getTextSize("PAUSED", cv2.FONT_HERSHEY_DUPLEX, 1.1, 2)
    cv2.putText(frame, "PAUSED", ((w - tw) // 2, h // 2 - 8),
                cv2.FONT_HERSHEY_DUPLEX, 1.1, (80, 190, 245), 2, cv2.LINE_AA)
    msg = "Press P to resume"
    (mw, _), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.putText(frame, msg, ((w - mw) // 2, h // 2 + 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (222, 226, 232), 1, cv2.LINE_AA)


def draw_toast(frame, message, until_time):
    if not message or time.time() >= until_time:
        return
    h, w = frame.shape[:2]
    (tw, th), _ = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)
    x = max(10, (w - tw) // 2 - 16)
    y = 78
    _draw_panel(frame, x, y, x + tw + 32, y + th + 24, color=(14, 16, 20),
                alpha=0.82, border=(70, 82, 96))
    cv2.putText(frame, message, (x + 16, y + th + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (230, 234, 238), 1,
                cv2.LINE_AA)


def save_snapshot(frame, label):
    folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "screenshots")
    os.makedirs(folder, exist_ok=True)
    safe_label = label.replace(" ", "_").replace(os.sep, "_")
    filename = f"posture_{time.strftime('%Y%m%d_%H%M%S')}_{safe_label}.jpg"
    path = os.path.join(folder, filename)
    ok = cv2.imwrite(path, frame)
    return path if ok else None


def run(args):
    print("=" * 52)
    print("  AI Posture Detection — starting up")
    print("=" * 52)

    model, le = load_model()
    cap, camera_index = open_camera(args.camera, args.max_camera)

    if cap is None:
        camera_text = f"camera {args.camera}" if args.camera != "auto" else f"any camera from 0 to {args.max_camera}"
        print(f"[ERROR] Cannot open {camera_text}.")
        print("        Close other camera apps, check Windows camera privacy settings,")
        print("        or plug in a webcam and run  python main.py --camera auto")
        sys.exit(1)

    from detector import PostureDetector

    detector    = PostureDetector()
    alerts      = AlertSystem(threshold=args.threshold, enable_audio=not args.no_audio)
    db          = init_db()

    # Session tracking
    session_start = time.time()
    good_frames   = 0
    total_frames  = 0
    issue_counts  = {}
    current_label = "unknown"
    session_id    = None

    # UI state
    debug_on      = args.debug
    help_visible  = False
    paused        = False
    last_ui_frame = None
    last_tick     = time.time()
    fps           = 0.0
    toast_text    = ""
    toast_until   = 0.0
    window_name   = "AI Posture Detection"

    def set_toast(message: str, seconds: float = 2.2):
        nonlocal toast_text, toast_until
        toast_text = message
        toast_until = time.time() + seconds

    def handle_key(key, frame_for_snapshot):
        nonlocal debug_on, help_visible, paused, session_start
        nonlocal good_frames, total_frames, current_label, last_tick

        if key in (-1, 255):
            return False

        if key in (ord("q"), ord("Q"), 27):
            return True
        if key in (ord("p"), ord("P")):
            paused = not paused
            last_tick = time.time()
            set_toast("Paused" if paused else "Resumed")
        elif key in (ord("d"), ord("D")):
            debug_on = not debug_on
            set_toast("Debug overlay on" if debug_on else "Debug overlay off")
        elif key in (ord("h"), ord("H")):
            help_visible = not help_visible
        elif key in (ord("r"), ord("R")):
            session_start = time.time()
            good_frames = 0
            total_frames = 0
            issue_counts.clear()
            alerts.reset()
            current_label = "unknown"
            set_toast("Session stats reset")
        elif key in (ord("s"), ord("S")) and frame_for_snapshot is not None:
            path = save_snapshot(frame_for_snapshot, current_label)
            if path:
                print(f"[SNAPSHOT] Saved: {path}")
                set_toast(f"Snapshot saved: {os.path.basename(path)}")
            else:
                set_toast("Snapshot failed")
        return False

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    print("  Controls: Q quit | P pause | D debug | S snapshot | R reset | H help\n")

    while True:
        if paused and last_ui_frame is not None:
            paused_frame = last_ui_frame.copy()
            draw_pause_overlay(paused_frame)
            draw_controls_bar(paused_frame, debug_on, paused)
            if help_visible:
                draw_help_overlay(paused_frame)
            draw_toast(paused_frame, toast_text, toast_until)
            cv2.imshow(window_name, paused_frame)
            if handle_key(cv2.waitKey(30) & 0xFF, paused_frame):
                break
            continue

        ret, frame = cap.read()
        if not ret:
            print("[WARN] Frame not received — retrying...")
            time.sleep(0.05)
            continue

        result = detector.process(frame)
        frame  = result["frame_annotated"]
        total_frames += 1

        now = time.time()
        frame_delta = max(0.001, now - last_tick)
        instant_fps = 1.0 / frame_delta
        fps = instant_fps if fps <= 0 else (fps * 0.88 + instant_fps * 0.12)
        last_tick = now

        # ── ML classification ──────────────────────────────────────────────
        confidence = 0.0
        if result["keypoint_row"]:
            pred       = model.predict([result["keypoint_row"]])[0]
            confidence = model.predict_proba([result["keypoint_row"]]).max()
            current_label = le.inverse_transform([pred])[0]
            if current_label == "upright":
                good_frames += 1

        active_issues = result["issues"].copy()

        # Add ML-detected issue
        if current_label not in ("upright", "unknown"):
            active_issues.insert(0, current_label)

        # Deduplicate
        seen = set()
        active_issues = [x for x in active_issues if not (x in seen or seen.add(x))]

        # ── Alert updates ──────────────────────────────────────────────────
        for issue in active_issues:
            msg = ALERT_MESSAGES.get(issue, "Please correct your posture.")
            sev = "critical" if issue in ("slouch", "lean_forward") else "info"
            fired = alerts.update(issue, is_bad=True, msg=msg, severity=sev)
            if fired:
                issue_counts[issue] = issue_counts.get(issue, 0) + 1
        # Reset resolved issues
        all_keys = list(POSTURE_CLASSES) + ["eye_closing", "too_close", "too_far"]
        for key in all_keys:
            if key not in active_issues:
                alerts.update(key, is_bad=False, msg="")

        # ── Overlays ───────────────────────────────────────────────────────
        primary_issue = active_issues[0] if active_issues else None
        good_pct = (good_frames / total_frames * 100) if total_frames > 0 else 0.0
        stats = {
            "elapsed": time.time() - session_start,
            "fps": fps,
            "camera_index": camera_index,
            "good_pct": good_pct,
            "total_frames": total_frames,
            "issue_counts": issue_counts,
            "alert_threshold": args.threshold,
        }
        draw_status_bar(frame, current_label, confidence, active_issues, stats)
        draw_session_dashboard(frame, result, current_label, active_issues, stats, alerts)
        if debug_on:
            draw_debug_info(frame, result)
        draw_controls_bar(frame, debug_on, paused)
        if help_visible:
            draw_help_overlay(frame)
        draw_toast(frame, toast_text, toast_until)

        last_ui_frame = frame.copy()
        cv2.imshow(window_name, frame)
        if handle_key(cv2.waitKey(1) & 0xFF, frame):
            break

    # ── Session end ────────────────────────────────────────────────────────
    duration = int(time.time() - session_start)
    good_pct = (good_frames / total_frames * 100) if total_frames > 0 else 0
    main_issue = max(issue_counts, key=issue_counts.get) if issue_counts else None

    session_id = log_session(db, duration, good_pct, total_frames, main_issue)
    print(f"\n[SESSION] Duration     : {duration}s")
    print(f"[SESSION] Good posture : {good_pct:.0f}%")
    print(f"[SESSION] Main issue   : {main_issue or 'none'}")
    print(f"[SESSION] Saved to DB  : session #{session_id}")

    cap.release()
    cv2.destroyAllWindows()
    detector.release()


if __name__ == "__main__":
    args = parse_args()
    run(args)
