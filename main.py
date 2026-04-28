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
import os
import sys

from detector import PostureDetector
from alert_system import AlertSystem
from session_logger import init_db, log_session, log_issue_event
from config import (
    MODEL_PATH, ENCODER_PATH,
    ALERT_MESSAGES, SUGGESTIONS,
    POSTURE_CLASSES,
)


# ── Argument parser ────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="AI Posture Detection")
    p.add_argument("--no-audio",    action="store_true", help="Disable TTS alerts")
    p.add_argument("--debug",       action="store_true", help="Show angle values")
    p.add_argument("--threshold",   type=int, default=5,  help="Alert threshold seconds")
    p.add_argument("--camera",      type=int, default=0,  help="Camera device index")
    return p.parse_args()


# ── Model loading ──────────────────────────────────────────────────────────

def load_model():
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model not found: {MODEL_PATH}")
        print("        Run  python train_model.py  first.")
        sys.exit(1)
    with open(MODEL_PATH,   "rb") as f: model = pickle.load(f)
    with open(ENCODER_PATH, "rb") as f: le    = pickle.load(f)
    return model, le


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

def run(args):
    print("=" * 52)
    print("  AI Posture Detection — starting up")
    print("=" * 52)

    model, le   = load_model()
    detector    = PostureDetector()
    alerts      = AlertSystem(threshold=args.threshold)
    db          = init_db()
    cap         = cv2.VideoCapture(args.camera)

    if not cap.isOpened():
        print("[ERROR] Cannot open camera.")
        sys.exit(1)

    # Session tracking
    session_start = time.time()
    good_frames   = 0
    total_frames  = 0
    issue_counts  = {}
    current_label = "unknown"
    session_id    = None

    print("  Press  Q  to quit and save session.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Frame not received — retrying...")
            time.sleep(0.05)
            continue

        result = detector.process(frame)
        frame  = result["frame_annotated"]
        total_frames += 1

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
        draw_status_bar(frame, current_label, confidence, active_issues)
        draw_suggestion_panel(frame, primary_issue)
        draw_posture_score(frame, good_frames, total_frames)
        if args.debug:
            draw_debug_info(frame, result)

        cv2.imshow("AI Posture Detection  [Q to quit]", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
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
