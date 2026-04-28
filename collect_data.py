"""
collect_data.py — Webcam posture image capture tool
Run this script to collect labeled training images.
For each class: get into posture → press SPACE → hold still.
"""

import cv2
import os
import time
from config import POSTURE_CLASSES, DATASET_DIR, IMAGES_PER_CLASS, CAPTURE_DELAY_MS


def capture_dataset():
    os.makedirs(DATASET_DIR, exist_ok=True)
    for cls in POSTURE_CLASSES:
        os.makedirs(os.path.join(DATASET_DIR, cls), exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam.")
        return

    print("=" * 50)
    print("  AI Posture Detection — Data Collection")
    print("=" * 50)
    print(f"  Classes : {POSTURE_CLASSES}")
    print(f"  Images  : {IMAGES_PER_CLASS} per class")
    print(f"  SPACE   : start capture  |  Q : quit")
    print("=" * 50)

    for cls in POSTURE_CLASSES:
        cls_dir   = os.path.join(DATASET_DIR, cls)
        existing  = len([f for f in os.listdir(cls_dir) if f.endswith(".jpg")])

        if existing >= IMAGES_PER_CLASS:
            print(f"\n[SKIP] {cls} — already has {existing} images.")
            continue

        print(f"\n>>> Next class: {cls.upper().replace('_', ' ')}")
        print(f"    Sit / stand in this posture, then press SPACE.")

        # ── Wait for spacebar ──────────────────────────────────────────────
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], 80), (20, 20, 20), -1)
            cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
            cv2.putText(frame, f"Posture: {cls.replace('_',' ').upper()}",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 100), 2)
            cv2.putText(frame, "Press SPACE to start capturing | Q to quit",
                        (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
            cv2.imshow("Data Collection", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                break
            if key == ord("q"):
                cap.release()
                cv2.destroyAllWindows()
                return

        # ── Countdown ─────────────────────────────────────────────────────
        for countdown in range(3, 0, -1):
            ret, frame = cap.read()
            cv2.putText(frame, str(countdown),
                        (frame.shape[1] // 2 - 30, frame.shape[0] // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 200, 80), 6)
            cv2.imshow("Data Collection", frame)
            cv2.waitKey(1000)

        # ── Capture frames ─────────────────────────────────────────────────
        count = 0
        while count < IMAGES_PER_CLASS:
            ret, frame = cap.read()
            if not ret:
                break
            img_path = os.path.join(cls_dir, f"{count:04d}.jpg")
            cv2.imwrite(img_path, frame)
            count += 1

            pct = int(count / IMAGES_PER_CLASS * 100)
            bar = "#" * (pct // 5) + "-" * (20 - pct // 5)
            display = frame.copy()
            cv2.rectangle(display, (0, 0), (frame.shape[1], 60), (0, 0, 0), -1)
            cv2.putText(display, f"[{bar}] {count}/{IMAGES_PER_CLASS}  {pct}%",
                        (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 0), 2)
            cv2.imshow("Data Collection", display)
            if cv2.waitKey(CAPTURE_DELAY_MS) & 0xFF == ord("q"):
                break

        print(f"    Saved {count} images for: {cls}")

    cap.release()
    cv2.destroyAllWindows()
    print("\n[DONE] Dataset collection complete!")
    print(f"       Images saved to: {DATASET_DIR}")


if __name__ == "__main__":
    capture_dataset()
