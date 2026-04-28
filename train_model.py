"""
train_model.py — Train the posture classifier
Reads keypoint CSV, trains Random Forest, evaluates,
and saves model + label encoder to models/.
"""

import os
import pickle
import csv
import numpy as np
import pandas as pd
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

from config import (
    DATASET_DIR, DATA_CSV, MODEL_PATH, ENCODER_PATH,
    POSTURE_CLASSES, N_ESTIMATORS, MAX_DEPTH,
    TEST_SIZE, RANDOM_STATE,
)


# ── Step 1: Extract keypoints from images ──────────────────────────────────

def extract_keypoints_to_csv(output_csv: str = DATA_CSV):
    """
    Run MediaPipe Pose over all captured images and save
    33 landmarks × 4 values + label to CSV.
    """
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    pose   = mp.solutions.pose.Pose(static_image_mode=True)
    header = ["label"] + [f"{ax}{i}" for i in range(33) for ax in ["x", "y", "z", "v"]]
    total  = 0

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for cls in POSTURE_CLASSES:
            cls_dir = os.path.join(DATASET_DIR, cls)
            if not os.path.isdir(cls_dir):
                print(f"  [SKIP] {cls} folder not found.")
                continue

            cls_count = 0
            for img_file in sorted(os.listdir(cls_dir)):
                if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                img = cv2.imread(os.path.join(cls_dir, img_file))
                if img is None:
                    continue
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                res = pose.process(rgb)
                if res.pose_landmarks:
                    row = [cls]
                    for lm in res.pose_landmarks.landmark:
                        row += [lm.x, lm.y, lm.z, lm.visibility]
                    writer.writerow(row)
                    cls_count += 1
                    total += 1

            print(f"  {cls}: {cls_count} samples extracted")

    pose.close()
    print(f"\n  Total samples: {total}")
    print(f"  CSV saved   : {output_csv}")
    return output_csv


# ── Step 2: Train model ────────────────────────────────────────────────────

def train(csv_path: str = DATA_CSV):
    """
    Load CSV, train Random Forest, print metrics, save model.
    """
    print("\n[TRAIN] Loading dataset...")
    df = pd.read_csv(csv_path)
    print(f"        {len(df)} rows | {df['label'].nunique()} classes")
    print(f"        {df['label'].value_counts().to_dict()}")

    le = LabelEncoder()
    y  = le.fit_transform(df["label"])
    X  = df.drop(columns=["label"]).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    print(f"\n[TRAIN] Training Random Forest ({N_ESTIMATORS} trees)...")
    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # ── Evaluation ─────────────────────────────────────────────────────────
    y_pred = model.predict(X_test)
    print("\n[EVAL] Classification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred,
        display_labels=le.classes_,
        ax=ax,
        xticks_rotation=45,
    )
    plt.title("Posture Classifier — Confusion Matrix")
    plt.tight_layout()
    cm_path = os.path.join(os.path.dirname(MODEL_PATH), "confusion_matrix.png")
    plt.savefig(cm_path)
    print(f"[EVAL] Confusion matrix saved: {cm_path}")
    plt.close()

    # ── Save ───────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH,   "wb") as f: pickle.dump(model, f)
    with open(ENCODER_PATH, "wb") as f: pickle.dump(le,    f)
    print(f"\n[SAVE] Model   : {MODEL_PATH}")
    print(f"[SAVE] Encoder : {ENCODER_PATH}")
    return model, le


# ── Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("  AI Posture Detection — Model Training")
    print("=" * 50)

    if not os.path.exists(DATA_CSV):
        print("\n[STEP 1] Extracting keypoints from captured images...")
        extract_keypoints_to_csv()
    else:
        print(f"\n[STEP 1] CSV already exists: {DATA_CSV}")

    print("\n[STEP 2] Training classifier...")
    train()
    print("\n[DONE] Training complete!")
