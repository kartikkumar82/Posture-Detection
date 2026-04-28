"""
config.py — Central configuration for AI Posture Detection Software
All thresholds, paths, and settings in one place.
"""

import os

# ── Paths ─────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH    = os.path.join(BASE_DIR, "models", "posture_model.pkl")
ENCODER_PATH  = os.path.join(BASE_DIR, "models", "label_encoder.pkl")
DB_PATH       = os.path.join(BASE_DIR, "data",   "sessions.db")
DATASET_DIR   = os.path.join(BASE_DIR, "dataset")
DATA_CSV      = os.path.join(BASE_DIR, "data",   "combined_dataset.csv")
ALERT_SOUND   = os.path.join(BASE_DIR, "assets", "alert.wav")

# ── Posture Classes ────────────────────────────────────────────────────────
POSTURE_CLASSES = [
    "upright",
    "slouch",
    "lean_forward",
    "lean_backward",
    "neck_tilt_left",
    "neck_tilt_right",
]

# ── Detection Thresholds ───────────────────────────────────────────────────
SPINE_ANGLE_THRESHOLD    = 160   # degrees — below = slouching
NECK_TILT_THRESHOLD      = 15    # degrees — above = neck tilt
HEAD_FORWARD_THRESHOLD   = 50    # pixels  — above = forward head
EAR_THRESHOLD            = 0.25  # ratio   — below = eyes closing
EYE_DIST_MIN             = 80    # pixels  — below = too close to screen
EYE_DIST_MAX             = 140   # pixels  — above = too far from screen

# ── Alert System ───────────────────────────────────────────────────────────
ALERT_THRESHOLD_SECONDS  = 5     # seconds before alert fires
ALERT_COOLDOWN_SECONDS   = 30    # seconds before same alert can repeat

# ── Data Collection ────────────────────────────────────────────────────────
IMAGES_PER_CLASS         = 200   # images to capture per posture class
CAPTURE_DELAY_MS         = 30    # milliseconds between captured frames

# ── Model Training ─────────────────────────────────────────────────────────
N_ESTIMATORS             = 200
MAX_DEPTH                = 15
TEST_SIZE                = 0.2
RANDOM_STATE             = 42

# ── MediaPipe Eye Landmarks ────────────────────────────────────────────────
LEFT_EYE_IDX  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDX = [33,  160, 158, 133, 153, 144]

# ── Alert Messages ─────────────────────────────────────────────────────────
ALERT_MESSAGES = {
    "slouch":         "You are slouching. Roll your shoulders back and sit up straight.",
    "lean_forward":   "You are leaning forward. Move your keyboard closer and use your backrest.",
    "lean_backward":  "You are leaning too far back. Sit upright against your backrest.",
    "neck_tilt_left": "Your neck is tilting left. Level your head so both ears are at the same height.",
    "neck_tilt_right":"Your neck is tilting right. Level your head so both ears are at the same height.",
    "eye_closing":    "Your eyes are closing. Take a 20-second break and look away from the screen.",
    "too_close":      "You are too close to the screen. Move back to 50 to 70 centimetres.",
    "too_far":        "You are too far from the screen. Move closer for comfortable viewing.",
}

# ── Correction Suggestions ────────────────────────────────────────────────
SUGGESTIONS = {
    "slouch": [
        "Roll your shoulders back and down",
        "Imagine a string pulling the top of your head upward",
        "Your ears should be directly above your shoulders",
        "Engage your core — tighten abdominal muscles slightly",
        "Try a lumbar roll behind your lower back",
    ],
    "lean_forward": [
        "Move your keyboard closer — arms should be relaxed at 90°",
        "Sit fully back in your chair and use the backrest",
        "Raise your monitor to eye level to avoid leaning in",
        "Position your mouse close to avoid overreaching",
    ],
    "lean_backward": [
        "Sit upright — your back should touch the chair backrest",
        "Tilt your chair so your hips are slightly higher than knees",
        "Bring your monitor closer so you don't lean back to see it",
    ],
    "neck_tilt_left": [
        "Level your head — both ears should be at the same height",
        "Gently stretch your neck to the right for 15 seconds",
        "Check if your monitor is centered directly in front of you",
    ],
    "neck_tilt_right": [
        "Level your head — both ears should be at the same height",
        "Gently stretch your neck to the left for 15 seconds",
        "Check if your monitor is centered directly in front of you",
    ],
    "eye_closing": [
        "Follow the 20-20-20 rule: look 20 feet away for 20 seconds",
        "Blink fully and slowly 10 times right now",
        "Reduce screen brightness to match your room lighting",
        "Consider blue-light filtering glasses for long sessions",
    ],
    "too_close": [
        "Move back — ideal screen distance is 50 to 70 cm",
        "Increase your font size or browser zoom instead of leaning in",
        "Apply 20-20-20 rule every 20 minutes to rest your eyes",
    ],
    "too_far": [
        "Move your chair closer to the screen",
        "Increase your font size if text is hard to read at distance",
        "Check your monitor brightness — dim screens cause squinting",
    ],
}
