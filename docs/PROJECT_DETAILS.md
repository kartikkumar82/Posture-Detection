# Project Details: AI Posture Detection Software

This document explains the structure, workflow, functions, logic, data flow, and current implementation details of the project.

## 1. Project Overview

This project is a webcam-based posture monitoring application. It uses:

- OpenCV for webcam capture, frame display, and image drawing.
- MediaPipe Pose for body landmark detection.
- MediaPipe Face Mesh for eye landmark detection.
- Scikit-learn Random Forest for posture classification.
- Pyttsx3 and Plyer for voice and desktop alerts.
- SQLite for session history.
- Streamlit and Plotly for the analytics dashboard.
- PyInstaller for executable packaging.

The application watches the user through a webcam, detects body posture and eye-related issues, shows visual feedback on the video frame, gives delayed alerts when bad posture persists, and saves session summary data for dashboard analysis.

## 2. Main Features

- Real-time webcam posture monitoring.
- Pose landmark extraction from 33 MediaPipe body landmarks.
- Machine learning classification into posture classes.
- Rule-based detection for slouching, neck tilt, eye closure, and screen distance.
- On-frame skeleton drawing, status bar, correction tips, debug metrics, and posture score.
- Timed alerts that only fire after an issue stays active for a threshold duration.
- Alert cooldown to avoid repeated alerts every frame.
- SQLite session logging.
- Streamlit dashboard for historical analytics.
- Data collection and model training scripts.
- PyInstaller build script for standalone distribution.

## 3. Project Structure

```text
posture_software/
|-- main.py              # Live detection entry point
|-- detector.py          # MediaPipe pose/face detection and feature extraction
|-- alert_system.py      # Timed desktop and voice alert engine
|-- session_logger.py    # SQLite database setup and logging helpers
|-- config.py            # Central paths, thresholds, classes, messages, suggestions
|-- collect_data.py      # Webcam image capture for training data
|-- train_model.py       # Keypoint CSV extraction, model training, evaluation, saving
|-- dashboard.py         # Streamlit analytics dashboard
|-- setup.py             # Setup wizard for dependencies, folders, webcam check
|-- build_exe.py         # PyInstaller packaging script
|-- requirements.txt     # Python dependencies
|-- README.md            # User-facing quick start documentation
|-- PROJECT_DETAILS.md   # This detailed technical documentation
|-- dataset/             # Captured posture images by label
|-- models/              # Created during setup/training; stores trained model files
|-- data/                # Created during setup/runtime; stores CSV and SQLite DB
|-- assets/              # Optional icon/audio assets
```

Current dataset folders visible in the project include `dataset/upright/` and `dataset/slouch/`. The configured posture class list also expects folders for `lean_forward`, `lean_backward`, `neck_tilt_left`, and `neck_tilt_right`.

## 4. Important Runtime Files

The project creates or expects these generated files:

```text
models/posture_model.pkl       # Trained RandomForestClassifier
models/label_encoder.pkl       # LabelEncoder used to map numeric class ids to labels
models/confusion_matrix.png    # Saved by train_model.py after evaluation
data/combined_dataset.csv      # Extracted MediaPipe keypoints plus labels
data/sessions.db               # SQLite session history database
dist/                          # PyInstaller output folder
build/                         # Temporary PyInstaller build folder
PostureGuard.spec              # PyInstaller spec file
```

## 5. Setup Workflow

Recommended setup flow:

```bash
python setup.py
python collect_data.py
python train_model.py
python main.py
streamlit run dashboard.py
```

### setup.py Workflow

`setup.py` is the first-run helper script.

1. Checks that Python is version 3.9 or newer.
2. Installs dependencies from `requirements.txt`.
3. Verifies important imports such as OpenCV, MediaPipe, NumPy, Streamlit, Plotly, and scikit-learn.
4. Creates required folders: `models`, `data`, `assets`, and `dataset`.
5. Tests whether webcam index `0` can be opened and read.
6. Optionally downloads a starter dataset from Zenodo.
7. Prints next-step commands.

Functions in `setup.py`:

- `check_python()`: Validates Python version.
- `install_packages()`: Runs `pip install -r requirements.txt -q`.
- `check_imports()`: Imports required packages and reports failures.
- `check_webcam()`: Opens camera `0` and attempts to read one frame.
- `create_folders()`: Creates the base runtime folders.
- `download_starter_dataset()`: Optionally downloads and extracts a starter dataset.
- `print_next_steps()`: Prints the normal collection, training, detection, and dashboard commands.

## 6. Data Collection Workflow

Data collection is handled by `collect_data.py`.

Command:

```bash
python collect_data.py
```

High-level flow:

1. Creates `dataset/` and one subfolder per class in `POSTURE_CLASSES`.
2. Opens webcam index `0`.
3. Loops through each posture class.
4. Skips a class if it already has at least `IMAGES_PER_CLASS` JPG files.
5. Shows a webcam preview and asks the user to press Space.
6. Shows a 3-second countdown.
7. Captures frames into the class folder as `0000.jpg`, `0001.jpg`, etc.
8. Stops when `IMAGES_PER_CLASS` images are captured or the user presses `q`.

Function in `collect_data.py`:

- `capture_dataset()`: Runs the full webcam capture process.

Important configuration values:

- `POSTURE_CLASSES`: Class labels to collect.
- `DATASET_DIR`: Dataset root folder.
- `IMAGES_PER_CLASS`: Number of images per class.
- `CAPTURE_DELAY_MS`: Delay between captured frames.

## 7. Model Training Workflow

Training is handled by `train_model.py`.

Command:

```bash
python train_model.py
```

High-level flow:

1. If `data/combined_dataset.csv` does not exist, extract pose keypoints from all images.
2. Load the CSV into Pandas.
3. Encode text labels using `LabelEncoder`.
4. Split data into training and test sets with stratification.
5. Train a `RandomForestClassifier`.
6. Print a classification report.
7. Save a confusion matrix image.
8. Save the trained model and label encoder as pickle files.

### Feature Extraction Logic

Each image is processed through MediaPipe Pose in static image mode. MediaPipe returns 33 body landmarks. Each landmark contributes:

- `x`
- `y`
- `z`
- `visibility`

So each sample has:

```text
33 landmarks * 4 values = 132 numeric features
```

The CSV stores:

```text
label,x0,y0,z0,v0,x1,y1,z1,v1,...,x32,y32,z32,v32
```

Functions in `train_model.py`:

- `extract_keypoints_to_csv(output_csv=DATA_CSV)`: Extracts MediaPipe pose landmarks from dataset images and writes labeled rows to CSV.
- `train(csv_path=DATA_CSV)`: Loads CSV data, trains the Random Forest model, evaluates it, and saves model artifacts.

Training configuration from `config.py`:

- `N_ESTIMATORS = 200`
- `MAX_DEPTH = 15`
- `TEST_SIZE = 0.2`
- `RANDOM_STATE = 42`

## 8. Live Detection Workflow

Live monitoring is handled by `main.py`.

Common commands:

```bash
python main.py
python main.py --debug
python main.py --threshold 3
python main.py --camera 1
python main.py --no-audio
```

### Live Runtime Flow

1. Parse command-line arguments.
2. Load the trained model and label encoder from `models/`.
3. Create a `PostureDetector`.
4. Create an `AlertSystem`.
5. Initialize the SQLite database.
6. Open the selected webcam.
7. Read frames in a loop.
8. Process each frame with MediaPipe pose and face detection.
9. If pose keypoints exist, classify posture using the trained model.
10. Merge ML-detected posture issues with rule-based detector issues.
11. Deduplicate active issues.
12. Update alert timers for active issues.
13. Reset alert timers for resolved issues.
14. Draw overlays and correction suggestions.
15. Show the annotated video frame.
16. On `q`, stop monitoring.
17. Calculate session duration, good posture percentage, and main issue.
18. Save the session summary to SQLite.
19. Release camera, OpenCV window, and MediaPipe resources.

### Main Data Flow

```text
Webcam frame
    -> PostureDetector.process(frame)
        -> MediaPipe Pose landmarks
        -> MediaPipe Face Mesh landmarks
        -> 132-value keypoint row
        -> rule-based issue list
        -> annotated frame
    -> RandomForestClassifier.predict(keypoint_row)
    -> LabelEncoder.inverse_transform(prediction)
    -> active issue list
    -> AlertSystem.update(issue)
    -> OpenCV overlay drawing
    -> session statistics
    -> SQLite session row
```

Functions in `main.py`:

- `parse_args()`: Defines CLI options: `--no-audio`, `--debug`, `--threshold`, and `--camera`.
- `load_model()`: Loads `posture_model.pkl` and `label_encoder.pkl`.
- `draw_status_bar(frame, label, confidence, issues)`: Draws the top status bar with current posture and active issues.
- `draw_suggestion_panel(frame, active_issue)`: Draws correction tips for the primary issue.
- `draw_debug_info(frame, result)`: Draws raw metrics such as spine angle, neck tilt, EAR, and eye distance.
- `draw_posture_score(frame, good_frames, total_frames)`: Draws the running good-posture score.
- `run(args)`: Main detection loop and session lifecycle.

## 9. Detector Logic

Core detection is implemented in `detector.py` through the `PostureDetector` class.

### PostureDetector Initialization

`PostureDetector.__init__()` creates:

- `mp.solutions.pose.Pose` with detection and tracking confidence of `0.6`.
- `mp.solutions.face_mesh.FaceMesh` with refined landmarks and confidence of `0.6`.
- MediaPipe drawing utilities and pose connections.

### process(frame)

`process(frame)` accepts one OpenCV BGR frame and returns a dictionary:

```text
pose_landmarks   # Raw MediaPipe pose landmarks or None
face_landmarks   # First Face Mesh result or None
keypoint_row     # 132-feature list for ML classification
spine_angle      # Computed spine angle in degrees
neck_tilt        # Computed neck tilt in degrees
ear              # Average eye aspect ratio
eye_dist         # Inter-eye pixel distance
issues           # Active rule-based issue keys
frame_annotated  # Frame with pose landmarks drawn
```

### Pose-Based Logic

When pose landmarks are found:

1. Draw the skeleton on the output frame.
2. Convert all 33 pose landmarks into the 132-value `keypoint_row`.
3. Calculate spine angle.
4. Calculate neck tilt.
5. Add `slouch` if spine angle is below `SPINE_ANGLE_THRESHOLD`.
6. Add `neck_tilt_left` or `neck_tilt_right` if absolute neck tilt is above `NECK_TILT_THRESHOLD`.

### Eye-Based Logic

When face landmarks are found:

1. Calculate average EAR from left and right eye landmarks.
2. Calculate inter-eye distance in pixels.
3. Add `eye_closing` if EAR is below `EAR_THRESHOLD`.
4. Add `too_close` if eye distance is below `EYE_DIST_MIN`.
5. Add `too_far` if eye distance is above `EYE_DIST_MAX`.

### Geometry Helpers

Methods in `PostureDetector`:

- `process(frame)`: Full frame analysis.
- `extract_features_from_image(image_path)`: Reads an image and returns pose keypoint features or `None`.
- `release()`: Closes MediaPipe resources.
- `_angle(a, b, c)`: Calculates the angle at point `b` formed by points `a-b-c`.
- `_pt(lm, idx, w, h)`: Converts a normalized MediaPipe landmark to pixel coordinates.
- `_spine_angle(lm, w, h)`: Calculates the angle between hips, shoulders, and a vertical reference.
- `_neck_tilt(lm, w, h)`: Calculates head tilt from nose position relative to shoulder midpoint.
- `_ear_one(lm, indices, w, h)`: Calculates Eye Aspect Ratio for one eye.
- `_avg_ear(lm, w, h)`: Averages left and right EAR values.
- `_inter_eye_dist(lm, w, h)`: Measures pixel distance between selected left and right eye landmarks.

## 10. Alert Logic

Alerts are handled by `alert_system.py` through the `AlertSystem` class.

The alert system does not fire immediately when an issue appears. It waits until the issue has stayed active for a configured threshold.

### Alert State

`AlertSystem` stores:

- `threshold`: Seconds an issue must persist before alerting.
- `bad_start`: Dictionary mapping issue key to the timestamp when it started.
- `alerted_at`: Dictionary mapping issue key to the timestamp of the last alert.
- `_tts_lock`: Threading lock to avoid overlapping text-to-speech calls.
- `_tts`: Pyttsx3 text-to-speech engine, if available.
- `_tts_ready`: Whether TTS initialized successfully.

### update(key, is_bad, msg, severity)

Logic:

1. If `is_bad` is true and the key is not already active, store the current timestamp.
2. Calculate how long the issue has stayed active.
3. Check whether the alert threshold has been reached.
4. Check whether the cooldown period has passed.
5. If both checks pass, fire the alert and store the alert timestamp.
6. If `is_bad` is false, remove the issue from `bad_start`.

Methods in `AlertSystem`:

- `update(key, is_bad, msg, severity="warning")`: Updates one issue state and returns `True` if an alert fired.
- `seconds_bad(key)`: Returns how many seconds the issue has been active.
- `reset(key=None)`: Resets one issue or all tracked issue state.
- `_fire(key, msg, severity)`: Sends desktop notification, starts voice thread, and prints alert text.
- `_speak(msg)`: Speaks the message through pyttsx3 inside a lock.

## 11. Session Logging

Session logging is handled by `session_logger.py`.

The database path comes from:

```text
config.DB_PATH = data/sessions.db
```

### Database Tables

`sessions` table:

```text
id            INTEGER PRIMARY KEY AUTOINCREMENT
date          TEXT NOT NULL
start_time    TEXT NOT NULL
duration_sec  INTEGER NOT NULL
good_pct      REAL NOT NULL
total_frames  INTEGER NOT NULL
main_issue    TEXT
notes         TEXT
```

`issue_events` table:

```text
id          INTEGER PRIMARY KEY AUTOINCREMENT
session_id  INTEGER NOT NULL
timestamp   TEXT NOT NULL
issue_key   TEXT NOT NULL
duration_s  REAL NOT NULL
```

Functions in `session_logger.py`:

- `init_db()`: Creates the database folder and tables, then returns a SQLite connection.
- `log_session(con, duration_sec, good_pct, total_frames, main_issue=None, notes=None)`: Inserts one completed session and returns its new ID.
- `log_issue_event(con, session_id, issue_key, duration_s)`: Inserts one issue event row.
- `get_all_sessions(con)`: Returns all sessions as dictionaries, newest first.
- `get_weekly_summary(con)`: Returns session count, average good posture percentage, and total monitored minutes for the last 7 days.

Current behavior: `main.py` saves a summary row to `sessions` at the end of a session. It imports `log_issue_event`, but the current main loop does not insert detailed `issue_events` rows.

## 12. Dashboard Workflow

The dashboard is implemented in `dashboard.py`.

Command:

```bash
streamlit run dashboard.py
```

Dashboard flow:

1. Configure Streamlit page title, icon, and wide layout.
2. Open a cached SQLite connection to `data/sessions.db`.
3. Load all rows from `sessions`, newest first.
4. If no data exists, show an empty-state message.
5. Show summary metrics:
   - Total sessions.
   - Average posture score.
   - Best session score.
   - Total monitored minutes.
6. Show posture score over time as a line chart.
7. Show most common main issues as a horizontal bar chart.
8. Show session quality distribution as a donut chart.
9. Show session history table.
10. Provide a refresh button that clears cached resources and reruns the app.

Dashboard functions:

- `get_connection()`: Cached SQLite connection factory.
- `score_label(pct)`: Maps a good posture percentage to `Excellent`, `Good`, `Poor`, or `Critical`.

## 13. Build and Distribution Workflow

Executable packaging is handled by `build_exe.py`.

Commands:

```bash
python build_exe.py
python build_exe.py --onefile
```

Default mode is folder mode (`--onedir`), which starts faster and is recommended. `--onefile` creates a single executable but startup may be slower.

Build flow:

1. Ensure PyInstaller is installed.
2. Delete previous `build`, `dist`, and spec artifacts.
3. Build a PyInstaller command for `main.py`.
4. Include MediaPipe, OpenCV, scikit-learn, pyttsx3, and Plyer hidden imports/collections.
5. Add project files and assets as data files.
6. Add `assets/icon.ico` if it exists.
7. Run PyInstaller.
8. Print output path and distribution instructions.

Functions in `build_exe.py`:

- `build(onefile=False)`: Runs the PyInstaller build.
- `print_distribution_instructions(onefile)`: Prints sharing instructions and reminds that model files are required.

## 14. Configuration Details

All main settings live in `config.py`.

### Paths

```text
BASE_DIR      # Root project directory
MODEL_PATH    # models/posture_model.pkl
ENCODER_PATH  # models/label_encoder.pkl
DB_PATH       # data/sessions.db
DATASET_DIR   # dataset/
DATA_CSV      # data/combined_dataset.csv
ALERT_SOUND   # assets/alert.wav
```

### Posture Classes

```text
upright
slouch
lean_forward
lean_backward
neck_tilt_left
neck_tilt_right
```

### Detection Thresholds

```text
SPINE_ANGLE_THRESHOLD  = 160
NECK_TILT_THRESHOLD    = 15
HEAD_FORWARD_THRESHOLD = 50
EAR_THRESHOLD          = 0.25
EYE_DIST_MIN           = 80
EYE_DIST_MAX           = 140
```

Notes:

- Spine angle below `160` degrees is treated as slouching.
- Absolute neck tilt above `15` degrees is treated as neck tilt.
- EAR below `0.25` is treated as eye closing.
- Eye distance below `80` pixels is treated as too close.
- Eye distance above `140` pixels is treated as too far.
- `HEAD_FORWARD_THRESHOLD` is defined but not currently used in `detector.py`.

### Alert Settings

```text
ALERT_THRESHOLD_SECONDS = 5
ALERT_COOLDOWN_SECONDS  = 30
```

### Data Collection Settings

```text
IMAGES_PER_CLASS = 200
CAPTURE_DELAY_MS = 30
```

### Training Settings

```text
N_ESTIMATORS = 200
MAX_DEPTH    = 15
TEST_SIZE    = 0.2
RANDOM_STATE = 42
```

### Eye Landmark Indices

```text
LEFT_EYE_IDX  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144]
```

These are MediaPipe Face Mesh landmark indices used to calculate Eye Aspect Ratio.

### Alert Messages and Suggestions

`ALERT_MESSAGES` maps issue keys to spoken/notification messages.

`SUGGESTIONS` maps issue keys to correction tips shown on the right-side video panel.

Issue keys include:

```text
slouch
lean_forward
lean_backward
neck_tilt_left
neck_tilt_right
eye_closing
too_close
too_far
```

## 15. Posture and Issue Logic

The project combines two types of detection:

1. Machine learning classification.
2. Rule-based checks.

### Machine Learning Issues

If the Random Forest predicts a label other than `upright`, `main.py` inserts that label into the active issue list.

Examples:

```text
slouch
lean_forward
lean_backward
neck_tilt_left
neck_tilt_right
```

### Rule-Based Issues

`detector.py` can also add issues directly:

```text
spine_angle < SPINE_ANGLE_THRESHOLD
    -> slouch

abs(neck_tilt) > NECK_TILT_THRESHOLD
    -> neck_tilt_left or neck_tilt_right

ear < EAR_THRESHOLD
    -> eye_closing

eye_dist < EYE_DIST_MIN
    -> too_close

eye_dist > EYE_DIST_MAX
    -> too_far
```

### Deduplication

`main.py` merges ML and rule-based issues, then removes duplicates while preserving order. The first issue becomes the primary issue for the correction panel.

## 16. Session Score Logic

During live monitoring:

- `total_frames` increments for every processed frame.
- `good_frames` increments when the ML label is `upright`.
- `good_pct` is calculated at the end:

```text
good_pct = good_frames / total_frames * 100
```

The on-screen score uses the same idea and updates during the session.

Score color logic in `draw_posture_score()`:

```text
>= 70%  -> green
>= 50%  -> orange
<  50%  -> red
```

Dashboard quality label logic:

```text
>= 90%  -> Excellent
>= 70%  -> Good
>= 50%  -> Poor
<  50%  -> Critical
```

## 17. External Dependencies

Dependencies listed in `requirements.txt`:

```text
opencv-python
mediapipe
numpy
pyttsx3
plyer
scikit-learn
pandas
matplotlib
streamlit
plotly
requests
pygame
Pillow
pyinstaller
```

Dependency purpose:

- `opencv-python`: Camera, frame display, image writing, overlays.
- `mediapipe`: Pose and face landmark detection.
- `numpy`: Geometry and numeric calculations.
- `pyttsx3`: Offline voice alerts.
- `plyer`: Desktop notifications.
- `scikit-learn`: Random Forest model and label encoding.
- `pandas`: CSV loading and dashboard data handling.
- `matplotlib`: Confusion matrix image saving.
- `streamlit`: Dashboard web UI.
- `plotly`: Interactive dashboard charts.
- `requests`: Optional dataset download.
- `pygame`: Listed dependency, not currently used by the visible source files.
- `Pillow`: Image support dependency, not directly used by the visible source files.
- `pyinstaller`: Standalone executable packaging.

## 18. Current Implementation Notes

- `main.py` imports `log_issue_event` but does not currently call it, so detailed alert events are not written to `issue_events`.
- `config.py` defines `HEAD_FORWARD_THRESHOLD`, but the current detector code does not use it.
- `alert_system.py` supports TTS and desktop notifications. The `--no-audio` CLI flag is parsed in `main.py`, but the current `AlertSystem` constructor does not receive or use that setting.
- `main.py` uses `os.path.exists()` inside `load_model()`. The current file should import `os` for this function to run correctly.
- The configured class list includes six classes, but the currently visible dataset folders include only `upright` and `slouch`. Training will skip missing class folders.
- `build_exe.py` reminds the user to copy model files into the distribution folder. The PyInstaller command currently adds code/assets but does not add `models/` directly.

## 19. Typical End-to-End Workflow

For a fresh machine:

```bash
python setup.py
python collect_data.py
python train_model.py
python main.py
```

For analytics:

```bash
streamlit run dashboard.py
```

For packaging:

```bash
python build_exe.py
```

## 20. Quick Troubleshooting

### Model not found

Run:

```bash
python train_model.py
```

Expected files:

```text
models/posture_model.pkl
models/label_encoder.pkl
```

### Webcam not opening

Try:

```bash
python main.py --camera 1
```

Also check camera permissions in the operating system.

### No dashboard data

Run at least one monitoring session:

```bash
python main.py
```

Quit with `q` so the session can be saved.

### Training has too few classes

Check `dataset/` and make sure there are class folders with enough images for each configured posture label.

### Alerts repeat too often or too slowly

Adjust:

```text
ALERT_THRESHOLD_SECONDS
ALERT_COOLDOWN_SECONDS
```

or run:

```bash
python main.py --threshold 3
```

