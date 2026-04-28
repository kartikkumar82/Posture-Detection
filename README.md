# PostureGuard — AI Posture Detection Software

Real-time posture and eye monitoring using your webcam, MediaPipe, and machine learning.

---

## Features

- Live posture classification (upright, slouch, lean forward/backward, neck tilt)
- Eye fatigue detection using Eye Aspect Ratio (EAR)
- Screen distance monitoring
- Timed voice alerts + OS toast notifications (alerts fire after 5 seconds, not every frame)
- On-screen correction tip panel
- Session history stored in SQLite
- Streamlit analytics dashboard
- Packagable as a standalone `.exe` (no Python needed on target machine)

---

## Quick Start

```bash
# 1. Clone or download the project folder
cd posture_software

# 2. Run setup (installs libraries, checks webcam, optional dataset download)
python setup.py

# 3. Collect training images (200 per posture class)
python collect_data.py

# 4. Train the model
python train_model.py

# 5. Run live detection
python main.py

# 6. View your dashboard (separate terminal)
streamlit run dashboard.py
```

---

## File Structure

```
posture_software/
├── main.py            ← Live detection entry point
├── collect_data.py    ← Webcam image capture
├── train_model.py     ← Model training + evaluation
├── detector.py        ← MediaPipe pose + eye engine
├── alert_system.py    ← Timed multi-channel alerts
├── session_logger.py  ← SQLite session recording
├── dashboard.py       ← Streamlit analytics dashboard
├── config.py          ← All settings and thresholds
├── setup.py           ← One-command setup wizard
├── build_exe.py       ← Package as standalone executable
├── requirements.txt   ← All dependencies
│
├── models/
│   ├── posture_model.pkl     ← Trained classifier (created by train_model.py)
│   └── label_encoder.pkl     ← Class encoder
│
├── data/
│   ├── sessions.db           ← SQLite session history
│   └── combined_dataset.csv  ← Training data CSV
│
├── dataset/                  ← Captured training images
│   ├── upright/
│   ├── slouch/
│   ├── lean_forward/
│   ├── lean_backward/
│   ├── neck_tilt_left/
│   └── neck_tilt_right/
│
└── assets/
    ├── icon.ico              ← App icon (optional)
    └── alert.wav             ← Custom alert sound (optional)
```

---

## Command Reference

```bash
# Detection options
python main.py                    # default mode
python main.py --debug            # show angle values on screen
python main.py --no-audio         # disable TTS voice alerts
python main.py --threshold 3      # alert after 3 seconds (default: 5)
python main.py --camera 1         # use second camera (default: 0)

# Build standalone executable
python build_exe.py               # folder mode (recommended)
python build_exe.py --onefile     # single .exe file
```

---

## Configuration

All thresholds and settings live in `config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `SPINE_ANGLE_THRESHOLD` | 160° | Below = slouching |
| `NECK_TILT_THRESHOLD` | 15° | Above = neck tilt |
| `EAR_THRESHOLD` | 0.25 | Below = eyes closing |
| `EYE_DIST_MIN` | 80px | Below = too close to screen |
| `EYE_DIST_MAX` | 140px | Above = too far from screen |
| `ALERT_THRESHOLD_SECONDS` | 5 | Seconds before alert fires |
| `IMAGES_PER_CLASS` | 200 | Images captured per posture class |

---

## Hardware Requirements

- Webcam (built-in laptop camera is fine)
- 4GB RAM minimum (8GB recommended)
- CPU: Intel i3 / AMD Ryzen 3 or better
- GPU: Not required (MediaPipe runs on CPU)
- OS: Windows 10/11, macOS, Linux

---

## Dependencies

See `requirements.txt`. Key libraries:

- `opencv-python` — video capture and drawing
- `mediapipe` — pose and face landmark detection
- `scikit-learn` — Random Forest classifier
- `pyttsx3` — offline text-to-speech
- `plyer` — cross-platform OS notifications
- `streamlit` + `plotly` — dashboard UI

---

## License

MIT License — free for personal and academic use.
