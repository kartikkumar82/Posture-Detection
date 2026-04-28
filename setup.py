"""
setup.py — One-command setup wizard
Run this first! It installs dependencies, checks webcam,
and optionally downloads the MultiPosture starter dataset.
"""

import subprocess
import sys
import os
import importlib

REQUIRED_PACKAGES = [
    "cv2",
    "mediapipe",
    "numpy",
    "pyttsx3",
    "plyer",
    "sklearn",
    "pandas",
    "matplotlib",
    "streamlit",
    "plotly",
    "requests",
]

PIP_NAMES = {
    "cv2":       "opencv-python",
    "sklearn":   "scikit-learn",
    "PIL":       "Pillow",
}


def check_python():
    major, minor = sys.version_info[:2]
    if major < 3 or minor < 9:
        print(f"[ERROR] Python 3.9+ required. You have {major}.{minor}")
        sys.exit(1)
    print(f"[OK] Python {major}.{minor}")


def install_packages():
    print("\n[INSTALL] Installing required packages...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "-q"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print("[WARN] Some packages may have failed to install:")
        print(result.stderr[:500])
    else:
        print("[OK] All packages installed.")


def check_imports():
    print("\n[CHECK] Verifying imports...")
    all_ok = True
    for pkg in REQUIRED_PACKAGES:
        try:
            importlib.import_module(pkg)
            print(f"  [OK] {pkg}")
        except ImportError:
            pip_name = PIP_NAMES.get(pkg, pkg)
            print(f"  [FAIL] {pkg} — try: pip install {pip_name}")
            all_ok = False
    return all_ok


def check_webcam():
    print("\n[CHECK] Testing webcam...")
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, _ = cap.read()
            cap.release()
            if ret:
                print("[OK] Webcam is working.")
                return True
        print("[WARN] Webcam not detected. Check your camera connection.")
        return False
    except Exception as e:
        print(f"[WARN] Webcam check failed: {e}")
        return False


def create_folders():
    print("\n[SETUP] Creating project folders...")
    folders = ["models", "data", "assets", "dataset"]
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        print(f"  [OK] {folder}/")


def download_starter_dataset():
    print("\n[DATASET] Would you like to download the MultiPosture starter dataset?")
    print("          (4,800 pre-labeled CSV samples — no capture needed)")
    ans = input("          Download? (y/n): ").strip().lower()
    if ans != "y":
        print("          Skipped. You can collect your own data with: python collect_data.py")
        return

    try:
        import requests, zipfile, io
        url = "https://zenodo.org/records/14230872/files/multiposture.zip"
        print("          Downloading from Zenodo...")
        r = requests.get(url, timeout=60)
        if r.status_code == 200:
            z = zipfile.ZipFile(io.BytesIO(r.content))
            z.extractall("data/multiposture/")
            print("[OK] Dataset downloaded to data/multiposture/")
        else:
            print(f"[WARN] Download failed (HTTP {r.status_code}). Download manually from:")
            print("       https://zenodo.org/records/14230872")
    except Exception as e:
        print(f"[WARN] Download error: {e}")
        print("       Download manually from: https://zenodo.org/records/14230872")


def print_next_steps():
    print("\n" + "=" * 52)
    print("  Setup complete! What to do next:")
    print("=" * 52)
    print()
    print("  STEP 1 — Collect your posture data:")
    print("           python collect_data.py")
    print()
    print("  STEP 2 — Train the model:")
    print("           python train_model.py")
    print()
    print("  STEP 3 — Run live detection:")
    print("           python main.py")
    print()
    print("  STEP 4 — View your dashboard:")
    print("           streamlit run dashboard.py")
    print()
    print("  OPTIONS:")
    print("           python main.py --debug       # show raw metrics")
    print("           python main.py --threshold 3 # faster alerts")
    print("           python main.py --no-audio    # silent mode")
    print("=" * 52)


if __name__ == "__main__":
    print("=" * 52)
    print("  AI Posture Detection — Setup Wizard")
    print("=" * 52)

    check_python()
    install_packages()
    ok = check_imports()
    create_folders()
    check_webcam()

    if ok:
        download_starter_dataset()

    print_next_steps()
