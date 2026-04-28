"""
build_exe.py — Package the software as a standalone executable
Uses PyInstaller to bundle Python + all dependencies into
a single distributable folder or .exe file.

Usage
-----
    python build_exe.py            # builds for current OS
    python build_exe.py --onefile  # single .exe (slower startup)
"""

import subprocess
import sys
import os
import argparse
import shutil

APP_NAME    = "PostureGuard"
MAIN_SCRIPT = "main.py"
ICON_FILE   = "assets/icon.ico"   # optional — remove if no icon


def build(onefile: bool = False):
    print("=" * 52)
    print(f"  Building {APP_NAME} executable")
    print("=" * 52)

    # Ensure PyInstaller is installed
    try:
        import PyInstaller
    except ImportError:
        print("[INSTALL] Installing PyInstaller...")
        subprocess.run([sys.executable, "-m", "pip", "install", "pyinstaller"], check=True)

    # Clean previous build
    for d in ["build", "dist", f"{APP_NAME}.spec"]:
        if os.path.exists(d):
            shutil.rmtree(d) if os.path.isdir(d) else os.remove(d)
    print("[CLEAN] Removed previous build artifacts.")

    # Build command
    cmd = [
        sys.executable, "-m", "PyInstaller",
        MAIN_SCRIPT,
        f"--name={APP_NAME}",
        "--noconsole",         # no terminal window (GUI mode)
        "--collect-all=mediapipe",
        "--collect-all=cv2",
        "--hidden-import=sklearn.tree._classes",
        "--hidden-import=sklearn.ensemble._forest",
        "--hidden-import=pyttsx3.drivers",
        "--hidden-import=pyttsx3.drivers.sapi5",  # Windows TTS
        "--hidden-import=plyer.platforms.win.notification",
        "--add-data=config.py;.",
        "--add-data=detector.py;.",
        "--add-data=alert_system.py;.",
        "--add-data=session_logger.py;.",
        "--add-data=assets;assets",
    ]

    if onefile:
        cmd.append("--onefile")
    else:
        cmd.append("--onedir")

    if os.path.exists(ICON_FILE):
        cmd.append(f"--icon={ICON_FILE}")

    print("\n[BUILD] Running PyInstaller...")
    result = subprocess.run(cmd, capture_output=False)

    if result.returncode == 0:
        print(f"\n[OK] Build complete!")
        output = f"dist/{APP_NAME}.exe" if onefile else f"dist/{APP_NAME}/"
        print(f"     Output: {os.path.abspath(output)}")
        print_distribution_instructions(onefile)
    else:
        print("\n[ERROR] Build failed. Check output above for details.")
        sys.exit(1)


def print_distribution_instructions(onefile: bool):
    print("\n" + "=" * 52)
    print("  Distribution Instructions")
    print("=" * 52)
    if onefile:
        print("""
  Single file mode:
  1. Share  dist/PostureGuard.exe  with any Windows user
  2. They double-click to run — no Python installation needed
  3. Note: first launch is slow (unpacking files)
""")
    else:
        print("""
  Folder mode (recommended):
  1. Zip the entire  dist/PostureGuard/  folder
  2. Share the zip file
  3. Recipient extracts and runs  PostureGuard.exe
  4. Faster startup than single-file mode
""")
    print("  Both modes require the model files to be present:")
    print("    models/posture_model.pkl")
    print("    models/label_encoder.pkl")
    print("  Copy these into the dist/ folder before distributing.")
    print("=" * 52)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--onefile", action="store_true",
                   help="Build a single .exe file instead of a folder")
    args = p.parse_args()
    build(onefile=args.onefile)
