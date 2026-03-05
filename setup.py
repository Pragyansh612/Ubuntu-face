#!/usr/bin/env python3
"""
setup.py — Face Gesture Unlock — One-shot setup script

Run this after cloning the repo:
    python setup.py

It will:
    1. Check system dependencies
    2. Create and configure virtualenv
    3. Install Python packages
    4. Collect face images
    5. Prepare dataset
    6. Generate embeddings
    7. Test face auth
    8. Test gesture auth
    9. Configure PAM
    10. Install systemd pre-scan service
    11. Configure GNOME keyring
"""

import os
import sys
import subprocess
import shutil
import time
import textwrap

# ── Colours ───────────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def green(s):  return f"{GREEN}{s}{RESET}"
def yellow(s): return f"{YELLOW}{s}{RESET}"
def red(s):    return f"{RED}{s}{RESET}"
def cyan(s):   return f"{CYAN}{s}{RESET}"
def bold(s):   return f"{BOLD}{s}{RESET}"

# ── Helpers ───────────────────────────────────────────────────────────────────
REPO_ROOT   = os.path.dirname(os.path.abspath(__file__))
VENV_DIR    = os.path.join(REPO_ROOT, "venv")
VENV_PYTHON = os.path.join(VENV_DIR, "bin", "python")


def header(title: str):
    print(f"\n{CYAN}{'─' * 60}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{CYAN}{'─' * 60}{RESET}\n")


def info(msg: str):
    print(f"  {GREEN}✔{RESET}  {msg}")


def warn(msg: str):
    print(f"  {YELLOW}⚠{RESET}  {msg}")


def error(msg: str):
    print(f"  {RED}✖{RESET}  {msg}")


def ask(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    try:
        val = input(f"\n  {BOLD}{prompt}{suffix}:{RESET} ").strip()
        return val if val else default
    except (EOFError, KeyboardInterrupt):
        print()
        return default


def confirm(prompt: str, default: bool = True) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    val = ask(f"{prompt} {suffix}", "y" if default else "n").lower()
    return val in ("y", "yes", "")


def run(cmd: list, capture: bool = False, env: dict = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        capture_output=capture,
        text=True,
        env=env or os.environ.copy()
    )


def run_venv(args: list, capture: bool = False) -> subprocess.CompletedProcess:
    """Run a command inside the virtualenv."""
    return run([VENV_PYTHON] + args, capture=capture)


def run_script(script_path: str, extra_args: list = []):
    """Run a project script inside the venv, streaming output."""
    result = run_venv([script_path] + extra_args)
    return result.returncode == 0


def abort(msg: str):
    error(msg)
    print(f"\n  {RED}Setup aborted.{RESET}\n")
    sys.exit(1)


# ── Step functions ────────────────────────────────────────────────────────────

def step_check_system():
    header("Step 1 — Checking system requirements")

    # Must be Linux
    if sys.platform != "linux":
        abort("This system is designed for Ubuntu Linux only.")
    info("Platform: Linux ✔")

    # Python version
    major, minor = sys.version_info[:2]
    if major < 3 or minor < 8:
        abort(f"Python 3.8+ required. Found {major}.{minor}")
    info(f"Python {major}.{minor} ✔")

    # Check for sudo
    result = run(["sudo", "-n", "true"], capture=True)
    if result.returncode != 0:
        warn("This script needs sudo for PAM and systemd setup.")
        warn("You will be prompted for your password when needed.")
    else:
        info("sudo available ✔")

    # Check webcam
    import glob
    cameras = glob.glob("/dev/video*")
    if not cameras:
        warn("No webcam detected at /dev/video*. Face/gesture auth may not work.")
    else:
        info(f"Webcam found: {', '.join(cameras)} ✔")

    # Check apt packages
    needed_apt = ["libpam-python", "pamtester", "python3-venv", "python3-pip"]
    missing    = []
    for pkg in needed_apt:
        r = run(["dpkg", "-s", pkg], capture=True)
        if r.returncode != 0:
            missing.append(pkg)

    if missing:
        warn(f"Missing apt packages: {', '.join(missing)}")
        if confirm("Install them now with sudo apt?"):
            run(["sudo", "apt", "install", "-y"] + missing)
            info("Apt packages installed ✔")
        else:
            abort("Required packages not installed.")
    else:
        info("All apt packages present ✔")

    # Find pam_python.so
    result = run(
        ["find", "/usr/lib", "/lib", "-name", "pam_python.so"],
        capture=True
    )
    pam_python_path = result.stdout.strip().splitlines()
    if not pam_python_path:
        abort("pam_python.so not found even after install. Try: sudo apt install libpam-python")

    pam_so = pam_python_path[0]
    info(f"pam_python.so found at: {pam_so} ✔")
    return pam_so


def step_virtualenv():
    header("Step 2 — Setting up Python virtual environment")

    if os.path.isdir(VENV_DIR):
        warn(f"Virtualenv already exists at {VENV_DIR}")
        if not confirm("Reuse existing virtualenv?"):
            shutil.rmtree(VENV_DIR)
            info("Removed old virtualenv.")
        else:
            info("Reusing existing virtualenv ✔")
            return

    info("Creating virtualenv…")
    result = run([sys.executable, "-m", "venv", VENV_DIR])
    if result.returncode != 0:
        abort("Failed to create virtualenv.")
    info(f"Virtualenv created at {VENV_DIR} ✔")


def step_install_packages():
    header("Step 3 — Installing Python packages")

    req_file = os.path.join(REPO_ROOT, "requirements.txt")
    if not os.path.isfile(req_file):
        abort(f"requirements.txt not found at {req_file}")

    info("Upgrading pip…")
    run_venv(["-m", "pip", "install", "--upgrade", "pip", "--quiet"])

    info("Installing packages from requirements.txt (this may take a few minutes)…")
    result = run_venv(["-m", "pip", "install", "-r", req_file])
    if result.returncode != 0:
        abort("pip install failed. Check your internet connection.")
    info("All packages installed ✔")


def step_collect_faces():
    header("Step 4 — Collecting face images")

    raw_dir = os.path.join(REPO_ROOT, "dataset", "user", "raw")
    os.makedirs(raw_dir, exist_ok=True)

    existing = [
        f for f in os.listdir(raw_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    if existing:
        info(f"Found {len(existing)} existing images in dataset/user/raw/")
        if not confirm("Capture more images?", default=False):
            return
    else:
        print(textwrap.dedent(f"""
  {YELLOW}No face images found.{RESET}
  You need at least 20–50 images of your face for good accuracy.
  The webcam will open and auto-capture images.
  Sit in good lighting, look at the camera, and move your head slightly.
        """))

    count = ask("How many images to capture?", "50")
    try:
        count = int(count)
    except ValueError:
        count = 50

    script = os.path.join(REPO_ROOT, "scripts", "capture_faces.py")
    info(f"Starting capture ({count} images)… Press Q in the webcam window to stop early.")
    success = run_script(script, ["--count", str(count)])

    if not success:
        warn("Face capture exited with an error. You can re-run this step later.")
    else:
        info("Face capture complete ✔")


def step_prepare_dataset():
    header("Step 5 — Preparing dataset")

    raw_dir  = os.path.join(REPO_ROOT, "dataset", "user", "raw")
    images   = [
        f for f in os.listdir(raw_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ] if os.path.isdir(raw_dir) else []

    if not images:
        abort("No images in dataset/user/raw/. Run step 4 first.")

    info(f"Found {len(images)} images. Renaming sequentially…")
    script  = os.path.join(REPO_ROOT, "scripts", "prepare_dataset.py")
    success = run_script(script)

    if not success:
        abort("Dataset preparation failed.")
    info("Dataset prepared ✔")


def step_generate_embeddings():
    header("Step 6 — Generating face embeddings")

    info("Loading InsightFace model and generating embeddings…")
    info("(First run will download the buffalo_sc model ~30 MB)")

    script  = os.path.join(REPO_ROOT, "scripts", "generate_embeddings.py")
    success = run_script(script)

    if not success:
        abort("Embedding generation failed.")

    emb_path = os.path.join(REPO_ROOT, "data", "embeddings.npy")
    if not os.path.isfile(emb_path):
        abort(f"Embeddings file not created at {emb_path}")

    info(f"Embeddings saved to data/embeddings.npy ✔")


def step_test_face():
    header("Step 7 — Testing face authentication")

    print(f"  {YELLOW}The webcam will open for 3 seconds. Look at it directly.{RESET}\n")
    if not confirm("Ready to test face recognition?"):
        warn("Skipping face test.")
        return

    script  = os.path.join(REPO_ROOT, "auth", "auth_controller.py")

    # Temporarily patch to run only face portion — just run full controller
    result  = run_venv([script], capture=True)

    if result.returncode == 0:
        info("Face authentication test PASSED ✔")
    else:
        warn("Face authentication test failed.")
        warn("Try: lower SIMILARITY_THRESHOLD in auth/face_auth.py (currently 0.40)")
        warn("Or recapture faces in better lighting and regenerate embeddings.")
        if not confirm("Continue setup anyway?", default=False):
            abort("Setup stopped at face test.")


def step_test_gesture():
    header("Step 8 — Testing gesture authentication")

    print(textwrap.dedent(f"""
  {YELLOW}Gesture: both hands showing the middle finger.{RESET}
  Make sure both hands are clearly visible in the frame.
  You will have 5 seconds.
    """))

    if not confirm("Ready to test gesture recognition?", default=False):
        warn("Skipping gesture test.")
        return

    test_script = os.path.join(REPO_ROOT, "test_gesture.py")

    if not os.path.isfile(test_script):
        # Write it inline if not present
        with open(test_script, "w") as f:
            f.write(textwrap.dedent("""
import cv2, sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from auth.gesture_auth import GestureAuthenticator

SCAN_DURATION = 5
print("[TEST] Show BOTH hands doing the middle finger. You have 5 seconds.")
gesture_auth = GestureAuthenticator()
cap = cv2.VideoCapture(0)
start = time.time()
passed = False
try:
    while (time.time() - start) < SCAN_DURATION:
        ret, frame = cap.read()
        if not ret: continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if gesture_auth.check_gesture(frame_rgb):
            passed = True
            break
        elapsed = SCAN_DURATION - (time.time() - start)
        cv2.putText(frame, f"Show middle finger (both hands) - {elapsed:.1f}s",
                    (10,40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,200,255), 2)
        cv2.imshow("Gesture Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
finally:
    cap.release()
    cv2.destroyAllWindows()
    gesture_auth.close()
sys.exit(0 if passed else 1)
"""))

    result = run_venv([test_script])
    if result.returncode == 0:
        info("Gesture authentication test PASSED ✔")
    else:
        warn("Gesture test failed or skipped.")
        if not confirm("Continue setup anyway?", default=False):
            abort("Setup stopped at gesture test.")


def step_configure_pam(pam_so_path: str):
    header("Step 9 — Configuring PAM authentication")

    pam_file    = "/etc/pam.d/gdm-password"
    pam_backup  = "/etc/pam.d/gdm-password.bak"
    marker      = "face_unlock.py"

    # Check if already configured
    try:
        with open(pam_file, "r") as f:
            content = f.read()
        if marker in content:
            info("PAM already configured ✔")
            return
    except PermissionError:
        pass  # will sudo below

    print(textwrap.dedent(f"""
  {YELLOW}This step edits /etc/pam.d/gdm-password to enable biometric login.{RESET}
  A backup will be saved to {pam_backup}.
  The line added uses 'sufficient' — if biometrics fail, password login still works.
    """))

    if not confirm("Configure PAM now?"):
        warn("PAM configuration skipped. Biometric login will not work.")
        return

    # Backup
    result = run(["sudo", "cp", pam_file, pam_backup])
    if result.returncode != 0:
        abort("Failed to backup PAM file.")
    info(f"Backup saved to {pam_backup} ✔")

    # Read current content
    result = run(["sudo", "cat", pam_file], capture=True)
    if result.returncode != 0:
        abort("Cannot read PAM file.")

    lines     = result.stdout.splitlines()
    new_line  = (
        f"auth    sufficient      {pam_so_path} "
        f"{REPO_ROOT}/pam/face_unlock.py"
    )

    # Insert after pam_succeed_if line
    new_lines = []
    inserted  = False
    for line in lines:
        new_lines.append(line)
        if "pam_succeed_if.so" in line and not inserted:
            new_lines.append(new_line)
            inserted = True

    if not inserted:
        # fallback — insert after first auth line
        for i, line in enumerate(new_lines):
            if line.strip().startswith("auth") and "pam_nologin" not in line:
                new_lines.insert(i + 1, new_line)
                inserted = True
                break

    if not inserted:
        new_lines.insert(1, new_line)

    new_content = "\n".join(new_lines) + "\n"

    # Write via sudo tee
    proc = subprocess.Popen(
        ["sudo", "tee", pam_file],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL
    )
    proc.communicate(input=new_content.encode())

    if proc.returncode != 0:
        abort("Failed to write PAM file.")

    info("PAM file updated ✔")

    # Update face_unlock.py to use venv python
    unlock_file = os.path.join(REPO_ROOT, "pam", "face_unlock.py")
    with open(unlock_file, "r") as f:
        unlock_content = f.read()

    unlock_content = unlock_content.replace(
        "_PYTHON = sys.executable",
        f'_PYTHON = "{VENV_PYTHON}"'
    )
    with open(unlock_file, "w") as f:
        f.write(unlock_content)

    info(f"face_unlock.py updated to use venv Python ✔")

    # Test with pamtester
    username = os.environ.get("SUDO_USER") or os.environ.get("USER")
    info(f"Testing PAM config with pamtester (user: {username})…")
    result = run(["sudo", "pamtester", "gdm-password", username, "authenticate"], capture=True)
    if result.returncode == 0:
        info("pamtester: successfully authenticated ✔")
    else:
        warn("pamtester did not authenticate. This may be okay — test by locking your screen.")


def step_install_service():
    header("Step 10 — Installing systemd pre-scan service")

    print(textwrap.dedent("""
  This installs a systemd service that runs the biometric scan at boot,
  so your face is already verified by the time you click your profile.
  This removes the 3-5 second delay at login.
    """))

    if not confirm("Install systemd pre-scan service?"):
        warn("Systemd service skipped. Login will have a short delay.")
        return

    service_content = textwrap.dedent(f"""
[Unit]
Description=Face unlock pre-scan
After=systemd-user-sessions.service
Before=gdm.service

[Service]
Type=oneshot
User=root
ExecStart={VENV_PYTHON} {REPO_ROOT}/auth/pre_scan.py
TimeoutSec=15
RemainAfterExit=no

[Install]
WantedBy=graphical.target
""").strip()

    service_path = "/etc/systemd/system/face-prescan.service"

    # Write via sudo tee
    proc = subprocess.Popen(
        ["sudo", "tee", service_path],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL
    )
    proc.communicate(input=service_content.encode())
    info("Service file written ✔")

    run(["sudo", "systemctl", "daemon-reload"])
    run(["sudo", "systemctl", "enable", "face-prescan.service"])
    info("Service enabled at boot ✔")

    # Test it
    info("Running service now to verify…")
    result = run(["sudo", "systemctl", "start", "face-prescan.service"], capture=True)
    time.sleep(2)

    cache_file = "/tmp/face_auth_cache"
    if os.path.isfile(cache_file):
        with open(cache_file) as f:
            cache = f.read().strip()
        info(f"Cache written: {cache} ✔")
    else:
        warn("Cache file not found after service run. Check: sudo journalctl -u face-prescan")


def step_keyring():
    header("Step 11 — GNOME Keyring configuration")

    print(textwrap.dedent(f"""
  {YELLOW}Issue:{RESET} When biometric login bypasses the password,
  GNOME Keyring stays locked and prompts you for a password
  the first time any app needs it (wifi, browser, etc).

  {YELLOW}Fix:{RESET} Set the keyring password to blank so it auto-unlocks.
  This is safe — the login gate (your face/gesture/password) is unchanged.
  The keyring is only accessible after you are already logged in.
    """))

    if not confirm("Open Seahorse to set keyring password to blank?"):
        warn("Keyring not configured. You may see keyring prompts after biometric login.")
        print(f"\n  You can fix this later by running: {cyan('seahorse')}")
        return

    result = run(["which", "seahorse"], capture=True)
    if result.returncode != 0:
        run(["sudo", "apt", "install", "-y", "seahorse"])

    print(f"\n  {YELLOW}Instructions:{RESET}")
    print("  1. Seahorse will open")
    print("  2. Find 'Login' keyring under Passwords")
    print("  3. Right-click → Change Password")
    print("  4. Enter your current password")
    print("  5. Leave new password BLANK (press Enter twice)")
    print("  6. Confirm the warning\n")
    input("  Press Enter to open Seahorse…")
    run(["seahorse"])


def step_done():
    header("Setup Complete!")

    print(textwrap.dedent(f"""
  {GREEN}Your face-gesture-unlock system is installed and ready.{RESET}

  {BOLD}How it works:{RESET}
    • At boot, the camera scans your face silently
    • When you click your profile, PAM checks the cached result
    • If face matched  → instant login ✅
    • If face failed   → try middle finger gesture ✅
    • If both failed   → normal password prompt ✅

  {BOLD}Useful commands:{RESET}
    Retrain face model:
      {cyan('cd ' + REPO_ROOT)}
      {cyan('source venv/bin/activate')}
      {cyan('python scripts/capture_faces.py --count 50')}
      {cyan('python scripts/prepare_dataset.py')}
      {cyan('python scripts/generate_embeddings.py')}

    Disable biometric login:
      {cyan('sudo cp /etc/pam.d/gdm-password.bak /etc/pam.d/gdm-password')}

    Check service logs:
      {cyan('sudo journalctl -u face-prescan.service')}

    Reboot to test the full flow:
      {cyan('sudo reboot')}
    """))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"""
{CYAN}{BOLD}
╔══════════════════════════════════════════════════════════╗
║          Face + Gesture Unlock — Setup Wizard            ║
║          Biometric login for Ubuntu (like Windows Hello) ║
╚══════════════════════════════════════════════════════════╝
{RESET}""")

    print(f"  Repo: {REPO_ROOT}")
    print(f"  User: {os.environ.get('USER', 'unknown')}\n")

    if not confirm("Start setup?"):
        print("  Cancelled.")
        sys.exit(0)

    pam_so = step_check_system()
    step_virtualenv()
    step_install_packages()
    step_collect_faces()
    step_prepare_dataset()
    step_generate_embeddings()
    step_test_face()
    step_test_gesture()
    step_configure_pam(pam_so)
    step_install_service()
    step_keyring()
    step_done()


if __name__ == "__main__":
    main()