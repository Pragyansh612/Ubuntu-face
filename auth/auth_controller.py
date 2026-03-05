"""
auth_controller.py

Orchestrates face authentication and gesture authentication.

Logic:
    1. Open webcam
    2. For SCAN_DURATION seconds, collect frames
    3. Try face recognition on every frame
    4. If face passes  → return True immediately
    5. If no face pass → try gesture on accumulated frames
    6. If gesture passes → return True
    7. Else → return False (fall through to password)

Returns exit code:
    0 → authentication succeeded
    1 → authentication failed
"""

import sys
import time
import cv2
import numpy as np

# Append project root so auth/ imports work when called from pam/
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from auth.face_auth    import FaceAuthenticator
from auth.gesture_auth import GestureAuthenticator

# ── Configuration ────────────────────────────────────────────────────────────
SCAN_DURATION   = 3       # seconds to scan before giving up
CAMERA_INDEX    = 0       # /dev/video0
FRAME_WIDTH     = 640
FRAME_HEIGHT    = 480
DET_SIZE        = (320, 320)   # InsightFace detection resolution


def _load_insightface():
    """Load InsightFace model (CPU). Returns app or None on failure."""
    try:
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(
            name="buffalo_sc",
            providers=["CPUExecutionProvider"]
        )
        app.prepare(ctx_id=-1, det_size=DET_SIZE)
        return app
    except Exception as exc:
        print(f"[AuthController] InsightFace load failed: {exc}")
        return None


def authenticate() -> bool:
    """
    Run biometric authentication.
    Returns True on success, False on failure.
    """
    print("[AuthController] Starting biometric authentication…")

    # ── Initialise models ────────────────────────────────────────────────────
    face_app = _load_insightface()

    try:
        face_auth    = FaceAuthenticator()
    except FileNotFoundError as exc:
        print(f"[AuthController] {exc}")
        face_auth = None

    gesture_auth = GestureAuthenticator()

    # ── Open camera ──────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("[AuthController] ERROR: Cannot open webcam.")
        gesture_auth.close()
        return False

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    start       = time.time()
    face_passed = False
    gesture_frames_rgb = []   # collect frames for gesture evaluation

    # ── Scan loop ─────────────────────────────────────────────────────────────
    try:
        while (time.time() - start) < SCAN_DURATION:
            ret, frame = cap.read()
            if not ret:
                continue

            # ── Face authentication ──────────────────────────────────────────
            if face_app is not None and face_auth is not None and not face_passed:
                faces = face_app.get(frame)
                if faces:
                    # Use largest face
                    face = max(
                        faces,
                        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
                    )
                    matched, score = face_auth.verify(face.normed_embedding)
                    if matched:
                        print("[AuthController] Face authentication PASSED.")
                        face_passed = True
                        break   # success — stop scanning immediately

            # Collect RGB frame for gesture check
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gesture_frames_rgb.append(frame_rgb)

    finally:
        cap.release()
        print("[AuthController] Camera released.")

    if face_passed:
        gesture_auth.close()
        return True

    # ── Gesture authentication ────────────────────────────────────────────────
    print("[AuthController] Face not matched. Checking gesture…")
    for frame_rgb in gesture_frames_rgb:
        if gesture_auth.check_gesture(frame_rgb):
            print("[AuthController] Gesture authentication PASSED.")
            gesture_auth.close()
            return True

    gesture_auth.close()
    print("[AuthController] Both biometric methods FAILED.")
    return False


if __name__ == "__main__":
    success = authenticate()
    sys.exit(0 if success else 1)