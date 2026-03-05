"""
auth_controller.py

Checks /tmp/face_auth_cache first (written by pre_scan.py at GDM startup).
If cache is fresh (< 30 seconds old) → use it instantly.
If cache is stale or missing → fall back to live scan.
"""

import sys
import time
import os
import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from auth.face_auth    import FaceAuthenticator
from auth.gesture_auth import GestureAuthenticator

# ── Configuration ─────────────────────────────────────────────────────────────
SCAN_DURATION   = 3
CAMERA_INDEX    = 0
FRAME_WIDTH     = 640
FRAME_HEIGHT    = 480
DET_SIZE        = (320, 320)
CACHE_FILE      = "/tmp/face_auth_cache"
CACHE_MAX_AGE   = 30    # seconds — discard cache older than this


def _read_cache() -> bool | None:
    """
    Read cached result from pre_scan.py.

    Returns:
        True  → cached SUCCESS and it's fresh
        False → cached FAIL and it's fresh
        None  → no cache, cache too old, or unreadable
    """
    try:
        if not os.path.isfile(CACHE_FILE):
            return None

        with open(CACHE_FILE, "r") as f:
            line = f.read().strip()

        parts = line.split()
        if len(parts) != 2:
            return None

        result, timestamp = parts[0], int(parts[1])
        age = time.time() - timestamp

        if age > CACHE_MAX_AGE:
            print(f"[AuthController] Cache expired ({age:.1f}s old) — running live scan.")
            os.remove(CACHE_FILE)
            return None

        print(f"[AuthController] Cache hit: {result} ({age:.1f}s old)")

        # Consume the cache so it can't be reused for a second login
        os.remove(CACHE_FILE)
        return result == "SUCCESS"

    except Exception as e:
        print(f"[AuthController] Cache read error: {e}")
        return None


def _load_insightface():
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


def _live_scan() -> bool:
    """Full live scan — used when cache is missing or stale."""
    print("[AuthController] Running live biometric scan…")

    face_app = _load_insightface()

    try:
        face_auth = FaceAuthenticator()
    except FileNotFoundError as exc:
        print(f"[AuthController] {exc}")
        face_auth = None

    gesture_auth = GestureAuthenticator()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("[AuthController] ERROR: Cannot open webcam.")
        gesture_auth.close()
        return False

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    start              = time.time()
    face_passed        = False
    gesture_frames_rgb = []

    try:
        while (time.time() - start) < SCAN_DURATION:
            ret, frame = cap.read()
            if not ret:
                continue

            if face_app and face_auth and not face_passed:
                faces = face_app.get(frame)
                if faces:
                    face = max(
                        faces,
                        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
                    )
                    matched, score = face_auth.verify(face.normed_embedding)
                    if matched:
                        print("[AuthController] Face authentication PASSED.")
                        face_passed = True
                        break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gesture_frames_rgb.append(frame_rgb)

    finally:
        cap.release()
        print("[AuthController] Camera released.")

    if face_passed:
        gesture_auth.close()
        return True

    print("[AuthController] Face not matched. Checking gesture…")
    for frame_rgb in gesture_frames_rgb:
        if gesture_auth.check_gesture(frame_rgb):
            print("[AuthController] Gesture authentication PASSED.")
            gesture_auth.close()
            return True

    gesture_auth.close()
    print("[AuthController] Both biometric methods FAILED.")
    return False


def authenticate() -> bool:
    # Try cache first
    cached = _read_cache()
    if cached is not None:
        print(f"[AuthController] Using cached result: {'PASS' if cached else 'FAIL'}")
        return cached

    # Fall back to live scan
    return _live_scan()


if __name__ == "__main__":
    success = authenticate()
    sys.exit(0 if success else 1)