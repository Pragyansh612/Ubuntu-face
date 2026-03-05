"""
pre_scan.py

Runs in the background when GDM greeter loads.
Scans face/gesture and caches the result to /tmp/face_auth_cache
so auth_controller.py can read it instantly when PAM fires.

Cache format:
    SUCCESS <unix_timestamp>
    FAIL    <unix_timestamp>
"""

import sys
import os
import time

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

CACHE_FILE = "/tmp/face_auth_cache"


def write_cache(result: str):
    timestamp = str(int(time.time()))
    with open(CACHE_FILE, "w") as f:
        f.write(f"{result} {timestamp}\n")
    # Readable by all users so PAM (running as root) and GDM can both access it
    os.chmod(CACHE_FILE, 0o644)


if __name__ == "__main__":
    try:
        # Import here so errors don't prevent the file being importable
        import cv2
        import numpy as np

        from auth.face_auth    import FaceAuthenticator
        from auth.gesture_auth import GestureAuthenticator

        SCAN_DURATION = 5      # slightly longer since user is still on profile screen
        CAMERA_INDEX  = 0
        DET_SIZE      = (320, 320)

        def _load_insightface():
            try:
                from insightface.app import FaceAnalysis
                app = FaceAnalysis(
                    name="buffalo_sc",
                    providers=["CPUExecutionProvider"]
                )
                app.prepare(ctx_id=-1, det_size=DET_SIZE)
                return app
            except Exception as e:
                print(f"[PreScan] InsightFace failed: {e}")
                return None

        face_app = _load_insightface()

        try:
            face_auth = FaceAuthenticator()
        except FileNotFoundError:
            face_auth = None

        gesture_auth = GestureAuthenticator()

        cap = cv2.VideoCapture(CAMERA_INDEX)
        if not cap.isOpened():
            write_cache("FAIL")
            sys.exit(1)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        start              = time.time()
        face_passed        = False
        gesture_frames_rgb = []

        try:
            while (time.time() - start) < SCAN_DURATION:
                ret, frame = cap.read()
                if not ret:
                    continue

                # Face check
                if face_app and face_auth and not face_passed:
                    faces = face_app.get(frame)
                    if faces:
                        face = max(
                            faces,
                            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
                        )
                        matched, score = face_auth.verify(face.normed_embedding)
                        if matched:
                            face_passed = True
                            break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                gesture_frames_rgb.append(frame_rgb)

        finally:
            cap.release()

        if face_passed:
            write_cache("SUCCESS")
            gesture_auth.close()
            sys.exit(0)

        # Gesture check
        for frame_rgb in gesture_frames_rgb:
            if gesture_auth.check_gesture(frame_rgb):
                write_cache("SUCCESS")
                gesture_auth.close()
                sys.exit(0)

        gesture_auth.close()
        write_cache("FAIL")
        sys.exit(1)

    except Exception as e:
        print(f"[PreScan] Error: {e}")
        write_cache("FAIL")
        sys.exit(1)