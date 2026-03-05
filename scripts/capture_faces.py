"""
capture_faces.py

Opens the webcam, detects a face using OpenCV's Haar cascade,
and automatically captures N images to dataset/user/raw/.

Usage:
    python scripts/capture_faces.py --count 50
"""

import cv2
import os
import argparse
import time

# ── Configuration ────────────────────────────────────────────────────────────
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset", "user", "raw")
HAAR_CASCADE = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
CAPTURE_INTERVAL = 0.3   # seconds between auto-captures
DISPLAY_SCALE    = 1.0   # set < 1 to shrink preview window


def parse_args():
    parser = argparse.ArgumentParser(description="Capture face images for dataset.")
    parser.add_argument(
        "--count", type=int, default=50,
        help="Number of images to capture (default: 50)"
    )
    return parser.parse_args()


def ensure_output_dir(path: str):
    os.makedirs(path, exist_ok=True)
    print(f"[INFO] Saving images to: {os.path.abspath(path)}")


def capture_faces(target_count: int):
    ensure_output_dir(OUTPUT_DIR)

    face_cascade = cv2.CascadeClassifier(HAAR_CASCADE)
    if face_cascade.empty():
        raise RuntimeError("Failed to load Haar cascade. Check OpenCV installation.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam.")

    # Count existing images so we don't overwrite them
    existing = [f for f in os.listdir(OUTPUT_DIR) if f.lower().endswith((".jpg", ".png"))]
    img_index = len(existing)
    captured  = 0
    last_time = 0.0

    print(f"[INFO] Starting capture. Need {target_count} images. Press Q to quit early.")

    try:
        while captured < target_count:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Failed to grab frame. Retrying…")
                continue

            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
            )

            now = time.time()
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Auto-capture at set interval
                if now - last_time >= CAPTURE_INTERVAL:
                    filename = os.path.join(OUTPUT_DIR, f"face_{img_index:04d}.jpg")
                    cv2.imwrite(filename, frame)
                    img_index  += 1
                    captured   += 1
                    last_time   = now
                    print(f"[INFO] Captured {captured}/{target_count}")

            # Overlay progress
            cv2.putText(
                frame, f"Captured: {captured}/{target_count}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2
            )

            if DISPLAY_SCALE != 1.0:
                frame = cv2.resize(frame, (0, 0), fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)

            cv2.imshow("Face Capture — press Q to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[INFO] Quit by user.")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"[DONE] Captured {captured} images to '{OUTPUT_DIR}'.")


if __name__ == "__main__":
    args = parse_args()
    capture_faces(args.count)