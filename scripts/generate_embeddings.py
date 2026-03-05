"""
generate_embeddings.py

Reads face images from dataset/user/raw/, generates ArcFace embeddings
using InsightFace, and saves them to data/embeddings.npy.

Usage:
    python scripts/generate_embeddings.py
"""

import os
import sys
import numpy as np
import cv2

# ── Configuration ────────────────────────────────────────────────────────────
RAW_DIR        = os.path.join(os.path.dirname(__file__), "..", "dataset", "user", "raw")
OUTPUT_DIR     = os.path.join(os.path.dirname(__file__), "..", "data")
EMBEDDINGS_PATH = os.path.join(OUTPUT_DIR, "embeddings.npy")
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_insightface_model():
    """Load InsightFace ArcFace model (CPU only)."""
    try:
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(
            name="buffalo_sc",          # lightweight single-model pack
            providers=["CPUExecutionProvider"]
        )
        app.prepare(ctx_id=-1, det_size=(320, 320))   # ctx_id=-1 → CPU
        print("[INFO] InsightFace model loaded.")
        return app
    except ImportError:
        print("[ERROR] insightface not installed. Run: pip install insightface onnxruntime")
        sys.exit(1)


def generate_embeddings(raw_dir: str, output_path: str):
    raw_dir     = os.path.abspath(raw_dir)
    output_path = os.path.abspath(output_path)

    if not os.path.isdir(raw_dir):
        print(f"[ERROR] Dataset directory not found: {raw_dir}")
        sys.exit(1)

    images = sorted([
        f for f in os.listdir(raw_dir)
        if os.path.splitext(f)[1].lower() in VALID_EXTENSIONS
    ])

    if not images:
        print("[ERROR] No images found in dataset. Run capture_faces.py first.")
        sys.exit(1)

    app = load_insightface_model()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    embeddings = []
    skipped    = 0

    for filename in images:
        img_path = os.path.join(raw_dir, filename)
        img      = cv2.imread(img_path)

        if img is None:
            print(f"[WARN] Cannot read image: {filename} — skipping.")
            skipped += 1
            continue

        faces = app.get(img)

        if not faces:
            print(f"[WARN] No face detected in: {filename} — skipping.")
            skipped += 1
            continue

        # Use the largest detected face
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        emb  = face.normed_embedding   # already L2-normalised
        embeddings.append(emb)
        print(f"[OK] {filename}  →  embedding shape {emb.shape}")

    if not embeddings:
        print("[ERROR] No embeddings were generated. Check your images.")
        sys.exit(1)

    emb_array = np.array(embeddings)   # shape (N, 512)
    np.save(output_path, emb_array)
    print(f"\n[DONE] Saved {len(embeddings)} embeddings to '{output_path}'.")
    if skipped:
        print(f"[WARN] {skipped} image(s) skipped (no face detected).")


if __name__ == "__main__":
    generate_embeddings(RAW_DIR, EMBEDDINGS_PATH)