"""
prepare_dataset.py

Reads every image inside dataset/user/raw/, renames them sequentially
(img1.jpg, img2.jpg, …) in-place, and reports the result.

Usage:
    python scripts/prepare_dataset.py
"""

import os
import shutil

# ── Configuration ────────────────────────────────────────────────────────────
RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset", "user", "raw")
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def prepare_dataset(raw_dir: str):
    raw_dir = os.path.abspath(raw_dir)

    if not os.path.isdir(raw_dir):
        print(f"[ERROR] Directory not found: {raw_dir}")
        return

    # Collect all valid image files
    images = sorted([
        f for f in os.listdir(raw_dir)
        if os.path.splitext(f)[1].lower() in VALID_EXTENSIONS
    ])

    if not images:
        print("[WARN] No images found. Add images to the raw/ folder first.")
        return

    print(f"[INFO] Found {len(images)} images. Renaming…")

    # Use a temp prefix to avoid name collisions during rename
    TEMP_PREFIX = "__tmp_rename_"

    # Step 1 — rename to temp names
    for filename in images:
        src = os.path.join(raw_dir, filename)
        tmp = os.path.join(raw_dir, TEMP_PREFIX + filename)
        os.rename(src, tmp)

    # Step 2 — rename temp names to final sequential names
    temp_files = sorted([
        f for f in os.listdir(raw_dir) if f.startswith(TEMP_PREFIX)
    ])

    for idx, filename in enumerate(temp_files, start=1):
        src  = os.path.join(raw_dir, filename)
        dest = os.path.join(raw_dir, f"img{idx}.jpg")
        shutil.move(src, dest)
        print(f"  {filename}  →  img{idx}.jpg")

    print(f"[DONE] Renamed {len(temp_files)} images.")


if __name__ == "__main__":
    prepare_dataset(RAW_DIR)