"""
face_auth.py

Loads stored ArcFace embeddings and verifies a webcam frame
against them using cosine similarity.
"""

import os
import numpy as np

# ── Configuration ────────────────────────────────────────────────────────────
EMBEDDINGS_PATH   = os.path.join(os.path.dirname(__file__), "..", "data", "embeddings.npy")
SIMILARITY_THRESHOLD = 0.40   # cosine similarity — tune if needed (0.0–1.0)


class FaceAuthenticator:
    """
    Loads stored face embeddings and exposes a single verify() method
    that accepts a pre-analysed InsightFace face object.
    """

    def __init__(self, embeddings_path: str = EMBEDDINGS_PATH):
        self.embeddings_path = embeddings_path
        self.stored_embeddings = self._load_embeddings()

    # ── Private ──────────────────────────────────────────────────────────────

    def _load_embeddings(self) -> np.ndarray:
        path = os.path.abspath(self.embeddings_path)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Embeddings file not found: {path}\n"
                "Run scripts/generate_embeddings.py first."
            )
        embeddings = np.load(path)
        print(f"[FaceAuth] Loaded {len(embeddings)} stored embeddings.")
        return embeddings   # shape (N, 512)

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Return cosine similarity in [−1, 1]. Embeddings are pre-normalised."""
        return float(np.dot(a, b))

    # ── Public ───────────────────────────────────────────────────────────────

    def verify(self, query_embedding: np.ndarray) -> tuple[bool, float]:
        """
        Compare query_embedding against all stored embeddings.

        Returns
        -------
        (matched: bool, best_similarity: float)
        """
        if self.stored_embeddings is None or len(self.stored_embeddings) == 0:
            return False, 0.0

        similarities = [
            self._cosine_similarity(query_embedding, stored)
            for stored in self.stored_embeddings
        ]
        best = max(similarities)
        matched = best >= SIMILARITY_THRESHOLD

        print(
            f"[FaceAuth] Best similarity: {best:.4f}  "
            f"(threshold: {SIMILARITY_THRESHOLD})  →  {'PASS' if matched else 'FAIL'}"
        )
        return matched, best