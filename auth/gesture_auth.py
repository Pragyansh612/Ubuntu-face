"""
gesture_auth.py

Uses MediaPipe Hands to detect a specific hand gesture:
    Both hands visible + both showing the middle finger raised.

Compatible with both old and new MediaPipe APIs.
"""

import numpy as np

# ── Landmark indices ──────────────────────────────────────────────────────────
FINGER_TIPS = {"index": 8,  "middle": 12, "ring": 16, "pinky": 20}
FINGER_PIPS = {"index": 6,  "middle": 10, "ring": 14, "pinky": 18}
THUMB_TIP = 4
THUMB_IP  = 3
THUMB_MCP = 2


class GestureAuthenticator:
    """
    Detects whether the user is showing the 'double middle finger' gesture.

    Gesture rules:
        - Exactly 2 hands detected
        - For each hand: middle finger extended, all other fingers folded
    """

    def __init__(self):
        self._init_mediapipe()
        print("[GestureAuth] MediaPipe Hands model initialised.")

    def _init_mediapipe(self):
        """Initialise MediaPipe, handling both old and new API styles."""
        import mediapipe as mp

        # ── Try new-style API (mediapipe >= 0.10) ────────────────────────────
        try:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision
            import urllib.request, os, tempfile

            # Download the hand-landmarker task model if not already present
            model_path = os.path.join(
                os.path.dirname(__file__), "..", "data", "hand_landmarker.task"
            )
            model_path = os.path.abspath(model_path)

            if not os.path.isfile(model_path):
                print("[GestureAuth] Downloading hand_landmarker.task model (~23 MB)…")
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                url = (
                    "https://storage.googleapis.com/mediapipe-models/"
                    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
                )
                urllib.request.urlretrieve(url, model_path)
                print("[GestureAuth] Model downloaded.")

            options = mp_vision.HandLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=model_path),
                num_hands=2,
                min_hand_detection_confidence=0.7,
                min_hand_presence_confidence=0.6,
                min_tracking_confidence=0.6,
                running_mode=mp_vision.RunningMode.IMAGE,
            )
            self._detector    = mp_vision.HandLandmarker.create_from_options(options)
            self._api_style   = "new"
            self._mp_image    = mp.Image
            self._image_format = mp.ImageFormat.SRGB
            return

        except Exception as e:
            print(f"[GestureAuth] New-style API unavailable ({e}), trying legacy API…")

        # ── Fall back to legacy API (mediapipe < 0.10) ────────────────────────
        try:
            self._hands_model = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.6,
            )
            self._api_style = "legacy"
            return
        except Exception as e:
            raise RuntimeError(f"[GestureAuth] Could not initialise MediaPipe: {e}")

    # ── Geometry helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _is_finger_extended(landmarks, tip_idx: int, pip_idx: int) -> bool:
        return landmarks[tip_idx].y < landmarks[pip_idx].y

    @staticmethod
    def _is_thumb_folded(landmarks) -> bool:
        tip = np.array([landmarks[THUMB_TIP].x, landmarks[THUMB_TIP].y])
        ip  = np.array([landmarks[THUMB_IP].x,  landmarks[THUMB_IP].y])
        mcp = np.array([landmarks[THUMB_MCP].x, landmarks[THUMB_MCP].y])
        return np.linalg.norm(tip - mcp) <= np.linalg.norm(ip - mcp) * 1.5

    def _is_middle_finger_only(self, landmarks) -> bool:
        middle_up    = self._is_finger_extended(landmarks, FINGER_TIPS["middle"], FINGER_PIPS["middle"])
        index_up     = self._is_finger_extended(landmarks, FINGER_TIPS["index"],  FINGER_PIPS["index"])
        ring_up      = self._is_finger_extended(landmarks, FINGER_TIPS["ring"],   FINGER_PIPS["ring"])
        pinky_up     = self._is_finger_extended(landmarks, FINGER_TIPS["pinky"],  FINGER_PIPS["pinky"])

        return middle_up and (not index_up) and (not ring_up) and (not pinky_up)

    # ── Normalise landmarks to a common list-like format ─────────────────────

    @staticmethod
    def _new_api_landmarks_to_list(hand_landmarks):
        """
        New API returns NormalizedLandmark objects in hand_landmarks.landmarks.
        Wrap them in a plain list so geometry helpers work identically.
        """
        class _LM:
            def __init__(self, x, y, z): self.x = x; self.y = y; self.z = z

        return [_LM(lm.x, lm.y, lm.z) for lm in hand_landmarks]

    # ── Public ────────────────────────────────────────────────────────────────

    def check_gesture(self, frame_rgb) -> bool:
        """
        Analyse a single RGB frame (numpy array, HxWx3 uint8).
        Returns True if both hands show the middle-finger gesture.
        """
        if self._api_style == "new":
            return self._check_gesture_new(frame_rgb)
        else:
            return self._check_gesture_legacy(frame_rgb)

    def _check_gesture_new(self, frame_rgb) -> bool:
        import mediapipe as mp
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result   = self._detector.detect(mp_image)

        if not result.hand_landmarks or len(result.hand_landmarks) < 2:
            return False

        confirmed = [
            self._is_middle_finger_only(
                self._new_api_landmarks_to_list(hand)
            )
            for hand in result.hand_landmarks
        ]
        both_match = all(confirmed)
        if both_match:
            print("[GestureAuth] Double middle-finger gesture confirmed → PASS")
        return both_match

    def _check_gesture_legacy(self, frame_rgb) -> bool:
        results = self._hands_model.process(frame_rgb)

        if not results.multi_hand_landmarks:
            return False
        if len(results.multi_hand_landmarks) < 2:
            return False

        confirmed = [
            self._is_middle_finger_only(hand.landmark)
            for hand in results.multi_hand_landmarks
        ]
        both_match = all(confirmed)
        if both_match:
            print("[GestureAuth] Double middle-finger gesture confirmed → PASS")
        return both_match

    def close(self):
        """Release MediaPipe resources."""
        if self._api_style == "new":
            self._detector.close()
        else:
            self._hands_model.close()