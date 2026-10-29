"""MediaPipe Hand Tracking wrapper for AirDesk.

Handles MediaPipe HandLandmarker initialization, frame processing, normalized & pixel landmark
extraction for up to 2 hands, and visualization overlays.
"""

import os
import urllib.request
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import cv2
import numpy as np
import mediapipe as mp

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "assets", "hand_landmarker.task")


@dataclass
class SingleHandData:
    """Dataclass holding normalized and pixel coordinates for a single hand."""
    label: str  # 'Left' or 'Right'
    score: float  # Detection confidence
    landmarks_norm: List[Tuple[float, float, float]]  # (x, y, z) in range [0, 1]
    landmarks_px: List[Tuple[int, int]]  # (x, y) in frame pixels


class HandTracker:
    """MediaPipe HandLandmarker wrapper class."""

    CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),        # Index finger
        (5, 9), (9, 10), (10, 11), (11, 12),   # Middle finger
        (9, 13), (13, 14), (14, 15), (15, 16), # Ring finger
        (13, 17), (17, 18), (18, 19), (19, 20),# Pinky
        (0, 17)                                # Palm base
    ]

    FINGERTIP_IDS = [4, 8, 12, 16, 20]

    def __init__(self, model_path: Optional[str] = None, max_hands: int = 2, min_detection_confidence: float = 0.5):
        """Initialize HandTracker with MediaPipe HandLandmarker."""
        if model_path is None:
            model_path = DEFAULT_MODEL_PATH
        
        self.model_path = os.path.abspath(model_path)
        self._ensure_model_exists()

        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        RunningMode = mp.tasks.vision.RunningMode

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.model_path),
            running_mode=RunningMode.IMAGE,
            num_hands=max_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_detection_confidence
        )

        self.landmarker = HandLandmarker.create_from_options(options)

    def _ensure_model_exists(self) -> None:
        """Download model file if not found locally."""
        if not os.path.exists(self.model_path):
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            print(f"[HandTracker] Model file missing. Downloading from {MODEL_URL}...")
            urllib.request.urlretrieve(MODEL_URL, self.model_path)
            print(f"[HandTracker] Saved model to {self.model_path}")

    def process_frame(self, frame_bgr: np.ndarray) -> List[SingleHandData]:
        """Detect hands in BGR frame and return structured list of detected hands."""
        if frame_bgr is None or frame_bgr.size == 0:
            return []

        h, w, _ = frame_bgr.shape
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        try:
            results = self.landmarker.detect(mp_image)
        except Exception as e:
            print(f"[HandTracker] Error during hand detection: {e}")
            return []

        detected_hands: List[SingleHandData] = []
        if not results.hand_landmarks:
            return detected_hands

        for idx, landmarks in enumerate(results.hand_landmarks):
            label = "Hand"
            score = 1.0
            if results.handedness and idx < len(results.handedness):
                categories = results.handedness[idx]
                if categories:
                    label = categories[0].category_name
                    score = categories[0].score

            norm_list = []
            px_list = []
            for lm in landmarks:
                norm_list.append((lm.x, lm.y, lm.z))
                px_x = int(np.clip(lm.x * w, 0, w - 1))
                px_y = int(np.clip(lm.y * h, 0, h - 1))
                px_list.append((px_x, px_y))

            detected_hands.append(SingleHandData(
                label=label,
                score=score,
                landmarks_norm=norm_list,
                landmarks_px=px_list
            ))

        return detected_hands

    def draw_landmarks(self, frame_bgr: np.ndarray, hands_data: List[SingleHandData]) -> np.ndarray:
        """Draw hand skeleton connections, joints, and tips overlay on the frame."""
        for hand in hands_data:
            pts = hand.landmarks_px

            # Draw connections
            for p1_idx, p2_idx in self.CONNECTIONS:
                pt1 = pts[p1_idx]
                pt2 = pts[p2_idx]
                cv2.line(frame_bgr, pt1, pt2, (255, 200, 100), 2, cv2.LINE_AA)

            # Draw joint nodes
            for idx, pt in enumerate(pts):
                if idx in self.FINGERTIP_IDS:
                    # Fingertips highlighted (Green for index, Cyan for thumb, White for rest)
                    if idx == 8:
                        color = (0, 255, 0)      # Index tip: Green
                        radius = 8
                    elif idx == 4:
                        color = (255, 255, 0)    # Thumb tip: Cyan/Yellow
                        radius = 8
                    else:
                        color = (0, 215, 255)    # Other tips: Orange/Gold
                        radius = 6
                    cv2.circle(frame_bgr, pt, radius, color, -1, cv2.LINE_AA)
                    cv2.circle(frame_bgr, pt, radius + 2, (0, 0, 0), 1, cv2.LINE_AA)
                else:
                    cv2.circle(frame_bgr, pt, 4, (200, 200, 250), -1, cv2.LINE_AA)

            # Draw label near wrist (landmark 0)
            wrist_pt = pts[0]
            label_text = f"{hand.label} ({int(hand.score * 100)}%)"
            cv2.putText(frame_bgr, label_text, (wrist_pt[0] - 20, wrist_pt[1] + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        return frame_bgr

    def close(self):
        """Release landmarker resource."""
        if hasattr(self, 'landmarker') and self.landmarker:
            self.landmarker.close()
