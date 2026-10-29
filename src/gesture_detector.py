"""Gesture detection module for AirDesk.

Classifies MediaPipe hand landmark data per hand per frame into typed gestures
with temporal hysteresis (N-frame confirmation threshold) to prevent spurious
gesture transitions caused by hand jitter or transitional hand poses.

Gestures (priority order):
- PINCH: Thumb tip and index tip within distance threshold (select/grab).
- OPEN_PALM: Index, middle, ring, and pinky fingers extended (stop/erase).
- POINT: ONLY index finger extended, all others curled (draw cursor).
- FIST: All fingers curled (neutral/idle).
- NONE: Unrecognized configuration.

Temporal hysteresis:
- Each gesture must be held for CONFIRM_FRAMES frames before being confirmed.
- This prevents single-frame blips from triggering state changes.
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import math

from hand_tracker import SingleHandData
from geometry import distance_2d, calculate_angle_2d, midpoint_2d


# Named constants for gesture thresholds (scale-invariant relative to palm size)
PINCH_THRESHOLD_PALM_RATIO = 0.32   # Pinch if thumb-index tip dist < 32% of palm size
FINGER_EXTEND_RATIO = 1.05          # Finger extended if dist(wrist, tip) > 1.05 * dist(wrist, pip)

# POINT gesture: index must be clearly extended, others must be clearly NOT extended
POINT_INDEX_RATIO = 1.10            # Index must be >1.10x ratio to be "pointing"
POINT_OTHER_RATIO = 0.95            # Other fingers must be <0.95x ratio to be "curled"

# Temporal stability: how many consecutive frames a gesture must hold before confirming
CONFIRM_FRAMES = 4                  # ~4 frames at 30fps ≈ 133ms confirmation delay
RELEASE_FRAMES = 2                  # How quickly a gesture is considered released


class GestureType(Enum):
    NONE = auto()
    POINT = auto()
    PINCH = auto()
    OPEN_PALM = auto()
    FIST = auto()

    def __str__(self) -> str:
        return self.name


@dataclass
class HandGesture:
    """Dataclass holding gesture detection outputs for a single hand."""
    hand_label: str                                   # 'Left' or 'Right'
    gesture: GestureType = GestureType.NONE
    raw_gesture: GestureType = GestureType.NONE       # Unfiltered raw gesture this frame
    index_tip: Tuple[float, float] = (0.0, 0.0)       # Pixel coordinates
    thumb_tip: Tuple[float, float] = (0.0, 0.0)       # Pixel coordinates
    pinch_point: Tuple[float, float] = (0.0, 0.0)     # Midpoint of thumb & index tips
    palm_center: Tuple[float, float] = (0.0, 0.0)     # Middle MCP joint (landmark 9)
    confidence: float = 0.0
    is_pinch: bool = False
    is_point: bool = False
    is_open_palm: bool = False
    is_fist: bool = False
    gesture_hold_frames: int = 0                       # How many frames current gesture confirmed


@dataclass
class FrameGestures:
    """Dataclass holding all gesture detection outputs for a frame."""
    hands: List[HandGesture] = field(default_factory=list)
    inter_hand_distance: Optional[float] = None       # Distance in pixels between 2 hands
    relative_hand_angle: Optional[float] = None       # Relative angle in degrees between 2 hands

    def get_hand_by_label(self, label: str) -> Optional[HandGesture]:
        """Find gesture data by hand label ('Left' or 'Right')."""
        for hand in self.hands:
            if hand.hand_label.lower() == label.lower():
                return hand
        return None


class _HandTemporalState:
    """Per-hand temporal state for gesture hysteresis."""
    def __init__(self):
        self.candidate_gesture: GestureType = GestureType.NONE
        self.candidate_count: int = 0
        self.confirmed_gesture: GestureType = GestureType.NONE
        self.confirmed_hold_frames: int = 0
        self.release_count: int = 0

    def update(self, raw_gesture: GestureType) -> GestureType:
        """
        Update temporal state and return the confirmed (stable) gesture.

        A gesture is confirmed only after CONFIRM_FRAMES consecutive frames
        of the same raw gesture. Once confirmed, it is held until RELEASE_FRAMES
        frames of a different gesture are seen (prevents micro-flickers on release).
        """
        if raw_gesture == self.candidate_gesture:
            self.candidate_count = min(self.candidate_count + 1, CONFIRM_FRAMES + 5)
        else:
            # New candidate gesture
            self.candidate_gesture = raw_gesture
            self.candidate_count = 1

        # Promote candidate to confirmed if held long enough
        if self.candidate_count >= CONFIRM_FRAMES:
            if raw_gesture != self.confirmed_gesture:
                # Count consecutive frames disagreeing with confirmed gesture
                if raw_gesture != self.confirmed_gesture:
                    self.release_count += 1
                if self.release_count >= RELEASE_FRAMES:
                    self.confirmed_gesture = raw_gesture
                    self.confirmed_hold_frames = self.candidate_count
                    self.release_count = 0
            else:
                self.release_count = 0
                self.confirmed_hold_frames += 1
        else:
            # Not yet confirmed — keep previous confirmed gesture (hysteresis)
            self.release_count = 0

        return self.confirmed_gesture


class GestureDetector:
    """Classifies MediaPipe hand landmark data into typed Gesture outputs with temporal filtering."""

    # Landmark Indices
    WRIST = 0
    THUMB_MCP = 2
    THUMB_TIP = 4
    INDEX_MCP = 5
    INDEX_PIP = 6
    INDEX_TIP = 8
    MIDDLE_MCP = 9
    MIDDLE_PIP = 10
    MIDDLE_TIP = 12
    RING_MCP = 13
    RING_PIP = 14
    RING_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_TIP = 20

    def __init__(self,
                 pinch_ratio: float = PINCH_THRESHOLD_PALM_RATIO,
                 finger_extend_ratio: float = FINGER_EXTEND_RATIO,
                 confirm_frames: int = CONFIRM_FRAMES):
        self.pinch_ratio = pinch_ratio
        self.finger_extend_ratio = finger_extend_ratio
        self.confirm_frames = confirm_frames
        # Per-hand temporal state tracked by label
        self._temporal: Dict[str, _HandTemporalState] = {}

    def _get_temporal(self, label: str) -> _HandTemporalState:
        if label not in self._temporal:
            self._temporal[label] = _HandTemporalState()
        return self._temporal[label]

    def classify_hand(self, hand_data: SingleHandData) -> HandGesture:
        """Classify gesture for a single hand based on landmarks with temporal hysteresis."""
        pts = hand_data.landmarks_px
        if len(pts) < 21:
            ts = self._get_temporal(hand_data.label)
            confirmed = ts.update(GestureType.NONE)
            return HandGesture(hand_label=hand_data.label, confidence=hand_data.score,
                               gesture=confirmed, raw_gesture=GestureType.NONE)

        wrist = pts[self.WRIST]
        thumb_tip = pts[self.THUMB_TIP]
        index_tip = pts[self.INDEX_TIP]
        middle_mcp = pts[self.MIDDLE_MCP]

        # Calculate scale reference: Palm Size = dist(WRIST, MIDDLE_MCP)
        palm_size = distance_2d(wrist, middle_mcp)
        if palm_size < 1.0:
            palm_size = 1.0

        # Pinch distance & midpoint
        pinch_dist = distance_2d(thumb_tip, index_tip)
        pinch_pt = midpoint_2d(thumb_tip, index_tip)
        is_pinching = (pinch_dist / palm_size) <= self.pinch_ratio

        # Strict per-finger extension checks (tip vs pip distance from wrist)
        def is_extended(tip_idx: int, pip_idx: int, ratio: float = self.finger_extend_ratio) -> bool:
            d_tip = distance_2d(wrist, pts[tip_idx])
            d_pip = distance_2d(wrist, pts[pip_idx])
            return d_tip > (d_pip * ratio)

        index_ext = is_extended(self.INDEX_TIP, self.INDEX_PIP, POINT_INDEX_RATIO)
        middle_ext = is_extended(self.MIDDLE_TIP, self.MIDDLE_PIP)
        ring_ext = is_extended(self.RING_TIP, self.RING_PIP)
        pinky_ext = is_extended(self.PINKY_TIP, self.PINKY_PIP)

        # Strict POINT: index clearly extended, ALL others clearly NOT extended
        # This is what prevents accidental drawing during transitions
        index_strict = is_extended(self.INDEX_TIP, self.INDEX_PIP, POINT_INDEX_RATIO)
        others_curled = (
            not is_extended(self.MIDDLE_TIP, self.MIDDLE_PIP, POINT_OTHER_RATIO) and
            not is_extended(self.RING_TIP, self.RING_PIP, POINT_OTHER_RATIO) and
            not is_extended(self.PINKY_TIP, self.PINKY_PIP, POINT_OTHER_RATIO)
        )
        is_strict_point = index_strict and others_curled and not is_pinching

        all_other_extended = middle_ext and ring_ext and pinky_ext
        all_curled = not index_ext and not middle_ext and not ring_ext and not pinky_ext

        # Raw gesture classification (priority order)
        if is_pinching:
            raw_gesture = GestureType.PINCH
        elif index_ext and all_other_extended:
            raw_gesture = GestureType.OPEN_PALM
        elif is_strict_point:
            raw_gesture = GestureType.POINT
        elif all_curled:
            raw_gesture = GestureType.FIST
        else:
            raw_gesture = GestureType.NONE

        # Apply temporal hysteresis
        ts = self._get_temporal(hand_data.label)
        confirmed_gesture = ts.update(raw_gesture)

        return HandGesture(
            hand_label=hand_data.label,
            gesture=confirmed_gesture,
            raw_gesture=raw_gesture,
            index_tip=index_tip,
            thumb_tip=thumb_tip,
            pinch_point=pinch_pt,
            palm_center=middle_mcp,
            confidence=hand_data.score,
            is_pinch=(confirmed_gesture == GestureType.PINCH),
            is_point=(confirmed_gesture == GestureType.POINT),
            is_open_palm=(confirmed_gesture == GestureType.OPEN_PALM),
            is_fist=(confirmed_gesture == GestureType.FIST),
            gesture_hold_frames=ts.confirmed_hold_frames
        )

    def detect(self, detected_hands: List[SingleHandData]) -> FrameGestures:
        """Classify gestures for all hands in the frame and compute multi-hand metrics."""
        # Remove temporal state for hands no longer visible
        active_labels = {h.label for h in detected_hands}
        for label in list(self._temporal.keys()):
            if label not in active_labels:
                del self._temporal[label]

        classified_hands: List[HandGesture] = []
        for hand in detected_hands:
            classified_hands.append(self.classify_hand(hand))

        inter_dist: Optional[float] = None
        rel_angle: Optional[float] = None

        if len(classified_hands) >= 2:
            h0 = classified_hands[0]
            h1 = classified_hands[1]
            p0 = h0.pinch_point if h0.is_pinch else h0.index_tip
            p1 = h1.pinch_point if h1.is_pinch else h1.index_tip

            inter_dist = distance_2d(p0, p1)
            rel_angle = calculate_angle_2d(p0, p1)

        return FrameGestures(
            hands=classified_hands,
            inter_hand_distance=inter_dist,
            relative_hand_angle=rel_angle
        )
