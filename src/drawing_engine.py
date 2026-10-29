"""Drawing engine module for AirDesk — Overhauled.

Key design changes:
- Drawing ONLY starts when the user explicitly activates DRAW MODE (via toolbar/key).
- A new stroke begins only on the RISING EDGE of the POINT gesture (not continuously
  while pointing). This prevents connecting lines between separate strokes.
- When the POINT gesture is released or hand tracking is lost, the stroke is finalized
  immediately and safely — no connecting line is created.
- Each completed stroke is a self-contained object with its own raw points preserved.
- Freehand strokes are stored as polylines, not converted to shapes.
- Dead zone: minimum movement threshold before recording next point.
"""

from typing import List, Tuple, Optional
import cv2
import numpy as np

from gesture_detector import FrameGestures, HandGesture, GestureType
from geometry import distance_2d, smooth_point
from shape_recognizer import ShapeRecognizer, RecognizedShape, ShapeType

# Constants for drawing engine
SMOOTHING_ALPHA = 0.35        # EMA smoothing factor (lower = smoother, higher = more responsive)
MIN_POINT_DIST_PX = 6.0       # Dead zone: minimum pixels moved to record another point
MIN_STROKE_POINTS = 5         # Minimum points for a stroke to be considered valid
MIN_STROKE_EXTENT_PX = 15.0   # Minimum bounding box size in any dimension (noise filter)
ACTIVE_STROKE_COLOR = (0, 255, 80)      # Bright Green (drawing in progress)
COMPLETED_STROKE_COLOR = (255, 200, 50) # Warm Gold (completed freehand stroke)
STROKE_THICKNESS = 3


class Stroke:
    """A single captured stroke — its raw points, smoothed path, and recognized shape."""

    def __init__(self, color: Tuple[int, int, int] = COMPLETED_STROKE_COLOR):
        self.points: List[Tuple[float, float]] = []
        self.color: Tuple[int, int, int] = color
        self.recognized_shape: Optional[RecognizedShape] = None

    def add_point(self, pt: Tuple[float, float]):
        self.points.append(pt)

    def is_valid(self) -> bool:
        """Return True if stroke has enough points and spatial extent to be meaningful."""
        if len(self.points) < MIN_STROKE_POINTS:
            return False
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        extent = max(max(xs) - min(xs), max(ys) - min(ys))
        return extent >= MIN_STROKE_EXTENT_PX


class DrawingEngine:
    """
    Manages air-drawing trajectory capture, EMA smoothing, and shape recognition.

    Drawing lifecycle:
        IDLE → (gesture becomes POINT and mode=DRAW) → DRAWING (rising edge only)
        DRAWING → add smoothed points each frame
        DRAWING → (gesture leaves POINT or hand lost) → finalize stroke → IDLE

    Critical invariant: two separate POINT activations are NEVER connected.
    """

    def __init__(self, smoothing_alpha: float = SMOOTHING_ALPHA, min_dist: float = MIN_POINT_DIST_PX):
        self.smoothing_alpha = smoothing_alpha
        self.min_dist = min_dist
        self.shape_recognizer = ShapeRecognizer()

        # Active stroke state
        self.current_stroke: Optional[Stroke] = None
        self.last_smoothed_pt: Optional[Tuple[float, float]] = None
        self._is_drawing: bool = False               # True while actively capturing a stroke
        self._was_pointing_last_frame: bool = False  # For rising-edge detection

        # Completed strokes (raw freehand polylines, not converted to objects)
        self.completed_strokes: List[Stroke] = []

    @property
    def is_drawing(self) -> bool:
        return self._is_drawing

    def update(self, frame_gestures: FrameGestures, mode_is_draw: bool = True) -> Optional[Stroke]:
        """
        Update drawing state based on detected gestures and active draw mode.

        Returns a newly completed Stroke when one is finalized, otherwise None.

        Args:
            frame_gestures: Per-frame gesture data from GestureDetector.
            mode_is_draw: Only capture strokes when True.
        """
        # Find the active pointing hand (only if drawing mode is active)
        pointing_hand: Optional[HandGesture] = None
        if mode_is_draw:
            for hand in frame_gestures.hands:
                if hand.is_point:
                    pointing_hand = hand
                    break

        newly_completed: Optional[Stroke] = None
        currently_pointing = pointing_hand is not None

        if currently_pointing:
            raw_pt = pointing_hand.index_tip

            if not self._is_drawing:
                # === RISING EDGE: Start a new stroke ===
                # Only on the first frame where POINT is detected.
                # This is the key fix: we do NOT create a stroke from any previous
                # position — we start fresh from exactly where the user is now.
                self._is_drawing = True
                self.last_smoothed_pt = raw_pt
                self.current_stroke = Stroke(color=ACTIVE_STROKE_COLOR)
                self.current_stroke.add_point(raw_pt)
            else:
                # === DRAWING: Add smoothed points to active stroke ===
                if self.last_smoothed_pt is not None:
                    smoothed_pt = smooth_point(raw_pt, self.last_smoothed_pt, alpha=self.smoothing_alpha)
                else:
                    smoothed_pt = raw_pt

                # Dead zone: only record if hand moved enough
                last_pt = self.current_stroke.points[-1]
                if distance_2d(last_pt, smoothed_pt) >= self.min_dist:
                    self.current_stroke.add_point(smoothed_pt)
                    self.last_smoothed_pt = smoothed_pt

        else:
            # === FALLING EDGE or TRACKING LOST: Finalize stroke ===
            if self._is_drawing:
                self._is_drawing = False
                if self.current_stroke is not None and self.current_stroke.is_valid():
                    self.current_stroke.color = COMPLETED_STROKE_COLOR
                    # Run shape recognizer on the completed stroke
                    rec_shape = self.shape_recognizer.recognize(self.current_stroke.points)
                    self.current_stroke.recognized_shape = rec_shape
                    self.completed_strokes.append(self.current_stroke)
                    newly_completed = self.current_stroke

                # Always reset — never carry over any state to next stroke
                self.current_stroke = None
                self.last_smoothed_pt = None

        self._was_pointing_last_frame = currently_pointing
        return newly_completed

    def cancel_active_stroke(self):
        """Cancel the current stroke without saving it (e.g. on mode switch or tool change)."""
        self._is_drawing = False
        self.current_stroke = None
        self.last_smoothed_pt = None
        self._was_pointing_last_frame = False

    def render_strokes(self, frame: np.ndarray) -> np.ndarray:
        """Render completed stroke polylines with shape labels, and the active stroke."""
        output = frame

        SHAPE_LABEL_COLORS = {
            ShapeType.RECTANGLE: (0, 255, 255),
            ShapeType.TRIANGLE: (255, 0, 255),
            ShapeType.CIRCLE: (0, 255, 0),
            ShapeType.LINE: (255, 150, 0),
            ShapeType.UNRECOGNIZED: (180, 180, 180)
        }

        # Draw completed strokes
        for stroke in self.completed_strokes:
            if len(stroke.points) > 1:
                pts_array = np.array(stroke.points, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(output, [pts_array], isClosed=False, color=stroke.color,
                              thickness=STROKE_THICKNESS, lineType=cv2.LINE_AA)

                if stroke.recognized_shape is not None:
                    rec = stroke.recognized_shape
                    label = f"[{rec.shape_type.name}]"
                    lx, ly = rec.label_position
                    color = SHAPE_LABEL_COLORS.get(rec.shape_type, (255, 255, 255))
                    cv2.putText(output, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 0, 0), 3, cv2.LINE_AA)
                    cv2.putText(output, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, color, 1, cv2.LINE_AA)

        # Draw active stroke in progress
        if self.current_stroke is not None and len(self.current_stroke.points) > 1:
            pts_array = np.array(self.current_stroke.points, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(output, [pts_array], isClosed=False, color=ACTIVE_STROKE_COLOR,
                          thickness=STROKE_THICKNESS + 1, lineType=cv2.LINE_AA)
            # Draw tip indicator
            tip = (int(self.current_stroke.points[-1][0]), int(self.current_stroke.points[-1][1]))
            cv2.circle(output, tip, 6, (0, 255, 80), -1, cv2.LINE_AA)
            cv2.circle(output, tip, 10, (0, 255, 80), 2, cv2.LINE_AA)

        return output

    def clear_all(self):
        """Clear all strokes (active and completed)."""
        self.cancel_active_stroke()
        self.completed_strokes.clear()

    def undo_last(self) -> Optional[Stroke]:
        """Remove the most recently completed stroke."""
        if self.completed_strokes:
            return self.completed_strokes.pop()
        return None
