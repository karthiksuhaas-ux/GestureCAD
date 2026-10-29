"""Shape recognition module for AirDesk.

Classifies completed hand-drawn air strokes into geometric primitive shapes:
- LINE
- TRIANGLE
- RECTANGLE (or SQUARE)
- CIRCLE
- UNRECOGNIZED

Uses OpenCV contour approximation (approxPolyDP), vertex counting, circularity,
and bounding box geometry for tolerant classification of hand-drawn strokes.
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import math
import cv2
import numpy as np

from geometry import distance_2d


class ShapeType(Enum):
    UNRECOGNIZED = auto()
    LINE = auto()
    TRIANGLE = auto()
    RECTANGLE = auto()
    CIRCLE = auto()

    def __str__(self) -> str:
        return self.name


@dataclass
class RecognizedShape:
    """Dataclass holding shape recognition results."""
    shape_type: ShapeType = ShapeType.UNRECOGNIZED
    confidence: float = 0.0
    bounding_box: Tuple[int, int, int, int] = (0, 0, 0, 0)  # (x, y, w, h)
    center: Tuple[float, float] = (0.0, 0.0)
    label_position: Tuple[int, int] = (0, 0)
    vertices: List[Tuple[int, int]] = field(default_factory=list)


class ShapeRecognizer:
    """Recognizes 2D primitive shapes from hand-drawn stroke point trajectories."""

    def __init__(self, approx_epsilon_ratio: float = 0.03):
        self.approx_epsilon_ratio = approx_epsilon_ratio

    def recognize(self, points: List[Tuple[float, float]]) -> RecognizedShape:
        """Classify a polyline stroke into a RecognizedShape."""
        if not points or len(points) < 5:
            return RecognizedShape(shape_type=ShapeType.UNRECOGNIZED)

        # Convert points to NumPy contour format (N, 1, 2) int32
        pts_np = np.array(points, dtype=np.int32).reshape((-1, 1, 2))

        # Bounding box & Center
        x, y, w, h = cv2.boundingRect(pts_np)
        center = (x + w / 2.0, y + h / 2.0)
        label_pos = (x + w + 10, y + h // 2)

        # Calculate polyline perimeter & start-to-end distance
        perimeter = cv2.arcLength(pts_np, closed=False)
        if perimeter <= 1.0:
            return RecognizedShape(shape_type=ShapeType.UNRECOGNIZED, bounding_box=(x, y, w, h), center=center, label_position=label_pos)

        start_end_dist = distance_2d(points[0], points[-1])
        closure_ratio = start_end_dist / max(w, h, 1.0)
        straightness = start_end_dist / max(perimeter, 1.0)

        # 1. LINE Check: Open stroke where start & end points are far apart relative to bounding box, or very straight trajectory
        if closure_ratio > 0.40 or straightness > 0.65:
            if straightness > 0.70 or max(w, h) / (min(w, h) + 1.0) > 3.5:
                return RecognizedShape(
                    shape_type=ShapeType.LINE,
                    confidence=min(straightness, 1.0),
                    bounding_box=(x, y, w, h),
                    center=center,
                    label_position=label_pos,
                    vertices=[(int(points[0][0]), int(points[0][1])), (int(points[-1][0]), int(points[-1][1]))]
                )

        # For closed shape analysis, use convex hull
        hull = cv2.convexHull(pts_np)
        hull_area = cv2.contourArea(hull)
        hull_perimeter = cv2.arcLength(hull, closed=True)

        if hull_perimeter <= 1.0:
            return RecognizedShape(shape_type=ShapeType.UNRECOGNIZED, bounding_box=(x, y, w, h), center=center, label_position=label_pos)

        # Contour approximation with approxPolyDP
        epsilon = self.approx_epsilon_ratio * hull_perimeter
        approx = cv2.approxPolyDP(hull, epsilon=epsilon, closed=True)
        vertices_count = len(approx)
        approx_vertices = [(int(pt[0][0]), int(pt[0][1])) for pt in approx]

        # 2. TRIANGLE Check: approxPolyDP yields 3 vertices
        if vertices_count == 3 and closure_ratio < 0.45:
            return RecognizedShape(
                shape_type=ShapeType.TRIANGLE,
                confidence=0.88,
                bounding_box=(x, y, w, h),
                center=center,
                label_position=label_pos,
                vertices=approx_vertices
            )

        # 3. RECTANGLE / SQUARE Check: approxPolyDP yields 4 vertices
        if vertices_count == 4 and closure_ratio < 0.45:
            return RecognizedShape(
                shape_type=ShapeType.RECTANGLE,
                confidence=0.88,
                bounding_box=(x, y, w, h),
                center=center,
                label_position=label_pos,
                vertices=approx_vertices
            )

        # Calculate Compactness / Circularity = 4 * pi * Area / (Perimeter^2)
        circularity = (4.0 * math.pi * hull_area) / (hull_perimeter ** 2) if hull_perimeter > 0 else 0.0

        # Min Enclosing Circle fill ratio
        (cx, cy), radius = cv2.minEnclosingCircle(pts_np)
        circle_area = math.pi * (radius ** 2)
        circle_fill_ratio = (hull_area / circle_area) if circle_area > 0 else 0.0
        aspect_ratio = min(w, h) / max(w, h, 1.0)

        # 4. CIRCLE Check: High circularity or high circle fill ratio with reasonable aspect ratio & >= 5 vertices
        if (circularity > 0.55 or circle_fill_ratio > 0.60) and aspect_ratio > 0.50 and closure_ratio < 0.45:
            confidence = min(0.5 * circularity + 0.5 * circle_fill_ratio, 1.0)
            return RecognizedShape(
                shape_type=ShapeType.CIRCLE,
                confidence=confidence,
                bounding_box=(x, y, w, h),
                center=center,
                label_position=label_pos,
                vertices=approx_vertices
            )

        # 5. Fallback check with adaptive coarse epsilon ratio (e.g. 0.05)
        epsilon_coarse = 0.05 * hull_perimeter
        approx_coarse = cv2.approxPolyDP(hull, epsilon=epsilon_coarse, closed=True)
        coarse_count = len(approx_coarse)

        if coarse_count == 3 and closure_ratio < 0.50:
            return RecognizedShape(
                shape_type=ShapeType.TRIANGLE,
                confidence=0.75,
                bounding_box=(x, y, w, h),
                center=center,
                label_position=label_pos,
                vertices=[(int(pt[0][0]), int(pt[0][1])) for pt in approx_coarse]
            )
        elif coarse_count == 4 and closure_ratio < 0.50:
            return RecognizedShape(
                shape_type=ShapeType.RECTANGLE,
                confidence=0.75,
                bounding_box=(x, y, w, h),
                center=center,
                label_position=label_pos,
                vertices=[(int(pt[0][0]), int(pt[0][1])) for pt in approx_coarse]
            )

        # Unrecognized shape
        return RecognizedShape(
            shape_type=ShapeType.UNRECOGNIZED,
            confidence=0.3,
            bounding_box=(x, y, w, h),
            center=center,
            label_position=label_pos,
            vertices=approx_vertices
        )
