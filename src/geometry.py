"""Geometry helper functions.

Provides vector math, distance calculations, smoothing, and landmark utility functions
used across AirDesk modules.
"""

from typing import Tuple, List, Optional
import math
import numpy as np


def distance_2d(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Calculate Euclidean distance between two 2D points (x, y)."""
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


def calculate_angle_2d(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Calculate angle in degrees from p1 to p2 relative to positive X axis."""
    return math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))


def midpoint_2d(p1: Tuple[float, float], p2: Tuple[float, float]) -> Tuple[float, float]:
    """Calculate 2D midpoint between two points."""
    return ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)


def smooth_point(new_point: Tuple[float, float], prev_point: Tuple[float, float], alpha: float = 0.5) -> Tuple[float, float]:
    """Apply exponential moving average smoothing between previous and new point."""
    x = alpha * new_point[0] + (1.0 - alpha) * prev_point[0]
    y = alpha * new_point[1] + (1.0 - alpha) * prev_point[1]
    return (x, y)

