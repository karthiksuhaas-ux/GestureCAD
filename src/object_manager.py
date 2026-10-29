"""Object manager module for AirDesk workspace objects — Overhauled.

Key improvements:
- WorkspaceObject stores raw_points for freehand strokes (not just bounding-box geometry).
- contains_point() uses proper geometry hit-testing:
    - Freehand strokes: point-to-segment distance across all segments.
    - Shapes: geometric containment / bounding box + center distance.
- select_object() deselects all others first.
- Undo/Redo snapshot history preserved.
"""

import copy
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import math
import numpy as np

from shape_recognizer import ShapeType, RecognizedShape
from geometry import distance_2d


# Distinct vibrant color palette for vector workspace objects (BGR)
COLOR_PALETTE = [
    (0, 230, 255),    # Vibrant Yellow/Cyan
    (255, 105, 180),  # Hot Pink / Magenta
    (0, 255, 127),    # Spring Green
    (255, 140, 0),    # Dark Orange
    (186, 85, 211),   # Medium Orchid / Purple
    (0, 191, 255)     # Deep Sky Blue
]

# Hit-test tolerance for shapes that have no explicit polygon (e.g. raw strokes)
DEFAULT_HIT_TOLERANCE_PX = 20.0


def _point_to_segment_dist(p: Tuple[float, float],
                            a: Tuple[float, float],
                            b: Tuple[float, float]) -> float:
    """Minimum distance from point p to line segment a→b."""
    ax, ay = a
    bx, by = b
    px, py = p
    dx, dy = bx - ax, by - ay
    len_sq = dx * dx + dy * dy
    if len_sq < 1e-9:
        return distance_2d(p, a)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / len_sq))
    proj = (ax + t * dx, ay + t * dy)
    return distance_2d(p, proj)


@dataclass
class WorkspaceObject:
    """Dataclass representing an active 2D workspace object."""
    id: int
    shape_type: ShapeType
    position: Tuple[float, float]               # Center position (cx, cy) in frame pixels
    size: Tuple[float, float]                   # (width, height) or (radius, radius)
    rotation: float = 0.0                       # Rotation angle in degrees
    scale: float = 1.0                          # Scale factor
    color: Tuple[int, int, int] = (0, 255, 0)   # BGR color tuple
    selected: bool = False                      # Active selection flag
    velocity: Tuple[float, float] = (0.0, 0.0)  # (vx, vy) for physics
    local_vertices: List[Tuple[float, float]] = field(default_factory=list)  # Vertices relative to center
    raw_points: List[Tuple[float, float]] = field(default_factory=list)      # Original stroke points (freehand)

    def get_world_vertices(self) -> List[Tuple[float, float]]:
        """Calculate world-space vertex coordinates applying position, scale, and rotation."""
        if not self.local_vertices:
            return []
        rad = math.radians(self.rotation)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)
        cx, cy = self.position
        world_pts = []
        for lx, ly in self.local_vertices:
            sx = lx * self.scale
            sy = ly * self.scale
            rx = sx * cos_a - sy * sin_a
            ry = sx * sin_a + sy * cos_a
            world_pts.append((cx + rx, cy + ry))
        return world_pts

    def get_world_raw_points(self) -> List[Tuple[float, float]]:
        """
        Return raw stroke points transformed to current world position.
        Used for hit-testing freehand strokes after they've been moved.
        """
        if not self.raw_points:
            return []
        # Compute original center from raw_points
        orig_cx = sum(p[0] for p in self.raw_points) / len(self.raw_points)
        orig_cy = sum(p[1] for p in self.raw_points) / len(self.raw_points)
        cx, cy = self.position
        dx, dy = cx - orig_cx, cy - orig_cy
        return [(p[0] + dx, p[1] + dy) for p in self.raw_points]

    def contains_point(self, point: Tuple[float, float], tolerance_px: float = DEFAULT_HIT_TOLERANCE_PX) -> bool:
        """
        Check if a point is within tolerance distance of this object's geometry.

        For freehand strokes (UNRECOGNIZED): uses point-to-segment distance against
        the actual stroke path. This gives proper hit-testing on the visual ink.

        For shapes: uses bounding-box containment + center-distance fallback.
        """
        if self.shape_type == ShapeType.UNRECOGNIZED and self.raw_points:
            # Hit-test the actual stroke path
            pts = self.get_world_raw_points()
            if len(pts) < 2:
                return distance_2d(point, pts[0]) <= tolerance_px
            for i in range(len(pts) - 1):
                d = _point_to_segment_dist(point, pts[i], pts[i + 1])
                if d <= tolerance_px:
                    return True
            return False

        elif self.shape_type == ShapeType.LINE:
            world_pts = self.get_world_vertices()
            if len(world_pts) >= 2:
                return _point_to_segment_dist(point, world_pts[0], world_pts[1]) <= tolerance_px
            # Fallback
            return distance_2d(self.position, point) <= tolerance_px + max(self.size) / 2.0

        elif self.shape_type == ShapeType.CIRCLE:
            radius = (self.size[0] / 2.0) * self.scale
            d = distance_2d(point, self.position)
            # Hit the ring (not just the center) - within tolerance of the circle edge or inside
            return d <= radius + tolerance_px

        else:
            # For rectangles, triangles, and other polygons: use world vertices bounding box
            world_pts = self.get_world_vertices()
            if world_pts:
                all_x = [p[0] for p in world_pts]
                all_y = [p[1] for p in world_pts]
                min_x = min(all_x) - tolerance_px
                max_x = max(all_x) + tolerance_px
                min_y = min(all_y) - tolerance_px
                max_y = max(all_y) + tolerance_px
                px, py = point
                return min_x <= px <= max_x and min_y <= py <= max_y
            # Fallback to center distance
            cx, cy = self.position
            hw = (self.size[0] / 2.0) * self.scale + tolerance_px
            hh = (self.size[1] / 2.0) * self.scale + tolerance_px
            px, py = point
            return abs(px - cx) <= hw and abs(py - cy) <= hh


class ObjectManager:
    """Manages creation, storage, spatial hit-testing, deletion, and Undo/Redo history."""

    def __init__(self):
        self.objects: List[WorkspaceObject] = []
        self.undo_stack: List[List[WorkspaceObject]] = []
        self.redo_stack: List[List[WorkspaceObject]] = []
        self._next_id: int = 1
        self._color_index: int = 0

    def _get_next_color(self) -> Tuple[int, int, int]:
        color = COLOR_PALETTE[self._color_index % len(COLOR_PALETTE)]
        self._color_index += 1
        return color

    def save_snapshot(self):
        """Save current workspace objects state to Undo stack and clear Redo stack."""
        snapshot = copy.deepcopy(self.objects)
        self.undo_stack.append(snapshot)
        self.redo_stack.clear()
        # Limit undo stack to 50 snapshots
        if len(self.undo_stack) > 50:
            self.undo_stack.pop(0)

    def undo(self) -> bool:
        """Undo last workspace object change."""
        if not self.undo_stack:
            return False
        self.redo_stack.append(copy.deepcopy(self.objects))
        self.objects = self.undo_stack.pop()
        return True

    def redo(self) -> bool:
        """Redo previously undone workspace object change."""
        if not self.redo_stack:
            return False
        self.undo_stack.append(copy.deepcopy(self.objects))
        self.objects = self.redo_stack.pop()
        return True

    def create_primitive_object(self, shape_type: ShapeType, center: Tuple[float, float] = (640.0, 360.0),
                                 size: Tuple[float, float] = (120.0, 120.0)) -> 'WorkspaceObject':
        """Create standard primitive object (e.g. from Toolbar button)."""
        self.save_snapshot()
        obj_id = self._next_id
        self._next_id += 1
        color = self._get_next_color()

        local_verts: List[Tuple[float, float]] = []
        w, h = size

        if shape_type == ShapeType.RECTANGLE:
            hw, hh = w / 2.0, h / 2.0
            local_verts = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
        elif shape_type == ShapeType.TRIANGLE:
            hw, hh = w / 2.0, h / 2.0
            local_verts = [(0.0, -hh), (hw, hh), (-hw, hh)]
        elif shape_type == ShapeType.CIRCLE:
            radius = max(w, h) / 2.0
            for i in range(32):
                angle = (i / 32.0) * 2.0 * math.pi
                local_verts.append((radius * math.cos(angle), radius * math.sin(angle)))
        else:  # LINE
            hw = w / 2.0
            local_verts = [(-hw, 0.0), (hw, 0.0)]

        obj = WorkspaceObject(
            id=obj_id, shape_type=shape_type, position=center, size=size,
            rotation=0.0, scale=1.0, color=color, selected=False,
            velocity=(0.0, 0.0), local_vertices=local_verts, raw_points=[]
        )
        self.objects.append(obj)
        return obj

    def create_object_from_shape(self, rec_shape: RecognizedShape,
                                  raw_points: List[Tuple[float, float]]) -> 'WorkspaceObject':
        """Convert a recognized shape into a persistent vector WorkspaceObject."""
        self.save_snapshot()
        obj_id = self._next_id
        self._next_id += 1

        cx, cy = rec_shape.center
        x, y, w, h = rec_shape.bounding_box
        color = self._get_next_color()
        shape_type = rec_shape.shape_type

        local_verts: List[Tuple[float, float]] = []

        if shape_type == ShapeType.RECTANGLE:
            half_w, half_h = w / 2.0, h / 2.0
            local_verts = [(-half_w, -half_h), (half_w, -half_h), (half_w, half_h), (-half_w, half_h)]
            size = (float(w), float(h))
        elif shape_type == ShapeType.TRIANGLE:
            half_w, half_h = w / 2.0, h / 2.0
            local_verts = [(0.0, -half_h), (half_w, half_h), (-half_w, half_h)]
            size = (float(w), float(h))
        elif shape_type == ShapeType.CIRCLE:
            radius = max(w, h) / 2.0
            size = (radius * 2.0, radius * 2.0)
            for i in range(32):
                angle = (i / 32.0) * 2.0 * math.pi
                local_verts.append((radius * math.cos(angle), radius * math.sin(angle)))
        elif shape_type == ShapeType.LINE:
            size = (float(w), float(h))
            if len(rec_shape.vertices) >= 2:
                p1, p2 = rec_shape.vertices[0], rec_shape.vertices[1]
                local_verts = [(p1[0] - cx, p1[1] - cy), (p2[0] - cx, p2[1] - cy)]
            else:
                local_verts = [(-w / 2.0, 0.0), (w / 2.0, 0.0)]
        else:  # UNRECOGNIZED — store raw points as local vertices
            size = (float(w), float(h))
            local_verts = [(px - cx, py - cy) for px, py in raw_points]

        obj = WorkspaceObject(
            id=obj_id, shape_type=shape_type, position=(cx, cy), size=size,
            rotation=0.0, scale=1.0, color=color, selected=False,
            velocity=(0.0, 0.0), local_vertices=local_verts,
            raw_points=list(raw_points)  # Preserve original stroke for proper hit-testing
        )
        self.objects.append(obj)
        return obj

    def delete_selected(self) -> Optional['WorkspaceObject']:
        """Delete currently selected object."""
        for i, obj in enumerate(self.objects):
            if obj.selected:
                self.save_snapshot()
                return self.objects.pop(i)
        return None

    def hit_test(self, point: Tuple[float, float],
                 tolerance_px: float = DEFAULT_HIT_TOLERANCE_PX) -> Optional['WorkspaceObject']:
        """
        Find the topmost object that geometrically contains the given point.

        Iterates in reverse (last drawn = topmost) so recent objects take priority.
        """
        for obj in reversed(self.objects):
            if obj.contains_point(point, tolerance_px=tolerance_px):
                return obj
        return None

    def remove_object(self, obj_id: int) -> bool:
        """Remove object by ID."""
        for i, obj in enumerate(self.objects):
            if obj.id == obj_id:
                self.save_snapshot()
                self.objects.pop(i)
                return True
        return False

    def clear_all(self):
        """Remove all objects from workspace."""
        if self.objects:
            self.save_snapshot()
            self.objects.clear()

    def get_object_by_id(self, obj_id: int) -> Optional['WorkspaceObject']:
        """Find object by ID."""
        for obj in self.objects:
            if obj.id == obj_id:
                return obj
        return None

    def deselect_all(self):
        """Deselect all active objects."""
        for obj in self.objects:
            obj.selected = False

    def select_object(self, obj_id: int) -> Optional['WorkspaceObject']:
        """Deselect all and select the specified object."""
        self.deselect_all()
        target = self.get_object_by_id(obj_id)
        if target:
            target.selected = True
        return target
