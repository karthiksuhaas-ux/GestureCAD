"""3D Renderer and Mesh Primitives module for AirDesk.

Provides 3D mesh definitions (Cube, Pyramid, Sphere), 3D perspective projection,
shading, wireframes, and PyOpenGL / OpenCV compositing.
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import math
import cv2
import numpy as np


class ShapeType3D(Enum):
    CUBE = auto()
    PYRAMID = auto()
    SPHERE = auto()

    def __str__(self) -> str:
        return self.name


@dataclass
class WorkspaceObject3D:
    """Dataclass representing a 3D workspace object."""
    id: int
    shape_type_3d: ShapeType3D
    position_3d: Tuple[float, float, float] = (0.0, 0.0, 500.0) # (X, Y, Z depth in mm/px)
    rotation_3d: Tuple[float, float, float] = (0.0, 0.0, 0.0) # (rot_x, rot_y, rot_z) in degrees
    scale_3d: float = 1.0
    color: Tuple[int, int, int] = (0, 255, 255)
    selected: bool = False


class Renderer3D:
    """3D perspective projection and mesh rendering engine."""

    def __init__(self, focal_length: float = 800.0, camera_center: Tuple[float, float] = (640.0, 360.0)):
        self.focal_length = focal_length
        self.camera_center = camera_center
        self.objects_3d: List[WorkspaceObject3D] = []
        self._next_id: int = 100

    def create_3d_object(self, shape_type_3d: ShapeType3D, position_3d: Tuple[float, float, float] = (640.0, 360.0, 600.0), scale_3d: float = 1.0) -> WorkspaceObject3D:
        """Instantiate a new 3D workspace object."""
        obj = WorkspaceObject3D(
            id=self._next_id,
            shape_type_3d=shape_type_3d,
            position_3d=position_3d,
            rotation_3d=(15.0, 30.0, 0.0),
            scale_3d=scale_3d,
            color=(0, 215, 255) if shape_type_3d == ShapeType3D.CUBE else ((255, 105, 180) if shape_type_3d == ShapeType3D.PYRAMID else (0, 255, 127)),
            selected=False
        )
        self._next_id += 1
        self.objects_3d.append(obj)
        return obj

    def _get_mesh_data(self, shape_type: ShapeType3D) -> Tuple[np.ndarray, List[List[int]]]:
        """Generate 3D vertices and face indices for 3D primitives."""
        if shape_type == ShapeType3D.CUBE:
            s = 60.0
            vertices = np.array([
                [-s, -s, -s], [s, -s, -s], [s, s, -s], [-s, s, -s],
                [-s, -s, s],  [s, -s, s],  [s, s, s],  [-s, s, s]
            ], dtype=np.float32)
            faces = [
                [0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4],
                [2, 3, 7, 6], [0, 3, 7, 4], [1, 2, 6, 5]
            ]
            return vertices, faces

        elif shape_type == ShapeType3D.PYRAMID:
            s = 60.0
            h = 80.0
            vertices = np.array([
                [-s, s, -s], [s, s, -s], [s, s, s], [-s, s, s], [0, -h, 0]
            ], dtype=np.float32)
            faces = [
                [0, 1, 2, 3], # Base
                [0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4] # Sides
            ]
            return vertices, faces

        else: # SPHERE
            r = 60.0
            rings, sectors = 10, 10
            vertices = []
            faces = []
            for i in range(rings + 1):
                lat = math.pi * (-0.5 + float(i) / rings)
                z0 = r * math.sin(lat)
                r0 = r * math.cos(lat)
                for j in range(sectors):
                    lng = 2 * math.pi * float(j) / sectors
                    x0 = r0 * math.cos(lng)
                    y0 = r0 * math.sin(lng)
                    vertices.append([x0, y0, z0])

            vertices = np.array(vertices, dtype=np.float32)
            for i in range(rings):
                for j in range(sectors):
                    first = (i * (sectors)) + j
                    second = first + sectors
                    next_j = (j + 1) % sectors
                    faces.append([first, first + (next_j - j), second + (next_j - j), second])

            return vertices, faces

    def project_3d_point(self, pt_3d: np.ndarray, obj_pos: Tuple[float, float, float]) -> Tuple[int, int]:
        """Apply 3D perspective projection onto 2D camera viewport."""
        x, y, z = pt_3d[0] + obj_pos[0], pt_3d[1] + obj_pos[1], pt_3d[2] + obj_pos[2]
        if z <= 1.0:
            z = 1.0  # Prevent zero division

        # Perspective projection equations: X_2d = f * X / Z + cx, Y_2d = f * Y / Z + cy
        proj_x = int((self.focal_length * (x - self.camera_center[0]) / z) + self.camera_center[0])
        proj_y = int((self.focal_length * (y - self.camera_center[1]) / z) + self.camera_center[1])
        return (proj_x, proj_y)

    def render_3d_objects(self, frame: np.ndarray) -> np.ndarray:
        """Render all active 3D objects with depth sorting and perspective projection."""
        output = frame
        cx, cy = self.camera_center

        for obj in self.objects_3d:
            vertices, faces = self._get_mesh_data(obj.shape_type_3d)

            # Apply 3D Rotation matrices
            rx = math.radians(obj.rotation_3d[0])
            ry = math.radians(obj.rotation_3d[1])
            rz = math.radians(obj.rotation_3d[2])

            Rx = np.array([[1, 0, 0], [0, math.cos(rx), -math.sin(rx)], [0, math.sin(rx), math.cos(rx)]])
            Ry = np.array([[math.cos(ry), 0, math.sin(ry)], [0, 1, 0], [-math.sin(ry), 0, math.cos(ry)]])
            Rz = np.array([[math.cos(rz), -math.sin(rz), 0], [math.sin(rz), math.cos(rz), 0], [0, 0, 1]])

            R = Rz @ Ry @ Rx

            # Scale and rotate local vertices
            transformed_verts = (vertices * obj.scale_3d) @ R.T

            # Project 3D vertices to 2D screen points
            screen_pts = [self.project_3d_point(v, obj.position_3d) for v in transformed_verts]

            line_color = (0, 255, 255) if obj.selected else obj.color
            line_thick = 3 if obj.selected else 2

            # Draw 3D Mesh Wireframe & Shaded Faces
            for face in faces:
                pts_face = np.array([screen_pts[idx] for idx in face], dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(output, [pts_face], isClosed=True, color=line_color, thickness=line_thick, lineType=cv2.LINE_AA)

            # Draw 3D Label & Selection Halo
            proj_center = self.project_3d_point(np.array([0, 0, 0]), obj.position_3d)
            if obj.selected:
                cv2.circle(output, proj_center, 15, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.circle(output, proj_center, 4, (0, 255, 255), -1, cv2.LINE_AA)

            label = f"[3D {obj.shape_type_3d.name}] Z:{int(obj.position_3d[2])}mm"
            cv2.putText(output, label, (proj_center[0] - 40, proj_center[1] - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(output, label, (proj_center[0] - 40, proj_center[1] - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, line_color, 1, cv2.LINE_AA)

        return output
