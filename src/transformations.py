"""Transformation logic module for AirDesk.

Provides 2D and 3D spatial transformation algorithms for translating (moving),
rotating (X/Y/Z), and scaling workspace objects based on hand gesture inputs.
Includes deadzone filtering and optional exponential smoothing for rock-solid stability.
"""

from typing import Tuple
from object_manager import WorkspaceObject
from renderer_3d import WorkspaceObject3D
from geometry import calculate_angle_2d, distance_2d, smooth_point


class Transformations:
    """Applies 2D and 3D spatial transformations to workspace objects."""

    # --- 2D Transformations ---

    @staticmethod
    def translate(obj: WorkspaceObject, pinch_point: Tuple[float, float], drag_offset: Tuple[float, float], alpha: float = 1.0) -> Tuple[float, float]:
        """Update 2D object position following pinch point while maintaining initial grab offset."""
        import math as _math
        # Guard against NaN input from tracking instability
        if not (_math.isfinite(pinch_point[0]) and _math.isfinite(pinch_point[1])):
            return obj.position

        target_x = pinch_point[0] + drag_offset[0]
        target_y = pinch_point[1] + drag_offset[1]
        
        if alpha < 1.0:
            new_pos = smooth_point((target_x, target_y), obj.position, alpha=alpha)
        else:
            new_pos = (target_x, target_y)

        obj.position = new_pos
        return obj.position


    @staticmethod
    def rotate(obj: WorkspaceObject, current_hand_angle: float, initial_hand_angle: float, initial_obj_rotation: float, deadzone_deg: float = 0.0, alpha: float = 1.0) -> float:
        """Update 2D object rotation angle with optional deadzone filtering and smoothing."""
        angle_delta = current_hand_angle - initial_hand_angle

        if abs(angle_delta) < deadzone_deg:
            angle_delta = 0.0

        target_rotation = (initial_obj_rotation + angle_delta) % 360.0
        
        if alpha < 1.0:
            rot_diff = (target_rotation - obj.rotation + 180.0) % 360.0 - 180.0
            obj.rotation = (obj.rotation + alpha * rot_diff) % 360.0
        else:
            obj.rotation = target_rotation

        return obj.rotation

    @staticmethod
    def scale(obj: WorkspaceObject, current_inter_dist: float, initial_inter_dist: float, initial_obj_scale: float, alpha: float = 1.0) -> float:
        """Update 2D object scale factor based on 2-hand inter-hand distance ratio."""
        if initial_inter_dist < 1e-3:
            return obj.scale
        
        ratio = current_inter_dist / initial_inter_dist
        target_scale = max(0.15, min(10.0, initial_obj_scale * ratio))
        
        if alpha < 1.0:
            obj.scale = (1.0 - alpha) * obj.scale + alpha * target_scale
        else:
            obj.scale = target_scale

        return obj.scale

    # --- 3D Transformations ---

    @staticmethod
    def translate_3d(obj_3d: WorkspaceObject3D, pinch_pt_2d: Tuple[float, float], palm_size_px: float, initial_palm_size: float, initial_z: float, alpha: float = 1.0) -> Tuple[float, float, float]:
        """Update 3D position (X, Y, Z depth) using 2D screen motion and palm size depth scaling."""
        target_x = pinch_pt_2d[0]
        target_y = pinch_pt_2d[1]

        if initial_palm_size > 1.0:
            depth_delta = (initial_palm_size - palm_size_px) * 4.0
            target_z = max(200.0, min(1500.0, initial_z + depth_delta))
        else:
            target_z = obj_3d.position_3d[2]

        if alpha < 1.0:
            new_x = (1.0 - alpha) * obj_3d.position_3d[0] + alpha * target_x
            new_y = (1.0 - alpha) * obj_3d.position_3d[1] + alpha * target_y
            new_z = (1.0 - alpha) * obj_3d.position_3d[2] + alpha * target_z
        else:
            new_x, new_y, new_z = target_x, target_y, target_z

        obj_3d.position_3d = (new_x, new_y, new_z)
        return obj_3d.position_3d

    @staticmethod
    def rotate_3d(obj_3d: WorkspaceObject3D, current_hand_angle: float, initial_hand_angle: float, initial_rotation_3d: Tuple[float, float, float], alpha: float = 1.0) -> Tuple[float, float, float]:
        """Update 3D rotation angles (rot_X, rot_Y, rot_Z) using hand orientation vector."""
        angle_delta = current_hand_angle - initial_hand_angle
        rx, ry, rz = initial_rotation_3d
        target_rz = (rz + angle_delta) % 360.0

        if alpha < 1.0:
            new_rz = (obj_3d.rotation_3d[2] + alpha * ((target_rz - obj_3d.rotation_3d[2] + 180.0) % 360.0 - 180.0)) % 360.0
        else:
            new_rz = target_rz

        obj_3d.rotation_3d = (rx, ry, new_rz)
        return obj_3d.rotation_3d

    @staticmethod
    def scale_3d(obj_3d: WorkspaceObject3D, current_inter_dist: float, initial_inter_dist: float, initial_obj_scale: float, alpha: float = 1.0) -> float:
        """Update 3D object scale factor based on 2-hand distance ratio."""
        if initial_inter_dist < 1e-3:
            return obj_3d.scale_3d

        ratio = current_inter_dist / initial_inter_dist
        target_scale = max(0.2, min(8.0, initial_obj_scale * ratio))

        if alpha < 1.0:
            obj_3d.scale_3d = (1.0 - alpha) * obj_3d.scale_3d + alpha * target_scale
        else:
            obj_3d.scale_3d = target_scale

        return obj_3d.scale_3d
