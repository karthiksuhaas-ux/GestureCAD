"""Renderer module for AirDesk.

Composites webcam video feed, hand tracking overlays, detected gestures,
multi-hand metrics, 2D vector workspace objects with hover pre-highlighting,
CAD-style rotation & scale handles, state info, draggable widgets, and HUD status bars.
"""

from typing import List, Optional, Tuple
import time
import math
import cv2
import numpy as np

from hand_tracker import SingleHandData
from gesture_detector import FrameGestures, GestureType
from object_manager import WorkspaceObject
from shape_recognizer import ShapeType


class Renderer:
    """Renders composite frame, vector objects, CAD handles, and demo-ready HUD."""

    def __init__(self, window_name: str = "AirDesk — Workspace"):
        self.window_name = window_name
        self.fps = 0.0
        self.last_frame_time = time.time()

    def update_fps(self) -> float:
        """Calculate frame rate."""
        current_time = time.time()
        dt = current_time - self.last_frame_time
        self.last_frame_time = current_time
        if dt > 0:
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt) if self.fps > 0 else 1.0 / dt
        return self.fps

    def render_objects(self, frame: np.ndarray, objects: List[WorkspaceObject], hovered_obj: Optional[WorkspaceObject] = None) -> np.ndarray:
        """Render vector workspace objects with hover pre-highlighting and CAD selection handles."""
        output = frame

        for obj in objects:
            world_pts = obj.get_world_vertices()
            cx, cy = int(obj.position[0]), int(obj.position[1])
            is_hovered = (hovered_obj is not None and hovered_obj.id == obj.id)

            if obj.selected:
                line_color = (0, 255, 255) # Bright Yellow
                line_thick = 4
            elif is_hovered:
                line_color = (0, 230, 255) # Hover Cyan-Yellow Glow
                line_thick = 3
            else:
                line_color = obj.color
                line_thick = 3

            # Draw Base Geometry Shape
            if obj.shape_type == ShapeType.CIRCLE:
                radius = int((obj.size[0] / 2.0) * obj.scale)
                cv2.circle(output, (cx, cy), radius, line_color, line_thick, cv2.LINE_AA)
                
                rad = math.radians(obj.rotation)
                rx = int(cx + radius * math.cos(rad))
                ry = int(cy + radius * math.sin(rad))
                cv2.line(output, (cx, cy), (rx, ry), line_color, 2, cv2.LINE_AA)

            elif obj.shape_type == ShapeType.LINE and len(world_pts) >= 2:
                p1 = (int(world_pts[0][0]), int(world_pts[0][1]))
                p2 = (int(world_pts[1][0]), int(world_pts[1][1]))
                cv2.line(output, p1, p2, line_color, line_thick, cv2.LINE_AA)
                cv2.circle(output, p1, 5, line_color, -1, cv2.LINE_AA)
                cv2.circle(output, p2, 5, line_color, -1, cv2.LINE_AA)

            elif len(world_pts) > 2:
                pts_array = np.array(world_pts, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(output, [pts_array], isClosed=True, color=line_color,
                              thickness=line_thick, lineType=cv2.LINE_AA)

            # Draw Center Node
            cv2.circle(output, (cx, cy), 4, line_color, -1, cv2.LINE_AA)

            # Draw Pre-Highlight Hover Ring if hovered (gives user instant feedback BEFORE pinching!)
            if is_hovered and not obj.selected:
                all_x = [pt[0] for pt in world_pts] if world_pts else [cx]
                all_y = [pt[1] for pt in world_pts] if world_pts else [cy]
                min_x, max_x = int(min(all_x)) - 10, int(max(all_x)) + 10
                min_y, max_y = int(min(all_y)) - 10, int(max(all_y)) + 10
                cv2.rectangle(output, (min_x, min_y), (max_x, max_y), (0, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(output, "PINCH TO SELECT", (min_x, min_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

            # Draw Full CAD Handles & Bounding Box if Selected
            if obj.selected:
                all_x = [pt[0] for pt in world_pts] if world_pts else [cx]
                all_y = [pt[1] for pt in world_pts] if world_pts else [cy]
                min_x, max_x = int(min(all_x)) - 12, int(max(all_x)) + 12
                min_y, max_y = int(min(all_y)) - 12, int(max(all_y)) + 12

                # Selection Bounding Rectangle
                cv2.rectangle(output, (min_x, min_y), (max_x, max_y), (0, 255, 255), 1, cv2.LINE_AA)
                
                # Corner Handles
                for handle in [(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)]:
                    cv2.rectangle(output, (handle[0]-4, handle[1]-4), (handle[0]+4, handle[1]+4), (0, 255, 255), -1)

                # CAD Rotation Stem & Knob (Top Center Handle)
                rot_knob_x = cx
                rot_knob_y = min_y - 28
                cv2.line(output, (cx, min_y), (rot_knob_x, rot_knob_y), (0, 255, 0), 2, cv2.LINE_AA)
                cv2.circle(output, (rot_knob_x, rot_knob_y), 8, (0, 255, 0), -1, cv2.LINE_AA)
                cv2.circle(output, (rot_knob_x, rot_knob_y), 10, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(output, "ROTATE", (rot_knob_x + 14, rot_knob_y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)

                # Selection Status Label
                obj_label = f"Selected: Obj #{obj.id} ({obj.shape_type.name})"
                cv2.putText(output, obj_label, (min_x, min_y - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(output, obj_label, (min_x, min_y - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA)

        return output

    def render_gestures(self, frame: np.ndarray, frame_gestures: FrameGestures) -> np.ndarray:
        """Render gesture state tags, cursor targets, and multi-hand connecting line."""
        output = frame

        GESTURE_COLORS = {
            GestureType.POINT: (0, 255, 0),
            GestureType.PINCH: (0, 255, 255),
            GestureType.OPEN_PALM: (255, 100, 100),
            GestureType.FIST: (0, 0, 255),
            GestureType.NONE: (180, 180, 180)
        }

        for hand in frame_gestures.hands:
            color = GESTURE_COLORS.get(hand.gesture, (200, 200, 200))
            
            if hand.is_pinch:
                px, py = int(hand.pinch_point[0]), int(hand.pinch_point[1])
                cv2.circle(output, (px, py), 14, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.circle(output, (px, py), 4, (0, 255, 255), -1, cv2.LINE_AA)
                # Crosshair lines
                cv2.line(output, (px - 20, py), (px + 20, py), (0, 255, 255), 1, cv2.LINE_AA)
                cv2.line(output, (px, py - 20), (px, py + 20), (0, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(output, "PINCH", (px + 18, py - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA)

            elif hand.is_point:
                ix, iy = int(hand.index_tip[0]), int(hand.index_tip[1])
                cv2.circle(output, (ix, iy), 10, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.circle(output, (ix, iy), 3, (0, 255, 0), -1, cv2.LINE_AA)
                cv2.putText(output, "POINT", (ix + 15, iy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

            cx, cy = int(hand.palm_center[0]), int(hand.palm_center[1])
            badge_text = f"{hand.hand_label}: {hand.gesture.name}"
            cv2.putText(output, badge_text, (cx - 40, cy - 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(output, badge_text, (cx - 40, cy - 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)

        if len(frame_gestures.hands) >= 2 and frame_gestures.inter_hand_distance is not None:
            p0 = (int(frame_gestures.hands[0].palm_center[0]), int(frame_gestures.hands[0].palm_center[1]))
            p1 = (int(frame_gestures.hands[1].palm_center[0]), int(frame_gestures.hands[1].palm_center[1]))
            
            cv2.line(output, p0, p1, (255, 200, 0), 1, cv2.LINE_AA)
            mid_pt = ((p0[0] + p1[0]) // 2, (p0[1] + p1[1]) // 2)
            
            metric_text = f"Dist: {int(frame_gestures.inter_hand_distance)}px | Angle: {int(frame_gestures.relative_hand_angle)}deg"
            cv2.putText(output, metric_text, (mid_pt[0] - 100, mid_pt[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 230, 100), 1, cv2.LINE_AA)

        return output

    def render_overlay(self, frame: np.ndarray, hands: List[SingleHandData], debug_text: str = "") -> np.ndarray:
        """Render demo-ready HUD header overlay and bottom shortcut footer."""
        output = frame.copy()
        h, w, _ = output.shape

        # Header Bar
        header = output.copy()
        cv2.rectangle(header, (0, 0), (w, 42), (18, 18, 28), -1)
        cv2.addWeighted(header, 0.80, output, 0.20, 0, output)
        cv2.line(output, (0, 42), (w, 42), (60, 60, 80), 1, cv2.LINE_AA)

        cv2.putText(output, "AirDesk", (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 215, 255), 2, cv2.LINE_AA)

        fps_text = f"FPS: {int(self.fps)}"
        cv2.putText(output, fps_text, (w - 105, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 255, 180), 1, cv2.LINE_AA)

        hand_status = f"Hands: {len(hands)}"
        cv2.putText(output, hand_status, (140, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)

        if debug_text:
            cv2.putText(output, debug_text, (260, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 120), 1, cv2.LINE_AA)

        # Footer Legend Bar
        footer = output.copy()
        cv2.rectangle(footer, (0, h - 30), (w, h), (18, 18, 28), -1)
        cv2.addWeighted(footer, 0.80, output, 0.20, 0, output)
        cv2.line(output, (0, h - 30), (w, h - 30), (60, 60, 80), 1, cv2.LINE_AA)

        footer_text = "[POINT]: Draw  |  [PINCH]: Grab & Move/Rotate  |  [2-HAND]: Scale  |  [p]: Physics  |  [c]: Clear  |  [u]: Undo  |  [q]: Exit"
        cv2.putText(output, footer_text, (15, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 220), 1, cv2.LINE_AA)

        return output
