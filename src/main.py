"""AirDesk Main Entrypoint — Overhauled.

PROJECT: AirDesk — Real-Time Gesture-Controlled Virtual Workspace.

Interaction Modes:
  CURSOR  — Move pointer without drawing. Default mode.
  DRAW    — POINT gesture draws freehand strokes. Active only when explicitly enabled.
  SELECT  — PINCH gesture selects and drags objects.
  ERASE   — OPEN_PALM gesture erases hovered objects.

Key Fixes:
  - Gestures have temporal hysteresis (4-frame confirmation) to prevent spurious activation.
  - POINT gesture is STRICT (only index finger extended, all others curled).
  - Drawing only starts on rising edge of POINT gesture in DRAW mode.
  - No connecting lines between strokes or when hand tracking is lost.
  - Hit-testing uses actual geometry (segment distance for strokes, bounding box for shapes).
  - Mouse fallback for every essential operation.
  - Confirmation dialog before Clear Canvas.
  - Mode switching cancels any active drawing stroke safely.
"""

import sys
import os
import argparse
import time
from enum import Enum, auto
from typing import Optional, List, Tuple

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from hand_tracker import HandTracker, SingleHandData
from gesture_detector import GestureDetector, GestureType, HandGesture, FrameGestures
from drawing_engine import DrawingEngine
from object_manager import ObjectManager, WorkspaceObject
from state_machine import StateMachine, State
from transformations import Transformations
from shape_recognizer import ShapeType
from renderer_3d import Renderer3D, ShapeType3D, WorkspaceObject3D
from physics_engine import PhysicsEngine
from widget_manager import WidgetManager, WorkspaceWidget
from geometry import calculate_angle_2d, distance_2d
from renderer import Renderer


# ─── App Mode Enum ──────────────────────────────────────────────────────────

class AppMode(Enum):
    CURSOR = auto()   # Move pointer, no drawing
    DRAW   = auto()   # POINT gesture → freehand strokes
    SELECT = auto()   # PINCH → select, drag, move objects
    ERASE  = auto()   # OPEN_PALM → hover-erase objects

    def __str__(self) -> str:
        return self.name


# Mode display colors (BGR)
MODE_COLORS = {
    AppMode.CURSOR: (200, 200, 200),
    AppMode.DRAW:   (0,   255,  80),
    AppMode.SELECT: (0,   220, 255),
    AppMode.ERASE:  (0,    80, 255),
}

# Pinch selection hysteresis: prevent immediate de-select on pinch release
SELECT_RELEASE_FRAMES = 3


# ─── Mouse Callback State ────────────────────────────────────────────────────

class MouseState:
    """Tracks mouse cursor position and button state for fallback interaction."""
    def __init__(self):
        self.pos: Tuple[int, int] = (0, 0)
        self.left_down: bool = False
        self.left_just_pressed: bool = False
        self.left_just_released: bool = False
        self._prev_left: bool = False

    def callback(self, event, x, y, flags, param):
        self.pos = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.left_down = True
        elif event == cv2.EVENT_LBUTTONUP:
            self.left_down = False

    def update(self):
        """Called once per frame to compute edge events."""
        self.left_just_pressed  = self.left_down and not self._prev_left
        self.left_just_released = not self.left_down and self._prev_left
        self._prev_left = self.left_down


# ─── Toolbar Renderer ────────────────────────────────────────────────────────

class ToolbarState:
    """Simple toolbar with clickable mode buttons and action buttons."""

    BUTTON_W = 110
    BUTTON_H = 30
    MARGIN = 8
    START_X = 10
    START_Y = 52  # Below header bar

    def __init__(self):
        self._buttons: List[dict] = []
        self._build()

    def _build(self):
        bw, bh = self.BUTTON_W, self.BUTTON_H
        x, y = self.START_X, self.START_Y
        gap = bh + self.MARGIN

        def btn(key, label, color=(60, 60, 80)):
            self._buttons.append({"key": key, "label": label,
                                   "rect": (x, y, bw, bh), "color": color})

        # Mode buttons
        btn("mode_cursor", "[ CURSOR ]", (50, 60, 80))
        y += gap
        btn("mode_draw",   "[ DRAW ]",   (0, 70, 30))
        y += gap
        btn("mode_select", "[ SELECT ]", (0, 60, 70))
        y += gap
        btn("mode_erase",  "[ ERASE ]",  (0, 30, 80))
        y += gap + 6

        # Action buttons
        btn("undo",   "Undo",   (50, 50, 60))
        y += gap
        btn("redo",   "Redo",   (50, 50, 60))
        y += gap
        btn("delete", "Delete", (60, 30, 30))
        y += gap
        btn("clear",  "Clear ✕", (80, 20, 20))
        y += gap + 6

        # Shape spawn buttons
        btn("rect",   "+ Rect",    (30, 50, 70))
        y += gap
        btn("circle", "+ Circle",  (30, 50, 70))
        y += gap
        btn("tri",    "+ Triangle",(30, 50, 70))
        y += gap + 6

        btn("physics","Physics ON", (40, 40, 60))

    def render(self, frame: np.ndarray, current_mode: AppMode,
               physics_on: bool, has_selection: bool) -> np.ndarray:
        bw, bh = self.BUTTON_W, self.BUTTON_H
        # Draw sidebar background
        h, w = frame.shape[:2]
        panel_w = bw + self.START_X * 2
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, self.START_Y - 4), (panel_w, h), (20, 22, 30), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        cv2.line(frame, (panel_w, self.START_Y - 4), (panel_w, h), (50, 55, 70), 1)

        for btn in self._buttons:
            key = btn["key"]
            label = btn["label"]
            bx, by, bw2, bh2 = btn["rect"]

            # Active mode highlight
            is_active = (
                (key == "mode_cursor" and current_mode == AppMode.CURSOR) or
                (key == "mode_draw"   and current_mode == AppMode.DRAW) or
                (key == "mode_select" and current_mode == AppMode.SELECT) or
                (key == "mode_erase"  and current_mode == AppMode.ERASE) or
                (key == "physics" and physics_on)
            )
            base_color = btn["color"]
            if is_active:
                bg_color = (
                    min(base_color[0] + 80, 255),
                    min(base_color[1] + 80, 255),
                    min(base_color[2] + 80, 255)
                )
                border_color = (0, 255, 180)
                text_color = (255, 255, 255)
            else:
                bg_color = base_color
                border_color = (80, 85, 100)
                text_color = (180, 185, 200)

            # Dim delete if nothing selected
            if key == "delete" and not has_selection:
                text_color = (80, 80, 90)

            cv2.rectangle(frame, (bx, by), (bx + bw2, by + bh2), bg_color, -1)
            cv2.rectangle(frame, (bx, by), (bx + bw2, by + bh2), border_color, 1)

            # Update physics label
            if key == "physics":
                label = f"Physics {'ON' if physics_on else 'OFF'}"

            cv2.putText(frame, label, (bx + 6, by + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, text_color, 1, cv2.LINE_AA)

        return frame

    def hit_test(self, point: Tuple[float, float]) -> Optional[str]:
        """Return button key if point is within any button rect."""
        px, py = point
        for btn in self._buttons:
            bx, by, bw, bh = btn["rect"]
            if bx <= px <= bx + bw and by <= py <= by + bh:
                return btn["key"]
        return None

    def get_panel_width(self) -> int:
        return self.BUTTON_W + self.START_X * 2


# ─── Clear Confirmation Overlay ──────────────────────────────────────────────

def ask_confirm_clear(frame: np.ndarray) -> np.ndarray:
    """Draw a confirmation overlay. Returns a copy of the frame with the dialog."""
    out = frame.copy()
    h, w = out.shape[:2]
    # Dim background
    overlay = np.zeros_like(out)
    cv2.addWeighted(overlay, 0.55, out, 0.45, 0, out)
    # Dialog box
    dw, dh = 380, 140
    dx, dy = (w - dw) // 2, (h - dh) // 2
    cv2.rectangle(out, (dx, dy), (dx + dw, dy + dh), (30, 32, 45), -1)
    cv2.rectangle(out, (dx, dy), (dx + dw, dy + dh), (0, 200, 255), 2)
    cv2.putText(out, "Clear canvas?", (dx + 20, dy + 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, "Press ENTER to confirm, ESC to cancel",
                (dx + 14, dy + 68), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 185, 200), 1, cv2.LINE_AA)
    # Buttons
    cv2.rectangle(out, (dx + 30,  dy + 88), (dx + 130, dy + 120), (0, 60, 30),  -1)
    cv2.rectangle(out, (dx + 30,  dy + 88), (dx + 130, dy + 120), (0, 200, 100), 1)
    cv2.putText(out, "ENTER: Yes", (dx + 38, dy + 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 120), 1, cv2.LINE_AA)
    cv2.rectangle(out, (dx + 250, dy + 88), (dx + 350, dy + 120), (50, 20, 20),  -1)
    cv2.rectangle(out, (dx + 250, dy + 88), (dx + 350, dy + 120), (0, 80, 200),  1)
    cv2.putText(out, "ESC: Cancel", (dx + 258, dy + 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (100, 150, 255), 1, cv2.LINE_AA)
    return out


# ─── HUD Renderer ────────────────────────────────────────────────────────────

def render_hud(frame: np.ndarray, fps: float, hands: List[SingleHandData],
               mode: AppMode, status_text: str, cam_preview: Optional[np.ndarray] = None) -> np.ndarray:
    """Render header bar, mode indicator, status text, and optional camera preview."""
    h, w = frame.shape[:2]
    mode_color = MODE_COLORS.get(mode, (200, 200, 200))

    # ── Header bar ──────────────────────────────────────────────────────────
    hdr = frame.copy()
    cv2.rectangle(hdr, (0, 0), (w, 44), (14, 16, 24), -1)
    cv2.addWeighted(hdr, 0.82, frame, 0.18, 0, frame)
    cv2.line(frame, (0, 44), (w, 44), (55, 60, 80), 1)

    # Logo
    cv2.putText(frame, "AirDesk", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.80, (0, 215, 255), 2, cv2.LINE_AA)

    # Hand count
    hand_txt = f"Hands: {len(hands)}"
    hand_col = (0, 255, 120) if hands else (80, 80, 100)
    cv2.putText(frame, hand_txt, (140, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, hand_col, 1, cv2.LINE_AA)
    if not hands:
        cv2.putText(frame, "No hand detected", (215, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 80, 160), 1, cv2.LINE_AA)

    # Mode pill
    mode_text = f" {mode.name} MODE "
    (tw, th), _ = cv2.getTextSize(mode_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    pill_x = 340
    cv2.rectangle(frame, (pill_x - 4, 10), (pill_x + tw + 8, 38), mode_color, -1, cv2.LINE_AA)
    cv2.putText(frame, mode_text, (pill_x, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (10, 12, 20), 2, cv2.LINE_AA)

    # Status text
    cv2.putText(frame, status_text, (pill_x + tw + 20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (210, 215, 230), 1, cv2.LINE_AA)

    # FPS
    fps_txt = f"FPS: {int(fps)}"
    cv2.putText(frame, fps_txt, (w - 105, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.60, (160, 255, 160), 1, cv2.LINE_AA)

    # ── Footer legend ────────────────────────────────────────────────────────
    ftr = frame.copy()
    cv2.rectangle(ftr, (0, h - 28), (w, h), (14, 16, 24), -1)
    cv2.addWeighted(ftr, 0.82, frame, 0.18, 0, frame)
    cv2.line(frame, (0, h - 28), (w, h - 28), (55, 60, 80), 1)
    legend = ("Keys: [1]=CURSOR  [2]=DRAW  [3]=SELECT  [4]=ERASE  "
              "[u]=Undo  [r]=Redo  [d]=Delete  [c]=Clear  [p]=Physics  [q]=Quit")
    cv2.putText(frame, legend, (130, h - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.40, (180, 185, 205), 1, cv2.LINE_AA)

    # ── Camera preview (top-right, small) ───────────────────────────────────
    if cam_preview is not None:
        ph, pw = 90, 120
        try:
            thumb = cv2.resize(cam_preview, (pw, ph))
            px_off = w - pw - 10
            py_off = h - ph - 35
            frame[py_off:py_off + ph, px_off:px_off + pw] = thumb
            cv2.rectangle(frame, (px_off - 1, py_off - 1),
                          (px_off + pw + 1, py_off + ph + 1), (55, 60, 80), 1)
        except Exception:
            pass

    return frame


def render_cursor(frame: np.ndarray, cursor_pt: Tuple[float, float],
                  mode: AppMode, is_confirmed: bool) -> np.ndarray:
    """Draw the gesture cursor at index-tip or pinch-point with mode-specific styling."""
    cx, cy = int(cursor_pt[0]), int(cursor_pt[1])
    color = MODE_COLORS.get(mode, (200, 200, 200))

    if mode == AppMode.DRAW and is_confirmed:
        # Green drawing dot with ring
        cv2.circle(frame, (cx, cy), 6, color, -1, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 12, color, 2, cv2.LINE_AA)
    elif mode == AppMode.SELECT:
        # Crosshair cursor
        cv2.circle(frame, (cx, cy), 10, color, 2, cv2.LINE_AA)
        cv2.line(frame, (cx - 16, cy), (cx + 16, cy), color, 1, cv2.LINE_AA)
        cv2.line(frame, (cx, cy - 16), (cx, cy + 16), color, 1, cv2.LINE_AA)
    elif mode == AppMode.ERASE:
        # Red X cursor
        cv2.circle(frame, (cx, cy), 14, (0, 80, 255), 2, cv2.LINE_AA)
        cv2.line(frame, (cx - 9, cy - 9), (cx + 9, cy + 9), (0, 80, 255), 2, cv2.LINE_AA)
        cv2.line(frame, (cx + 9, cy - 9), (cx - 9, cy + 9), (0, 80, 255), 2, cv2.LINE_AA)
    else:
        # Simple dot cursor
        cv2.circle(frame, (cx, cy), 5, color, -1, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 9, color, 1, cv2.LINE_AA)

    return frame


def render_objects(frame: np.ndarray, objects: List[WorkspaceObject],
                   hovered_obj: Optional[WorkspaceObject] = None) -> np.ndarray:
    """Render workspace objects with hover highlight and CAD selection handles."""
    import math as _math
    for obj in objects:
        # Skip objects with invalid (NaN/Inf) positions — can happen during physics instability
        if not (_math.isfinite(obj.position[0]) and _math.isfinite(obj.position[1])):
            continue
        world_pts = obj.get_world_vertices()
        cx, cy = int(obj.position[0]), int(obj.position[1])
        is_hovered = (hovered_obj is not None and hovered_obj.id == obj.id)


        if obj.selected:
            draw_color = (0, 255, 255)
            line_thick = 4
        elif is_hovered:
            draw_color = (0, 230, 255)
            line_thick = 3
        else:
            draw_color = obj.color
            line_thick = 2

        # Draw shape geometry
        if obj.shape_type == ShapeType.CIRCLE:
            radius = int((obj.size[0] / 2.0) * obj.scale)
            cv2.circle(frame, (cx, cy), radius, draw_color, line_thick, cv2.LINE_AA)
            import math
            rad = math.radians(obj.rotation)
            rx = int(cx + radius * math.cos(rad))
            ry = int(cy + radius * math.sin(rad))
            cv2.line(frame, (cx, cy), (rx, ry), draw_color, 1, cv2.LINE_AA)

        elif obj.shape_type == ShapeType.LINE and len(world_pts) >= 2:
            p1 = (int(world_pts[0][0]), int(world_pts[0][1]))
            p2 = (int(world_pts[1][0]), int(world_pts[1][1]))
            cv2.line(frame, p1, p2, draw_color, line_thick, cv2.LINE_AA)
            cv2.circle(frame, p1, 5, draw_color, -1, cv2.LINE_AA)
            cv2.circle(frame, p2, 5, draw_color, -1, cv2.LINE_AA)

        elif obj.shape_type == ShapeType.UNRECOGNIZED and obj.raw_points:
            # Render freehand stroke as polyline
            world_raw = obj.get_world_raw_points()
            if len(world_raw) > 1:
                pts_array = np.array(world_raw, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [pts_array], isClosed=False, color=draw_color,
                              thickness=line_thick, lineType=cv2.LINE_AA)

        elif len(world_pts) > 2:
            pts_array = np.array(world_pts, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts_array], isClosed=True, color=draw_color,
                          thickness=line_thick, lineType=cv2.LINE_AA)

        # Center node
        cv2.circle(frame, (cx, cy), 3, draw_color, -1, cv2.LINE_AA)

        # Hover ring (before pinching)
        if is_hovered and not obj.selected:
            all_x = [pt[0] for pt in world_pts] if world_pts else [cx]
            all_y = [pt[1] for pt in world_pts] if world_pts else [cy]
            if obj.raw_points:
                wrp = obj.get_world_raw_points()
                all_x = [p[0] for p in wrp]
                all_y = [p[1] for p in wrp]
            min_x, max_x = int(min(all_x)) - 10, int(max(all_x)) + 10
            min_y, max_y = int(min(all_y)) - 10, int(max(all_y)) + 10
            cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (0, 230, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, "PINCH TO SELECT", (min_x, min_y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 230, 255), 1, cv2.LINE_AA)

        # CAD selection bounding box + corner handles + rotation knob
        if obj.selected:
            all_x = [pt[0] for pt in world_pts] if world_pts else [cx]
            all_y = [pt[1] for pt in world_pts] if world_pts else [cy]
            if obj.raw_points:
                wrp = obj.get_world_raw_points()
                all_x = [p[0] for p in wrp]
                all_y = [p[1] for p in wrp]
            min_x, max_x = int(min(all_x)) - 12, int(max(all_x)) + 12
            min_y, max_y = int(min(all_y)) - 12, int(max(all_y)) + 12

            cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (0, 255, 255), 1, cv2.LINE_AA)
            for hx, hy in [(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)]:
                cv2.rectangle(frame, (hx - 4, hy - 4), (hx + 4, hy + 4), (0, 255, 255), -1)

            # Rotation knob
            knob_x, knob_y = cx, min_y - 26
            cv2.line(frame, (cx, min_y), (knob_x, knob_y), (0, 255, 120), 2, cv2.LINE_AA)
            cv2.circle(frame, (knob_x, knob_y), 7, (0, 255, 120), -1, cv2.LINE_AA)
            cv2.circle(frame, (knob_x, knob_y), 9, (255, 255, 255), 1, cv2.LINE_AA)

            # Label
            label = f"#{obj.id} {obj.shape_type.name}"
            cv2.putText(frame, label, (min_x, min_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, label, (min_x, min_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 255, 255), 1, cv2.LINE_AA)

    return frame


# ─── Main Application Loop ───────────────────────────────────────────────────

def run_app(camera_id: int = 0, width: int = 1280, height: int = 720, synthetic: bool = False):
    """Run AirDesk application loop."""
    print("=====================================================================")
    print("AirDesk — Real-Time Gesture-Controlled Virtual Workspace")
    print("=====================================================================")
    print("[AirDesk] Initializing modules...")

    tracker       = HandTracker(max_hands=2, min_detection_confidence=0.5)
    detector      = GestureDetector()
    drawing_engine = DrawingEngine()
    object_manager = ObjectManager()
    renderer_3d   = Renderer3D(camera_center=(width / 2.0, height / 2.0))
    physics_engine = PhysicsEngine(viewport_size=(width, height))
    widget_manager = WidgetManager(viewport_size=(width, height))
    state_machine  = StateMachine()
    toolbar        = ToolbarState()
    mouse          = MouseState()
    renderer       = Renderer(window_name="AirDesk — Virtual Workspace")

    # App state
    current_mode: AppMode = AppMode.CURSOR
    status_text: str = "Ready — Use gestures or mouse"
    physics_on: bool = True
    confirm_clear_active: bool = False

    # Selection & drag state
    selected_widget: Optional[WorkspaceWidget] = None
    selected_3d_obj: Optional[WorkspaceObject3D] = None
    pinch_release_frames: int = 0

    # For velocity-based throw
    hand_pos_history: List[Tuple[float, float, float]] = []

    fps_tracker: float = 0.0
    last_frame_time = time.time()

    if synthetic:
        print("[AirDesk] Running synthetic test...")
        obj = object_manager.create_primitive_object(ShapeType.RECTANGLE, center=(width // 2, 200.0))
        renderer_3d.create_3d_object(ShapeType3D.CUBE, position_3d=(width // 2 + 200, 300.0, 500.0))
        for i in range(30):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            hands = tracker.process_frame(frame)
            frame_gestures = detector.detect(hands)
            physics_engine.step(0.033, object_manager.objects)
            renderer.update_fps()
            time.sleep(0.01)
        fps_tracker = renderer.fps
        print(f"[AirDesk] Synthetic test complete (~{int(fps_tracker)} FPS).")
        tracker.close()
        return

    print(f"[AirDesk] Opening camera device {camera_id}...")
    cap_flags = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    cap = cv2.VideoCapture(camera_id, cap_flags)
    if not cap.isOpened():
        cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"[AirDesk] Error: Cannot open camera {camera_id}.")
        print("[AirDesk] Run with --synthetic for testing without a webcam.")
        tracker.close()
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    cv2.namedWindow(renderer.window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(renderer.window_name, width, height)
    cv2.setMouseCallback(renderer.window_name, mouse.callback)

    print("[AirDesk] Workspace Active!")
    print("  Gesture modes:")
    print("  - [1] CURSOR: Move hand freely — no drawing")
    print("  - [2] DRAW:   POINT (index only) → draw freehand")
    print("  - [3] SELECT: PINCH → select, drag objects")
    print("  - [4] ERASE:  OPEN_PALM → hover-erase objects")
    print("  - [u] Undo  [r] Redo  [d] Delete  [c] Clear  [p] Physics  [q] Quit")
    print("  Mouse: Click on toolbar buttons or canvas objects.")

    try:
        while cap.isOpened():
            current_time = time.time()
            dt = max(current_time - last_frame_time, 1e-4)
            last_frame_time = current_time

            ret, raw_frame = cap.read()
            if not ret or raw_frame is None:
                print("[AirDesk] Frame capture failed.")
                break

            frame = cv2.flip(raw_frame, 1)
            cam_preview = cv2.resize(frame, (120, 90))  # Small preview copy

            # ── Hand tracking & gesture detection ────────────────────────────
            hands = tracker.process_frame(frame)
            frame_gestures = detector.detect(hands)
            mouse.update()

            # Resolve active hands
            pinching_hand: Optional[HandGesture] = None
            pointing_hand: Optional[HandGesture] = None
            open_palm_hand: Optional[HandGesture] = None

            for h in frame_gestures.hands:
                if h.is_pinch   and pinching_hand  is None: pinching_hand  = h
                if h.is_point   and pointing_hand  is None: pointing_hand  = h
                if h.is_open_palm and open_palm_hand is None: open_palm_hand = h

            # Cursor point (used for hover & toolbar)
            cursor_pt: Optional[Tuple[float, float]] = None
            if pinching_hand:
                cursor_pt = pinching_hand.pinch_point
            elif pointing_hand:
                cursor_pt = pointing_hand.index_tip
            elif open_palm_hand:
                cursor_pt = open_palm_hand.palm_center

            # Mouse cursor as fallback when no hand gesture
            if cursor_pt is None:
                cursor_pt = mouse.pos

            # ── Velocity tracking for physics throw ──────────────────────────
            hand_vel = (0.0, 0.0)
            if pinching_hand:
                hand_pos_history.append((pinching_hand.pinch_point[0],
                                         pinching_hand.pinch_point[1], current_time))
                if len(hand_pos_history) > 5:
                    hand_pos_history.pop(0)
                if len(hand_pos_history) >= 2:
                    dt_pos = hand_pos_history[-1][2] - hand_pos_history[0][2]
                    if dt_pos > 1e-4:
                        vx = (hand_pos_history[-1][0] - hand_pos_history[0][0]) / dt_pos
                        vy = (hand_pos_history[-1][1] - hand_pos_history[0][1]) / dt_pos
                        hand_vel = (vx, vy)
            else:
                hand_pos_history.clear()

            # ── Confirmation dialog ──────────────────────────────────────────
            if confirm_clear_active:
                key = cv2.waitKey(1) & 0xFF
                if key == 13:  # ENTER
                    drawing_engine.clear_all()
                    object_manager.clear_all()
                    renderer_3d.objects_3d.clear()
                    physics_engine.clear_all()
                    selected_3d_obj = None
                    selected_widget = None
                    state_machine.reset_to_idle()
                    status_text = "Canvas cleared"
                    confirm_clear_active = False
                elif key == 27:  # ESC
                    status_text = "Clear cancelled"
                    confirm_clear_active = False

                diag_frame = ask_confirm_clear(frame)
                diag_frame = render_hud(diag_frame, renderer.fps, hands, current_mode, status_text)
                cv2.imshow(renderer.window_name, diag_frame)
                renderer.update_fps()
                continue

            # ── Toolbar interaction (gesture pinch OR mouse click) ────────────
            toolbar_panel_w = toolbar.get_panel_width()
            pinch_in_toolbar = (cursor_pt is not None and
                                cursor_pt[0] <= toolbar_panel_w and
                                (pinching_hand is not None or mouse.left_just_pressed))

            if pinch_in_toolbar:
                hit_key = toolbar.hit_test(cursor_pt)
                if hit_key:
                    if hit_key == "mode_cursor":
                        drawing_engine.cancel_active_stroke()
                        current_mode = AppMode.CURSOR
                        status_text = "CURSOR MODE — Move freely"
                    elif hit_key == "mode_draw":
                        drawing_engine.cancel_active_stroke()
                        current_mode = AppMode.DRAW
                        status_text = "DRAW MODE — POINT to draw"
                    elif hit_key == "mode_select":
                        drawing_engine.cancel_active_stroke()
                        current_mode = AppMode.SELECT
                        status_text = "SELECT MODE — PINCH to select & drag"
                    elif hit_key == "mode_erase":
                        drawing_engine.cancel_active_stroke()
                        current_mode = AppMode.ERASE
                        status_text = "ERASE MODE — OPEN_PALM to erase"
                    elif hit_key == "undo":
                        if object_manager.undo():
                            state_machine.reset_to_idle()
                            status_text = "Undo"
                    elif hit_key == "redo":
                        if object_manager.redo():
                            state_machine.reset_to_idle()
                            status_text = "Redo"
                    elif hit_key == "delete":
                        deleted = object_manager.delete_selected()
                        if deleted:
                            physics_engine.unregister_object(deleted.id)
                            state_machine.reset_to_idle()
                            status_text = f"Deleted #{deleted.id}"
                        else:
                            status_text = "Nothing selected"
                    elif hit_key == "clear":
                        confirm_clear_active = True
                        status_text = "Confirm clear? (ENTER/ESC)"
                    elif hit_key == "rect":
                        obj = object_manager.create_primitive_object(
                            ShapeType.RECTANGLE, center=(width // 2, height // 2 - 100.0))
                        physics_engine.register_object(obj)
                        status_text = f"Created Rect #{obj.id}"
                    elif hit_key == "circle":
                        obj = object_manager.create_primitive_object(
                            ShapeType.CIRCLE, center=(width // 2, height // 2 - 100.0))
                        physics_engine.register_object(obj)
                        status_text = f"Created Circle #{obj.id}"
                    elif hit_key == "tri":
                        obj = object_manager.create_primitive_object(
                            ShapeType.TRIANGLE, center=(width // 2, height // 2 - 100.0))
                        physics_engine.register_object(obj)
                        status_text = f"Created Triangle #{obj.id}"
                    elif hit_key == "physics":
                        physics_on = physics_engine.toggle_physics()
                        status_text = f"Physics {'ON' if physics_on else 'OFF'}"

            # ── Mode-specific workspace interactions ─────────────────────────
            else:
                # --- CURSOR MODE: No drawing, no selection ---
                if current_mode == AppMode.CURSOR:
                    # Draggable widget windows (pinch drag)
                    if pinching_hand:
                        ppt = pinching_hand.pinch_point
                        if not state_machine.is_selected():
                            hit_w = widget_manager.hit_test_title_bar(ppt)
                            if hit_w:
                                selected_widget = hit_w
                                selected_widget.selected = True
                                wx, wy, ww, wh = selected_widget.rect
                                selected_widget.drag_offset = (wx - ppt[0], wy - ppt[1])
                                state_machine.transition_to(State.SELECTED)
                        else:
                            if selected_widget:
                                ox, oy = selected_widget.drag_offset
                                selected_widget.rect = (
                                    int(ppt[0] + ox), int(ppt[1] + oy),
                                    selected_widget.rect[2], selected_widget.rect[3])
                    else:
                        if state_machine.is_selected() and selected_widget:
                            selected_widget.selected = False
                            selected_widget = None
                            state_machine.reset_to_idle()

                # --- DRAW MODE: POINT gesture → freehand stroke ---
                elif current_mode == AppMode.DRAW:
                    newly_completed = drawing_engine.update(frame_gestures, mode_is_draw=True)
                    if drawing_engine.is_drawing:
                        state_machine.transition_to(State.DRAWING)
                        status_text = "Drawing..."
                    else:
                        if state_machine.is_drawing():
                            state_machine.transition_to(State.IDLE)

                    if newly_completed and newly_completed.recognized_shape:
                        new_obj = object_manager.create_object_from_shape(
                            newly_completed.recognized_shape, newly_completed.points)
                        physics_engine.register_object(new_obj)
                        state_machine.transition_to(State.SHAPE_CREATED, object_id=new_obj.id)
                        status_text = f"Drew {new_obj.shape_type.name} #{new_obj.id}"
                        state_machine.transition_to(State.IDLE)

                # --- SELECT MODE: PINCH → select / drag objects ---
                elif current_mode == AppMode.SELECT:
                    if pinching_hand:
                        pinch_release_frames = 0
                        ppt = pinching_hand.pinch_point

                        if not state_machine.is_selected():
                            # Try widget title bar first
                            hit_w = widget_manager.hit_test_title_bar(ppt)
                            if hit_w:
                                selected_widget = hit_w
                                selected_widget.selected = True
                                wx, wy, ww, wh = selected_widget.rect
                                selected_widget.drag_offset = (wx - ppt[0], wy - ppt[1])
                                state_machine.transition_to(State.SELECTED)
                            else:
                                # Spatial hit-test 2D objects
                                target = object_manager.hit_test(ppt, tolerance_px=20.0)
                                if target:
                                    object_manager.select_object(target.id)
                                    state_machine.drag_offset = (
                                        target.position[0] - ppt[0],
                                        target.position[1] - ppt[1])
                                    state_machine.initial_hand_angle = calculate_angle_2d(
                                        pinching_hand.palm_center, pinching_hand.index_tip)
                                    state_machine.initial_object_rotation = target.rotation
                                    state_machine.initial_object_scale = target.scale
                                    if frame_gestures.inter_hand_distance is not None:
                                        state_machine.initial_inter_hand_dist = frame_gestures.inter_hand_distance
                                    state_machine.transition_to(State.SELECTED, object_id=target.id)
                                    status_text = f"Selected #{target.id} — drag to move"
                                else:
                                    # Click on empty area → deselect
                                    object_manager.deselect_all()
                                    state_machine.reset_to_idle()
                                    status_text = "Deselected"
                        else:
                            # Drag selected widget
                            if selected_widget:
                                ox, oy = selected_widget.drag_offset
                                selected_widget.rect = (
                                    int(ppt[0] + ox), int(ppt[1] + oy),
                                    selected_widget.rect[2], selected_widget.rect[3])
                            else:
                                selected_obj = object_manager.get_object_by_id(state_machine.active_object_id)
                                if selected_obj:
                                    new_pos = Transformations.translate(
                                        selected_obj, ppt, state_machine.drag_offset)
                                    physics_engine.sync_held_object(selected_obj, new_pos, hand_vel)

                                    curr_angle = calculate_angle_2d(
                                        pinching_hand.palm_center, pinching_hand.index_tip)
                                    Transformations.rotate(
                                        selected_obj, curr_angle,
                                        state_machine.initial_hand_angle,
                                        state_machine.initial_object_rotation)

                                    if (len(frame_gestures.hands) >= 2 and
                                            frame_gestures.inter_hand_distance is not None):
                                        Transformations.scale(
                                            selected_obj, frame_gestures.inter_hand_distance,
                                            state_machine.initial_inter_hand_dist,
                                            state_machine.initial_object_scale)

                    else:
                        # Pinch released — hysteresis before deselecting
                        if state_machine.is_selected():
                            pinch_release_frames += 1
                            if pinch_release_frames >= SELECT_RELEASE_FRAMES:
                                if selected_widget:
                                    selected_widget.selected = False
                                    selected_widget = None
                                else:
                                    selected_obj = object_manager.get_object_by_id(
                                        state_machine.active_object_id)
                                    if selected_obj:
                                        physics_engine.release_object_with_throw(selected_obj, hand_vel)
                                        status_text = f"Released #{selected_obj.id}"

                                object_manager.deselect_all()
                                state_machine.reset_to_idle()
                                pinch_release_frames = 0

                    # Mouse fallback: click on object to select
                    if mouse.left_just_pressed and cursor_pt is not None:
                        if cursor_pt[0] > toolbar_panel_w:
                            target = object_manager.hit_test(cursor_pt, tolerance_px=20.0)
                            if target:
                                object_manager.select_object(target.id)
                                state_machine.transition_to(State.SELECTED, object_id=target.id)
                                state_machine.drag_offset = (
                                    target.position[0] - cursor_pt[0],
                                    target.position[1] - cursor_pt[1])
                                status_text = f"Mouse selected #{target.id}"
                            else:
                                object_manager.deselect_all()
                                state_machine.reset_to_idle()

                # --- ERASE MODE: OPEN_PALM over object → erase it ---
                elif current_mode == AppMode.ERASE:
                    if open_palm_hand:
                        erase_pt = open_palm_hand.palm_center
                        target = object_manager.hit_test(erase_pt, tolerance_px=30.0)
                        if target:
                            object_manager.remove_object(target.id)
                            physics_engine.unregister_object(target.id)
                            status_text = f"Erased #{target.id}"

                # --- MOUSE CLICK fallback for drawing in DRAW mode ---
                if current_mode == AppMode.DRAW and mouse.left_just_pressed:
                    # Mouse click places a small circle object
                    if cursor_pt[0] > toolbar_panel_w:
                        obj = object_manager.create_primitive_object(
                            ShapeType.CIRCLE,
                            center=(float(cursor_pt[0]), float(cursor_pt[1])),
                            size=(60.0, 60.0))
                        physics_engine.register_object(obj)
                        status_text = f"Mouse-placed circle #{obj.id}"

            # ── Hover pre-highlight (outside toolbar, in SELECT mode) ─────────
            hovered_obj: Optional[WorkspaceObject] = None
            if cursor_pt and current_mode == AppMode.SELECT and not state_machine.is_selected():
                if cursor_pt[0] > toolbar_panel_w:
                    hovered_obj = object_manager.hit_test(cursor_pt, tolerance_px=20.0)

            # ── Physics step ─────────────────────────────────────────────────
            physics_engine.step(dt, object_manager.objects)

            # ── Rendering pipeline ───────────────────────────────────────────
            # 1. Hand skeleton
            frame = tracker.draw_landmarks(frame, hands)

            # 2. 2D workspace objects
            frame = render_objects(frame, object_manager.objects, hovered_obj=hovered_obj)

            # 3. 3D objects
            frame = renderer_3d.render_3d_objects(frame)

            # 4. Draggable widgets
            frame = widget_manager.render(frame)

            # 5. Drawing strokes (in-progress and completed)
            if current_mode == AppMode.DRAW:
                frame = drawing_engine.render_strokes(frame)

            # 6. Gesture cursor overlay
            is_drawing_confirmed = drawing_engine.is_drawing
            if cursor_pt is not None:
                frame = render_cursor(frame, cursor_pt, current_mode, is_drawing_confirmed)

            # 7. Toolbar sidebar
            has_selection = state_machine.is_selected()
            frame = toolbar.render(frame, current_mode, physics_on=physics_on,
                                   has_selection=has_selection)

            # 8. HUD header + footer
            frame = render_hud(frame, renderer.fps, hands, current_mode,
                               status_text, cam_preview=cam_preview)

            renderer.update_fps()
            cv2.imshow(renderer.window_name, frame)

            # ── Keyboard shortcuts ───────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                print("[AirDesk] Exit requested.")
                break
            elif key == ord("1"):
                drawing_engine.cancel_active_stroke()
                current_mode = AppMode.CURSOR
                status_text = "CURSOR MODE"
            elif key == ord("2"):
                drawing_engine.cancel_active_stroke()
                current_mode = AppMode.DRAW
                status_text = "DRAW MODE — POINT to draw"
            elif key == ord("3"):
                drawing_engine.cancel_active_stroke()
                current_mode = AppMode.SELECT
                status_text = "SELECT MODE"
            elif key == ord("4"):
                drawing_engine.cancel_active_stroke()
                current_mode = AppMode.ERASE
                status_text = "ERASE MODE"
            elif key == ord("u"):
                if object_manager.undo():
                    state_machine.reset_to_idle()
                    status_text = "Undo"
            elif key == ord("r"):
                if object_manager.redo():
                    state_machine.reset_to_idle()
                    status_text = "Redo"
            elif key == ord("d"):
                deleted = object_manager.delete_selected()
                if deleted:
                    physics_engine.unregister_object(deleted.id)
                    state_machine.reset_to_idle()
                    status_text = f"Deleted #{deleted.id}"
            elif key == ord("c"):
                confirm_clear_active = True
                status_text = "Confirm clear? (ENTER/ESC)"
            elif key == ord("p"):
                physics_on = physics_engine.toggle_physics()
                status_text = f"Physics {'ON' if physics_on else 'OFF'}"
            elif key == ord("m"):
                # Cycle modes
                modes = list(AppMode)
                idx = modes.index(current_mode)
                drawing_engine.cancel_active_stroke()
                current_mode = modes[(idx + 1) % len(modes)]
                status_text = f"{current_mode.name} MODE"

    finally:
        cap.release()
        cv2.destroyAllWindows()
        tracker.close()
        print("[AirDesk] Shutdown complete.")


def main():
    parser = argparse.ArgumentParser(description="AirDesk — Real-Time Gesture-Controlled Virtual Workspace")
    parser.add_argument("--cam",       type=int,  default=0,    help="Camera device index (default: 0)")
    parser.add_argument("--width",     type=int,  default=1280, help="Video width (default: 1280)")
    parser.add_argument("--height",    type=int,  default=720,  help="Video height (default: 720)")
    parser.add_argument("--synthetic", action="store_true",     help="Run synthetic frames (no camera)")
    args = parser.parse_args()
    run_app(camera_id=args.cam, width=args.width, height=args.height, synthetic=args.synthetic)


if __name__ == "__main__":
    main()
