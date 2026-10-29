"""UI module for AirDesk on-screen toolbar and interactive widgets.

Provides OpenCV-rendered interactive buttons for:
- Workspace Mode Toggle: Mode: SELECT (default) vs Mode: DRAW
- 2D Shape Shortcuts: Create Rectangle, Create Circle, Create Triangle
- 3D Shape Shortcuts: Create 3D Cube, Create 3D Sphere, Create 3D Pyramid
- Workspace Actions: Physics Toggle, Delete Selected, Undo, Redo, Clear Workspace

Buttons support hover visual highlighting and pinch-click interaction with debounce protection.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import time
import cv2
import numpy as np


@dataclass
class ToolbarButton:
    """Dataclass representing an interactive toolbar button."""
    id: str
    label: str
    rect: Tuple[int, int, int, int]              # (x, y, w, h)
    action_code: str                             # Identifier for triggered action
    bg_color: Tuple[int, int, int] = (40, 40, 60) # Default BGR background
    text_color: Tuple[int, int, int] = (240, 240, 240)
    is_hovered: bool = False
    is_active: bool = False

    def contains(self, point: Tuple[float, float]) -> bool:
        """Check if 2D point is inside button rectangle."""
        px, py = point
        x, y, w, h = self.rect
        return x <= px <= (x + w) and y <= py <= (y + h)


class ToolbarUI:
    """Manages on-screen toolbar rendering and pinch-click interaction hit testing."""

    def __init__(self, panel_pos: Tuple[int, int] = (15, 45), button_size: Tuple[int, int] = (120, 28), padding: int = 6):
        self.panel_pos = panel_pos
        self.button_w, self.button_h = button_size
        self.padding = padding
        self.last_click_time: float = 0.0
        self.debounce_sec: float = 0.35

        self.buttons: List[ToolbarButton] = []
        self._init_buttons()

    def _init_buttons(self):
        """Construct toolbar button layout."""
        button_specs = [
            ("btn_mode", "Mode: SELECT", "toggle_mode", (0, 215, 255)),
            ("btn_rect", "+ 2D Rect", "create_rect", (0, 180, 255)),
            ("btn_circle", "+ 2D Circle", "create_circle", (0, 220, 150)),
            ("btn_tri", "+ 2D Tri", "create_triangle", (220, 100, 220)),
            ("btn_cube", "+ 3D Cube", "create_3d_cube", (0, 200, 255)),
            ("btn_sphere", "+ 3D Sphere", "create_3d_sphere", (0, 255, 180)),
            ("btn_pyram", "+ 3D Pyram", "create_3d_pyramid", (255, 100, 180)),
            ("btn_physics", "Physics: ON", "toggle_physics", (0, 180, 200)),
            ("btn_delete", "Delete", "delete_selected", (60, 60, 220)),
            ("btn_undo", "Undo", "undo", (200, 160, 0)),
            ("btn_redo", "Redo", "redo", (180, 140, 0)),
            ("btn_clear", "Clear", "clear", (80, 80, 100))
        ]

        start_x, start_y = self.panel_pos
        for i, (btn_id, label, action, color) in enumerate(button_specs):
            btn_y = start_y + i * (self.button_h + self.padding)
            rect = (start_x, btn_y, self.button_w, self.button_h)
            self.buttons.append(ToolbarButton(
                id=btn_id,
                label=label,
                rect=rect,
                action_code=action,
                bg_color=(45, 45, 65) if btn_id != "btn_mode" else (0, 120, 160),
                text_color=(255, 255, 255)
            ))

    def set_button_label(self, btn_id: str, new_label: str):
        """Update button label text dynamically."""
        for btn in self.buttons:
            if btn.id == btn_id:
                btn.label = new_label

    def update_interaction(self, cursor_point: Optional[Tuple[float, float]], is_pinch: bool) -> Optional[str]:
        """Check cursor hover and pinch-click status."""
        current_time = time.time()
        triggered_action: Optional[str] = None

        for btn in self.buttons:
            if cursor_point is not None and btn.contains(cursor_point):
                btn.is_hovered = True
                if is_pinch and (current_time - self.last_click_time) >= self.debounce_sec:
                    self.last_click_time = current_time
                    btn.is_active = True
                    triggered_action = btn.action_code
                    print(f"[ToolbarUI] Clicked button: {btn.label} (Action: {btn.action_code})")
            else:
                btn.is_hovered = False
                btn.is_active = False

        return triggered_action

    def render(self, frame: np.ndarray) -> np.ndarray:
        """Render toolbar panel and interactive buttons onto frame."""
        output = frame
        if not self.buttons:
            return output

        min_x = self.panel_pos[0] - 8
        min_y = self.panel_pos[1] - 8
        panel_w = self.button_w + 16
        panel_h = len(self.buttons) * (self.button_h + self.padding) + 8

        overlay = output.copy()
        cv2.rectangle(overlay, (min_x, min_y), (min_x + panel_w, min_y + panel_h), (25, 25, 35), -1)
        cv2.addWeighted(overlay, 0.75, output, 0.25, 0, output)
        cv2.rectangle(output, (min_x, min_y), (min_x + panel_w, min_y + panel_h), (80, 80, 100), 1, cv2.LINE_AA)

        cv2.putText(output, "TOOLBAR", (min_x + 10, min_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 215, 255), 1, cv2.LINE_AA)

        for btn in self.buttons:
            bx, by, bw, bh = btn.rect

            if btn.is_active:
                fill_color = (0, 255, 255)
                text_color = (0, 0, 0)
                border_color = (255, 255, 255)
            elif btn.is_hovered:
                fill_color = (70, 70, 100)
                text_color = (0, 255, 255)
                border_color = (0, 255, 255)
            else:
                fill_color = btn.bg_color
                text_color = btn.text_color
                border_color = (90, 90, 110)

            cv2.rectangle(output, (bx, by), (bx + bw, by + bh), fill_color, -1)
            cv2.rectangle(output, (bx, by), (bx + bw, by + bh), border_color, 1, cv2.LINE_AA)

            text_size, _ = cv2.getTextSize(btn.label, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
            tx = bx + (bw - text_size[0]) // 2
            ty = by + (bh + text_size[1]) // 2

            cv2.putText(output, btn.label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.38, text_color, 1, cv2.LINE_AA)

        return output
