"""Widget manager module for AirDesk draggable workspace widgets.

Provides interactive, draggable widget windows (e.g., Quick Notes / Instruction Widget)
that can be grabbed via PINCH gesture and moved across the workspace.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import cv2
import numpy as np


@dataclass
class WorkspaceWidget:
    """Dataclass representing a draggable workspace widget window."""
    id: str
    title: str
    rect: Tuple[int, int, int, int]          # (x, y, w, h)
    lines: List[str]                         # Text lines content inside widget
    bg_color: Tuple[int, int, int] = (25, 25, 35)
    border_color: Tuple[int, int, int] = (0, 215, 255)
    selected: bool = False
    drag_offset: Tuple[float, float] = (0.0, 0.0)

    def contains_title_bar(self, point: Tuple[float, float]) -> bool:
        """Check if point is inside title bar handle for dragging."""
        px, py = point
        x, y, w, h = self.rect
        return x <= px <= (x + w) and y <= py <= (y + 28)

    def contains(self, point: Tuple[float, float]) -> bool:
        """Check if point is anywhere inside widget window."""
        px, py = point
        x, y, w, h = self.rect
        return x <= px <= (x + w) and y <= py <= (y + h)


class WidgetManager:
    """Manages draggable workspace widgets."""

    def __init__(self, viewport_size: Tuple[int, int] = (1280, 720)):
        self.viewport_w, self.viewport_h = viewport_size
        self.widgets: List[WorkspaceWidget] = []
        self._init_default_widgets()

    def _init_default_widgets(self):
        """Construct default workspace widgets (Notes & Controls Widget)."""
        notes_rect = (self.viewport_w - 260, 60, 240, 160)
        notes_lines = [
            "Quick Gestures:",
            "- POINT: Air Drawing",
            "- PINCH: Select & Move",
            "- PINCH + Rotate: Spin",
            "- 2-HAND OPEN: Scale",
            "- Fast PINCH Release: Throw"
        ]

        self.widgets.append(WorkspaceWidget(
            id="widget_notes",
            title="AirDesk Notes",
            rect=notes_rect,
            lines=notes_lines,
            bg_color=(20, 20, 30),
            border_color=(0, 215, 255)
        ))

    def hit_test_title_bar(self, point: Tuple[float, float]) -> Optional[WorkspaceWidget]:
        """Spatial hit test for widget title bar handle."""
        for widget in reversed(self.widgets):
            if widget.contains_title_bar(point):
                return widget
        return None

    def render(self, frame: np.ndarray) -> np.ndarray:
        """Render draggable widget windows onto frame."""
        output = frame

        for widget in self.widgets:
            wx, wy, ww, wh = widget.rect

            # Window Background & Translucent Overlay
            overlay = output.copy()
            cv2.rectangle(overlay, (wx, wy), (wx + ww, wy + wh), widget.bg_color, -1)
            cv2.addWeighted(overlay, 0.82, output, 0.18, 0, output)

            # Window Border & Selection Glow
            border_color = (0, 255, 255) if widget.selected else widget.border_color
            border_thick = 2 if widget.selected else 1
            cv2.rectangle(output, (wx, wy), (wx + ww, wy + wh), border_color, border_thick, cv2.LINE_AA)

            # Title Bar Header
            cv2.rectangle(output, (wx, wy), (wx + ww, wy + 26), (40, 40, 60), -1)
            cv2.line(output, (wx, wy + 26), (wx + ww, wy + 26), border_color, 1, cv2.LINE_AA)
            cv2.putText(output, widget.title, (wx + 10, wy + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 215, 255), 1, cv2.LINE_AA)

            # Drag indicator handle dots
            for dx in [ww - 25, ww - 18, ww - 11]:
                cv2.circle(output, (wx + dx, wy + 13), 2, (180, 180, 180), -1, cv2.LINE_AA)

            # Widget Text Content
            for idx, line in enumerate(widget.lines):
                line_y = wy + 46 + idx * 20
                if line_y < wy + wh - 10:
                    color = (255, 255, 255) if not line.startswith("-") else (200, 220, 255)
                    cv2.putText(output, line, (wx + 12, line_y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)

        return output
