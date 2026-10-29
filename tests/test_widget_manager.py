"""Unit tests for Phase 10 Draggable Widget Manager."""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from widget_manager import WidgetManager, WorkspaceWidget


class TestWidgetManager(unittest.TestCase):

    def setUp(self):
        self.manager = WidgetManager(viewport_size=(1280, 720))

    def test_default_widget_creation(self):
        """Verify default Notes widget initialization."""
        self.assertGreaterEqual(len(self.manager.widgets), 1)
        notes = self.manager.widgets[0]
        self.assertEqual(notes.id, "widget_notes")
        self.assertEqual(notes.title, "AirDesk Notes")

    def test_title_bar_hit_test(self):
        """Verify title bar hit testing for dragging widgets."""
        notes = self.manager.widgets[0]
        wx, wy, ww, wh = notes.rect

        # Point inside title bar (wx + 10, wy + 10) -> hit
        hit = self.manager.hit_test_title_bar((wx + 10.0, wy + 10.0))
        self.assertIsNotNone(hit)
        self.assertEqual(hit.id, notes.id)

        # Point inside body of widget (wx + 10, wy + 80) -> not title bar hit
        miss = self.manager.hit_test_title_bar((wx + 10.0, wy + 80.0))
        self.assertIsNone(miss)


if __name__ == '__main__':
    unittest.main()
