"""Unit tests for Phase 7 Toolbar UI & Undo/Redo History."""

import os
import sys
import unittest
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from ui import ToolbarUI, ToolbarButton
from object_manager import ObjectManager, WorkspaceObject
from shape_recognizer import ShapeType


class TestToolbarUI(unittest.TestCase):

    def setUp(self):
        self.toolbar = ToolbarUI(panel_pos=(10, 50), button_size=(100, 30))
        self.manager = ObjectManager()

    def test_button_hover_and_click(self):
        """Verify button hover hit-testing and pinch click detection."""
        # Find first button rect (10, 50, 100, 30)
        btn1 = self.toolbar.buttons[0]
        self.assertEqual(btn1.rect, (10, 50, 100, 30))

        # Hover point at center of button (60, 65)
        hover_pt = (60.0, 65.0)

        # Update without pinch -> should highlight hover, no action
        act_hover = self.toolbar.update_interaction(hover_pt, is_pinch=False)
        self.assertTrue(btn1.is_hovered)
        self.assertIsNone(act_hover)

        # Update with pinch -> should trigger action_code
        act_click = self.toolbar.update_interaction(hover_pt, is_pinch=True)
        self.assertEqual(act_click, btn1.action_code)

    def test_debounce_timer(self):
        """Verify click debounce timer prevents accidental double clicks."""
        hover_pt = (60.0, 65.0)
        
        # Click 1
        act1 = self.toolbar.update_interaction(hover_pt, is_pinch=True)
        self.assertIsNotNone(act1)

        # Immediate Click 2 (0ms later) -> should be debounced and return None
        act2 = self.toolbar.update_interaction(hover_pt, is_pinch=True)
        self.assertIsNone(act2)

    def test_undo_redo_history(self):
        """Verify Undo/Redo snapshot history for multi-object state management."""
        self.assertEqual(len(self.manager.objects), 0)

        # Create Object 1
        obj1 = self.manager.create_primitive_object(ShapeType.RECTANGLE)
        self.assertEqual(len(self.manager.objects), 1)

        # Create Object 2
        obj2 = self.manager.create_primitive_object(ShapeType.CIRCLE)
        self.assertEqual(len(self.manager.objects), 2)

        # Test Undo
        self.assertTrue(self.manager.undo())
        self.assertEqual(len(self.manager.objects), 1)
        self.assertEqual(self.manager.objects[0].id, obj1.id)

        # Test Redo
        self.assertTrue(self.manager.redo())
        self.assertEqual(len(self.manager.objects), 2)


if __name__ == '__main__':
    unittest.main()
