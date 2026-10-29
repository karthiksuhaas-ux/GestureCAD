"""Unit tests for Phase 6 State Machine & Spatial Transformations."""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from shape_recognizer import RecognizedShape, ShapeType
from object_manager import ObjectManager, WorkspaceObject
from state_machine import StateMachine, State
from transformations import Transformations


class TestTransformations(unittest.TestCase):

    def setUp(self):
        self.manager = ObjectManager()
        self.state_machine = StateMachine()
        
        # Create a test object centered at (200, 200) size (100, 100)
        rec = RecognizedShape(shape_type=ShapeType.RECTANGLE, bounding_box=(150, 150, 100, 100), center=(200.0, 200.0))
        self.test_obj = self.manager.create_object_from_shape(rec, raw_points=[])

    def test_spatial_hit_test(self):
        """Verify spatial hit test correctly identifies object at pinch point."""
        # Pinch at (210, 205) -> should hit test_obj
        hit = self.manager.hit_test((210.0, 205.0), tolerance_px=30.0)
        self.assertIsNotNone(hit)
        self.assertEqual(hit.id, self.test_obj.id)

        # Pinch far away at (800, 800) -> should return None
        miss = self.manager.hit_test((800.0, 800.0), tolerance_px=30.0)
        self.assertIsNone(miss)

    def test_translation(self):
        """Verify translate updates position accurately following pinch point with offset."""
        # Grab with offset (-10, -5)
        drag_offset = (-10.0, -5.0)
        pinch_pt = (300.0, 350.0)
        new_pos = Transformations.translate(self.test_obj, pinch_pt, drag_offset)
        
        self.assertEqual(new_pos, (290.0, 345.0))
        self.assertEqual(self.test_obj.position, (290.0, 345.0))

    def test_rotation(self):
        """Verify relative hand angle delta updates object rotation."""
        initial_hand_angle = 45.0
        initial_obj_rotation = 0.0
        current_hand_angle = 90.0  # +45 deg rotation

        new_rot = Transformations.rotate(self.test_obj, current_hand_angle, initial_hand_angle, initial_obj_rotation)
        self.assertEqual(new_rot, 45.0)
        self.assertEqual(self.test_obj.rotation, 45.0)

    def test_scaling(self):
        """Verify inter-hand distance ratio scales object correctly."""
        initial_inter_dist = 200.0
        initial_scale = 1.0
        current_inter_dist = 300.0  # 1.5x scale change

        new_scale = Transformations.scale(self.test_obj, current_inter_dist, initial_inter_dist, initial_scale)
        self.assertEqual(new_scale, 1.5)
        self.assertEqual(self.test_obj.scale, 1.5)

    def test_state_machine_flow(self):
        """Verify StateMachine state transitions."""
        self.assertTrue(self.state_machine.is_idle())
        
        # Transition to SELECTED
        self.state_machine.transition_to(State.SELECTED, object_id=1)
        self.assertTrue(self.state_machine.is_selected())
        self.assertEqual(self.state_machine.active_object_id, 1)

        # Transition to MOVING
        self.state_machine.transition_to(State.MOVING, object_id=1)
        self.assertEqual(self.state_machine.current_state, State.MOVING)

        # Reset to IDLE
        self.state_machine.reset_to_idle()
        self.assertTrue(self.state_machine.is_idle())
        self.assertIsNone(self.state_machine.active_object_id)


if __name__ == '__main__':
    unittest.main()
