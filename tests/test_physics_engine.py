"""Unit tests for Phase 9 Physics Engine (Pymunk)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from shape_recognizer import ShapeType, RecognizedShape
from object_manager import ObjectManager, WorkspaceObject
from physics_engine import PhysicsEngine


class TestPhysicsEngine(unittest.TestCase):

    def setUp(self):
        self.physics = PhysicsEngine(viewport_size=(1280, 720))
        self.manager = ObjectManager()
        
        # Create test object at (640, 200)
        rec = RecognizedShape(shape_type=ShapeType.RECTANGLE, bounding_box=(590, 150, 100, 100), center=(640.0, 200.0))
        self.test_obj = self.manager.create_object_from_shape(rec, raw_points=[])

    def test_gravity_stepping(self):
        """Verify object falls downward under gravity when physics steps across frame iterations."""
        self.physics.register_object(self.test_obj)
        initial_y = self.test_obj.position[1]

        # Step physics over 5 frame iterations
        for _ in range(5):
            self.physics.step(0.033, [self.test_obj])

        # Object y position should increase due to downward gravity
        self.assertGreater(self.test_obj.position[1], initial_y)

    def test_throw_momentum(self):
        """Verify releasing object with hand velocity imparts momentum."""
        self.physics.register_object(self.test_obj)

        hand_velocity = (300.0, -150.0)
        self.physics.release_object_with_throw(self.test_obj, hand_velocity)

        body = self.physics.bodies[self.test_obj.id]
        self.assertGreater(body.velocity.x, 0.0)

    def test_toggle_physics(self):
        """Verify toggling physics simulation on and off."""
        self.assertTrue(self.physics.enabled)
        state = self.physics.toggle_physics()
        self.assertFalse(state)
        self.assertFalse(self.physics.enabled)


if __name__ == '__main__':
    unittest.main()
