"""Unit tests for Phase 5 Object Manager & Vector Objects."""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from shape_recognizer import RecognizedShape, ShapeType
from object_manager import ObjectManager, WorkspaceObject


class TestObjectManager(unittest.TestCase):

    def setUp(self):
        self.manager = ObjectManager()

    def test_create_rectangle_object(self):
        """Verify vector RECTANGLE object instantiation from recognized shape."""
        rec = RecognizedShape(
            shape_type=ShapeType.RECTANGLE,
            confidence=0.9,
            bounding_box=(100, 100, 100, 80),
            center=(150.0, 140.0)
        )
        obj = self.manager.create_object_from_shape(rec, raw_points=[])
        self.assertEqual(obj.id, 1)
        self.assertEqual(obj.shape_type, ShapeType.RECTANGLE)
        self.assertEqual(obj.position, (150.0, 140.0))
        self.assertEqual(obj.size, (100, 80))
        self.assertEqual(len(self.manager.objects), 1)

    def test_create_circle_object(self):
        """Verify vector CIRCLE object instantiation with generated smooth vertices."""
        rec = RecognizedShape(
            shape_type=ShapeType.CIRCLE,
            confidence=0.85,
            bounding_box=(200, 200, 100, 100),
            center=(250.0, 250.0)
        )
        obj = self.manager.create_object_from_shape(rec, raw_points=[])
        self.assertEqual(obj.shape_type, ShapeType.CIRCLE)
        self.assertEqual(obj.size, (100.0, 100.0))
        self.assertEqual(len(obj.local_vertices), 32)

    def test_world_vertices_transformation(self):
        """Verify get_world_vertices correctly translates and rotates local points."""
        rec = RecognizedShape(
            shape_type=ShapeType.RECTANGLE,
            bounding_box=(100, 100, 100, 100),
            center=(200.0, 200.0)
        )
        obj = self.manager.create_object_from_shape(rec, raw_points=[])
        
        # Unrotated world vertices at center (200, 200) with size 100x100
        world_pts = obj.get_world_vertices()
        self.assertEqual(len(world_pts), 4)
        # Top-left vertex should be at (200 - 50, 200 - 50) = (150, 150)
        self.assertAlmostEqual(world_pts[0][0], 150.0, places=1)
        self.assertAlmostEqual(world_pts[0][1], 150.0, places=1)

        # Apply 90 degree rotation
        obj.rotation = 90.0
        world_pts_rot = obj.get_world_vertices()
        self.assertEqual(len(world_pts_rot), 4)

    def test_remove_and_clear_objects(self):
        """Verify remove_object and clear_all management functions."""
        rec1 = RecognizedShape(shape_type=ShapeType.LINE, bounding_box=(0, 0, 50, 50), center=(25.0, 25.0))
        rec2 = RecognizedShape(shape_type=ShapeType.TRIANGLE, bounding_box=(100, 100, 50, 50), center=(125.0, 125.0))
        
        obj1 = self.manager.create_object_from_shape(rec1, raw_points=[])
        obj2 = self.manager.create_object_from_shape(rec2, raw_points=[])
        self.assertEqual(len(self.manager.objects), 2)

        removed = self.manager.remove_object(obj1.id)
        self.assertTrue(removed)
        self.assertEqual(len(self.manager.objects), 1)

        self.manager.clear_all()
        self.assertEqual(len(self.manager.objects), 0)


if __name__ == '__main__':
    unittest.main()
