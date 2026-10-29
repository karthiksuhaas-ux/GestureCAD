"""Unit tests for Phase 8 3D Objects & Renderer3D Pipeline."""

import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from renderer_3d import Renderer3D, ShapeType3D, WorkspaceObject3D
from transformations import Transformations


class TestRenderer3D(unittest.TestCase):

    def setUp(self):
        self.renderer_3d = Renderer3D(focal_length=800.0, camera_center=(640.0, 360.0))

    def test_create_3d_primitives(self):
        """Verify creation of 3D primitive objects (Cube, Sphere, Pyramid)."""
        cube = self.renderer_3d.create_3d_object(ShapeType3D.CUBE)
        sphere = self.renderer_3d.create_3d_object(ShapeType3D.SPHERE)
        pyramid = self.renderer_3d.create_3d_object(ShapeType3D.PYRAMID)

        self.assertEqual(len(self.renderer_3d.objects_3d), 3)
        self.assertEqual(cube.shape_type_3d, ShapeType3D.CUBE)
        self.assertEqual(sphere.shape_type_3d, ShapeType3D.SPHERE)
        self.assertEqual(pyramid.shape_type_3d, ShapeType3D.PYRAMID)

    def test_3d_perspective_projection(self):
        """Verify 3D perspective projection math maps 3D point to 2D screen coordinate."""
        # 3D point at origin relative to camera center (640, 360) at Z depth 800mm
        pt_3d = np.array([0.0, 0.0, 0.0])
        obj_pos = (640.0, 360.0, 800.0)

        proj_2d = self.renderer_3d.project_3d_point(pt_3d, obj_pos)
        self.assertEqual(proj_2d, (640, 360))

    def test_3d_transformations(self):
        """Verify 3D translation (X, Y, Z depth), rotation, and scaling."""
        cube = self.renderer_3d.create_3d_object(ShapeType3D.CUBE, position_3d=(400.0, 300.0, 500.0))

        # Test 3D Translation
        new_pos = Transformations.translate_3d(cube, (450.0, 320.0), palm_size_px=120.0, initial_palm_size=100.0, initial_z=500.0)
        self.assertEqual(new_pos[0], 450.0)
        self.assertEqual(new_pos[1], 320.0)
        self.assertLess(new_pos[2], 500.0) # Larger palm size -> hand closer -> smaller Z depth

        # Test 3D Rotation
        new_rot = Transformations.rotate_3d(cube, current_hand_angle=90.0, initial_hand_angle=45.0, initial_rotation_3d=(0.0, 0.0, 0.0))
        self.assertEqual(new_rot[2], 45.0)

        # Test 3D Scaling
        new_scale = Transformations.scale_3d(cube, current_inter_dist=300.0, initial_inter_dist=200.0, initial_obj_scale=1.0)
        self.assertEqual(new_scale, 1.5)


if __name__ == '__main__':
    unittest.main()
