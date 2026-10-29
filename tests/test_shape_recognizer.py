"""Unit tests for Phase 4 Shape Recognizer."""

import os
import sys
import unittest
import math

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from shape_recognizer import ShapeRecognizer, ShapeType, RecognizedShape


class TestShapeRecognizer(unittest.TestCase):

    def setUp(self):
        self.recognizer = ShapeRecognizer()

    def test_line_recognition(self):
        """Verify open straight line stroke recognition."""
        # Horizontal straight line points from (100, 100) to (300, 100)
        pts = [(100.0 + i * 5.0, 100.0) for i in range(41)]
        res = self.recognizer.recognize(pts)
        self.assertEqual(res.shape_type, ShapeType.LINE)
        self.assertGreaterEqual(res.confidence, 0.7)

    def test_triangle_recognition(self):
        """Verify 3-sided closed polygon recognition."""
        pts = []
        # Side 1: (100, 300) -> (200, 100)
        for i in range(20):
            t = i / 20.0
            pts.append((100.0 + t * 100.0, 300.0 - t * 200.0))
        # Side 2: (200, 100) -> (300, 300)
        for i in range(20):
            t = i / 20.0
            pts.append((200.0 + t * 100.0, 100.0 + t * 200.0))
        # Side 3: (300, 300) -> (100, 300)
        for i in range(20):
            t = i / 20.0
            pts.append((300.0 - t * 200.0, 300.0))

        res = self.recognizer.recognize(pts)
        self.assertEqual(res.shape_type, ShapeType.TRIANGLE)

    def test_rectangle_recognition(self):
        """Verify 4-sided rectangular closed polygon recognition."""
        pts = []
        # Top: (100, 100) -> (300, 100)
        for i in range(20):
            pts.append((100.0 + i * 10.0, 100.0))
        # Right: (300, 100) -> (300, 250)
        for i in range(15):
            pts.append((300.0, 100.0 + i * 10.0))
        # Bottom: (300, 250) -> (100, 250)
        for i in range(20):
            pts.append((300.0 - i * 10.0, 250.0))
        # Left: (100, 250) -> (100, 100)
        for i in range(15):
            pts.append((100.0, 250.0 - i * 10.0))

        res = self.recognizer.recognize(pts)
        self.assertEqual(res.shape_type, ShapeType.RECTANGLE)

    def test_circle_recognition(self):
        """Verify smooth circular closed loop recognition."""
        pts = []
        cx, cy, radius = 300.0, 300.0, 100.0
        for i in range(60):
            angle = (i / 60.0) * 2.0 * math.pi
            pts.append((cx + radius * math.cos(angle), cy + radius * math.sin(angle)))

        res = self.recognizer.recognize(pts)
        self.assertEqual(res.shape_type, ShapeType.CIRCLE)
        self.assertGreaterEqual(res.confidence, 0.6)


if __name__ == '__main__':
    unittest.main()
