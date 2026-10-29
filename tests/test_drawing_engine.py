"""Unit tests for Phase 3 Air Drawing Engine."""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from gesture_detector import FrameGestures, HandGesture, GestureType
from drawing_engine import DrawingEngine, Stroke


class TestDrawingEngine(unittest.TestCase):

    def setUp(self):
        self.engine = DrawingEngine(smoothing_alpha=0.5, min_dist=5.0)

    def test_stroke_creation_and_smoothing(self):
        """Verify stroke points are captured and distance-filtered when POINT gesture is active."""
        # Frame 1: Hand pointing at (100, 100)
        hand = HandGesture(hand_label='Right', gesture=GestureType.POINT, index_tip=(100.0, 100.0), is_point=True)
        frame_g1 = FrameGestures(hands=[hand])
        self.engine.update(frame_g1)

        self.assertTrue(self.engine.is_drawing)
        self.assertIsNotNone(self.engine.current_stroke)
        self.assertEqual(len(self.engine.current_stroke.points), 1)
        self.assertEqual(self.engine.current_stroke.points[0], (100.0, 100.0))

        # Frame 2: Hand moves to (120, 100) (dist 20px > min_dist 5px)
        hand2 = HandGesture(hand_label='Right', gesture=GestureType.POINT, index_tip=(120.0, 100.0), is_point=True)
        frame_g2 = FrameGestures(hands=[hand2])
        self.engine.update(frame_g2)

        self.assertEqual(len(self.engine.current_stroke.points), 2)

    def test_jitter_filtering(self):
        """Verify tiny jitter moves (< min_dist) are filtered out."""
        hand = HandGesture(hand_label='Right', gesture=GestureType.POINT, index_tip=(100.0, 100.0), is_point=True)
        self.engine.update(FrameGestures(hands=[hand]))

        # Tiny move (101, 100) -> distance = 1px < min_dist (5px)
        hand_jitter = HandGesture(hand_label='Right', gesture=GestureType.POINT, index_tip=(101.0, 100.0), is_point=True)
        self.engine.update(FrameGestures(hands=[hand_jitter]))

        # Points count should remain 1 because jitter move was < 5px
        self.assertEqual(len(self.engine.current_stroke.points), 1)

    def test_stroke_freezing_on_gesture_change(self):
        """Verify stroke is finalized and moved to completed_strokes when gesture changes away from POINT."""
        # Add 6 distinct points while POINT gesture is active
        for i in range(6):
            pt = (100.0 + i * 20.0, 100.0)
            h = HandGesture(hand_label='Right', gesture=GestureType.POINT, index_tip=pt, is_point=True)
            self.engine.update(FrameGestures(hands=[h]))

        self.assertIsNotNone(self.engine.current_stroke)
        self.assertEqual(len(self.engine.completed_strokes), 0)

        # Transition away to OPEN_PALM
        h_open = HandGesture(hand_label='Right', gesture=GestureType.OPEN_PALM, is_open_palm=True)
        newly_completed = self.engine.update(FrameGestures(hands=[h_open]))

        self.assertFalse(self.engine.is_drawing)
        self.assertIsNone(self.engine.current_stroke)
        self.assertIsNotNone(newly_completed)
        self.assertEqual(len(self.engine.completed_strokes), 1)

    def test_clear_and_undo(self):
        """Verify clear_all and undo_last functions."""
        # Build completed stroke
        for i in range(6):
            pt = (100.0 + i * 20.0, 100.0)
            self.engine.update(FrameGestures(hands=[HandGesture(hand_label='Right', gesture=GestureType.POINT, index_tip=pt, is_point=True)]))
        self.engine.update(FrameGestures(hands=[HandGesture(hand_label='Right', gesture=GestureType.NONE)]))

        self.assertEqual(len(self.engine.completed_strokes), 1)

        # Test Undo
        undone = self.engine.undo_last()
        self.assertIsNotNone(undone)
        self.assertEqual(len(self.engine.completed_strokes), 0)


if __name__ == '__main__':
    unittest.main()
