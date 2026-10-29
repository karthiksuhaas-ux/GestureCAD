"""Unit tests for Phase 1 Hand Tracker."""

import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from hand_tracker import HandTracker, SingleHandData


class TestHandTracker(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tracker = HandTracker(max_hands=2, min_detection_confidence=0.5)

    @classmethod
    def tearDownClass(cls):
        cls.tracker.close()

    def test_empty_frame_handling(self):
        """Verify process_frame handles empty or zero-sized frames without raising exceptions."""
        result = self.tracker.process_frame(np.array([]))
        self.assertEqual(result, [])

    def test_blank_image_no_hand(self):
        """Verify process_frame on a black image returns 0 hands cleanly."""
        black_img = np.zeros((480, 640, 3), dtype=np.uint8)
        hands = self.tracker.process_frame(black_img)
        self.assertIsInstance(hands, list)
        self.assertEqual(len(hands), 0)

    def test_draw_landmarks_no_hands(self):
        """Verify draw_landmarks runs cleanly when hand list is empty."""
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        rendered = self.tracker.draw_landmarks(img, [])
        self.assertEqual(rendered.shape, (480, 640, 3))


if __name__ == '__main__':
    unittest.main()
