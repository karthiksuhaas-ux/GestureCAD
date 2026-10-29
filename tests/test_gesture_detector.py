"""Unit tests for Phase 2 Gesture Detector."""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from hand_tracker import SingleHandData
from gesture_detector import GestureDetector, GestureType, HandGesture, FrameGestures


class TestGestureDetector(unittest.TestCase):

    def setUp(self):
        self.detector = GestureDetector()

    def _classify_stable(self, hand):
        """Call classify_hand enough times to satisfy temporal hysteresis."""
        from gesture_detector import CONFIRM_FRAMES
        result = None
        for _ in range(CONFIRM_FRAMES + 1):
            result = self.detector.classify_hand(hand)
        return result

    def _create_synthetic_hand(self, label: str,
                               index_tip_y: int,
                               middle_tip_y: int,
                               ring_tip_y: int,
                               pinky_tip_y: int,
                               thumb_tip_pos: tuple = (100, 250),
                               wrist_pos: tuple = (100, 300),
                               middle_mcp_pos: tuple = (100, 200)) -> SingleHandData:
        """Helper to build a synthetic 21-landmark hand dataset."""
        # 21 default points at wrist
        pts = [wrist_pos] * 21
        
        pts[GestureDetector.WRIST] = wrist_pos
        pts[GestureDetector.MIDDLE_MCP] = middle_mcp_pos # Palm reference = 100px
        
        # Index joints
        pts[GestureDetector.INDEX_MCP] = (80, 200)
        pts[GestureDetector.INDEX_PIP] = (80, 150)
        pts[GestureDetector.INDEX_TIP] = (80, index_tip_y)

        # Middle joints
        pts[GestureDetector.MIDDLE_PIP] = (100, 150)
        pts[GestureDetector.MIDDLE_TIP] = (100, middle_tip_y)

        # Ring joints
        pts[GestureDetector.RING_MCP] = (120, 200)
        pts[GestureDetector.RING_PIP] = (120, 150)
        pts[GestureDetector.RING_TIP] = (120, ring_tip_y)

        # Pinky joints
        pts[GestureDetector.PINKY_MCP] = (140, 200)
        pts[GestureDetector.PINKY_PIP] = (140, 150)
        pts[GestureDetector.PINKY_TIP] = (140, pinky_tip_y)

        # Thumb tip
        pts[GestureDetector.THUMB_TIP] = thumb_tip_pos

        norm_pts = [(x / 640.0, y / 480.0, 0.0) for x, y in pts]

        return SingleHandData(
            label=label,
            score=0.95,
            landmarks_norm=norm_pts,
            landmarks_px=pts
        )

    def test_point_gesture(self):
        """Verify POINT gesture classification when only index finger is extended."""
        hand = self._create_synthetic_hand(
            label='Right',
            index_tip_y=50,   # Extended (tip_y 50 < pip_y 150)
            middle_tip_y=220, # Curled (tip_y 220 > pip_y 150)
            ring_tip_y=220,   # Curled
            pinky_tip_y=220   # Curled
        )
        gesture_res = self._classify_stable(hand)
        self.assertEqual(gesture_res.gesture, GestureType.POINT)
        self.assertTrue(gesture_res.is_point)

    def test_pinch_gesture(self):
        """Verify PINCH gesture classification when thumb & index tips are touching."""
        hand = self._create_synthetic_hand(
            label='Right',
            index_tip_y=120,
            middle_tip_y=220,
            ring_tip_y=220,
            pinky_tip_y=220,
            thumb_tip_pos=(82, 120)  # Distance to index tip (80, 120) is 2px << palm size (100px)
        )
        gesture_res = self._classify_stable(hand)
        self.assertEqual(gesture_res.gesture, GestureType.PINCH)
        self.assertTrue(gesture_res.is_pinch)
        self.assertEqual(gesture_res.pinch_point, (81.0, 120.0))

    def test_open_palm_gesture(self):
        """Verify OPEN_PALM gesture classification when all 4 fingers are extended."""
        hand = self._create_synthetic_hand(
            label='Right',
            index_tip_y=50,
            middle_tip_y=40,
            ring_tip_y=45,
            pinky_tip_y=50
        )
        gesture_res = self._classify_stable(hand)
        self.assertEqual(gesture_res.gesture, GestureType.OPEN_PALM)
        self.assertTrue(gesture_res.is_open_palm)

    def test_fist_gesture(self):
        """Verify FIST gesture classification when all 4 fingers are curled."""
        hand = self._create_synthetic_hand(
            label='Right',
            index_tip_y=220,
            middle_tip_y=220,
            ring_tip_y=220,
            pinky_tip_y=220
        )
        gesture_res = self._classify_stable(hand)
        self.assertEqual(gesture_res.gesture, GestureType.FIST)
        self.assertTrue(gesture_res.is_fist)


    def test_multi_hand_metrics(self):
        """Verify 2-hand distance and angle calculations."""
        hand1 = self._create_synthetic_hand('Right', 50, 220, 220, 220, middle_mcp_pos=(100, 200))
        hand2 = self._create_synthetic_hand('Left', 50, 220, 220, 220, middle_mcp_pos=(400, 200))

        frame_gestures = self.detector.detect([hand1, hand2])
        self.assertEqual(len(frame_gestures.hands), 2)
        self.assertIsNotNone(frame_gestures.inter_hand_distance)
        self.assertIsNotNone(frame_gestures.relative_hand_angle)
        # Distance between index tips (80, 50) and (80, 50) for right and left synthetic hands
        # (80, 50) for both: distance is 0 between index tips, angle is 0
        self.assertAlmostEqual(frame_gestures.inter_hand_distance, 0.0, places=1)


if __name__ == '__main__':
    unittest.main()
