"""Recorded body occlusion and grid traversal at a measured fractional pose."""
import math
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from go2_navigation import GroundGrid, HEADINGS, SELF_SHADOW_TTL, self_shadow_ground


class LiveGridTests(unittest.TestCase):




    def test_enclosed_shadow_outside_the_standing_envelope_remains_unknown(self):
        observed = np.ones((31, 31), bool)
        observed[15, 26] = False
        shadow = self_shadow_ground(observed, np.zeros_like(observed), np.array([-15,-15]),
                                    np.zeros(2), .05)
        self.assertFalse(shadow.any())

    def test_connected_unknown_and_measured_drop_are_not_filled(self):
        observed = np.ones((21, 21), bool)
        observed[8:14, 8:14] = False
        lower = np.zeros_like(observed)
        origin, start = np.array([-10, -10]), np.zeros(2)
        self.assertEqual(self_shadow_ground(observed, lower, origin, start, .05).sum(), 36)
        lower[12, 12] = True
        self.assertFalse(self_shadow_ground(observed, lower, origin, start, .05).any())
        lower[:] = False
        observed[13, :8] = False
        self.assertFalse(self_shadow_ground(observed, lower, origin, start, .05).any())



if __name__ == '__main__':
    unittest.main()
