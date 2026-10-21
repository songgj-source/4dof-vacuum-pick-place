import math
import unittest
from test_unknown_mode import ROOT, LOC
import numpy as np
import tray.tray_capacity as capacity
from tray.tray_capacity import fit_unknown_in_slot, require_slot_capacity, suggest_unknown_turn

class ShapeFitTests(unittest.TestCase):
    def setUp(self):
        self._tight, capacity.ALLOW_TIGHT_SLOT = capacity.ALLOW_TIGHT_SLOT, False   # 여유 검사 자체를 시험한다
        self.addCleanup(setattr, capacity, 'ALLOW_TIGHT_SLOT', self._tight)
        self.depth = np.full((480, 640), 500, np.uint16)
        self.depth[175:275, 160:310] = 490
        self.slot = dict(x_mm=180., y_mm=30., floor_height_mm=10.)
        self.geometry = dict(footprint_polygon_mm=[[-50,-10],[50,-10],[50,10],[-50,10]],
                             footprint_radius_mm=52., height_uncertainty_mm=3.)
        self.pick = [.180, .030]

    def test_rectangle_fits_when_enclosing_circle_does_not(self):
        with self.assertRaises(ValueError):
            require_slot_capacity(self.depth, ROOT, self.slot, 52., locator=LOC)
        fitted, gap = fit_unknown_in_slot(self.depth, ROOT, self.slot, self.geometry, self.pick,
                                          locator=LOC, allow_shift=False)
        self.assertEqual({k: v for k, v in fitted.items() if k != 'tight'}, self.slot)
        self.assertGreaterEqual(gap, 5.)

    def test_rotation_is_accounted_for(self):
        yaw = math.atan2(.030, .180) - math.pi/2
        with self.assertRaises(ValueError):
            fit_unknown_in_slot(self.depth, ROOT, self.slot, self.geometry,
                                [math.cos(yaw)*.2, math.sin(yaw)*.2], locator=LOC, allow_shift=False)

    def test_turn_suggestion_finds_fitting_angle(self):
        yaw = math.atan2(.030, .180) - math.pi/2
        pick = [math.cos(yaw)*.2, math.sin(yaw)*.2]
        found = suggest_unknown_turn(self.depth, ROOT, {1: self.slot}, self.geometry, pick, locator=LOC)
        self.assertEqual([n for n, _, _ in found], [1])
        self.assertEqual(abs(found[0][1]), 90)
        self.assertGreaterEqual(found[0][2], 5.)

    def test_measured_slot_ignores_wall_shadow_but_not_objects(self):
        square = dict(self.geometry, footprint_polygon_mm=[[-30,-30],[30,-30],[30,30],[-30,30]])
        slot = dict(self.slot, axis_xy=[1., 0.])
        self.depth[175:275, 160:185] = 0      # 벽 그림자: 깊이 없음
        fitted, gap = fit_unknown_in_slot(self.depth, ROOT, slot, square, self.pick, locator=LOC, allow_shift=False)
        self.assertGreaterEqual(gap, 5.)
        self.depth[215:235, 225:245] = 460    # 칸 가운데 솟은 물체
        with self.assertRaises(ValueError):
            fit_unknown_in_slot(self.depth, ROOT, slot, square, self.pick, locator=LOC)

    def test_recheck_keeps_slot_rectangle_at_slot_center(self):
        square = dict(self.geometry, footprint_polygon_mm=[[-30,-30],[30,-30],[30,30],[-30,30]])
        slot = dict(self.slot, axis_xy=[1., 0.])
        fitted, gap = fit_unknown_in_slot(self.depth, ROOT, dict(slot, x_mm=slot['x_mm']+8.), square, self.pick,
                                          locator=LOC, allow_shift=False)
        self.assertEqual(fitted['center_mm'], [slot['x_mm']+8., slot['y_mm']])
        _, recheck = fit_unknown_in_slot(self.depth, ROOT, dict(fitted, x_mm=slot['x_mm']+8., center_mm=[slot['x_mm'], slot['y_mm']]),
                                         square, self.pick, locator=LOC, allow_shift=False)
        self.assertLess(recheck, gap)         # 같은 놓기 위치라도 사각형이 칸 중심에 있으면 벽에 8mm 가까워진다

    def test_tight_slot_runs_with_flag_when_allowed(self):
        capacity.ALLOW_TIGHT_SLOT = True
        yaw = math.atan2(.030, .180) - math.pi/2
        fitted, gap = fit_unknown_in_slot(self.depth, ROOT, self.slot, self.geometry,
                                          [math.cos(yaw)*.2, math.sin(yaw)*.2], locator=LOC, allow_shift=False)
        self.assertTrue(fitted['tight'])
        self.assertLess(gap, 5.)

    def test_small_obstacle_inside_footprint_is_not_ignored(self):
        self.depth[202:206, 262:266] = 460
        with self.assertRaises(ValueError):
            fit_unknown_in_slot(self.depth, ROOT, self.slot, self.geometry, self.pick,
                                locator=LOC, allow_shift=False)

    def test_missing_floor_is_rejected(self):
        with self.assertRaises(ValueError):
            fit_unknown_in_slot(self.depth, ROOT, dict(self.slot, floor_height_mm=None),
                                self.geometry, self.pick, locator=LOC)

if __name__ == '__main__':
    unittest.main()
