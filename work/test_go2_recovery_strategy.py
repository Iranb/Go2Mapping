"""Bounded mid-route obstacle/stall recovery, without robot transport."""
import asyncio
import math
import unittest
from unittest.mock import patch
import numpy as np
import go2_live_demo as app
from go2_navigation import GroundGrid, ARRIVAL
from test_go2_navigation import NavigationDemo, scene
from test_go2_motion_recovery import Clock


class FailedDirectionDemo(NavigationDemo):
    def __init__(self, all_directions=False):
        super().__init__()
        self.all_directions = all_directions

    def move(self, yaw, vx=0., vy=0.):
        before = self.sport['position'][:2].copy()
        heading = self.sport['imu_state']['rpy'][2]
        super().move(yaw, vx, vy)
        if vx and (self.all_directions or
                   (np.linalg.norm(before) <= .30 and abs(heading) < .55)):
            self.sport['position'][:2] = before


class RecoveryStrategyTests(unittest.IsolatedAsyncioTestCase):
    def test_failed_direction_is_not_fabricated_occupancy_and_shortcuts_obey_it(self):
        grid = GroundGrid(scene(), [0, 0, .3], 0.)
        occupied = grid.occupied.copy()
        grid.avoid_failed_departures([{'position': [0., 0.], 'yaw': 0.}])
        plan = grid.plan([1.2, 0.])
        self.assertGreater(max(abs(p[1]) for p in plan['path']), .20)
        self.assertTrue(grid.route_safe(plan['path'], 0.))
        self.assertFalse(grid.route_safe([[0., 0.], [1.2, 0.]], 0.))
        self.assertTrue(grid.segment_safe([0., 0.], [1.2, 0.], 0.))
        np.testing.assert_array_equal(grid.occupied, occupied)

    async def navigate(self, all_directions=False, response_gain=None):
        demo, clock, real_sleep = FailedDirectionDemo(all_directions), Clock(), asyncio.sleep
        if response_gain is not None:
            demo = NavigationDemo()
            original_move = demo.move
            def fast_move(yaw, vx=0., vy=0.):
                original_move(yaw, vx, vy)
                demo.sport['velocity'] = [response_gain*vx, 0., 0.]
            demo.move = fast_move
        with patch.object(app, 'time', clock.namespace()):
            demo.refresh()
            plan = {**GroundGrid(demo.cloud, demo.sport['position'], 0.).plan([1.2, 0.]),
                    'target': [1.2, 0.], 'yaw': 0., 'epoch': 0, 'start': [0., 0.], 'used': True}
            async def tick(seconds):
                clock.value += seconds
                demo.refresh()
                await real_sleep(0)
            with patch.object(app.asyncio, 'sleep', tick):
                await demo.run_navigation('operator', plan)
        return demo

    async def test_stall_parks_changes_departure_then_reaches_goal(self):
        demo = await self.navigate()
        self.assertEqual(demo.phase, 'complete', demo.message)
        self.assertLessEqual(demo.report['distance_m'], ARRIVAL)
        self.assertEqual(len(demo.navigation['failed_departures']), 1)
        self.assertEqual(demo.report['replans'][0]['trigger']['kind'], 'stall')
        self.assertEqual(demo.report['corners'][0]['outcome'], 'stationary')
        self.assertEqual(len(demo.report['recovery_events']), 1)

    async def test_global_nonresponse_stops_after_one_stall_detour(self):
        demo = await self.navigate(all_directions=True)
        self.assertEqual(demo.phase, 'error')
        self.assertIn('No sustained displacement after the detour', demo.message)
        self.assertEqual(len(demo.navigation['failed_departures']), 1)
        self.assertEqual(len(demo.report['replans']), 1)
        self.assertIn('stop_response', demo.report)
        self.assertIn('release_response', demo.report)

    async def test_high_measured_speed_reduces_commands_without_raising_guard(self):
        demo = await self.navigate(response_gain=2.8)
        self.assertEqual(demo.phase, 'complete', demo.message)
        changes = demo.report['speed_feedback_adjustments']
        self.assertTrue(changes)
        self.assertLess(changes[-1]['command_scale'], 1.)
        self.assertGreaterEqual(changes[-1]['command_scale'], .65)
        moving = [s for s in demo.records[0][1] if s['stage']=='moving']
        self.assertLess(moving[-1]['vx'], .12)
        self.assertLessEqual(moving[-1]['velocity'][0], .34)

    async def test_obstacle_wait_needs_two_distinct_scans_and_dwell(self):
        demo, clock, real_sleep = NavigationDemo(), Clock(), asyncio.sleep
        with patch.object(app, 'time', clock.namespace()):
            demo.refresh()
            started, first = clock.value, demo.lidar_time
            async def tick(seconds):
                clock.value += seconds;demo.refresh()
                demo.lidar_time = first+(.2 if clock.value-started < .6 else .6)
                await real_sleep(0)
            with patch.object(app.asyncio, 'sleep', tick):
                _, stamp = await demo.rescan_navigation(dwell=.45, frames=2)
            self.assertGreaterEqual(clock.value-started, .6)
            self.assertGreaterEqual(stamp, first+.6)
        self.assertTrue(all(c['vx']==c['wz']==0 for c in demo.command_samples))


if __name__ == '__main__':
    unittest.main()
