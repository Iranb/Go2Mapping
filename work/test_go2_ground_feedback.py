"""Ground classification and changing lidar coverage; no robot connection."""
import asyncio
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import go2_live_demo as app
from go2_navigation import GroundGrid, GROUND_TTL, wrap
from test_go2_navigation import NavigationDemo, scene
from test_go2_motion_recovery import Clock


def missing_strip(points):
    centers = points[:, :2]+.025
    return points[~((centers[:, 0] > .45) & (centers[:, 0] < 1.))]


class GroundTests(unittest.TestCase):
    def test_ground_tolerance_does_not_create_obstacle_halo(self):
        cloud = np.vstack((scene(wall=[(.8, 0)]), [[.675, -.025, .025], [.775, .025, .075]]))
        grid = GroundGrid(cloud, [0, 0, .3], 0)
        x, y = grid.cell([.7, 0])
        self.assertTrue(grid.observed[y, x])
        self.assertFalse(grid.occupied[y, x])
        for point in ([.8, 0], [.8, .05]):
            x, y = grid.cell(point)
            self.assertTrue(grid.occupied[y, x])

    def test_short_gap_keeps_measured_ground_but_expires(self):
        first = GroundGrid(scene(), [0, 0, .3], 0, stamp=10.)
        cloud = missing_strip(scene())
        self.assertFalse(GroundGrid(cloud, [0, 0, .3], 0).segment_safe([0, 0], [.6, 0], 0))
        recent = GroundGrid(cloud, [.15, 0, .3], 0, memory=first.ground_memory, stamp=11.)
        self.assertTrue(recent.segment_safe([.15, 0], [.6, 0], 0))
        expired = GroundGrid(cloud, [.15, 0, .3], 0, memory=recent.ground_memory, stamp=10.+GROUND_TTL+.1)
        self.assertEqual(expired.segment_issue([.15, 0], [.6, 0], 0)['kind'], 'unknown')

    def test_new_occupancy_overrides_recent_ground_immediately(self):
        first = GroundGrid(scene(), [0, 0, .3], 0, stamp=10.)
        new = GroundGrid(scene(wall=[(.65, 0)]), [0, 0, .3], 0,
                         memory=first.ground_memory, stamp=10.1)
        issue = new.segment_issue([0, 0], [.6, 0], 0)
        self.assertEqual(issue['kind'], 'occupied')
        self.assertAlmostEqual(issue['point'][0], .65)

    def test_ground_under_feet_is_not_promoted_to_measured_history(self):
        cloud = scene()
        cloud = cloud[np.linalg.norm(cloud[:, :2]+.025, axis=1) > .65]
        first = GroundGrid(cloud, [0, 0, .3], 0, stamp=10.)
        x, y = first.cell([-.3, 0])
        self.assertTrue(first.observed[y, x])
        later = GroundGrid(cloud, [.6, 0, .3], 0, memory=first.ground_memory, stamp=10.2)
        x, y = later.cell([-.3, 0])
        self.assertFalse(np.isfinite(later.ground_memory['seen'][y, x]))
        self.assertTrue(later.inferred_ground[y, x])
        expired = GroundGrid(cloud, [.6, 0, .3], 0, memory=later.ground_memory, stamp=15.1)
        x, y = expired.cell([-.3, 0])
        self.assertFalse(expired.observed[y, x])

    def test_different_floor_does_not_reuse_old_ground(self):
        first = GroundGrid(scene(), [0, 0, .3], 0, stamp=10.)
        cloud = missing_strip(scene())
        cloud[:, 2] -= .15
        later = GroundGrid(cloud, [0, 0, .3], 0, memory=first.ground_memory, stamp=10.1)
        self.assertFalse(later.segment_safe([0, 0], [.6, 0], 0))

    def test_failure_snapshot_preserves_replay_grid(self):
        grid = GroundGrid(scene(wall=[(.65, 0)]), [0, 0, .3], 0)
        issue = grid.segment_issue([0, 0], [.6, 0], 0)
        with tempfile.TemporaryDirectory() as folder, patch.object(app, 'SITE', Path(folder)):
            app.Demo.save_navigation({'route_check': issue}, [], [], grid)
            saved = np.load(next(Path(folder).glob('navigation/*/blocked-grid.npz')))
            np.testing.assert_array_equal(saved['occupied'], grid.occupied)
            np.testing.assert_array_equal(saved['observed'], grid.observed)
            np.testing.assert_array_equal(saved['origin'], grid.origin)


class CoverageDemo(NavigationDemo):
    def __init__(self, clock, scenario):
        super().__init__(yaw=math.pi/2 if scenario.startswith('turn') else 0)
        self.clock, self.scenario, self.changed_at = clock, scenario, None

    async def navigation_grid(self, yaw):
        if self.changed_at is not None:
            elapsed = self.clock.value-self.changed_at
            if self.scenario == 'turn-far-obstacle':
                self.cloud = scene(wall=[(.8, y*.05) for y in range(-60, 61)])
            elif self.scenario == 'turn-near-obstacle':
                self.cloud = scene(wall=[(.3, 0)])
            elif self.scenario == 'permanent-gap' or elapsed < 2.7:
                self.cloud = missing_strip(scene())
            else:
                self.cloud = scene()
        self.refresh()
        return await app.Demo.navigation_grid(self, yaw)


class CoverageControllerTests(unittest.IsolatedAsyncioTestCase):
    async def navigate(self, scenario):
        clock, real_sleep = Clock(), asyncio.sleep
        with patch.object(app, 'time', clock.namespace()):
            demo = CoverageDemo(clock, scenario)
            yaw = demo.sport['imu_state']['rpy'][2]
            grid, _ = await demo.navigation_grid(yaw)
            plan = {**grid.plan([.6, 0]), 'target': [.6, 0], 'yaw': yaw, 'epoch': 0,
                    'start': [0, 0], 'used': True, 'state': 'checking'}

            async def tick(seconds):
                clock.value += seconds
                demo.refresh()
                changed = (len(demo.translations) >= 8 if not scenario.startswith('turn') else
                           any(c.get('wz') for c in demo.command_samples))
                if changed and demo.changed_at is None:
                    demo.changed_at = clock.value
                await real_sleep(0)

            with patch.object(app.asyncio, 'sleep', tick):
                await demo.run_navigation('operator', plan)
        return demo

    async def test_short_missing_ground_parks_then_resumes_on_new_measurements(self):
        demo = await self.navigate('temporary-gap')
        self.assertEqual(demo.phase, 'complete', demo.message)
        wait = demo.report['ground_waits'][0]
        self.assertEqual(wait['outcome'], 'resumed')
        commands = [c for c in demo.command_samples if wait['started_t'] <= c['t'] < wait['started_t']+wait['duration_s']]
        self.assertGreater(len(commands), 2)
        self.assertTrue(all(c['vx'] == c['vy'] == c['wz'] == 0 for c in commands))
        self.assertNotIn('blocked_at', demo.navigation)

    async def test_persistent_unknown_stops_with_location_and_never_drives_blind(self):
        demo = await self.navigate('permanent-gap')
        self.assertEqual(demo.phase, 'error', demo.message)
        self.assertIn('coverage incomplete', demo.message)
        self.assertEqual(demo.report['route_check']['kind'], 'unknown')
        wait = demo.report['ground_waits'][0]
        self.assertLess(wait['duration_s'], 2.)
        commands = [c for c in demo.command_samples if c['t'] >= wait['started_t']]
        self.assertTrue(all(c['vx'] == c['vy'] == c['wz'] == 0 for c in commands))
        self.assertIn('stop_response', demo.report)

    async def test_distant_route_change_does_not_interrupt_safe_alignment(self):
        demo = await self.navigate('turn-far-obstacle')
        self.assertEqual(demo.phase, 'error', demo.message)
        replan = demo.report['replans'][0]
        self.assertEqual(replan['trigger']['kind'], 'occupied')
        self.assertEqual(replan['trigger']['stage'], 'moving')
        corner = demo.report['corners'][0]
        before_park = [s for s in demo.records[0][1] if s['t'] <= corner['started_t']]
        self.assertTrue(before_park)
        self.assertTrue(all(s['vx'] == 0 for s in before_park))
        grid = GroundGrid(demo.cloud, [0, 0, .3], 0)
        self.assertTrue(grid.route_safe(replan['path'], 0))
        self.assertTrue(grid.pose_safe(demo.sport['position'][:2], demo.sport['imu_state']['rpy'][2]))
        self.assertGreater(demo.report['distance_m'], .2)
        self.assertIn('stop_response', demo.report)

    async def test_near_obstacle_stops_alignment_immediately(self):
        demo = await self.navigate('turn-near-obstacle')
        self.assertEqual(demo.phase, 'error', demo.message)
        self.assertEqual(demo.report['route_check']['stage'], 'aligning')
        self.assertEqual(demo.report['route_check']['kind'], 'occupied')
        self.assertGreater(abs(demo.sport['imu_state']['rpy'][2]), .05)
        self.assertEqual(demo.translations, [])


class TurnDeadbandDemo(NavigationDemo):
    def __init__(self, yaw, threshold=.34):
        super().__init__(yaw=yaw)
        self.threshold = threshold

    def move(self, yaw, vx=0., vy=0.):
        before = self.sport['imu_state']['rpy'][2]
        super().move(yaw, vx, vy)
        measured = math.copysign(.13, yaw) if abs(yaw) >= self.threshold else 0.
        self.sport['imu_state']['rpy'][2] = before+measured*.05
        self.sport['yaw_speed'] = measured


class TurnResponseTests(unittest.IsolatedAsyncioTestCase):
    async def navigate(self, yaw, threshold=.34):
        clock, real_sleep = Clock(), asyncio.sleep
        with patch.object(app, 'time', clock.namespace()):
            demo = TurnDeadbandDemo(yaw, threshold)
            grid = GroundGrid(demo.cloud, demo.sport['position'], yaw)
            plan = {**grid.plan([.6, 0]), 'target': [.6, 0], 'yaw': yaw, 'epoch': 0,
                    'start': [0, 0], 'used': True, 'state': 'checking'}

            async def tick(seconds):
                clock.value += seconds
                demo.refresh()
                await real_sleep(0)

            with patch.object(app.asyncio, 'sleep', tick):
                await demo.run_navigation('operator', plan)
        return demo

    async def test_turn_threshold_does_not_stall_with_fourteen_degrees_remaining(self):
        for yaw in (.25, -1.95, -math.pi):
            demo = await self.navigate(yaw)
            self.assertEqual(demo.phase, 'complete', demo.message)
            first = next(c for c in demo.records[0][1] if c['vx'] > 0)
            self.assertLessEqual(abs(first['heading_error']), .05)
            self.assertLessEqual(max(abs(c['wz']) for c in demo.command_samples), app.TURN_SPEED)
            self.assertTrue(all(c['vy'] == 0 for c in demo.command_samples))

    async def test_no_turn_response_still_stops_without_increasing_maximum(self):
        demo = await self.navigate(.25, threshold=app.TURN_SPEED+.1)
        self.assertEqual(demo.phase, 'error', demo.message)
        self.assertIn('Navigation turn progress stalled', demo.message)
        self.assertEqual(demo.translations, [])
        self.assertLessEqual(max(abs(c['wz']) for c in demo.command_samples), app.TURN_SPEED)
        self.assertIn('stop_response', demo.report)


if __name__ == '__main__':
    unittest.main()
