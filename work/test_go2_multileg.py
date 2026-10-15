"""Heading-aware detours and stop/turn/walk execution without a robot."""
import asyncio
import math
import unittest
from unittest.mock import patch

import numpy as np

import go2_live_demo as app
from go2_navigation import GroundGrid, HEADINGS, ARRIVAL, MAX_PATH
from test_go2_navigation import NavigationDemo, scene
from test_go2_motion_recovery import Clock


def corridor(kind='U'):
    cloud = scene()
    x, y = (cloud[:, :2]+.025).T
    bottom = (x >= -.8) & (x <= 1.8) & (np.abs(y) <= .35)
    pocket = (x >= .7) & (x <= 1.8) & (y >= -.55) & (y <= .55)
    vertical = (x >= .9) & (x <= 1.6) & (y >= 0) & (y <= (1.95 if kind == 'L' else 1.6))
    top = (x >= -.8) & (x <= 1.8) & (y >= .9) & (y <= 1.6)
    top_pocket = (x >= .7) & (x <= 1.8) & (y >= .7) & (y <= 1.8)
    keep = bottom | pocket | vertical
    if kind != 'L':
        keep |= top | top_pocket
    return cloud[keep]


class MultiLegPlannerTests(unittest.TestCase):
    def test_goal_can_be_entered_at_a_different_heading(self):
        grid = GroundGrid(corridor('L'), [0, 0, .3], 0)
        goal = [1.25, 1.4]
        self.assertFalse(grid.free(goal))
        plan = grid.plan(goal)
        self.assertTrue(grid.route_safe(plan['path'], 0))
        self.assertGreaterEqual(plan['walk_count'], 2)
        self.assertGreaterEqual(plan['turn_count'], 1)
        self.assertLess(np.linalg.norm(np.asarray(plan['path'][-1])-goal), ARRIVAL)

    def test_u_corridor_requires_multiple_checked_turns_and_walks(self):
        grid = GroundGrid(corridor(), [0, 0, .3], 0)
        self.assertFalse(grid.route_safe([[0, 0], [0, 1.25]], 0))
        plan = grid.plan([0, 1.25])
        self.assertTrue(grid.route_safe(plan['path'], 0))
        self.assertGreaterEqual(plan['turn_count'], 2)
        self.assertGreaterEqual(plan['walk_count'], 3)
        self.assertLessEqual(plan['length_m'], MAX_PATH)
        self.assertEqual(sum(a['type'] == 'walk' for a in plan['actions']), plan['walk_count'])
        self.assertEqual(sum(a['type'] == 'turn' for a in plan['actions']), plan['turn_count'])
        for a, b in zip(plan['actions'], plan['actions'][1:]):
            if b['type'] == 'turn':
                np.testing.assert_allclose(a['to'], b['at'])

    def test_fractional_measured_start_keeps_route_in_world_coordinates(self):
        start = [.013, -.017, .3]
        grid = GroundGrid(corridor(), start, 0)
        plan = grid.plan([0, 1.25])
        np.testing.assert_allclose(plan['path'][0], start[:2])
        self.assertTrue(grid.route_safe(plan['path'], 0))
        self.assertLess(np.linalg.norm(np.asarray(plan['path'][-1])-[0, 1.25]), ARRIVAL)

    def test_unknown_gap_cannot_be_used_as_a_shortcut(self):
        cloud = corridor()
        y = cloud[:, 1]+.025
        grid = GroundGrid(cloud[(y < .45) | (y > .9)], [0, 0, .3], 0)
        with self.assertRaises(ValueError):
            grid.plan([0, 1.25])

    def test_asymmetric_turn_mask_matches_direct_head_clearance(self):
        for yaw in (0., math.pi):
            grid = GroundGrid(scene(wall=[(.3, 0)]), [0, 0, .3], yaw)
            heading = HEADINGS.index(yaw)
            x, y = grid.cell([0, 0])
            direct = grid.turn_safe([0, 0], yaw, yaw)
            self.assertEqual(not grid.transition_mask(len(HEADINGS), heading)[y, x], direct)
            self.assertEqual(direct, yaw == math.pi)


class CorridorDemo(NavigationDemo):
    def __init__(self):
        super().__init__()
        self.cloud = corridor()


class MultiLegControllerTests(unittest.IsolatedAsyncioTestCase):
    async def turn_obstacle_case(self, sealed=False):
        demo, clock, real_sleep = NavigationDemo(), Clock(), asyncio.sleep
        original_grid = demo.navigation_grid
        appeared = False

        async def navigation_grid(yaw):
            nonlocal appeared
            if yaw > .5:
                appeared = True
            if appeared:
                wall = [(.15, .35)]
                if sealed:
                    wall += [(x*.05, .45) for x in range(-60, 61)]
                demo.cloud = scene(wall=wall)
            return await original_grid(yaw)

        demo.navigation_grid = navigation_grid
        with patch.object(app, 'time', clock.namespace()):
            demo.refresh()
            route = GroundGrid(demo.cloud, demo.sport['position'], 0).plan([0, 1.5])
            plan = {**route, 'target': [0, 1.5], 'yaw': 0, 'epoch': 0,
                    'start': [0, 0], 'used': True, 'state': 'checking'}

            async def tick(seconds):
                clock.value += seconds
                demo.refresh()
                await real_sleep(0)

            with patch.object(app.asyncio, 'sleep', tick):
                await demo.run_navigation('operator', plan)
        self.assertTrue(appeared)
        return demo

    async def test_new_obstacle_during_turn_parks_then_checks_an_alternative(self):
        demo = await self.turn_obstacle_case()
        self.assertEqual(demo.phase, 'complete', demo.message)
        self.assertLessEqual(demo.report['distance_m'], ARRIVAL)
        replanned = demo.report['replans'][0]
        self.assertEqual(replanned['trigger']['stage'], 'aligning')
        self.assertEqual(replanned['trigger']['kind'], 'occupied')
        self.assertEqual(demo.report['corners'][0]['outcome'], 'stationary')
        check = GroundGrid(demo.cloud, [*replanned['path'][0], .3],
                           replanned['trigger']['yaw'])
        self.assertTrue(check.route_safe(replanned['path'], check.yaw))
        self.assertTrue(any(abs(p[0]) > .3 for p in replanned['path']))
        self.assertIn('stop_response', demo.report)

    async def test_blocked_turn_without_an_alternative_stops_without_false_arrival(self):
        demo = await self.turn_obstacle_case(sealed=True)
        self.assertEqual(demo.phase, 'error', demo.message)
        self.assertGreater(demo.report['distance_m'], ARRIVAL)
        self.assertEqual(demo.report['replan_failure']['trigger']['stage'], 'aligning')
        self.assertEqual(demo.report['corners'][0]['outcome'], 'stationary')
        self.assertEqual(demo.translations, [])
        self.assertIn('stop_response', demo.report)
        self.assertIn('release_response', demo.report)

    async def test_rescan_waits_for_a_new_frame_and_only_sends_neutral(self):
        demo, clock, real_sleep = NavigationDemo(), Clock(), asyncio.sleep
        with patch.object(app, 'time', clock.namespace()):
            demo.refresh()
            first_scan, started = demo.lidar_time, clock.value

            async def tick(seconds):
                clock.value += seconds
                demo.refresh()
                if clock.value-started < .25:
                    demo.lidar_time = first_scan
                await real_sleep(0)

            with patch.object(app.asyncio, 'sleep', tick):
                grid, stamp = await demo.rescan_navigation()
        self.assertGreaterEqual(stamp-started, .25)
        self.assertGreater(stamp, first_scan)
        self.assertGreaterEqual(len(demo.command_samples), 5)
        self.assertTrue(all(c['vx'] == c['vy'] == c['wz'] == 0 for c in demo.command_samples))

    async def test_operator_stop_during_rescan_prevents_planning(self):
        demo, clock, real_sleep = NavigationDemo(), Clock(), asyncio.sleep
        with patch.object(app, 'time', clock.namespace()):
            demo.refresh()

            async def tick(seconds):
                clock.value += seconds
                demo.refresh()
                demo.request_stop('operator stop during scan')
                await real_sleep(0)

            with patch.object(app.asyncio, 'sleep', tick):
                with self.assertRaises(asyncio.CancelledError):
                    await demo.rescan_navigation()
        self.assertEqual(demo.grid_calls, 0)
        self.assertEqual(demo.translations, [])

    async def test_missing_new_scan_times_out_within_the_rescan_budget(self):
        demo, clock, real_sleep = NavigationDemo(), Clock(), asyncio.sleep
        with patch.object(app, 'time', clock.namespace()):
            demo.refresh()
            started, first_scan = clock.value, demo.lidar_time

            async def tick(seconds):
                clock.value += seconds
                demo.refresh()
                demo.lidar_time = first_scan
                await real_sleep(0)

            with patch.object(app.asyncio, 'sleep', tick):
                with self.assertRaisesRegex(RuntimeError, 'No fresh lidar scan'):
                    await demo.rescan_navigation()
            self.assertLessEqual(clock.value-started, 1.6)
        self.assertEqual(demo.grid_calls, 0)
        self.assertTrue(all(c['vx'] == c['vy'] == c['wz'] == 0 for c in demo.command_samples))

    async def test_closed_loop_u_route_parks_at_every_bend_and_arrives(self):
        demo, clock, real_sleep = CorridorDemo(), Clock(), asyncio.sleep
        with patch.object(app, 'time', clock.namespace()):
            demo.refresh()
            grid = GroundGrid(demo.cloud, demo.sport['position'], 0)
            route = grid.plan([0, 1.25])
            plan = {**route, 'target': [0, 1.25], 'yaw': 0, 'epoch': 0,
                    'start': [0, 0], 'used': True, 'state': 'checking'}
            async def tick(seconds):
                clock.value += seconds
                demo.refresh()
                await real_sleep(0)
            with patch.object(app.asyncio, 'sleep', tick):
                await demo.run_navigation('operator', plan)
        self.assertEqual(demo.phase, 'complete', demo.message)
        self.assertLessEqual(demo.report['distance_m'], ARRIVAL)
        corners = demo.report['corners']
        self.assertEqual(len(corners), demo.report['walk_count']-1)
        for corner in corners:
            self.assertEqual(corner['outcome'], 'stationary')
            commands = [c for c in demo.command_samples if corner['started_t'] <= c['t'] <
                        corner['started_t']+corner['duration_s']]
            self.assertGreaterEqual(len(commands), 3)
            self.assertTrue(all(c['vx'] == c['vy'] == c['wz'] == 0 for c in commands))
        self.assertEqual(plan['completed_segments'], plan['walk_count'])
        self.assertIn('stop_response', demo.report)
        self.assertTrue(all(vy == 0 for _, vy, _ in demo.translations))

    async def test_new_obstacle_parks_and_replans_a_checked_detour(self):
        demo, clock, real_sleep = NavigationDemo(), Clock(), asyncio.sleep
        original_grid = demo.navigation_grid
        async def navigation_grid(yaw):
            if len(demo.translations) >= 8:
                demo.cloud = scene(wall=[(.8, 0.)])
            return await original_grid(yaw)
        demo.navigation_grid = navigation_grid
        with patch.object(app, 'time', clock.namespace()):
            demo.refresh()
            route = GroundGrid(demo.cloud, demo.sport['position'], 0).plan([1.5, 0])
            plan = {**route, 'target': [1.5, 0], 'yaw': 0, 'epoch': 0,
                    'start': [0, 0], 'used': True, 'state': 'checking'}
            async def tick(seconds):
                clock.value += seconds
                demo.refresh()
                await real_sleep(0)
            with patch.object(app.asyncio, 'sleep', tick):
                await demo.run_navigation('operator', plan)
        self.assertEqual(demo.phase, 'complete', demo.message)
        self.assertEqual(len(demo.report['replans']), 1)
        replanned = demo.report['replans'][0]
        check = GroundGrid(demo.cloud, [*replanned['path'][0], .3], 0.)
        self.assertTrue(check.route_safe(replanned['path'], 0.))
        self.assertGreater(max(abs(p[1]) for p in replanned['path']), .3)
        parked = demo.report['corners'][0]
        commands = [c for c in demo.command_samples if parked['started_t'] <= c['t'] <
                    parked['started_t']+parked['duration_s']]
        self.assertTrue(commands)
        self.assertTrue(all(c['vx'] == c['wz'] == 0. for c in commands))
        self.assertLessEqual(demo.report['distance_m'], ARRIVAL)

    async def test_no_turn_when_corner_cannot_be_confirmed_stationary(self):
        demo, clock, real_sleep = NavigationDemo(), Clock(), asyncio.sleep
        demo.owner, demo.report = 'operator', {}
        demo.sport['velocity'] = [.05, 0., 0.]
        async def tick(seconds):
            clock.value += seconds
            demo.refresh()
            await real_sleep(0)
        with patch.object(app, 'time', clock.namespace()), patch.object(app.asyncio, 'sleep', tick):
            demo.refresh()
            with self.assertRaisesRegex(RuntimeError, 'Corner stop'):
                await demo.park_navigation_corner({}, 1)
        self.assertTrue(all(c['vx'] == c['vy'] == c['wz'] == 0 for c in demo.command_samples))


if __name__ == '__main__':
    unittest.main()
