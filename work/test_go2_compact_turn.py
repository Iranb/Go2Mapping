"""Compact swept turns, joystick wire format, and intermittent feedback recovery."""
import asyncio
import math
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import go2_live_demo as app
from go2_demo_core import rotation_obstacles, swept_clearance
from test_go2_demo import FakeDemo
from test_go2_ground_feedback import TurnDeadbandDemo
from test_go2_motion_recovery import Clock
from test_go2_navigation import NavigationDemo
from go2_navigation import GroundGrid


class CompactGeometryTests(unittest.TestCase):
    def test_turn_point_filter_preserves_full_geometry_for_both_directions(self):
        from go2_demo_core import body_distance
        rng = np.random.default_rng(117)
        for offset in (0., .5):
            centers = rng.uniform(-1.5, 1.5, (1200, 2))+offset
            for angle in (-.5, .18, .5):
                yaw, position = .8, np.array([.1, -.2])
                reference = math.inf
                for heading in np.linspace(yaw, yaw+angle, math.ceil(abs(angle)/.04)+1):
                    c, s = math.cos(heading), math.sin(heading)
                    local = (centers-position)@np.array([[c, -s], [s, c]])
                    reference = min(reference, body_distance(local).min())
                self.assertAlmostEqual(swept_clearance(centers, position, yaw, angle, .05,
                    body_only=True), reference-.05/math.sqrt(2)-.01, places=12)

    def test_turn_point_filter_keeps_near_contact_among_distant_walls(self):
        centers = np.array([[.34, 0.], [1.2, -.8], [-1.1, 1.3], [1., 1.]])
        self.assertLess(swept_clearance(centers, [0, 0], 0., .5, .05, body_only=True), .02)

    def test_body_model_rejects_contact_without_filling_the_stepping_rectangle(self):
        from go2_demo_core import body_distance, BODY_POLYGON
        self.assertLess(body_distance(np.array([[.32, 0]]))[0], 0)
        self.assertLess(body_distance(np.array([[-.18, 0]]))[0], 0)
        self.assertGreater(body_distance(np.array([[-.329, .286]]))[0], .20)
        self.assertLess(np.linalg.norm(BODY_POLYGON, axis=1).max(), .34)
        obstacle = np.array([[-.329, .286]])
        self.assertLess(swept_clearance(obstacle, [0, 0], 0, .18, .05), .02)
        self.assertGreater(swept_clearance(obstacle, [0, 0], 0, .18, .05, body_only=True), .02)
    def test_short_turn_can_pass_obstacle_inside_full_circle(self):
        obstacles = np.array([[0., .36]])
        # The obstacle is inside the complete body-and-feet rotation circle.
        self.assertLess(np.linalg.norm(obstacles[0]), math.hypot(.38, .23))
        self.assertGreater(swept_clearance(obstacles, [0, 0], 0., .18, .05), .02)
        self.assertLess(swept_clearance(obstacles, [0, 0], 0., math.pi/2, .05), .02)

    def test_signed_sweep_rejects_only_the_approaching_corner(self):
        obstacles = np.array([[.33, .29]])
        self.assertGreater(swept_clearance(obstacles, [0, 0], 0., -.3, .05), .02)
        self.assertLess(swept_clearance(obstacles, [0, 0], 0., .3, .05), .02)

    def test_body_and_feet_are_both_kept_without_filled_walking_corners(self):
        from go2_demo_core import turn_distance
        gaps = turn_distance(np.array([[.34, .14], [.28, .22], [.37, .24]]))
        self.assertLess(gaps[0], 0)
        self.assertLess(gaps[1], 0)
        self.assertGreater(gaps[2], .06)

    def test_ground_tolerance_next_to_furniture_is_not_an_obstacle(self):
        floor = [[x*.05-.025, y*.05-.025, -.075] for x in range(-10, 11) for y in range(-10, 11)]
        points = np.array(floor+[[.675, -.025, -.025], [.775, -.025, .075]], np.float32)
        occupied, _ = rotation_obstacles(points, [0, 0, .3], .05)
        self.assertEqual(len(occupied), 1)
        np.testing.assert_allclose(occupied[0], [.8, 0], atol=1e-7)


class CommandTests(unittest.IsolatedAsyncioTestCase):
    async def test_virtual_right_stick_has_no_translation_or_buttons(self):
        demo = FakeDemo()
        sent = []
        demo.conn = types.SimpleNamespace(datachannel=types.SimpleNamespace(
            channel=types.SimpleNamespace(readyState='open'),
            pub_sub=types.SimpleNamespace(publish_without_callback=lambda *args: sent.append(args))))
        app.Demo.move_joystick(demo, -.15)
        self.assertEqual(sent[0], (app.RTC_TOPIC['WIRELESS_CONTROLLER'],
            {'lx': 0., 'ly': 0., 'rx': -.15, 'ry': 0., 'keys': 0}))
        self.assertTrue(demo.joystick_active)
        self.assertEqual(demo.command_samples[-1]['rx'], -.15)

    async def test_sport_stop_or_translation_neutralizes_nonzero_right_stick_first(self):
        demo, sent = FakeDemo(), []
        demo.conn = types.SimpleNamespace(datachannel=types.SimpleNamespace(
            channel=types.SimpleNamespace(readyState='open'),
            pub_sub=types.SimpleNamespace(publish_without_callback=lambda *args: sent.append(args))))
        demo.move_joystick = types.MethodType(app.Demo.move_joystick, demo)
        demo.move_joystick(-.15)
        app.Demo.move(demo, 0., .16)
        self.assertEqual(sent[-2][0], app.RTC_TOPIC['WIRELESS_CONTROLLER'])
        self.assertEqual(sent[-2][1]['rx'], 0.)
        self.assertEqual(sent[-1][0], app.RTC_TOPIC['SPORT_MOD'])

    async def test_stop_retries_only_timed_out_stop_request_and_neutralizes_stick(self):
        demo = FakeDemo()
        calls, sticks = [], []
        async def rpc(topic, api, parameter=None, timeout=1.):
            calls.append((topic, api))
            if len(calls) == 1:
                raise asyncio.TimeoutError()
            return {'code': 0}
        demo.rpc = rpc
        demo.joystick_active = True
        demo.move_joystick = lambda rx: sticks.append(rx)
        real_sleep = asyncio.sleep
        async def tick(_):
            await real_sleep(0)
        with patch.object(app.asyncio, 'sleep', tick):
            await demo.stop_robot()
        self.assertEqual([api for _, api in calls], [1003, 1003, 1004])
        self.assertEqual(sticks, [0.]*8)
        self.assertNotIn('stop_response_error', demo.report)
        self.assertIn('release_response', demo.report)


class SteppingTurnDemo(TurnDeadbandDemo):
    def move(self, yaw, vx=0., vy=0.):
        super().move(yaw, vx, vy)
        # In-place steps can produce small instantaneous translation velocity.
        self.sport['velocity'] = [.05, 0., 0.] if yaw and not vx else [0., 0., 0.]


class ControllerTests(unittest.IsolatedAsyncioTestCase):
    async def test_720_right_stick_uses_measured_yaw_and_stops_at_two_turns(self):
        demo, clock, real_sleep = FakeDemo(), Clock(), asyncio.sleep
        sticks = []
        def stick(rx):
            demo.joystick_active = True
            sticks.append(rx)
            demo.record_command(0.)
            if rx:
                demo.report['nonzero_commands'] += 1
                demo.advance(-rx)
        demo.move_joystick = stick
        async def tick(seconds):
            clock.value += seconds
            demo.refresh()
            await real_sleep(0)
        with tempfile.TemporaryDirectory() as folder, patch.object(app, 'SITE', Path(folder)), \
             patch.object(app, 'time', clock.namespace()), patch.object(app.asyncio, 'sleep', tick):
            demo.refresh()
            await demo.run_motion('operator', 'joystick')
        self.assertEqual(demo.phase, 'complete', demo.message)
        self.assertAlmostEqual(demo.report['actual_degrees'], 720)
        self.assertEqual(sum(bool(rx) for rx in sticks), 36)
        self.assertTrue(all(-app.TURN_SPEED*app.RIGHT_STICK_PER_YAW-1e-8 <= rx < 0 for rx in sticks if rx))
        self.assertEqual(sticks[-8:], [0.]*8)
        self.assertNotIn('move_ack', demo.report)

    async def test_bounded_probe_sign_stop_and_stale_feedback(self):
        for direction, stale in ((1, False), (-1, False), (1, True)):
            demo, clock, real_sleep = NavigationDemo(), Clock(), asyncio.sleep
            sticks = []
            def stick(rx):
                demo.joystick_active = True
                sticks.append(rx)
                demo.record_command(0.)
                demo.sport['imu_state']['rpy'][2] -= rx*.1
                demo.report['nonzero_commands'] += bool(rx)
            demo.move_joystick = stick
            demo.save_turn_probe = lambda *args: None
            async def tick(seconds):
                clock.value += seconds
                demo.refresh()
                if stale and any(sticks):
                    demo.state_time -= .4
                await real_sleep(0)
            with patch.object(app, 'time', clock.namespace()), patch.object(app.asyncio, 'sleep', tick):
                demo.refresh()
                await demo.run_turn_probe('operator', direction, 'joystick')
            self.assertIn('stop_response', demo.report)
            self.assertIn('release_response', demo.report)
            self.assertTrue(all(direction*rx < 0 for rx in sticks if rx))
            self.assertTrue(all(rx == 0 for rx in sticks[-8:]))
            if stale:
                self.assertEqual(demo.report['outcome'], 'error')
                self.assertEqual(sum(bool(rx) for rx in sticks), 1)
            else:
                self.assertEqual(demo.report['outcome'], 'complete')
                self.assertLessEqual(demo.report['nonzero_duration_s'], 1.25)
                self.assertGreater(direction*demo.report['actual_degrees'], 10.)

    async def test_in_place_steps_do_not_repeatedly_cancel_navigation_turn(self):
        clock, real_sleep = Clock(), asyncio.sleep
        with patch.object(app, 'time', clock.namespace()):
            demo = SteppingTurnDemo(.6)
            grid = GroundGrid(demo.cloud, demo.sport['position'], .6)
            plan = {**grid.plan([.6, 0]), 'target': [.6, 0], 'yaw': .6, 'epoch': 0,
                    'start': [0, 0], 'used': True, 'state': 'checking'}
            async def tick(seconds):
                clock.value += seconds
                demo.refresh()
                await real_sleep(0)
            with patch.object(app.asyncio, 'sleep', tick):
                await demo.run_navigation('operator', plan)
        self.assertEqual(demo.phase, 'complete', demo.message)
        self.assertLess(abs(demo.records[0][1][0]['heading_error']), 1.)
        self.assertTrue(any(c['wz'] and np.linalg.norm(c['velocity'][:2]) > .025
                            for c in demo.command_samples))
        turns = [c for c in demo.command_samples if c.get('channel') == 'joystick' and c.get('rx')]
        self.assertTrue(turns)
        first_forward = next(i for i, c in enumerate(demo.command_samples) if c['vx'])
        neutral = max(i for i, c in enumerate(demo.command_samples[:first_forward])
                      if c.get('channel') == 'joystick')
        self.assertEqual(demo.command_samples[neutral]['rx'], 0.)
        self.assertTrue(all(not c['vx'] and not c['wz']
                            for c in demo.command_samples[neutral:first_forward]))


if __name__ == '__main__':
    unittest.main()
