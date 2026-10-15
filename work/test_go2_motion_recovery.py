"""Feedback interruptions and motion response regressions; no robot connection."""
import asyncio
import json
import math
import multiprocessing
import tempfile
import time
import types
import unittest
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from unittest.mock import patch

import numpy as np

import go2_live_demo as app
from go2_demo_core import YawCommand, map_frame
from go2_navigation import GroundGrid, MAX_SPEED, wrap
from test_go2_demo import FakeDemo
from test_go2_navigation import NavigationDemo, scene


class Clock:
    def __init__(self):
        self.value = 1000.

    def monotonic(self):
        return self.value

    def namespace(self):
        return types.SimpleNamespace(monotonic=self.monotonic, time=time.time)


class RecoveryTests(unittest.IsolatedAsyncioTestCase):
    async def turn(self, interruption):
        clock = Clock()
        real_sleep = asyncio.sleep
        triggered, until = False, 0.
        with tempfile.TemporaryDirectory() as folder, patch.object(app, 'SITE', Path(folder)), patch.object(app, 'time', clock.namespace()):
            demo = FakeDemo()

            async def tick(seconds):
                nonlocal triggered, until
                clock.value += seconds
                if demo.tracker and demo.tracker.degrees >= 100 and not triggered:
                    triggered, until = True, clock.value+.6
                waiting = triggered and clock.value < until
                if waiting and interruption in ('transient', 'persistent', 'stop', 'disconnect', 'epoch', 'obstacle', 'tilt'):
                    demo.state_time, demo.lidar_time, demo.geometry_time = clock.value-.4, clock.value-.6, clock.value-.9
                else:
                    demo.refresh()
                if triggered and interruption == 'persistent':
                    demo.state_time = clock.value-.4
                if waiting and interruption == 'speed':
                    demo.sport['velocity'] = [.24, 0., 0.]
                else:
                    demo.sport['velocity'] = [0., 0., 0.]
                if demo.phase == 'paused':
                    if interruption == 'stop':
                        demo.request_stop('operator stop during pause')
                    elif interruption == 'disconnect':
                        demo.sockets.pop('operator', None)
                    elif interruption == 'epoch':
                        demo.connection_epoch += 1
                    elif interruption == 'obstacle' and not waiting:
                        demo.clearance = -.01
                    elif interruption == 'tilt' and not waiting:
                        demo.sport['imu_state']['rpy'][0] = .3
                await real_sleep(0)

            with patch.object(app.asyncio, 'sleep', tick):
                await demo.run_motion('operator')
        return demo

    async def test_brief_stream_gap_resumes_remaining_angle_with_zero_commands(self):
        demo = await self.turn('transient')
        self.assertEqual(demo.phase, 'complete', demo.message)
        self.assertAlmostEqual(demo.report['actual_degrees'], 720)
        pause = demo.report['pauses'][0]
        self.assertEqual(pause['outcome'], 'resumed')
        self.assertGreaterEqual(pause['duration_s'], .25)
        commands = [c for c in demo.command_samples if pause['started_t'] <= c['t'] < pause['started_t']+pause['duration_s']]
        self.assertGreater(len(commands), 3)
        self.assertTrue(all(c['vx'] == c['vy'] == c['wz'] == 0 for c in commands))
        self.assertIn('stop_response', demo.report)
        self.assertIn('release_response', demo.report)

    async def test_brief_speed_exceedance_stops_then_resumes_without_raising_limit(self):
        demo = await self.turn('speed')
        self.assertEqual(demo.phase, 'complete', demo.message)
        self.assertIn('speed', demo.report['pauses'][0]['reason'])
        self.assertLessEqual(max(abs(c['wz']) for c in demo.command_samples), app.TURN_SPEED)

    async def test_persistent_gap_ends_after_bounded_wait(self):
        demo = await self.turn('persistent')
        self.assertEqual(demo.phase, 'error')
        self.assertIn('3 seconds', demo.message)
        self.assertEqual(demo.report['pauses'][0]['outcome'], 'stopped')
        self.assertAlmostEqual(demo.report['actual_degrees'], 100)
        self.assertIn('stop_response', demo.report)

    async def test_stop_browser_loss_reconnection_and_obstacle_prevent_resume(self):
        for reason in ('stop', 'disconnect', 'epoch', 'obstacle', 'tilt'):
            demo = await self.turn(reason)
            self.assertEqual(demo.phase, 'stopped' if reason == 'stop' else 'error', (reason, demo.message))
            self.assertAlmostEqual(demo.report['actual_degrees'], 100)
            self.assertEqual(demo.report['pauses'][0]['outcome'], 'stopped')
            self.assertIn('stop_response', demo.report)

    async def test_missing_new_frames_prevents_resume_and_pause_time_is_bounded(self):
        demo = FakeDemo()
        demo.owner = 'operator'
        demo.report = {'pauses': []}
        demo.state_time -= .4
        clock, real_sleep = Clock(), asyncio.sleep
        ticks = 0
        async def tick(seconds):
            nonlocal ticks
            ticks += 1
            clock.value += seconds
            demo.refresh()
            # Receiving one old burst cannot qualify as three continuing frames.
            if ticks > 1:
                demo.sport_sequence -= 1
            await real_sleep(0)
        with patch.object(app, 'time', clock.namespace()), patch.object(app.asyncio, 'sleep', tick):
            demo.state_time = clock.value-.4
            with self.assertRaisesRegex(RuntimeError, '3 seconds'):
                await demo.motion_feedback()

    async def test_recovered_gaps_do_not_exhaust_a_lifetime_pause_budget(self):
        demo, clock, real_sleep = FakeDemo(), Clock(), asyncio.sleep
        demo.owner = 'operator'
        demo.report = {'pauses': [{'duration_s': 1., 'outcome': 'resumed'} for _ in range(12)]}
        async def tick(seconds):
            clock.value += seconds
            demo.refresh()
            await real_sleep(0)
        with patch.object(app, 'time', clock.namespace()), patch.object(app.asyncio, 'sleep', tick):
            demo.refresh()
            elapsed = await demo.motion_feedback(reason='Brief feedback interruption')
        self.assertGreaterEqual(elapsed, .25)
        self.assertEqual(demo.report['pauses'][-1]['outcome'], 'resumed')
        self.assertGreater(demo.report['paused_s'], 12.)

    async def test_duplicate_or_older_source_stamps_do_not_refresh_pose(self):
        clock = Clock()
        with patch.object(app, 'time', clock.namespace()):
            demo = FakeDemo()
            data = {**demo.sport, 'stamp': {'sec': 1, 'nanosec': 50000000}}
            demo.sport_callback({'data': data})
            stamp, sequence = demo.state_time, demo.sport_sequence
            clock.value += .5
            for nanos in (50000000, 40000000, 50000000):
                demo.sport_callback({'data': {**data, 'stamp': {'sec': 1, 'nanosec': nanos}}})
            self.assertEqual(demo.state_time, stamp)
            self.assertEqual(demo.sport_sequence, sequence)
            demo.sport_callback({'data': {**data, 'stamp': {'sec': 1, 'nanosec': 100000000}}})
            self.assertGreater(demo.state_time, stamp)
            self.assertEqual(demo.sport_sequence, sequence+1)

    async def test_saved_map_startup_keeps_controls_idle(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            site = root/'outputs/go2-live-demo'
            site.mkdir(parents=True)
            (site/'turn_state.json').write_text(json.dumps({'token': 'saved-token', 'latched': True}))
            recorded = root/'outputs/go2-room-scan/rotation-area/rotation-full-03'
            recorded.mkdir(parents=True)
            np.savez_compressed(recorded/'after_cloud.npz', points=scene(extent=.8))
            (recorded/'rotation_result.json').write_text(json.dumps({'lidar_metadata': {'resolution': .05}}))
            demo = app.Demo()
            async def no_connection():
                await asyncio.Event().wait()
            demo.connect_loop = no_connection
            with patch.object(app, 'SITE', site), patch.object(app, 'ROOT', root):
                try:
                    await demo.startup(None)
                    await asyncio.sleep(0)
                    self.assertTrue(demo.recorded_map)
                    self.assertGreater(len(demo.map_keys), 0)
                    self.assertTrue(demo.turn_latched)
                    self.assertFalse(demo.motion_active())
                    self.assertEqual(demo.command_samples, [])
                finally:
                    await demo.shutdown(None)

    async def test_worker_process_preserves_map_packet(self):
        demo = FakeDemo()
        demo.compute_pool = ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context('spawn'))
        points = scene(extent=.8)
        args = (np.empty((0, 3), np.int32), points, .05, [0, 0, .3])
        try:
            result = await demo.compute(map_frame, *args)
            expected = map_frame(*args)
            np.testing.assert_array_equal(result[0], expected[0])
            self.assertEqual(result[1:], expected[1:])
            grid = await demo.compute(GroundGrid, scene(), [0, 0, .3], 0)
            plan = await demo.compute(grid.plan, [.6, 0])
            self.assertTrue(grid.route_safe(plan['path'], 0))
        finally:
            await asyncio.to_thread(demo.compute_pool.shutdown, wait=True)


class ForwardDemo(NavigationDemo):
    def __init__(self, deadband=0., yaw=0):
        super().__init__(yaw=yaw)
        self.deadband = deadband
        self.first_forward_yaw = None

    def move(self, yaw, vx=0., vy=0.):
        before = list(self.sport['position'])
        if vx and self.first_forward_yaw is None:
            self.first_forward_yaw = self.sport['imu_state']['rpy'][2]
        super().move(yaw, vx, vy)
        if abs(vx) < self.deadband:
            self.sport['position'][:] = before


class NoisyForwardDemo(ForwardDemo):
    def move(self, yaw, vx=0., vy=0.):
        super().move(yaw, vx, vy)
        if vx and abs(vx) < self.deadband:
            self.sport['position'][1] = .012*math.sin(len(self.translations)*.31)


class ForwardTests(unittest.IsolatedAsyncioTestCase):
    async def navigate(self, deadband, yaw=0, pause=False, obstacle=False, noisy=False, pause_after_max=False):
        clock, real_sleep = Clock(), asyncio.sleep
        triggered, until = False, 0.
        with patch.object(app, 'time', clock.namespace()):
            demo = (NoisyForwardDemo if noisy else ForwardDemo)(deadband, yaw)
            grid = GroundGrid(demo.cloud, demo.sport['position'], yaw)
            plan = {**grid.plan([.6, 0]), 'target': [.6, 0], 'yaw': yaw, 'epoch': 0,
                    'start': [0, 0], 'used': True, 'state': 'checking'}
            async def tick(seconds):
                nonlocal triggered, until
                clock.value += seconds
                reached_speed = demo.translations and demo.translations[-1][0] >= MAX_SPEED-1e-8
                if pause and len(demo.translations) >= 8 and not triggered and (not pause_after_max or reached_speed):
                    triggered, until = True, clock.value+.6
                if triggered and clock.value < until:
                    demo.state_time = clock.value-.4
                    demo.low_time = clock.value-.6
                else:
                    demo.refresh()
                    if triggered and obstacle:
                        demo.cloud = scene(wall=[(.7, y*.05) for y in range(-60, 61)])
                await real_sleep(0)
            with patch.object(app.asyncio, 'sleep', tick):
                await demo.run_navigation('operator', plan)
        return demo

    async def test_deadband_feedback_adjusts_bounded_forward_speed(self):
        demo = await self.navigate(.11)
        self.assertEqual(demo.phase, 'complete', demo.message)
        self.assertGreaterEqual(max(c[0] for c in demo.translations), .12-1e-8)
        self.assertLessEqual(max(c[0] for c in demo.translations), MAX_SPEED)
        self.assertTrue(all(c[1] == 0 for c in demo.translations))
        self.assertIn('velocity', demo.records[0][1][0])
        self.assertIn('vx', demo.records[0][1][0])

    async def test_lateral_jitter_does_not_prevent_effective_startup_adjustment(self):
        demo = await self.navigate(.19, noisy=True)
        self.assertEqual(demo.phase, 'complete', demo.message)
        self.assertEqual([e['to_m_s'] for e in demo.report['forward_adjustments']], [.18, .20])
        self.assertLessEqual(max(c[0] for c in demo.translations), MAX_SPEED)

    async def test_pause_keeps_adjusted_command_and_ramps_again_from_zero(self):
        demo = await self.navigate(.19, pause=True, pause_after_max=True)
        self.assertEqual(demo.phase, 'complete', demo.message)
        pause = demo.report['pauses'][0]
        samples = [s for s in demo.records[0][1] if s['t'] >= pause['started_t']+pause['duration_s']]
        self.assertEqual(samples[0]['speed_limit'], MAX_SPEED)
        self.assertGreater(samples[0]['vx'], 0)
        self.assertLess(samples[0]['vx'], MAX_SPEED)
        self.assertEqual(len(demo.report['forward_adjustments']), 2)

    async def test_sideways_jitter_cannot_hide_forward_stall(self):
        demo = await self.navigate(1., noisy=True)
        self.assertEqual(demo.phase, 'error')
        self.assertIn('displacement', demo.message)
        self.assertLess(demo.report['forward_stall']['progress_m'], .02)
        self.assertLess(demo.report['forward_stall']['duration_s'], 5.1)
        self.assertIn('stop_response', demo.report)

    async def test_turns_before_forward_motion(self):
        demo = await self.navigate(0., yaw=math.pi/2)
        self.assertEqual(demo.phase, 'complete', demo.message)
        heading = next(a['yaw'] for a in demo.report['actions'] if a['type'] == 'walk')
        self.assertLess(abs(wrap(demo.first_forward_yaw-heading)), .06)
        self.assertTrue(all(c[1] == 0 for c in demo.translations))

    async def test_no_response_stops_with_specific_diagnosis(self):
        demo = await self.navigate(1.)
        self.assertEqual(demo.phase, 'error')
        self.assertIn('displacement', demo.message)
        self.assertLessEqual(max(c[0] for c in demo.translations), MAX_SPEED)
        self.assertIn('stop_response', demo.report)

    async def test_navigation_resumes_only_after_rechecking_live_route(self):
        demo = await self.navigate(0., pause=True)
        self.assertEqual(demo.phase, 'complete', demo.message)
        self.assertEqual(demo.report['pauses'][0]['outcome'], 'resumed')
        demo = await self.navigate(0., pause=True, obstacle=True)
        self.assertEqual(demo.phase, 'error', demo.message)
        self.assertIn('new observations', demo.message)
        replan = demo.report['replans'][0]
        self.assertEqual(replan['trigger']['kind'], 'occupied')
        self.assertTrue(replan['partial'])
        corner = demo.report['corners'][0]
        stopped = [c for c in demo.command_samples
                   if corner['started_t'] <= c['t'] < corner['started_t']+corner['duration_s']]
        self.assertTrue(stopped)
        self.assertTrue(all(c['vx'] == c['vy'] == c['wz'] == 0 for c in stopped))
        grid = GroundGrid(demo.cloud, replan['path'][0]+[.3], 0)
        self.assertTrue(grid.route_safe(replan['path'], 0))
        self.assertTrue(grid.pose_safe(demo.sport['position'][:2], 0))
        self.assertGreater(demo.report['distance_m'], .2)
        self.assertIn('stop_response', demo.report)

    async def test_narrow_forward_corridor_does_not_allow_unsafe_turn(self):
        wall = [(x*.05, y) for x in range(-50, 51) for y in (-.35, .35)]
        grid = GroundGrid(scene(wall=wall), [0, 0, .3], 0)
        self.assertTrue(grid.route_safe(grid.plan([.6, 0])['path'], 0))
        self.assertFalse(grid.turn_safe([0, 0], 0, math.pi/2))
        self.assertFalse(grid.pose_safe(grid.world((1, 1)), math.pi/2))

    async def test_angular_ramp_and_feedback_braking(self):
        yaw = YawCommand()
        commands = [yaw.update(.4, 0., .05) for _ in range(20)]
        self.assertLessEqual(max(commands), .4)
        self.assertTrue(all(abs(b-a) <= .040000001 for a, b in zip(commands, commands[1:])))
        self.assertLess(yaw.update(.4, .7, .05), commands[-1])


if __name__ == '__main__':
    unittest.main()
