"""Offline checks; these tests never connect to a robot."""
import asyncio
import json
import math
import struct
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import go2_live_demo as app
from go2_demo_core import TurnProgress, merge_voxels, point_packet, rotation_clearance


class GeometryTests(unittest.TestCase):
    def test_two_turns_across_yaw_wrap(self):
        turn = TurnProgress(math.radians(170))
        for degrees in range(190, 891, 20):
            turn.update(math.atan2(math.sin(math.radians(degrees)), math.cos(math.radians(degrees))))
        self.assertAlmostEqual(turn.degrees, 720)
        turn.update(math.radians(150))
        self.assertAlmostEqual(turn.degrees, 700)

    def test_world_voxels_and_binary_packet(self):
        cloud = np.array([[-.025, -.025, -.025], [.025, -.025, .075], [np.nan, 0, 0]], np.float32)
        keys = merge_voxels(np.empty((0, 3), np.int32), cloud, .05)
        again = merge_voxels(keys, cloud, .05)
        np.testing.assert_array_equal(keys, again)
        self.assertEqual(len(keys), 2)
        np.testing.assert_array_equal(keys, [[0, 0, 0], [1, 0, 2]])
        packet = point_packet(1, keys.astype(np.float32)*.05)
        self.assertEqual(struct.unpack('<II', packet[:8]), (1, 2))
        self.assertEqual(len(packet), 8+24)

    def test_near_obstacle_inside_rotation_envelope(self):
        floor = [[x*.05, y*.05, -.25] for x in range(-5, 6) for y in range(-5, 6)]
        cloud = np.array(floor+[[.35, 0, 0], [.35, 0, -.2]], np.float32)
        gap, ground = rotation_clearance(cloud, np.zeros(3), .05)
        self.assertLess(gap, .02)
        self.assertAlmostEqual(ground, -.25)


class FakeDemo(app.Demo):
    def __init__(self, interruption=None):
        super().__init__()
        self.connected = self.faults_seen = True
        self.clearance = .6
        self.low = {'motor_state': [{'reserve': [0, 0], 'temperature': 40} for _ in range(12)]}
        self.sport = {'position': [0., 0., .3], 'imu_state': {'rpy': [0., 0., 0.]},
                      'velocity': [0., 0., 0.], 'yaw_speed': 0., 'body_height': .30}
        self.sockets = {'operator': {'heartbeat': time.monotonic()}}
        self.commands = []
        self.interruption = interruption
        self.refresh()

    def refresh(self):
        self.state_time = self.lidar_time = self.geometry_time = self.low_time = app.time.monotonic()
        self.sport_sequence += 1
        if 'operator' in self.sockets:
            self.sockets['operator']['heartbeat'] = app.time.monotonic()

    def advance(self, yaw):
        self.commands.append(('move', yaw))
        if not yaw:
            return
        state = {**self.sport, 'imu_state': {'rpy': [0., 0., self.sport['imu_state']['rpy'][2]+math.radians(20)]}}
        self.sport_callback({'data': state})
        if self.tracker.degrees >= 100:
            if self.interruption == 'stop':
                self.request_stop('test stop')
            elif self.interruption == 'disconnect':
                self.sockets.pop('operator', None)

    def move(self, yaw, vx=0., vy=0.):
        self.record_command(yaw, vx, vy)
        self.advance(yaw)

    def move_joystick(self, rx):
        self.joystick_active, self.joystick_rx = True, rx
        self.move(max(-app.TURN_SPEED, min(app.TURN_SPEED, -rx/app.RIGHT_STICK_PER_YAW)))
        self.command_samples[-1].update(channel='joystick', rx=rx)
        if rx:
            self.report['nonzero_commands'] = self.report.get('nonzero_commands', 0)+1

    async def rpc(self, topic, api, parameter=None, timeout=1):
        self.commands.append(('rpc', api))
        if topic == app.RTC_TOPIC['SPORT_MOD'] and api == 1004:
            self.commands.append(('stand', 1004))
            self.sport['body_height'] = .30
            self.sport['imu_state']['rpy'][:2] = [0., 0.]
        if api == 1008:
            self.advance(parameter['z'])
        data = {'name': 'mcf'} if api == 1001 else {'enable': True}
        return {'data': {'data': json.dumps(data), 'header': {'status': {'code': 0}}}}

    def save_session(self):
        self.saved = True


class MotionTests(unittest.IsolatedAsyncioTestCase):
    async def simulate(self, interruption=None):
        demo = FakeDemo(interruption)
        real_sleep = asyncio.sleep

        async def tick(_):
            demo.refresh()
            await real_sleep(0)

        with tempfile.TemporaryDirectory() as folder, patch.object(app, 'SITE', Path(folder)), patch.object(app.asyncio, 'sleep', tick):
            demo.motion_task = asyncio.create_task(demo.run_motion('operator'))
            await demo.motion_task
        return demo

    async def test_measured_720_and_stop_acknowledgements(self):
        demo = await self.simulate()
        self.assertEqual(demo.phase, 'complete')
        self.assertAlmostEqual(demo.report['actual_degrees'], 720)
        self.assertIn('stop_response', demo.report)
        self.assertIn('release_response', demo.report)
        self.assertEqual(demo.commands[-2:], [('rpc', 1003), ('rpc', 1004)])

    async def test_operator_stop_and_lost_browser(self):
        for mode, outcome in [('stop', 'stopped'), ('disconnect', 'error')]:
            demo = await self.simulate(mode)
            self.assertEqual(demo.phase, outcome)
            self.assertAlmostEqual(demo.report['actual_degrees'], 100)
            self.assertEqual(len([c for c in demo.commands if c[0] == 'move' and c[1]]), 5)
            self.assertIn('stop_response', demo.report)

    async def test_slow_session_directory_keeps_sensor_feedback_live(self):
        demo = FakeDemo()
        real_sleep, mkdir = asyncio.sleep, Path.mkdir

        def slow_mkdir(path, *args, **kwargs):
            time.sleep(.4)
            mkdir(path, *args, **kwargs)

        async def tick(_):
            demo.refresh()
            await real_sleep(.001)

        async def feedback():
            while True:
                demo.refresh()
                await real_sleep(.01)

        stream = asyncio.create_task(feedback())
        try:
            with tempfile.TemporaryDirectory() as folder, patch.object(app, 'SITE', Path(folder)), \
                 patch.object(Path, 'mkdir', slow_mkdir), patch.object(app.asyncio, 'sleep', tick):
                await demo.run_motion('operator')
            self.assertEqual(demo.phase, 'complete', demo.message)
            self.assertAlmostEqual(demo.report['actual_degrees'], 720)
            self.assertIn('stop_response', demo.report)
        finally:
            stream.cancel()
            await asyncio.gather(stream, return_exceptions=True)

    async def test_abort_before_first_command(self):
        demo = FakeDemo()
        demo.abort.set()
        await demo.run_motion('operator')
        self.assertEqual(demo.phase, 'stopped')
        self.assertEqual(demo.commands, [])

    async def test_faults_and_tilt_disable_start(self):
        demo = FakeDemo()
        self.assertTrue(demo.status()['ready'])
        demo.fault_callback({'type': 'errors', 'data': [[1, 306, 4]]})
        self.assertFalse(demo.status()['ready'])
        self.assertIn('306-0x4', demo.status()['message'])
        demo.fault_callback({'type': 'rm_error', 'data': [1, 306, 4]})
        self.assertTrue(demo.status()['ready'])
        demo.sport['imu_state']['rpy'][0] = .78
        self.assertFalse(demo.status()['ready'])
        self.assertIn('stable standing pose', demo.status()['message'])

    async def test_prone_stands_before_turning(self):
        demo = FakeDemo()
        demo.sport['body_height'] = .12
        self.assertFalse(demo.status()['ready'])
        self.assertTrue(demo.status()['stand_ready'])
        real_sleep = asyncio.sleep

        async def tick(_):
            demo.refresh()
            await real_sleep(0)

        with patch.object(app.asyncio, 'sleep', tick):
            await demo.run_stand('operator')
        self.assertIn(('stand', 1004), demo.commands)
        self.assertIn('balance_ack', demo.report)
        self.assertEqual(demo.phase, 'idle')
        self.assertFalse(any(c == ('move', .5) for c in demo.commands))

    async def test_joint_fault_blocks_stand(self):
        demo = FakeDemo()
        demo.low['motor_state'][6]['reserve'][0] = 4
        self.assertFalse(demo.status()['stand_ready'])
        await demo.run_stand('operator')
        self.assertNotIn(('stand', 1004), demo.commands)
        self.assertEqual(demo.phase, 'error')


if __name__ == '__main__':
    unittest.main()
