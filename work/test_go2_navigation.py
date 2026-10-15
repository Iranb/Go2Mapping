"""Planner, endpoint concurrency, and closed-loop tests. No robot connection."""
import asyncio
import json
import math
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

import go2_live_demo as app
from go2_navigation import GroundGrid, wrap, ARRIVAL, MAX_SPEED, path_distance
from test_go2_demo import FakeDemo


def scene(wall=None, extent=3):
    floor = [[x*.05-.025, y*.05-.025, -.025]
             for x in range(-int(extent/.05), int(extent/.05)+1)
             for y in range(-int(extent/.05), int(extent/.05)+1)]
    if wall:
        floor += [[x-.025, y-.025, z-.025] for x, y in wall for z in (.15, .25, .4)]
    return np.asarray(floor, np.float32)


class PlannerTests(unittest.TestCase):
    def test_open_ground_and_wall_detour(self):
        cloud = scene(wall=[(.6, y*.05) for y in range(-7, 8)])
        grid = GroundGrid(cloud, [0, 0, .3], 0)
        plan = grid.plan([1.3, 0])
        self.assertGreater(plan['length_m'], 1.3)
        self.assertGreater(max(abs(p[1]) for p in plan['path']), .55)
        self.assertTrue(grid.route_safe(plan['path'], grid.yaw))

    def test_closed_wall_unknown_ground_and_body_footprint(self):
        grid = GroundGrid(scene(wall=[(.6, y*.05) for y in range(-60, 61)]), [0, 0, .3], 0)
        with self.assertRaises(ValueError):
            grid.plan([1.3, 0])
        grid = GroundGrid(scene(extent=.7), [0, 0, .3], 0)
        with self.assertRaises(ValueError):
            grid.plan([1.5, 0])
        grid = GroundGrid(scene(wall=[(1., .25), (1., -.25)]), [0, 0, .3], 0)
        with self.assertRaises(ValueError):
            grid.plan([1., 0])

    def test_invalid_goal_and_no_diagonal_corner_cut(self):
        grid = GroundGrid(scene(), [0, 0, .3], 0)
        for target in ([float('nan'), 0], [float('inf'), 0], [5, 0]):
            with self.assertRaises(ValueError):
                grid.plan(target)
        # Isolated diagonal ground cells cannot form a walkable corridor.
        floor = np.array([[i*.05-.025, i*.05-.025, -.025] for i in range(-40, 41)], np.float32)
        grid = GroundGrid(floor, [0, 0, .3], 0)
        with self.assertRaises(ValueError):
            grid.plan([.8, .8])


class NavigationDemo(FakeDemo):
    def __init__(self, interruption=None, yaw=0):
        super().__init__()
        self.sport['imu_state']['rpy'][2] = yaw
        self.cloud, self.metadata = scene(), {'resolution': .05}
        self.interruption, self.translations, self.grid_calls = interruption, [], 0
        self.records = []

    async def compute(self, function, *args):
        result = await super().compute(function, *args)
        self.refresh()
        return result

    def move(self, yaw, vx=0., vy=0.):
        self.record_command(yaw, vx, vy)
        self.commands.append(('velocity', vx, vy, yaw))
        self.sport['imu_state']['rpy'][2] = wrap(self.sport['imu_state']['rpy'][2]+yaw*.05)
        if vx or vy:
            self.translations.append((vx, vy, yaw))
            heading = self.sport['imu_state']['rpy'][2]
            c, s = math.cos(heading), math.sin(heading)
            self.sport['position'][0] += (c*vx-s*vy)*.05
            self.sport['position'][1] += (s*vx+c*vy)*.05
            if len(self.translations) == 8:
                if self.interruption == 'stop':
                    self.request_stop('test stop')
                elif self.interruption == 'disconnect':
                    self.sockets.pop('operator', None)
                elif self.interruption == 'jump':
                    self.sport['position'][0] += .4
                elif self.interruption == 'tilt':
                    self.sport['imu_state']['rpy'][0] = .3
                elif self.interruption == 'stale':
                    self.connected = False

    def move_navigation(self, yaw, vx):
        self.move(yaw, vx)

    async def rpc(self, topic, api, parameter=None, timeout=1.):
        self.commands.append(('rpc', api))
        if topic == app.RTC_TOPIC['SPORT_MOD'] and api == 1008:
            self.move(parameter['z'], parameter['x'], parameter['y'])
        return {'data': {'data': json.dumps({'name': 'mcf', 'enable': True}), 'header': {'status': {'code': 0}}}}

    async def navigation_grid(self, yaw):
        self.grid_calls += 1
        self.refresh()
        if self.interruption == 'obstacle' and self.grid_calls >= 2:
            self.cloud = scene(wall=[(.7, y*.05) for y in range(-60, 61)])
        grid = GroundGrid(self.cloud, self.sport['position'], yaw)
        return grid, self.lidar_time

    def save_navigation(self, report, samples, commands, grid=None):
        self.records.append((report, samples))


class ControllerTests(unittest.IsolatedAsyncioTestCase):
    async def simulate(self, interruption=None, yaw=0):
        demo = NavigationDemo(interruption, yaw)
        grid = GroundGrid(demo.cloud, demo.sport['position'], yaw)
        plan = {**grid.plan([.6, 0]), 'target': [.6, 0], 'yaw': yaw, 'epoch': 0,
                'start': [0, 0], 'used': True, 'state': 'checking'}
        real_sleep = asyncio.sleep

        async def tick(_):
            demo.refresh()
            await real_sleep(.005 if interruption == "obstacle" else .001)
        demo.refresh()
        with patch.object(app.asyncio, 'sleep', tick):
            await demo.run_navigation('operator', plan)
        return demo

    async def test_arrival_and_forward_heading_alignment(self):
        for yaw in (0, math.pi/2):
            demo = await self.simulate(yaw=yaw)
            self.assertEqual(demo.phase, 'complete', demo.message)
            self.assertLessEqual(demo.report['distance_m'], ARRIVAL)
            self.assertIn('stop_response', demo.report)
            self.assertIn('release_response', demo.report)
            self.assertGreater(len(demo.records[0][1]), 1)
            self.assertLessEqual(max(math.hypot(x, y) for x, y, _ in demo.translations), MAX_SPEED+1e-8)
            if yaw:
                self.assertTrue(all(vy == 0 for _, vy, _ in demo.translations))
                self.assertLess(path_distance(demo.sport['position'][:2], demo.navigation['path']), .05)
                self.assertTrue(any(c[0] == 'velocity' and c[1] == 0 and c[3] for c in demo.commands))

    async def test_stop_feedback_and_operator_loss(self):
        for reason in ('stop', 'disconnect', 'jump', 'tilt', 'stale'):
            demo = await self.simulate(reason)
            self.assertEqual(demo.phase, 'stopped' if reason == 'stop' else 'error', demo.message)
            self.assertEqual(len(demo.translations), 8, reason)
            self.assertIn('stop_response', demo.report)

    async def test_new_obstacle_stops_before_arrival(self):
        demo = await self.simulate('obstacle')
        self.assertEqual(demo.phase, 'error')
        self.assertIn('new observations', demo.message)
        self.assertGreater(demo.report['distance_m'], ARRIVAL)
        self.assertIn('stop_response', demo.report)


class EndpointTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.site_patch = patch.object(app, 'SITE', Path(self.folder.name))
        self.site_patch.start()
        self.demo = NavigationDemo()
        self.executions = 0
        self.release = asyncio.Event()

        async def wait_turn(owner, channel='sport'):
            self.executions += 1
            await self.release.wait()
        self.demo.run_motion = wait_turn
        server = web.Application()
        for path, handler in [('/api/start', self.demo.start_handler), ('/api/turn/rearm', self.demo.rearm_handler),
                              ('/api/plan', self.demo.plan_handler), ('/api/navigate', self.demo.navigate_handler),
                              ('/api/stand', self.demo.stand_handler), ('/api/reset', self.demo.reset_handler)]:
            server.router.add_post(path, handler)
        self.client = TestClient(TestServer(server))
        await self.client.start_server()

    async def asyncTearDown(self):
        self.release.set()
        if self.demo.motion_active():
            self.demo.abort.set()
            await self.demo.motion_task
        await self.client.close()
        self.site_patch.stop()
        self.folder.cleanup()

    async def test_repeat_and_delayed_clicks_accept_one_turn_until_explicit_rearm(self):
        token = self.demo.turn_token
        body = {'client_id': 'operator', 'turn_token': token, 'channel': 'joystick'}
        replies = await asyncio.gather(*(self.client.post('/api/start', json=body) for _ in range(12)))
        results = [await r.json() for r in replies]
        self.assertEqual(sum(r['started'] for r in results), 1)
        self.assertEqual(self.executions, 1)
        rearm = await self.client.post('/api/turn/rearm', json=body)
        self.assertEqual(rearm.status, 409)
        self.release.set()
        await self.demo.motion_task
        late = await self.client.post('/api/start', json=body)
        self.assertTrue((await late.json())['duplicate'])
        self.assertEqual(self.executions, 1)
        await self.client.post('/api/reset', json={})
        self.assertTrue(self.demo.turn_latched)
        self.assertTrue(json.loads((Path(self.folder.name)/'turn_state.json').read_text())['latched'])
        self.demo.cloud = scene()
        self.demo.clearance = .6
        self.demo.refresh()
        rearm = await self.client.post('/api/turn/rearm', json=body)
        self.assertEqual(rearm.status, 200)
        old = await self.client.post('/api/start', json=body)
        self.assertEqual(old.status, 409)
        new = await self.client.post('/api/start', json={**body, 'turn_token': self.demo.turn_token})
        self.assertEqual(new.status, 200)
        await self.demo.motion_task
        self.assertEqual(self.executions, 2)

    async def test_turn_captures_camera_settings_and_rejects_invalid_settings_before_motion(self):
        body = {'client_id': 'operator', 'turn_token': self.demo.turn_token}
        invalid = await self.client.post('/api/start', json={**body, 'camera': {'hfov_deg': float('nan')}})
        self.assertEqual(invalid.status, 400)
        self.assertFalse(self.demo.turn_latched)
        self.assertEqual(self.executions, 0)
        valid = await self.client.post('/api/start', json={**body, 'camera': {'hfov_deg': 112., 'pitch_deg': 4.}})
        self.assertEqual(valid.status, 200)
        self.assertEqual(self.demo.fusion_camera.hfov_deg, 112.)
        self.assertEqual(self.demo.fusion_camera.pitch_deg, 4.)
        self.assertEqual(self.executions, 1)

    async def test_slow_turn_state_storage_keeps_sensor_feedback_live(self):
        save = self.demo.save_turn_state
        saving, feedback_during_save = False, 0
        guard_errors = []

        def slow_save():
            nonlocal saving
            saving = True
            try:
                time.sleep(.4)
                save()
            finally:
                saving = False

        async def feedback():
            nonlocal feedback_during_save
            while True:
                self.demo.refresh()
                if saving:
                    feedback_during_save += 1
                await asyncio.sleep(.01)

        async def checked_turn(owner):
            try:
                self.demo.owner = owner
                self.demo.guard()
                self.executions += 1
            except RuntimeError as error:
                guard_errors.append(str(error))

        self.demo.run_motion = checked_turn
        stream = asyncio.create_task(feedback())
        try:
            body = {'client_id': 'operator', 'turn_token': self.demo.turn_token}
            with patch.object(self.demo, 'save_turn_state', slow_save):
                response = await self.client.post('/api/start', json=body)
                self.assertEqual(response.status, 200)
                await self.demo.motion_task
                self.assertGreater(feedback_during_save, 5)
                self.assertEqual(guard_errors, [])
                self.assertEqual(self.executions, 1)
                feedback_during_save = 0
                response = await self.client.post('/api/turn/rearm', json=body)
                self.assertEqual(response.status, 200)
                self.assertGreater(feedback_during_save, 5)
                self.assertFalse(self.demo.turn_latched)
        finally:
            stream.cancel()
            await asyncio.gather(stream, return_exceptions=True)

    async def test_plan_then_single_navigation_with_shared_motion_lock(self):
        response = await self.client.post('/api/plan', json={'client_id': 'operator', 'target': [.6, 0]})
        self.assertEqual(response.status, 200, await response.text())
        plan = await response.json()
        self.assertTrue(plan['reachable'])
        runs = []

        async def wait_nav(owner, plan):
            runs.append(plan['id'])
            await self.release.wait()
        self.demo.run_navigation = wait_nav
        body = {'client_id': 'operator', 'plan_id': plan['id']}
        replies = await asyncio.gather(*(self.client.post('/api/navigate', json=body) for _ in range(8)))
        self.assertEqual(sum(r.status == 200 for r in replies), 1)
        self.assertEqual(runs, [plan['id']])
        rotate = await self.client.post('/api/start', json={'client_id': 'operator', 'turn_token': self.demo.turn_token})
        self.assertEqual(rotate.status, 409)
        stand = await self.client.post('/api/stand', json={'client_id': 'operator'})
        self.assertEqual(stand.status, 409)
        self.release.set()
        await self.demo.motion_task
        late = await self.client.post('/api/navigate', json=body)
        self.assertEqual(late.status, 409)

    async def test_expired_reconnected_moved_and_other_operator_cannot_start(self):
        for condition in ('expired', 'reconnected', 'moved', 'other'):
            self.demo.refresh()
            response = await self.client.post('/api/plan', json={'client_id': 'operator', 'target': [.6, 0]})
            plan = await response.json()
            self.assertTrue(plan['reachable'])
            owner = 'operator'
            if condition == 'expired':
                self.demo.navigation['expires_at'] = time.monotonic()-1
            elif condition == 'reconnected':
                self.demo.connection_epoch += 1
            elif condition == 'moved':
                self.demo.sport['position'][0] += .2
            else:
                self.demo.sockets['other'] = {'heartbeat': time.monotonic()}
                owner = 'other'
            result = await self.client.post('/api/navigate', json={'client_id': owner, 'plan_id': plan['id']})
            self.assertEqual(result.status, 409, condition)
            self.demo.sport['position'][0] = 0
        self.assertEqual(self.demo.translations, [])

    async def test_invalid_targets_reject_and_unknown_goal_only_offers_checked_approach(self):
        for target in ([float('nan'), 0], [float('inf'), 0], ['a', 0], [True, 0]):
            r = await self.client.post('/api/plan', json={'client_id': 'operator', 'target': target})
            self.assertEqual(r.status, 400)
        self.demo.cloud = scene(extent=.7)
        r = await self.client.post('/api/plan', json={'client_id': 'operator', 'target': [1.5, 0]})
        plan = await r.json()
        self.assertTrue(plan['reachable'])
        self.assertTrue(plan['partial'])
        self.assertGreater(plan['goal_offset_m'], ARRIVAL)
        grid = GroundGrid(self.demo.cloud, self.demo.sport['position'], 0)
        self.assertTrue(grid.route_safe(plan['path'], 0))
        self.assertEqual(plan['remaining_obstruction']['kind'], 'unknown')
        self.assertEqual(self.demo.translations, [])

    async def test_pose_changes_during_search_do_not_publish_a_stale_preview(self):
        for change in ('position', 'yaw'):
            with self.subTest(change=change):
                self.demo.sport['position'] = [0., 0., .3]
                self.demo.sport['imu_state']['rpy'][2] = 0.
                self.demo.refresh()
                compute = self.demo.compute

                async def changed(function, *args):
                    result = await compute(function, *args)
                    if change == 'position':
                        self.demo.sport['position'][0] += .11
                    else:
                        self.demo.sport['imu_state']['rpy'][2] += .11
                    return result

                with patch.object(self.demo, 'compute', changed):
                    response = await self.client.post('/api/plan', json={
                        'client_id': 'operator', 'target': [.6, 0]})
                self.assertEqual(response.status, 409)
                self.assertIn('during planning', await response.text())
                self.assertIsNone(self.demo.navigation)
                self.assertFalse(self.demo.motion_active())
                self.assertEqual(self.demo.translations, [])

    async def test_reset_while_route_search_runs_cannot_restore_old_preview(self):
        entered, release = asyncio.Event(), asyncio.Event()
        compute = self.demo.compute

        async def delayed(function, *args):
            entered.set()
            await release.wait()
            return await compute(function, *args)

        with patch.object(self.demo, 'compute', delayed):
            pending = asyncio.create_task(self.client.post('/api/plan', json={
                'client_id': 'operator', 'target': [.6, 0]}))
            await entered.wait()
            try:
                reset = await self.client.post('/api/reset', json={})
                self.assertEqual(reset.status, 200)
            finally:
                release.set()
            response = await pending
        self.assertEqual(response.status, 409)
        self.assertIsNone(self.demo.navigation)
        self.assertEqual(self.demo.translations, [])


if __name__ == '__main__':
    unittest.main()
