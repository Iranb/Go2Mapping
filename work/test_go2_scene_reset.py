"""Scene cache reset, asynchronous results and restart isolation. No robot connection."""
import asyncio
import json
import struct
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

import go2_live_demo as app
from go2_demo_core import point_packet
from go2_navigation import GroundGrid
from go2_photo_depth import CameraModel
from test_go2_navigation import NavigationDemo, scene
from test_go2_photo_depth import photo


class SceneResetTests(unittest.IsolatedAsyncioTestCase):
    async def test_button_endpoint_clears_scene_and_fresh_scan_rebuilds_without_motion(self):
        demo = NavigationDemo()
        demo.map_keys = np.array([[200, 0, 6]], np.int32)
        demo.map_packet = point_packet(1, demo.map_keys.astype(np.float32)*.05)
        demo.live_packet = point_packet(2, demo.map_keys.astype(np.float32)*.05)
        demo.photos = {0: photo()}
        demo.jpeg = b'old-camera'
        demo.ground_memory = {'old': True}
        demo.navigation = {'id': 5, 'used': False}
        demo.session = Path('old-session')
        demo.fusion = {'id': 'old-fusion'}
        demo.turn_latched = True
        token = demo.turn_token
        version = demo.map_version
        server = web.Application()
        server.router.add_post('/api/reset', demo.reset_handler)
        server.router.add_get('/photos/{slot}.jpg', demo.photo_handler)
        with tempfile.TemporaryDirectory() as folder, patch.object(app, 'SITE', Path(folder)):
            async with TestClient(TestServer(server)) as client:
                result = await client.post('/api/reset', json={})
                self.assertEqual(result.status, 200)
                status = (await result.json())['status']
                self.assertEqual(status['map_points'], 0)
                self.assertEqual(status['photo_count'], 0)
                self.assertIsNone(status['fusion_id'])
                self.assertIsNone(status['navigation'])
                self.assertIsNone(demo.ground_memory)
                self.assertIsNone(demo.cloud)
                self.assertIsNone(demo.jpeg)
                self.assertIsNone(demo.session)
                self.assertGreater(demo.map_version, version)
                self.assertEqual(struct.unpack('<II', demo.live_packet), (2, 0))
                self.assertTrue(demo.turn_latched)
                self.assertEqual(demo.turn_token, token)
                self.assertEqual(demo.command_samples, [])
                self.assertEqual((await client.get('/photos/0.jpg')).status, 404)
                self.assertGreater(json.loads((Path(folder)/'scene_state.json').read_text())['started_unix'], 0)
                demo.lidar_callback({'data': {'resolution': .05, 'data': {'points': scene(extent=.8)}}})
                task = asyncio.create_task(demo.map_loop())
                try:
                    for _ in range(100):
                        if len(demo.map_keys):
                            break
                        await asyncio.sleep(.01)
                    self.assertGreater(len(demo.map_keys), 0)
                    self.assertLess(float(np.max(demo.map_keys[:, 0])), 30)
                    self.assertEqual(demo.command_samples, [])
                finally:
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)

    async def test_active_motion_rejects_reset_and_preserves_scene(self):
        demo = NavigationDemo()
        demo.map_keys = np.ones((1, 3), np.int32)
        demo.motion_task = asyncio.create_task(asyncio.Event().wait())
        try:
            with self.assertRaises(web.HTTPConflict):
                await demo.reset_handler(None)
            self.assertEqual(len(demo.map_keys), 1)
            self.assertEqual(demo.scene_generation, 0)
        finally:
            demo.motion_task.cancel()
            await asyncio.gather(demo.motion_task, return_exceptions=True)

    async def test_pending_plan_cannot_restore_old_ground_after_reset(self):
        demo = NavigationDemo()
        entered, release = asyncio.Event(), asyncio.Event()

        async def compute(function, *args):
            entered.set()
            await release.wait()
            return function(*args)

        demo.compute = compute
        task = asyncio.create_task(app.Demo.navigation_grid(demo, 0.))
        await entered.wait()
        await demo.reset_map(new_scene=True)
        release.set()
        with self.assertRaisesRegex(RuntimeError, 'scene changed'):
            await task
        self.assertIsNone(demo.ground_memory)

    async def test_inflight_fusion_cannot_publish_or_replace_reset_message(self):
        for failed in (False, True):
            demo = app.Demo()
            entered, release = threading.Event(), threading.Event()

            def build(*args):
                entered.set()
                if not release.wait(3):
                    raise RuntimeError('test barrier timeout')
                if failed:
                    raise RuntimeError('old build failed')
                return {'id': 'old-fusion'}

            with patch.object(app, 'build_fusion', build):
                task = asyncio.create_task(demo.run_fusion(np.zeros((1, 3)), [photo()], CameraModel(), 'old', 0, 0.))
                self.assertTrue(await asyncio.to_thread(entered.wait, 2))
                await demo.reset_map(new_scene=True)
                message = demo.fusion_message
                release.set()
                await task
                self.assertIsNone(demo.fusion)
                self.assertEqual(demo.fusion_message, message)

    async def test_restart_after_clear_skips_old_preview_and_old_completed_session(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            site = root/'outputs/go2-live-demo'
            saved = site/'sessions/old-session'
            saved.mkdir(parents=True)
            (saved/'result.json').write_text(json.dumps({'outcome': 'complete', 'started_unix': 1.}))
            recorded = root/'outputs/go2-room-scan/rotation-area/rotation-full-03'
            recorded.mkdir(parents=True)
            np.savez_compressed(recorded/'after_cloud.npz', points=scene(extent=.8))
            (recorded/'rotation_result.json').write_text(json.dumps({'lidar_metadata': {'resolution': .05}}))
            with patch.object(app, 'SITE', site), patch.object(app, 'ROOT', root):
                await app.Demo().reset_handler(None)
                demo = app.Demo()

                async def disconnected():
                    await asyncio.Event().wait()

                demo.connect_loop = disconnected
                try:
                    await demo.startup(None)
                    self.assertEqual(len(demo.map_keys), 0)
                    self.assertEqual(demo.photos, {})
                    self.assertIsNone(demo.fusion)
                    self.assertIsNone(demo.session)
                    self.assertFalse(demo.motion_active())
                finally:
                    await demo.shutdown(None)

    async def test_new_completed_capture_can_reload_after_scene_reset(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(app, 'SITE', Path(folder)):
            demo = NavigationDemo()
            await demo.reset_handler(None)
            demo.session = Path(folder)/'sessions/new-session'
            demo.session.mkdir(parents=True)
            demo.photos = {0: photo()}
            demo.map_keys = np.array([[20, 0, 6]], np.int32)
            demo.report = {'outcome': 'complete', 'started_unix': demo.scene_started_unix+1,
                           'before': demo.sport, 'after': demo.sport, 'actual_degrees': 720., 'peak_shift_m': .01}
            app.Demo.save_session(demo)
            restarted = app.Demo()
            restarted.scene_started_unix = demo.scene_started_unix
            restarted.load_saved_scene()
            self.assertEqual(restarted.session, demo.session)
            self.assertEqual(len(restarted.photos), 1)
            self.assertEqual(len(restarted.map_keys), 1)


if __name__ == '__main__':
    unittest.main()
