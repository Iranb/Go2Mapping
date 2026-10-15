"""Spherical geometry, image alignment and scene-isolation checks; no robot connection."""
import asyncio
import json
import math
import unittest

import cv2
import numpy as np
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

import go2_live_demo as app
from go2_panorama import OPTICAL, build_panorama, decode, feature_pairs, orientations, rays, refine
from go2_photo_depth import CameraModel, rotation


def photo(yaw, color=100, roll=0., pitch=0.):
    image = np.full((180, 320, 3), color, np.uint8)
    ok, data = cv2.imencode('.jpg', image)
    assert ok
    return {'slot': round(math.degrees(yaw)/15) % 24, 'version': 1, 'yaw': yaw,
            'rpy': [roll, pitch, yaw], 'position': [0., 0., .3], 'jpeg': data.tobytes()}


def synthetic_photos():
    rng = np.random.default_rng(19)
    texture = cv2.GaussianBlur(rng.integers(0, 256, (600, 1600, 3), dtype=np.uint8), (3, 3), .7)
    for _ in range(250):
        x, y = rng.integers(0, 1600), rng.integers(0, 600)
        cv2.circle(texture, (int(x), int(y)), int(rng.integers(2, 9)), tuple(int(v) for v in rng.integers(20, 240, 3)), -1)
    photos = []
    yy, xx = np.mgrid[:180, :320]
    points = np.column_stack((xx.ravel(), yy.ravel()))
    for i, yaw in enumerate(np.radians([-30, -15, 0, 15, 30])):
        world = rays(points, (180, 320), rotation([0., 0., yaw])@OPTICAL)
        u = (np.arctan2(world[:, 1], world[:, 0]) % (2*math.pi))*1600/(2*math.pi)
        v = (math.pi/2-np.arcsin(world[:, 2]))*600/math.pi
        image = cv2.remap(texture, u.reshape(180, 320).astype(np.float32), v.reshape(180, 320).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        frame = photo(yaw+math.radians((i % 3-1)*2), roll=math.radians((i % 2)*1.2))
        frame['jpeg'] = cv2.imencode('.jpg', image)[1].tobytes()
        photos.append(frame)
    return photos


class PanoramaGeometryTests(unittest.TestCase):
    def test_projection_uses_roll_pitch_and_world_axis(self):
        frame = photo(math.pi/2, roll=.1, pitch=.2)
        orient = orientations([frame], CameraModel(), np.zeros(5))[0]
        ray = rays([[159.5, 89.5]], (180, 320), orient)[0]
        np.testing.assert_allclose(ray, rotation(frame['rpy'])[:, 0], atol=1e-8)
        left = rays([[0, 89.5]], (180, 320), OPTICAL)[0]
        self.assertGreater(left[1], 0)

    def test_feature_alignment_reduces_known_pose_errors(self):
        frames = synthetic_photos()
        images = decode(frames)
        pairs = feature_pairs(images, frames)
        parameters, stats = refine(images, frames, CameraModel(), pairs)
        self.assertGreaterEqual(stats['matched_pairs'], 4)
        self.assertLess(stats['after_px'], stats['before_px']*.5)
        self.assertTrue(np.all(np.isfinite(parameters)))
        self.assertLessEqual(float(np.max(np.abs(parameters[:-2:3]))), math.radians(5)+1e-8)

    def test_periodic_blend_covers_zero_degrees_and_keeps_unseen_space_empty(self):
        result = build_panorama([photo(math.radians(-10), 95), photo(math.radians(10), 140)], width=768)
        image = cv2.imdecode(np.frombuffer(result['image'], np.uint8), cv2.IMREAD_UNCHANGED)
        row = image.shape[0]//2
        self.assertEqual(image.shape, (192, 768, 4))
        self.assertTrue(np.all(image[row, [0, 767], 3] == 255))
        self.assertLess(float(np.max(np.abs(image[row, 0, :3].astype(float)-image[row, -1, :3]))), 4)
        self.assertEqual(int(image[row, 384, 3]), 0)
        self.assertEqual(result['matched_pairs'], 0)
        self.assertIsNone(result['after_px'])

    def test_no_photos_rejected(self):
        with self.assertRaises(ValueError):
            build_panorama([])


class PanoramaServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_async_result_discarded_after_scene_reset(self):
        demo = app.Demo()
        demo.photos = {0: photo(0.), 1: photo(.25)}
        demo.photo_sequence = 2
        entered, release = asyncio.Event(), asyncio.Event()

        async def compute(*args):
            entered.set()
            await release.wait()
            return {'image': b'old-image', 'frames': 2, 'matched_pairs': 1}

        demo.compute = compute
        task = asyncio.create_task(demo.panorama_loop())
        try:
            await entered.wait()
            await demo.reset_map(new_scene=True)
            demo.running = False
            release.set()
            await task
            self.assertIsNone(demo.panorama)
            self.assertNotIn('blended', demo.panorama_message)
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def test_endpoint_returns_only_current_version_and_reset_removes_image(self):
        demo = app.Demo()
        demo.panorama_version = 3
        demo.panorama = {'image': b'current-image'}
        server = web.Application()
        server.router.add_get('/panorama.png', demo.panorama_handler)
        async with TestClient(TestServer(server)) as client:
            old = await client.get('/panorama.png?v=2')
            self.assertEqual(old.status, 404)
            current = await client.get('/panorama.png?v=3')
            self.assertEqual(await current.read(), b'current-image')
            self.assertEqual(current.headers['Cache-Control'], 'no-store')
            await demo.reset_map()
            self.assertEqual((await client.get('/panorama.png?v=3')).status, 404)

    async def test_capture_in_progress_does_not_start_heavy_alignment(self):
        demo = app.Demo()
        demo.photos = {0: photo(0.), 1: photo(.25)}
        demo.photo_sequence = 2
        demo.motion_task = asyncio.create_task(asyncio.Event().wait())
        calls = []

        async def compute(*args):
            calls.append(args)

        demo.compute = compute
        task = asyncio.create_task(demo.panorama_loop())
        await asyncio.sleep(.03)
        task.cancel()
        demo.motion_task.cancel()
        await asyncio.gather(task, demo.motion_task, return_exceptions=True)
        self.assertEqual(calls, [])


if __name__ == '__main__':
    unittest.main()
