"""Offline geometry, occlusion, storage and dataset-isolation checks."""
import asyncio
import io
import json
import struct
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

import go2_live_demo as app
from go2_photo_depth import (CameraModel, build_fusion, camera_pose, depth_surface,
                            fill_small_holes, project, raster_depth, voxel_support)
from go2_photo_depth import surface_packet
from test_go2_demo import FakeDemo


def photo():
    output=io.BytesIO()
    Image.new('RGB',(160,90),(220,40,20)).save(output,format='JPEG')
    return {'slot':0,'yaw':0.,'position':[0.,0.,0.], 'jpeg':output.getvalue(),
            'saved_monotonic':1.,'historical':True,'span_deg':30}


class DepthGeometryTests(unittest.TestCase):
    def test_optical_axes_and_rotated_camera(self):
        model=CameraModel(offset_x=.3,offset_y=0,offset_z=0)
        frame=photo()
        ids,uv,z=project(np.array([[2.3,0,0],[2.3,.2,0],[2.3,0,.2],[-1,0,0]]),frame,model)
        np.testing.assert_array_equal(ids,[0,1,2])
        np.testing.assert_allclose(uv[0],[79.5,44.5])
        self.assertLess(uv[1,0],uv[0,0])
        self.assertLess(uv[2,1],uv[0,1])
        frame['yaw']=np.pi/2
        ids,uv,z=project(np.array([[0,2.3,0]]),frame,model)
        np.testing.assert_allclose(uv[0],[79.5,44.5])

    def test_depth_zbuffer_and_bounded_voxel_projection(self):
        frame,model=photo(),CameraModel(offset_x=.3,offset_y=0,offset_z=0)
        depth,measured=raster_depth(np.array([[1.3,0,0],[2.3,0,0],[-1,0,0]]),frame,model)
        self.assertEqual(measured.sum(),1)
        self.assertAlmostEqual(float(depth.max()),1.)
        supported=voxel_support(depth,model)
        yy,xx=np.where(supported>0)
        self.assertLessEqual(xx.max()-xx.min(),6)
        self.assertLessEqual(yy.max()-yy.min(),6)
        self.assertGreater(np.count_nonzero(supported),1)
        self.assertEqual(supported[0,0],0)

    def test_only_small_consistent_holes_are_filled(self):
        depth=np.zeros((9,12),np.float32)
        depth[2:7,2:7]=2.;depth[4,4]=0
        filled,mask=fill_small_holes(depth)
        self.assertEqual(filled[4,4],2.)
        self.assertTrue(mask[4,4])
        self.assertEqual(filled[0,0],0)
        depth[4,5]=4.
        filled,mask=fill_small_holes(depth)
        self.assertEqual(filled[4,4],0)
        self.assertFalse(mask[4,4])

    def test_backprojection_uses_full_pose_and_does_not_bridge_depth_edges(self):
        model=CameraModel()
        frame={**photo(),'rpy':[.04,.03,1.2]}
        depth=np.full((90,160),2.,np.float32)
        depth[:,80:]=4.
        vertices,uv,indices=depth_surface(depth,frame,model,60)
        self.assertGreater(len(indices),0)
        self.assertEqual(indices.ndim,1)
        packet=surface_packet(vertices,uv,indices)
        self.assertEqual(struct.unpack('<III',packet[:12]),(1,len(vertices),indices.size))
        self.assertEqual(len(packet),12+len(vertices)*20+indices.size*4)
        origin,orient=camera_pose(frame,model)
        recovered=(vertices-origin)@orient
        self.assertTrue(np.all(np.isclose(recovered[:,2],2.,atol=1e-6)|np.isclose(recovered[:,2],4.,atol=1e-6)))
        triangle_depth=recovered[indices.reshape(-1,3),2]
        self.assertLess(float(np.ptp(triangle_depth,axis=1).max()),1e-5)
        self.assertTrue(np.all((uv>=0)&(uv<=1)))
        # Optical y increases down, while texture v increases up.
        self.assertLess(np.corrcoef(recovered[:,1]/recovered[:,2],uv[:,1])[0,1],-.99)

    def test_world_geometry_color_visibility_and_depth_provenance_exports(self):
        frame,model=photo(),CameraModel(offset_x=.3,offset_y=0,offset_z=0)
        points=np.array([[1.3,0,0],[2.3,0,0]],np.float32)
        frame['depth_points']=points
        frame['rpy']=[0.,0.,0.]
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary)/'model'
            manifest=build_fusion(folder,points,[frame],model,'unit-test',0.)
            self.assertEqual(manifest['colored_points'],1)
            self.assertFalse(manifest['calibrated'])
            self.assertEqual(manifest['frames'][0]['pairing'],'host-time-paired')
            self.assertEqual(manifest['frames'][0]['pose'],'full-rpy')
            data=np.load(folder/'colored_cloud.npz')
            np.testing.assert_array_equal(data['points'],points)
            np.testing.assert_array_equal(data['has_color'],[True,False])
            self.assertGreater(data['colors'][0,0],.8)
            depth=np.load(folder/'frame-00-depth.npz')
            self.assertFalse(np.any(depth['measured']&depth['interpolated']))
            self.assertTrue(np.all(depth['depth_m'][~(depth['measured']|depth['interpolated'])]==0))
            raw=(folder/'colored.bin').read_bytes()
            self.assertEqual(struct.unpack('<II',raw[:8]),(3,1))
            ply=(folder/'colored.ply').read_bytes().split(b'end_header\n',1)
            self.assertIn(b'property uchar has_color',ply[0])
            self.assertEqual(len(ply[1]),len(points)*16)

    def test_session_persists_paired_cloud_without_exposing_arrays_in_manifest(self):
        demo=app.Demo();frame=photo()
        frame['depth_points']=np.array([[1.,2.,3.]],np.float32)
        frame['rpy']=[.01,.02,0.]
        demo.photos={0:frame};demo.metadata={'resolution':.05}
        with tempfile.TemporaryDirectory() as folder:
            demo.session=Path(folder)
            demo.save_session()
            saved=json.loads((Path(folder)/'photos.json').read_text())[0]
            self.assertNotIn('depth_points',saved)
            self.assertEqual(saved['rpy'],frame['rpy'])
            np.testing.assert_array_equal(np.load(Path(folder)/saved['depth_file'])['points'],frame['depth_points'])
            self.assertNotIn('depth_points',demo.photo_manifest()[0])


class FusionServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_offline_build_and_reject_mismatched_live_map(self):
        demo=app.Demo();demo.photos={0:photo()};demo.recorded_map=True
        demo.session=Path('test-session');demo.map_keys=np.array([[20,0,0]],np.int32)
        application=web.Application();application.router.add_post('/build',demo.fusion_handler)
        with tempfile.TemporaryDirectory() as folder,patch.object(app,'SITE',Path(folder)):
            async with TestClient(TestServer(application)) as client:
                response=await client.post('/build',json={'camera':{}})
                self.assertEqual(response.status,200)
                await demo.fusion_task
                self.assertIsNotNone(demo.fusion)
                self.assertFalse(demo.motion_active())
                demo.recorded_map=False
                response=await client.post('/build',json={'camera':{}})
                self.assertEqual(response.status,409)
                response=await client.post('/build',json={'camera':{'hfov_deg':float('nan')}})
                self.assertEqual(response.status,400)

    async def test_reset_during_build_does_not_publish_old_coordinates(self):
        demo=app.Demo();entered=threading.Event();release=threading.Event()
        def build(*args):
            entered.set()
            if not release.wait(3):raise RuntimeError('test barrier timeout')
            return {'id':'old-dataset'}
        with patch.object(app,'build_fusion',build):
            task=asyncio.create_task(demo.run_fusion(np.zeros((1,3)),[photo()],CameraModel(),'old',0,0.))
            self.assertTrue(await asyncio.to_thread(entered.wait,2))
            await demo.reset_map()
            release.set();await task
            self.assertIsNone(demo.fusion)
            self.assertIn('Map updated',demo.fusion_message)


class AutoFusionDemo(FakeDemo):
    def __init__(self, interruption=None, stop_failure=False):
        super().__init__(interruption)
        self.stop_failure = stop_failure
        self.fusion_camera = CameraModel(hfov_deg=110., pitch_deg=3.)

    async def stop_robot(self):
        await super().stop_robot()
        if self.stop_failure:
            self.report['stop_response_error'] = 'stop acknowledgement missing'

    def save_session(self):
        self.metadata = {'resolution': .05}
        self.map_keys = np.array([[20, 0, 6], [30, 0, 6]], np.int32)
        frame = {**photo(), 'historical': False, 'position': list(self.sport['position']),
                 'rpy': [0., 0., 0.], 'depth_points': self.map_keys.astype(np.float32)*.05}
        self.photos = {0: frame}
        app.Demo.save_session(self)


class AutoFusionTests(unittest.IsolatedAsyncioTestCase):
    async def turn(self, demo):
        real_sleep = asyncio.sleep

        async def tick(_):
            demo.refresh()
            await real_sleep(0)

        with patch.object(app.asyncio, 'sleep', tick):
            demo.motion_task = asyncio.create_task(demo.run_motion('operator'))
            await demo.motion_task

    async def test_success_builds_saved_dataset_once_and_manual_clicks_reuse_task(self):
        demo = AutoFusionDemo()
        entered, release = threading.Event(), threading.Event()
        builds = []

        def build(*args):
            self.assertEqual(demo.report['outcome'], 'complete')
            self.assertIn('stop_response', demo.report)
            self.assertIn('release_response', demo.report)
            self.assertTrue((demo.session/'photos.json').is_file())
            self.assertTrue((demo.session/'room_cloud.npz').is_file())
            builds.append(args)
            entered.set()
            if not release.wait(3):
                raise RuntimeError('test barrier timeout')
            return build_fusion(*args)

        with tempfile.TemporaryDirectory() as folder, patch.object(app, 'SITE', Path(folder)), patch.object(app, 'build_fusion', build):
            try:
                await self.turn(demo)
                self.assertEqual(demo.phase, 'complete', demo.message)
                self.assertTrue(await asyncio.to_thread(entered.wait, 2))
                task = demo.fusion_task
                application = web.Application()
                application.router.add_post('/build', demo.fusion_handler)
                async with TestClient(TestServer(application)) as client:
                    responses = await asyncio.gather(*(client.post('/build', json={}) for _ in range(4)))
                    self.assertTrue(all(r.status == 200 for r in responses))
                self.assertIs(demo.fusion_task, task)
            finally:
                release.set()
                if demo.fusion_task:
                    await demo.fusion_task
            self.assertEqual(len(builds), 1)
            self.assertEqual(demo.fusion['dataset_session'], demo.session.name)
            self.assertEqual(demo.fusion['camera']['hfov_deg'], 110.)
            self.assertEqual(demo.fusion['camera']['pitch_deg'], 3.)
            self.assertGreater(demo.fusion['colored_points'], 0)
            self.assertIn('Built', demo.fusion_message)

    async def test_stopped_or_unconfirmed_turn_does_not_build(self):
        for demo in (AutoFusionDemo('stop'), AutoFusionDemo(stop_failure=True)):
            with tempfile.TemporaryDirectory() as folder, patch.object(app, 'SITE', Path(folder)), patch.object(app, 'build_fusion') as build:
                await self.turn(demo)
                self.assertIn(demo.phase, ('stopped', 'error'))
                self.assertIsNone(demo.fusion_task)
                build.assert_not_called()

    async def test_missing_capture_leaves_turn_complete_with_retry_message(self):
        demo = FakeDemo()
        with tempfile.TemporaryDirectory() as folder, patch.object(app, 'SITE', Path(folder)):
            await self.turn(demo)
            self.assertEqual(demo.phase, 'complete', demo.message)
            self.assertIsNone(demo.fusion_task)
            self.assertIn('could not build automatically', demo.fusion_message)
            self.assertIn('retry manually', demo.fusion_message)


if __name__=='__main__':
    unittest.main()
