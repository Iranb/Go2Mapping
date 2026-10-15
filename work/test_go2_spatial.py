"""Multiple translated viewpoints, isolated epochs, storage and read-only capture."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
import go2_live_demo as app
from go2_spatial import stations, select_keyframe, save_capture, build_spatial_panorama, MAX_PHOTOS
from go2_photo_depth import CameraModel, build_fusion
from test_go2_photo_depth import photo
from test_go2_navigation import NavigationDemo


def frames():
    a, b = photo(), photo()
    for i, p in enumerate((a,b)):
        p.update(slot=i, position=[float(i),0.,.3], historical=False, connection_epoch=0,
                 rpy=[0.,0.,0.], depth_points=np.array([[2.,0.,.3]], np.float32))
    return [a,b]


class SpatialGeometryTests(unittest.TestCase):
    def test_same_heading_at_different_positions_is_retained(self):
        a,b = frames(); photos={0:a}
        pose={'position':a['position'],'imu_state':{'rpy':[0.,0.,0.]}}
        self.assertIsNone(select_keyframe(photos,pose,0,1.))
        pose['position']=b['position']
        self.assertEqual(select_keyframe(photos,pose,0,1.),1)
        self.assertEqual(len(stations([a,b])),2)

    def test_grouping_does_not_chain_across_space_or_connection_epochs(self):
        photos=[]
        for i,x in enumerate((0.,.3,.6)):
            photos.append({**photo(),'slot':i,'position':[x,0.,0.],'connection_epoch':0})
        groups=stations(photos)
        self.assertEqual([len(g['photos']) for g in groups],[2,1])
        photos.append({**photos[0],'slot':3,'connection_epoch':1})
        self.assertEqual(len(stations(photos)),3)

    def test_capture_is_bounded_without_evicting_existing_views(self):
        photos={i:{**photo(),'slot':i,'position':[float(i),0.,0.]} for i in range(MAX_PHOTOS)}
        pose={'position':[500.,0.,0.],'imu_state':{'rpy':[0.,0.,0.]}}
        self.assertIsNone(select_keyframe(photos,pose,0,1.))
        self.assertEqual(len(photos),MAX_PHOTOS)

    def test_panorama_is_built_separately_for_translated_positions(self):
        photos=frames()
        with patch('go2_panorama.build_panorama',return_value={'image':b'image','matched_pairs':0}) as build:
            manifest=build_spatial_panorama(photos,CameraModel())
        self.assertEqual(manifest['station_count'],2)
        self.assertEqual(build.call_count,2)
        self.assertTrue(all(len(call.args[0])==1 for call in build.call_args_list))
        self.assertNotEqual(manifest['stations'][0]['position'],manifest['stations'][1]['position'])

    def test_surface_span_is_per_position_instead_of_total_photo_count(self):
        photos=[{**photo(),'slot':i,'position':[float(i),0.,0.]} for i in range(24)]
        empty=(np.empty((0,3),np.float32),np.empty((0,2),np.float32),np.empty(0,np.uint32))
        with tempfile.TemporaryDirectory() as temporary, patch('go2_photo_depth.depth_surface',return_value=empty) as surface:
            manifest=build_fusion(Path(temporary)/'dataset',np.array([[2.,0.,0.]],np.float32),photos,CameraModel(),'test',0.)
        self.assertEqual(manifest['station_count'],24)
        self.assertTrue(manifest['spatial'])
        self.assertTrue(all(call.args[3]==45. for call in surface.call_args_list))

    def test_spatial_capture_reloads_pose_and_paired_depth(self):
        photos=frames()
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);folder=root/'spatial'/'20000101-235900-000001'
            save_capture(folder,photos,np.array([[2.,0.,.3]],np.float32),.05)
            with patch.object(app,'SITE',root),patch.object(app,'ROOT',root):
                demo=app.Demo();demo.load_saved_scene()
            self.assertEqual(len(stations(demo.photos.values())),2)
            self.assertTrue(all(p['historical'] for p in demo.photos.values()))
            np.testing.assert_array_equal(demo.photos[1]['depth_points'],photos[1]['depth_points'])


class SpatialServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_capture_start_stop_and_persistence_never_send_motion(self):
        demo=NavigationDemo()
        application=web.Application();application.router.add_post('/capture',demo.spatial_handler)
        with tempfile.TemporaryDirectory() as temporary,patch.object(app,'SITE',Path(temporary)):
            async with TestClient(TestServer(application)) as client:
                demo.refresh()
                response=await client.post('/capture',json={'client_id':'operator','enabled':True})
                self.assertEqual(response.status,200,await response.text())
                self.assertTrue(demo.spatial_capture)
                demo.photos={p['slot']:p for p in frames()}
                response=await client.post('/capture',json={'client_id':'operator','enabled':False})
                self.assertEqual(response.status,200)
                self.assertFalse(demo.spatial_capture)
                self.assertEqual(len(json.loads((demo.session/'photos.json').read_text())),2)
                self.assertEqual(demo.commands,[])

    async def test_station_images_are_versioned_and_never_cross_selected_position(self):
        demo=app.Demo();demo.panorama_version=3
        demo.panorama={'image':b'first','stations':[{'station':0,'image':b'first'},{'station':1,'image':b'second'}]}
        application=web.Application();application.router.add_get('/panorama.png',demo.panorama_handler)
        async with TestClient(TestServer(application)) as client:
            response=await client.get('/panorama.png?v=3&station=1')
            self.assertEqual(await response.read(),b'second')
            self.assertEqual((await client.get('/panorama.png?v=2&station=1')).status,404)
            self.assertEqual((await client.get('/panorama.png?v=3&station=9')).status,404)


if __name__ == '__main__':
    unittest.main()
