"""Offline preview must never create a robot transport or accept controls."""
import unittest
from unittest.mock import patch
from aiohttp.test_utils import TestClient, TestServer
import go2_live_demo
from preview_ui import preview_app


class PreviewTests(unittest.IsolatedAsyncioTestCase):
    async def test_preview_has_synthetic_status_and_rejects_motion(self):
        with patch.object(go2_live_demo, 'UnitreeWebRTCConnection') as transport:
            async with TestClient(TestServer(preview_app())) as client:
                response = await client.get('/api/status')
                status = await response.json()
                self.assertTrue(status['preview'])
                self.assertFalse(status['connected'])
                self.assertFalse(status['active'])
                self.assertEqual(status['map_source'], 'synthetic')
                self.assertGreater(status['map_points'], 1000)
                for endpoint in ('/api/start', '/api/stand', '/api/navigate', '/api/reset', '/api/spatial/capture'):
                    response = await client.post(endpoint, json={})
                    self.assertEqual(response.status, 403, endpoint)
                async with client.ws_connect('/ws') as websocket:
                    self.assertEqual((await websocket.receive_json())['type'], 'hello')
                    self.assertTrue((await websocket.receive_json())['preview'])
            transport.assert_not_called()
