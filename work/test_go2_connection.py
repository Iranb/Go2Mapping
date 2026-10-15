"""LAN signaling must not block the operator heartbeat or wait without a bound."""
import asyncio
import time
import types
import unittest
from unittest.mock import patch

from unitree_webrtc_connect import webrtc_driver, unitree_auth


class SignalingTests(unittest.IsolatedAsyncioTestCase):
    async def test_slow_lan_signaling_leaves_event_loop_responsive(self):
        conn = types.SimpleNamespace(connectionMethod=webrtc_driver.WebRTCConnectionMethod.LocalSTA,
                                     token='', aes_128_key='test-key')
        peer = types.SimpleNamespace(localDescription=types.SimpleNamespace(sdp='test-sdp', type='offer'))
        ticks = []
        def slow(*args, **kwargs):
            time.sleep(.15)
            ticks.append('answer')
            return 'answer'
        with patch.object(webrtc_driver, 'send_sdp_to_local_peer', slow):
            task = asyncio.create_task(webrtc_driver.UnitreeWebRTCConnection.get_answer_from_local_peer(
                conn, peer, '192.0.2.10'))
            await asyncio.sleep(.03)
            ticks.append('heartbeat')
            self.assertEqual(await task, 'answer')
        self.assertEqual(ticks, ['heartbeat', 'answer'])

    async def test_lan_post_has_connect_and_read_timeout(self):
        with patch.object(unitree_auth.requests, 'post', side_effect=unitree_auth.requests.exceptions.Timeout()) as post:
            self.assertIsNone(unitree_auth.make_local_request('http://192.0.2.10:9991/con_notify'))
        self.assertEqual(post.call_args.kwargs['timeout'], (3, 8))


if __name__ == '__main__':
    unittest.main()
