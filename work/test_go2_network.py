"""LAN changes must retain robot identity without blocking the control loop."""
import asyncio
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from go2_network import DiscoveryError, read_device, resolve_robot_ip


class NetworkTests(unittest.IsolatedAsyncioTestCase):
    async def test_manual_ip_does_not_require_multicast(self):
        with patch("go2_network.discover_ip_sn") as scan:
            self.assertEqual(await resolve_robot_ip("198.51.100.10", "dog-a"), "198.51.100.10")
        scan.assert_not_called()

    async def test_reconnect_uses_new_address_for_same_serial(self):
        replies = [{"dog-b": "192.0.2.20", "dog-a": "192.0.2.10"},
                   {"dog-a": "198.51.100.55"}]
        with patch("go2_network.discover_ip_sn", side_effect=replies):
            self.assertEqual(await resolve_robot_ip("auto", "dog-a"), "192.0.2.10")
            self.assertEqual(await resolve_robot_ip("auto", "dog-a"), "198.51.100.55")

    async def test_other_robot_is_never_selected(self):
        with patch("go2_network.discover_ip_sn", return_value={"dog-b": "192.0.2.20"}):
            with self.assertRaises(DiscoveryError):
                await resolve_robot_ip("auto", "dog-a")

    async def test_missing_serial_needs_explicit_ip(self):
        with self.assertRaises(DiscoveryError):
            await resolve_robot_ip("auto", "")

    async def test_discovery_leaves_event_loop_responsive(self):
        events = []
        def scan(**kwargs):
            time.sleep(.1)
            events.append("discovered")
            return {"dog-a": "192.0.2.10"}
        with patch("go2_network.discover_ip_sn", side_effect=scan):
            task = asyncio.create_task(resolve_robot_ip("auto", "dog-a"))
            await asyncio.sleep(.02)
            events.append("heartbeat")
            await task
        self.assertEqual(events, ["heartbeat", "discovered"])

    async def test_invalid_key_error_does_not_echo_credentials(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"device.json"
            path.write_text(json.dumps({"aes_128_key": "private-invalid-value"}))
            with self.assertRaises(ValueError) as caught:
                read_device(path)
            self.assertNotIn("private-invalid-value", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
