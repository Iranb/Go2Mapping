"""Fault evidence preserves power and command feedback without moving the robot."""
import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import go2_live_demo as app
from test_go2_demo import FakeDemo


class FaultTests(unittest.IsolatedAsyncioTestCase):
    async def test_simultaneous_motor_undervoltage_is_saved_with_pre_fault_power(self):
        demo = FakeDemo()
        low = {'power_v': 28.7, 'bms_state': {'soc': 58, 'current': -3415},
               'motor_state': [{'temperature': 40, 'lost': 2, 'reserve': [0, 597]}]*12}
        demo.record_command(0., .16)
        demo.command_samples[-1].update(channel='joystick', rx=0., ly=.24)
        demo.low_callback({'data': low})
        with tempfile.TemporaryDirectory() as folder, patch.object(app, 'SITE', Path(folder)):
            for motor in range(300, 308):
                demo.fault_callback({'type': 'add_error', 'data': [1792063451, motor, 8]})
            with self.assertRaisesRegex(RuntimeError, 'bus undervoltage'):
                demo.check_navigation(stationary=True)
            task = demo.fault_capture_task
            demo.fault_callback({'type': 'errors', 'data': [[1792063451, m, 8] for m in range(300, 308)]})
            self.assertIs(task, demo.fault_capture_task)
            await task
            records = list((Path(folder)/'faults').glob('*.json'))
            self.assertEqual(len(records), 1)
            report = json.loads(records[0].read_text())
            self.assertEqual(len(report['events']), 8)
            self.assertEqual(len(report['active_faults']), 8)
            sample = report['power_samples'][0]
            self.assertEqual(sample['power_v'], 28.7)
            self.assertEqual(sample['bms']['current'], -3415)
            self.assertEqual(sample['command']['ly'], .24)
            self.assertEqual(len(sample['motors']), 12)
            self.assertEqual(demo.last_fault_record, str(records[0]))
            low['bms_state']['soc'] = 55
            self.assertEqual(sample['bms']['soc'], 58)

    async def test_invalid_json_cannot_clear_an_active_fault(self):
        demo = FakeDemo()
        with tempfile.TemporaryDirectory() as folder, patch.object(app, 'SITE', Path(folder)):
            demo.fault_callback({'type': 'errors', 'data': [[1, 301, 8]]})
            demo.channel_faults('{"type":"errors","data":invalid}')
            self.assertEqual(demo.faults[(301, 8)]['code'], '8')
            with self.assertRaisesRegex(RuntimeError, 'bus undervoltage'):
                demo.check_robot()
            await demo.fault_capture_task
        demo.fault_callback({'type': 'rm_error', 'data': [1, 301, 8]})
        self.assertFalse(demo.faults)

    def test_fault_description_preserves_source_and_combined_bits(self):
        text = app.fault_text({'source': 307, 'code': '28'})
        self.assertIn('307-0x28', text)
        self.assertIn('bus undervoltage', text)
        self.assertIn('encoder fault', text)
        self.assertEqual(app.fault_text({'source': 400, 'code': '8'}), '400-0x8')


if __name__ == '__main__':
    unittest.main()
