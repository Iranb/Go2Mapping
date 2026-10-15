"""Far recorded-map navigation and isolation from the live goal limit."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import go2_navigation as navigation
from go2_navigation_sim import main, simulate


class SimulationScopeTests(unittest.IsolatedAsyncioTestCase):



    async def test_cli_records_planning_rejection_without_movement(self):
        with tempfile.TemporaryDirectory() as folder:
            capture, output = Path(folder)/'capture', Path(folder)/'output'
            capture.mkdir()
            points = np.array([[x*.05, y*.05, 0.] for x in range(-20, 21) for y in range(-20, 21)], np.float32)
            np.savez_compressed(capture/'live-frames-centres.npz', cloud_0=points, pose_0=[0., 0., .3], yaw_0=0.)
            (capture/'diagnostics-after.json').write_text(json.dumps({'turn_space': {'resolution': .05}}))
            args = types.SimpleNamespace(capture=capture, output=output, frames='last',
                                         target=[5.4, 1.75], goal_limit=None, profiles=['nominal'])
            with contextlib.redirect_stdout(io.StringIO()):
                await main(args)
            result = json.loads((output/'simulation-results.json').read_text())
            self.assertEqual(result['cases'], 1)
            self.assertEqual(result['completed'], 0)
            self.assertEqual(result['results'][0]['outcome'], 'unreachable')
            self.assertIsNone(result['results'][0]['plan'])
            self.assertEqual(result['results'][0]['report']['nonzero_commands'], 0)
            trace = np.load(output/'trajectories.npz')
            self.assertEqual(trace['frame_0_nominal'].shape, (0, 6))


if __name__ == '__main__':
    unittest.main()
