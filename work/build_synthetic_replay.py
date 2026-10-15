"""Regenerate the public replay from a procedural scene, without a robot."""
import asyncio
import json
from pathlib import Path
import numpy as np
from go2_navigation import GroundGrid
from go2_navigation_sim import PROFILES, simulate
from preview_ui import synthetic_scene


async def main():
    points = synthetic_scene()
    position, yaw, target = [0., 0., .3], 0., [1.8, 0.]
    grid = GroundGrid(points, position, yaw)
    yy, xx = np.nonzero(grid.occupied)
    data = {'provenance': 'Procedurally generated floor and wall; no field recordings',
            'target': target, 'start': position[:2],
            'occupied': (np.column_stack((xx, yy))+grid.origin).tolist(),
            'resolution': grid.resolution, 'cases': {}}
    data['occupied'] = (np.asarray(data['occupied'])*grid.resolution).round(4).tolist()
    for profile in PROFILES:
        result, trace, _ = await simulate(points, position, yaw, target, profile)
        data['cases'][profile] = {k: result[k] for k in
            ('outcome', 'message', 'elapsed_s', 'final_distance_m', 'response_model', 'plan')}
        trace = trace[:, :4].copy()
        trace[:, 0] -= 1000.
        data['cases'][profile]['trace'] = trace.round(5).tolist()
        if result['outcome'] != 'complete' or result['violations']:
            raise RuntimeError(f'Synthetic replay failed: {profile}: {result["message"]}')
    output = Path(__file__).resolve().parent.parent/'outputs/go2-live-demo/assets/navigation-synthetic.json'
    output.write_text(json.dumps(data, indent=None, separators=(',', ':'))+'\n')
    print('Built three synthetic controller replays; all arrived with no modeled map violations.')


if __name__ == '__main__':
    asyncio.run(main())
