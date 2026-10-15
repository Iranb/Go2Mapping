"""Serve the English interface with synthetic data and no robot transport."""
import argparse
import asyncio
from aiohttp import web
import numpy as np

from go2_demo_core import point_packet
from go2_live_demo import Demo, create_app


def synthetic_scene():
    floor = [[x*.05-.025, y*.05-.025, -.025]
             for x in range(-40, 61) for y in range(-40, 41)]
    obstacle = [[.8-.025, y*.05-.025, z-.025]
                for y in range(-7, 8) for z in (.15, .25, .4)]
    return np.asarray(floor+obstacle, np.float32)


class PreviewDemo(Demo):
    def __init__(self):
        super().__init__()
        self.phase, self.message = 'preview', 'Offline preview · Synthetic scene · No robot connected'
        self.metadata = {'resolution': .05}
        points = synthetic_scene()
        self.map_keys = np.rint(points/.05).astype(np.int32)
        self.map_packet, self.map_version = point_packet(1, points+.025), 1
        self.fusion_message = 'Capture photos with your own robot to build Spatial 3D'
        self.sport = {'position': [0., 0., .3], 'imu_state': {'rpy': [0., 0., 0.]},
                      'velocity': [0., 0., 0.], 'yaw_speed': 0., 'body_height': .3}

    def status(self):
        return {**super().status(), 'preview': True, 'map_source': 'synthetic'}

    async def startup(self, app):
        # Deliberately omit connection, saved-scene loading and control tasks.
        self.tasks = [asyncio.create_task(self.broadcast())]


@web.middleware
async def read_only(request, handler):
    if request.method not in ('GET', 'HEAD'):
        raise web.HTTPForbidden(text='Offline preview is read-only; no robot transport exists')
    return await handler(request)


def preview_app():
    app = create_app(PreviewDemo())
    app.middlewares.append(read_only)
    return app


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=63542)
    args = parser.parse_args()
    web.run_app(preview_app(), host='127.0.0.1', port=args.port, access_log=None)
