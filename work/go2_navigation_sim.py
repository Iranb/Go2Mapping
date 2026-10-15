"""Offline navigation on fixed recorded lidar with assumed planar actuators.

Command latency, acceleration and deadband are simulated. This is not a Go2
rigid-body model. No robot connection or transport is created.
"""
import argparse
import asyncio
from collections import deque
import json
import math
from pathlib import Path
import time
import types
from unittest.mock import patch
import numpy as np
import go2_live_demo as controller
import go2_navigation as navigation
from go2_navigation import GroundGrid, footprint_offsets, wrap

PROFILES = {
    'nominal': {'latency_s': .05, 'forward_gain': 1., 'yaw_gain': 1.},
    'delayed': {'latency_s': .15, 'forward_gain': .85, 'yaw_gain': .85},
    'slow': {'latency_s': .20, 'forward_gain': .70, 'yaw_gain': .75},
}


class NavigationSimulation(controller.Demo):
    def __init__(self, points, position, yaw, profile='nominal', resolution=.05):
        super().__init__()
        self.profile = dict(PROFILES[profile])
        self.now, self.body_speed, self.angular_speed = 1000., 0., 0.
        self.pending, self.target_speed, self.target_yaw = deque(), 0., 0.
        self.connected = self.faults_seen = True
        self.low = {'motor_state': [{'reserve': [0, 0], 'temperature': 40} for _ in range(12)]}
        self.sport = {'position': list(map(float, position)), 'imu_state': {'rpy': [0., 0., float(yaw)]},
                      'velocity': [0., 0., 0.], 'yaw_speed': 0., 'body_height': .30}
        self.cloud, self.metadata = points, {'resolution': resolution}
        self.sockets = {'simulation': {'heartbeat': self.now}}
        self.reference = GroundGrid(points, position, yaw, resolution)
        self.trace, self.saved, self.violations = [], None, []
        self.refresh()

    def refresh(self):
        self.state_time = self.low_time = self.lidar_time = self.geometry_time = self.now
        self.sport_sequence += 1
        self.sockets['simulation']['heartbeat'] = self.now

    def move(self, yaw, vx=0., vy=0.):
        if vy: raise ValueError('Simulation supports forward walking and turning only')
        self.record_command(yaw, vx, vy)
        self.pending.append((self.now+self.profile['latency_s'], vx, yaw))
        if vx or yaw: self.report['nonzero_commands'] = self.report.get('nonzero_commands', 0)+1

    def move_joystick(self, rx):
        self.joystick_active, self.joystick_rx = True, rx
        self.move(-rx/controller.RIGHT_STICK_PER_YAW)
        self.command_samples[-1].update(channel='joystick', rx=rx)

    def move_navigation(self, yaw, vx):
        self.move(yaw, vx)

    async def rpc(self, topic, api, parameter=None, timeout=1.):
        if topic == controller.RTC_TOPIC['SPORT_MOD'] and api == 1008:
            self.move(parameter['z'], parameter['x'], parameter['y'])
        if topic == controller.RTC_TOPIC['SPORT_MOD'] and api == 1003: self.move(0.)
        return {'data': {'data': json.dumps({'name': 'mcf', 'enable': True}),
                         'header': {'status': {'code': 0}}}}

    async def navigation_grid(self, yaw):
        grid = GroundGrid(self.cloud, self.sport['position'], yaw, self.metadata['resolution'],
                          self.ground_memory, self.now)
        self.ground_memory = grid.ground_memory
        return grid, self.now

    def save_navigation(self, report, samples, commands, grid=None):
        self.saved = {'report': report, 'samples': samples, 'commands': commands}

    def advance(self, seconds):
        steps = max(1, math.ceil(seconds/.05))
        for _ in range(steps):
            dt = seconds/steps
            previous = np.asarray(self.sport['position'][:2]).copy()
            previous_yaw = self.sport['imu_state']['rpy'][2]
            self.now += dt
            while self.pending and self.pending[0][0] <= self.now+1e-9:
                _, self.target_speed, self.target_yaw = self.pending.popleft()
            desired_v = self.target_speed*self.profile['forward_gain'] if abs(self.target_speed) >= .08 else 0.
            desired_w = self.target_yaw*self.profile['yaw_gain'] if abs(self.target_yaw) >= .15 else 0.
            self.body_speed += float(np.clip(desired_v-self.body_speed, -.35*dt, .35*dt))
            self.angular_speed += float(np.clip(desired_w-self.angular_speed, -1.2*dt, 1.2*dt))
            middle = previous_yaw+self.angular_speed*dt/2
            position = previous+self.body_speed*dt*np.array([math.cos(middle), math.sin(middle)])
            yaw = wrap(previous_yaw+self.angular_speed*dt)
            self.sport['position'][:2] = position.tolist()
            self.sport['imu_state']['rpy'][2] = yaw
            self.sport['velocity'] = [self.body_speed*math.cos(yaw), self.body_speed*math.sin(yaw), 0.]
            self.sport['yaw_speed'] = self.angular_speed
            issue = None
            if abs(self.body_speed) > .002:
                for point, heading in ((previous, previous_yaw), (position, yaw)):
                    cells = np.asarray(footprint_offsets(heading, self.metadata['resolution'], 0.))+self.reference.cell(point)
                    inside = np.all((cells >= 0) & (cells < self.reference.unsafe.shape[0]), axis=1)
                    if not np.all(inside):
                        issue = {'kind': 'outside', 'point': self.reference.world(cells[~inside][0]).tolist(),
                                 'robot_position': point.tolist()}
                        break
                    bad = self.reference.unsafe[cells[:, 1], cells[:, 0]]
                    if np.any(bad):
                        trigger = cells[bad][0]
                        issue = {'kind': 'occupied' if self.reference.occupied[trigger[1],trigger[0]] else 'unknown',
                                 'point': self.reference.world(trigger).tolist(), 'robot_position': point.tolist()}
                        break
            else:
                issue = self.reference.turn_issue(position, previous_yaw, yaw)
            if issue is not None and not self.violations:
                self.violations.append({'t': self.now, **issue})
                self.request_stop('Simulated body entered occupied or unobserved space')
            self.trace.append([self.now, *position, yaw, self.body_speed, self.angular_speed])
            self.refresh()


async def simulate(points, position, yaw, target, profile='nominal', resolution=.05, goal_limit=None):
    demo = NavigationSimulation(points, position, yaw, profile, resolution)
    limit = navigation.MAX_GOAL if goal_limit is None else float(goal_limit)
    with patch.object(navigation, 'MAX_GOAL', limit):
        route = demo.reference.plan(target, True)
    plan = {**route, 'target': list(target), 'yaw': float(yaw), 'epoch': 0,
            'start': list(position[:2]), 'used': True, 'state': 'checking'}
    real_sleep = asyncio.sleep
    async def tick(seconds):
        demo.advance(seconds)
        await real_sleep(0)
    clock = types.SimpleNamespace(monotonic=lambda: demo.now, time=time.time)
    with patch.object(navigation, 'MAX_GOAL', limit), patch.object(controller, 'time', clock), patch.object(controller.asyncio, 'sleep', tick):
        demo.refresh()
        await demo.run_navigation('simulation', plan)
    result = {'profile': profile, 'response_model': demo.profile, 'outcome': demo.phase,
              'message': demo.message, 'elapsed_s': demo.now-1000., 'goal_limit_m': limit,
              'final_position': demo.sport['position'][:2],
              'final_distance_m': float(np.linalg.norm(np.asarray(demo.sport['position'][:2])-target)),
              'final_speed_m_s': abs(demo.body_speed), 'final_yaw_speed_rad_s': abs(demo.angular_speed),
              'violations': demo.violations, 'plan': route,
              'report': demo.saved['report'] if demo.saved else demo.report}
    return result, np.asarray(demo.trace), demo.command_samples


async def main(args):
    capture = np.load(args.capture/'live-frames-centres.npz')
    diagnostic = json.loads((args.capture/'diagnostics-after.json').read_text())
    resolution = diagnostic['turn_space']['resolution']
    target = args.target or json.loads((args.capture/'reported-plan-diagnostics.json').read_text())['status']['navigation']['target']
    indices = sorted(int(k.split('_')[1]) for k in capture.files if k.startswith('cloud_'))
    if args.frames == 'last': indices = indices[-1:]
    args.output.mkdir(parents=True, exist_ok=True)
    results, trajectories = [], {}
    for index in indices:
        points = np.rint(capture[f'cloud_{index}']/resolution)*resolution-resolution/2
        for profile in args.profiles:
            position = capture[f'pose_{index}']
            try:
                result, trace, commands = await simulate(points, position,
                    float(capture[f'yaw_{index}']), target, profile, resolution, args.goal_limit)
            except ValueError as error:
                result = {'profile': profile, 'response_model': dict(PROFILES[profile]),
                    'outcome': 'unreachable', 'message': str(error), 'elapsed_s': 0.,
                    'goal_limit_m': navigation.MAX_GOAL if args.goal_limit is None else args.goal_limit,
                    'final_position': position[:2].tolist(),
                    'final_distance_m': float(np.linalg.norm(position[:2]-target)),
                    'final_speed_m_s': 0., 'final_yaw_speed_rad_s': 0.,
                    'violations': [], 'plan': None, 'report': {'outcome': 'unreachable', 'nonzero_commands': 0}}
                trace, commands = np.empty((0, 6)), []
            result['frame'] = index//4
            results.append(result)
            key = f'frame_{index//4}_{profile}'
            trajectories[key] = trace
            trajectories[key+'_commands'] = np.array([[c['t'], c['vx'], c['wz']] for c in commands])
            print(json.dumps({k:result[k] for k in ('frame','profile','outcome','final_distance_m','elapsed_s')},ensure_ascii=False),flush=True)
    summary = {'environment':'fixed_recorded_lidar_map', 'actuator_model':'assumed_planar_response',
        'target':target,'cases':len(results),'completed':sum(r['outcome']=='complete' for r in results),
        'map_violations':sum(bool(r['violations']) for r in results),'results':results}
    (args.output/'simulation-results.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    np.savez_compressed(args.output/'trajectories.npz',**trajectories)
    print(json.dumps({k:v for k,v in summary.items() if k!='results'},ensure_ascii=False))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--target', type=float, nargs=2, metavar=('X', 'Y'), help='Offline target in recorded odometry coordinates (metres)')
    parser.add_argument('--goal-limit', type=float, help='Offline goal distance limit; never changes the live server')
    parser.add_argument('--frames',choices=('all','last'),default='all')
    parser.add_argument('--profiles',nargs='+',choices=tuple(PROFILES),default=list(PROFILES))
    asyncio.run(main(parser.parse_args()))
