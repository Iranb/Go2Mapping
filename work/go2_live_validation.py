import argparse
import asyncio
import json
import struct
import time
from pathlib import Path

import aiohttp
import numpy as np

BASE = 'http://127.0.0.1:63541'

async def main(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    statuses, frames, plans = [], {}, []
    latest = {}
    client_id = None
    started = False
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=12)) as session:
        async def save_url(route, name):
            async with session.get(BASE+route) as response:
                response.raise_for_status()
                (out/name).write_bytes(await response.read())

        async def post(route, body):
            async with session.post(BASE+route, json={'client_id': client_id, **body}) as response:
                text = await response.text()
                if response.status != 200:
                    raise RuntimeError(f'{route}: {response.status} {text}')
                return json.loads(text)

        await save_url('/api/diagnostics', 'diagnostics-before.json')
        await save_url('/camera.jpg', 'camera-before.jpg')
        async with session.ws_connect(BASE+'/ws') as ws:
            async def receive():
                nonlocal client_id, latest
                async for message in ws:
                    if message.type == aiohttp.WSMsgType.TEXT:
                        item = json.loads(message.data)
                        if item.get('type') == 'hello':
                            client_id = item['client_id']
                        elif item.get('type') == 'status':
                            latest = item
                            statuses.append({'received_monotonic': time.monotonic(), 'status': item})
                    elif message.type == aiohttp.WSMsgType.BINARY:
                        kind, count = struct.unpack('<II', message.data[:8])
                        points = np.frombuffer(message.data[8:], dtype='<f4').reshape(count, 3).copy()
                        if kind == 2 and count and latest.get('pose'):
                            index = len(frames)//4
                            frames[f'cloud_{index}'] = points
                            frames[f'time_{index}'] = np.array(time.monotonic())
                            frames[f'pose_{index}'] = np.array(latest['pose']['position'])
                            frames[f'yaw_{index}'] = np.array(latest['pose']['yaw'])
                        elif kind == 1 and count:
                            np.save(out/'accumulated-map-centres.npy', points)

            async def heartbeat():
                while True:
                    await ws.send_str('heartbeat')
                    await asyncio.sleep(.75)

            receiver = asyncio.create_task(receive())
            heart = asyncio.create_task(heartbeat())
            try:
                until = time.monotonic()+5
                while not client_id or not latest:
                    if time.monotonic() > until:
                        raise RuntimeError('No operator/status connection')
                    await asyncio.sleep(.05)
                if latest['active']:
                    raise RuntimeError('Another action is active')
                await asyncio.sleep(args.capture_seconds)
                if args.turn:
                    if not latest.get('ready'):
                        raise RuntimeError('Turn is not ready')
                    if latest['turn_latched']:
                        await post('/api/turn/rearm', {'turn_token': latest['turn_token']})
                        async with session.get(BASE+'/api/status') as response:
                            current = await response.json()
                    else:
                        current = latest
                    result = await post('/api/start', {'turn_token': current['turn_token'], 'channel': 'joystick'})
                    started = bool(result['started'])
                if args.target:
                    if not latest.get('nav_ready') or latest.get('map_source') != 'live':
                        raise RuntimeError('Live navigation is not ready')
                    plan = await post('/api/plan', {'target': args.target})
                    plans.append(plan)
                    (out/'plan.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2))
                    print(json.dumps({'event': 'plan', **{k: plan.get(k) for k in ['reachable', 'reason', 'path', 'walk_count', 'turn_count', 'arrival_target']}}, ensure_ascii=False), flush=True)
                    if args.execute and plan['reachable']:
                        result = await post('/api/navigate', {'plan_id': plan['id']})
                        started = bool(result['started'])
                if started:
                        begin, last_print = time.monotonic(), -1e9
                        stop_sent = False
                        while time.monotonic()-begin < 125:
                            await asyncio.sleep(.25)
                            if args.turn and args.stop_degrees and not stop_sent and latest.get('degrees', 0) >= args.stop_degrees:
                                await post('/api/stop', {})
                                stop_sent = True
                            if time.monotonic()-last_print >= 5:
                                print(json.dumps({'event': 'progress', 'elapsed_s': round(time.monotonic()-begin, 1), **{k: latest.get(k) for k in ['active', 'phase', 'message', 'pose', 'nav_blocked', 'degrees']}, 'navigation': latest.get('navigation')}, ensure_ascii=False), flush=True)
                                last_print = time.monotonic()
                            if not latest.get('active') and latest.get('phase') in ('complete', 'error', 'stopped') and time.monotonic()-begin > 1:
                                break
                        else:
                            await post('/api/stop', {})
                            raise RuntimeError('Test deadline reached; stop requested')
                        print(json.dumps({'event': 'finished', 'phase': latest.get('phase'), 'message': latest.get('message'), 'pose': latest.get('pose'), 'report': latest.get('report')}, ensure_ascii=False), flush=True)
                await save_url('/api/diagnostics', 'diagnostics-after.json')
                await save_url('/camera.jpg', 'camera-after.jpg')
            finally:
                if started and latest.get('active'):
                    await post('/api/stop', {})
                    until = time.monotonic()+8
                    while latest.get('active') and time.monotonic() < until:
                        await asyncio.sleep(.1)
                np.savez_compressed(out/'live-frames-centres.npz', **frames)
                (out/'status-sequence.json').write_text(json.dumps(statuses, ensure_ascii=False))
                heart.cancel()
                receiver.cancel()
                await asyncio.gather(heart, receiver, return_exceptions=True)
    print(json.dumps({'folder': str(out), 'frames': len(frames)//4, 'pose': latest.get('pose'), 'battery': latest.get('battery'), 'active': latest.get('active')}, ensure_ascii=False), flush=True)

parser = argparse.ArgumentParser()
parser.add_argument('--output', required=True)
parser.add_argument('--target', type=float, nargs=2)
parser.add_argument('--capture-seconds', type=float, default=4.)
parser.add_argument('--execute', action='store_true')
parser.add_argument('--turn', action='store_true')
parser.add_argument('--stop-degrees', type=float)
asyncio.run(main(parser.parse_args()))
