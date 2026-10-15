"""Spatial keyframes and separate panorama stations in one odometry frame."""
import math
import json
from pathlib import Path
import numpy as np

STATION_RADIUS = .35
MAX_PHOTOS = 192


def stations(photos):
    groups = []
    for photo in sorted(photos, key=lambda p: p['slot']):
        position = np.asarray(photo['position'], float)
        epoch = photo.get('connection_epoch')
        candidates = [s for s in groups if s['epoch'] == epoch and
                      np.linalg.norm(position-np.asarray(s['position'])) <= STATION_RADIUS]
        station = min(candidates, key=lambda s: np.linalg.norm(position-s['position'])) if candidates else None
        if station is None:
            station = {'id': len(groups), 'position': position.tolist(), 'epoch': epoch, 'photos': []}
            groups.append(station)
        station['photos'].append(photo)
    return groups


def select_keyframe(photos, pose, epoch, now):
    """Keep the same heading at different locations; bound memory per capture."""
    groups = stations(photos.values())
    position, yaw = np.asarray(pose['position']), float(pose['imu_state']['rpy'][2])
    near = [s for s in groups if s['epoch'] == epoch and
            np.linalg.norm(position-s['position']) <= STATION_RADIUS]
    group = min(near, key=lambda s: np.linalg.norm(position-s['position'])) if near else None
    if group:
        for photo in group['photos']:
            angle = math.atan2(math.sin(yaw-photo['yaw']), math.cos(yaw-photo['yaw']))
            if abs(angle) < math.radians(15):
                return None
    if len(photos) >= MAX_PHOTOS:
        return None
    return max(photos, default=-1)+1


def build_spatial_panorama(photos, model):
    # Never stitch translations into a common spherical camera centre.
    from go2_panorama import build_panorama
    groups = stations(photos)
    views = []
    for group in groups:
        result = build_panorama(group['photos'], model, width=1536 if len(groups)>1 else 3072)
        views.append({**result, 'station': group['id'], 'position': group['position'],
                      'slots': [p['slot'] for p in group['photos']]})
    if not views:
        raise ValueError('Capture spatial photos first')
    return {**views[0], 'stations': views, 'station_count': len(views),
            'frames': len(photos), 'matched_pairs': sum(v['matched_pairs'] for v in views)}


def save_capture(folder, photos, points, resolution):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    records = []
    for photo in photos:
        stem = f"photo-{photo['slot']:03d}"
        (folder/(stem+'.jpg')).write_bytes(photo['jpeg'])
        record = {k: v for k, v in photo.items() if k not in ('jpeg', 'depth_points')}
        record['file'] = stem+'.jpg'
        if photo.get('depth_points') is not None:
            record['depth_file'] = stem+'-lidar.npz'
            np.savez_compressed(folder/record['depth_file'], points=photo['depth_points'])
        records.append(record)
    np.savez_compressed(folder/'room_cloud.tmp.npz', points=points)
    (folder/'room_cloud.tmp.npz').replace(folder/'room_cloud.npz')
    (folder/'capture.json').write_text(json.dumps({'resolution': resolution, 'photos': len(records),
        'stations': len(stations(photos)), 'coordinate_frame': 'odom-z-up-metres'}))
    temporary = folder/'photos.tmp.json'
    temporary.write_text(json.dumps(records, ensure_ascii=False, indent=2))
    temporary.replace(folder/'photos.json')
