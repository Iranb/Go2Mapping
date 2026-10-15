"""Approximate photo-depth reconstruction anchored to measured odometry voxels."""
import io
import json
import math
import struct
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from go2_spatial import stations


@dataclass
class CameraModel:
    hfov_deg: float = 100.
    pitch_deg: float = 0.
    yaw_deg: float = 0.
    # Unitree's public Go2 URDF; optical orientation and intrinsics remain approximate.
    offset_x: float = .32715
    offset_y: float = -.00003
    offset_z: float = .04297

    def __post_init__(self):
        values = asdict(self)
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in values.values()):
            raise ValueError("Camera parameters must be finite numbers")
        if not 60 <= self.hfov_deg <= 140 or abs(self.pitch_deg) > 35 or abs(self.yaw_deg) > 25:
            raise ValueError("FOV must be 60–140°, pitch within ±35° and yaw offset within ±25°")
        if not (.1 <= self.offset_x <= .5 and abs(self.offset_y) <= .15 and -.1 <= self.offset_z <= .2):
            raise ValueError("Camera position is outside the supported demo range")

    def intrinsics(self, width, height):
        focal = width/(2*math.tan(math.radians(self.hfov_deg)/2))
        return focal, focal, (width-1)/2, (height-1)/2


def rotation(rpy):
    roll, pitch, yaw = rpy
    cr, sr, cp, sp, cy, sy = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
    return np.array([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                     [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr], [-sp, cp*sr, cp*cr]])


def camera_pose(photo, model):
    body = rotation(photo.get("rpy", [0., 0., photo["yaw"]]))
    origin = np.asarray(photo["position"])+body@np.array([model.offset_x, model.offset_y, model.offset_z])
    optical_to_body = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
    orient = body@rotation([0., math.radians(model.pitch_deg), math.radians(model.yaw_deg)])@optical_to_body
    return origin, orient


def project(points, photo, model, width=160, height=90):
    origin, orient = camera_pose(photo, model)
    optical = (np.asarray(points)-origin)@orient
    fx, fy, cx, cy = model.intrinsics(width, height)
    depth = optical[:, 2]
    valid = np.all(np.isfinite(optical), axis=1) & (depth > .18) & (depth < 8.)
    ids = np.flatnonzero(valid)
    u = fx*optical[ids, 0]/depth[ids]+cx
    v = fy*optical[ids, 1]/depth[ids]+cy
    inside = (u >= 0) & (u <= width-1) & (v >= 0) & (v <= height-1)
    return ids[inside], np.column_stack((u[inside], v[inside])), depth[ids[inside]]


def raster_depth(points, photo, model, width=160, height=90):
    ids, uv, z = project(points, photo, model, width, height)
    pixels = np.rint(uv).astype(int)
    depth = np.full(width*height, np.inf, np.float32)
    np.minimum.at(depth, pixels[:, 1]*width+pixels[:, 0], z)
    depth = depth.reshape(height, width)
    measured = np.isfinite(depth)
    depth[~measured] = 0
    return depth, measured


def fill_small_holes(depth):
    """Fill enclosed one-pixel gaps only when neighbouring depths agree."""
    depth = depth.copy()
    padded = np.pad(depth, 1)
    h, w = depth.shape
    neighbours = np.stack([padded[y:y+h, x:x+w] for y in range(3) for x in range(3) if (x, y) != (1, 1)])
    valid = neighbours > 0
    count = valid.sum(axis=0)
    low = np.where(valid, neighbours, np.inf).min(axis=0)
    high = np.where(valid, neighbours, -np.inf).max(axis=0)
    mean = neighbours.sum(axis=0)/np.maximum(count, 1)
    enclosed = ((padded[1:1+h, :w] > 0) & (padded[1:1+h, 2:2+w] > 0) |
                (padded[:h, 1:1+w] > 0) & (padded[2:2+h, 1:1+w] > 0))
    filled = (depth == 0) & (count >= 5) & enclosed & (high-low <= np.maximum(.12, .06*mean))
    depth[filled] = mean[filled]
    return depth, filled


def voxel_support(depth, model, resolution=.05):
    """Approximate the image footprint of a finite lidar voxel, capped at 3 px."""
    h, w = depth.shape
    focal = model.intrinsics(w, h)[0]
    radius = np.minimum(3., focal*resolution/(2*np.maximum(depth, .18)))
    supported = np.where(depth > 0, depth, np.inf)
    for dy in range(-3, 4):
        for dx in range(-3, 4):
            if dx == dy == 0:
                continue
            y0, y1, x0, x1 = max(0, dy), min(h, h+dy), max(0, dx), min(w, w+dx)
            source = depth[y0-dy:y1-dy, x0-dx:x1-dx]
            covers = (source > 0) & (radius[y0-dy:y1-dy, x0-dx:x1-dx]+.5 >= math.hypot(dx, dy))
            supported[y0:y1, x0:x1] = np.minimum(supported[y0:y1, x0:x1], np.where(covers, source, np.inf))
    supported[~np.isfinite(supported)] = 0
    return supported


def depth_surface(depth, photo, model, half_span_deg):
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    fx, fy, cx, cy = model.intrinsics(w, h)
    optical = np.stack(((u-cx)*depth/fx, (v-cy)*depth/fy, depth), axis=-1)
    origin, orient = camera_pose(photo, model)
    vertices = (optical.reshape(-1, 3)@orient.T+origin).astype(np.float32)
    indices = np.arange(w*h).reshape(h, w)
    a, b, c, d = indices[:-1, :-1], indices[:-1, 1:], indices[1:, :-1], indices[1:, 1:]
    triangles = np.concatenate((np.stack((a, b, c), axis=-1).reshape(-1, 3),
                                np.stack((b, d, c), axis=-1).reshape(-1, 3)))
    depths = depth.ravel()[triangles]
    good = np.all(depths > 0, axis=1)
    good &= np.ptp(depths, axis=1) <= np.maximum(.12, .06*depths.min(axis=1))
    cam = optical.reshape(-1, 3)
    center = cam[triangles].mean(axis=1)
    good &= np.abs(np.arctan2(center[:, 0], center[:, 2])) <= math.radians(half_span_deg)
    xyz = vertices[triangles]
    max_edge = np.maximum.reduce([np.linalg.norm(xyz[:, 0]-xyz[:, 1], axis=1),
                                  np.linalg.norm(xyz[:, 1]-xyz[:, 2], axis=1),
                                  np.linalg.norm(xyz[:, 2]-xyz[:, 0], axis=1)])
    good &= max_edge <= np.minimum(.45, .12+.06*depths.min(axis=1))
    triangles = triangles[good]
    if not len(triangles):
        return np.empty((0, 3), np.float32), np.empty((0, 2), np.float32), np.empty(0, np.uint32)
    used, remap = np.unique(triangles, return_inverse=True)
    # Three.js textures use bottom-origin UV; PNG/JPEG rows use top-origin pixels.
    uv = np.column_stack((u.ravel()[used]/(w-1), 1-v.ravel()[used]/(h-1))).astype(np.float32)
    return vertices[used], uv, remap.reshape(-1).astype(np.uint32)


def surface_packet(vertices, uv, indices):
    return (struct.pack('<III', 1, len(vertices), len(indices))+
            np.asarray(vertices, '<f4').tobytes()+np.asarray(uv, '<f4').tobytes()+np.asarray(indices, '<u4').tobytes())


def cloud_packet(points, colors=None):
    data = points if colors is None else np.column_stack((points, colors))
    return struct.pack('<II', 1 if colors is None else 3, len(points))+np.asarray(data, '<f4').tobytes()


def depth_preview(depth):
    t = np.clip((depth-.2)/5.8, 0, 1)
    colors = np.stack((np.clip(1.7-2*t, 0, 1), np.clip(1.5-np.abs(t-.5)*3, 0, 1), np.clip(2*t-.2, 0, 1)), axis=-1)
    colors[depth <= 0] = 0
    output = io.BytesIO()
    Image.fromarray(np.uint8(colors*255)).resize((480, 270), Image.Resampling.NEAREST).save(output, format='PNG')
    return output.getvalue()


def colored_ply(points, colors, has_color):
    head = (f'ply\nformat binary_little_endian 1.0\ncomment odom voxel centres in metres; approximate camera registration\n'
            f'element vertex {len(points)}\nproperty float x\nproperty float y\nproperty float z\n'
            'property uchar red\nproperty uchar green\nproperty uchar blue\nproperty uchar has_color\nend_header\n')
    rows = np.empty(len(points), dtype=[('p', '<f4', 3), ('c', 'u1', 3), ('valid', 'u1')])
    rows['p'], rows['c'], rows['valid'] = points, np.uint8(np.clip(colors, 0, 1)*255), has_color
    return head.encode()+rows.tobytes()


def build_fusion(folder, points, photos, model, dataset_session, floor):
    """Write one immutable photo-depth dataset; publish the manifest last."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    points = np.asarray(points, np.float32)
    points = points[np.all(np.isfinite(points), axis=1)]
    if not len(points) or not photos:
        raise ValueError('Photos and lidar must share the same coordinate frame')
    (folder/'lidar.bin').write_bytes(cloud_packet(points))
    scores, colors = np.zeros(len(points)), np.full((len(points), 3), .38, np.float32)
    frames = []
    groups = stations(photos)
    station_for = {photo['slot']: group for group in groups for photo in group['photos']}
    for photo in sorted(photos, key=lambda p: p['slot']):
        slot = photo['slot']
        group = station_for[slot]
        half_span = min(model.hfov_deg*.45, max(10., 180/len(group['photos'])*1.15))
        rgb = np.asarray(Image.open(io.BytesIO(photo['jpeg'])).convert('RGB'))
        paired = photo.get('depth_points')
        depth_points = paired if paired is not None else points
        sparse, measured = raster_depth(depth_points, photo, model)
        supported = voxel_support(sparse, model, photo.get('lidar_resolution', .05))
        depth, filled = fill_small_holes(supported)
        measured &= np.abs(depth-sparse) < 1e-5
        filled = (depth > 0) & ~measured
        vertices, uv, indices = depth_surface(depth, photo, model, half_span)
        prefix = f'frame-{slot:02d}'
        (folder/f'{prefix}.jpg').write_bytes(photo['jpeg'])
        (folder/f'{prefix}.bin').write_bytes(surface_packet(vertices, uv, indices))
        (folder/f'{prefix}-depth.png').write_bytes(depth_preview(depth))
        np.savez_compressed(folder/f'{prefix}-depth.npz', depth_m=depth, sparse_depth_m=sparse,
                            measured=measured, interpolated=filled)
        ids, projected, z = project(points, photo, model)
        pixel = np.rint(projected).astype(int)
        if len(ids):
            nearest = supported[pixel[:, 1], pixel[:, 0]]
            visible = (nearest > 0) & (z <= nearest+.07+.015*z)
            fx, _, cx, _ = model.intrinsics(160, 90)
            angle = np.arctan((projected[:, 0]-cx)/fx)
            score = np.cos(angle)**4/(1+.08*z)
            replace = visible & (score > scores[ids])
            chosen = ids[replace]
            # Sample at the normalized pixel positions used by depth projection.
            xy = projected[replace]/[159, 89]*[rgb.shape[1]-1, rgb.shape[0]-1]
            xy = np.rint(xy).astype(int)
            colors[chosen] = rgb[xy[:, 1], xy[:, 0]]/255.
            scores[chosen] = score[replace]
        frames.append({'slot': slot, 'station': group['id'], 'yaw': photo['yaw'], 'position': photo['position'],
                       'image': f'{prefix}.jpg', 'surface': f'{prefix}.bin', 'depth_preview': f'{prefix}-depth.png',
                       'depth_data': f'{prefix}-depth.npz', 'vertices': len(vertices), 'triangles': len(indices)//3,
                       'measured_pixels': int(measured.sum()), 'interpolated_pixels': int(filled.sum()),
                       'depth_width': 160, 'depth_height': 90,
                       'pairing': 'host-time-paired' if paired is not None else 'historical-static-map',
                       'pose': 'full-rpy' if 'rpy' in photo else 'yaw-only',
                       'timing': {k: photo.get(k) for k in ('camera_monotonic', 'state_monotonic', 'lidar_monotonic')}})
    valid = scores > 0
    (folder/'colored.bin').write_bytes(cloud_packet(points[valid], colors[valid]))
    (folder/'colored.ply').write_bytes(colored_ply(points, colors, valid))
    np.savez_compressed(folder/'colored_cloud.npz', points=points, colors=colors, has_color=valid)
    low, high = np.percentile(points, [2, 98], axis=0)
    manifest = {'id': folder.name, 'dataset_session': dataset_session, 'coordinate_frame': 'odom-z-up-metres',
                'calibrated': False, 'camera': asdict(model), 'floor': floor,
                'lidar_points': len(points), 'colored_points': int(valid.sum()),
                'surface_triangles': sum(f['triangles'] for f in frames), 'frames': frames,
                'center': ((low+high)/2).tolist(), 'radius': max(2., float(np.linalg.norm(high-low))/2),
                'lidar': 'lidar.bin', 'colored': 'colored.bin', 'export': 'colored.ply',
                'stations': [{'id': g['id'], 'position': g['position'],
                              'slots': [p['slot'] for p in g['photos']]} for g in groups],
                'station_count': len(groups), 'spatial': len(groups)>1,
                'depth_type': 'optical-z-metres',
                'mesh_half_span_deg': 'per-station',
                'note': 'Camera calibration is approximate. Missing depth stays empty; texturing and local hole filling are for display only.'}
    (folder/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest
