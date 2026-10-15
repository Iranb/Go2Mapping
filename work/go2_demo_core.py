"""Geometry and measured turn progress for the Go2 room demonstration."""
import math
import struct

import numpy as np

# Body/head envelope and the separate region used by in-place foot placement.
TURN_ENVELOPES = ((.35, .155), (.30, .23))
TURN_SPEED = .7
TURN_ACCELERATION = 1.2
# Outer projection of Unitree go2_description/dae/base.dae and trunk collision,
# using the model's node transform, 16 support directions and 5 mm model padding.
BODY_POLYGON = np.array([
    [.336958, .013740], [.331156, .027747], [.308252, .050651], [.184823, .101777],
    [-.128992, .101777], [-.148030, .093892], [-.192339, .049582], [-.193100, .047745],
    [-.193100, -.047745], [-.192339, -.049582], [-.147268, -.094653], [-.128992, -.102223],
    [.184823, -.102223], [.308252, -.051097], [.331156, -.028193], [.336958, -.014185]])
_BODY_EDGES = np.roll(BODY_POLYGON, -1, axis=0)-BODY_POLYGON
_BODY_EDGE_SQUARED = np.sum(_BODY_EDGES*_BODY_EDGES, axis=-1)
_BODY_RADIUS = float(np.linalg.norm(BODY_POLYGON, axis=1).max())
_BODY_INRADIUS = float(np.min((BODY_POLYGON[:, 0]*_BODY_EDGES[:, 1]-
    BODY_POLYGON[:, 1]*_BODY_EDGES[:, 0])/np.sqrt(_BODY_EDGE_SQUARED)))


def collision_ceiling(position_z, floor_z, resolution):
    """Upper collision band for the bare Go2 within the 0.18 rad tilt limit."""
    # The body model is below 0.10 m above its origin; allow 0.15 m for
    # the complete robot, and at least the nominal 0.40 m standing height.
    # Cover simultaneous roll/pitch, 2 cm clearance and the voxel half-height.
    tilt = _BODY_RADIUS*math.sqrt(2)*math.sin(.18)
    return max(float(position_z)+.15, float(floor_z)+.40)+tilt+.02+resolution/2


def body_distance(local):
    """Signed distance to the convex horizontal body/head outline."""
    edges = _BODY_EDGES
    q = np.asarray(local)[..., None, :]-BODY_POLYGON
    t = np.clip(np.sum(q*edges, axis=-1)/_BODY_EDGE_SQUARED, 0, 1)
    distance = np.linalg.norm(q-t[..., None]*edges, axis=-1).min(axis=-1)
    inside = np.all(edges[:, 0]*q[..., 1]-edges[:, 1]*q[..., 0] >= -1e-10, axis=-1)
    return np.where(inside, -distance, distance)


def turn_distance(local):
    """Signed distance to the union of the body and in-place stepping region."""
    distances = []
    for half_size in TURN_ENVELOPES:
        d = np.abs(local)-half_size
        distances.append(np.linalg.norm(np.maximum(d, 0), axis=-1)+np.minimum(d.max(axis=-1), 0))
    return np.minimum.reduce(distances)


class TurnProgress:
    def __init__(self, yaw):
        self.last = float(yaw)
        self.radians = 0.

    def update(self, yaw):
        delta = float(yaw)-self.last
        self.radians += math.atan2(math.sin(delta), math.cos(delta))
        self.last = float(yaw)
        return self.degrees

    @property
    def degrees(self):
        return math.degrees(self.radians)


class YawCommand:
    """Ramp angular commands and reduce them before the speed guard trips."""
    def __init__(self, limit=.4):
        self.value = 0.
        self.limit = limit

    def update(self, desired, measured, dt):
        desired = float(np.clip(desired, -self.limit, self.limit))
        if abs(measured) > max(.6, self.limit*1.25):
            desired = 0.
        step = (TURN_ACCELERATION if self.limit > .4 else .8)*min(max(dt, 0.), .1)
        self.value += float(np.clip(desired-self.value, -step, step))
        return self.value


def map_frame(existing, points, resolution, position):
    keys = merge_voxels(existing, points, resolution)
    if len(keys) > 400000 and position is not None:
        center = np.asarray(position)/resolution
        order = np.argsort(np.sum((keys-center)**2, axis=1))[:400000]
        keys = keys[order]
    return keys, point_packet(1, keys.astype(np.float32)*resolution), point_packet(2, points+resolution/2)


def merge_voxels(existing, points, resolution):
    """Native Go2 points already use odom coordinates and denote voxel corners."""
    valid = points[np.all(np.isfinite(points), axis=1)]
    keys = np.rint((valid+resolution/2)/resolution).astype(np.int32)
    return np.unique(np.concatenate((existing, keys)), axis=0)


def point_packet(kind, points):
    return struct.pack("<II", kind, len(points)) + np.asarray(points, dtype="<f4").tobytes()


def rotation_obstacles(points, position, resolution):
    """Classify nearby occupied voxel centres using the measured ground band."""
    floor_candidates = points[points[:, 2] < position[2]-.15, 2]
    if len(floor_candidates) < 20:
        raise ValueError("Not enough ground points to verify the turning envelope")
    z, counts = np.unique(floor_candidates, return_counts=True)
    floor = float(z[np.argmax(counts)])
    near = points[np.linalg.norm(points[:, :2]-position[:2], axis=1) < 1.5]
    if len(near) < 20:
        raise ValueError("Insufficient nearby lidar coverage")
    layers = np.rint((near[:, 2]-floor)/resolution).astype(int)
    ceiling = collision_ceiling(position[2], floor+resolution/2, resolution)
    high = near[(layers >= 3) & (near[:, 2]+resolution/2 <= ceiling)]
    low = near[layers == 2]
    cells = {tuple(p) for p in np.rint((high[:, :2]+resolution/2)/resolution).astype(int)}
    steps = math.ceil(.10/resolution)
    offsets = [(x, y) for x in range(-steps, steps+1) for y in range(-steps, steps+1)
               if (x*resolution)**2+(y*resolution)**2 <= .10**2+1e-8]
    keep = [p for p, (x, y) in zip(low, np.rint((low[:, :2]+resolution/2)/resolution).astype(int))
            if any((x+dx, y+dy) in cells for dx, dy in offsets)]
    occupied = np.concatenate((high, np.asarray(keep).reshape(-1, 3)))
    return np.unique(occupied[:, :2]+resolution/2, axis=0), floor


def turn_lookahead(speed):
    """Cover the next command, feedback gap and braking at the measured rate."""
    return max(.18, min(.5, abs(speed)*.65+.05))


def swept_clearance(centers, position, yaw, angle, resolution, body_only=False):
    """Minimum clearance of the selected envelope during a short signed turn."""
    if not len(centers):
        return 1.
    delta = np.asarray(centers)-np.asarray(position[:2])
    if body_only:
        radii = np.linalg.norm(delta, axis=1)
        # Inner/outer circles bound the closest distance at every heading.
        # Points outside this radius cannot change the minimum clearance.
        delta = delta[radii <= radii.min()-_BODY_INRADIUS+_BODY_RADIUS+1e-10]
    gap = math.inf
    count = max(1, math.ceil(abs(angle)/.04))
    for heading in np.linspace(yaw, yaw+angle, count+1):
        c, s = math.cos(heading), math.sin(heading)
        local = delta@np.array([[c, -s], [s, c]])
        signed = body_distance(local) if body_only else turn_distance(local)
        gap = min(gap, float(signed.min()))
    # The sampling allowance covers the corner's arc between tested headings.
    return gap-resolution/math.sqrt(2)-.01


def rotation_clearance(points, position, resolution, yaw=0., angle=.18):
    centers, floor = rotation_obstacles(points, position, resolution)
    return swept_clearance(centers, position, yaw, angle, resolution), floor
