"""Local ground-grid planning in Go2 odometry coordinates (metres)."""
import heapq
from collections import deque
from functools import lru_cache
import math

import numpy as np
from go2_demo_core import BODY_POLYGON, body_distance, collision_ceiling, TURN_SPEED

RESERVE = .02
MAX_GOAL = 4.0
MAX_PATH = 6.0
ARRIVAL = .20
STOP_RADIUS = ARRIVAL-.02
# Keep room for tracking and braking inside the requested arrival region.
GOAL_REGION = STOP_RADIUS-.02
CRUISE_SPEED = .16
MAX_SPEED = .20
HEADING_LIMIT = .10
GROUND_TTL = 2.0
SELF_SHADOW_TTL = 5.0
CORNER_ARRIVAL = .04
MIN_APPROACH_PROGRESS = .15
# Integer lattice directions preserve grid alignment while offering 16 headings.
DIRECTIONS = ((1, 0), (2, 1), (1, 1), (1, 2), (0, 1), (-1, 2), (-1, 1), (-2, 1),
              (-1, 0), (-2, -1), (-1, -1), (-1, -2), (0, -1), (1, -2), (1, -1), (2, -1))
HEADINGS = tuple(math.atan2(y, x) for x, y in DIRECTIONS)


class PlanningError(ValueError):
    def __init__(self, message, issue):
        super().__init__(message)
        self.issue = issue

    def __reduce__(self):
        return type(self), (str(self), self.issue)


def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def dilate(mask, offsets):
    result = np.zeros_like(mask)
    height, width = mask.shape
    for dx, dy in offsets:
        if abs(dx) >= width or abs(dy) >= height:
            continue
        x0, x1 = max(0, dx), min(width, width+dx)
        y0, y1 = max(0, dy), min(height, height+dy)
        result[y0:y1, x0:x1] |= mask[y0-dy:y1-dy, x0-dx:x1-dx]
    return result


def disk(radius, resolution):
    n = math.ceil(radius/resolution)
    return [(x, y) for x in range(-n, n+1) for y in range(-n, n+1)
            if math.hypot(x, y)*resolution <= radius+1e-6]


def segment_positions(start, end, resolution):
    """Sample every rounded grid cell crossed by a measured line segment."""
    start, end = np.asarray(start, float), np.asarray(end, float)
    delta = end-start
    count = max(1, math.ceil(float(np.linalg.norm(delta))/(resolution/2)))
    times = list(np.linspace(0., 1., count+1))
    for a, b in zip(start/resolution, end/resolution):
        if abs(b-a) < 1e-12:
            continue
        low, high = sorted((a, b))
        for boundary in np.arange(math.ceil(low-.5), math.floor(high-.5)+1)+.5:
            t = (boundary-a)/(b-a)
            # Include each side of a rounding boundary, including ties.
            times.extend((max(0., t-1e-8), t, min(1., t+1e-8)))
    return start+np.unique(np.clip(times, 0., 1.))[:, None]*delta


def self_shadow_ground(observed, lower, origin, start, resolution):
    """Infer only enclosed lidar shadows directly under the standing body."""
    yy, xx = np.indices(observed.shape)
    world = (np.stack((xx, yy), axis=-1)+origin)*resolution
    local = ~observed & (np.linalg.norm(world-start, axis=-1) <= .60)
    visited, inferred = np.zeros_like(observed), np.zeros_like(observed)
    height, width = observed.shape
    for sy, sx in np.argwhere(local):
        if visited[sy, sx]:
            continue
        visited[sy, sx] = True
        pending, cells, enclosed = deque([(int(sx), int(sy))]), [], True
        while pending:
            x, y = pending.popleft()
            cells.append((x, y))
            for nx, ny in ((x+1, y), (x-1, y), (x, y+1), (x, y-1)):
                if not (0 <= nx < width and 0 <= ny < height):
                    enclosed = False
                elif not observed[ny, nx]:
                    if not local[ny, nx]:
                        enclosed = False
                    elif not visited[ny, nx]:
                        visited[ny, nx] = True
                        pending.append((nx, ny))
        cells = np.asarray(cells)
        x, y = cells.T
        center = (cells.mean(axis=0)+origin)*resolution
        if (enclosed and np.linalg.norm(center-start) <= math.hypot(.38+RESERVE, .23+RESERVE)+resolution/math.sqrt(2)
                and np.max(np.ptp(cells, axis=0))*resolution <= .6+1e-9
                and not np.any(lower[y, x])):
            inferred[y, x] = True
    return inferred


@lru_cache(maxsize=256)
def footprint_offsets(yaw, resolution, heading_tolerance=HEADING_LIMIT):
    half_length, half_width = .38+RESERVE, .23+RESERVE
    padding = resolution/math.sqrt(2)
    bound = math.hypot(half_length+padding, half_width+padding)
    n = math.ceil(bound/resolution)
    x, y = np.meshgrid(np.arange(-n, n+1), np.arange(-n, n+1))
    inside = np.zeros_like(x, bool)
    for heading in (yaw-heading_tolerance, yaw, yaw+heading_tolerance):
        c, s = math.cos(heading), math.sin(heading)
        inside |= ((np.abs(c*x+s*y)*resolution <= half_length+padding) &
                   (np.abs(-s*x+c*y)*resolution <= half_width+padding))
    return list(zip(x[inside].tolist(), y[inside].tolist()))


@lru_cache(maxsize=256)
def turn_footprint_offsets(yaw, resolution):
    padding = RESERVE+resolution/math.sqrt(2)+.01
    bound = float(np.linalg.norm(BODY_POLYGON, axis=1).max())+padding
    n = math.ceil(bound/resolution)
    x, y = np.meshgrid(np.arange(-n, n+1), np.arange(-n, n+1))
    c, s = math.cos(yaw), math.sin(yaw)
    local = np.stack((c*x+s*y, -s*x+c*y), axis=-1)*resolution
    inside = body_distance(local) <= padding
    return list(zip(x[inside].tolist(), y[inside].tolist()))


class GroundGrid:
    def __init__(self, raw_points, position, yaw, resolution=.05, memory=None, stamp=0.):
        if not .04 <= resolution <= .06:
            raise ValueError("Lidar resolution is not supported by this navigation demo")
        self.resolution, self.yaw = resolution, float(yaw)
        self.start = np.asarray(position[:2], float)
        map_radius = MAX_GOAL+.7
        self.origin = np.rint(self.start/resolution).astype(int)-math.ceil(map_radius/resolution)
        size = 2*math.ceil(map_radius/resolution)+1
        points = np.asarray(raw_points)
        points = points[np.all(np.isfinite(points), axis=1)]
        points = points[np.linalg.norm(points[:, :2]-self.start, axis=1) < map_radius-resolution]
        floor_points = points[points[:, 2] < position[2]-.15]
        if len(floor_points) < 30:
            raise ValueError("Insufficient ground observations")
        # Retain the native half-grid corner height.
        floor_z, floor_counts = np.unique(floor_points[:, 2], return_counts=True)
        self.floor = float(floor_z[np.argmax(floor_counts)])+resolution/2
        layers = np.rint((points[:, 2]+resolution/2-self.floor)/resolution).astype(int)
        xy = np.rint((points[:, :2]+resolution/2)/resolution).astype(int)-self.origin
        inside = np.all((xy >= 0) & (xy < size), axis=1)
        xy, layers, points = xy[inside], layers[inside], points[inside]
        observed = np.zeros((size, size), bool)
        ground = xy[(layers >= -1) & (layers <= 1)]
        observed[ground[:, 1], ground[:, 0]] = True
        # Keep only measured ground, never the inferred ground under the feet.
        seen = np.full(observed.shape, -np.inf)
        shadow_seen = np.full(observed.shape, -np.inf)
        if (memory is not None and memory["resolution"] == resolution and
                abs(memory["floor"]-self.floor) < resolution/4 and stamp >= memory["stamp"]):
            shift = np.asarray(memory["origin"])-self.origin
            old_y, old_x = np.nonzero(stamp-memory["seen"] <= GROUND_TTL)
            new_x, new_y = old_x+shift[0], old_y+shift[1]
            keep = (new_x >= 0) & (new_x < size) & (new_y >= 0) & (new_y < size)
            seen[new_y[keep], new_x[keep]] = memory["seen"][old_y[keep], old_x[keep]]
            if "shadow_seen" in memory:
                old_y, old_x = np.nonzero(stamp-memory["shadow_seen"] <= SELF_SHADOW_TTL)
                new_x, new_y = old_x+shift[0], old_y+shift[1]
                keep = (new_x >= 0) & (new_x < size) & (new_y >= 0) & (new_y < size)
                shadow_seen[new_y[keep], new_x[keep]] = memory["shadow_seen"][old_y[keep], old_x[keep]]
        seen[observed] = stamp
        self.ground_memory = {"seen": seen, "origin": self.origin.copy(), "stamp": stamp,
                              "floor": self.floor, "resolution": resolution}
        observed = stamp-seen <= GROUND_TTL
        # Close scan gaps by one cell; larger unobserved areas remain blocked.
        observed = dilate(observed, disk(resolution*1.01, resolution))
        lower = np.zeros_like(observed)
        below = xy[layers <= -2]
        lower[below[:, 1], below[:, 0]] = True
        lower &= ~observed
        # The current standing envelope already has support. Use it as a
        # boundary when closing enclosed body shadows; keep it out of measured
        # ground history, and never infer a connection to exterior unknown space.
        standing_support = np.zeros_like(observed)
        center = self.cell(self.start)
        offsets = footprint_offsets(float(yaw), resolution)
        for dx, dy in offsets:
            x, y = center[0]+dx, center[1]+dy
            if 0 <= x < size and 0 <= y < size:
                standing_support[y, x] = True
        self.inferred_ground = self_shadow_ground(observed | standing_support, lower,
                                                 self.origin, self.start, resolution)
        # Retain recently supported ground separately from measured returns.
        # Otherwise a rear-foot shadow is forgotten as soon as the body moves.
        shadow_seen[(self.inferred_ground | standing_support) & ~observed] = stamp
        shadow_seen[observed | lower] = -np.inf
        self.ground_memory["shadow_seen"] = shadow_seen
        self.inferred_ground = stamp-shadow_seen <= SELF_SHADOW_TTL
        observed |= self.inferred_ground
        occupied = np.zeros_like(observed)
        self.ceiling = collision_ceiling(position[2], self.floor, resolution)
        high = xy[(layers >= 3) & (points[:, 2]+resolution/2 <= self.ceiling)]
        occupied[high[:, 1], high[:, 0]] = True
        # Layer 1 belongs to the ground tolerance. It must not also become an
        # obstacle merely because a vertical object is within 10 cm.
        low = xy[layers == 2]
        supported = dilate(occupied, disk(.10, resolution))
        near_high = low[supported[low[:, 1], low[:, 0]]]
        occupied[near_high[:, 1], near_high[:, 0]] = True
        occupied |= lower
        # Estimated walking envelope: body and feet, plus nominal 2 cm reserve,
        # voxel half diagonal and the permitted heading correction sweep.
        offsets = footprint_offsets(float(yaw), resolution)
        bound = math.hypot(.38+RESERVE+resolution/math.sqrt(2),
                           .23+RESERVE+resolution/math.sqrt(2))
        observed |= standing_support
        unsafe = occupied | ~observed
        self.unsafe = unsafe
        self.blocked = dilate(unsafe, offsets)
        edge = self.edge = math.ceil(bound/resolution)
        self.blocked[:edge] = self.blocked[-edge:] = True
        self.blocked[:, :edge] = self.blocked[:, -edge:] = True
        self.occupied, self.observed = occupied, observed
        self.heading_masks = {float(yaw): self.blocked}
        self.turn_masks, self.motion_masks = {}, {}
        self.risk_masks = {}
        self.failed_departures = []

    def avoid_failed_departures(self, departures):
        """Exclude failed travel directions locally, without inventing occupancy."""
        self.failed_departures = list(departures)
        self.motion_masks.clear()

    def departure_safe(self, start, end):
        delta = np.asarray(end)-start
        if np.linalg.norm(delta) < 1e-8:
            return True
        heading = math.atan2(delta[1], delta[0])
        positions = segment_positions(start, end, self.resolution)
        return not any(abs(wrap(heading-record['yaw'])) < .55 and
                       np.any(np.linalg.norm(positions-record['position'], axis=1) <= .30)
                       for record in self.failed_departures)

    def departure_mask(self, heading):
        mask = np.zeros_like(self.unsafe)
        if not self.failed_departures:
            return mask
        yy, xx = np.indices(mask.shape)
        world = (np.stack((xx, yy), axis=-1)-self.cell(self.start))*self.resolution+self.start
        for record in self.failed_departures:
            if abs(wrap(heading-record['yaw'])) < .55:
                mask |= np.linalg.norm(world-record['position'], axis=-1) <= .30
        return mask

    def cell(self, xy):
        return tuple((np.rint(np.asarray(xy)/self.resolution).astype(int)-self.origin).tolist())

    def world(self, cell):
        return (np.asarray(cell)+self.origin)*self.resolution

    def free(self, xy):
        x, y = self.cell(xy)
        return 0 <= y < self.blocked.shape[0] and 0 <= x < self.blocked.shape[1] and not self.blocked[y, x]

    def heading_mask(self, yaw):
        yaw = float(yaw)
        if yaw not in self.heading_masks:
            if len(self.heading_masks) >= 64:
                self.heading_masks.clear()
            mask = dilate(self.unsafe, footprint_offsets(yaw, self.resolution))
            mask[:self.edge] = mask[-self.edge:] = True
            mask[:, :self.edge] = mask[:, -self.edge:] = True
            self.heading_masks[yaw] = mask
        return self.heading_masks[yaw]

    def pose_safe(self, position, yaw):
        x, y = self.cell(position)
        mask = self.heading_mask(yaw)
        return 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and not mask[y, x]

    def turn_safe(self, position, start_yaw, end_yaw):
        angle = wrap(end_yaw-start_yaw)
        count = max(1, math.ceil(abs(angle)/.04))
        # Check the body/head sweep; the native controller places the feet.
        x, y = self.cell(position)
        for yaw in np.linspace(start_yaw, start_yaw+angle, count+1):
            offsets = np.asarray(turn_footprint_offsets(float(yaw), self.resolution))
            cells = offsets+[x, y]
            if np.any(cells < 0) or np.any(cells[:, 0] >= self.unsafe.shape[1]) or np.any(cells[:, 1] >= self.unsafe.shape[0]):
                return False
            if np.any(self.unsafe[cells[:, 1], cells[:, 0]]):
                return False
        return True

    def segment_safe(self, start, end, yaw=None):
        mask = self.blocked if yaw is None else self.heading_mask(yaw)
        cells = np.rint(segment_positions(start, end, self.resolution)/self.resolution).astype(int)-self.origin
        return bool(np.all(cells >= 0) and np.all(cells[:, 0] < mask.shape[1]) and
                    np.all(cells[:, 1] < mask.shape[0]) and not np.any(mask[cells[:, 1], cells[:, 0]]))

    def envelope_issue(self, positions, headings, turning=False, heading_tolerance=HEADING_LIMIT):
        """Identify measured occupancy before reporting missing ground."""
        unknown = None
        for position, yaw in zip(positions, headings):
            center = np.asarray(self.cell(position))
            offsets = (turn_footprint_offsets(float(yaw), self.resolution) if turning else
                       footprint_offsets(float(yaw), self.resolution, heading_tolerance))
            cells = np.asarray(offsets)+center
            inside = np.all((cells >= 0) & (cells < self.unsafe.shape[0]), axis=1)
            if not np.all(inside):
                return {"kind": "outside", "point": self.world(cells[~inside][0]).tolist(),
                        "robot_position": np.asarray(position).tolist()}
            occupied = self.occupied[cells[:, 1], cells[:, 0]]
            if np.any(occupied):
                return {"kind": "occupied", "point": self.world(cells[occupied][0]).tolist(),
                        "robot_position": np.asarray(position).tolist()}
            missing = ~self.observed[cells[:, 1], cells[:, 0]]
            if unknown is None and np.any(missing):
                unknown = {"kind": "unknown", "point": self.world(cells[missing][0]).tolist(),
                           "robot_position": np.asarray(position).tolist()}
        return unknown

    def segment_issue(self, start, end, yaw):
        if self.segment_safe(start, end, yaw):
            return None
        positions = segment_positions(start, end, self.resolution)
        return self.envelope_issue(positions, [yaw]*len(positions))

    def turn_issue(self, position, start_yaw, end_yaw):
        if self.turn_safe(position, start_yaw, end_yaw):
            return None
        angle = wrap(end_yaw-start_yaw)
        count = max(1, math.ceil(abs(angle)/.04))
        return self.envelope_issue([position]*(count+1), np.linspace(start_yaw, start_yaw+angle, count+1), turning=True)

    def route_safe(self, path, yaw):
        for a, b in zip(path, path[1:]):
            delta = np.asarray(b)-a
            if np.linalg.norm(delta) < 1e-8:
                continue
            heading = math.atan2(delta[1], delta[0])
            if (not self.departure_safe(a, b) or not self.turn_safe(a, yaw, heading)
                    or not self.segment_safe(a, b, heading)):
                return False
            yaw = heading
        return True

    def result(self, path, goal):
        length, yaw, actions, turns = 0., self.yaw, [], 0
        for segment, (a, b) in enumerate(zip(path, path[1:]), 1):
            a, b = np.asarray(a), np.asarray(b)
            delta = b-a
            distance = float(np.linalg.norm(delta))
            heading = math.atan2(delta[1], delta[0])
            angle = wrap(heading-yaw)
            if abs(angle) > .05:
                actions.append({"type": "turn", "at": a.tolist(), "from_yaw": yaw,
                                "to_yaw": heading, "angle_rad": angle, "segment": segment})
                turns += 1
            actions.append({"type": "walk", "from": a.tolist(), "to": b.tolist(),
                            "yaw": heading, "distance_m": distance, "segment": segment})
            length, yaw = length+distance, heading
        duration = length/CRUISE_SPEED+sum(abs(a["angle_rad"])/TURN_SPEED+.4 for a in actions
                                          if a["type"] == "turn")+.4*max(0, len(path)-2)
        return {"path": [np.asarray(p).tolist() for p in path], "length_m": length, "floor": self.floor,
                "arrival_target": np.asarray(path[-1]).tolist(),
                "goal_offset_m": float(np.linalg.norm(np.asarray(path[-1])-goal)),
                "goal_region_m": GOAL_REGION, "collision_ceiling_m": self.ceiling,
                "self_shadow_cells": int(self.inferred_ground.sum()),
                "resolution_m": self.resolution, "reserve_m": RESERVE,
                "arrival_m": ARRIVAL, "stop_radius_m": STOP_RADIUS, "max_speed_m_s": MAX_SPEED,
                "motion_model": "turn_then_forward", "planner": "heading_lattice_16_region",
                "actions": actions, "walk_count": len(path)-1, "turn_count": turns,
                "estimated_motion_s": duration, "partial": False}

    def plan(self, goal, approach=False):
        goal = np.asarray(goal, dtype=float)
        if goal.shape != (2,) or not np.all(np.isfinite(goal)):
            raise ValueError("Target must contain finite ground coordinates")
        distance = float(np.linalg.norm(goal-self.start))
        if distance > MAX_GOAL:
            raise ValueError(f"Select a target within {MAX_GOAL:g} m of the robot")
        if distance <= ARRIVAL:
            return self.result([self.start], goal)
        # The controller accepts an arrival region, so the search must not
        # require the exact clicked centre to fit at a lattice heading.
        center = np.rint((goal-self.start)/self.resolution).astype(int)
        n = math.ceil(GOAL_REGION/self.resolution)
        candidates = [self.start+(center+[dx, dy])*self.resolution
                      for dx in range(-n, n+1) for dy in range(-n, n+1)
                      if np.linalg.norm(self.start+(center+[dx, dy])*self.resolution-goal) <= GOAL_REGION+1e-9]
        direct = [[self.start, p] for p in [goal, *candidates]
                  if self.route_safe([self.start, p], self.yaw)]
        if direct:
            return self.result(min(direct, key=self.route_cost), goal)
        goal_issue, goal_message = None, None
        if not any(self.pose_safe(p, yaw) for p in candidates for yaw in HEADINGS):
            issues = [self.envelope_issue([p], [yaw]) for p in candidates for yaw in HEADINGS]
            issue = min(issues, key=lambda i: {"unknown": 0, "occupied": 1, "outside": 2}[i["kind"]])
            issue.update(stage="goal", floor=self.floor, collision_ceiling_m=self.ceiling)
            if issue["kind"] == "unknown":
                message = "Ground coverage near the target is incomplete; scan again to verify reachability"
            elif issue["kind"] == "occupied":
                message = "All candidate endpoints near the target have occupied body or foot space; trigger marked on the map"
            else:
                message = "Target lies outside the current local lidar map; select a new target"
            goal_issue, goal_message = issue, message
            if not approach:
                raise PlanningError(message, issue)
        try:
            result = self._plan(goal, approach)
        except ValueError:
            if goal_issue is not None:
                raise PlanningError(goal_message, goal_issue)
            raise
        if result["partial"]:
            result["approach_reason"] = goal_message or "Current observations do not connect the full route"
            result["remaining_obstruction"] = goal_issue
        return result

    def risk_mask(self, yaw):
        """Soft cost for routes close to the existing collision/ground boundary."""
        yaw = float(yaw)
        if yaw not in self.risk_masks:
            blocked = self.heading_mask(yaw)
            near = dilate(blocked, disk(self.resolution*1.01, self.resolution))
            farther = dilate(near, disk(self.resolution*1.01, self.resolution))
            self.risk_masks[yaw] = .75*near+.25*farther
        return self.risk_masks[yaw]

    def leg_cost(self, a, b, previous_yaw, initial=False):
        delta = np.asarray(b)-a
        distance = float(np.linalg.norm(delta))
        heading = math.atan2(delta[1], delta[0])
        angle = abs(wrap(heading-previous_yaw))
        count = max(1, math.ceil(distance/(self.resolution/2)))
        cells = np.rint(np.linspace(a, b, count+1)/self.resolution).astype(int)-self.origin
        risk = float(self.risk_mask(heading)[cells[:, 1], cells[:, 0]].mean())
        return distance/CRUISE_SPEED*(1+risk)+angle/TURN_SPEED+(.4 if angle > .05 else 0.)+(.4 if not initial else 0.)

    def route_cost(self, path):
        yaw, cost = self.yaw, 0.
        for index, (a, b) in enumerate(zip(path, path[1:])):
            cost += self.leg_cost(a, b, yaw, initial=index == 0)
            yaw = math.atan2(*(np.asarray(b)-a)[::-1])
        return cost

    def motion_mask(self, heading):
        if heading not in self.motion_masks:
            delta = np.array(DIRECTIONS[heading])
            phase = self.start/self.resolution-np.rint(self.start/self.resolution)
            positions = segment_positions(phase, phase+delta, 1.)
            cells = np.unique(np.rint(positions).astype(int), axis=0)
            # Obstacles are expanded back toward every possible segment origin.
            mask = self.heading_mask(HEADINGS[heading]) | self.departure_mask(HEADINGS[heading])
            self.motion_masks[heading] = dilate(mask, -cells)
        return self.motion_masks[heading]

    def transition_mask(self, before, after):
        key = before, after
        if key not in self.turn_masks:
            start_yaw = self.yaw if before == len(HEADINGS) else HEADINGS[before]
            angle = wrap(HEADINGS[after]-start_yaw)
            count = max(1, math.ceil(abs(angle)/.04))
            offsets = set()
            for yaw in np.linspace(start_yaw, start_yaw+angle, count+1):
                offsets.update(turn_footprint_offsets(float(yaw), self.resolution))
            # The head outline is asymmetric, so reflection is required here.
            self.turn_masks[key] = dilate(self.unsafe, [(-x, -y) for x, y in offsets])
        return self.turn_masks[key]

    def _plan(self, goal, approach=False):
        """Search checked poses using time/clearance cost and the configured path budget."""
        goal = np.asarray(goal, float)
        start_cell = np.asarray(self.cell(self.start))
        start = (*start_cell, len(HEADINGS))
        def world(node):
            return self.start+(np.asarray(node[:2])-start_cell)*self.resolution
        def remaining(node):
            return max(0., float(np.linalg.norm(goal-world(node)))-GOAL_REGION)
        # Keep nondominated (distance, cost) labels. A faster but longer arrival
        # at a pose must not discard a shorter route needed by the path budget.
        records = [(start, 0., 0., None)]
        labels, active = {start: [0]}, {0}
        queue = [(remaining(start)/CRUISE_SPEED, 0., 0)]
        finish, nearest = None, None
        start_distance = float(np.linalg.norm(goal-self.start))
        progress = min(.50, max(MIN_APPROACH_PROGRESS, .10*start_distance))
        nearest_distance = start_distance-progress
        while queue:
            _, elapsed, label = heapq.heappop(queue)
            if label not in active:
                continue
            node, travelled, _, _ = records[label]
            x, y, before = node
            position = world(node)
            if approach and before < len(HEADINGS):
                distance = float(np.linalg.norm(position-goal))
                if distance < nearest_distance-1e-9:
                    nearest, nearest_distance = label, distance
            if before < len(HEADINGS) and np.linalg.norm(position-goal) <= GOAL_REGION+1e-9:
                if self.pose_safe(position, HEADINGS[before]):
                    finish = label
                    break
            previous_yaw = self.yaw if before == len(HEADINGS) else HEADINGS[before]
            for after, (dx, dy) in enumerate(DIRECTIONS):
                nx, ny = x+dx, y+dy
                if not (0 <= nx < self.unsafe.shape[1] and 0 <= ny < self.unsafe.shape[0]):
                    continue
                step = math.hypot(dx, dy)*self.resolution
                length, nxt = travelled+step, (nx, ny, after)
                if length+remaining(nxt) > MAX_PATH+1e-9 or self.motion_mask(after)[y, x]:
                    continue
                angle = abs(wrap(HEADINGS[after]-previous_yaw))
                risk = self.risk_mask(HEADINGS[after])
                effort = elapsed+step/CRUISE_SPEED*(1+max(risk[y, x], risk[ny, nx]))+angle/TURN_SPEED
                if angle > .05:
                    effort += .4+(.4 if before < len(HEADINGS) else 0.)
                old = labels.get(nxt, [])
                if any(records[i][1] <= length+1e-9 and records[i][2] <= effort+1e-9 for i in old):
                    continue
                if self.transition_mask(before, after)[y, x]:
                    continue
                keep = []
                for i in old:
                    if length <= records[i][1]+1e-9 and effort <= records[i][2]+1e-9:
                        active.discard(i)
                    else:
                        keep.append(i)
                index = len(records)
                records.append((nxt, length, effort, label))
                labels[nxt] = keep+[index]
                active.add(index)
                heapq.heappush(queue, (effort+remaining(nxt)/CRUISE_SPEED, effort, index))
        partial = finish is None and nearest is not None
        if partial:
            finish = nearest
        if finish is None:
            raise ValueError(f"No checked route of at most {MAX_PATH:g} m with walking and turning in the observed area")
        points = []
        while finish is not None:
            node, _, _, finish = records[finish]
            points.append(world(node))
        points.reverse()
        path = [points[0]]
        for a, b, c in zip(points, points[1:], points[2:]):
            u, v = b-a, c-b
            if abs(u[0]*v[1]-u[1]*v[0]) > 1e-9 or np.dot(u, v) <= 0:
                path.append(b)
        path.append(points[-1])
        # A shortcut must improve the same time/clearance objective used by
        # search, as well as pass every turn and full-footprint walk check.
        cost = self.route_cost(path)
        while True:
            best, best_cost = None, cost
            for start in range(len(path)-2):
                for end in range(start+2, len(path)):
                    candidate = path[:start+1]+path[end:]
                    score = self.route_cost(candidate)
                    if score >= best_cost-1e-6 or not self.route_safe(candidate, self.yaw):
                        continue
                    best, best_cost = candidate, score
            if best is None:
                break
            path, cost = best, best_cost
        # Centre waypoints in their voxels so a straight leg does not track
        # a grid-rounding boundary with only a few millimetres to spare.
        centred = [path[0]]+[self.world(self.cell(p)) for p in path[1:]]
        if ((partial or np.linalg.norm(centred[-1]-goal) <= GOAL_REGION+1e-9) and
            (not partial or start_distance-np.linalg.norm(centred[-1]-goal) >= progress-1e-9) and
            sum(np.linalg.norm(b-a) for a, b in zip(centred, centred[1:])) <= MAX_PATH and
            self.route_safe(centred, self.yaw)):
            path = centred
        if not self.route_safe(path, self.yaw):
            raise ValueError("Insufficient turning space along the route; rescan before selecting a target")
        result = self.result(path, goal)
        result["partial"] = partial
        return result


def path_distance(position, path):
    p = np.asarray(position)
    distances = []
    for a, b in zip(path, path[1:]):
        a, b = np.asarray(a), np.asarray(b)
        d = b-a
        t = np.clip(np.dot(p-a, d)/max(np.dot(d, d), 1e-10), 0, 1)
        distances.append(np.linalg.norm(p-(a+t*d)))
    return float(min(distances))
