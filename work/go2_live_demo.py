"""Local point-cloud/photo viewer with planned navigation and a two-turn demo."""
import argparse
import asyncio
import contextlib
import io
import json
import logging
import math
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from collections import deque
import signal
import time
import uuid
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from PIL import Image
from aiohttp import web, WSMsgType
from aiortc.mediastreams import MediaStreamError
from unitree_webrtc_connect import UnitreeWebRTCConnection, WebRTCConnectionMethod, RTC_TOPIC
from unitree_webrtc_connect.unitree_auth import RobotBusyError

from go2_demo_core import (TurnProgress, YawCommand, map_frame, merge_voxels, point_packet,
                           rotation_obstacles, swept_clearance, turn_lookahead, TURN_SPEED, TURN_ACCELERATION)
from go2_navigation import GroundGrid, PlanningError, ARRIVAL, STOP_RADIUS, CRUISE_SPEED, MAX_SPEED, MAX_PATH, HEADING_LIMIT, CORNER_ARRIVAL, wrap, path_distance
from go2_photo_depth import CameraModel, build_fusion
from go2_panorama import build_panorama
from go2_spatial import stations, select_keyframe, build_spatial_panorama, save_capture, MAX_PHOTOS
from go2_network import DEFAULT_DEVICE_FILE, DiscoveryError, read_device, resolve_robot_ip, robot_address

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / "outputs/go2-live-demo"
RIGHT_STICK_PER_YAW = .375
MOTOR_FAULT_LABELS = {1: "overcurrent", 2: "overvoltage", 4: "driver overtemperature", 8: "bus undervoltage",
                      16: "winding overtemperature", 32: "encoder fault", 256: "motor communication lost"}


def fault_text(fault):
    source, code = fault["source"], int(fault["code"], 16)
    labels = [text for bit, text in MOTOR_FAULT_LABELS.items() if code & bit] if 300 <= source < 312 else []
    detail = " / ".join(labels)
    return f"{source}-0x{code:X}"+(f"（{detail}）" if detail else "")


class MotionPause(RuntimeError):
    """A condition that requires zero velocity and fresh stationary feedback."""


class Demo:
    def __init__(self, robot_ip="auto", key_file=DEFAULT_DEVICE_FILE):
        self.robot_address = robot_ip
        self.key_file = Path(key_file).expanduser()
        self.robot_ip = None
        self.compute_pool = None
        self.loop_lag_s = 0.
        self.sport_sequence = 0
        self.sport_source_stamp = None
        self.control_epoch = None
        self.command_samples = []
        self.conn = None
        self.connected = False
        self.faults = {}
        self.faults_seen = False
        self.power_history = deque(maxlen=600)
        self.fault_events = deque(maxlen=128)
        self.fault_capture_task = None
        self.last_fault_record = None
        self.sport = None
        self.low = None
        self.cloud = None
        self.metadata = None
        self.state_time = self.lidar_time = self.camera_time = -1e9
        self.low_time = -1e9
        self.geometry_time = -1e9
        self.clearance = None
        self.floor = 0.
        self.lidar_frames = self.camera_frames = 0
        self.map_keys = np.empty((0, 3), np.int32)
        self.map_packet = point_packet(1, np.empty((0, 3)))
        self.live_packet = point_packet(2, np.empty((0, 3)))
        self.map_version = 0
        self.scene_generation = 0
        self.scene_reset_at = -1e9
        self.scene_started_unix = 0.
        self.recorded_map = False
        self.map_lock = asyncio.Lock()
        self.jpeg = None
        self.photos = {}
        self.photo_sequence = 0
        self.spatial_capture = False
        self.spatial_message = 'Capture multiple stations along a route, then build spatial views'
        self.spatial_save_lock = asyncio.Lock()
        self.spatial_saved_at = -math.inf
        self.sockets = {}
        self.client_sequence = 0
        self.owner = None
        self.phase = "connecting"
        self.message = "Connecting to the robot"
        self.abort = asyncio.Event()
        self.abort_reason = ""
        self.motion_task = None
        self.tracker = None
        self.measuring = False
        self.origin = None
        self.peak_shift = 0.
        self.session = None
        self.samples = []
        self.report = {}
        self.video_tasks = set()
        self.tasks = []
        self.running = True
        self.action_lock = asyncio.Lock()
        self.turn_token = uuid.uuid4().hex
        self.turn_latched = False
        self.connection_epoch = 0
        self.navigation = None
        self.ground_memory = None
        self.plan_sequence = 0
        self.fusion = None
        self.fusion_version = 0
        self.fusion_generation = 0
        self.fusion_task = None
        self.fusion_camera = CameraModel()
        self.fusion_message = "Photo 3D builds after the two-turn demo, or update manually"
        self.panorama = None
        self.panorama_version = 0
        self.panorama_message = "Pose-aligned panorama during capture; seams align and blend after capture"
        self.turn_points = None
        self.turn_resolution = .05
        self.turn_direction = 1
        self.turn_channel = "joystick"
        self.joystick_active = False
        self.joystick_ly = 0.
        self.joystick_rx = 0.

    def age(self, stamp):
        return round(time.monotonic()-stamp, 3) if stamp > 0 else None

    def motion_active(self):
        return self.motion_task is not None and not self.motion_task.done()

    def fusion_busy(self):
        return self.fusion_task is not None and not self.fusion_task.done()

    def status(self):
        state_age, lidar_age = self.age(self.state_time), self.age(self.lidar_time)
        ready = (self.connected and state_age is not None and state_age < .3 and
                 lidar_age is not None and lidar_age < .5 and self.clearance is not None and
                 time.monotonic()-self.geometry_time < .8)
        blocked = None
        stand_blocked = None
        if self.connected:
            try:
                self.check_stand()
            except RuntimeError as error:
                stand_blocked = str(error)
            try:
                self.check_robot(include_origin=False)
                if np.linalg.norm(self.sport["velocity"][:2]) > .025 or abs(self.sport["yaw_speed"]) > .04:
                    blocked = "Robot has not settled"
            except RuntimeError as error:
                blocked = str(error)
            ready = blocked is None
        pose = None
        shift = 0.
        if self.sport:
            p = self.sport["position"]
            rpy = self.sport["imu_state"]["rpy"]
            pose = {"position": p, "yaw": rpy[2], "rpy": rpy}
            if self.origin is not None:
                shift = float(np.linalg.norm(np.asarray(p[:2])-self.origin))
        nav_blocked = None
        try:
            self.check_navigation(stationary=not self.motion_active())
        except RuntimeError as error:
            nav_blocked = str(error)
        return {"type": "status", "connected": self.connected, "ready": ready,
                "connection_epoch": self.connection_epoch,
                "robot_ip": self.robot_ip, "robot_address": self.robot_address,
                "phase": self.phase, "message": blocked if blocked and not self.motion_active() else self.message,
                "blocked": blocked, "faults": list(self.faults.values()), "active": self.motion_active(),
                "stand_ready": self.connected and stand_blocked is None,
                "stand_blocked": stand_blocked, "map_source": "recorded" if self.recorded_map else "live",
                "degrees": self.tracker.degrees if self.tracker else 0., "target": 720,
                "shift_m": shift, "peak_shift_m": self.peak_shift,
                "pose": pose, "floor": self.floor, "clearance_m": self.clearance,
                "map_points": len(self.map_keys), "lidar_frames": self.lidar_frames,
                "camera_frames": self.camera_frames, "state_age_s": state_age,
                "lidar_age_s": lidar_age, "camera_age_s": self.age(self.camera_time), "loop_lag_s": self.loop_lag_s,
                "photo_count": len(self.photos), "photo_coverage": len(self.photos)/24,
                "photo_stations": len(stations(self.photos.values())),
                "spatial_capture": self.spatial_capture, "spatial_message": self.spatial_message,
                "battery": (self.low or {}).get("bms_state", {}).get("soc"),
                "turn_latched": self.turn_latched, "turn_token": self.turn_token,
                "navigation": ({**self.navigation, "expired": time.monotonic() > self.navigation["expires_at"]}
                               if self.navigation else None), "nav_ready": nav_blocked is None,
                "nav_blocked": nav_blocked,
                "fusion_busy": self.fusion_busy(), "fusion_message": self.fusion_message,
                "fusion_id": self.fusion["id"] if self.fusion else None,
                "scene_generation": self.scene_generation,
                "panorama_message": self.panorama_message,
                "report": self.report if self.phase in ("complete", "stopped", "error") else None}

    def sport_callback(self, msg):
        data = msg["data"]
        stamp = data.get("stamp")
        if stamp is not None:
            source_stamp = int(stamp["sec"])*1000000000+int(stamp["nanosec"])
            if self.sport_source_stamp is not None and source_stamp <= self.sport_source_stamp:
                return
            self.sport_source_stamp = source_stamp
        self.sport = data
        self.sport_sequence += 1
        self.state_time = time.monotonic()
        if self.tracker is not None and self.measuring:
            self.tracker.update(self.sport["imu_state"]["rpy"][2])
            if self.origin is not None:
                shift = float(np.linalg.norm(np.asarray(self.sport["position"][:2])-self.origin))
                self.peak_shift = max(self.peak_shift, shift)
            self.samples.append({"t": self.state_time, "yaw": self.sport["imu_state"]["rpy"][2],
                                 "degrees": self.tracker.degrees, "position": self.sport["position"],
                                 **self.feedback_sample()})

    def low_callback(self, msg):
        data = msg["data"]
        self.low = json.loads(data) if isinstance(data, str) else data
        self.low_time = time.monotonic()
        sport = self.sport or {}
        self.power_history.append({"t": self.low_time, "unix": time.time(),
            "power_v": self.low.get("power_v"), "bms": dict(self.low.get("bms_state", {})),
            "board_temperature": self.low.get("temperature_ntc1"),
            "motors": [{"temperature": m.get("temperature"), "lost": m.get("lost"),
                        "fault_bits": (m.get("reserve") or [0])[0]}
                       for m in self.low.get("motor_state", [])[:12]],
            "sport_stamp": sport.get("stamp"), "position": list(sport.get("position", [])),
            "rpy": list(sport.get("imu_state", {}).get("rpy", [])),
            "velocity": list(sport.get("velocity", [])), "yaw_speed": sport.get("yaw_speed"),
            "command": dict(self.command_samples[-1]) if self.command_samples else None})

    def fault_callback(self, msg):
        kind = msg.get("type")
        if kind not in ("errors", "add_error", "rm_error"):
            return
        previous = set(self.faults)
        rows = msg.get("data") or []
        if rows and not isinstance(rows[0], (list, tuple)):
            rows = [rows]
        if kind == "errors":
            self.faults = {}
            self.faults_seen = True
        for stamp, source, code in rows:
            key = (source, code)
            if kind == "rm_error":
                self.faults.pop(key, None)
            else:
                self.faults[key] = {"source": source, "code": format(code, "X"), "since": stamp}
        added = set(self.faults)-previous
        if added:
            self.fault_events.append({"t": time.monotonic(), "unix": time.time(),
                "faults": [dict(self.faults[key]) for key in sorted(added)]})
            if self.fault_capture_task is None or self.fault_capture_task.done():
                self.fault_capture_task = asyncio.create_task(self.capture_fault())

    async def capture_fault(self):
        started = time.monotonic()
        before = list(self.power_history)
        await asyncio.sleep(.5)
        payload = {"recorded_unix": time.time(), "connection_epoch": self.connection_epoch,
            "events": [event for event in self.fault_events if event["t"] >= started-.5],
            "power_samples": before+[sample for sample in self.power_history if sample["t"] > started],
            "active_faults": list(self.faults.values()), "phase": self.phase,
            "sport_state": self.sport, "low_state": self.low}
        folder = SITE / "faults"
        name = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d-%H%M%S-%f")+".json"
        def save():
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / name
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
            return str(path)
        try:
            self.last_fault_record = await asyncio.to_thread(save)
            logging.warning("Robot fault evidence saved: %s", self.last_fault_record)
        except OSError:
            logging.exception("Robot fault evidence could not be saved")

    def channel_faults(self, message):
        if isinstance(message, str):
            try:
                parsed = json.loads(message)
            except json.JSONDecodeError as error:
                logging.warning("Invalid robot JSON on fault listener: %s", error)
                return
            self.fault_callback(parsed)

    def lidar_callback(self, msg):
        payload = msg.get("data", {})
        points = payload.get("data", {}).get("points")
        if points is not None:
            self.cloud = np.asarray(points, np.float32)
            self.metadata = {k: v for k, v in payload.items() if k != "data"}
            self.lidar_time = time.monotonic()
            self.lidar_frames += 1

    @staticmethod
    def encode_image(frame):
        image = frame.to_image()
        image.thumbnail((960, 540))
        output = io.BytesIO()
        image.save(output, format="JPEG", quality=84)
        return output.getvalue()

    async def video(self, track):
        task = asyncio.current_task()
        self.video_tasks.add(task)
        encoded_at = -1e9
        try:
            while self.running:
                frame = await track.recv()
                self.camera_frames += 1
                now = time.monotonic()
                if now-encoded_at < .35:
                    continue
                pose = self.sport
                pose_time, cloud_time = self.state_time, self.lidar_time
                cloud, metadata = self.cloud, self.metadata
                generation, epoch = self.scene_generation, self.connection_epoch
                jpeg = await asyncio.to_thread(self.encode_image, frame)
                if generation != self.scene_generation or epoch != self.connection_epoch:
                    continue
                self.jpeg, self.camera_time = jpeg, now
                encoded_at = now
                if ((self.measuring or self.spatial_capture) and pose is not None and 0 <= now-pose_time < .15 and
                    cloud is not None and metadata is not None and 0 <= now-cloud_time < .25 and
                    max(abs(v) for v in pose["imu_state"]["rpy"][:2]) < .18):
                    yaw = float(pose["imu_state"]["rpy"][2])
                    slot = select_keyframe(self.photos, pose, epoch, now)
                    if slot is not None:
                        self.photo_sequence += 1
                        self.photos[slot] = {"slot": slot, "version": self.photo_sequence,
                            "yaw": yaw, "position": pose["position"], "jpeg": jpeg,
                            "rpy": list(pose["imu_state"]["rpy"]),
                            "camera_monotonic": now, "state_monotonic": pose_time,
                            "lidar_monotonic": cloud_time, "connection_epoch": self.connection_epoch,
                            "depth_points": cloud+float(metadata["resolution"])/2,
                            "lidar_resolution": float(metadata["resolution"]),
                            "saved_monotonic": now, "historical": False, "span_deg": 30}
                        if (self.spatial_capture and now-self.spatial_saved_at >= 2. and
                                self.session and self.session.parent == SITE/'spatial'):
                            await self.checkpoint_spatial()
                    if self.spatial_capture and len(self.photos) >= MAX_PHOTOS:
                        self.spatial_capture = False
                        self.spatial_message = '192-photo limit reached. Capture paused; stop moving, then build and save Spatial 3D'
        except (MediaStreamError, asyncio.CancelledError):
            pass
        finally:
            self.video_tasks.discard(task)

    async def rpc(self, topic, api, parameter=None, timeout=1.):
        if not self.conn or not self.connected:
            raise RuntimeError("Robot is not connected")
        options = {"api_id": api}
        if parameter is not None:
            options["parameter"] = parameter
        response = await asyncio.wait_for(self.conn.datachannel.pub_sub.publish_request_new(topic, options), timeout)
        code = response["data"]["header"]["status"]["code"]
        if code != 0:
            raise RuntimeError(f"Robot API {api} returned {code}")
        return response

    async def compute(self, function, *args):
        if self.compute_pool is None:
            return await asyncio.to_thread(function, *args)
        return await asyncio.get_running_loop().run_in_executor(self.compute_pool, function, *args)

    async def timing_loop(self):
        expected = time.monotonic()+.05
        while self.running:
            await asyncio.sleep(.05)
            now = time.monotonic()
            self.loop_lag_s = max(0., now-expected)
            expected = now+.05

    def feedback_sample(self):
        sport = self.sport or {}
        return {"sport_stamp": sport.get("stamp"), "state_age_s": self.age(self.state_time),
                "lidar_age_s": self.age(self.lidar_time), "loop_lag_s": self.loop_lag_s,
                "velocity": list(sport.get("velocity", [])), "yaw_speed": sport.get("yaw_speed"),
                "mode": sport.get("mode"), "gait_type": sport.get("gait_type"),
                "low_age_s": self.age(self.low_time)}

    def record_command(self, yaw, vx=0., vy=0.):
        self.command_samples.append({"t": time.monotonic(), "vx": vx, "vy": vy, "wz": yaw,
                                     **self.feedback_sample()})

    def move(self, yaw, vx=0., vy=0.):
        if not self.conn or self.conn.datachannel.channel.readyState != "open":
            raise RuntimeError("Robot control connection lost")
        if self.joystick_active and (self.joystick_rx or self.joystick_ly):
            self.move_joystick(0.)
        payload = {"header": {"identity": {"api_id": 1008, "id": time.monotonic_ns() % 2147483647},
                   "policy": {"priority": 0, "noreply": True}},
                   "parameter": json.dumps({"x": vx, "y": vy, "z": yaw}), "binary": []}
        self.conn.datachannel.pub_sub.publish_without_callback(RTC_TOPIC["SPORT_MOD"], payload, "req")
        self.record_command(yaw, vx, vy)
        self.command_samples[-1]["channel"] = "sport"
        if yaw or vx or vy:
            self.report["nonzero_commands"] = self.report.get("nonzero_commands", 0)+1

    def move_joystick(self, rx, ly=0.):
        if not self.conn or self.conn.datachannel.channel.readyState != "open":
            raise RuntimeError("Robot control connection lost")
        self.joystick_active = True
        self.joystick_rx, self.joystick_ly = rx, ly
        self.conn.datachannel.pub_sub.publish_without_callback(RTC_TOPIC["WIRELESS_CONTROLLER"],
            {"lx": 0., "ly": ly, "rx": rx, "ry": 0., "keys": 0})
        self.record_command(0.)
        self.command_samples[-1].update(channel="joystick", rx=rx, ly=ly)
        if rx or ly:
            self.report["nonzero_commands"] = self.report.get("nonzero_commands", 0)+1

    def move_navigation(self, yaw, vx):
        # Normalized forward stick; actual speed remains checked from odometry.
        ly = float(np.clip(1.5*vx, 0., .35))
        self.move_joystick(-yaw*RIGHT_STICK_PER_YAW, ly)
        self.command_samples[-1].update(vx=vx, wz=yaw)

    def request_stop(self, reason):
        self.abort_reason = reason
        self.abort.set()
        if self.motion_active():
            self.phase, self.message = "stopping", reason+"; stopping"
            with contextlib.suppress(Exception):
                if self.joystick_active:
                    self.move_joystick(0.)
                self.move(0.)

    def operator_guard(self):
        if self.abort.is_set():
            raise asyncio.CancelledError()
        if self.control_epoch is not None and self.control_epoch != self.connection_epoch:
            raise RuntimeError("Robot connection changed; motion will not resume automatically")
        if not self.connected:
            raise RuntimeError("Robot connection lost")
        if self.owner is not None:
            client = self.sockets.get(self.owner)
            if not client or time.monotonic()-client["heartbeat"] > 3.:
                raise RuntimeError("Operator page disconnected; stop requested")

    def guard(self):
        self.operator_guard()
        self.check_robot()

    async def motion_feedback(self, navigation=False, reason=None):
        check = self.navigation_guard if navigation else self.guard
        try:
            if reason is not None:
                raise MotionPause(reason)
            check()
            return 0.
        except MotionPause as error:
            reason = str(error)
        pauses = self.report.setdefault("pauses", [])
        if self.joystick_active:
            self.move_joystick(0.)
        self.move(0.)
        started = time.monotonic()
        before = np.asarray(self.sport["position"][:2], float).copy()
        event = {"started_t": started, "reason": reason, "outcome": "stopped", **self.feedback_sample()}
        pauses.append(event)
        self.phase, self.message = "paused", reason+" · Zero velocity requested; waiting for stable feedback"
        if navigation:
            self.navigation["state"] = "paused"
        stable_at, sequence = None, self.sport_sequence
        try:
            while True:
                self.operator_guard()
                if self.joystick_active:
                    self.move_joystick(0.)
                self.move(0.)
                now = time.monotonic()
                if now-started > 3.:
                    raise RuntimeError(reason+" persisted for over 3 seconds; stopped")
                try:
                    check()
                except MotionPause:
                    stable_at, sequence = None, self.sport_sequence
                else:
                    if np.linalg.norm(np.asarray(self.sport["position"][:2])-before) > .10:
                        raise RuntimeError("Position changed by over 10 cm while paused; stopped")
                    stationary = (np.linalg.norm(self.sport["velocity"][:2]) <= .025 and
                                  abs(self.sport["yaw_speed"]) <= .08)
                    if not stationary:
                        stable_at, sequence = None, self.sport_sequence
                    elif stable_at is None:
                        stable_at, sequence = now, self.sport_sequence
                    elif now-stable_at >= .25 and self.sport_sequence-sequence >= 3:
                        event["outcome"] = "resumed"
                        return now-started
                await asyncio.sleep(.05)
        finally:
            event["duration_s"] = time.monotonic()-started
            event["after"] = self.feedback_sample()
            self.report["paused_s"] = sum(p.get("duration_s", 0.) for p in pauses)

    def check_robot(self, include_origin=True, rotation=True):
        now = time.monotonic()
        if self.faults:
            labels = ", ".join(fault_text(f) for f in self.faults.values())
            raise RuntimeError("Robot reports active faults: "+labels+"; check the official app")
        if not self.faults_seen:
            raise RuntimeError("Waiting for robot fault status")
        if not self.connected:
            raise RuntimeError("Robot connection lost")
        if not self.sport or now-self.state_time > .3 or now-self.lidar_time > .5:
            raise MotionPause("Live pose or lidar is not ready")
        if rotation and (now-self.geometry_time > .8 or self.clearance is None):
            raise MotionPause("Waiting for an updated turning envelope")
        if rotation and self.turn_points is not None:
            self.clearance = swept_clearance(self.turn_points, self.sport["position"],
                self.sport["imu_state"]["rpy"][2], self.turn_direction*turn_lookahead(self.sport["yaw_speed"]),
                self.turn_resolution, body_only=self.turn_channel == "joystick")
        if self.sport.get("body_height", 0) < .22:
            raise RuntimeError("Robot is not standing steadily; stand up first")
        if rotation and self.clearance < .02:
            raise RuntimeError(f"Obstacle in the turning envelope: estimated clearance {max(0., self.clearance)*100:.1f} cm, below 2 cm")
        rpy = self.sport["imu_state"]["rpy"]
        raw = (self.low or {}).get("imu_state", {}).get("rpy", rpy)
        if max(abs(v) for v in (*rpy[:2], *raw[:2])) > .18:
            raise RuntimeError("Body tilt exceeds the demo limit; restore a stable standing pose")
        if include_origin and self.origin is not None and np.linalg.norm(np.asarray(self.sport["position"][:2])-self.origin) > .20:
            raise RuntimeError("Drift from the start exceeds 20 cm")
        if np.linalg.norm(self.sport["velocity"][:2]) > (.20 if rotation else .40) or abs(self.sport["yaw_speed"]) > 1.1:
            raise MotionPause("Motion speed exceeds the demo limit")

    async def stop_robot(self):
        for _ in range(8):
            with contextlib.suppress(Exception):
                if self.joystick_active:
                    self.move_joystick(0.)
                self.move(0.)
            await asyncio.sleep(.05)
        for key, topic, api, parameter in [
            ("stop_response", RTC_TOPIC["SPORT_MOD"], 1003, {}),
            ("release_response", RTC_TOPIC["OBSTACLES_AVOID"], 1004, {"is_remote_commands_from_api": False})]:
            for attempt in range(2):
                try:
                    self.report[key] = await self.rpc(topic, api, parameter, timeout=1.5)
                    break
                except asyncio.TimeoutError:
                    self.report[key+"_timeouts"] = attempt+1
                    if attempt:
                        self.report[key+"_error"] = "Both stop confirmation requests timed out"
                    with contextlib.suppress(Exception):
                        self.move(0.)
                except Exception as error:
                    self.report[key+"_error"] = str(error)
                    break
        self.joystick_active = False

    async def reset_map(self, new_scene=False):
        async with self.map_lock:
            self.scene_generation += 1
            self.invalidate_fusion()
            self.panorama = None
            self.panorama_version += 1
            self.panorama_message = "Capture again to align and blend panorama seams"
            self.plan_sequence += 1
            self.navigation = None
            self.ground_memory = None
            self.map_keys = np.empty((0, 3), np.int32)
            self.map_packet = point_packet(1, np.empty((0, 3)))
            self.live_packet = point_packet(2, np.empty((0, 3)))
            self.map_version += 1
            self.recorded_map = False
            self.photos.clear()
            self.spatial_capture = False
            self.spatial_message = 'Capture multiple stations along a route, then build spatial views'
            self.photo_sequence += 1
            self.session = None
            if new_scene:
                self.turn_points = None
                self.scene_reset_at = time.monotonic()
                self.cloud = None
                self.lidar_time = self.geometry_time = -1e9
                self.clearance = None
                self.jpeg = None
                self.camera_time = -1e9
                self.lidar_frames = self.camera_frames = 0
                self.tracker = self.origin = None
                self.peak_shift = 0.
                self.report = {}
                self.phase, self.message = "idle", "Scene reset; waiting for new data"
                self.fusion_message = "Scene reset; capture new photos before building photo 3D"

    async def run_motion(self, owner, channel="joystick"):
        self.turn_direction = 1
        self.turn_channel = channel
        self.owner = owner
        self.control_epoch = self.connection_epoch
        self.command_samples = []
        self.abort_reason = ""
        self.tracker = self.origin = None
        self.session = None
        self.measuring = False
        self.navigation = None
        self.samples = []
        self.report = {"started_unix": time.time(), "target_degrees": 720., "commanded_yaw_rad_s": TURN_SPEED,
                       "nonzero_commands": 0, "local_reserve_m": .02, "max_shift_m": .20,
                       "control_channel": channel}
        controlled = False
        outcome, detail = "error", "Demo did not start"
        try:
            self.phase, self.message = "preparing", "Checking the turning envelope"
            self.guard()
            if np.linalg.norm(self.sport["velocity"][:2]) > .025 or abs(self.sport["yaw_speed"]) > .04:
                raise RuntimeError("Robot has not settled; wait before starting")
            mode = await self.rpc(RTC_TOPIC["MOTION_SWITCHER"], 1001)
            if json.loads(mode["data"]["data"])["name"] != "mcf":
                raise RuntimeError("Robot is not in the validated MCF mode")
            self.report["avoidance_query"] = await self.rpc(RTC_TOPIC["OBSTACLES_AVOID"], 1002)
            if not json.loads(self.report["avoidance_query"]["data"]["data"])["enable"]:
                raise RuntimeError("Onboard obstacle avoidance is not enabled")
            await self.reset_map()
            self.session = SITE / "sessions" / datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d-%H%M%S")
            await asyncio.to_thread(self.session.mkdir, parents=True, exist_ok=False)
            self.fusion_message = "Capturing photos and lidar; photo 3D builds after two turns"
            self.origin = np.asarray(self.sport["position"][:2])
            self.peak_shift = 0.
            for i in range(3, 0, -1):
                self.phase, self.message = "countdown", f"{i} seconds until the two-turn demo; supervise the robot"
                for _ in range(20):
                    if self.abort.is_set():
                        raise asyncio.CancelledError()
                    await self.motion_feedback()
                    await asyncio.sleep(.05)
            self.report["control_source_response"] = await self.rpc(RTC_TOPIC["OBSTACLES_AVOID"], 1004,
                {"is_remote_commands_from_api": False})
            controlled = True
            await self.motion_feedback()
            self.tracker = TurnProgress(self.sport["imu_state"]["rpy"][2])
            self.measuring = True
            self.report["before"] = self.sport
            self.phase, self.message = "rotating", "Turning twice · Updating the cloud and photos"
            started = time.monotonic()
            paused_before_motion = self.report.get("paused_s", 0.)
            yaw_command = YawCommand(TURN_SPEED)
            first_yaw = yaw_command.update(TURN_SPEED, self.sport["yaw_speed"], .05)
            if channel == "joystick":
                self.report["maximum_stick_rx"] = TURN_SPEED*RIGHT_STICK_PER_YAW
                self.report["stick_scale_per_rad_s"] = RIGHT_STICK_PER_YAW
                self.move_joystick(-first_yaw*RIGHT_STICK_PER_YAW)
            else:
                self.report["nonzero_commands"] = 1
                self.record_command(first_yaw)
                self.report["move_ack"] = await self.rpc(RTC_TOPIC["SPORT_MOD"], 1008,
                    {"x": 0., "y": 0., "z": first_yaw}, timeout=.4)
            progress_at, progress_degrees = started, 0.
            command_at = started
            while self.tracker.degrees < 720-1e-6:
                if self.abort.is_set():
                    raise asyncio.CancelledError()
                paused = await self.motion_feedback()
                now = time.monotonic()
                if paused:
                    yaw_command = YawCommand(TURN_SPEED)
                    progress_at, progress_degrees = now, self.tracker.degrees
                    command_at = now-.05
                self.phase, self.message = "rotating", "Turning twice · Updating the cloud and photos"
                if now-started > 120 or now-started-(self.report.get("paused_s", 0.)-paused_before_motion) > 90:
                    raise RuntimeError("Demo time limit exceeded")
                if self.tracker.degrees < -3:
                    raise RuntimeError("Measured turn direction differs from the command")
                if now-progress_at > 5:
                    if self.tracker.degrees-progress_degrees < 3:
                        raise RuntimeError("Turn progress stalled; stop requested")
                    progress_at, progress_degrees = now, self.tracker.degrees
                remaining = math.radians(max(0., 720-self.tracker.degrees))
                desired = min(TURN_SPEED, max(.4, remaining*1.2)) if remaining > 1e-6 else 0.
                command = yaw_command.update(desired, self.sport["yaw_speed"], now-command_at)
                try:
                    self.guard()
                except MotionPause:
                    if self.joystick_active:
                        self.move_joystick(0.)
                    self.move(0.)
                    continue
                if channel == "joystick":
                    self.move_joystick(-command*RIGHT_STICK_PER_YAW)
                else:
                    self.move(command)
                command_at = now
                await asyncio.sleep(.05)
            outcome, detail = "complete", "Two-turn demo completed"
        except asyncio.CancelledError:
            outcome, detail = "stopped", self.abort_reason or "Demo stopped"
        except Exception as error:
            outcome, detail = "error", str(error) or f"{type(error).__name__}: robot control API did not confirm in time"
        finally:
            if controlled:
                self.phase, self.message = "stopping", "Sending stop and confirming feedback"
                await self.stop_robot()
                await asyncio.sleep(2)
            self.measuring = False
            self.report.update({"finished_unix": time.time(), "actual_degrees": self.tracker.degrees if self.tracker else 0.,
                                "peak_shift_m": self.peak_shift, "after": self.sport})
            if self.origin is not None and self.sport:
                self.report["final_shift_m"] = float(np.linalg.norm(np.asarray(self.sport["position"][:2])-self.origin))
            if controlled and ("stop_response_error" in self.report or "release_response_error" in self.report):
                detail = (detail+"; " if outcome == "error" else "")+"Stop confirmation incomplete; check the robot"
                outcome = "error"
            self.report["outcome"] = outcome
            self.report["message"] = detail
            if self.session:
                await asyncio.to_thread(self.save_session)
            self.phase, self.message = outcome, detail
            self.owner = None
            if outcome == "complete":
                async with self.action_lock:
                    try:
                        await self.start_fusion(self.fusion_camera)
                    except RuntimeError as error:
                        self.fusion_message = "Photo 3D could not build automatically: "+str(error)+"; retry manually"

    def save_session(self):
        session = self.session
        (session / "result.json").write_text(json.dumps(self.report, ensure_ascii=False, indent=2))
        (session / "turn_samples.json").write_text(json.dumps(self.samples))
        (session / "command_samples.json").write_text(json.dumps(self.command_samples))
        resolution = self.metadata["resolution"] if self.metadata else .05
        np.savez_compressed(session / "room_cloud.npz", points=self.map_keys.astype(np.float32)*resolution)
        if self.jpeg:
            (session / "camera_after.jpg").write_bytes(self.jpeg)
        photos = []
        for photo in list(self.photos.values()):
            name = f"photo-{photo['slot']:02d}.jpg"
            (session / name).write_bytes(photo["jpeg"])
            record = {k: v for k, v in photo.items() if k not in ("jpeg", "depth_points")}
            if photo.get("depth_points") is not None:
                record["depth_file"] = f"photo-{photo['slot']:02d}-lidar.npz"
                np.savez_compressed(session / record["depth_file"], points=photo["depth_points"])
            photos.append({**record, "file": name})
        (session / "photos.json").write_text(json.dumps(photos, ensure_ascii=False, indent=2))

    async def geometry_loop(self):
        last = -1e9
        while self.running:
            if self.cloud is not None and self.sport and self.lidar_time != last:
                cloud, stamp = self.cloud, self.lidar_time
                generation = self.scene_generation
                position = np.asarray(self.sport["position"])
                resolution = float(self.metadata["resolution"])
                try:
                    centers, floor = await self.compute(rotation_obstacles, cloud, position, resolution)
                    if generation == self.scene_generation:
                        self.turn_points, self.turn_resolution, self.floor = centers, resolution, floor
                        self.clearance = swept_clearance(centers, self.sport["position"],
                            self.sport["imu_state"]["rpy"][2],
                            self.turn_direction*turn_lookahead(self.sport["yaw_speed"]), resolution,
                            body_only=self.turn_channel == "joystick")
                        self.geometry_time = stamp
                except ValueError:
                    if generation == self.scene_generation:
                        self.clearance = None
                        self.turn_points = None
                last = stamp
            await asyncio.sleep(.05)

    async def map_loop(self):
        last = -1e9
        while self.running:
            if self.cloud is not None and self.lidar_time != last:
                cloud, stamp = self.cloud, self.lidar_time
                generation = self.scene_generation
                resolution = float(self.metadata["resolution"])
                async with self.map_lock:
                    if generation == self.scene_generation and stamp > self.scene_reset_at:
                        if self.recorded_map:
                            self.invalidate_fusion()
                            self.map_keys = np.empty((0, 3), np.int32)
                            self.recorded_map = False
                        position = self.sport["position"] if self.sport else None
                        keys, packet, live_packet = await self.compute(map_frame, self.map_keys, cloud, resolution, position)
                        self.map_keys, self.map_packet = keys, packet
                        self.live_packet = live_packet
                        self.map_version += 1
                last = stamp
            await asyncio.sleep(.4)

    async def connect_loop(self):
        device = read_device(self.key_file)
        while self.running:
            conn = None
            try:
                self.robot_ip = await resolve_robot_ip(self.robot_address, device["sn"])
                conn = UnitreeWebRTCConnection(WebRTCConnectionMethod.LocalSTA, ip=self.robot_ip,
                                              aes_128_key=device["aes_128_key"])
                self.conn = conn
                self.faults, self.faults_seen = {}, False
                self.power_history.clear()
                connecting = asyncio.create_task(conn.connect())
                # LocalSTA creates the data channel before its first network await.
                await asyncio.sleep(0)
                conn.datachannel.channel.on("message", self.channel_faults)
                await asyncio.wait_for(connecting, 40)
                self.connection_epoch += 1
                self.sport_source_stamp = None
                self.navigation = None
                self.ground_memory = None
                self.connected = True
                conn.datachannel.set_decoder("native")
                conn.datachannel.pub_sub.subscribe(RTC_TOPIC["LF_SPORT_MOD_STATE"], self.sport_callback)
                conn.datachannel.pub_sub.subscribe(RTC_TOPIC["LOW_STATE"], self.low_callback)
                conn.datachannel.pub_sub.subscribe(RTC_TOPIC["ULIDAR_ARRAY"], self.lidar_callback)
                conn.video.add_track_callback(self.video)
                conn.video.switchVideoChannel(True)
                await asyncio.wait_for(conn.datachannel.disableTrafficSaving(True), 3)
                conn.datachannel.pub_sub.publish_without_callback(RTC_TOPIC["ULIDAR_SWITCH"], "on")
                if not self.motion_active():
                    if self.report.get("outcome") == "complete":
                        self.phase, self.message = "complete", "Connected · Showing the last result; arm before another demo"
                    else:
                        self.phase, self.message = "idle", "Connected · Ready to start the two-turn demo"
                connected_at = time.monotonic()
                while self.running and conn.isConnected and conn.datachannel.channel.readyState == "open":
                    if time.monotonic()-max(self.state_time, connected_at) > 5:
                        raise RuntimeError("Pose stream interrupted; reconnecting for telemetry")
                    await asyncio.sleep(.5)
            except asyncio.CancelledError:
                break
            except Exception as error:
                if not self.motion_active():
                    self.phase = "connecting"
                    self.message = ("Robot is busy with another client; close phone video or remote control" if isinstance(error, RobotBusyError)
                                    else str(error) if isinstance(error, DiscoveryError)
                                    else "Connection unavailable; retrying")
                logging.warning("Robot connection: %s", error)
            finally:
                self.connected = False
                was_spatial = self.spatial_capture
                self.spatial_capture = False
                self.connection_epoch += 1
                if not self.motion_active():
                    self.navigation = None
                if self.motion_active():
                    self.request_stop("Robot disconnected")
                    with contextlib.suppress(Exception, asyncio.CancelledError):
                        await self.motion_task
                if was_spatial and self.session and self.session.parent == SITE/'spatial':
                    await self.checkpoint_spatial()
                if conn is not None:
                    with contextlib.suppress(Exception):
                        await conn.disconnect()
                self.conn = None
                for task in list(self.video_tasks):
                    task.cancel()
                self.state_time = self.lidar_time = self.geometry_time = -1e9
                self.low_time = -1e9
                self.cloud = self.sport = self.low = None
                self.turn_points = None
                self.ground_memory = None
                self.recorded_map = bool(len(self.map_keys))
                for photo in self.photos.values():
                    self.photo_sequence += 1
                    photo.update(historical=True, version=self.photo_sequence)
            if self.running:
                await asyncio.sleep(4)

    def photo_manifest(self):
        return [{k: v for k, v in p.items() if k not in ("jpeg", "saved_monotonic", "depth_points")}
                for p in list(self.photos.values())]

    async def panorama_loop(self):
        processed = -1
        while self.running:
            sequence, generation = self.photo_sequence, self.scene_generation
            if (sequence != processed and len(self.photos) >= 2 and not self.motion_active() and not self.spatial_capture and
                    not self.fusion_busy()):
                photos = [{k: p[k] for k in ("slot", "yaw", "rpy", "position", "jpeg", "connection_epoch") if k in p}
                          for p in self.photos.values()]
                self.panorama_message = "Aligning photo edges and blending seams…"
                try:
                    result = await self.compute(build_spatial_panorama, photos, self.fusion_camera)
                    if sequence == self.photo_sequence and generation == self.scene_generation:
                        self.panorama_version += 1
                        self.panorama = {**result, "id": self.panorama_version,
                                         "image_url": f"/panorama.png?v={self.panorama_version}"}
                        self.panorama_message = (f"{result.get('station_count', 1)} spatial stations · {result['frames']} photos, blended separately at each station" +
                            (f" · {result['matched_pairs']} overlapping feature pairs aligned" if result['matched_pairs'] else " · Low texture; using recorded camera poses"))
                except Exception:
                    logging.exception("Panorama alignment")
                    if generation == self.scene_generation and sequence == self.photo_sequence:
                        self.panorama_message = "Seam optimization incomplete; showing the pose-aligned panorama"
                processed = sequence
            await asyncio.sleep(1.)

    async def broadcast(self):
        async def send(client):
            ws = client["ws"]
            await ws.send_json(self.status())
            if client.get("panorama_version") != self.panorama_version:
                manifest = ({k: v for k, v in self.panorama.items() if k not in ('image', 'stations')} if self.panorama else None)
                if manifest:
                    manifest["image"] = manifest.pop("image_url")
                    manifest['stations'] = [{**{k: v for k, v in view.items() if k != 'image'},
                        'image': f"/panorama.png?v={self.panorama_version}&station={view['station']}"}
                        for view in self.panorama.get('stations', [])]
                await ws.send_json({"type": "panorama", "manifest": manifest})
                client["panorama_version"] = self.panorama_version
            if client.get("fusion_version") != self.fusion_version:
                await ws.send_json({"type": "fusion", "manifest": self.fusion})
                client["fusion_version"] = self.fusion_version
            if client.get("map_version") != self.map_version:
                await ws.send_bytes(self.map_packet)
                await ws.send_bytes(self.live_packet)
                client["map_version"] = self.map_version
            if client.get("photo_version") != self.photo_sequence:
                await ws.send_json({"type": "photos", "photos": self.photo_manifest()})
                client["photo_version"] = self.photo_sequence
        while self.running:
            for client in list(self.sockets.values()):
                try:
                    await asyncio.wait_for(send(client), .5)
                except Exception:
                    pass
            await asyncio.sleep(.25)

    async def ws_handler(self, request):
        ws = web.WebSocketResponse(heartbeat=5)
        await ws.prepare(request)
        self.client_sequence += 1
        client_id = str(self.client_sequence)
        self.sockets[client_id] = {"ws": ws, "heartbeat": time.monotonic()}
        await ws.send_json({"type": "hello", "client_id": client_id})
        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    self.sockets[client_id]["heartbeat"] = time.monotonic()
        finally:
            self.sockets.pop(client_id, None)
            if self.owner == client_id and self.motion_active():
                self.request_stop("Demo page disconnected")
        return ws

    def save_turn_state(self):
        path = SITE / "turn_state.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps({"token": self.turn_token, "latched": self.turn_latched}))
        temporary.replace(path)

    def operator(self, body):
        owner = body.get("client_id")
        if owner not in self.sockets or time.monotonic()-self.sockets[owner]["heartbeat"] > 3:
            raise web.HTTPConflict(text="Connect the live page first")
        return owner

    def schedule(self, owner, coroutine):
        self.abort.clear()
        self.owner = owner
        self.phase, self.message = "preparing", "Task accepted; checking readiness"
        self.motion_task = asyncio.create_task(coroutine)

    async def turn_probe_handler(self, request):
        body = await request.json()
        direction, channel = body.get("direction"), body.get("channel", "joystick")
        if type(direction) is not int or direction not in (-1, 1) or channel not in ("joystick", "sport"):
            raise web.HTTPBadRequest(text="Turn test requires direction -1 or 1 and channel joystick or sport")
        async with self.action_lock:
            owner = self.operator(body)
            if self.motion_active() or self.fusion_busy():
                raise web.HTTPConflict(text="Wait for the current task to finish")
            try:
                self.check_navigation(stationary=True)
            except RuntimeError as error:
                raise web.HTTPConflict(text=str(error))
            self.schedule(owner, self.run_turn_probe(owner, direction, channel))
        return web.json_response({"started": True, "maximum_seconds": 1.2, "target_degrees": 12})

    async def run_turn_probe(self, owner, direction, channel):
        self.owner, self.turn_direction = owner, direction
        self.command_samples = []
        self.report = {"action": "turn_probe", "channel": channel, "direction": direction,
                       "started_unix": time.time(), "nonzero_commands": 0,
                       "maximum_seconds": 1.2, "target_degrees": 12., "maximum_shift_m": .05}
        controlled = False
        outcome, detail = "error", "Turn test did not start"
        tracker = None
        try:
            self.operator_guard()
            self.check_navigation(stationary=True)
            self.check_robot(include_origin=False)
            mode = await self.rpc(RTC_TOPIC["MOTION_SWITCHER"], 1001)
            if json.loads(mode["data"]["data"])["name"] != "mcf":
                raise RuntimeError("Robot is not in the validated MCF mode")
            avoid = await self.rpc(RTC_TOPIC["OBSTACLES_AVOID"], 1002)
            if not json.loads(avoid["data"]["data"])["enable"]:
                raise RuntimeError("Onboard obstacle avoidance is not enabled")
            controlled = True
            self.report["control_source_response"] = await self.rpc(RTC_TOPIC["OBSTACLES_AVOID"], 1004,
                {"is_remote_commands_from_api": False})
            self.report["before"] = self.sport
            origin = np.asarray(self.sport["position"][:2], float).copy()
            tracker = TurnProgress(self.sport["imu_state"]["rpy"][2])
            self.report["clearance_before_m"] = self.clearance
            started = time.monotonic()
            command = YawCommand()
            peak_shift = 0.
            self.phase, self.message = "rotating", "Running a brief turn-only test"
            while time.monotonic()-started < 1.2:
                self.operator_guard()
                self.check_robot(include_origin=False)
                tracker.update(self.sport["imu_state"]["rpy"][2])
                shift = float(np.linalg.norm(np.asarray(self.sport["position"][:2])-origin))
                peak_shift = max(peak_shift, shift)
                if shift > .05:
                    raise RuntimeError("Turn test translation exceeded 5 cm; stopped")
                if direction*tracker.degrees < -2:
                    raise RuntimeError("Measured turn differs from the test direction; stopped")
                if direction*tracker.degrees >= 12:
                    break
                if channel == "joystick":
                    # Right-stick positive X turns clockwise; odometry yaw is counterclockwise.
                    self.move_joystick(-direction*.15)
                else:
                    self.move(command.update(direction*.3, self.sport["yaw_speed"], .05))
                await asyncio.sleep(.05)
            self.report["nonzero_duration_s"] = time.monotonic()-started
            self.report["peak_shift_m"] = peak_shift
            outcome, detail = "complete", "Turn test completed"
        except asyncio.CancelledError:
            outcome, detail = "stopped", self.abort_reason or "Turn test stopped"
        except Exception as error:
            outcome, detail = "error", str(error) or f"{type(error).__name__}: robot control API did not confirm in time"
        finally:
            if controlled:
                await self.stop_robot()
                await asyncio.sleep(.5)
            if tracker and self.sport:
                tracker.update(self.sport["imu_state"]["rpy"][2])
            if "stop_response_error" in self.report or "release_response_error" in self.report:
                outcome, detail = "error", detail+"; Stop confirmation incomplete; check the robot"
            self.report.update(outcome=outcome, message=detail, after=self.sport,
                               actual_degrees=tracker.degrees if tracker else 0., finished_unix=time.time())
            try:
                await asyncio.to_thread(self.save_turn_probe, self.report.copy(), self.command_samples)
            except Exception:
                logging.exception("Turn probe record could not be saved")
            self.turn_direction = 1
            self.phase, self.message, self.owner = ("idle" if outcome == "complete" else outcome), detail, None

    @staticmethod
    def save_turn_probe(report, commands):
        folder = SITE / "turn-tests" / datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d-%H%M%S-%f")
        folder.mkdir(parents=True)
        (folder / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
        (folder / "command_samples.json").write_text(json.dumps(commands))

    async def start_handler(self, request):
        body = await request.json()
        channel = body.get("channel", "joystick")
        if channel not in ("sport", "joystick"):
            raise web.HTTPBadRequest(text="Turn channel must be sport or joystick")
        async with self.action_lock:
            owner = self.operator(body)
            if body.get("turn_token") != self.turn_token:
                raise web.HTTPConflict(text="Demo arm token changed; refresh status and retry")
            if self.turn_latched:
                return web.json_response({"started": False, "duplicate": True})
            if self.motion_active():
                raise web.HTTPConflict(text="Another motion task is active; cannot start a turn")
            if self.fusion_busy():
                raise web.HTTPConflict(text="Photo 3D is building; wait before moving")
            if not self.status()["ready"]:
                raise web.HTTPConflict(text=self.status()["blocked"] or "Waiting for lidar")
            try:
                model = CameraModel(**body["camera"]) if "camera" in body else self.fusion_camera
            except (TypeError, ValueError) as error:
                raise web.HTTPBadRequest(text=str(error))
            self.fusion_camera = model
            self.turn_latched = True
            try:
                await asyncio.to_thread(self.save_turn_state)
            except OSError:
                self.turn_latched = False
                raise
            self.schedule(owner, self.run_motion(owner) if channel == "joystick" else self.run_motion(owner, channel))
            return web.json_response({"started": True})

    async def rearm_handler(self, request):
        body = await request.json()
        async with self.action_lock:
            self.operator(body)
            if self.motion_active():
                raise web.HTTPConflict(text="Cannot arm while a task is running or stopping")
            if body.get("turn_token") != self.turn_token or not self.turn_latched:
                raise web.HTTPConflict(text="Demo arm state changed")
            self.turn_token, self.turn_latched = uuid.uuid4().hex, False
            await asyncio.to_thread(self.save_turn_state)
            self.phase, self.message = "idle", "Armed · Click Start two-turn demo"
            return web.json_response({"prepared": True})

    def check_navigation(self, stationary=False):
        self.check_stand()
        self.check_robot(include_origin=False, rotation=False)
        if len((self.low or {}).get("motor_state", [])) < 12:
            raise RuntimeError("Waiting for complete joint feedback")
        if self.cloud is None or self.metadata is None:
            raise RuntimeError("Waiting for the current live lidar map")
        if stationary and (np.linalg.norm(self.sport["velocity"][:2]) > .025 or abs(self.sport["yaw_speed"]) > .04):
            raise RuntimeError("Let the robot settle before planning or navigating")

    def navigation_guard(self):
        self.operator_guard()
        self.check_navigation()

    async def navigation_grid(self, yaw):
        epoch, stamp = self.connection_epoch, self.lidar_time
        generation = self.scene_generation
        cloud, pose = self.cloud, self.sport
        grid = await self.compute(GroundGrid, cloud, pose["position"], yaw,
                                 float(self.metadata["resolution"]), self.ground_memory, stamp)
        if epoch != self.connection_epoch or generation != self.scene_generation:
            raise RuntimeError("Connection or scene changed during planning; select a new target")
        if time.monotonic()-stamp > .5:
            raise MotionPause("Lidar became stale during planning")
        if self.ground_memory is None or stamp >= self.ground_memory["stamp"]:
            self.ground_memory = grid.ground_memory
        return grid, stamp

    async def plan_handler(self, request):
        body = await request.json()
        owner = self.operator(body)
        target = body.get("target")
        if not isinstance(target, list) or len(target) != 2 or any(
                isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in target):
            raise web.HTTPBadRequest(text="Target requires two finite ground coordinates")
        async with self.action_lock:
            if self.motion_active():
                raise web.HTTPConflict(text="Stop the current action before selecting a target")
            self.plan_sequence += 1
            request_id = self.plan_sequence
            self.navigation = None
        try:
            self.check_navigation(stationary=True)
            epoch = self.connection_epoch
            generation = self.scene_generation
            yaw = float(self.sport["imu_state"]["rpy"][2])
            start = list(self.sport["position"][:2])
            grid, stamp = await self.navigation_grid(yaw)
            try:
                result = await self.compute(grid.plan, target, True)
                reachable = True
                reason = (f"Staged approach: first detour to a stop {result['goal_offset_m']:.2f} m from the goal, settle and rescan; full route unverified"
                          if result["partial"] else
                          f"Reachable in current observations: {result['walk_count']} walking legs, {result['turn_count']} turns; checked again before departure")
            except PlanningError as error:
                result = {"blocked_at": error.issue, "floor": grid.floor,
                          "collision_ceiling_m": grid.ceiling}
                reachable, reason = False, str(error)
            except ValueError as error:
                result, reachable, reason = {}, False, str(error)
            async with self.action_lock:
                if self.motion_active() or epoch != self.connection_epoch or request_id != self.plan_sequence:
                    raise RuntimeError("Robot state changed; select a new target")
                self.check_navigation(stationary=True)
                if generation != self.scene_generation or np.linalg.norm(
                        np.asarray(self.sport["position"][:2])-grid.start) > .10 or abs(wrap(
                        self.sport["imu_state"]["rpy"][2]-yaw)) > HEADING_LIMIT:
                    raise RuntimeError("Pose or scene changed during planning; replan the target")
                self.navigation = {**result, "id": request_id, "owner": owner, "epoch": epoch,
                    "scene_generation": generation,
                    "target": target, "start": start, "yaw": yaw, "reachable": reachable,
                    "reason": reason, "state": "planned" if reachable else "blocked", "used": False,
                    "expires_at": time.monotonic()+45, "map_stamp": stamp}
                return web.json_response(self.navigation)
        except (RuntimeError, ValueError) as error:
            raise web.HTTPConflict(text=str(error))

    async def navigate_handler(self, request):
        body = await request.json()
        async with self.action_lock:
            owner = self.operator(body)
            plan = self.navigation
            if self.motion_active():
                raise web.HTTPConflict(text="Another action is running; navigation is not queued")
            if self.fusion_busy():
                raise web.HTTPConflict(text="Photo 3D is building; wait before moving")
            if not plan or body.get("plan_id") != plan["id"] or plan["owner"] != owner:
                raise web.HTTPConflict(text="Select and plan a new target in this page")
            if plan["used"] or not plan["reachable"] or time.monotonic() > plan["expires_at"]:
                raise web.HTTPConflict(text="Route used, unreachable or expired; select a new target")
            if plan["epoch"] != self.connection_epoch:
                raise web.HTTPConflict(text="Robot reconnected; select a new target")
            try:
                self.check_navigation(stationary=True)
                if np.linalg.norm(np.asarray(self.sport["position"][:2])-plan["start"]) > .10 or abs(wrap(
                        self.sport["imu_state"]["rpy"][2]-plan["yaw"])) > HEADING_LIMIT:
                    raise RuntimeError("Robot pose changed; select a new target")
            except RuntimeError as error:
                raise web.HTTPConflict(text=str(error))
            plan["used"], plan["state"] = True, "checking"
            self.schedule(owner, self.run_navigation(owner, plan))
            return web.json_response({"started": True, "plan_id": plan["id"]})

    async def park_navigation_corner(self, plan, waypoint):
        """Confirm a stationary corner before changing the walking direction."""
        started = time.monotonic()
        position = np.asarray(self.sport["position"][:2], float)
        event = {"segment": waypoint, "started_t": started, "position": position.tolist(), "outcome": "stopped"}
        self.report.setdefault("corners", []).append(event)
        stable_at, sequence = None, self.sport_sequence
        try:
            while True:
                self.move(0.)
                await self.motion_feedback(navigation=True)
                now = time.monotonic()
                self.phase, plan["state"] = "corner", "settling"
                self.message = f"Leg {waypoint} reached · Settling before the next turn"
                if now-started > 3.:
                    raise RuntimeError("Corner stop did not settle within 3 seconds; stopped")
                if np.linalg.norm(np.asarray(self.sport["position"][:2])-position) > .10:
                    raise RuntimeError("Corner stop drift exceeded 10 cm; stopped")
                stationary = (np.linalg.norm(self.sport["velocity"][:2]) <= .025 and
                              abs(self.sport["yaw_speed"]) <= .08)
                if not stationary:
                    stable_at, sequence = None, self.sport_sequence
                elif stable_at is None:
                    stable_at, sequence = now, self.sport_sequence
                elif now-stable_at >= .15 and self.sport_sequence-sequence >= 3:
                    event["outcome"] = "stationary"
                    return
                await asyncio.sleep(.05)
        finally:
            event["duration_s"] = time.monotonic()-started
            event["after"] = self.feedback_sample()

    async def rescan_navigation(self, dwell=0., frames=1):
        """Hold neutral until a scan received after settling can be checked."""
        started, previous_scan = time.monotonic(), self.lidar_time
        scans, last_scan = 0, previous_scan
        while True:
            self.move(0.)
            try:
                self.navigation_guard()
            except MotionPause:
                pass
            else:
                if self.lidar_time > last_scan:
                    scans, last_scan = scans+1, self.lidar_time
                if scans >= frames and time.monotonic()-started >= dwell:
                    return await self.navigation_grid(self.sport["imu_state"]["rpy"][2])
            if time.monotonic()-started >= 1.5:
                raise RuntimeError("No fresh lidar scan within 1.5 seconds of stopping; stopped short of the goal")
            await asyncio.sleep(.05)

    async def replan_navigation(self, plan, goal, path, waypoint, travelled, trigger):
        if len(self.report.get("replans", [])) >= 3:
            raise RuntimeError("Staged replan limit reached; stopped short of the goal")
        await self.park_navigation_corner(plan, waypoint)
        plan['state'] = 'replanning'
        self.message = "Stopped; rescanning and replanning from the measured pose"
        grid, stamp = await self.rescan_navigation(
            dwell=.45 if trigger['kind'] in ('occupied', 'stall') else 0.,
            frames=2 if trigger['kind'] in ('occupied', 'stall') else 1)
        if trigger['kind'] == 'stall':
            plan.setdefault('failed_departures', []).append({
                'position': grid.start.tolist(), 'yaw': trigger['yaw']})
        grid.avoid_failed_departures(plan.get('failed_departures', []))
        self.report.setdefault('recovery_events', []).append({
            'trigger': trigger, 'scan_stamp': stamp, 'position': grid.start.tolist(),
            'failed_departures': plan.get('failed_departures', [])[:]})
        try:
            updated = await self.compute(grid.plan, goal, True)
        except ValueError as error:
            obstruction = getattr(error, "issue", None)
            if obstruction is not None:
                plan["blocked_at"] = obstruction
            self.report["replan_failure"] = {"reason": str(error), "blocked_at": obstruction,
                                           "trigger": trigger}
            distance = float(np.linalg.norm(np.asarray(self.sport["position"][:2])-goal))
            newly_blocked = trigger["kind"] == "occupied" or any(
                r["trigger"]["kind"] == "occupied" for r in self.report.get("replans", []))
            label = "Stopped at a verified endpoint after new observations" if newly_blocked else "Stopped at a verified endpoint"
            raise RuntimeError(f"{label}; distance to goal {distance:.2f} m; "+str(error)) from error
        if travelled+updated["length_m"] > MAX_PATH:
            raise RuntimeError("Replanned travel exceeds the navigation limit; stopped")
        self.report.setdefault("replans", []).append({"t": time.monotonic(), "trigger": trigger,
            "previous_path": path.tolist(), "path": updated["path"], "partial": updated["partial"]})
        plan.update(updated)
        plan.update(replan_count=len(self.report["replans"]), replan_limit=3,
                    recovery_strategy='stop_scan_avoid_failed_departure')
        plan.pop("blocked_at", None)
        plan["reason"] = (f"Staged approach: this stop is {updated['goal_offset_m']:.2f} m from the goal; settle and rescan"
                          if updated["partial"] else "New route checked; continuing to the goal")
        self.report.update(actions=updated["actions"], walk_count=updated["walk_count"],
            turn_count=updated["turn_count"], arrival_target=updated["arrival_target"],
            goal_offset_m=updated["goal_offset_m"], partial=updated["partial"])
        return updated, grid, stamp

    async def run_navigation(self, owner, plan):
        self.owner = owner
        self.control_epoch = self.connection_epoch
        self.command_samples = []
        self.navigation = plan
        self.measuring = False
        self.report = {"action": "navigate", "target": plan["target"], "started_unix": time.time(),
                       "nonzero_commands": 0, "before": self.sport, "motion_model": "turn_then_forward",
                       "forward_control_channel": "joystick",
                       "turn_control_channel": "joystick", "stick_scale_per_rad_s": RIGHT_STICK_PER_YAW}
        samples, controlled, failure_grid = [], False, None
        outcome, detail = "error", "Navigation did not start"
        goal = np.asarray(plan["target"])
        try:
            self.navigation_guard()
            if plan["epoch"] != self.connection_epoch:
                raise RuntimeError("Connection coordinate frame changed; select a new target")
            mode = await self.rpc(RTC_TOPIC["MOTION_SWITCHER"], 1001)
            if json.loads(mode["data"]["data"])["name"] != "mcf":
                raise RuntimeError("Robot is not in the validated MCF mode")
            self.report["avoidance_query"] = await self.rpc(RTC_TOPIC["OBSTACLES_AVOID"], 1002)
            if not json.loads(self.report["avoidance_query"]["data"]["data"])["enable"]:
                raise RuntimeError("Onboard obstacle avoidance is not enabled")
            for i in (2, 1):
                self.message = f"{i} seconds until navigation; supervise the robot"
                for _ in range(20):
                    await self.motion_feedback(navigation=True)
                    if np.linalg.norm(np.asarray(self.sport["position"][:2])-plan["start"]) > .10:
                        raise RuntimeError("Position changed during preparation; select a new target")
                    if np.linalg.norm(self.sport["velocity"][:2]) > .025 or abs(self.sport["yaw_speed"]) > .04:
                        raise RuntimeError("Robot moved during preparation; select a new target")
                    await asyncio.sleep(.05)
            grid, grid_stamp = await self.navigation_grid(self.sport["imu_state"]["rpy"][2])
            result = await self.compute(grid.plan, goal, True)
            plan.update(result)
            path = np.asarray(result["path"])
            plan.update(segment_index=1, completed_segments=0)
            self.report.update(actions=result["actions"], walk_count=result["walk_count"],
                               turn_count=result["turn_count"], planner=result["planner"],
                               arrival_target=result["arrival_target"], goal_offset_m=result["goal_offset_m"],
                               collision_ceiling_m=result["collision_ceiling_m"], partial=result["partial"])
            self.navigation_guard()
            if len(path) == 1:
                plan["completed_segments"] = 0
                outcome, detail = "complete", "Already within the arrival radius; no movement needed"
                return
            controlled = True
            self.report["balance_response"] = await self.rpc(RTC_TOPIC["SPORT_MOD"], 1002, {}, timeout=2.)
            for _ in range(10):
                await self.motion_feedback(navigation=True)
                await asyncio.sleep(.05)
            self.report["control_source_response"] = await self.rpc(RTC_TOPIC["OBSTACLES_AVOID"], 1004,
                {"is_remote_commands_from_api": False}, timeout=2.)
            started = time.monotonic()
            previous = np.asarray(self.sport["position"][:2], float)
            previous_time, refresh_at, waypoint = started, -math.inf, 1
            travelled = 0.
            command_at, vx, yaw_command = started-.05, 0., YawCommand(TURN_SPEED)
            stage, pending_pause = None, None
            route_wait = None
            speed_limit = CRUISE_SPEED
            forward_gain, speed_sequence = 1., self.sport_sequence
            while True:
                paused = await self.motion_feedback(navigation=True, reason=pending_pause)
                pending_pause = None
                now = time.monotonic()
                position = np.asarray(self.sport["position"][:2], float)
                yaw = float(self.sport["imu_state"]["rpy"][2])
                distance = float(np.linalg.norm(goal-position))
                plan["remaining_m"] = distance
                if plan["epoch"] != self.connection_epoch:
                    raise RuntimeError("Connection coordinate frame changed")
                if np.linalg.norm(position-previous) > .10+MAX_SPEED*(now-previous_time):
                    raise RuntimeError("Localization jumped; stopped")
                travelled += float(np.linalg.norm(position-previous))
                previous, previous_time = position.copy(), now
                if now-started > 120:
                    raise RuntimeError("Navigation exceeded the 120-second limit")
                if path_distance(position, path) > .15:
                    raise RuntimeError("Route deviation exceeded 15 cm; stopped")
                if waypoint == len(path)-1 and distance <= STOP_RADIUS:
                    plan["completed_segments"] = len(path)-1
                    outcome, detail = "complete", "Arrived within the target radius"
                    break
                if paused:
                    stage, vx, yaw_command = None, 0., YawCommand(TURN_SPEED)
                    command_at = now-.05
                    if self.report.get('pauses', [{}])[-1].get('reason') == 'Motion speed exceeds the demo limit':
                        forward_gain = max(.65, forward_gain*.85)
                try:
                    if paused or now-refresh_at >= .20:
                        grid, grid_stamp = await self.navigation_grid(yaw)
                        refresh_at = time.monotonic()
                    self.navigation_guard()
                    if time.monotonic()-grid_stamp > .5:
                        raise MotionPause("Route check data is stale")
                except MotionPause as error:
                    self.move(0.)
                    pending_pause = str(error)
                    continue
                current_position = np.asarray(self.sport["position"][:2], float)
                if np.linalg.norm(current_position-position) > .10+MAX_SPEED*(time.monotonic()-now):
                    raise RuntimeError("Localization jumped during the route update; stopped")
                position = current_position
                yaw = float(self.sport["imu_state"]["rpy"][2])
                if path_distance(position, path) > .15:
                    raise RuntimeError("Route deviation exceeded 15 cm; stopped")
                corner_delta = position-path[waypoint]
                incoming = path[waypoint]-path[waypoint-1]
                at_corner = (np.linalg.norm(corner_delta) <= CORNER_ARRIVAL or
                             (np.dot(corner_delta, incoming) >= 0 and np.linalg.norm(corner_delta) <= .08))
                if waypoint == len(path)-1 and plan.get("partial") and at_corner:
                    updated, grid, grid_stamp = await self.replan_navigation(plan, goal, path, waypoint,
                        travelled, {"kind": "approach", "point": position.tolist()})
                    path, waypoint = np.asarray(updated["path"]), 1
                    if len(path) == 1:
                        outcome, detail = "complete", "Arrived within the target radius"
                        break
                    stage, vx, yaw_command = None, 0., YawCommand(TURN_SPEED)
                    command_at, refresh_at = time.monotonic()-.05, -math.inf
                    route_wait = None
                    continue
                if waypoint < len(path)-1 and at_corner:
                    await self.park_navigation_corner(plan, waypoint)
                    waypoint += 1
                    stage, vx, yaw_command = None, 0., YawCommand(TURN_SPEED)
                    command_at, refresh_at = time.monotonic()-.05, -math.inf
                    # Recheck the new segment from the settled, measured pose.
                    continue
                plan.update(segment_index=waypoint, completed_segments=waypoint-1,
                    remaining_path_m=float(np.linalg.norm(path[waypoint]-position))+
                        sum(float(np.linalg.norm(b-a)) for a, b in zip(path[waypoint:], path[waypoint+1:])))
                error = path[waypoint]-position
                direction = incoming/np.linalg.norm(incoming)
                cross_track = float(np.dot(position-path[waypoint-1], [-direction[1], direction[0]]))
                # Follow the leg tangent with a bounded lateral correction.
                # Bearing directly at a nearby corner magnifies small position
                # errors and repeatedly restarts alignment before the stop.
                correction = float(np.clip(math.atan2(-cross_track, .30), -.08, .08))
                travel_heading = math.atan2(direction[1], direction[0])
                heading = wrap(travel_heading+correction)
                angle = wrap(heading-yaw)
                desired_stage = "aligning" if stage in (None, "aligning") or abs(angle) > HEADING_LIMIT else "moving"
                if desired_stage == "aligning" and abs(angle) <= .025 and abs(self.sport["yaw_speed"]) <= .05:
                    desired_stage = "moving"
                # Check the swept turn while aligning, then check translation
                # before sending any forward command.
                next_heading = yaw+float(np.clip(angle, -turn_lookahead(self.sport["yaw_speed"]),
                                                 turn_lookahead(self.sport["yaw_speed"])))
                issue = (grid.turn_issue(position, yaw, next_heading) if desired_stage == "aligning" else
                         grid.segment_issue(position, path[waypoint], travel_heading))
                if issue is None and desired_stage == "moving":
                    # The route already includes the permitted heading sweep.
                    # Check the measured pose without adding that sweep twice.
                    issue = grid.envelope_issue([position], [yaw], heading_tolerance=0.)
                if issue is not None:
                    failure_grid = grid
                    issue.update(stage=desired_stage, floor=grid.floor, yaw=yaw, lidar_stamp=grid_stamp,
                                 distance_m=float(np.linalg.norm(np.asarray(issue["point"])-position)))
                    self.report["route_check"] = issue
                    plan["blocked_at"] = issue
                    label = "Turning envelope" if desired_stage == "aligning" else "Walking route"
                    if issue["kind"] == "occupied" and len(self.report.get("replans", [])) < 3:
                        # A newly blocked turn can require walking to a different
                        # turning point. Settle and rescan before checking that route.
                        updated, grid, grid_stamp = await self.replan_navigation(
                            plan, goal, path, waypoint, travelled, issue)
                        path, waypoint = np.asarray(updated["path"]), 1
                        if len(path) == 1:
                            outcome, detail = "complete", "Arrived within the target radius"
                            break
                        stage, vx, yaw_command = None, 0., YawCommand(TURN_SPEED)
                        command_at, refresh_at = time.monotonic()-.05, -math.inf
                        route_wait = None
                        continue
                    if issue["kind"] != "unknown":
                        reason = "New observed obstacle entered the " if issue["kind"] == "occupied" else "Local lidar map boundary entered the "
                        raise RuntimeError(reason+label+f" (distance from body {issue['distance_m']:.2f} m); stopped; trigger marked on the map")
                    if route_wait is None:
                        route_wait = {"started_t": now, "outcome": "stopped", **issue}
                        self.report.setdefault("ground_waits", []).append(route_wait)
                    route_wait["duration_s"] = now-route_wait["started_t"]
                    if route_wait["duration_s"] >= 1.5:
                        raise RuntimeError(label+" ground coverage incomplete for 1.5 seconds; stopped; missing region marked on the map")
                    self.move(0.)
                    pending_pause = label+" ground coverage incomplete; stopping for a fresh scan"
                    continue
                if route_wait is not None:
                    route_wait.update(outcome="resumed", duration_s=now-route_wait["started_t"])
                    route_wait = None
                failure_grid = None
                plan.pop("blocked_at", None)
                self.report.pop("route_check", None)
                if desired_stage != stage:
                    if self.joystick_active:
                        self.move_joystick(0.)
                    self.move(0.)
                    # Neutralize both stick axes, then refresh the measured pose
                    # and checked route before issuing the next motion.
                    now = time.monotonic()
                    stage, stage_at = desired_stage, now
                    progress_at, progress_position, progress_yaw = now, position.copy(), yaw
                    progress_direction = error/np.linalg.norm(error)
                    speed_at, speed_position = now, position.copy()
                    vx, yaw_command = 0., YawCommand(TURN_SPEED)
                    turn_settled = False
                    turn_rate_floor, response_at, response_yaw = .22, now, yaw
                    turn_timeout = max(25., abs(angle)/.10+5.)
                    continue
                dt = min(max(now-command_at, 0.), .1)
                if stage == "aligning":
                    self.phase, self.message, plan["state"] = "aligning", f"Leg {waypoint}/{len(path)-1} · Turning to the route", "aligning"
                    if np.linalg.norm(position-progress_position) > .10:
                        raise RuntimeError("Navigation turn translation exceeded 10 cm; stopped")
                    if now-stage_at > turn_timeout:
                        raise RuntimeError(f"Navigation turn exceeded {turn_timeout:.0f} seconds; stopped")
                    if now-progress_at > 5:
                        required = min(math.radians(3), max(math.radians(.5), abs(angle)-.05))
                        if abs(wrap(yaw-progress_yaw)) < required:
                            raise RuntimeError("Navigation turn progress stalled; stopped")
                        progress_at, progress_yaw = now, yaw
                    # Increase small-angle commands only when feedback confirms
                    # a turn deadband. Keep final corrections brief and brake
                    # using both angular speed and transport delay.
                    if now-response_at >= .8:
                        if abs(wrap(yaw-response_yaw)) < .01 and abs(angle) > .025:
                            turn_rate_floor = min(TURN_SPEED, turn_rate_floor+.10)
                            self.report.setdefault("turn_adjustments", []).append(
                                {"t": now, "minimum_rad_s": turn_rate_floor, "heading_error": angle})
                        response_at, response_yaw = now, yaw
                    measured_w = abs(self.sport["yaw_speed"])
                    brake_angle = .02+min(.25, measured_w*.15+measured_w**2/(2*TURN_ACCELERATION))
                    turn_rate = min(TURN_SPEED, max(turn_rate_floor, 2*abs(angle)))
                    desired_yaw = math.copysign(turn_rate, angle) if abs(angle) > brake_angle else 0.
                    if not turn_settled:
                        turn_settled = np.linalg.norm(self.sport["velocity"][:2]) <= .04
                        if not turn_settled:
                            desired_yaw = 0.
                    vx = 0.
                else:
                    self.phase, self.message, plan["state"] = "navigating", f"Leg {waypoint}/{len(path)-1} · Walking along the route", "moving"
                    if now-progress_at > 5:
                        directed_progress = float(np.dot(position-progress_position, progress_direction))
                        if directed_progress < .02:
                            self.report["forward_stall"] = {"command_m_s": speed_limit,
                                "duration_s": now-progress_at, "progress_m": directed_progress,
                                **self.feedback_sample()}
                            if plan.get('failed_departures') or len(self.report.get('replans', [])) >= 3:
                                raise RuntimeError("No sustained displacement after the detour; stopped; check gait, onboard avoidance and control state")
                            trigger = {'kind': 'stall', 'stage': 'moving', 'yaw': travel_heading,
                                       'point': position.tolist(), **self.report['forward_stall']}
                            updated, grid, grid_stamp = await self.replan_navigation(
                                plan, goal, path, waypoint, travelled, trigger)
                            path, waypoint = np.asarray(updated['path']), 1
                            if len(path) == 1:
                                outcome, detail = 'complete', 'Arrived within the target radius'
                                break
                            stage, vx, yaw_command = None, 0., YawCommand(TURN_SPEED)
                            speed_limit = CRUISE_SPEED
                            command_at, refresh_at = time.monotonic()-.05, -math.inf
                            route_wait = None
                            continue
                        progress_at, progress_position = now, position.copy()
                        progress_direction = error/np.linalg.norm(error)
                    if now-speed_at >= 1.:
                        directed_progress = float(np.dot(position-speed_position, error/np.linalg.norm(error)))
                        if directed_progress/(now-speed_at) < .02 and speed_limit < MAX_SPEED:
                            old_speed = speed_limit
                            speed_limit = min(MAX_SPEED, round(speed_limit+.02, 2))
                            self.report.setdefault("forward_adjustments", []).append({"t": now,
                                "from_m_s": old_speed, "to_m_s": speed_limit,
                                "progress_m": directed_progress, "duration_s": now-speed_at})
                        speed_at, speed_position = now, position.copy()
                    if self.sport_sequence != speed_sequence:
                        speed_sequence = self.sport_sequence
                        measured_speed = float(np.linalg.norm(self.sport['velocity'][:2]))
                        if measured_speed > .28 and vx > .08 and forward_gain > .65:
                            forward_gain = max(.65, forward_gain-.03)
                            self.report.setdefault('speed_feedback_adjustments', []).append({
                                't': now, 'measured_m_s': measured_speed, 'command_scale': forward_gain})
                    desired_vx = max(.10, speed_limit*forward_gain)
                    vx += float(np.clip(desired_vx-vx, -.25*dt, .25*dt))
                    desired_yaw = (math.copysign(max(.18, min(.22, 1.8*abs(angle))), angle)
                                   if abs(angle) > .015 else 0.)
                wz = yaw_command.update(desired_yaw, self.sport["yaw_speed"], dt)
                try:
                    self.navigation_guard()
                except MotionPause as error:
                    self.move(0.)
                    pending_pause = str(error)
                    continue
                if stage == "aligning":
                    self.move_joystick(-wz*RIGHT_STICK_PER_YAW)
                else:
                    self.move_navigation(wz, vx)
                samples.append({"t": now, "position": position.tolist(), "distance_m": distance,
                                "stage": stage, "heading_error": angle, "speed_limit": speed_limit,
                                "vx": vx, "vy": 0., "wz": wz, **self.feedback_sample()})
                command_at = now
                await asyncio.sleep(.05)
        except asyncio.CancelledError:
            outcome, detail = "stopped", self.abort_reason or "Navigation stopped"
        except Exception as error:
            outcome, detail = "error", str(error) or f"{type(error).__name__}: robot control API did not confirm in time"
        finally:
            if controlled:
                self.phase, self.message = "stopping", "Stopping and confirming position"
                await self.stop_robot()
                await asyncio.sleep(1)
            distance = float(np.linalg.norm(np.asarray(self.sport["position"][:2])-goal)) if self.sport else None
            if outcome == "complete" and (distance is None or distance > ARRIVAL or time.monotonic()-self.state_time > .3):
                outcome, detail = "error", "Arrival not confirmed after stopping; check the position"
            if outcome == "complete" and (np.linalg.norm(self.sport["velocity"][:2]) > .025 or abs(self.sport["yaw_speed"]) > .04):
                outcome, detail = "error", "Motion feedback persists after stopping; check the robot"
            if "stop_response_error" in self.report or "release_response_error" in self.report:
                detail = (detail+"; " if outcome == "error" else "")+"Stop confirmation incomplete; check the robot"
                outcome = "error"
            plan.update(state=outcome, reason=detail, remaining_m=distance)
            self.report.update(outcome=outcome, message=detail, after=self.sport, distance_m=distance,
                               finished_unix=time.time(), path=plan.get("path", []))
            try:
                await asyncio.to_thread(self.save_navigation, self.report.copy(), samples, self.command_samples, failure_grid)
            except Exception:
                logging.exception("Navigation record could not be saved")
            self.phase, self.message, self.owner = outcome, detail, None

    @staticmethod
    def save_navigation(report, samples, commands, grid=None):
        folder = SITE / "navigation" / datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d-%H%M%S-%f")
        folder.mkdir(parents=True)
        (folder / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
        (folder / "samples.json").write_text(json.dumps(samples))
        (folder / "command_samples.json").write_text(json.dumps(commands))
        if grid is not None:
            np.savez_compressed(folder / "blocked-grid.npz", occupied=grid.occupied, observed=grid.observed,
                                origin=grid.origin, resolution=grid.resolution, floor=grid.floor,
                                start=grid.start, yaw=grid.yaw, ceiling=grid.ceiling)

    def check_stand(self):
        if not self.connected:
            raise RuntimeError("Robot connection lost")
        if time.monotonic()-self.state_time > .3 or time.monotonic()-self.low_time > .5:
            raise MotionPause("Waiting for live body and joint feedback")
        faults = [f for f in self.faults.values() if f["source"] != 400]
        if faults:
            raise RuntimeError("Robot reports active faults: "+", ".join(fault_text(f) for f in faults))
        if any(m.get("reserve", [0])[0] for m in (self.low or {}).get("motor_state", [])[:12]):
            raise RuntimeError("Joint or body fault still active; stand-up blocked")
        if not self.faults_seen:
            raise RuntimeError("Waiting for robot fault status")
        if max(abs(v) for v in self.sport["imu_state"]["rpy"][:2]) > .6:
            raise RuntimeError("Excessive body roll; adjust the resting pose at the robot")

    async def stand_handler(self, request):
        body = await request.json()
        async with self.action_lock:
            if self.motion_active():
                raise web.HTTPConflict(text="Stop the current action first")
            if self.fusion_busy():
                raise web.HTTPConflict(text="Photo 3D is building; wait before moving")
            try:
                self.check_stand()
            except RuntimeError as error:
                raise web.HTTPConflict(text=str(error))
            owner = self.operator(body)
            self.navigation = None
            self.schedule(owner, self.run_stand(owner))
            return web.json_response({"standing": True})

    async def run_stand(self, owner):
        self.owner = owner
        self.report = {"action": "stand", "before": self.sport}
        self.phase, self.message = "standing", "Standing up; supervise the robot"
        try:
            self.check_stand()
            mode = await self.rpc(RTC_TOPIC["MOTION_SWITCHER"], 1001)
            if json.loads(mode["data"]["data"])["name"] != "mcf":
                raise RuntimeError("Robot is not in the validated MCF mode")
            if self.abort.is_set():
                raise asyncio.CancelledError()
            self.check_stand()
            self.report["stand_ack"] = await self.rpc(RTC_TOPIC["SPORT_MOD"], 1004, {})
            for _ in range(120):
                if self.abort.is_set():
                    raise asyncio.CancelledError()
                self.check_stand()
                client = self.sockets.get(owner)
                if not client or time.monotonic()-client["heartbeat"] > 3:
                    raise RuntimeError("Stand-up page disconnected")
                await asyncio.sleep(.05)
            rpy = self.sport["imu_state"]["rpy"]
            if self.sport.get("body_height", 0) < .22 or max(abs(v) for v in rpy[:2]) > .18:
                raise RuntimeError("Stand command acknowledged, but standing pose not confirmed; check the robot")
            self.report["balance_ack"] = await self.rpc(RTC_TOPIC["SPORT_MOD"], 1002, {})
            self.phase, self.message = "idle", "Standing confirmed; waiting for lidar and turning envelope"
        except asyncio.CancelledError:
            await self.stop_robot()
            self.phase, self.message = "stopped", "Stop requested during stand-up; check the robot"
        except Exception as error:
            await self.stop_robot()
            self.phase, self.message = "error", str(error)
        finally:
            self.report["after"] = self.sport
            self.owner = None

    async def stop_handler(self, request):
        async with self.action_lock:
            if self.motion_active():
                self.request_stop("Operator clicked Stop now")
            elif self.connected:
                self.phase, self.message = "stopping", "Requesting stop"
                self.motion_task = asyncio.create_task(self.idle_stop())
            if self.navigation and not self.navigation["used"]:
                self.navigation = None
        return web.json_response({"stop_requested": True})

    async def idle_stop(self):
        self.report = {}
        await self.stop_robot()
        failed = "stop_response_error" in self.report or "release_response_error" in self.report
        self.phase = "error" if failed else "stopped"
        self.message = "Stop confirmation incomplete; check the robot" if failed else "Stop requested"

    async def reset_handler(self, request):
        async with self.action_lock:
            if self.motion_active():
                raise web.HTTPConflict(text="Stop motion before resetting the scene")
            started = time.time()
            await asyncio.to_thread(self.save_scene_state, started)
            self.scene_started_unix = started
            await self.reset_map(new_scene=True)
        return web.json_response({"reset": True, "status": self.status()})

    @staticmethod
    def save_scene_state(started):
        path = SITE / "scene_state.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps({"started_unix": started}))
        temporary.replace(path)

    def invalidate_fusion(self):
        self.fusion_generation += 1
        self.fusion = None
        self.fusion_version += 1
        self.fusion_message = "Map updated; rebuild photo 3D using photos from the same capture"

    async def fusion_handler(self, request):
        body = await request.json()
        try:
            model = CameraModel(**body["camera"]) if "camera" in body else self.fusion_camera
        except (TypeError, ValueError) as error:
            raise web.HTTPBadRequest(text=str(error))
        async with self.action_lock:
            if self.motion_active():
                raise web.HTTPConflict(text="Wait for motion to finish before building photo 3D")
            try:
                await self.start_fusion(model)
            except RuntimeError as error:
                raise web.HTTPConflict(text=str(error))
        return web.json_response({"building": True})

    async def spatial_handler(self, request):
        body = await request.json()
        self.operator(body)
        enabled = body.get('enabled')
        if not isinstance(enabled, bool):
            raise web.HTTPBadRequest(text='Specify whether capture is enabled')
        async with self.action_lock:
            if self.fusion_busy():
                raise web.HTTPConflict(text='Wait for Spatial 3D to finish building')
            if enabled:
                if self.motion_active():
                    raise web.HTTPConflict(text='Start spatial capture before navigation')
                if (not self.connected or self.sport is None or time.monotonic()-self.state_time > .5
                        or self.cloud is None or self.metadata is None or time.monotonic()-self.lidar_time > .5):
                    raise web.HTTPConflict(text='Wait for live pose and lidar from this connection')
                if self.spatial_capture:
                    return web.json_response({'capturing': True})
                if any(p.get('historical') or p.get('connection_epoch') != self.connection_epoch
                       for p in self.photos.values()):
                    self.photos.clear()
                    self.photo_sequence += 1
                    self.panorama = None
                    self.panorama_version += 1
                    self.invalidate_fusion()
                if len(self.photos) >= MAX_PHOTOS:
                    raise web.HTTPConflict(text='Photo limit reached; save Spatial 3D, then reset the scene before another capture')
                self.session = SITE/'spatial'/datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d-%H%M%S-%f')
                await asyncio.to_thread(self.session.mkdir, parents=True, exist_ok=False)
                self.spatial_capture = True
                self.spatial_message = 'Spatial capture active: saving positions and headings along the route'
            else:
                self.spatial_capture = False
                self.spatial_message = ('Capture finished; build Spatial 3D after navigation ends' if self.motion_active()
                                        else 'Capture finished; building Spatial 3D')
                if self.session and self.session.parent == SITE/'spatial':
                    await self.checkpoint_spatial()
                if not self.motion_active() and self.photos and len(self.map_keys):
                    await self.start_fusion(self.fusion_camera)
        return web.json_response({'capturing': self.spatial_capture})

    async def checkpoint_spatial(self):
        async with self.spatial_save_lock:
            folder, photos = self.session, [dict(p) for p in self.photos.values()]
            resolution = float((self.metadata or {}).get('resolution', .05))
            points = self.map_keys.astype(np.float32)*resolution
            try:
                await asyncio.to_thread(save_capture, folder, photos, points, resolution)
                self.spatial_saved_at = time.monotonic()
            except Exception:
                logging.exception('Spatial capture checkpoint')
                self.spatial_message = 'Could not save spatial capture; check disk space and finish capture again'

    async def start_fusion(self, model):
        """Snapshot one dataset while the caller holds action_lock."""
        if self.fusion_busy():
            return
        if (not self.spatial_capture and self.session and self.session.parent == SITE/'spatial'
                and self.photos and not any(p.get('historical') for p in self.photos.values())):
            await self.checkpoint_spatial()
        async with self.map_lock:
            if self.spatial_capture:
                raise RuntimeError('Finish spatial capture before building Spatial 3D')
            if not self.photos or not len(self.map_keys):
                raise RuntimeError("Capture photos and lidar first")
            archived = (all(p.get('historical') for p in self.photos.values()) and self.session
                        and (self.session/'room_cloud.npz').is_file())
            if any(p.get("historical") for p in self.photos.values()) and not self.recorded_map and not archived:
                raise RuntimeError("Photos are from a previous connection; capture again in the current map to align coordinates")
            if self.recorded_map and self.session is None:
                raise RuntimeError("No paired capture metadata; complete a new photo capture first")
            resolution = float((self.metadata or {}).get("resolution", .05))
            points = (np.load(self.session/'room_cloud.npz')['points'] if archived
                      else self.map_keys.astype(np.float32)*resolution)
            photos = [dict(p) for p in self.photos.values()]
            dataset = self.session.name if self.session else f"connection-{self.connection_epoch}"
            self.fusion_camera = model
            self.fusion_message = "Building photo depth, textured surfaces and colored cloud…"
            self.fusion_task = asyncio.create_task(self.run_fusion(points, photos, model, dataset,
                                                                   self.fusion_generation, self.floor))

    async def run_fusion(self, points, photos, model, dataset, generation, floor):
        try:
            folder = SITE / "fusion" / datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d-%H%M%S-%f")
            manifest = await asyncio.to_thread(build_fusion, folder, points, photos, model, dataset, floor)
            if generation != self.fusion_generation:
                return
            self.fusion = manifest
            self.fusion_version += 1
            self.fusion_message = f"Built {manifest['colored_points']:,} colored points and {manifest['surface_triangles']:,} textured triangles"
        except Exception as error:
            if generation != self.fusion_generation:
                return
            self.fusion_message = "Photo 3D build failed: "+str(error)
            logging.exception("Photo-depth fusion")

    async def camera_handler(self, request):
        if not self.jpeg:
            raise web.HTTPNotFound(text="Waiting for camera")
        return web.Response(body=self.jpeg, content_type="image/jpeg", headers={"Cache-Control": "no-store"})

    async def photo_handler(self, request):
        photo = self.photos.get(int(request.match_info["slot"]))
        if not photo:
            raise web.HTTPNotFound()
        return web.Response(body=photo["jpeg"], content_type="image/jpeg", headers={"Cache-Control": "no-store"})

    async def panorama_handler(self, request):
        if self.panorama is None or request.query.get("v") != str(self.panorama_version):
            raise web.HTTPNotFound(text="Panorama has changed")
        result = self.panorama
        if 'station' in request.query:
            result = next((v for v in self.panorama.get('stations', [])
                           if str(v['station']) == request.query['station']), None)
            if result is None:
                raise web.HTTPNotFound(text='No panorama at this station')
        return web.Response(body=result["image"], content_type="image/png", headers={"Cache-Control": "no-store"})

    async def export_handler(self, request):
        resolution = self.metadata["resolution"] if self.metadata else .05
        points = self.map_keys.astype(np.float32)*resolution
        def encode():
            head = (f"ply\nformat ascii 1.0\ncomment Go2 odom voxel centers, metres\nelement vertex {len(points)}\n"
                    "property float x\nproperty float y\nproperty float z\nend_header\n")
            return (head+"".join("%.3f %.3f %.3f\n" % tuple(p) for p in points)).encode()
        return web.Response(body=await asyncio.to_thread(encode), content_type="application/octet-stream",
                            headers={"Content-Disposition": 'attachment; filename="go2-room.ply"'})

    def load_saved_scene(self):
        if not self.scene_started_unix:
            recorded = ROOT / "outputs/go2-room-scan/rotation-area/rotation-full-03"
            if (recorded / "after_cloud.npz").is_file():
                self.metadata = json.loads((recorded / "rotation_result.json").read_text())["lidar_metadata"]
                points = np.load(recorded / "after_cloud.npz")["points"]
                self.map_keys = merge_voxels(self.map_keys, points, self.metadata["resolution"])
                self.map_packet = point_packet(1, self.map_keys.astype(np.float32)*self.metadata["resolution"])
                self.recorded_map = True
                self.floor = -.075
            # Previously recorded photos provide a rough scene preview before a new run.
            area = ROOT / "outputs/go2-room-scan/rotation-area"
            for folder, filename, state in [("rotation-api-01", "camera_before.jpg", "before"),
                                             ("rotation-sport-02", "camera_after.jpg", "after"),
                                             ("rotation-full-02", "camera_after.jpg", "after")]:
                source = area / folder
                if (source / filename).is_file():
                    record = json.loads((source / "rotation_result.json").read_text())[state]
                    yaw = record["imu_state"]["rpy"][2]
                    slot = int(round((math.degrees(yaw) % 360)/15)) % 24
                    self.photo_sequence += 1
                    self.photos[slot] = {"slot": slot, "version": self.photo_sequence, "yaw": yaw,
                        "position": record["position"], "jpeg": (source / filename).read_bytes(),
                        "saved_monotonic": -1e9, "historical": True, "span_deg": 90}
        sessions = sorted((SITE / "sessions").glob("*/result.json"), reverse=True)
        if sessions:
            saved = sessions[0].parent
            report = json.loads(sessions[0].read_text())
            if report.get("outcome") == "complete" and report.get("started_unix", 0.) >= self.scene_started_unix:
                self.report, self.session = report, saved
                self.metadata = {"resolution": .05}
                points = np.load(saved / "room_cloud.npz")["points"]
                self.map_keys = np.rint(points/.05).astype(np.int32)
                self.map_packet = point_packet(1, points)
                self.recorded_map = True
                self.photos.clear()
                for photo in json.loads((saved / "photos.json").read_text()):
                    self.photo_sequence += 1
                    self.photos[photo["slot"]] = {**photo, "version": self.photo_sequence,
                        "jpeg": (saved / photo["file"]).read_bytes(), "historical": True,
                        "saved_monotonic": -1e9}
                    if photo.get("depth_file"):
                        self.photos[photo["slot"]]["depth_points"] = np.load(saved / photo["depth_file"])["points"]
                self.tracker = TurnProgress(report["after"]["imu_state"]["rpy"][2])
                self.tracker.radians = math.radians(report["actual_degrees"])
                self.origin = np.asarray(report["before"]["position"][:2])
                self.peak_shift = report["peak_shift_m"]
                self.phase, self.message = "complete", "Last demo result loaded · Arm before another demo"
        captures = sorted((SITE/'spatial').glob('*/photos.json'), reverse=True)
        if captures:
            source = captures[0]
            newer = self.session is None or source.parent.name > self.session.name
            if newer and source.stat().st_mtime >= self.scene_started_unix:
                saved = source.parent
                records = json.loads(source.read_text())
                if records and (saved/'room_cloud.npz').exists():
                    self.session = saved
                    resolution = json.loads((saved/'capture.json').read_text())['resolution']
                    self.metadata = {'resolution': resolution}
                    points = np.load(saved/'room_cloud.npz')['points']
                    self.map_keys = np.rint(points/resolution).astype(np.int32)
                    self.map_packet, self.recorded_map = point_packet(1, points), True
                    self.photos.clear()
                    for photo in records:
                        self.photo_sequence += 1
                        self.photos[photo['slot']] = {**photo, 'version': self.photo_sequence,
                            'historical': True, 'saved_monotonic': -1e9,
                            'jpeg': (saved/photo['file']).read_bytes()}
                        if photo.get('depth_file'):
                            self.photos[photo['slot']]['depth_points'] = np.load(saved/photo['depth_file'])['points']
                    self.spatial_message = f'Restored capture at {len(stations(self.photos.values()))} stations'
        if self.session:
            for source in sorted((SITE / "fusion").glob("*/manifest.json"), reverse=True):
                manifest = json.loads(source.read_text())
                if manifest.get("dataset_session") == self.session.name:
                    self.fusion = manifest
                    self.fusion_camera = CameraModel(**manifest["camera"])
                    self.fusion_version += 1
                    self.fusion_message = "Loaded paired photo 3D · Camera calibration is approximate"
                    break

    async def startup(self, app):
        saved_turn = SITE / "turn_state.json"
        if saved_turn.exists():
            state = json.loads(saved_turn.read_text())
            self.turn_token, self.turn_latched = state["token"], state["latched"]
        elif any((SITE / "sessions").glob("*/result.json")):
            self.turn_latched = True
            self.save_turn_state()
        scene_state = SITE / "scene_state.json"
        if scene_state.exists():
            self.scene_started_unix = json.loads(scene_state.read_text())["started_unix"]
        await asyncio.to_thread(self.load_saved_scene)
        self.compute_pool = ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context("spawn"))
        self.tasks = [asyncio.create_task(f()) for f in
                      (self.connect_loop, self.geometry_loop, self.map_loop, self.broadcast, self.timing_loop, self.panorama_loop)]

    async def shutdown(self, app):
        self.request_stop("Demo service is shutting down")
        if self.motion_active():
            with contextlib.suppress(Exception, asyncio.CancelledError):
                await asyncio.wait_for(self.motion_task, 10)
        self.running = False
        if self.fault_capture_task is not None:
            await self.fault_capture_task
        if self.fusion_busy():
            await self.fusion_task
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        if self.compute_pool is not None:
            await asyncio.to_thread(self.compute_pool.shutdown, wait=True, cancel_futures=True)
        for client in list(self.sockets.values()):
            await client["ws"].close()


def create_app(demo):
    app = web.Application(client_max_size=8192)
    async def index(request):
        return web.FileResponse(SITE / "index.html", headers={"Cache-Control": "no-cache"})
    async def status(request):
        return web.json_response(demo.status())
    async def diagnostics(request):
        control = None
        if request.query.get("control") == "1":
            if demo.motion_active():
                raise web.HTTPConflict(text="Read control status after motion finishes")
            control = {}
            for name, topic, api, parameter in (
                ("mode", RTC_TOPIC["MOTION_SWITCHER"], 1001, None),
                ("sport", RTC_TOPIC["SPORT_MOD"], 1034,
                 ["state", "bodyHeight", "speedLevel", "gait", "continuousGait", "economicGait"]),
                ("avoidance", RTC_TOPIC["OBSTACLES_AVOID"], 1002, None)):
                try:
                    control[name] = await demo.rpc(topic, api, parameter, timeout=2.)
                except Exception as error:
                    control[name] = {"error": str(error) or type(error).__name__}
        turn_space = None
        if demo.turn_points is not None and demo.sport:
            turn_space = {"occupied_xy": demo.turn_points.tolist(), "resolution": demo.turn_resolution,
                          "scan_age_s": demo.age(demo.geometry_time), "checked_angle_deg": math.degrees(turn_lookahead(demo.sport["yaw_speed"]))}
            for name, direction in (("left", 1), ("right", -1)):
                turn_space[name+"_clearance_m"] = swept_clearance(demo.turn_points, demo.sport["position"],
                    demo.sport["imu_state"]["rpy"][2], direction*turn_lookahead(demo.sport["yaw_speed"]), demo.turn_resolution,
                    body_only=demo.turn_channel == "joystick")
        return web.json_response({"status": demo.status(), "low_state": demo.low, "sport_state": demo.sport,
                                  "turn_space": turn_space, "control": control,
                                  "last_fault_record": demo.last_fault_record,
                                  "power_history": list(demo.power_history) if request.query.get("power") == "1" else None})
    app.router.add_get("/", index)
    app.router.add_get("/api/status", status)
    app.router.add_get("/api/diagnostics", diagnostics)
    app.router.add_get("/ws", demo.ws_handler)
    app.router.add_post("/api/start", demo.start_handler)
    app.router.add_post("/api/turn/rearm", demo.rearm_handler)
    app.router.add_post("/api/turn/probe", demo.turn_probe_handler)
    app.router.add_post("/api/plan", demo.plan_handler)
    app.router.add_post("/api/navigate", demo.navigate_handler)
    app.router.add_post("/api/stand", demo.stand_handler)
    app.router.add_post("/api/stop", demo.stop_handler)
    app.router.add_post("/api/reset", demo.reset_handler)
    app.router.add_post("/api/fusion/build", demo.fusion_handler)
    app.router.add_get("/camera.jpg", demo.camera_handler)
    app.router.add_get("/photos/{slot:[0-9]+}.jpg", demo.photo_handler)
    app.router.add_get("/panorama.png", demo.panorama_handler)
    app.router.add_post('/api/spatial/capture', demo.spatial_handler)
    app.router.add_get("/map.ply", demo.export_handler)
    app.router.add_static("/vendor/", SITE / "vendor")
    app.router.add_static("/assets/", SITE / "assets")
    (SITE / "fusion").mkdir(parents=True, exist_ok=True)
    app.router.add_static("/fusion/data/", SITE / "fusion")
    app.on_startup.append(demo.startup)
    app.on_shutdown.append(demo.shutdown)
    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=63541)
    parser.add_argument("--ip", type=robot_address, default="auto", help="Robot IPv4 address; auto discovers the configured serial number")
    parser.add_argument("--key-file", type=Path, default=DEFAULT_DEVICE_FILE, help="Path to the local device JSON")
    args = parser.parse_args()
    try:
        device = read_device(args.key_file)
        if args.ip == "auto" and not device["sn"]:
            parser.error("Auto discovery requires sn in the device config, or specify --ip")
    except ValueError as error:
        parser.error(str(error))
    (SITE / "assets").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("aiortc.codecs.h264").setLevel(logging.ERROR)
    web.run_app(create_app(Demo(args.ip, args.key_file)), host="127.0.0.1", port=args.port, access_log=None)
