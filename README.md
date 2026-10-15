# Go2Mapping

**Local mapping, click-to-go navigation and spatial photo views for the Unitree Go2.**

[中文说明](README.zh-CN.md) · [MIT License](LICENSE) · [Third-party notices](THIRD_PARTY_NOTICES.md)

Go2Mapping connects to a Go2 over the local network using WebRTC and serves an
English web interface on your computer. It combines live lidar, route previews,
feedback-driven motion and photos captured at multiple positions in one odometry
frame. Browser assets are bundled, so the running interface needs no CDN.

## Features

- **Live point cloud:** accumulated height-colored lidar, current scan, robot heading,
  top/orbit views, snapshots and PLY export.
- **Click-to-go navigation:** preview walking legs and turns before starting;
  body-envelope checks, ground-observation checks and arrival verification.
- **Obstacle recovery:** stop and confirm feedback, wait for fresh scans, then
  replan from the measured pose. Partial routes are distinguished from a complete
  route to the goal. A stalled departure gets one checked alternative attempt.
- **Spatial photos:** capture along a route and browse separate photo stations,
  instead of projecting the entire space onto a single panorama.
- **Spatial 3D:** paired lidar supplies metric photo depth, colored points and
  textured surfaces in the shared odometry frame. Missing depth stays empty.
- **Two-turn demo:** an explicitly armed, single-use scan task with measured
  angular progress, telemetry guards and stop requests.
- **Offline tools:** synthetic controller replay, robot-free UI preview and
  regression tests. No field recordings are distributed.

## Requirements

The supplied launch scripts target **macOS** with Python 3.12, Homebrew and
PortAudio. Setup downloads a project-local Python runtime and dependencies.
A Go2 with a compatible WebRTC interface, its bound-device AES-128 key and local
network access is required for live operation. The validated motion path uses
MCF mode with onboard obstacle avoidance enabled. Firmware compatibility varies.

## Install

```sh
git clone https://github.com/Iranb/Go2Mapping.git
cd Go2Mapping
chmod +x *.command
./setup.command
```

Install [Homebrew](https://brew.sh/) first if it is unavailable. Setup prepares
`.venv`, `.python`, `.tools` and a local `wheelhouse`. These directories are not
part of the source release. After preparation, `./install-offline.command`
recreates the environment using the local runtime and wheelhouse. PortAudio must
already be installed; offline caches must match the destination OS/architecture.

## Configure your robot

1. Connect your computer and robot to the same trusted local network.
2. Obtain the key for a robot bound to your own Unitree account:

   ```sh
   .venv/bin/python work/fetch_go2_key.py --region global
   ```

   Use `--region cn` for the China cloud. The helper prompts locally, uses the
   cloud for this login and saves only the selected serial number and AES key.
   It does not save your account password.
3. The default private configuration is
   `~/Library/Application Support/Go2WiFi/device.json`. Alternatively, create a
   private file based on [device.example.json](examples/device.example.json)
   and supply `--key-file /path/to/device.json`. Replace both placeholders.
   Restrict its permissions with `chmod 600 /path/to/device.json`.
4. Automatic discovery uses the saved serial number. To set an explicit address,
   copy [connection.example.json](examples/connection.example.json) to
   `connection.json` and change `ip` to your robot address. This file is ignored
   by Git. You can also pass `--ip YOUR_ROBOT_IP` when starting.

## Run

```sh
./start.command
```

Open **http://127.0.0.1:63541/**. Double-clicking `start.command` also works.
Keep the terminal open; press Ctrl+C there to stop the service. Starting the
service connects for telemetry. Motion requires a separate web action.
Close other phone video/control clients if the robot reports that it is busy.

For a different key file or address:

```sh
./start.command --key-file /path/to/device.json --ip YOUR_ROBOT_IP
```

The server binds to loopback. It has no remote-user authentication; do not expose
or port-forward the interface to an untrusted network.

### Navigate

Stand the robot up, wait for fresh feedback and lidar, then choose **Pick target**.
Click the ground, review the route and actions, and choose **Go to target**.
**Replan target** updates the preview; it does not itself start motion.

The current limits are a target within 4 m, total planned/cumulative travel within
6 m, up to three replans, 120 seconds and a settled arrival within 20 cm.
Green shows a checked route; an amber connection to the final goal remains
unverified. Reaching a temporary stop is not final arrival. Select a new target
after reconnection or a coordinate-frame change.

### Capture a space

Choose **Start spatial capture** while stationary, then navigate through several
positions. Stop and choose **Finish capture** to save and build. Switch between
**Spatial photos** for station-based look-around views and **Spatial 3D** for the
shared scene. Capture itself does not issue robot motion commands.

Camera parameters are approximate. Positive pitch points down. Rebuild after
changing camera parameters. Odometry drift, uncalibrated camera extrinsics,
parallax and occlusion can cause misalignment; this is not a calibrated SLAM or
photogrammetry pipeline. Navigation uses raw lidar, not the photo surface.

## Try it without a robot

```sh
.venv/bin/python work/preview_ui.py
```

Open **http://127.0.0.1:63542/**. This read-only preview uses a procedural scene,
creates no robot transport and rejects control requests. The navigation replay
link shows three simulated actuator responses. To regenerate that replay:

```sh
.venv/bin/python work/build_synthetic_replay.py
```

## Tests and development

```sh
.venv/bin/python work/verify_install.py
.venv/bin/python -m unittest discover -s work -p 'test_go2_*.py'
node work/test_go2_navigation_ui.mjs
node work/test_go2_spatial_ui.mjs
.venv/bin/python work/check_public_source.py
```

Node.js is only needed for the JavaScript regression tests. The public suite
uses synthetic inputs and mocked transports. Private field captures and their
capture-specific regressions are excluded. See [testing notes](docs/TESTING.md)
for simulation limits and [privacy notes](docs/PRIVACY.md) before sharing captures.

The main backend is `work/go2_live_demo.py`; navigation, spatial capture, panorama
and photo-depth processing are separate `go2_*.py` modules. The browser entry is
`outputs/go2-live-demo/index.html`. GitHub Actions runs the offline checks.

## Operating limits

This is experimental robot-control software. Use level ground, clear space and
an operator ready to intervene. Lidar can miss glass, cables, small objects and
drops. A software stop request depends on the connection and robot response;
it does not replace a physical emergency stop. New avoidance behavior still
requires supervised hardware validation for your robot and environment.

## License

Project source is available under the [MIT License](LICENSE). Vendored components
retain their original licenses and attribution; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
