# Offline testing

The public suite uses synthetic geometry, simulated time, mocked transports and
HTTP/WebSocket test servers. It does not connect to a robot. Run it from the
repository root using the commands in the README.

Coverage includes ground/body envelopes, missing observations, turn clearance,
arrival regions, multi-leg execution, obstacle replanning, stop confirmation,
stall recovery, feedback gaps, scene resets, request ownership, stale plans,
fault capture, spatial grouping and photo/lidar pairing. JavaScript scenarios
exercise route rendering and spatial station behavior without a browser runtime.
`verify_install.py` checks imports and web assets without starting robot tasks.

Thirteen historical tests depended on private field point clouds. Those test
cases and their recordings are excluded from the public release; no skipped
field tests are counted as passed. Shared synthetic coverage remains available.
Private field recordings do not establish a success rate for other environments.

## Synthetic controller replay

`build_synthetic_replay.py` constructs a floor and wall procedurally and runs the
actual navigation controller under three assumed planar actuator profiles. The
saved JSON is used by `/assets/navigation-simulation.html` and contains no field
measurements. Latency, gain, acceleration and command deadband are modeled;
contacts, slip, leg motion, onboard avoidance internals and odometry drift are not.
Geometric map checks share the planner representation rather than providing an
independent physical collision detector.

`go2_navigation_sim.py` can also process recordings that you collect locally.
`go2_navigation_replay.py` compares matched before/after runs. Generated reports
and maps can disclose your space: keep them outside source control.

The offline preview uses `PreviewDemo`, whose startup runs only WebSocket
broadcasting. It never loads local captures or starts a robot connection, and
middleware rejects non-read-only HTTP methods. Use the live launcher only when
you intend to connect your configured robot.
