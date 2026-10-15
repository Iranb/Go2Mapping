# Third-party notices

Go2Mapping source is released under the MIT License in [LICENSE](LICENSE).
Third-party code retains its own copyright and license notices.

| Component | Included path | License and attribution |
|---|---|---|
| unitree_webrtc_connect 2.2.0 | `vendor/unitree_webrtc_connect/` | MIT, Copyright (c) 2025 Konstantin Severov. See its [LICENSE](vendor/unitree_webrtc_connect/LICENSE). [Upstream](https://github.com/legion1581/unitree_webrtc_connect). This vendored snapshot includes compatibility changes used by this application. |
| Three.js and OrbitControls | `outputs/go2-live-demo/vendor/` | MIT, Copyright 2010–2024 Three.js Authors. See [THREE-LICENSE.txt](outputs/go2-live-demo/vendor/THREE-LICENSE.txt). [Upstream](https://github.com/mrdoob/three.js). |

The vendored Unitree client includes its lidar decoder and `libvoxel.wasm`
package asset. Keep the accompanying vendor notices when redistributing it.
Python dependencies are installed separately and retain their respective licenses.
No robot firmware, account credentials, field photos or field point clouds are
included in this repository. Unitree and Go2 are names of their respective owners;
this is an independent community project.
