# Go2Mapping

**面向宇树 Go2 的本地建图、地图选点导航与空间实景浏览工具。**

[English](README.md) · [MIT 协议](LICENSE) · [第三方声明](THIRD_PARTY_NOTICES.md)

通过局域网 WebRTC 连接机器狗，在本机浏览器显示实时雷达、规划路线和沿途采集的空间照片。
系统界面默认英文，网页资源随源码提供，运行时不依赖外部 CDN。

## 主要功能

- 实时点云累积、当前扫描、机头方向、俯视与三维视角、截图及 PLY 导出。
- 地图选点后预览行走段和转向动作，再点击执行；检查机身包络、地面观测与实际到达位置。
- 遇到新障碍先停车并确认反馈，等待新扫描，从实际位置重新规划。临时落点与最终目标分别显示。
- 行走停滞时允许一次经过检查的替代出发方向；再次停滞则停止。
- 沿途多个位置独立采集照片，按位置浏览环视；不会把整个空间压在单点球面上。
- 使用配对雷达提供米制深度，生成同一里程计坐标系中的彩色点云与照片纹理面。
- 显式准备、单次受理的两圈建图演示，按实测转角统计进度。
- 无需机器狗的只读界面预览、合成场景控制回放与离线测试。

公开仓库不包含设备密钥、个人配置、实测照片、室内点云、日志或交接备份。

## 安装环境

启动脚本面向 **macOS**，使用 Python 3.12、Homebrew 和 PortAudio。
实机运行还需要兼容的 Go2 WebRTC 接口、本人绑定设备的 AES-128 密钥，以及同一可信局域网。
当前验证过的运动路径使用 MCF 模式，并要求机载避障开启；不同固件的兼容性可能不同。

```sh
git clone https://github.com/Iranb/Go2Mapping.git
cd Go2Mapping
chmod +x *.command
./setup.command
```

未安装 Homebrew 时请先访问 [brew.sh](https://brew.sh/)。
安装会准备项目内的 `.venv`、`.python`、`.tools` 与 `wheelhouse`，这些环境文件不进入源码仓库。
准备完成后，`./install-offline.command` 可用本地缓存重建环境；目标电脑需要已安装 PortAudio，
缓存必须匹配其操作系统和处理器架构。

## 配置机器狗

1. 让电脑与机器狗连接到同一可信局域网。
2. 在本机交互式获取本人绑定设备的密钥：

   ```sh
   .venv/bin/python work/fetch_go2_key.py --region cn
   ```

   全球地区账号使用 `--region global`。工具通过对应云端登录，仅在本机保存所选设备的序列号和 AES 密钥，不保存账号密码。
3. 默认配置位置为 `~/Library/Application Support/Go2WiFi/device.json`。
   也可参考 [device.example.json](examples/device.example.json) 创建私有配置，用真实值替换两个占位符，
   执行 `chmod 600 /path/to/device.json`，再通过 `--key-file` 指定。
4. 默认按序列号自动发现设备。若要指定地址，把
   [connection.example.json](examples/connection.example.json) 复制为根目录 `connection.json`，
   将 `ip` 改成机器狗地址。该文件已被 Git 忽略；启动时也可用 `--ip YOUR_ROBOT_IP` 覆盖。

## 启动与使用

```sh
./start.command
```

打开 **http://127.0.0.1:63541/**，或双击 `start.command`。
保留终端窗口，按 Ctrl+C 退出。服务启动后连接遥测；运动需要在网页中单独发起。
若提示设备被占用，请退出其他手机图传或遥控客户端。

```sh
./start.command --key-file /path/to/device.json --ip YOUR_ROBOT_IP
```

服务仅监听本机回环地址，没有面向远程用户的认证机制，请勿转发端口或暴露到不可信网络。

### 地图导航

先点击 **Stand up** 并等待实时反馈和雷达就绪，再点击 **Pick target**，在地面选点。
检查路线与转向步骤后，点击 **Go to target**。
**Replan target** 只更新预览，需再次点击前往才会执行。

当前限制为目标距离 4 m 内、累计路线不超过 6 m、最多三次重规划、120 秒，
并在目标 20 cm 内确认停稳。绿色表示已检查路线，琥珀色连接最终目标的部分仍未确认可通行。
到达临时落点不代表最终到达。重连或坐标系改变后需要重新选点。

### 空间采集与浏览

静止时点击 **Start spatial capture**，再导航经过多个位置。
停止后点击 **Finish capture** 保存并生成。使用 **Spatial photos** 按位置环视，
使用 **Spatial 3D** 浏览共享坐标系中的空间。采集开关本身不会触发机器狗运动。

相机参数仍为近似值，正俯仰表示向下，修改后需重新生成。
里程计漂移、外参误差、近景视差与遮挡可能造成错位；这不是经过标定的 SLAM 或摄影测量系统。
导航继续使用原始雷达，不使用照片纹理面作为避障依据。

## 无机器狗预览

```sh
.venv/bin/python work/preview_ui.py
```

打开 **http://127.0.0.1:63542/**。只读预览使用程序生成的场景，不建立机器人传输连接，并拒绝控制请求。
导航回放链接展示三种假定执行器响应。重新生成合成回放：

```sh
.venv/bin/python work/build_synthetic_replay.py
```

## 测试与开发

```sh
.venv/bin/python work/verify_install.py
.venv/bin/python -m unittest discover -s work -p 'test_go2_*.py'
node work/test_go2_navigation_ui.mjs
node work/test_go2_spatial_ui.mjs
.venv/bin/python work/check_public_source.py
```

Node.js 仅用于 JavaScript 回归测试。公开测试使用合成输入与模拟传输，
依赖私有实测数据的专项回归未公开。参见 [测试说明](docs/TESTING.md) 与 [隐私说明](docs/PRIVACY.md)。
后端入口为 `work/go2_live_demo.py`，导航、空间采集、环视与照片深度模块分别位于 `go2_*.py`。
网页入口为 `outputs/go2-live-demo/index.html`。GitHub Actions 配置会运行离线检查。

## 使用边界

这是实验性机器人控制软件，需要在平地、空旷区域由现场人员看护。
雷达可能漏检玻璃、线缆、低矮物体和落差。软件停止请求依赖连接与机器人响应，不能代替物理急停。
新的避障行为仍需针对具体机器狗与环境开展有人看护的实机验证。

## 开源协议

项目源码采用 [MIT 协议](LICENSE)。第三方代码保留原始许可与署名，详见
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
