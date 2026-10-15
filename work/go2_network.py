"""Select the same Go2 on a new LAN by its saved serial number."""
import asyncio
import ipaddress
import json
from pathlib import Path

from unitree_webrtc_connect import discover_ip_sn

DEFAULT_DEVICE_FILE = Path.home() / "Library/Application Support/Go2WiFi/device.json"


class DiscoveryError(RuntimeError):
    pass


def robot_address(value):
    if value == "auto":
        return value
    try:
        return str(ipaddress.IPv4Address(value))
    except ipaddress.AddressValueError:
        raise ValueError("Robot address must be auto or an IPv4 address") from None


def read_device(path):
    try:
        data = json.loads(Path(path).expanduser().read_text())
        key = data["aes_128_key"].strip()
        if len(key) != 32 or len(bytes.fromhex(key)) != 16:
            raise ValueError()
        serial = str(data.get("sn") or "").strip()
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        raise ValueError("Could not read device configuration; check device.json and the AES-128 key format") from None
    return {"sn": serial, "aes_128_key": key}


async def resolve_robot_ip(address, serial):
    address = robot_address(address)
    if address != "auto":
        return address
    if not serial:
        raise DiscoveryError("Auto discovery requires sn in device.json; alternatively specify --ip")
    devices = await asyncio.to_thread(discover_ip_sn, timeout=2, device_type="Go2", sn=serial)
    if serial not in devices:
        raise DiscoveryError("Robot not found; check the LAN connection or specify --ip")
    return robot_address(devices[serial])
