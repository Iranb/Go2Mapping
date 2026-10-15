"""Fetch a bound Go2 key with local interactive account input."""

import argparse
import getpass
import json
import os
from pathlib import Path

KEY_FILE = Path.home() / "Library/Application Support/Go2WiFi/device.json"


def main():
    parser = argparse.ArgumentParser(description="Fetch the key for a bound Go2 and save it locally")
    parser.add_argument("--region", choices=("cn", "global"), default="cn")
    args = parser.parse_args()

    from unitree_webrtc_connect import UnitreeCloud, UnitreeCloudError

    print("Credentials are used for this login only. Only the selected serial number and AES key are saved locally.")
    print("The cloud response determines which account formats are supported.")
    account = input("Unitree account (phone or email): ").strip()
    if not account:
        raise SystemExit("No account entered.")
    password = getpass.getpass("Unitree password (hidden): ")
    if not password:
        raise SystemExit("No password entered.")

    cloud = UnitreeCloud(region=args.region, device_type="Go2")
    try:
        cloud.login_email(account, password)
        devices = cloud.list_devices()
    except UnitreeCloudError as exc:
        raise SystemExit(f"Unitree cloud response: {exc}") from None
    finally:
        password = None

    if not devices:
        raise SystemExit("No bound devices returned. Check the account and region.")
    for number, device in enumerate(devices, start=1):
        print(f"{number}. SN: {device.sn}  Name: {device.alias or 'Unnamed'}")

    selected = 0
    if len(devices) > 1:
        try:
            selected = int(input("Select the device number: ")) - 1
        except ValueError:
            raise SystemExit("Invalid device number.") from None
        if not 0 <= selected < len(devices):
            raise SystemExit("Invalid device number.")
    device = devices[selected]
    key = device.key.strip()
    try:
        valid = len(key) == 32 and len(bytes.fromhex(key)) == 16
    except ValueError:
        valid = False
    if not valid:
        raise SystemExit("The selected device did not return a valid AES-128 key.")

    KEY_FILE.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(KEY_FILE, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        os.fchmod(stream.fileno(), 0o600)
        json.dump({"sn": device.sn, "aes_128_key": key}, stream, indent=2)
        stream.write("\n")
    print(f"Saved device {device.sn} key to: {KEY_FILE}")
    print("Run start.command to open Go2Mapping.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        raise SystemExit(f"Key retrieval failed: {type(exc).__name__}: {exc}") from None
