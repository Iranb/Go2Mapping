"""Check the staged Git snapshot for private runtime data and common secrets.

Print file names and rule names only, never the matched credential itself.
This supplements human review; it is not a general anonymization guarantee.
"""
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
FORBIDDEN_SUFFIXES = {'.npz', '.npy', '.ply', '.pcd', '.bag', '.jpg', '.jpeg', '.png',
                      '.mp4', '.mov', '.zip', '.log', '.key', '.pem', '.db'}
BINARY_ASSET = 'vendor/unitree_webrtc_connect/unitree_webrtc_connect/lidar/libvoxel.wasm'
PATTERNS = {
    'private key': re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
    'GitHub token': re.compile(r'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{50,})'),
    'AWS access key': re.compile(r'AKIA[0-9A-Z]{16}'),
    'home directory': re.compile(r'/(?:Users|home)/[A-Za-z0-9_.-]+/'),
    'literal device AES key': re.compile(r'''["']aes_128_key["']\s*:\s*["'][0-9a-fA-F]{32}["']'''),
}
WEB_FILES = {
    'outputs/go2-live-demo/index.html',
    'outputs/go2-live-demo/assets/photo-depth.js',
    'outputs/go2-live-demo/assets/panorama-view.js',
    'outputs/go2-live-demo/assets/spatial-photos.js',
    'outputs/go2-live-demo/assets/navigation-simulation.html',
    'outputs/go2-live-demo/assets/navigation-synthetic.json',
    'outputs/go2-live-demo/vendor/three.module.js',
    'outputs/go2-live-demo/vendor/OrbitControls.js',
    'outputs/go2-live-demo/vendor/THREE-LICENSE.txt',
}


def main():
    names = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0')
    failures, count = [], 0
    for name in filter(None, names):
        path = Path(name)
        count += 1
        if (path.suffix.lower() in FORBIDDEN_SUFFIXES or
            path.name in {'device.json', 'connection.json', '.env'} or
            path.name.startswith('HANDOFF') or
            any(part in {'.venv', '.python', '.tools', 'wheelhouse', 'fixtures', 'evidence'} for part in path.parts) or
            (name.startswith('outputs/') and name not in WEB_FILES)):
            failures.append((name, 'private data path'))
        data = subprocess.check_output(['git', 'show', ':'+name], cwd=ROOT)
        if name == BINARY_ASSET:
            if not data.startswith(b'\0asm'):
                failures.append((name, 'unexpected WASM format'))
            continue
        try:
            text = data.decode('utf-8')
        except UnicodeDecodeError:
            failures.append((name, 'unreviewed binary'))
            continue
        for rule, pattern in PATTERNS.items():
            if pattern.search(text):
                failures.append((name, rule))
        if ((name.startswith('work/') or name in WEB_FILES or name.endswith('.command'))
                and not name.startswith('outputs/go2-live-demo/vendor/')
                and re.search(r'[\u4e00-\u9fff]', text)):
            failures.append((name, 'non-English runtime text'))
    if failures:
        for name, rule in failures:
            print(f'{name}: {rule}')
        return 1
    print(f'Publication check passed: {count} staged source files, no forbidden data paths or matched credential patterns.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
