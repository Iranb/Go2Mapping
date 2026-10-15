#!/bin/zsh
set -eu
cd "${0:A:h}"
if [[ ! -x .venv/bin/python ]]; then
  print 'Run setup.command to install dependencies first.'
  exit 1
fi
print 'Web interface: http://127.0.0.1:63541/'
print 'Use the web interface after startup. Press Ctrl+C here to quit.'
GO2_IP="$(.venv/bin/python -c 'import json; from pathlib import Path; p=Path("connection.json"); print(json.loads(p.read_text()).get("ip", "auto") if p.is_file() else "auto")')"
exec .venv/bin/python -u work/go2_live_demo.py --ip "$GO2_IP" "$@"
