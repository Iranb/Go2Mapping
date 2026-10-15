#!/bin/zsh
set -eu
cd "${0:A:h}"
if [[ ! -x .tools/uv || ! -d .python || ! -d wheelhouse ]]; then
  print 'Run setup.command online first to prepare Python, PortAudio and the wheelhouse.'
  exit 1
fi
export UV_PYTHON_INSTALL_DIR="$PWD/.python"
export UV_OFFLINE=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
GO2_PYTHON="$(.tools/uv python find --managed-python --system 3.12)"
"$GO2_PYTHON" -m venv --clear .venv
.venv/bin/python -m pip install --no-index --find-links wheelhouse -r requirements-offline.txt
.venv/bin/python -m pip check
.venv/bin/python work/verify_install.py
print 'Offline installation complete.'
