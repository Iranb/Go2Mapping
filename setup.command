#!/bin/zsh
set -eu
cd "${0:A:h}"

GO2_BREW="$(command -v brew || true)"
if [[ -z "$GO2_BREW" ]]; then
  for candidate in /opt/homebrew/bin/brew /usr/local/bin/brew; do
    if [[ -x "$candidate" ]]; then GO2_BREW="$candidate"; break; fi
  done
fi
if [[ -z "$GO2_BREW" ]]; then
  print 'Install Homebrew from https://brew.sh/ and run setup.command again.'
  exit 1
fi
export HOMEBREW_NO_AUTO_UPDATE=1
export HOMEBREW_NO_INSTALL_CLEANUP=1
GO2_AUDIO="$("$GO2_BREW" --prefix portaudio 2>/dev/null || true)"
if [[ ! -f "$GO2_AUDIO/lib/libportaudio.dylib" ]]; then
  "$GO2_BREW" install portaudio
  GO2_AUDIO="$("$GO2_BREW" --prefix portaudio)"
fi
mkdir -p .tools
if [[ ! -x .tools/uv ]]; then
  /usr/bin/curl -fLsS --connect-timeout 10 --max-time 120 https://astral.sh/uv/install.sh -o .tools/uv-install.sh
  UV_UNMANAGED_INSTALL="$PWD/.tools" /bin/sh .tools/uv-install.sh
fi
export UV_PYTHON_INSTALL_DIR="$PWD/.python"
export UV_PYTHON_BIN_DIR="$PWD/.tools"
.tools/uv python install 3.12
GO2_PYTHON="$(.tools/uv python find --managed-python --system 3.12)"
"$GO2_PYTHON" -m venv --clear .venv
export CPPFLAGS="-I$GO2_AUDIO/include ${CPPFLAGS:-}"
export LDFLAGS="-L$GO2_AUDIO/lib ${LDFLAGS:-}"
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip check
.venv/bin/python work/verify_install.py
.venv/bin/python -m pip wheel --wheel-dir wheelhouse -r requirements.txt
print 'Setup complete. Configure your device, then double-click start.command.'
