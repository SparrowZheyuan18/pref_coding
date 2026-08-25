#!/usr/bin/env bash
# One-shot installer for study participants.
#   git clone <this repo> && cd pref_tool && ./install.sh
set -euo pipefail

cd "$(dirname "$0")"

if command -v uv >/dev/null 2>&1; then
  uv venv --python 3.11 .venv
  uv pip install --python .venv/bin/python -e .
else
  PY=""
  for c in python3.13 python3.12 python3.11 python3; do
    if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(sys.version_info < (3,11))'; then
      PY="$c"; break
    fi
  done
  if [ -z "$PY" ]; then
    echo "need Python 3.11+ (or install uv: https://docs.astral.sh/uv/)" >&2
    exit 1
  fi
  "$PY" -m venv .venv
  .venv/bin/python -m pip install -q --upgrade pip
  .venv/bin/python -m pip install -q -e .
fi

BIN="$(cd .venv/bin && pwd)"
echo
echo "Installed. Add preftool to your PATH for this terminal:"
echo
echo "    export PATH=\"$BIN:\$PATH\""
echo
echo "Then, inside the repo you will be working in:"
echo
echo "    preftool start <your-participant-id>"
echo
