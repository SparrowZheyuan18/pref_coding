#!/usr/bin/env bash
# One-shot installer for study participants.
#   git clone <this repo> && cd pref_tool && ./install.sh
set -euo pipefail

cd "$(dirname "$0")"

have_py311() {
  for c in python3.13 python3.12 python3.11 python3; do
    if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(sys.version_info < (3,11))' 2>/dev/null; then
      echo "$c"; return 0
    fi
  done
  return 1
}

# macOS ships Python 3.9, so most participants land here. Rather than telling
# them to go install something, offer to do it.
if ! command -v uv >/dev/null 2>&1 && ! have_py311 >/dev/null; then
  echo "This tool needs Python 3.11+, which was not found."
  echo "The easiest fix is uv (https://docs.astral.sh/uv/), a single binary."
  echo
  REPLY=""
  if [ -t 0 ]; then
    printf "Install uv now? [Y/n] "
    read -r REPLY || REPLY="y"
  else
    REPLY="y"
  fi
  case "${REPLY:-y}" in
    [nN]*) echo "Aborted. Install Python 3.11+ or uv, then re-run ./install.sh"; exit 1 ;;
    *)     curl -LsSf https://astral.sh/uv/install.sh | sh
           for d in "$HOME/.local/bin" "$HOME/.cargo/bin"; do
             [ -x "$d/uv" ] && PATH="$d:$PATH"
           done
           export PATH ;;
  esac
fi

if command -v uv >/dev/null 2>&1; then
  # Re-running the installer must be safe: participants will do it.
  [ -x .venv/bin/python ] || uv venv --python 3.11 .venv
  uv pip install --python .venv/bin/python -e .
else
  PY="$(have_py311)" || {
    echo "need Python 3.11+ (or install uv: https://docs.astral.sh/uv/)" >&2
    exit 1
  }
  [ -x .venv/bin/python ] || "$PY" -m venv .venv
  .venv/bin/python -m pip install -q --upgrade pip
  .venv/bin/python -m pip install -q -e .
fi

BIN="$(cd .venv/bin && pwd)"

# Put preftool on PATH permanently. A participant should not have to remember
# an `export` line every time they open a terminal.
add_to_path() {
  local rc="$1"
  [ -n "$rc" ] || return 1
  if [ -f "$rc" ] && grep -q "preftool PATH" "$rc"; then
    echo "already in $rc"
    return 0
  fi
  {
    echo ""
    echo "# preftool PATH (added by install.sh)"
    echo "export PATH=\"$BIN:\$PATH\""
  } >> "$rc"
  echo "added to $rc"
}

case "${SHELL:-}" in
  */zsh)  RC="$HOME/.zshrc" ;;
  */bash) RC="$HOME/.bash_profile"; [ -f "$HOME/.bashrc" ] && RC="$HOME/.bashrc" ;;
  *)      RC="" ;;
esac

if [ "${1:-}" = "--no-path" ] || [ -z "$RC" ]; then
  echo
  echo "Add preftool to your PATH:"
  echo
  echo "    export PATH=\"$BIN:\$PATH\""
else
  echo
  echo -n "PATH:            "
  add_to_path "$RC"
  export PATH="$BIN:$PATH"
fi

echo
echo "Open a new terminal (or run: source ${RC:-your shell config}), then check:"
echo
echo "    preftool --help"
echo
echo "Then, inside the repo you will be working in:"
echo
echo "    preftool start <your-participant-id>"
echo
