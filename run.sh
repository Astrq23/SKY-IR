#!/usr/bin/env bash
# SKY-IR: one command to set up and run everything.
#
#   ./run.sh            install whatever is missing, build the UI, serve on http://127.0.0.1:8000
#   ./run.sh dev        same setup, then backend :8000 + Vite hot-reload UI on :5173
#   ./run.sh test       backend unit tests
#   ./run.sh setup      only install / update dependencies, don't start the server
#   ./run.sh help       show this help
#
# Environment variables:
#   HOST=0.0.0.0        listen on all interfaces (LAN access)      default 127.0.0.1
#   PORT=9000           server port                                default 8000
#   TORCH_INDEX=cpu     force a PyTorch build: cu128, cu126, cpu   default: cu128 if an NVIDIA GPU is found
#   PYTHON=python3.11   interpreter used to create .venv           default: newest of 3.12 / 3.11 / 3.10
#
# Every setup step is skipped when it is already up to date, so repeat runs start immediately.
set -euo pipefail
cd "$(dirname "$0")"
ROOT=$(pwd)
VENV="$ROOT/.venv"
PY="$VENV/bin/python"
STAMP_DIR="$VENV/.skyir"

c_info=$'\e[36m'; c_ok=$'\e[32m'; c_warn=$'\e[33m'; c_err=$'\e[31m'; c_off=$'\e[0m'
[ -t 1 ] || { c_info=; c_ok=; c_warn=; c_err=; c_off=; }
step() { echo "${c_info}==>${c_off} $*"; }
ok()   { echo "${c_ok}  ✓${c_off} $*"; }
warn() { echo "${c_warn}  !${c_off} $*"; }
die()  { echo "${c_err}  ✗ $*${c_off}" >&2; exit 1; }

fingerprint() { cat "$@" 2>/dev/null | sha256sum | cut -c1-16; }

# ------------------------------------------------------------------ python
find_python() {
  if [ -n "${PYTHON:-}" ]; then command -v "$PYTHON" && return; die "PYTHON=$PYTHON not found"; fi
  for v in python3.12 python3.11 python3.10; do command -v "$v" >/dev/null && { command -v "$v"; return; }; done
  if command -v python3 >/dev/null; then
    local mv; mv=$(python3 -c 'import sys; print(sys.version_info[:2] >= (3, 10) and sys.version_info[:2] <= (3, 12))')
    [ "$mv" = True ] && { command -v python3; return; }
  fi
  return 1
}

setup_python() {
  step "Python environment"
  if [ ! -x "$PY" ]; then
    if command -v uv >/dev/null; then
      uv venv -q --python 3.12 "$VENV" || die "uv could not create a Python 3.12 environment"
    else
      local base; base=$(find_python) || die "need Python 3.10-3.12 (PyTorch does not support newer versions yet). Install it or uv (https://docs.astral.sh/uv/)."
      "$base" -m venv "$VENV" || die "could not create .venv (on Debian/Ubuntu: sudo apt install python3-venv)"
    fi
    ok "created .venv ($("$PY" --version))"
  fi
  mkdir -p "$STAMP_DIR"
  local pip=("$PY" -m pip install -q)
  if command -v uv >/dev/null; then pip=(uv pip install -q --python "$PY"); else "$PY" -m pip install -q --upgrade pip; fi

  local index=${TORCH_INDEX:-}
  if [ -z "$index" ]; then
    if command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1; then index=cu128; else index=cpu; fi
  fi
  if [ "$(cat "$STAMP_DIR/torch" 2>/dev/null)" != "$index" ] || ! "$PY" -c "import torch" 2>/dev/null; then
    step "Installing PyTorch ($index), this can take a few minutes the first time"
    "${pip[@]}" torch torchvision --index-url "https://download.pytorch.org/whl/$index"
    echo "$index" > "$STAMP_DIR/torch"
  fi
  ok "PyTorch $("$PY" -c 'import torch; print(torch.__version__, "· CUDA" if torch.cuda.is_available() else "· CPU only")')"

  local fp; fp=$(fingerprint backend/requirements.txt)
  if [ "$(cat "$STAMP_DIR/requirements" 2>/dev/null)" != "$fp" ]; then
    step "Installing Python packages"
    if [ "$index" = cpu ]; then
      # TensorRT is GPU-only: skip it on CPU machines (the app falls back to PyTorch)
      grep -v '^tensorrt' backend/requirements.txt > "$STAMP_DIR/req-cpu.txt"
      "${pip[@]}" -r "$STAMP_DIR/req-cpu.txt"
    else
      "${pip[@]}" -r backend/requirements.txt || {
        warn "full install failed; retrying without TensorRT (PyTorch FP16 will be used)"
        grep -v '^tensorrt' backend/requirements.txt > "$STAMP_DIR/req-nott.txt"
        "${pip[@]}" -r "$STAMP_DIR/req-nott.txt"
      }
    fi
    echo "$fp" > "$STAMP_DIR/requirements"
  fi
  ok "Python packages up to date"
}

# ------------------------------------------------------------------ frontend
setup_frontend() {
  step "Web UI"
  command -v npm >/dev/null || die "need Node.js 18+ and npm (https://nodejs.org)"
  local major; major=$(node -p 'process.versions.node.split(".")[0]')
  [ "$major" -ge 18 ] || die "Node.js $major is too old, need 18+"
  local fp; fp=$(fingerprint frontend/package-lock.json)
  if [ ! -d frontend/node_modules ] || [ "$(cat frontend/node_modules/.skyir-lock 2>/dev/null)" != "$fp" ]; then
    (cd frontend && npm ci --no-audit --no-fund --loglevel=error)
    echo "$fp" > frontend/node_modules/.skyir-lock
  fi
  ok "node packages up to date"
}

build_frontend() {
  if [ ! -f frontend/dist/index.html ] || [ -n "$(find frontend/src frontend/index.html -newer frontend/dist/index.html -print -quit)" ]; then
    step "Building web UI"
    (cd frontend && npm run build --silent >/dev/null) || die "UI build failed: run 'cd frontend && npm run build' to see the errors"
  fi
  ok "web UI built"
}

# ------------------------------------------------------------------ model
check_model() {
  step "Model weights"
  local w; w=$(find models -maxdepth 1 -name '*.pt' -size +1k 2>/dev/null | head -1)
  [ -n "$w" ] || die "no model weights in models/. Download the .pt file from the link in README.md into models/ and run again."
  ok "$(basename "$w")"
  if [ -f "${w%.pt}.engine" ]; then ok "TensorRT engine ready"
  elif "$PY" -c "import tensorrt, torch; assert torch.cuda.is_available()" 2>/dev/null; then
    warn "first start will build a TensorRT engine for this GPU (2-3 minutes, once)"
  else
    warn "TensorRT not available: running on PyTorch (slower, still works)"
  fi
}

setup_all() { setup_python; setup_frontend; check_model; }

serve() {
  local host=${HOST:-127.0.0.1} port=${PORT:-8000}
  local shown=$host; [ "$host" = 0.0.0.0 ] && shown=$(hostname -I 2>/dev/null | awk '{print $1}')
  echo
  echo "${c_ok}SKY-IR starting on http://${shown:-$host}:$port${c_off}  (Ctrl+C to stop)"
  echo
  cd backend && exec "$PY" -m skyir --host "$host" --port "$port"
}

case "${1:-serve}" in
  serve | start | "")
    setup_all; build_frontend; serve ;;
  dev)
    setup_all
    (cd backend && exec "$PY" -m skyir --host "${HOST:-127.0.0.1}" --port "${PORT:-8000}") &
    trap 'kill $! 2>/dev/null' EXIT
    echo "${c_ok}Dev UI: http://localhost:5173  (API on :${PORT:-8000})${c_off}"
    cd frontend && SKYIR_BACKEND="http://127.0.0.1:${PORT:-8000}" npm run dev ;;
  test)
    setup_python
    "$PY" -c "import pytest" 2>/dev/null || { if command -v uv >/dev/null; then uv pip install -q --python "$PY" pytest; else "$PY" -m pip install -q pytest; fi; }
    cd backend && exec "$PY" -m pytest -q tests ;;
  setup)
    setup_all; build_frontend; echo; ok "setup complete. Start with: ./run.sh" ;;
  help | -h | --help)
    sed -n '2,17p' "$0" | sed 's/^# \{0,1\}//' ;;
  *)
    die "unknown command '$1' (try: ./run.sh help)" ;;
esac
