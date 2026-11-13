#!/usr/bin/env bash
# SKY-IR launcher.
#   ./run.sh          build the UI (if needed) and serve everything on http://127.0.0.1:8000
#   ./run.sh dev      backend on :8000 + Vite dev server with hot reload on :5173
#   ./run.sh test     backend unit tests
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-.venv/bin/python}
[ -x "$PY" ] || PY=python3

case "${1:-serve}" in
  dev)
    (cd frontend && [ -d node_modules ] || npm install)
    (cd backend && exec "../$PY" -m skyir --port 8000) &
    trap 'kill $!' EXIT
    cd frontend && npm run dev
    ;;
  test)
    cd backend && exec "../$PY" -m pytest -q tests
    ;;
  serve|*)
    if [ ! -d frontend/dist ] || [ -n "$(find frontend/src -newer frontend/dist -print -quit)" ]; then
      (cd frontend && { [ -d node_modules ] || npm install; } && npm run build)
    fi
    cd backend && exec "../$PY" -m skyir --host "${HOST:-127.0.0.1}" --port "${PORT:-8000}"
    ;;
esac
