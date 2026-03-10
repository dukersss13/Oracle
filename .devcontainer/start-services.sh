#!/usr/bin/env bash
set -euo pipefail

cd /workspaces/Oracle

if ! pgrep -f "uvicorn api.app:app" >/dev/null 2>&1; then
  nohup python -m uvicorn api.app:app --host 0.0.0.0 --port 8000 > /tmp/oracle-api.log 2>&1 &
fi

if [ -f "ui/package.json" ]; then
  cd /workspaces/Oracle/ui
  if ! pgrep -f "vite --host 0.0.0.0 --port 5173" >/dev/null 2>&1; then
    nohup npm run dev -- --host 0.0.0.0 --port 5173 > /tmp/oracle-ui.log 2>&1 &
  fi
fi
