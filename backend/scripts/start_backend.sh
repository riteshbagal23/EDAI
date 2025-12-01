#!/usr/bin/env bash
# Simple helper to activate venv and start the backend with uvicorn
# Usage: ./start_backend.sh
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
if [ -d "venv" ]; then
  echo "Activating venv..."
  # shellcheck disable=SC1091
  source venv/bin/activate
else
  echo "No venv found in $ROOT_DIR/venv. Create one with: python3.11 -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
  exit 1
fi
# Start uvicorn
echo "Starting backend (uvicorn server:app) on 127.0.0.1:8000"
uvicorn server:app --host 127.0.0.1 --port 8000 --reload
