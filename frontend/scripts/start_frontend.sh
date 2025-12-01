#!/usr/bin/env bash
# Simple helper to start the frontend dev server
# Usage: ./scripts/start_frontend.sh
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
if [ -f "package.json" ]; then
  echo "Installing frontend deps (if needed) and starting dev server..."
  # prefer yarn if available
  if command -v yarn >/dev/null 2>&1; then
    yarn install --silent || true
    yarn start
  else
    npm install --silent || true
    npm start
  fi
else
  echo "package.json not found in frontend directory"
  exit 1
fi
