#!/usr/bin/env bash
# run_all.sh — compatibility wrapper for the Python orchestrator.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

exec uv run python run_all.py "$@"
