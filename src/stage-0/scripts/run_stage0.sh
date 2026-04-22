#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_DIR="$(cd "${ROOT_DIR}/.." && pwd)"

cd "${SRC_DIR}"
source .venv/bin/activate

cd "${ROOT_DIR}"
python scripts/discover_repos.py --max-search-pages 3 --per-page 40 --sleep-seconds 0.2
python scripts/clone_repos.py
python scripts/build_instances.py
python scripts/plot_contamination.py

echo "Stage 0 complete. Outputs are in ${ROOT_DIR}/outputs"
