#!/usr/bin/env bash
set -euo pipefail

# End-to-end easy evaluation over all five artifact repos:
# - GPU 6: precompute embedding-router JSONLs.
# - GPU 4: oracle-routed predictions/eval.
# - GPU 5: embedding-routed predictions/eval.
#
# By default this runs only local model predictions. To run the full
# predict -> judge -> report path, set EVAL_SUBCOMMAND=all ALLOW_API_JUDGE=1.

cd "$(dirname "$0")/.."

ORACLE_CONFIG="${ORACLE_CONFIG:-config/eval/all_repos_easy_oracle.yaml}"
EMBEDDING_CONFIG="${EMBEDDING_CONFIG:-config/eval/all_repos_easy_embedding_top1.yaml}"
EVAL_SUBCOMMAND="${EVAL_SUBCOMMAND:-predict}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-Qwen/Qwen3-Embedding-0.6B}"
ROUTING_VISIBLE_GPU="${ROUTING_VISIBLE_GPU:-6}"
FORCE_ROUTING="${FORCE_ROUTING:-0}"
REQUIRE_LORAS="${REQUIRE_LORAS:-1}"

ARTIFACT_ROOTS=(
  artifacts/antirez__kilo
  artifacts/fogleman__Craft
  artifacts/marimo-team__marimo
  artifacts/psf__requests
  artifacts/pytest-dev__pytest
)

case "$EVAL_SUBCOMMAND" in
  all|predict) ;;
  *)
    echo "EVAL_SUBCOMMAND must be either 'all' or 'predict'." >&2
    exit 2
    ;;
esac

if [[ "$EVAL_SUBCOMMAND" == "all" && "${ALLOW_API_JUDGE:-0}" != "1" ]]; then
  echo "Refusing to run OpenAI judge without ALLOW_API_JUDGE=1." >&2
  echo "Use EVAL_SUBCOMMAND=predict for local predictions only." >&2
  exit 2
fi

if [[ "$REQUIRE_LORAS" == "1" ]]; then
  uv run python - "${ARTIFACT_ROOTS[@]}" <<'PY'
import json
import sys
from pathlib import Path

missing = []
for raw_root in sys.argv[1:]:
    root = Path(raw_root)
    ledger_path = root / "ledger.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    documents = ledger.get("documents") or {}
    for document_id, entry in documents.items():
        files = entry.get("files") or {}
        lora_rel = files.get("lora")
        if not lora_rel:
            missing.append(f"{root}:{document_id} has no files.lora")
            continue
        lora_path = root / lora_rel
        if not lora_path.exists():
            missing.append(f"{root}:{document_id} files.lora is missing on disk: {lora_rel}")

if missing:
    print("Cannot run a complete shine evaluation until these LoRAs exist:", file=sys.stderr)
    for item in missing:
        print(f"  - {item}", file=sys.stderr)
    print("Set REQUIRE_LORAS=0 only if you intentionally want partial shine coverage.", file=sys.stderr)
    sys.exit(1)
PY
fi

tmp_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_dir"' EXIT

for artifact_root in "${ARTIFACT_ROOTS[@]}"; do
  routing_output="$artifact_root/easy/routing_results.qwen3.all.top1.jsonl"
  if [[ "$FORCE_ROUTING" != "1" && -s "$routing_output" ]]; then
    echo "Using existing routing results: $routing_output"
    continue
  fi

  echo "Generating embedding routing results: $routing_output"
  : > "$routing_output"
  for qa_path in "$artifact_root"/easy/qas/*.json; do
    tmp_output="$tmp_dir/$(basename "$artifact_root").$(basename "$qa_path" .json).jsonl"
    CUDA_VISIBLE_DEVICES="$ROUTING_VISIBLE_GPU" uv run python scripts/embedding_router.py \
      --lora-store "$artifact_root/ledger.json" \
      --qa-pairs "$qa_path" \
      --output "$tmp_output" \
      --embedding-model "$EMBEDDING_MODEL" \
      --top-k 1 \
      --device cuda
    cat "$tmp_output" >> "$routing_output"
  done
done

echo "Starting oracle-routed evaluation on cuda:4"
uv run python scripts/run_eval.py "$EVAL_SUBCOMMAND" --config "$ORACLE_CONFIG" &
oracle_pid=$!

echo "Starting embedding-routed evaluation on cuda:5"
uv run python scripts/run_eval.py "$EVAL_SUBCOMMAND" --config "$EMBEDDING_CONFIG" &
embedding_pid=$!

status=0
wait "$oracle_pid" || status=1
wait "$embedding_pid" || status=1
exit "$status"
