#!/usr/bin/env bash
# run_all.sh — Run all LoRA recall diagnostics and the main experiment.
#
# Diagnostics (run in order of likelihood):
#   1. Checkpoint validation: run doc-to-lora's own eval on the checkpoint
#   2. Context window check:  token-length analysis + tiny-doc probe
#   3. Synthetic canary test: fictitious doc with zero pretraining signal
#   4. Reset verification:    confirm model.reset() fully clears LoRA
#   5. Main experiment:       the full recall probe suite (run_experiment.py)
#
# Usage:
#   cd experiments/lora-recall
#   bash run_all.sh           # run everything
#   bash run_all.sh 2 3       # run only diagnostics 2 and 3
#   bash run_all.sh 5         # run only the main experiment

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RESULTS_DIR="$SCRIPT_DIR/results"
mkdir -p "$RESULTS_DIR"

TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_FILE="$RESULTS_DIR/run_all_${TIMESTAMP}.log"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

log() {
    local msg="[$(date -u +%H:%M:%S)] $*"
    echo -e "$msg" | tee -a "$LOG_FILE"
}

run_step() {
    local step_num="$1"
    local step_name="$2"
    local script="$3"

    log ""
    log "${CYAN}══════════════════════════════════════════════════════════════════${NC}"
    log "${CYAN}  Step ${step_num}: ${step_name}${NC}"
    log "${CYAN}══════════════════════════════════════════════════════════════════${NC}"

    if [[ ! -f "$script" ]]; then
        log "${RED}  ERROR: Script not found: ${script}${NC}"
        return 1
    fi

    local step_start
    step_start=$(date +%s)

    if uv run python "$script" 2>&1 | tee -a "$LOG_FILE"; then
        local step_end
        step_end=$(date +%s)
        local elapsed=$(( step_end - step_start ))
        log "${GREEN}  ✓ Step ${step_num} completed (${elapsed}s)${NC}"
        return 0
    else
        local rc=$?
        local step_end
        step_end=$(date +%s)
        local elapsed=$(( step_end - step_start ))
        log "${RED}  ✗ Step ${step_num} FAILED with exit code ${rc} (${elapsed}s)${NC}"
        return $rc
    fi
}

# ---------------------------------------------------------------------------
# Determine which steps to run
# ---------------------------------------------------------------------------
ALL_STEPS=(1 2 3 4 5)

if [[ $# -gt 0 ]]; then
    STEPS=("$@")
else
    STEPS=("${ALL_STEPS[@]}")
fi

log "Starting run_all.sh at $(date -u)"
log "Steps to run: ${STEPS[*]}"
log "Results directory: $RESULTS_DIR"
log "Log file: $LOG_FILE"

overall_start=$(date +%s)
passed=0
failed=0
skipped=0

for step in "${STEPS[@]}"; do
    case "$step" in
        1)
            if run_step 1 "Checkpoint validation (vendor eval)" "diag_checkpoint_eval.py"; then
                ((passed+=1))
            else
                ((failed+=1))
                log "${YELLOW}  ⚠ Checkpoint eval failed — remaining diagnostics may still be useful.${NC}"
            fi
            ;;
        2)
            if run_step 2 "Context window & token length check" "diag_context_window.py"; then
                ((passed+=1))
            else
                ((failed+=1))
            fi
            ;;
        3)
            if run_step 3 "Synthetic canary test" "diag_canary.py"; then
                ((passed+=1))
            else
                ((failed+=1))
            fi
            ;;
        4)
            if run_step 4 "Reset verification" "diag_reset_verify.py"; then
                ((passed+=1))
            else
                ((failed+=1))
            fi
            ;;
        5)
            if run_step 5 "Main recall experiment" "run_experiment.py"; then
                ((passed+=1))
            else
                ((failed+=1))
            fi
            ;;
        *)
            log "${YELLOW}  Unknown step: ${step} (valid: 1-5)${NC}"
            ((skipped+=1))
            ;;
    esac
done

overall_end=$(date +%s)
overall_elapsed=$(( overall_end - overall_start ))

log ""
log "${CYAN}══════════════════════════════════════════════════════════════════${NC}"
log "${CYAN}  SUMMARY${NC}"
log "${CYAN}══════════════════════════════════════════════════════════════════${NC}"
log "  Passed:  ${passed}"
log "  Failed:  ${failed}"
log "  Skipped: ${skipped}"
log "  Total time: ${overall_elapsed}s"
log ""
log "  Results in: ${RESULTS_DIR}/"
log "  Log file:   ${LOG_FILE}"
log ""

# List all result files from this run
log "  Result files:"
find "$RESULTS_DIR" -maxdepth 1 -name "*.json" -o -name "*.csv" -o -name "*.txt" | sort | while read -r f; do
    log "    $(basename "$f")"
done

if [[ $failed -gt 0 ]]; then
    log "${RED}  Some steps failed — review the log above.${NC}"
    exit 1
fi
