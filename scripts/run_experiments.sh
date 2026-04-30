#!/usr/bin/env bash
# run_experiments.sh — run one or more Elliptic2 ablation cells serially.
#
# Usage:
#   bash scripts/run_experiments.sh A8 A8b A5 A7 A6 A1
#
# Each arg must match a YAML file under config/ablation/<NAME>.yml. The
# runner creates logs/<NAME>/ with stdout.log (full console) and
# train.jsonl (per-epoch metrics). Failures in one cell do NOT stop the
# rest — we print the status and move on so an overnight run finishes
# as much as possible.
#
# Background / SSH-safe launch:
#   nohup bash scripts/run_experiments.sh A8 A8b A5 A7 A6 A1 \
#       > logs/runner.log 2>&1 &
#   # then `disown` if your shell requires it; nohup alone is fine on bash.
#
# Environment:
#   CUDA_VISIBLE_DEVICES   — defaults to 0 if unset.
set -u

# Resolve repo root as the parent of this script's directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

if [[ $# -eq 0 ]]; then
    echo "Usage: $0 <EXP> [EXP ...]"
    echo "Example: $0 A8 A8b A5 A7 A6 A1"
    exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

summary_file="logs/runner_summary.log"
mkdir -p logs
echo "=== run_experiments.sh started $(date) ===" | tee -a "$summary_file"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" | tee -a "$summary_file"
echo "experiments: $*" | tee -a "$summary_file"

for exp in "$@"; do
    cfg="config/ablation/${exp}.yml"
    log_dir="logs/${exp}"

    if [[ ! -f "$cfg" ]]; then
        echo "[ERROR] $(date) missing config $cfg — skipping" | tee -a "$summary_file"
        continue
    fi

    mkdir -p "$log_dir"
    echo "---" | tee -a "$summary_file"
    echo "[START] $(date) exp=$exp cfg=$cfg" | tee -a "$summary_file"

    start_ts=$(date +%s)
    python run_ablation_elliptic2.py \
        --config "$cfg" \
        --log_dir "$log_dir" \
        > "$log_dir/stdout.log" 2>&1
    rc=$?
    end_ts=$(date +%s)
    dur=$((end_ts - start_ts))

    if [[ $rc -eq 0 ]]; then
        echo "[DONE]  $(date) exp=$exp rc=$rc duration=${dur}s" | tee -a "$summary_file"
        # Extract the last final record from the JSONL for a quick summary.
        if [[ -f "$log_dir/train.jsonl" ]]; then
            tail -n 1 "$log_dir/train.jsonl" | tee -a "$summary_file"
        fi
    else
        echo "[FAIL]  $(date) exp=$exp rc=$rc duration=${dur}s — see $log_dir/stdout.log" | tee -a "$summary_file"
    fi
done

echo "=== run_experiments.sh finished $(date) ===" | tee -a "$summary_file"
