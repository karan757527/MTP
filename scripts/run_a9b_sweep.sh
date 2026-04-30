#!/usr/bin/env bash
# run_a9b_sweep.sh — 5-seed A9b sweep on GPU 1.
#
# A9b = Approach B = additive edge-feature injection (Linear(95, H) +
# scatter_add by destination, added to adj@x pre-norm). Param-matched
# against A9's scalar-gate approach, so any PR-AUC delta is attributable
# to integration style, not capacity.
#
# Summary file: logs/A9b_runner_summary.log (separate from A9's).
#
# Usage (GPU 1, detached):
#   nohup env CUDA_VISIBLE_DEVICES=1 bash scripts/run_a9b_sweep.sh \
#       >> logs/A9b_runner.log 2>&1 &
#   disown

trap "" HUP
set -u

NSEEDS="${NSEEDS:-5}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
mkdir -p logs

SUMMARY=logs/A9b_runner_summary.log

{
echo "============================================================"
echo "=== run_a9b_sweep.sh started $(date)"
echo "=== NSEEDS=$NSEEDS"
echo "=== CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "=== EXPERIMENTS: A9b"
echo "=== PID: $$"
echo "============================================================"
} | tee -a "$SUMMARY"

OVERALL_START=$(date +%s)

exp=A9b
cfg="config/ablation/${exp}.yml"
if [[ ! -f "$cfg" ]]; then
    echo "[ERROR] $(date) missing config $cfg" | tee -a "$SUMMARY"
    exit 1
fi

{
echo ""
echo "------------------------------------------------------------"
echo "[CELL_START] $(date) exp=$exp cfg=$cfg seeds=0..$((NSEEDS-1))"
echo "------------------------------------------------------------"
} | tee -a "$SUMMARY"

cell_start=$(date +%s)

for ((seed=0; seed<NSEEDS; seed++)); do
    log_dir="logs/${exp}_seed${seed}"
    mkdir -p "$log_dir"

    echo "[SEED_START] $(date) exp=$exp seed=$seed log=$log_dir" \
        | tee -a "$SUMMARY"
    seed_start=$(date +%s)

    python -u run_ablation_elliptic2.py \
        --config "$cfg" \
        --log_dir "$log_dir" \
        --seed_override "$seed" \
        > "$log_dir/stdout.log" 2>&1
    rc=$?

    seed_end=$(date +%s)
    dur=$((seed_end - seed_start))

    if [[ $rc -eq 0 ]]; then
        echo "[SEED_DONE] $(date) exp=$exp seed=$seed rc=0 dur=${dur}s" \
            | tee -a "$SUMMARY"
        if [[ -f "$log_dir/train.jsonl" ]]; then
            tail -n 1 "$log_dir/train.jsonl" 2>/dev/null \
                | tee -a "$SUMMARY" || true
        fi
    else
        echo "[SEED_FAIL] $(date) exp=$exp seed=$seed rc=$rc dur=${dur}s — see $log_dir/stdout.log" \
            | tee -a "$SUMMARY"
    fi
done

cell_end=$(date +%s)
cell_dur=$((cell_end - cell_start))
echo "[CELL_AGG] $(date) exp=$exp total_dur=${cell_dur}s" | tee -a "$SUMMARY"

python -u scripts/aggregate_seeds.py \
    --exp "$exp" \
    --nseeds "$NSEEDS" \
    --out "logs/${exp}_summary.json" 2>&1 \
    | tee -a "$SUMMARY" \
    || echo "[AGG_FAIL] $(date) exp=$exp aggregator crashed" \
    | tee -a "$SUMMARY"

OVERALL_END=$(date +%s)
OVERALL_DUR=$((OVERALL_END - OVERALL_START))

{
echo ""
echo "============================================================"
echo "=== run_a9b_sweep.sh finished $(date)"
echo "=== total wall time: ${OVERALL_DUR}s"
echo "============================================================"
} | tee -a "$SUMMARY"

echo "=== A9b sweep done. summary in logs/A9b_summary.json" | tee -a "$SUMMARY"
