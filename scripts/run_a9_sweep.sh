#!/usr/bin/env bash
# run_a9_sweep.sh — 5-seed A9 sweep on GPU 1.
#
# Mirrors run_all_experiments.sh (same SEED_START / SEED_DONE / CELL_AGG
# markers, same aggregator invocation) so a downstream viewer can't tell
# the two sweeps apart in format. Differences:
#   * Only cell A9.
#   * Writes to its OWN summary file (logs/A9_runner_summary.log) so it
#     does not interleave with the main GPU-0 sweep summary
#     (logs/runner_summary.log).
#   * seed 0 was already completed under logs/A9_seed0 before this script
#     was written — we skip it and run seeds 1..4 only, but include seed 0
#     in the aggregator pass so the final logs/A9_summary.json spans all 5.
#
# Usage (GPU 1, detached):
#   nohup env CUDA_VISIBLE_DEVICES=1 bash scripts/run_a9_sweep.sh \
#       >> logs/A9_runner.log 2>&1 &
#   disown

trap "" HUP
set -u

NSEEDS="${NSEEDS:-5}"
SEED_START_INDEX="${SEED_START_INDEX:-1}"   # skip seed 0 — already done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
mkdir -p logs

SUMMARY=logs/A9_runner_summary.log

{
echo "============================================================"
echo "=== run_a9_sweep.sh started $(date)"
echo "=== NSEEDS=$NSEEDS  SEED_START_INDEX=$SEED_START_INDEX"
echo "=== CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "=== EXPERIMENTS: A9"
echo "=== PID: $$"
echo "============================================================"
} | tee -a "$SUMMARY"

OVERALL_START=$(date +%s)

exp=A9
cfg="config/ablation/${exp}.yml"
if [[ ! -f "$cfg" ]]; then
    echo "[ERROR] $(date) missing config $cfg" | tee -a "$SUMMARY"
    exit 1
fi

{
echo ""
echo "------------------------------------------------------------"
echo "[CELL_START] $(date) exp=$exp cfg=$cfg seeds=${SEED_START_INDEX}..$((NSEEDS-1))"
echo "------------------------------------------------------------"
} | tee -a "$SUMMARY"

cell_start=$(date +%s)

for ((seed=SEED_START_INDEX; seed<NSEEDS; seed++)); do
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

# Aggregate over ALL 5 seeds (0..4). seed 0 lives in logs/A9_seed0 from the
# earlier standalone run, so the aggregator finds the full set.
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
echo "=== run_a9_sweep.sh finished $(date)"
echo "=== total wall time: ${OVERALL_DUR}s"
echo "============================================================"
} | tee -a "$SUMMARY"

echo "=== A9 sweep done. summary in logs/A9_summary.json" | tee -a "$SUMMARY"
