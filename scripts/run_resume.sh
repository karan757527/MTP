#!/usr/bin/env bash
# run_resume.sh — trimmed resume sweep after the main runner is stopped.
#
# Why this exists
# ---------------
# The original scripts/run_all_experiments.sh was running 5 seeds × 8 cells
# with each cell's full max_epochs=500 / patience=20 budget. For the three
# slow cells (A2, A1, A6) that worked out to ~8–9 days of wall time, outside
# the thesis timeline. On 2026-04-17 we trimmed those three configs
# (max_epochs=80, A2 patience 20→10) and cut their seed count to 3 (plenty
# for mean±std). The fast cells (A4, A8b, A7, A8, A5) already have 5 seeds
# each under their original configs and are untouched.
#
# This script resumes only what's left:
#   A2: seeds 1, 2   (seed 0 already completed under the original config)
#   A1: seeds 0, 1, 2
#   A6: seeds 0, 1, 2
#
# Appends to the SAME logs/runner_summary.log the original runner uses, so
# grepping for best_tst_prauc across the full sweep still works.
#
# Usage:
#   nohup bash scripts/run_resume.sh >> logs/runner.log 2>&1 &
#   disown
#
# Tunables:
#   CUDA_VISIBLE_DEVICES  — GPU index (default 0; inherit the original slot)

trap "" HUP
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
mkdir -p logs

SUMMARY=logs/runner_summary.log

{
echo ""
echo "============================================================"
echo "=== run_resume.sh started $(date)"
echo "=== CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "=== plan: A2 seeds 1-2, A1 seeds 0-2, A6 seeds 0-2 (trimmed configs)"
echo "=== PID: $$"
echo "============================================================"
} | tee -a "$SUMMARY"

OVERALL_START=$(date +%s)

# "exp seed_lo seed_hi" triples.
JOBS=(
    "A2 1 2"
    "A1 0 2"
    "A6 0 2"
)

for job in "${JOBS[@]}"; do
    read -r exp lo hi <<< "$job"
    cfg="config/ablation/${exp}.yml"
    if [[ ! -f "$cfg" ]]; then
        echo "[ERROR] $(date) missing config $cfg — skipping cell" \
            | tee -a "$SUMMARY"
        continue
    fi

    {
    echo ""
    echo "------------------------------------------------------------"
    echo "[CELL_START] $(date) exp=$exp cfg=$cfg seeds=${lo}..${hi} (resume)"
    echo "------------------------------------------------------------"
    } | tee -a "$SUMMARY"

    cell_start=$(date +%s)

    for ((seed=lo; seed<=hi; seed++)); do
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

    # nseeds=3 reflects what we actually aggregate over for these cells:
    #   A2 = 1 existing seed0 (original cfg) + 2 trimmed seeds = 3
    #   A1, A6 = 3 trimmed seeds
    # Fast cells already have their own _summary.json at nseeds=5.
    python -u scripts/aggregate_seeds.py \
        --exp "$exp" \
        --nseeds 3 \
        --out "logs/${exp}_summary.json" 2>&1 \
        | tee -a "$SUMMARY" \
        || echo "[AGG_FAIL] $(date) exp=$exp aggregator crashed" \
        | tee -a "$SUMMARY"
done

OVERALL_END=$(date +%s)
OVERALL_DUR=$((OVERALL_END - OVERALL_START))

{
echo ""
echo "============================================================"
echo "=== run_resume.sh finished $(date)"
echo "=== total wall time: ${OVERALL_DUR}s"
echo "============================================================"
} | tee -a "$SUMMARY"

# Rebuild the cross-cell rollup against the updated per-cell summaries.
python -u scripts/aggregate_seeds.py --all --nseeds 5 \
    --out logs/all_summary.json 2>&1 \
    | tee -a "$SUMMARY" \
    || echo "[ROLLUP_FAIL] $(date) cross-cell rollup failed" | tee -a "$SUMMARY"

echo "=== resume done. per-cell summaries in logs/<EXP>_summary.json" \
    | tee -a "$SUMMARY"
