#!/usr/bin/env bash
# run_seed0_a9c_a10_a11.sh — seed 0 of three new cells, sequentially.
#
# Order (fast to slow):
#   1. A9c — A9b + normalized edges + presence mask  (~15 min)
#   2. A10 — A6 (2L) + additive edges                 (~80 min)
#   3. A11 — A2 (2L + GraphNorm + k=200) + edges      (~3-4 h)
#
# A9c needs the normalized edge_attr blob. If absent, this script first
# runs scripts/compute_edge_attr_norm.py (CPU, ~25-35 min, ~68 GB disk).
#
# Per-cell logs land in logs/<exp>_seed0/ exactly like the multi-seed
# runners (run_a9_sweep.sh, run_a9b_sweep.sh) so a later 5-seed sweep
# with SEED_START_INDEX=1 picks up seed 0 from the existing dir.
#
# Summary marker format matches the multi-seed runners:
#   [SEED_START] / [SEED_DONE] / [SEED_FAIL] / [CELL_AGG]
#
# Usage (GPU 1, detached):
#   nohup env CUDA_VISIBLE_DEVICES=1 bash scripts/run_seed0_a9c_a10_a11.sh \
#       >> logs/seed0_combo_runner.log 2>&1 &
#   disown

trap "" HUP
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
mkdir -p logs

SUMMARY=logs/seed0_combo_runner_summary.log
EXPS=(A9c A10 A11)
NORM_BLOB="dataset_/elliptic2/processed/elliptic2_k2_edge_attr_norm.pt"

{
echo "============================================================"
echo "=== run_seed0_a9c_a10_a11.sh started $(date)"
echo "=== CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "=== EXPERIMENTS: ${EXPS[*]} (seed 0 each)"
echo "=== PID: $$"
echo "============================================================"
} | tee -a "$SUMMARY"

OVERALL_START=$(date +%s)

# --- Step 0: build normalized edge_attr if missing (A9c needs it) ---
if [[ ! -f "$NORM_BLOB" ]]; then
    {
    echo ""
    echo "------------------------------------------------------------"
    echo "[PREP_START] $(date) building $NORM_BLOB via compute_edge_attr_norm.py"
    echo "------------------------------------------------------------"
    } | tee -a "$SUMMARY"
    prep_start=$(date +%s)
    python -u scripts/compute_edge_attr_norm.py \
        > logs/compute_edge_attr_norm.log 2>&1
    prep_rc=$?
    prep_dur=$(( $(date +%s) - prep_start ))
    if [[ $prep_rc -ne 0 ]]; then
        echo "[PREP_FAIL] $(date) compute_edge_attr_norm.py rc=$prep_rc dur=${prep_dur}s — see logs/compute_edge_attr_norm.log" \
            | tee -a "$SUMMARY"
        echo "[ABORT] cannot run A9c without normalized blob" | tee -a "$SUMMARY"
        exit 1
    fi
    echo "[PREP_DONE] $(date) rc=0 dur=${prep_dur}s" | tee -a "$SUMMARY"
else
    echo "[PREP_SKIP] $(date) $NORM_BLOB already exists" | tee -a "$SUMMARY"
fi

# --- Step 1..3: run seed 0 of each cell ---
for exp in "${EXPS[@]}"; do
    cfg="config/ablation/${exp}.yml"
    if [[ ! -f "$cfg" ]]; then
        echo "[ERROR] $(date) missing config $cfg" | tee -a "$SUMMARY"
        continue
    fi

    {
    echo ""
    echo "------------------------------------------------------------"
    echo "[CELL_START] $(date) exp=$exp cfg=$cfg seeds=0..0"
    echo "------------------------------------------------------------"
    } | tee -a "$SUMMARY"
    cell_start=$(date +%s)

    seed=0
    log_dir="logs/${exp}_seed${seed}"
    mkdir -p "$log_dir"

    echo "[SEED_START] $(date) exp=$exp seed=$seed log=$log_dir" | tee -a "$SUMMARY"
    seed_start=$(date +%s)

    python -u run_ablation_elliptic2.py \
        --config "$cfg" \
        --log_dir "$log_dir" \
        --seed_override "$seed" \
        > "$log_dir/stdout.log" 2>&1
    rc=$?

    seed_dur=$(( $(date +%s) - seed_start ))

    if [[ $rc -eq 0 ]]; then
        echo "[SEED_DONE] $(date) exp=$exp seed=$seed rc=0 dur=${seed_dur}s" \
            | tee -a "$SUMMARY"
        if [[ -f "$log_dir/train.jsonl" ]]; then
            tail -n 1 "$log_dir/train.jsonl" 2>/dev/null \
                | tee -a "$SUMMARY" || true
        fi
    else
        echo "[SEED_FAIL] $(date) exp=$exp seed=$seed rc=$rc dur=${seed_dur}s — see $log_dir/stdout.log" \
            | tee -a "$SUMMARY"
    fi

    cell_dur=$(( $(date +%s) - cell_start ))
    echo "[CELL_AGG] $(date) exp=$exp total_dur=${cell_dur}s (n=1, aggregator skipped — re-run scripts/aggregate_seeds.py after multi-seed sweep)" \
        | tee -a "$SUMMARY"
done

OVERALL_DUR=$(( $(date +%s) - OVERALL_START ))

{
echo ""
echo "============================================================"
echo "=== run_seed0_a9c_a10_a11.sh finished $(date)"
echo "=== total wall time: ${OVERALL_DUR}s"
echo "============================================================"
} | tee -a "$SUMMARY"
