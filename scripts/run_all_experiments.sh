#!/usr/bin/env bash
# run_all_experiments.sh — full multi-seed sweep of the Elliptic2 ablation.
#
# For every cell in EXPERIMENTS, runs N_SEEDS independent training runs (each
# with a different seed). After every cell's seed sweep completes, calls the
# aggregator to compute mean / std of the headline metrics across the seeds
# and writes a per-cell summary JSON to logs/<EXP>_summary.json.
#
# Design goals:
#   * Resilient.  No `set -e`. A single seed crashing must NOT stop the rest.
#                 We trap SIGHUP (so SSH disconnect doesn't kill the run) and
#                 isolate each python invocation with `|| true`.
#   * SSH-safe.   Use with `nohup ... &`. Trap on HUP is belt + suspenders.
#   * Ordered.    Cheapest cells first so partial results are useful even if
#                 the run is interrupted.
#   * Reproducible. Per-seed log dir is logs/<EXP>_seed<SEED>/, never reused.
#                 Old contents are NOT auto-deleted by this script — wipe
#                 logs/ manually before launching if you want a fresh start.
#
# Usage:
#   nohup bash scripts/run_all_experiments.sh > logs/runner.log 2>&1 &
#   disown
#
# Tunables (env vars):
#   NSEEDS                — number of seeds per cell (default 5)
#   CUDA_VISIBLE_DEVICES  — GPU index (default 0)
#   EXPERIMENTS_OVERRIDE  — space-separated cell list to override the default
#                           order (e.g. EXPERIMENTS_OVERRIDE="A4 A8b").

trap "" HUP                       # ignore SIGHUP — survive SSH disconnect
set -u                            # NB: deliberately NO `-e`

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
NSEEDS="${NSEEDS:-5}"

# Order: easiest / fastest first → slowest last.
# Wall-time per single seed (from prior runs):
#   A4 ~2 min, A8b ~5 min, A7 ~7 min, A8 ~13 min, A5 ~30 min,
#   A2 ~30-60 min, A1 ~60-120 min, A6 ~150 min.
DEFAULT_EXPERIMENTS=(A4 A8b A7 A8 A5 A2 A1 A6)
if [[ -n "${EXPERIMENTS_OVERRIDE:-}" ]]; then
    read -r -a EXPERIMENTS <<< "$EXPERIMENTS_OVERRIDE"
else
    EXPERIMENTS=("${DEFAULT_EXPERIMENTS[@]}")
fi

# ---------------------------------------------------------------------------
# Resolve repo root (parent of this script's dir).
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
mkdir -p logs

SUMMARY=logs/runner_summary.log

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
{
echo "============================================================"
echo "=== run_all_experiments.sh started $(date)"
echo "=== NSEEDS=$NSEEDS"
echo "=== CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "=== EXPERIMENTS: ${EXPERIMENTS[*]}"
echo "=== PID: $$"
echo "============================================================"
} | tee -a "$SUMMARY"

OVERALL_START=$(date +%s)

# ---------------------------------------------------------------------------
# Outer loop — one cell at a time.
# ---------------------------------------------------------------------------
for exp in "${EXPERIMENTS[@]}"; do
    cfg="config/ablation/${exp}.yml"
    if [[ ! -f "$cfg" ]]; then
        echo "[ERROR] $(date) missing config $cfg — skipping cell" \
            | tee -a "$SUMMARY"
        continue
    fi

    {
    echo ""
    echo "------------------------------------------------------------"
    echo "[CELL_START] $(date) exp=$exp cfg=$cfg seeds=$NSEEDS"
    echo "------------------------------------------------------------"
    } | tee -a "$SUMMARY"

    cell_start=$(date +%s)

    # -----------------------------------------------------------------------
    # Inner loop — one seed at a time.
    # -----------------------------------------------------------------------
    for ((seed=0; seed<NSEEDS; seed++)); do
        log_dir="logs/${exp}_seed${seed}"
        mkdir -p "$log_dir"

        echo "[SEED_START] $(date) exp=$exp seed=$seed log=$log_dir" \
            | tee -a "$SUMMARY"

        seed_start=$(date +%s)

        # `python -u` for unbuffered stdout so `tail -f` works in real time.
        # `|| true` so a non-zero exit code can never abort the loop.
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
            # Append the JSONL final record (one line) to the summary so
            # users can grep "best_tst_prauc" in runner_summary.log.
            if [[ -f "$log_dir/train.jsonl" ]]; then
                tail -n 1 "$log_dir/train.jsonl" 2>/dev/null \
                    | tee -a "$SUMMARY" || true
            fi
        else
            echo "[SEED_FAIL] $(date) exp=$exp seed=$seed rc=$rc dur=${dur}s — see $log_dir/stdout.log" \
                | tee -a "$SUMMARY"
        fi
    done

    # -----------------------------------------------------------------------
    # Per-cell aggregation.  Wrapped in `|| true` so an aggregator failure
    # doesn't taint the rest of the sweep.
    # -----------------------------------------------------------------------
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
done

# ---------------------------------------------------------------------------
# Final summary across all cells.
# ---------------------------------------------------------------------------
OVERALL_END=$(date +%s)
OVERALL_DUR=$((OVERALL_END - OVERALL_START))

{
echo ""
echo "============================================================"
echo "=== run_all_experiments.sh finished $(date)"
echo "=== total wall time: ${OVERALL_DUR}s"
echo "============================================================"
} | tee -a "$SUMMARY"

# Best-effort cross-cell roll-up (won't fail the script if it crashes).
python -u scripts/aggregate_seeds.py --all --nseeds "$NSEEDS" \
    --out logs/all_summary.json 2>&1 \
    | tee -a "$SUMMARY" \
    || echo "[ROLLUP_FAIL] $(date) cross-cell rollup failed" | tee -a "$SUMMARY"

echo "=== done. summaries in logs/<EXP>_summary.json and logs/all_summary.json" \
    | tee -a "$SUMMARY"
