#!/usr/bin/env bash
# watch_and_resume.sh — watches for A2 seed0 completion, then stops the
# main runner and launches the trimmed resume sweep.
#
# What it does (in order):
#   1. Polls logs/runner_summary.log every 30s.
#   2. When it sees `[SEED_DONE] ... exp=A2 seed=0`, sends SIGTERM to the
#      main runner bash (RUNNER_PID). Then kills any in-flight A2 seed1
#      python child (the runner may have already started it — up to ~30s
#      of compute lost, acceptable).
#   3. Waits a few seconds for processes to clear, then launches
#      scripts/run_resume.sh under nohup on the same GPU slot, appending
#      to logs/runner.log and logs/runner_summary.log.
#
# Safety:
#   - Only kills RUNNER_PID (passed in) and python processes whose cmdline
#     matches 'A2_seed1'. Never touches the A9 training on GPU 1.
#   - If RUNNER_PID is not alive when triggered, skips the kill and just
#     launches the resume (the runner must have finished some other way).
#
# Usage:
#   nohup env RUNNER_PID=524833 CUDA_VISIBLE_DEVICES=0 \
#       bash scripts/watch_and_resume.sh >> logs/watcher.log 2>&1 &
#   disown

trap "" HUP
set -u

: "${RUNNER_PID:?RUNNER_PID must be set (pid of run_all_experiments.sh)}"
: "${CUDA_VISIBLE_DEVICES:=0}"
export CUDA_VISIBLE_DEVICES

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

SUMMARY=logs/runner_summary.log
LOG_RUNNER=logs/runner.log

echo "[watcher] $(date) started, watching for A2 seed0 completion" \
    "(RUNNER_PID=$RUNNER_PID, GPU=$CUDA_VISIBLE_DEVICES)"

# ----------------------------------------------------------------------------
# Phase 1: wait for A2 seed0 DONE marker.
# ----------------------------------------------------------------------------
while true; do
    if grep -q 'exp=A2 seed=0 rc=0' "$SUMMARY" 2>/dev/null; then
        echo "[watcher] $(date) detected A2 seed0 SEED_DONE"
        break
    fi
    if grep -q 'exp=A2 seed=0 rc=[^0]' "$SUMMARY" 2>/dev/null; then
        echo "[watcher] $(date) A2 seed0 FAILED — still proceeding with resume"
        break
    fi
    # Bail out if the runner died unexpectedly (no point watching forever).
    if ! kill -0 "$RUNNER_PID" 2>/dev/null; then
        echo "[watcher] $(date) runner PID $RUNNER_PID is gone — assuming" \
             "A2 seed0 is done and proceeding"
        break
    fi
    sleep 30
done

# ----------------------------------------------------------------------------
# Phase 2: stop the main runner + any A2 seed1 that was started.
# ----------------------------------------------------------------------------
if kill -0 "$RUNNER_PID" 2>/dev/null; then
    echo "[watcher] $(date) SIGTERM -> runner PID $RUNNER_PID"
    kill -TERM "$RUNNER_PID" || true
    # The runner is currently blocked on its python child (A2 seed1). Killing
    # the bash script doesn't kill the child, so hunt it down explicitly.
    sleep 2
    pkill -TERM -f 'run_ablation_elliptic2.py.*A2_seed1' 2>/dev/null || true
    sleep 3
    # Last-ditch SIGKILL if anything survived.
    kill -KILL "$RUNNER_PID" 2>/dev/null || true
    pkill -KILL -f 'run_ablation_elliptic2.py.*A2_seed1' 2>/dev/null || true
else
    echo "[watcher] $(date) runner PID already gone, skipping kill"
fi

sleep 3

# Sanity: confirm no A2_seed1 python is still running. A9 on GPU 1 and
# A2 seed0 (if still finishing) are unaffected because we only matched
# 'A2_seed1' in the cmdline.
echo "[watcher] $(date) surviving run_ablation processes:"
pgrep -af run_ablation_elliptic2.py || echo "  (none)"

# ----------------------------------------------------------------------------
# Phase 3: launch the trimmed resume under nohup, appending to the same logs.
# ----------------------------------------------------------------------------
echo "[watcher] $(date) launching scripts/run_resume.sh"
nohup bash scripts/run_resume.sh >> "$LOG_RUNNER" 2>&1 &
RESUME_PID=$!
disown "$RESUME_PID" 2>/dev/null || true
echo "[watcher] $(date) resume launched PID=$RESUME_PID"
echo "[watcher] $(date) exiting"
