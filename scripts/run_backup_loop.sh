#!/usr/bin/env bash
# run_backup_loop.sh — fires backup_to_aqua.sh every 6 hours indefinitely.
# Each tick defers to the backup script's own change detection, so ticks
# where nothing has changed locally just record a SKIP in the summary.
#
# Launch:
#   nohup bash scripts/run_backup_loop.sh >> logs/backup/backup_loop.log 2>&1 &
#   disown
#
# Stop:
#   pkill -f run_backup_loop.sh

trap "" HUP
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="/data/cs24m035/BPTT/GLASS/logs/backup"
mkdir -p "$LOG_DIR"
LOOP_LOG="$LOG_DIR/backup_loop.log"
INTERVAL_SEC=$((6 * 3600))   # 6 hours

{
  echo "===== [$(date '+%Y-%m-%d %H:%M:%S')] run_backup_loop.sh started PID=$$ interval=${INTERVAL_SEC}s"
} | tee -a "$LOOP_LOG"

while true; do
    echo "----- [$(date '+%Y-%m-%d %H:%M:%S')] tick start" | tee -a "$LOOP_LOG"
    bash "$SCRIPT_DIR/backup_to_aqua.sh" >> "$LOOP_LOG" 2>&1
    rc=$?
    echo "----- [$(date '+%Y-%m-%d %H:%M:%S')] tick end rc=$rc — sleeping ${INTERVAL_SEC}s" | tee -a "$LOOP_LOG"
    sleep "$INTERVAL_SEC"
done
