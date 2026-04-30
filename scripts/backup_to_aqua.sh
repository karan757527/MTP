#!/usr/bin/env bash
# backup_to_aqua.sh — one-shot incremental rsync of our work to aqua.
#
# Skips the run if no file under the source trees has been modified since the
# last successful backup (detected via find -newer <marker>). First run (no
# marker yet) always executes.
#
# Backs up:
#   /data/cs24m035/BPTT/                            -> $REMOTE_ROOT/BPTT/
#   /home/cs24m035/.claude/projects/.../memory/     -> $REMOTE_ROOT/memory/
#
# Additive (no --delete): orphaned remote files are kept.
#
# Usage:
#   bash scripts/backup_to_aqua.sh           # normal, honors skip logic
#   FORCE=1 bash scripts/backup_to_aqua.sh   # ignore marker, always backup
#
# Prerequisite: passwordless SSH via key ~/.ssh/id_aqua to aqua:40826.

set -u

REMOTE_USER="cs24m021"
REMOTE_HOST="aqua.iitm.ac.in"
REMOTE_PORT="40826"
REMOTE_ROOT="/lfs/usrhome/mtech/cs24m021/backups/cs24m035_glass"
SSH_KEY="$HOME/.ssh/id_aqua"

SRC_BPTT="/data/cs24m035/BPTT"
SRC_MEMORY="/home/cs24m035/.claude/projects/-data-cs24m035-BPTT/memory"

LOG_DIR="/data/cs24m035/BPTT/GLASS/logs/backup"
mkdir -p "$LOG_DIR"
SUMMARY="$LOG_DIR/backup_summary.log"
MARKER="$LOG_DIR/.last_backup_marker"
TS="$(date +%Y%m%d_%H%M%S)"
DETAIL="$LOG_DIR/backup_${TS}.log"

FORCE="${FORCE:-0}"

log_both() { echo "$*" | tee -a "$SUMMARY" ; }

log_both "===== [$(date '+%Y-%m-%d %H:%M:%S')] backup_to_aqua.sh attempt TS=$TS"

# --- Change detection ---
# find -quit exits on the first hit so this is fast even for large trees.
# We exclude this script's own per-run detail logs so they don't self-trigger
# (backup_summary.log and .last_backup_marker are fine to include).
if [[ "$FORCE" != "1" && -f "$MARKER" ]]; then
    changed_sample=$(find "$SRC_BPTT" "$SRC_MEMORY" \
        -type f -newer "$MARKER" \
        -not -path "$LOG_DIR/backup_20*.log" \
        -print -quit 2>/dev/null)
    if [[ -z "$changed_sample" ]]; then
        log_both "[SKIP]  [$(date '+%Y-%m-%d %H:%M:%S')] no files changed since $(date -r "$MARKER" '+%Y-%m-%d %H:%M:%S') — rsync not invoked"
        exit 0
    fi
    log_both "[RUN]   [$(date '+%Y-%m-%d %H:%M:%S')] changes detected (first hit: ${changed_sample}) — invoking rsync"
else
    log_both "[RUN]   [$(date '+%Y-%m-%d %H:%M:%S')] FORCE=$FORCE marker_exists=$([[ -f $MARKER ]] && echo yes || echo no) — invoking rsync (full scan)"
fi

START=$(date +%s)

# Ensure remote dirs exist (idempotent)
ssh -i "$SSH_KEY" -p "$REMOTE_PORT" "${REMOTE_USER}@${REMOTE_HOST}" \
    "mkdir -p '$REMOTE_ROOT/BPTT' '$REMOTE_ROOT/memory'" >> "$DETAIL" 2>&1
rc_mkdir=$?

RSYNC_OPTS=(
    -avz
    --partial
    --stats
    --human-readable
    -e "ssh -i $SSH_KEY -p $REMOTE_PORT -o ServerAliveInterval=60 -o ServerAliveCountMax=3"
)

# Excluded from BPTT sync: large tensors that blow past aqua's user quota.
# These are regeneratable by running scripts/preprocess_elliptic2.py and
# scripts/preprocess_elliptic2_edges.py from the raw CSVs.
BPTT_EXCLUDES=(
    --exclude='GLASS/dataset_/elliptic2/processed/*.pt'
    --exclude='GLASS/dataset_/elliptic2/raw/raw_emb.pt'
)

{
  echo "----- [$(date '+%Y-%m-%d %H:%M:%S')] rsync $SRC_BPTT/ -> $REMOTE_ROOT/BPTT/"
  echo "       excludes: ${BPTT_EXCLUDES[*]}"
} >> "$DETAIL"
rsync "${RSYNC_OPTS[@]}" "${BPTT_EXCLUDES[@]}" \
    "$SRC_BPTT/" \
    "${REMOTE_USER}@${REMOTE_HOST}:$REMOTE_ROOT/BPTT/" \
    >> "$DETAIL" 2>&1
rc_bptt=$?

{
  echo "----- [$(date '+%Y-%m-%d %H:%M:%S')] rsync $SRC_MEMORY/ -> $REMOTE_ROOT/memory/"
} >> "$DETAIL"
rsync "${RSYNC_OPTS[@]}" \
    "$SRC_MEMORY/" \
    "${REMOTE_USER}@${REMOTE_HOST}:$REMOTE_ROOT/memory/" \
    >> "$DETAIL" 2>&1
rc_mem=$?

DUR=$(( $(date +%s) - START ))

# Extract a compact transfer-stats summary (sum across both rsync runs).
sent_lines=$(grep -E "^(Total transferred file size|sent [0-9])" "$DETAIL" 2>/dev/null | tr '\n' ' | ' | sed 's/ | $//')

if [[ $rc_bptt -eq 0 && $rc_mem -eq 0 && $rc_mkdir -eq 0 ]]; then
    touch "$MARKER"
    log_both "[OK]    [$(date '+%Y-%m-%d %H:%M:%S')] rc_mkdir=$rc_mkdir rc_bptt=$rc_bptt rc_mem=$rc_mem dur=${DUR}s detail=$(basename $DETAIL)"
    [[ -n "$sent_lines" ]] && log_both "        stats: ${sent_lines:0:300}"
else
    log_both "[FAIL]  [$(date '+%Y-%m-%d %H:%M:%S')] rc_mkdir=$rc_mkdir rc_bptt=$rc_bptt rc_mem=$rc_mem dur=${DUR}s detail=$(basename $DETAIL)"
fi
