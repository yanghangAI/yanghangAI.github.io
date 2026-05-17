#!/usr/bin/env bash
# Cron replacement: run the dashboard updater every 15 minutes forever.
# Intended to be left running in a tmux session on a Unity login node:
#   tmux new -s dashboard
#   ~/yanghangAI.github.io/scripts/cluster-dashboard-loop.sh
#   (Ctrl-b d to detach)

set -u

INTERVAL="${INTERVAL:-900}"  # seconds
SCRIPT="/home/hangyang_umass_edu/yanghangAI.github.io/scripts/update-cluster-dashboard.sh"
LOG="$HOME/cluster-dashboard.log"

while true; do
  {
    echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) tick ==="
    "$SCRIPT"
  } >> "$LOG" 2>&1 || echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) updater exited $?" >> "$LOG"
  sleep "$INTERVAL"
done
