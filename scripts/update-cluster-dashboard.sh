#!/usr/bin/env bash
# Collects SLURM/quota state on Unity and publishes it as JSON to the site repo.
# Intended to run via cron on a Unity login node. See bottom of file for cron setup.

set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/hangyang_umass_edu/yanghangAI.github.io}"
OUT_FILE="$REPO_DIR/assets/cluster-data.json"
USER_NAME="$(whoami)"
TS="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

cd "$REPO_DIR"

# Make sure we are up to date so we never push a stale base.
git fetch --quiet origin master
git checkout --quiet master
git reset --quiet --hard origin/master

json_escape() {
  python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))'
}

# --- squeue --me ---------------------------------------------------------
# Format: JOBID|PARTITION|NAME|STATE|TIME|TIME_LIMIT|NODES|NODELIST(REASON)
SQUEUE_RAW="$(squeue --me --noheader \
  -o '%i|%P|%j|%T|%M|%l|%D|%R' 2>/dev/null || true)"

squeue_jobs_json="[]"
if [[ -n "$SQUEUE_RAW" ]]; then
  squeue_jobs_json="$(printf '%s\n' "$SQUEUE_RAW" | python3 -c '
import json, sys
rows = []
for line in sys.stdin:
    line = line.rstrip("\n")
    if not line:
        continue
    parts = line.split("|")
    if len(parts) < 8:
        continue
    rows.append({
        "jobid": parts[0], "partition": parts[1], "name": parts[2],
        "state": parts[3], "time": parts[4], "time_limit": parts[5],
        "nodes": parts[6], "nodelist": parts[7],
    })
print(json.dumps(rows))
')"
fi

# --- sacct recent (last 24h, completed/failed/cancelled/timeout) ---------
SACCT_RAW="$(sacct -u "$USER_NAME" --starttime now-1days \
  --format=JobID,JobName%30,Partition,State,Elapsed,End,ExitCode \
  --noheader --parsable2 -X 2>/dev/null || true)"

sacct_jobs_json="$(printf '%s\n' "$SACCT_RAW" | python3 -c '
import json, sys
rows = []
for line in sys.stdin:
    line = line.rstrip("\n")
    if not line:
        continue
    parts = line.split("|")
    if len(parts) < 7:
        continue
    rows.append({
        "jobid": parts[0], "name": parts[1], "partition": parts[2],
        "state": parts[3], "elapsed": parts[4], "end": parts[5],
        "exit_code": parts[6],
    })
# newest first, cap to 30
rows.reverse()
print(json.dumps(rows[:30]))
')"

# --- sinfo partition load ------------------------------------------------
# %P partition  %a avail  %l timelimit  %D nodes  %t state  %C cpus(A/I/O/T)
SINFO_RAW="$(sinfo --noheader -o '%P|%a|%l|%D|%t|%C' 2>/dev/null || true)"
sinfo_json="$(printf '%s\n' "$SINFO_RAW" | python3 -c '
import json, sys
rows = []
for line in sys.stdin:
    line = line.rstrip("\n")
    if not line:
        continue
    parts = line.split("|")
    if len(parts) < 6:
        continue
    rows.append({
        "partition": parts[0], "avail": parts[1], "timelimit": parts[2],
        "nodes": parts[3], "state": parts[4], "cpus_aiot": parts[5],
    })
print(json.dumps(rows))
')"

# --- quota / disk --------------------------------------------------------
# Generic df-based view of the paths the user cares about. Skip silently if absent.
QUOTA_PATHS=("$HOME" "/work/pi_nwycoff_umass_edu/hang")
quota_json="$(python3 - "${QUOTA_PATHS[@]}" <<'PY'
import json, os, shutil, sys
rows = []
for p in sys.argv[1:]:
    if not os.path.exists(p):
        continue
    try:
        u = shutil.disk_usage(p)
        rows.append({
            "path": p,
            "total_gb": round(u.total / 2**30, 1),
            "used_gb":  round(u.used  / 2**30, 1),
            "free_gb":  round(u.free  / 2**30, 1),
            "percent":  round(u.used / u.total * 100, 1) if u.total else 0,
        })
    except Exception as e:
        rows.append({"path": p, "error": str(e)})
print(json.dumps(rows))
PY
)"

# --- assemble ------------------------------------------------------------
mkdir -p "$(dirname "$OUT_FILE")"
python3 - "$TS" "$USER_NAME" "$squeue_jobs_json" "$sacct_jobs_json" "$sinfo_json" "$quota_json" > "$OUT_FILE" <<'PY'
import json, sys
ts, user, squeue_s, sacct_s, sinfo_s, quota_s = sys.argv[1:7]
out = {
    "updated_utc": ts,
    "user": user,
    "squeue": json.loads(squeue_s),
    "sacct_recent": json.loads(sacct_s),
    "sinfo": json.loads(sinfo_s),
    "quota": json.loads(quota_s),
}
print(json.dumps(out, indent=2))
PY

# --- commit & push only if changed ---------------------------------------
if git diff --quiet -- "$OUT_FILE"; then
  echo "no change"
  exit 0
fi

git add "$OUT_FILE"
git -c user.email="cluster-dashboard@local" \
    -c user.name="cluster-dashboard" \
    commit -m "chore(dashboard): refresh cluster data $TS" --quiet
git push --quiet origin master
echo "pushed"

# ---------------------------------------------------------------------------
# Cron setup (run on a Unity login node):
#   crontab -e
#   */15 * * * * /home/hangyang_umass_edu/yanghangAI.github.io/scripts/update-cluster-dashboard.sh >> $HOME/cluster-dashboard.log 2>&1
#
# The script pushes over the repo's configured remote. If the remote is HTTPS
# and prompts for credentials, switch to SSH once:
#   git -C /home/hangyang_umass_edu/yanghangAI.github.io remote set-url origin git@github.com:yanghangAI/yanghangAI.github.io.git
# ---------------------------------------------------------------------------
