#!/bin/bash
# with_lock.sh <label> <cmd...>: run one controlled ae-dev probe under the shared bench lock.
# Waits up to 3 h for the lock. The child runs in the background so TERM/INT reach
# this script; on any exit the child is stopped and reaped, and this script's own
# children are confirmed gone, before the lock is released. Exits with the child's status.
set -euo pipefail
K=${WITH_LOCK_PATH:-/home/ae/Work/recall-performance-ae-dev-bench.lock}
E=/home/ae/Work/alleneubank/recall.worktrees/recall-performance-ids-93/scratch/evidence
label=$1; shift
start=$(date +%s); deadline=$((start + 10800))
until mkdir $K 2>/dev/null; do
  if [ "$(date +%s)" -ge $deadline ]; then echo "lock wait timed out" >&2; exit 75; fi
  sleep 15
done
echo "ids-93 $(date -u +%FT%TZ) $label" > $K/owner
got=$(date +%s)
child=""
status=1
release() {
  trap - EXIT TERM INT
  if [ -n "$child" ] && kill -0 "$child" 2>/dev/null; then
    kill -TERM "$child" 2>/dev/null || true
    wait "$child" 2>/dev/null || true
  fi
  # Builtin read, so no helper process of ours appears in the list.
  remaining=""
  read -r remaining < /proc/$$/task/$$/children || true
  if [ -n "${remaining// /}" ]; then
    echo "owned children still running: $remaining; lock kept for inspection" >&2
    exit 70
  fi
  rm -f $K/owner && rmdir $K
  printf '%s\t%s\t%s\t%s\t%s\n' "$label" "$(date -u -d @$start +%FT%TZ)" "$((got - start))" \
    "$(( $(date +%s) - got ))" "$status" >> $E/locks.tsv
  echo "{\"lock\":\"released\",\"label\":\"$label\",\"child_status\":$status}" >&2
  # Leave a window for the other lane's poller before this lane's next run.
  sleep 30
  exit $status
}
trap 'status=143; release' TERM
trap 'status=130; release' INT
trap release EXIT
"$@" &
child=$!
set +e
wait $child
status=$?
set -e
child=""
release
