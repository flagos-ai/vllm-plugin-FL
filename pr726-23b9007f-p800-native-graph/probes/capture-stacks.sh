#!/bin/bash
set -euo pipefail
role=$1
name="codex-flagcx729-stack-20261120-direct-original-$role"
label="stack-20261120-direct-original-$role"
root=/data/pr726-e8-crosshost/native-stack-20261120
log="$root/direct-original-role-$role.log"
for iteration in $(seq 1 240); do
  if [[ -f "$log" ]] && [[ $(grep -c '^START_REPLAY [0-3] 1 1 2$' "$log" || true) -ge 2 ]]; then break; fi
  sleep 1
done
[[ "$(docker inspect --format '{{index .Config.Labels "codex.flagcx729"}}' "$name")" == "$label" ]]
date -u > "$root/stacks-role-$role.timestamp"
grep '^WORKER_PID' "$log" | sort -u > "$root/worker-pids-role-$role.txt"
while read -r marker rank pid; do
  [[ "$marker" == WORKER_PID && "$rank" =~ ^[0-3]$ && "$pid" =~ ^[0-9]+$ ]]
  timeout --signal=TERM --kill-after=2s 35s docker exec "$name" /bin/bash -ec '
    export LD_LIBRARY_PATH=/probe/debugger/root/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}
    /probe/debugger/root/usr/bin/gdb --data-directory=/probe/debugger/root/usr/share/gdb -q -batch \
      -ex "set pagination off" -ex "set auto-load safe-path /" \
      -ex "set debuginfod enabled off" -ex "thread apply all bt 35" \
      -ex "info sharedlibrary" -ex "detach" -p "$1"
  ' bash "$pid" > "$root/gdb-rank-$rank.txt" 2>&1
done < "$root/worker-pids-role-$role.txt"
date -u >> "$root/stacks-role-$role.timestamp"
