#!/bin/bash
set -euo pipefail
D=${TASK_DIR:-/data/codex/gla-padding-offline-20261120/attempt7}
WORK=/data/codex/flaggems-vllm-h20-marlin-0d86ec1
cd "$D"
sha256sum -c SHA256SUMS > package-verification.log
mkdir outputs
for phase in candidate base; do
  O="$D/outputs/$phase"
  mkdir "$O"
  cleanup_owned() {
    if [[ -s "$O/cid" ]]; then
      local cid
      cid=$(cat "$O/cid")
      if [[ "$(docker inspect --format '{{.State.Running}} {{index .Config.Labels "codex.task"}}' "$cid" 2>/dev/null || true)" == 'true gla-padding-offline-20261120' ]]; then
        docker stop --time 15 "$cid" > "$O/administrative-cleanup.log" 2>&1 || true
        if [[ "$(docker inspect --format '{{.State.Running}} {{index .Config.Labels "codex.task"}}' "$cid" 2>/dev/null || true)" == 'true gla-padding-offline-20261120' ]]; then
          docker kill "$cid" >> "$O/administrative-cleanup.log" 2>&1 || true
        fi
      fi
    fi
  }
  trap cleanup_owned EXIT
  date -u --iso-8601=seconds > "$O/started.utc"
  status=0
  timeout --signal=TERM --kill-after=20s 30m docker run --rm --pull=never \
    --runtime=runc --cpus=2 --memory=16g --network none --init \
    --cidfile "$O/cid" --name "codex-gla-padding-offline-$phase-20261120" \
    --label codex.task=gla-padding-offline-20261120 \
    -e NVIDIA_VISIBLE_DEVICES=void -e CUDA_VISIBLE_DEVICES= \
    -v "$WORK:/work:ro" \
    -v /data/codex/flaggems-vllm-pr1249-marlin-20261117/toolchain-22:/toolchain:ro \
    -v "$D:/probe" --workdir /probe --entrypoint /bin/bash \
    sha256:bd8a5d7a8774800c92d714c0c226b32c1863f929150f36efd610aebcd12a335d \
    /probe/inside.sh "$phase" > "$O/run.log" 2>&1 || status=$?
  cleanup_owned
  trap - EXIT
  printf '%s\n' "$status" > "$O/status"
  date -u --iso-8601=seconds > "$O/completed.utc"
  if [[ "$status" != 0 && "$status" != 124 && "$status" != 137 ]]; then
    echo "Compile API/environment failure; stop without trying another runtime/compiler/GPU." >&2
    exit "$status"
  fi
done
