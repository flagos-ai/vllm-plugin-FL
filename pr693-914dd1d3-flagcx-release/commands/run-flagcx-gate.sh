#!/usr/bin/env bash
set -Eeuo pipefail
cd /work/qualification
exec > >(tee -a launch.log) 2>&1
exec 9>/data/codex/metax032/.codex-pr693-serving.lock
flock -n 9
cid=''
cleanup() {
    local status=$?
    printf '%s\n' "$status" > exit
    if [ -n "$cid" ]; then docker rm -f "$cid" >/dev/null 2>&1 || true; fi
}
trap cleanup EXIT
mx-smi | tee gpu-before.log
grep -qi 'no process found' gpu-before.log
# A runner can be newly claimed before its first accelerator context appears.
container_names="$(docker ps --format '{{.Names}}')"
current_epoch="$(date +%s)"
while IFS= read -r name; do
    if [ -z "$name" ]; then continue; fi
    start_time="$(docker inspect --format '{{.State.StartedAt}}' "$name")"
    age=$(( current_epoch - $(date -d "$start_time" +%s) ))
    if (( age < 1800 )); then
        printf 'Newly started container: %s (%s seconds). Leaving shared node untouched.\n' "$name" "$age" >&2
        exit 75
    fi
done <<< "$container_names"
printf '08e22f00a7244c3849ae0fe94ae78690cd086e95d428860b1a479f8d14182595  source.bundle\n' | sha256sum -c
shallow_commit=914dd1d30f2212c8cee3184a3dc3cabaa44409a2
if [ ! -d src/.git ]; then
    git init --quiet src
    printf '%s\n' "$shallow_commit" > src/.git/shallow
    git -C src bundle unbundle "$PWD/source.bundle"
    git -C src checkout --quiet --detach "$shallow_commit"
fi
test "$(git -C src rev-parse HEAD^{tree})" = 2c375e273dd48b7752bfe1faabf16473a5ec41c3
image='sha256:b8579e0baa84898822e7592580c360e2d1405ba7b67d231f3d99740f6e03df4f'
docker image inspect --format '{{.Id}}' "$image"
cid="$(docker create --name fl-pr693-flagcx-gate --label purpose=pr693-gate758 --init --pull never --network none --ipc host --shm-size 64g \
    --device /dev/dri:/dev/dri:rwm --device /dev/mxcd:/dev/mxcd:rwm \
    -e FLAGCX_DEBUG=INFO -e FLAGCX_DEBUG_SUBSYS=INIT,GRAPH -e MACA_VISIBLE_DEVICES=0,1,2,3 -e GEMS_VENDOR=metax -e VLLM_PLUGINS=fl -e USE_FLAGGEMS=1 \
    -v "$PWD:/qualification" \
    -v /public-flash/models/Qwen3.6-35B-A3B:/data/models/Qwen/Qwen3.6-35B-A3B:ro \
    -v /public-flash/models/Qwen3.6-27B:/data/models/Qwen/Qwen3.6-27B:ro \
    --entrypoint bash "$image" /qualification/inside-flagcx-gate.sh "$@")"
printf '%s\n' "$cid" > cid
docker start -a "$cid"
docker inspect -f '{{.State.ExitCode}}' "$cid" | tee container-exit
test "$(cat container-exit)" = 0
