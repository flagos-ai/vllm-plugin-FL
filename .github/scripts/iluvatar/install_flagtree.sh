#!/usr/bin/env bash
# Download the pinned vendor wheel with resumable, bounded retries.
set -euo pipefail
wheel=flagtree-0.7.0+iluvatar3.6-cp312-cp312-linux_x86_64.whl
url="https://resource.flagos.net/repository/flagos-pypi-hosted/packages/flagtree/0.7.0+iluvatar3.6/${wheel}"
digest=af6a7c9eddf167441fd872c3e6d673e9da572608dee79dba655b0775db740176
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
archive="$work/$wheel"
for attempt in 1 2 3 4; do
  if curl --fail --location --continue-at - --connect-timeout 20 \
      --max-time 1800 "$url" -o "$archive"; then
    printf '%s  %s\n' "$digest" "$archive" | sha256sum -c -
    python -m pip install --no-deps "$archive"
    exit 0
  else
    status=$?
  fi
  # A server without Range support requires a fresh download on the next try.
  if [[ "$status" == 33 ]]; then
    rm -f "$archive"
  fi
  if [[ "$attempt" == 4 ]]; then
    echo 'FlagTree download failed after four attempts' >&2
    exit "$status"
  fi
  echo "FlagTree download interrupted; retrying (attempt $((attempt + 1))/4)" >&2
  sleep 5
done
