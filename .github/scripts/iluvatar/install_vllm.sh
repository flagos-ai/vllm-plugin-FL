#!/usr/bin/env bash
# Install exact EMPTY vLLM without replacing the CoreX torch/FlagTree stack.
set -euo pipefail
archive=${VLLM_SOURCE_ARCHIVE:-}
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
if [[ -z "$archive" ]]; then
  archive="$work/vllm-v0.32.0.tar.gz"
  url=${VLLM_SOURCE_URL:-https://github.com/vllm-project/vllm/archive/refs/tags/v0.32.0.tar.gz}
  curl --fail --location --retry 3 --connect-timeout 20 --max-time 300 "$url" -o "$archive"
fi
printf '%s  %s\n' bf592c132f27bbb5f65cff0bbd2ee873eae722b79c8286934fd5b5cf88b43d61 "$archive" | sha256sum -c -
tar -xzf "$archive" -C "$work"
VLLM_TARGET_DEVICE=empty VLLM_VERSION_OVERRIDE=0.32.0 \
  python -m pip install --no-build-isolation --no-deps "$work/vllm-0.32.0"
python - <<'PY'
import vllm
assert vllm.__version__.split('+')[0] == '0.32.0', vllm.__version__
print('Exact vLLM EMPTY import:', vllm.__version__, vllm.__file__)
PY
