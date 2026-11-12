#!/usr/bin/env bash
# Install the validated commits without pip's unbounded VCS network operation.
set -euo pipefail
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

install_archive() {
  local repository=$1 revision=$2 digest=$3
  local archive="$work/$repository.tar.gz"
  local url="https://codeload.github.com/flagos-ai/$repository/tar.gz/$revision"
  if ! curl --fail --location --retry 2 --connect-timeout 10 \
      --max-time 120 "$url" -o "$archive"; then
    # Some CI containers inherit an unreachable runner proxy. Retry directly;
    # do not change the process-wide proxy needed by other install steps.
    rm -f "$archive"
    curl --noproxy '*' --fail --location --retry 2 --connect-timeout 10 \
      --max-time 120 "$url" -o "$archive"
  fi
  printf '%s  %s\n' "$digest" "$archive" | sha256sum -c -
  tar -xzf "$archive" -C "$work"
  # GitHub archives omit .git; preserve the validated SCM-derived version.
  SETUPTOOLS_SCM_PRETEND_VERSION_FOR_FLAG_GEMS=5.5.0.dev0 \
    python -m pip install --no-build-isolation --no-deps \
      "$work/$repository-$revision"
}

install_archive FlagGems 2bcfe2656dad0b654077fcfe7d66e6a93542afab \
  d376a37f5516ce2209209bbb44baf2e723b5db7df1018f37b2c862bb30f4e23b
install_archive FlagGems-vllm b434eafb1715849d1019ddcd480c5fc39461665c \
  827d65c8aa9d25a7c4ca8f4953d7eae56fa25808717c7755d91e5862061b10f3
