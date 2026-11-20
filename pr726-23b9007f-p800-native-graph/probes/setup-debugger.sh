#!/bin/bash
set -euo pipefail
root=/data/pr726-e8-crosshost/native-stack-20261120
image=harbor.baai.ac.cn/plugin/vllm-plugin-fl@sha256:93caedbfc36d665c2e7d9f70a38bbb2501a860c9389be660888b0b6d66bc4402
mkdir -p "$root/debugger"
docker run --rm --pull=never --name codex-flagcx729-gdb-setup-20261120 \
  --label codex.flagcx729=debugger-setup-20261120 \
  -v "$root/debugger:/debug" --entrypoint /bin/bash "$image" -ec '
    unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
    mkdir -p /debug/packages/partial /debug/root
    apt-get update
    apt-get -y -o Dir::Cache::archives=/debug/packages --download-only install gdb
    for package in /debug/packages/*.deb; do dpkg-deb -x "$package" /debug/root; done
    find /debug/packages -name "*.deb" -print0 | sort -z | xargs -0 sha256sum > /debug/packages.sha256
    LD_LIBRARY_PATH=/debug/root/usr/lib/x86_64-linux-gnu /debug/root/usr/bin/gdb --version
  ' > "$root/debugger-setup.log" 2>&1
printf '0\n' > "$root/debugger-setup.exit"
