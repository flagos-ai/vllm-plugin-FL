#!/bin/bash
# Copyright (c) 2025 BAAI. All rights reserved.
# Setup script for Ascend NPU CI environment.
set -eo pipefail

# Source CANN Toolkit and ATB environment variables,
# then persist any changes to $GITHUB_ENV so they survive across steps.
_before=$(mktemp)
printenv | sort > "$_before"

# set_env.sh references $ZSH_VERSION which is unset in bash; pre-define it to
# avoid "unbound variable" errors when the vendor script runs with set -u.
ZSH_VERSION=${ZSH_VERSION:-}

[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ] && source /usr/local/Ascend/ascend-toolkit/set_env.sh
[ -f /usr/local/Ascend/nnal/atb/set_env.sh ]       && source /usr/local/Ascend/nnal/atb/set_env.sh

if [ -n "${GITHUB_ENV:-}" ]; then
    printenv | sort | diff "$_before" - | grep '^>' | sed 's/^> //' >> "$GITHUB_ENV" || true
fi
rm -f "$_before"

# Install vLLM plugins and dependencies
pip install --upgrade pip "setuptools>=77.0.3"
pip install --no-build-isolation -e ".[test]"
