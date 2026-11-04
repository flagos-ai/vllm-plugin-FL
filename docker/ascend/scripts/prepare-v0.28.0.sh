#!/bin/bash
# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p sources
cd sources
if ! test -d vllm/.git; then
  git clone --depth 1 --branch v0.28.0 https://github.com/vllm-project/vllm.git vllm
fi
test "$(git -C vllm rev-parse HEAD)" = 2cf0a6915ce544dc493a0990f2ea38d81601128a
if ! test -d plugin/.git; then
  git clone --no-checkout https://github.com/flagos-ai/vllm-plugin-FL.git plugin
fi
if ! git -C plugin cat-file -e 9a7eaf1191999422682bf5c6d1cf7b3f88d253b5^{commit} 2>/dev/null; then
  git -C plugin fetch origin refs/pull/487/head
fi
git -C plugin checkout --detach 9a7eaf1191999422682bf5c6d1cf7b3f88d253b5
test "$(git -C plugin rev-parse HEAD)" = 9a7eaf1191999422682bf5c6d1cf7b3f88d253b5
git -C vllm rev-parse HEAD > vllm.commit
git -C plugin rev-parse HEAD > plugin.commit
if ! test -d FlagGems/.git; then
  git clone https://github.com/flagos-ai/FlagGems.git FlagGems
fi
git -C FlagGems checkout --detach 3b406c36212744b98b9720bf6d0a5387c09fe96b
test "$(git -C FlagGems rev-parse HEAD)" = 3b406c36212744b98b9720bf6d0a5387c09fe96b
