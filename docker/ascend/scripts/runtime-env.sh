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

# Fixed release stack: use the installed compiler in /flagos, not the base's
# selectable compiler trees. Do not download or upgrade packages at startup.
case ":$PATH:" in
  *:/flagos/bin:*) ;;
  *) export PATH="/flagos/bin:$PATH" ;;
esac
_vllm_pythonpath=""
IFS=: read -ra _vllm_paths <<< "${PYTHONPATH:-}"
for _vllm_path in "${_vllm_paths[@]}"; do
  case "$_vllm_path" in
    /opt/flagtree|/opt/triton|"") continue ;;
  esac
  _vllm_pythonpath="${_vllm_pythonpath:+$_vllm_pythonpath:}$_vllm_path"
done
export PYTHONPATH="$_vllm_pythonpath"
unset _vllm_pythonpath _vllm_paths _vllm_path
