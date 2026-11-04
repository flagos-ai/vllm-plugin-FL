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

"""FL worker package.

``worker`` and ``model_runner`` are adapted from vLLM v0.28's GPU worker and
resolve a few helpers that vLLM 0.32 removed. Re-publish them here, at the
moment this package is first imported: that is after the platform and operator
schemas are installed by ``vllm_fl.register()``, and before any worker
submodule can look the names up.
"""

from vllm_fl.patches.worker_symbols_032 import apply_worker_symbols_032

apply_worker_symbols_032()
