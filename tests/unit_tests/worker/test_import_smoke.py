# Copyright (c) 2025 BAAI. All rights reserved.
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

"""Import health smoke tests for vllm_fl.worker modules.

These tests exist to catch broken top-level imports that the rest of the
unit-test suite (and the CI stack check) never exercise. The CI stack
check only imports ``vllm_fl.worker.nvidia_worker.NvidiaWorkerFL``, which
uses lazy method patching and never imports ``model_runner`` at module
level. When the v2 model runner is the default (as on vLLM 0.32.0 with
Triton present), ``vllm_fl.worker.model_runner`` is never imported in any
CI job, so a stale top-level ``import`` of a removed upstream symbol would
slip through. This module closes that gap.
"""


def test_model_runner_imports():
    """Top-level import of vllm_fl.worker.model_runner must succeed.

    This guards against stale imports of symbols removed by upstream vLLM
    version bumps (e.g. _reshape_attention_kv_cache removed in 0.32.0).
    """
    import vllm_fl.worker.model_runner as mod  # noqa: F401

    assert mod is not None
