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

"""Single-GPU test for vllm_fl.attention.sparse_indexer.

The CPU branches (fp4 rejection, layer registration, op-argument plumbing)
live in ``coverage_tests/unit/compilation/test_cov_sparse_indexer.py``. The
fake/workspace path allocates its radix workspace through the workspace
manager, which requires an initialized accelerator device, so this test
carries ``pytest.mark.gpu`` and lives in the functional tree.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

# The shim import is load-bearing: it must precede vllm_fl/vllm
# imports on hosts without triton. Do not alphabetize.
# isort: split
from . import _flaggems_shim  # noqa: F401  (must run before vllm_fl import)

# isort: split
import vllm_fl.attention.sparse_indexer as si


@pytest.fixture(autouse=True)
def _layer_operations_sandbox():
    """Keep _LAYER_OPERATIONS scoped to each test."""
    saved = dict(si._LAYER_OPERATIONS)
    yield
    si._LAYER_OPERATIONS.clear()
    si._LAYER_OPERATIONS.update(saved)


@pytest.fixture(autouse=True)
def _workspace_manager_sandbox(monkeypatch):
    """Shrink the transient logits buffer and drop the workspace singleton.

    The fake path allocates a VLLM_SPARSE_INDEXER_MAX_LOGITS_MB-sized uint8
    buffer; the environment variable is read live at call time, so pinning
    it to 1 MB keeps the GPU-memory footprint negligible. The workspace
    manager singleton is restored to its pre-test value afterwards.
    """
    import vllm.v1.worker.workspace as wsm

    monkeypatch.setenv("VLLM_SPARSE_INDEXER_MAX_LOGITS_MB", "1")
    saved_manager = wsm._manager
    yield
    wsm._manager = saved_manager


def _make_indexer(prefix: str, operations=None):
    """Build a PortableSparseAttnIndexer with a recording k-cache handle."""
    k_cache = SimpleNamespace(
        prefix=prefix,
        kv_cache=torch.zeros(4, 8, 32, dtype=torch.uint8),
    )
    indexer = si.PortableSparseAttnIndexer(
        k_cache,
        quant_block_size=128,
        scale_fmt="e4m3",
        topk_tokens=8,
        head_dim=128,
        max_model_len=1024,
        max_total_seq_len=4096,
        topk_indices_buffer=torch.zeros(16, 8, dtype=torch.int32),
        operations=operations or {},
    )
    return indexer, k_cache


@pytest.mark.gpu
def test_indexer_function_fake_path_returns_the_buffer():
    """Non-dict attn_metadata takes the fake/workspace path returning the buffer.

    The fake path allocates the radix workspace through the workspace
    manager, which requires an initialized accelerator device — hence the
    gpu marker (the repo conftest auto-skips it on accelerator-less hosts).
    """
    from vllm.forward_context import ForwardContext, override_forward_context
    from vllm.platforms import current_platform
    from vllm.v1.worker.workspace import init_workspace_manager

    init_workspace_manager(torch.device(current_platform.device_type))
    device = current_platform.device_type
    prefix = "cov_indexer_fake"
    _make_indexer(prefix)
    buffer = torch.full((16, 8), -3, dtype=torch.int32, device=device)
    ctx = ForwardContext(
        no_compile_layers={},
        attn_metadata=object(),  # non-dict metadata selects the fake branch
        slot_mapping=None,
    )
    with override_forward_context(ctx):
        result = si.portable_sparse_attn_indexer(
            torch.zeros(2, 128, device=device),
            si._encode_layer_name(prefix),
            torch.zeros(4, 8, 32, dtype=torch.uint8, device=device),
            torch.zeros(2, 32, 128, device=device),
            None,
            torch.zeros(2, 128, device=device),
            torch.ones(2, 32, device=device),
            128,
            "e4m3",
            8,
            128,
            1024,
            4096,
            buffer,
            True,
            False,
        )
    assert result is buffer
    assert bool((buffer == -3).all())
