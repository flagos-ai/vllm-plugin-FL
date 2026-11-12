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

"""CPU-side tests for vllm_fl.attention.sparse_indexer.

The module's import chain passes through vllm.v1 worker modules that pull
Linux-only packages (uvloop, xgrammar), so this file cannot import on the
Windows authoring host: locally it is validated with
``pytest --collect-only`` (per the campaign work order); the Linux CI images
execute it for real. The pure-Python branches (fp4 rejection, layer
registration, op-argument plumbing) run on CPU tensors; the fake/workspace
path carries ``pytest.mark.gpu`` because the workspace manager allocates on
the accelerator device (auto-skipped by the repo conftest elsewhere).

C1 tranche of the supplementary coverage suite (flagcicd_cov).
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


def test_indexer_registers_layer_operations_at_construction():
    """Construction publishes the operations dict under the cache prefix."""
    operations = {"fp8_fp4_mqa_logits": object()}
    prefix = "cov_indexer_register"
    indexer, _ = _make_indexer(prefix, operations)
    assert si._LAYER_OPERATIONS[prefix] is operations


def test_indexer_stores_configuration_verbatim():
    """Every constructor argument lands on the module unchanged."""
    prefix = "cov_indexer_config"
    indexer, k_cache = _make_indexer(prefix)
    assert indexer.k_cache is k_cache
    assert indexer.quant_block_size == 128
    assert indexer.scale_fmt == "e4m3"
    assert indexer.topk_tokens == 8
    assert indexer.head_dim == 128
    assert indexer.max_model_len == 1024
    assert indexer.max_total_seq_len == 4096
    assert indexer.topk_indices_buffer.dtype == torch.int32
    assert indexer.topk_indices_buffer.shape == (16, 8)


def test_indexer_forward_passes_fixed_flags_and_prefix_encoding():
    """The op dispatch passes False/False flags and the encoded layer name."""
    prefix = "cov_indexer_forward"
    seen = {}

    def recording_op(*args):
        seen["args"] = args
        seen["encoded"] = args[1]
        return torch.zeros(4, 8, dtype=torch.int32)

    # torch.ops.vllm.portable_sparse_attn_indexer dispatches to the Python
    # function registered at import; substitute a recorder via the module's
    # own op handle.
    original_op = torch.ops.vllm.portable_sparse_attn_indexer
    try:
        torch.ops.vllm.portable_sparse_attn_indexer = recording_op
        indexer, _ = _make_indexer(prefix)
        hidden = torch.zeros(2, 128)
        q = torch.zeros(2, 32, 128)
        k = torch.zeros(2, 128)
        weights = torch.ones(2, 32)
        result = indexer.forward(hidden, q, k, weights)
        args = seen["args"]
        # Positional contract: hidden, encoded prefix, kv cache, q, q_scale=None
        assert args[0] is hidden
        assert args[1] == si._encode_layer_name(prefix)
        assert args[3] is q
        assert args[4] is None
        assert args[5] is k
        assert args[6] is weights
        # Fixed configuration tail: quant block, scale fmt, dims, then the
        # buffer and both fixed False flags (skip_k_cache_insert,
        # use_fp4_cache) close out the 16-argument positional contract.
        assert args[7:13] == (128, "e4m3", 8, 128, 1024, 4096)
        assert args[13] is indexer.topk_indices_buffer
        assert args[14] is False  # skip_k_cache_insert
        assert args[15] is False  # use_fp4_cache
        assert len(args) == 16
        assert result is not None
    finally:
        torch.ops.vllm.portable_sparse_attn_indexer = original_op


def test_indexer_function_rejects_fp4_cache():
    """use_fp4_cache=True raises the documented ValueError."""
    prefix = "cov_indexer_fp4"
    _make_indexer(prefix)
    with pytest.raises(ValueError, match="requires FP8 cache") as excinfo:
        si.portable_sparse_attn_indexer(
            torch.zeros(2, 128),
            si._encode_layer_name(prefix),
            torch.zeros(4, 8, 32, dtype=torch.uint8),
            torch.zeros(2, 32, 128),
            None,
            torch.zeros(2, 128),
            torch.ones(2, 32),
            128,
            "e4m3",
            8,
            128,
            1024,
            4096,
            torch.zeros(16, 8, dtype=torch.int32),
            True,
            True,  # use_fp4_cache
        )
    assert "requires FP8 cache" in str(excinfo.value)


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
