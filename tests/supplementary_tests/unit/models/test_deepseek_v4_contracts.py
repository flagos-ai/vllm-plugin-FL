# Copyright (c) 2026 BAAI. All rights reserved.

"""CPU-only contract tests for the DeepSeek-V4 Ascend adapter surface.

``vllm_fl/models/deepseek_v4.py`` is an Ascend 910C (torch_npu) execution
adapter: the module import itself pulls in ``vllm.forward_context``,
``vllm.models.deepseek_v4.*`` and checks ``current_platform.device_type ==
"npu"``, and the numeric paths in
``tests/unit_tests/models/test_deepseek_v4.py`` require torch_npu. What can
be validated without any accelerator is the *source contract*:

* the Ascend-only guard structure (npu-gated patches and class wiring),
* the registration strings the platform plugin installs,
* the numeric helpers' cache-layout preconditions (pure torch CPU).

The cache helpers are extracted from the source with an AST split (the same
technique as ``tests/unit_tests/attention/test_txda_attention.py``), so the
tests exercise the real production code, not a copy.
"""

import ast
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch

_SOURCE = Path(__file__).resolve().parents[4] / "vllm_fl/models/deepseek_v4.py"
_MODULE_NAME = "supp_deepseek_v4_helpers"


@pytest.fixture
def helpers(monkeypatch):
    """Extract and execute only the pure-torch cache helpers from the source.

    ``_cache_rows`` and ``_store_cache_rows`` have no vllm/torch_npu imports
    (only ``torch``, already loaded), so pulling just those two functions out
    of the AST gives the real production logic on CPU.
    """
    tree = ast.parse(_SOURCE.read_text(encoding="utf-8"))
    wanted = {"_cache_rows", "_store_cache_rows"}
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    assert {node.name for node in functions} == wanted

    module = ModuleType(_MODULE_NAME)
    module.__dict__["torch"] = torch
    exec(
        compile(
            ast.Module(body=functions, type_ignores=[]),
            str(_SOURCE),
            "exec",
        ),
        module.__dict__,
    )
    monkeypatch.setitem(sys.modules, _MODULE_NAME, module)
    return module


@pytest.mark.supplementary
@pytest.mark.parametrize(
    "cache_ndim,block_size",
    [(2, 4), (4, 4), (3, 8)],
)
def test_cache_helpers_reject_wrong_cache_shape(helpers, cache_ndim, block_size):
    """A cache not shaped [blocks, block_size, ...] is a hard ValueError."""
    cache = torch.zeros((2,) * cache_ndim)
    slots = torch.tensor([0, 1])

    with pytest.raises(ValueError, match="Unexpected DeepSeek-V4 KV cache shape"):
        helpers._cache_rows(cache, slots, block_size)
    with pytest.raises(ValueError, match="Unexpected DeepSeek-V4 KV cache shape"):
        helpers._store_cache_rows(
            cache, slots, torch.zeros(2, cache.shape[-1]), block_size
        )


@pytest.mark.supplementary
@pytest.mark.parametrize("block_size", [1, 2, 4])
def test_store_then_read_round_trips_across_pages(helpers, block_size):
    """Store and read logical rows through non-contiguous page assignment."""
    num_blocks = 3
    cache = torch.zeros((num_blocks, block_size, 5), dtype=torch.float32)
    width = cache.shape[-1]
    # Scrambled logical order: slot i maps to (page, offset) pairs that are
    # deliberately not in ascending order.
    targets = [(2, block_size - 1), (0, 0), (1, min(block_size - 1, 1))]
    slots = torch.tensor([page * block_size + offset for page, offset in targets])
    rows = torch.arange(3 * width, dtype=torch.float32).reshape(3, width) + 1.0

    helpers._store_cache_rows(cache, slots, rows, block_size)

    read_back = helpers._cache_rows(cache, slots, block_size)
    torch.testing.assert_close(read_back, rows, rtol=0, atol=0)
    # Nothing leaked into rows we did not write.
    expected_cache = torch.zeros_like(cache)
    for (page, offset), row in zip(targets, rows):
        expected_cache[page, offset] = row
    torch.testing.assert_close(cache, expected_cache, rtol=0, atol=0)


@pytest.mark.supplementary
def test_store_casts_rows_to_cache_dtype(helpers):
    """FP32 rows stored into a BF16 cache are downcast, not an error."""
    cache = torch.zeros((2, 2, 3), dtype=torch.bfloat16)
    slots = torch.tensor([0, 3])
    rows = torch.full((2, 3), 1.5, dtype=torch.float32)

    helpers._store_cache_rows(cache, slots, rows, block_size=2)

    assert cache.dtype == torch.bfloat16
    torch.testing.assert_close(
        helpers._cache_rows(cache, slots, 2), rows.to(torch.bfloat16), rtol=0, atol=0
    )


@pytest.mark.supplementary
def test_module_imports_only_under_npu_or_stubbed_vllm():
    """Document the import requirements: vllm + torch_npu, npu platform.

    The module is NPU-only by design; without vllm installed the import must
    fail, and with it the numeric tests live in unit_tests (torch_npu-gated).
    This test pins the guard lines so a refactor cannot silently make the
    module import eagerly patch non-Ascend platforms.
    """
    source = _SOURCE.read_text(encoding="utf-8")

    # The HCHead fallback and the XPU model override run only on npu.
    assert 'current_platform.device_type == "npu"' in source
    assert source.count('if current_platform.device_type == "npu":') == 2
    # The registration entry used by vllm_fl.register().
    vllm_fl_init = (_SOURCE.parents[1] / "__init__.py").read_text(encoding="utf-8")
    assert '"vllm_fl.models.deepseek_v4:DeepseekV4ForCausalLM"' in vllm_fl_init
    # Import-time hard dependencies that make this NPU-only.
    for required_import in (
        "from vllm.forward_context import get_forward_context",
        "from vllm.models.deepseek_v4.attention import DeepseekV4Attention",
    ):
        assert required_import in source


@pytest.mark.supplementary
def test_validate_short_context_contract_is_source_pinned():
    """Pin the validation branches without importing the module.

    ``_validate_short_context_attention`` enforces: allowed compress ratios,
    max_model_len <= min(sliding_window, 128), and the C4A index_topk bound.
    Its live behavior is covered by the torch_npu-gated core tests; here the
    source contract (all three branches present, exact error markers) is
    asserted so silent removal is caught in CPU-only CI.
    """
    source = _SOURCE.read_text(encoding="utf-8")
    assert "ratio not in (0, 1, 4, 128)" in source
    assert "compress_ratio={ratio}" in source
    assert "max_model_len <= " in source
    assert "(max_model_len + 3) // 4 > config.index_topk" in source
    assert "all compressed candidates" in source


@pytest.mark.supplementary
def test_backend_names_are_distinct_and_prefixed():
    """The two custom backends expose distinct FL names for registry lookup."""
    source = _SOURCE.read_text(encoding="utf-8")
    assert '"FL_ASCEND_V4_BF16_SWA_CACHE"' in source
    assert '"FL_ASCEND_V4_BF16_SWA"' in source
    assert "supported_kv_cache_dtypes = " in source
