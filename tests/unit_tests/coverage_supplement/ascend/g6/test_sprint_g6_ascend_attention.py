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

"""G6: CPU coverage for the Ascend regular-attention backend module.

Only phase selection, metadata building and the graph-fallback guards are
exercised. ``torch_npu`` kernels are never called; the mask builder and
metadata inputs are local stand-ins.
"""

from __future__ import annotations

import importlib.util
import os
import pathlib
import sys
import types
from types import SimpleNamespace

import pytest
import torch

_TARGET_REL = (
    "vllm_fl/dispatch/backends/vendor/ascend/impl/attention.py"
)
_IMPL_REL = "vllm_fl/dispatch/backends/vendor/ascend/impl"


def _find_repo():
    env = os.environ.get("FL_REPO_ROOT") or os.environ.get("FL_WORKSPACE")
    candidates = []
    if env:
        candidates.append(pathlib.Path(env))
    here = pathlib.Path(__file__).resolve()
    candidates.extend([here.parent, *here.parents])
    candidates.append(pathlib.Path.cwd())
    for entry in sys.path:
        if entry:
            candidates.append(pathlib.Path(entry))
    for base in candidates:
        if (base / "vllm_fl" / "dispatch" / "backends").is_dir():
            return base
    return None


def _exec_attention(name, path):
    impl_dir = path.parent
    package = types.ModuleType(name.rsplit(".", 1)[0])
    package.__path__ = [str(impl_dir)]
    package.__package__ = name.rsplit(".", 1)[0]
    sys.modules[package.__name__] = package

    import vllm.v1.attention.backends.registry as registry

    original = registry.register_backend
    registry.register_backend = lambda *_a, **_k: (lambda cls: cls)
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        registry.register_backend = original
    return module


_REPO = _find_repo()
if _REPO is None:
    pytest.skip("cannot locate vllm_fl tree on this host", allow_module_level=True)
_TARGET = _REPO / _TARGET_REL
if not _TARGET.is_file():
    pytest.skip(f"target module missing: {_TARGET}", allow_module_level=True)

try:
    attn = _exec_attention("g6_asc_impl.attention", _TARGET)
except Exception as exc:  # torch_npu / vllm not importable here
    attn = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


@pytest.fixture()
def mod():
    if attn is None:
        pytest.skip(f"attention module unavailable: {_IMPORT_ERROR}")
    return attn


def _fake_mask(monkeypatch, device=None):
    device = device or torch.device("cpu")

    class _Mask:
        def get_attn_mask(self, *_a, **_k):
            return "attn"

        def get_mla_mask(self, *_a, **_k):
            return "mla"

        def get_splitfuse_attn_mask(self, *_a, **_k):
            return "splitfuse"

    monkeypatch.setattr(
        attn.AscendAttentionMetadataBuilder, "_mask_builder", _Mask()
    )
    monkeypatch.setattr(
        attn.AscendAttentionMetadataBuilder, "_mask_builder_device", device
    )


def _allow_cpu_pin(monkeypatch):
    try:
        monkeypatch.setattr(torch.Tensor, "pin_memory", lambda self, *a, **k: self)
    except (TypeError, AttributeError):
        pass


def _config(mode, speculative=None, additional="missing"):
    return SimpleNamespace(
        speculative_config=speculative,
        compilation_config=SimpleNamespace(cudagraph_mode=mode),
        additional_config={} if additional == "missing" else additional,
    )


def test_using_paged_attention_matrix(mod):
    from vllm.config import CUDAGraphMode

    assert (
        mod.using_paged_attention(
            4, _config(CUDAGraphMode.FULL_DECODE_ONLY, speculative=object())
        )
        is False
    )
    assert (
        mod.using_paged_attention(4, _config(CUDAGraphMode.NONE), head_size=512)
        is True
    )
    assert mod.using_paged_attention(4, _config(CUDAGraphMode.NONE)) is False
    assert (
        mod.using_paged_attention(
            2,
            _config(CUDAGraphMode.FULL_DECODE_ONLY, additional={"pa_shape_list": [1, 2]}),
        )
        is True
    )
    assert (
        mod.using_paged_attention(
            3,
            _config(CUDAGraphMode.FULL_DECODE_ONLY, additional={"pa_shape_list": [1, 2]}),
        )
        is False
    )
    assert (
        mod.using_paged_attention(1, _config(CUDAGraphMode.FULL_DECODE_ONLY))
        is False
    )
    with pytest.raises(TypeError):
        mod.using_paged_attention(
            1,
            _config(CUDAGraphMode.FULL_DECODE_ONLY, additional={"pa_shape_list": "1,2"}),
        )
    with pytest.raises(TypeError):
        mod.using_paged_attention(
            1,
            _config(CUDAGraphMode.FULL_DECODE_ONLY, additional={"pa_shape_list": [1, True]}),
        )


def test_torch_npu_available_flag(mod):
    assert mod.is_torch_npu_available() is True


def test_classify_attention_states(mod):
    state = mod.AscendAttentionState
    classify = mod.classify_ascend_attention_state
    assert classify([0, 0], [1, 1], [1, 1], False, None) is state.PrefillNoCache
    assert classify([1, 1], [1, 1], [1, 1], False, "mtp") is state.SpecDecoding
    assert classify([2, 2], [1, 1], [1, 1], False, None) is state.DecodeOnly
    assert classify([3, 3], [2, 2], [1, 1], False, "mtp") is state.SpecDecoding
    assert (
        classify([3, 3], [2, 2], [1, 1], False, "eagle")
        is state.ChunkedPrefill
    )
    assert classify([3, 3], [2, 2], [2, 2], True, None) is state.ChunkedPrefill
    assert classify([3, 3], [2, 2], [2, 2], False, None) is state.PrefillCacheHit


def test_v2_state_uses_active_queries(mod):
    state = mod.AscendAttentionState
    cm = SimpleNamespace(
        num_reqs=3,
        query_start_loc_cpu=torch.tensor([0, 2, 3, 3], dtype=torch.int32),
    )
    seq_lens = torch.tensor([4, 3, 0])
    assert (
        mod.get_ascend_v2_attention_state(cm, seq_lens, True, None)
        is state.ChunkedPrefill
    )


def test_pad_v2_decode_metadata(mod):
    cm = SimpleNamespace(num_reqs=3, num_actual_tokens=3)
    qsl = torch.tensor([0, 1, 2, 2], dtype=torch.int32)
    seq_lens = torch.tensor([5, 6, 0])
    padded_qsl, padded_seq = mod.pad_ascend_v2_decode_metadata(cm, qsl, seq_lens)
    assert padded_qsl.tolist() == [0, 1, 2, 3]
    assert padded_seq.tolist() == [5, 6, 1]

    cm_ok = SimpleNamespace(num_reqs=3, num_actual_tokens=2)
    out_qsl, out_seq = mod.pad_ascend_v2_decode_metadata(cm_ok, qsl, seq_lens)
    assert out_qsl is not None and out_seq is seq_lens

    bad = torch.tensor([0, 2, 2, 2], dtype=torch.int32)
    with pytest.raises(ValueError):
        mod.pad_ascend_v2_decode_metadata(cm, bad, seq_lens)




def test_builder_init_counts_speculative_tokens(mod, monkeypatch):
    monkeypatch.setattr(mod.AttentionMetadataBuilder, "__init__", lambda self, *a, **k: None)
    cfg = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=1024),
        speculative_config=SimpleNamespace(num_speculative_tokens=2),
        scheduler_config=SimpleNamespace(enable_chunked_prefill=True),
    )
    builder = mod.AscendAttentionMetadataBuilder(object(), ["l0"], cfg, torch.device("cpu"))
    assert builder.decode_threshold == 3
    assert builder.chunked_prefill_enabled is True
    assert builder.max_num_blocks_per_req == 8

    cfg.speculative_config = None
    plain = mod.AscendAttentionMetadataBuilder(object(), ["l0"], cfg, torch.device("cpu"))
    assert plain.decode_threshold == 1


def test_mask_builder_cache_reused(mod, monkeypatch):
    device = torch.device("cpu")
    monkeypatch.setattr(mod.AscendAttentionMetadataBuilder, "_mask_builder", None)
    monkeypatch.setattr(mod.AscendAttentionMetadataBuilder, "_mask_builder_device", None)
    created = []

    class _Factory:
        def __init__(self, dev):
            created.append(dev)

    monkeypatch.setattr(mod, "AttentionMaskBuilder", _Factory)
    builder = object.__new__(mod.AscendAttentionMetadataBuilder)
    builder.device = device
    first = builder._get_mask_builder()
    second = builder._get_mask_builder()
    assert first is second
    assert created == [device]


def test_make_attention_mask_variants(mod, monkeypatch):
    _fake_mask(monkeypatch)
    builder = object.__new__(mod.AscendAttentionMetadataBuilder)
    builder.device = torch.device("cpu")

    builder.model_config = SimpleNamespace(runner_type="pooling")
    assert builder._make_attention_mask(mod.AscendAttentionState.DecodeOnly) == "attn"

    builder.model_config = SimpleNamespace(runner_type="generation", use_mla=True)
    assert builder._make_attention_mask(mod.AscendAttentionState.DecodeOnly) == "mla"

    builder.model_config = SimpleNamespace(runner_type="generation", use_mla=False)
    assert (
        builder._make_attention_mask(mod.AscendAttentionState.DecodeOnly)
        == "splitfuse"
    )


def test_reorder_batch_is_noop(mod):
    builder = object.__new__(mod.AscendAttentionMetadataBuilder)
    assert builder.reorder_batch(object(), object()) is False


def _builder(mod):
    builder = object.__new__(mod.AscendAttentionMetadataBuilder)
    builder.device = torch.device("cpu")
    builder.decode_threshold = 1
    builder.chunked_prefill_enabled = True
    builder.speculative_config = None
    builder.model_config = SimpleNamespace(runner_type="generation", use_mla=False)
    builder.vllm_config = SimpleNamespace(use_v2_model_runner=False)
    return builder


def test_build_pads_short_metadata(mod, monkeypatch):
    _fake_mask(monkeypatch)
    _allow_cpu_pin(monkeypatch)
    builder = _builder(mod)
    cm = SimpleNamespace(
        num_reqs=2,
        num_actual_tokens=3,
        max_query_len=1,
        query_start_loc_cpu=torch.tensor([0, 1, 2], dtype=torch.int32),
        block_table_tensor=torch.zeros(1, 4, dtype=torch.int32),
        _seq_lens_cpu=torch.tensor([5], dtype=torch.int32),
        slot_mapping=torch.arange(5),
        attn_state=mod.AscendAttentionState.DecodeOnly,
        causal=True,
    )
    metadata = builder.build(0, cm)
    assert metadata.attn_state is mod.AscendAttentionState.DecodeOnly
    assert metadata.seq_lens_list == [5, 1]
    assert metadata.block_tables.shape[0] == 2
    assert metadata.actual_seq_lengths_q == [1, 2]
    assert metadata.attn_mask == "splitfuse"


def test_build_derives_v2_state(mod, monkeypatch):
    _fake_mask(monkeypatch)
    _allow_cpu_pin(monkeypatch)
    builder = _builder(mod)
    builder.vllm_config = SimpleNamespace(use_v2_model_runner=True)
    cm = SimpleNamespace(
        num_reqs=2,
        num_actual_tokens=4,
        max_query_len=3,
        query_start_loc_cpu=torch.tensor([0, 1, 3], dtype=torch.int32),
        block_table_tensor=torch.zeros(2, 4, dtype=torch.int32),
        _seq_lens_cpu=torch.tensor([2, 4], dtype=torch.int32),
        slot_mapping=torch.arange(6),
        attn_state=None,
        causal=True,
    )
    metadata = builder.build(0, cm)
    assert metadata.attn_state is mod.AscendAttentionState.ChunkedPrefill


def test_build_pads_v2_decode(mod, monkeypatch):
    _fake_mask(monkeypatch)
    _allow_cpu_pin(monkeypatch)
    builder = _builder(mod)
    builder.vllm_config = SimpleNamespace(use_v2_model_runner=True)
    cm = SimpleNamespace(
        num_reqs=3,
        num_actual_tokens=3,
        max_query_len=1,
        query_start_loc_cpu=torch.tensor([0, 1, 2, 2], dtype=torch.int32),
        block_table_tensor=torch.zeros(3, 4, dtype=torch.int32),
        _seq_lens_cpu=torch.tensor([2, 2, 0], dtype=torch.int32),
        slot_mapping=torch.arange(6),
        attn_state=None,
        causal=True,
    )
    metadata = builder.build(0, cm)
    assert metadata.attn_state is mod.AscendAttentionState.DecodeOnly
    assert metadata.actual_seq_lengths_q == [1, 2, 3]


def test_build_seq_lens_fallbacks(mod, monkeypatch):
    _fake_mask(monkeypatch)
    _allow_cpu_pin(monkeypatch)
    builder = _builder(mod)

    base = dict(
        num_reqs=1,
        num_actual_tokens=2,
        max_query_len=2,
        query_start_loc_cpu=torch.tensor([0, 2], dtype=torch.int32),
        block_table_tensor=torch.zeros(1, 4, dtype=torch.int32),
        slot_mapping=torch.arange(3),
        attn_state=mod.AscendAttentionState.PrefillNoCache,
        causal=True,
    )
    without_private = SimpleNamespace(seq_lens=torch.tensor([2]), **base)
    m1 = builder.build(0, without_private)
    assert m1.seq_lens_list == [2]

    with_public = SimpleNamespace(
        seq_lens=torch.tensor([9]), seq_lens_cpu=torch.tensor([2]), **base
    )
    m2 = builder.build(0, with_public)
    assert m2.seq_lens_list == [2]


def test_split_decodes_and_prefills_paths(mod):
    builder = object.__new__(mod.AscendAttentionMetadataBuilder)
    builder.decode_threshold = 1

    decode_cm = SimpleNamespace(
        max_query_len=1,
        num_reqs=3,
        num_actual_tokens=3,
        query_start_loc_cpu=torch.tensor([0, 1, 2, 3], dtype=torch.int32),
    )
    assert builder._split_decodes_and_prefills(decode_cm) == (3, 0, 3, 0)

    mixed = SimpleNamespace(
        max_query_len=3,
        num_reqs=3,
        num_actual_tokens=5,
        query_start_loc_cpu=torch.tensor([0, 1, 4, 5], dtype=torch.int32),
    )
    assert builder._split_decodes_and_prefills(mixed) == (1, 2, 1, 4)

    no_prefill = SimpleNamespace(
        max_query_len=3,
        num_reqs=3,
        num_actual_tokens=3,
        query_start_loc_cpu=torch.tensor([0, 1, 2, 3], dtype=torch.int32),
    )
    assert builder._split_decodes_and_prefills(no_prefill) == (3, 0, 3, 0)


def test_graph_capture_entry_points(mod):
    builder = object.__new__(mod.AscendAttentionMetadataBuilder)
    builder.build = lambda **kwargs: SimpleNamespace(attn_state=None)
    built = builder.build_for_graph_capture(object(), mod.AscendAttentionState.DecodeOnly)
    assert built.attn_state is mod.AscendAttentionState.DecodeOnly

    other = object.__new__(mod.AscendAttentionMetadataBuilder)
    seen = {}
    other.build_for_graph_capture = (
        lambda cm, attn_state=None, model=None: seen.setdefault("state", attn_state)
    )
    other.build_for_cudagraph_capture(object())
    assert seen["state"] is mod.AscendAttentionState.DecodeOnly


def test_import_requires_torch_npu():
    target = _TARGET
    name = "g6_asc_nonpu.attention"
    package_name = "g6_asc_nonpu"
    package = types.ModuleType(package_name)
    package.__path__ = [str(target.parent)]
    sys.modules[package_name] = package

    had = "torch_npu" in sys.modules
    previous = sys.modules.get("torch_npu")
    sys.modules["torch_npu"] = None
    spec = importlib.util.spec_from_file_location(name, str(target))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        with pytest.raises(ImportError):
            spec.loader.exec_module(module)
    finally:
        if had:
            sys.modules["torch_npu"] = previous
        else:
            sys.modules.pop("torch_npu", None)
        sys.modules.pop(name, None)
        sys.modules.pop(package_name, None)
        sys.modules.pop(f"{package_name}.attention_mask", None)
