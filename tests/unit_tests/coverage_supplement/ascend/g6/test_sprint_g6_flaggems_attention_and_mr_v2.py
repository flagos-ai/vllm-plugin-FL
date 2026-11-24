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

"""G6: CPU coverage for FlagGems attention and the Ascend V2 runner adapter.

FlagGems' flash-attention kernels and every ``torch_npu`` operator are
replaced by local recorders. The Ascend V2 model runner is imported through
synthetic package shims so its Ascend-only sibling modules are not executed.
"""

from __future__ import annotations

import contextlib
import importlib.util
import os
import pathlib
import sys
import types
from types import SimpleNamespace

import pytest
import torch

_FLAGGEMS_REL = "vllm_fl/dispatch/backends/flaggems/impl/attention.py"
_ASCEND_REL = "vllm_fl/dispatch/backends/vendor/ascend"


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


def _load_path(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_REPO = _find_repo()
if _REPO is None:
    pytest.skip("cannot locate vllm_fl tree on this host", allow_module_level=True)

if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def _load_flaggems():
    try:
        from flaggems_vllm import (  # noqa: F401
            flash_attn_varlen_func,
            flash_mla,
            reshape_and_cache_flash,
        )
    except Exception:
        if "flaggems_vllm" not in sys.modules:
            stub = types.ModuleType("flaggems_vllm")
            stub.flash_attn_varlen_func = lambda *a, **k: None
            stub.flash_mla = lambda *a, **k: None
            stub.reshape_and_cache_flash = lambda *a, **k: None
            sys.modules["flaggems_vllm"] = stub
    path = _REPO / _FLAGGEMS_REL
    return _load_path("g6_flaggems_attention", path)


def _load_v2_runner():
    ascend = _REPO / _ASCEND_REL
    worker = ascend / "worker"
    patches = ascend / "patches"

    pkg = types.ModuleType("g6mrv")
    pkg.__path__ = [str(ascend)]
    sys.modules["g6mrv"] = pkg

    worker_pkg = types.ModuleType("g6mrv.worker")
    worker_pkg.__path__ = [str(worker)]
    sys.modules["g6mrv.worker"] = worker_pkg

    patches_pkg = types.ModuleType("g6mrv.patches")
    patches_pkg.__path__ = [str(patches)]
    sys.modules["g6mrv.patches"] = patches_pkg

    device_shim = types.ModuleType("g6mrv.patches.ascend_v2_device")
    device_shim.install_ascend_v2_device_shims = lambda: False
    sys.modules["g6mrv.patches.ascend_v2_device"] = device_shim

    sampling_shim = types.ModuleType("g6mrv.patches.ascend_v2_sampling")
    sampling_shim.install_ascend_v2_sampling_shims = lambda: False
    sys.modules["g6mrv.patches.ascend_v2_sampling"] = sampling_shim

    block_table = types.ModuleType("g6mrv.worker.ascend_v2_block_table")

    class _AscendBlockTables:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    block_table.AscendBlockTables = _AscendBlockTables
    sys.modules["g6mrv.worker.ascend_v2_block_table"] = block_table

    graph = types.ModuleType("g6mrv.worker.ascend_v2_graph")

    class _AscendModelGraphManager:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def release_graphs(self):
            return None

    graph.AscendModelGraphManager = _AscendModelGraphManager
    graph.install_graph_metadata_bridge = lambda *a, **k: None
    sys.modules["g6mrv.worker.ascend_v2_graph"] = graph

    return _load_path(
        "g6mrv.worker.model_runner_v2", worker / "model_runner_v2.py"
    )


try:
    fg = _load_flaggems()
except Exception as exc:  # flaggems / vllm unavailable
    fg = None
    _FG_ERROR = exc
else:
    _FG_ERROR = None

try:
    mrv2 = _load_v2_runner()
except Exception as exc:  # vllm V2 modules unavailable
    mrv2 = None
    _MRV2_ERROR = exc
else:
    _MRV2_ERROR = None


@pytest.fixture()
def fl():
    if fg is None:
        pytest.skip(f"flaggems attention unavailable: {_FG_ERROR}")
    return fg


@pytest.fixture()
def mr():
    if mrv2 is None:
        pytest.skip(f"model_runner_v2 unavailable: {_MRV2_ERROR}")
    return mrv2


def _block_sizes_config(hybrid, ssm_dtype="auto", mamba_dtype="auto"):
    return SimpleNamespace(
        model_config=SimpleNamespace(is_hybrid=hybrid),
        cache_config=SimpleNamespace(
            mamba_ssm_cache_dtype=ssm_dtype, mamba_cache_dtype=mamba_dtype
        ),
    )




def test_backend_capability_helpers(fl):
    from vllm.v1.attention.backend import AttentionType

    backend = fl.AttentionFLBackend
    assert backend.get_name() == "CUSTOM"
    assert backend.get_impl_cls() is fl.AttentionFLImpl
    assert backend.get_builder_cls() is fl.AttentionFLMetadataBuilder
    assert backend.supports_sink() is False
    assert backend.supports_attn_type(AttentionType.DECODER) is True
    assert backend.supports_attn_type("bogus") is False
    assert backend.supports_kv_cache_dtype(None) is True
    assert backend.supports_kv_cache_dtype("auto") is True
    assert backend.supports_kv_cache_dtype("fp8") is False
    assert backend.supports_head_size(128) is True
    assert backend.supports_head_size(12) is False
    assert backend.supports_head_size(512) is False

    kwargs = dict(
        head_size=128,
        dtype=torch.float16,
        kv_cache_dtype="auto",
        block_size=16,
        use_mla=False,
        use_sparse=False,
        device_capability=None,
    )
    assert (
        backend.supports_combination(has_sink=True, use_mm_prefix=False, **kwargs)
        == "not support sink"
    )
    assert (
        backend.supports_combination(has_sink=False, use_mm_prefix=True, **kwargs)
        == "FlagGems FA2 does not support multimodal prefix attention"
    )
    assert (
        backend.supports_combination(has_sink=False, use_mm_prefix=False, **kwargs)
        is None
    )




def _fg_builder(fl):
    builder = object.__new__(fl.AttentionFLMetadataBuilder)
    builder.vllm_config = SimpleNamespace()
    builder.device = torch.device("cpu")
    builder.aot_schedule = False
    builder.aot_sliding_window = None
    builder.use_full_cuda_graph = False
    builder.max_cudagraph_size = 0
    builder.max_num_splits = 0
    builder.dcp_world_size = 1
    builder.dcp_rank = 0
    builder.cp_kv_cache_interleave_size = 1
    return builder


def _fg_metadata(num_reqs=2, num_tokens=4, causal=True):
    return SimpleNamespace(
        num_reqs=num_reqs,
        num_actual_tokens=num_tokens,
        max_query_len=2,
        max_seq_len=16,
        query_start_loc=torch.tensor([0, 2, num_tokens], dtype=torch.int32),
        seq_lens=torch.tensor([8, 12], dtype=torch.int32),
        block_table_tensor=torch.zeros(num_reqs, 2, dtype=torch.int32),
        slot_mapping=torch.arange(num_tokens),
        causal=causal,
    )


def test_builder_build_plain_and_graph(fl, monkeypatch):
    builder = _fg_builder(fl)
    metadata = builder.build(0, _fg_metadata())
    assert metadata.use_cascade is False
    assert metadata.max_num_splits == 0

    builder.use_full_cuda_graph = True
    builder.max_cudagraph_size = 8
    capped = builder.build(0, _fg_metadata())
    assert capped.max_num_splits == 0


def test_builder_build_aot_disables_on_mixed_windows(fl, monkeypatch):
    builder = _fg_builder(fl)
    builder.aot_schedule = True
    monkeypatch.setattr(
        fl, "_get_sliding_window_configs", lambda cfg: {(4095, 0), (4095, 4095)}
    )
    metadata = builder.build(0, _fg_metadata())
    assert builder.aot_schedule is False
    assert builder.aot_sliding_window == (-1, -1)
    assert metadata.use_cascade is False


def test_builder_build_cascade(fl):
    builder = _fg_builder(fl)
    metadata = builder.build(64, _fg_metadata())
    assert metadata.use_cascade is True
    assert metadata.cu_prefix_query_lens.tolist() == [0, 4]
    assert metadata.prefix_kv_lens.tolist() == [64]
    assert metadata.suffix_kv_lens.tolist() == [-56, -52]


def test_split_kv_cache_variants(fl):
    head = 8
    cache = torch.arange(2 * 2 * 4 * (2 * head), dtype=torch.float32).reshape(
        2, 2, 4, 2 * head
    )
    key, value = fl._split_kv_cache(cache, head)
    assert key.shape == (2, 4, 2, head)
    assert value.shape == (2, 4, 2, head)

    with pytest.raises(ValueError):
        fl._split_kv_cache(torch.zeros(2, 2, 4, head), head)
    with pytest.raises(ValueError):
        fl._split_kv_cache(torch.zeros(2, 2, 4), head)


def test_do_kv_cache_update_encoder_and_decoder(fl, monkeypatch):
    impl = object.__new__(fl.AttentionFLImpl)
    impl.attn_type = fl.AttentionType.ENCODER_ONLY
    impl.head_size = 8
    assert impl.do_kv_cache_update(None, None, None, None, None) is None

    impl.attn_type = fl.AttentionType.DECODER
    impl.kv_cache_dtype = "auto"
    calls = []

    import vllm.v1.attention.ops.triton_reshape_and_cache_flash as triton_cache

    monkeypatch.setattr(
        triton_cache,
        "triton_reshape_and_cache_flash",
        lambda *a, **k: calls.append((a, k)),
    )
    head, num_tokens = 8, 4
    key = torch.zeros(num_tokens, 2, head)
    value = torch.zeros(num_tokens, 2, head)
    kv_cache = torch.zeros(2, 2, 4, 2 * head)
    layer = SimpleNamespace(_k_scale=torch.ones(1), _v_scale=torch.ones(1))
    impl.do_kv_cache_update(
        layer, key, value, kv_cache, torch.arange(num_tokens)
    )
    assert len(calls) == 1


def _fg_impl(fl):
    impl = object.__new__(fl.AttentionFLImpl)
    impl.attn_type = fl.AttentionType.DECODER
    impl.head_size = 8
    impl.num_heads = 4
    impl.num_kv_heads = 2
    impl.scale = 0.5
    impl.alibi_slopes = None
    impl.sliding_window = (-1, -1)
    impl.logits_soft_cap = 0
    impl.vllm_flash_attn_version = 2
    impl.dcp_world_size = 1
    return impl


def _fg_layer():
    return SimpleNamespace(
        _q_scale=torch.ones(1),
        _k_scale=torch.ones(1),
        _v_scale=torch.ones(1),
    )


def test_forward_decoder_plain(fl, monkeypatch):
    impl = _fg_impl(fl)
    calls = []
    monkeypatch.setattr(fl, "flash_attn_varlen_func", lambda **k: calls.append(k))
    num_tokens = 4
    query = torch.zeros(num_tokens, 4, 8)
    key = torch.zeros(num_tokens, 2, 8)
    value = torch.zeros(num_tokens, 2, 8)
    kv_cache = torch.zeros(2, 2, 4, 16)
    output = torch.zeros(num_tokens, 4, 8)
    metadata = SimpleNamespace(
        num_actual_tokens=num_tokens,
        query_start_loc=torch.tensor([0, 2, 4], dtype=torch.int32),
        seq_lens=torch.tensor([8, 12], dtype=torch.int32),
        max_query_len=2,
        max_seq_len=16,
        block_table=torch.zeros(2, 2, dtype=torch.int32),
        scheduler_metadata=None,
        use_cascade=False,
        causal=True,
    )
    out = impl.forward(
        _fg_layer(), query, key, value, kv_cache, metadata, output=output
    )
    assert out is output
    assert len(calls) == 1


def test_forward_profiling_and_bad_scales(fl):
    impl = _fg_impl(fl)
    output = torch.zeros(4, 4, 8)
    out = impl.forward(
        _fg_layer(), torch.zeros(4, 4, 8), None, None, None, None, output=output
    )
    assert torch.count_nonzero(out) == 0

    with pytest.raises(NotImplementedError):
        impl.forward(
            _fg_layer(),
            torch.zeros(4, 4, 8),
            None,
            None,
            None,
            None,
            output=output,
            output_scale=torch.ones(1),
        )




def test_forward_dcp_branch(fl, monkeypatch):
    impl = _fg_impl(fl)
    impl.dcp_world_size = 2
    monkeypatch.setattr(impl, "_forward_with_dcp", lambda *a, **k: None)
    num_tokens = 4
    output = torch.zeros(num_tokens, 4, 8)
    metadata = SimpleNamespace(
        num_actual_tokens=num_tokens,
        query_start_loc=torch.tensor([0, 2, 4], dtype=torch.int32),
        seq_lens=torch.tensor([8, 12], dtype=torch.int32),
        max_query_len=2,
        max_seq_len=16,
        block_table=torch.zeros(2, 2, dtype=torch.int32),
        scheduler_metadata=None,
        use_cascade=False,
        causal=True,
    )
    out = impl.forward(
        _fg_layer(),
        torch.zeros(num_tokens, 4, 8),
        torch.zeros(num_tokens, 2, 8),
        torch.zeros(num_tokens, 2, 8),
        torch.zeros(2, 2, 4, 16),
        metadata,
        output=output,
    )
    assert out is output


def test_forward_cascade_branch(fl, monkeypatch):
    impl = _fg_impl(fl)
    calls = []
    monkeypatch.setattr(fl, "cascade_attention", lambda *a, **k: calls.append(k))
    num_tokens = 4
    output = torch.zeros(num_tokens, 4, 8)
    metadata = SimpleNamespace(
        num_actual_tokens=num_tokens,
        query_start_loc=torch.tensor([0, 2, 4], dtype=torch.int32),
        max_query_len=2,
        max_seq_len=16,
        block_table=torch.zeros(2, 2, dtype=torch.int32),
        use_cascade=True,
        cu_prefix_query_lens=torch.tensor([0, 4], dtype=torch.int32),
        prefix_kv_lens=torch.tensor([64], dtype=torch.int32),
        suffix_kv_lens=torch.tensor([-56, -52], dtype=torch.int32),
        common_prefix_len=64,
        prefix_scheduler_metadata=None,
        scheduler_metadata=None,
    )
    out = impl.forward(
        _fg_layer(),
        torch.zeros(num_tokens, 4, 8),
        torch.zeros(num_tokens, 2, 8),
        torch.zeros(num_tokens, 2, 8),
        torch.zeros(2, 2, 4, 16),
        metadata,
        output=output,
    )
    assert out is output
    assert len(calls) == 1


def test_forward_with_dcp_merges(fl, monkeypatch):
    impl = _fg_impl(fl)
    num_tokens, heads, head_size = 4, 4, 8
    query = torch.zeros(num_tokens, heads, head_size)
    key_cache = torch.zeros(2, 2, 4, head_size)
    value_cache = torch.zeros(2, 2, 4, head_size)
    output = torch.zeros(num_tokens, heads, head_size)
    metadata = SimpleNamespace(
        query_start_loc=torch.tensor([0, 2, 4], dtype=torch.int32),
        max_query_len=2,
        block_table=torch.zeros(2, 2, dtype=torch.int32),
        dcp_context_kv_lens=torch.tensor([6, 10], dtype=torch.int32),
        max_dcp_context_kv_len=8,
        scheduler_metadata=None,
        causal=True,
    )

    class _Group:
        def all_gather(self, tensor, dim=1):
            return torch.cat([tensor, tensor], dim=dim)

    monkeypatch.setattr(fl, "get_dcp_group", lambda: _Group())

    results = [
        (torch.zeros(num_tokens, 2 * heads, head_size), torch.zeros(2, num_tokens)),
        (torch.zeros(num_tokens, heads, head_size), torch.zeros(num_tokens, 1)),
    ]
    monkeypatch.setattr(fl, "flash_attn_varlen_func", lambda **k: results.pop(0))
    monkeypatch.setattr(
        fl,
        "cp_lse_ag_out_rs",
        lambda out, lse, group, return_lse=True: (
            torch.zeros(num_tokens, heads, head_size),
            torch.zeros(1, num_tokens),
        ),
    )
    merged = []
    monkeypatch.setattr(fl, "merge_attn_states", lambda *a: merged.append(a))

    impl._forward_with_dcp(
        query, torch.zeros(num_tokens, 2, head_size),
        torch.zeros(num_tokens, 2, head_size), key_cache, value_cache,
        output, metadata,
    )
    assert len(merged) == 1


def test_encoder_attention_rejects_fp8(fl):
    impl = _fg_impl(fl)
    impl.kv_cache_dtype = "fp8_e4m3"
    with pytest.raises(NotImplementedError):
        impl._forward_encoder_attention(
            torch.zeros(4, 4, 8), None, None, torch.zeros(4, 4, 8), None, None
        )


def test_encoder_attention_calls_flash(fl, monkeypatch):
    impl = _fg_impl(fl)
    impl.kv_cache_dtype = "auto"
    calls = []
    monkeypatch.setattr(fl, "flash_attn_varlen_func", lambda **k: calls.append(k))
    num_tokens = 4
    output = torch.zeros(num_tokens, 4, 8)
    metadata = SimpleNamespace(
        query_start_loc=torch.tensor([0, 2, 4], dtype=torch.int32),
        max_query_len=2,
    )
    out = impl._forward_encoder_attention(
        torch.zeros(num_tokens, 4, 8),
        torch.zeros(num_tokens, 2, 8),
        torch.zeros(num_tokens, 2, 8),
        output,
        metadata,
        _fg_layer(),
    )
    assert out is output
    assert len(calls) == 1


class _CgSupport:
    min_cg_support = "min"
    min_cg_attn_backend = "backend"

    def narrow(self, *_args):
        return self


def _v2_runner(m, *, draft, adaptive, profiling, pcp_manager, fast_prefill, aux):
    runner = object.__new__(m.GPUModelRunnerFLV2)
    runner._ascend_manager_views = {}
    runner._ascend_block_planes = []
    runner.parallel_config = SimpleNamespace(cp_kv_cache_interleave_size=1)
    runner.cudagraph_manager = object()
    runner.max_model_len = 100
    runner.is_encoder_decoder = True
    runner.scheduler_config = SimpleNamespace(max_num_encoder_input_tokens=64)
    runner.model_config = SimpleNamespace(
        hf_config=SimpleNamespace(max_source_positions=32, model_type="qwen")
    )
    runner.vllm_config = SimpleNamespace(
        use_v2_model_runner=True,
        mamba_config=object(),
        aux_output_config=SimpleNamespace(enabled=aux),
    )
    spec = SimpleNamespace(
        block_size=128,
        uses_slot_mapping=True,
        dcp_sharded=False,
        block_table_token_alignment=1,
        max_num_blocks_per_req=lambda cfg, maxlen: 4,
    )
    group = SimpleNamespace(kv_cache_spec=spec, layer_names=["a", "b"])
    runner.kv_cache_config = SimpleNamespace(kv_cache_groups=[group])

    class _ModelState:
        num_new_sampled_tokens_per_step = 1

        def get_additional_cg_support(self):
            return (slice(None),)

        def initialize_kv_cache(self, *args):
            self.inited = args

    runner.model_state = _ModelState()
    runner.jit_warmup_registry = SimpleNamespace(
        activate=lambda: contextlib.nullcontext()
    )
    runner.req_states = object()
    runner.input_buffers = SimpleNamespace(query_start_loc=object())
    runner.vocab_size = 1000
    runner.max_num_reqs = 4
    runner.max_num_tokens = 64
    runner.device = torch.device("cpu")
    runner.dcp_size = 1
    runner.dcp_rank = 0
    runner.supports_mm_inputs = False
    runner.pcp_manager_cls = object
    runner.cache_config = SimpleNamespace(
        use_replayssm=False, kv_sharing_fast_prefill=fast_prefill
    )
    runner.model = object()
    runner.decode_query_len = 1
    runner.lora_capture_cases = []
    runner.compilation_config = SimpleNamespace(
        cudagraph_mode=m.CUDAGraphMode.NONE,
        static_forward_context={},
        resolve_cudagraph_mode_and_sizes=lambda *a, **k: "RESOLVED",
    )
    runner.speculator = None
    if draft:
        runner.speculator = m.DraftModelSpeculator()
        runner.speculator.draft_attn_layer_names = {"a"}
    runner._allocate_and_bind_kv_cache = lambda **k: {
        "a": SimpleNamespace(device=torch.device("cpu")),
        "skip": SimpleNamespace(device=torch.device("cuda")),
    }
    return runner


def test_v2_runner_init_rejects_non_npu(mr):
    fake = object.__new__(mr.GPUModelRunnerFLV2)
    cfg = SimpleNamespace(
        use_v2_model_runner=True,
        compilation_config=SimpleNamespace(cudagraph_mode=mr.CUDAGraphMode.NONE),
    )
    with pytest.raises(ValueError):
        mr.GPUModelRunnerFLV2.__init__(fake, cfg, torch.device("cpu"))


def test_allocate_and_bind_kv_cache(mr, monkeypatch):
    records = {}
    monkeypatch.setattr(
        mr,
        "allocate_ascend_kv_cache",
        lambda *a, **k: ({"a": "va", "b": "vb"}, {"a": "ma"}, ["plane"]),
    )
    monkeypatch.setattr(mr, "get_shared_kv_cache_layers", lambda cfg: {"b": "a"})
    monkeypatch.setattr(
        mr, "bind_kv_cache_to_layers", lambda *a, **k: records.setdefault("bind", a)
    )
    runner = SimpleNamespace(
        vllm_config=SimpleNamespace(
            attention_config=SimpleNamespace(hisparse_config=None)
        ),
        device=torch.device("cpu"),
        kv_cache_config=SimpleNamespace(kv_cache_groups=["g"]),
        cache_config=SimpleNamespace(get_resolved_kv_cache_layout=lambda: "NHD"),
        kernel_block_sizes=[128],
        attn_groups=[["g0"]],
        use_ascend_hybrid_cache=False,
        compilation_config=SimpleNamespace(static_forward_context={"a": 1, "b": 2}),
        model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type="qwen")),
    )
    managers = mr.GPUModelRunnerFLV2._allocate_and_bind_kv_cache(runner)
    assert managers == {"a": "ma", "b": "ma"}
    assert runner._ascend_manager_views is managers
    assert runner._ascend_block_planes == ["plane"]
    assert "bind" in records

    runner.vllm_config.attention_config.hisparse_config = object()
    with pytest.raises(NotImplementedError):
        mr.GPUModelRunnerFLV2._allocate_and_bind_kv_cache(runner)


def _patch_init(mr, monkeypatch, records):
    class _Draft:
        def __init__(self):
            self.draft_attn_layer_names = {"a"}
            self.enable_adaptive_verification = True
            self.pcp_manager = None

        def set_attn(self, *args):
            records["set_attn"] = args

        def init_cudagraph_manager(self, mode):
            records["spec_mode"] = mode

    monkeypatch.setattr(mr, "DraftModelSpeculator", _Draft)
    monkeypatch.setattr(mr, "install_graph_metadata_bridge", lambda *a: records.setdefault("bridge", a))
    monkeypatch.setattr(mr, "get_block_table_width", lambda n, bs, token_alignment=1: n + 1)
    monkeypatch.setattr(mr, "init_attn_backend", lambda *a, **k: ([["g"]], _CgSupport(), [128]))
    monkeypatch.setattr(mr, "maybe_create_adaptive_verification_manager", records["adaptive_fn"])
    monkeypatch.setattr(mr, "get_max_chunk_logits", lambda size: 10)
    monkeypatch.setattr(mr, "AscendBlockTables", lambda **k: records.setdefault("block_tables", k))
    monkeypatch.setattr(mr, "pcp", SimpleNamespace(maybe_build_pcp_manager=records["pcp_fn"]))
    monkeypatch.setattr(mr, "maybe_build_ubatch_runner", lambda *a, **k: "UB")
    monkeypatch.setattr(mr, "initialize_mamba_ssu_backend", lambda *a, **k: records.setdefault("mamba", (a, k)))
    monkeypatch.setattr(mr, "envs", SimpleNamespace(VLLM_USE_BREAKABLE_CUDAGRAPH=True))
    monkeypatch.setattr(mr, "has_compiled_submodule", lambda model: False)
    monkeypatch.setattr(mr, "resolve_adaptive_cudagraph_mode", lambda *a, **k: "ADAPTIVE")
    monkeypatch.setattr(mr, "AscendModelGraphManager", lambda *a, **k: records.setdefault("graph", k))
    monkeypatch.setattr(mr, "FastPrefillHelper", lambda *a: "FP")
    monkeypatch.setattr(mr, "check_attention_cp_compatibility", lambda *a: records.setdefault("cp_check", a))
    monkeypatch.setattr(mr, "get_kv_connector", lambda *a: "KV")
    monkeypatch.setattr(mr, "NO_OP_KV_CONNECTOR", "NOOP")
    monkeypatch.setattr(mr, "get_aux_output_connector", lambda *a: "AUX")






def test_profile_cudagraph_memory_cleanup(mr, monkeypatch):
    monkeypatch.setattr(mr.GPUModelRunner, "profile_cudagraph_memory", lambda self: 7)
    cleared = []
    try:
        import vllm_fl.compilation.graph_params as graph_params

        monkeypatch.setattr(graph_params, "clear_graph_params", lambda: cleared.append(True))
    except Exception:
        pytest.skip("vllm_fl.compilation.graph_params unavailable")

    runner = object.__new__(mr.GPUModelRunnerFLV2)
    runner.compilation_config = SimpleNamespace(cudagraph_mode=mr.CUDAGraphMode.NONE)
    assert mr.GPUModelRunnerFLV2.profile_cudagraph_memory(runner) == 7

    runner.compilation_config = SimpleNamespace(
        cudagraph_mode=mr.CUDAGraphMode.FULL_DECODE_ONLY
    )
    runner._ascend_manager_views = {"a": 1}
    runner._ascend_block_planes = ["p"]
    runner.kv_block_zeroer = "z"
    assert mr.GPUModelRunnerFLV2.profile_cudagraph_memory(runner) == 7
    assert runner._ascend_manager_views == {}
    assert runner._ascend_block_planes == []
    assert runner.kv_block_zeroer is None
    assert cleared == [True]
