# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""G36 tail coverage (814b) for four nearly-complete modules.

Targets the residual uncovered regions left by g4/g5/g26/g27:

  * vllm_fl/ops/fused_moe/fused_moe_utils.py -- the three early-return
    dispatches in ``TritonExpertsFL.apply`` (Ascend grouped, Kunlunxin,
    NVIDIA FlagGems), the NPU ``workspace_shapes`` reservation and
    ``moe_sum``.
  * vllm_fl/attention/utils.py -- ``_gcu_forward_mm_attention`` (delegate +
    packed/unpacked varlen) and the Kunlunxin query-chunked SDPA adapter,
    plus ``maybe_get_vit_flash_attn_backend`` selection.
  * vllm_fl/dispatch/backends/vendor/ascend/worker/model_runner_v2.py --
    ``__init__`` success/policy branches, ``profile_cudagraph_memory`` and
    ``initialize_kv_cache``.
  * vllm_fl/distributed/kv_transfer/flagcx_connector.py -- the pure
    TP/region planning helpers, scheduler clipping/finalisation and the
    worker connection cache.

Everything is CPU-only: kernels, the FlagCX library, ZMQ and vLLM V2
super-classes are replaced with recording fakes.  Nothing touches NPU/triton.
"""

from __future__ import annotations

import contextlib
import importlib
import importlib.util
import pathlib
import sys
import threading
import types
from types import SimpleNamespace

import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# optional module imports
# ---------------------------------------------------------------------------

try:
    import vllm_fl.ops.fused_moe.fused_moe_utils as fmoe
except Exception as exc:  # pragma: no cover - env specific
    fmoe = None
    _FMOE_ERR = exc
else:
    _FMOE_ERR = None

try:
    import vllm_fl.attention.utils as attn_utils
except Exception as exc:  # pragma: no cover - env specific
    attn_utils = None
    _ATTN_ERR = exc
else:
    _ATTN_ERR = None


def _need(module, err, label):
    if module is None:
        pytest.skip(f"{label} unavailable: {err}")


def _install_module(monkeypatch, name, attrs):
    """Install ``name`` into ``sys.modules``; create missing parent packages."""
    parts = name.split(".")
    for i in range(1, len(parts)):
        parent = ".".join(parts[:i])
        if parent not in sys.modules:
            pkg = types.ModuleType(parent)
            pkg.__path__ = []
            monkeypatch.setitem(sys.modules, parent, pkg)
    mod = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(mod, key, value)
    monkeypatch.setitem(sys.modules, name, mod)
    return mod


# ===========================================================================
# vllm_fl/ops/fused_moe/fused_moe_utils.py
# ===========================================================================


def _fm_platform(**attrs):
    base = dict(
        vendor_name="nvidia",
        device_type="cuda",
        is_cuda=lambda: False,
        is_out_of_tree=lambda: False,
        is_cpu=lambda: False,
    )
    base.update(attrs)
    return SimpleNamespace(**base)


def _fm_quant(**attrs):
    base = dict(
        use_fp8_w8a8=False,
        use_int8_w8a8=False,
        use_int8_w8a16=False,
        use_int4_w4a16=False,
        quant_dtype=None,
        weight_quant_dtype=None,
        config_name=lambda dtype: "dummy",
    )
    base.update(attrs)
    return SimpleNamespace(**base)


def _fm_experts(quant=None):
    experts = object.__new__(fmoe.TritonExpertsFL)
    experts._lora_context = None
    experts.quant_config = quant or _fm_quant()
    experts.per_act_token_quant = False
    experts.block_shape = None
    experts.w1_scale = None
    experts.w2_scale = None
    experts.w1_bias = "w1_bias"
    experts.w2_bias = "w2_bias"
    experts.quant_dtype = None
    experts.quantization_emulation = None
    experts.activation_config = SimpleNamespace(clamp_limit=1.5)
    experts.moe_config = SimpleNamespace(max_num_tokens=64)
    return experts


class _Out:
    def __init__(self):
        self.value = None

    def copy_(self, value):
        self.value = value
        return self












# ===========================================================================
# vllm_fl/attention/utils.py
# ===========================================================================


def _fake_flash_module(monkeypatch):
    sentinel = lambda *a, **k: None
    _install_module(
        monkeypatch, "flash_attn", {"__path__": []}
    )
    _install_module(
        monkeypatch,
        "flash_attn.vllm_flash_attn",
        {"flash_attn_varlen_func": sentinel},
    )
    return sentinel


def test_gcu_forward_delegates_for_non_flash_backend(monkeypatch):
    _need(attn_utils, _ATTN_ERR, "attention.utils")
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    self = SimpleNamespace(
        attn_backend=object(),
        _fl_gcu_original_forward_native=lambda *a: "orig",
    )
    out = attn_utils._gcu_forward_mm_attention(
        self, torch.zeros(2, 3, 4, 8), None, None
    )
    assert out == "orig"
    assert AttentionBackendEnum.FLASH_ATTN is not None


def test_gcu_forward_unpacked_builds_seqlens(monkeypatch):
    _need(attn_utils, _ATTN_ERR, "attention.utils")
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    seen = []

    def _flash(**kwargs):
        seen.append(kwargs)
        return torch.zeros_like(kwargs["q"])

    _install_module(
        monkeypatch,
        "flash_attn.vllm_flash_attn",
        {"flash_attn_varlen_func": _flash},
    )
    self = SimpleNamespace(
        attn_backend=AttentionBackendEnum.FLASH_ATTN,
        scale=0.5,
        num_heads=4,
        head_size=8,
        view_qkv_to_4d=lambda q, k, v, b, l, kv: (q, k, v),
    )
    q = torch.zeros(2, 3, 4, 8)
    k = torch.zeros(2, 5, 4, 8)
    out = attn_utils._gcu_forward_mm_attention(self, q, k, k)
    assert tuple(out.shape) == (2, 3, 4, 8)
    assert seen[0]["max_seqlen_q"] == 3
    assert seen[0]["max_seqlen_k"] == 5
    assert seen[0]["causal"] is False


def test_gcu_forward_reshaped_with_cu_seqlens(monkeypatch):
    _need(attn_utils, _ATTN_ERR, "attention.utils")
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    seen = []

    def _flash(**kwargs):
        seen.append(kwargs)
        return torch.zeros_like(kwargs["q"])

    _install_module(
        monkeypatch,
        "flash_attn.vllm_flash_attn",
        {"flash_attn_varlen_func": _flash},
    )
    self = SimpleNamespace(
        attn_backend=AttentionBackendEnum.FLASH_ATTN,
        scale=0.5,
        num_heads=4,
        head_size=8,
        view_qkv_to_4d=lambda q, k, v, b, l, kv: (
            q.reshape(b, l, 4, 8),
            k.reshape(b, l, 4, 8),
            v.reshape(b, l, 4, 8),
        ),
    )
    q = torch.zeros(2, 3, 32)
    cu = torch.tensor([0, 4, 6], dtype=torch.int32)
    out = attn_utils._gcu_forward_mm_attention(
        self, q, q, q, cu_seqlens=cu, max_seqlen=torch.tensor(4)
    )
    assert tuple(out.shape) == (2, 3, 32)
    assert seen[0]["max_seqlen_q"] == 4
    assert seen[0]["cu_seqlens_q"] is cu


def _fake_mm(monkeypatch):
    mm = types.ModuleType(
        "vllm.model_executor.layers.attention.mm_encoder_attention"
    )

    class MMEncoderAttention:
        def forward_native(self, *a, **k):
            return "native"

        def forward_cuda(self, *a, **k):
            return "cuda"

    mm.MMEncoderAttention = MMEncoderAttention
    mm.maybe_get_vit_flash_attn_backend = lambda backend: None
    monkeypatch.setitem(sys.modules, mm.__name__, mm)
    return mm






# ===========================================================================
# vllm_fl/dispatch/backends/vendor/ascend/worker/model_runner_v2.py
# ===========================================================================

_ASCEND_REL = "vllm_fl/dispatch/backends/vendor/ascend"


def _find_repo():
    import os

    env = os.environ.get("FL_REPO_ROOT") or os.environ.get("FL_WORKSPACE")
    cands = [pathlib.Path(env)] if env else []
    here = pathlib.Path(__file__).resolve()
    cands.extend([here.parent, *here.parents, pathlib.Path.cwd()])
    cands.extend(pathlib.Path(p) for p in sys.path if p)
    for base in cands:
        if (base / "vllm_fl" / "dispatch" / "backends").is_dir():
            return base
    return None


def _load_path(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_v2_runner():
    repo = _find_repo()
    if repo is None:
        return None, RuntimeError("cannot locate vllm_fl tree")
    ascend = repo / _ASCEND_REL
    worker = ascend / "worker"
    patches = ascend / "patches"

    pkg = types.ModuleType("g36mrv")
    pkg.__path__ = [str(ascend)]
    sys.modules["g36mrv"] = pkg

    worker_pkg = types.ModuleType("g36mrv.worker")
    worker_pkg.__path__ = [str(worker)]
    sys.modules["g36mrv.worker"] = worker_pkg

    patches_pkg = types.ModuleType("g36mrv.patches")
    patches_pkg.__path__ = [str(patches)]
    sys.modules["g36mrv.patches"] = patches_pkg

    dev = types.ModuleType("g36mrv.patches.ascend_v2_device")
    dev.install_ascend_v2_device_shims = lambda: False
    sys.modules["g36mrv.patches.ascend_v2_device"] = dev

    samp = types.ModuleType("g36mrv.patches.ascend_v2_sampling")
    samp.install_ascend_v2_sampling_shims = lambda: False
    sys.modules["g36mrv.patches.ascend_v2_sampling"] = samp

    bt = types.ModuleType("g36mrv.worker.ascend_v2_block_table")

    class _AscendBlockTables:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    bt.AscendBlockTables = _AscendBlockTables
    sys.modules["g36mrv.worker.ascend_v2_block_table"] = bt

    graph = types.ModuleType("g36mrv.worker.ascend_v2_graph")

    class _AscendModelGraphManager:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def release_graphs(self):
            return None

    graph.AscendModelGraphManager = _AscendModelGraphManager
    graph.install_graph_metadata_bridge = lambda *a, **k: None
    sys.modules["g36mrv.worker.ascend_v2_graph"] = graph

    return _load_path(
        "g36mrv.worker.model_runner_v2", worker / "model_runner_v2.py"
    ), None


try:
    mrv2, _MRV2_ERROR = _load_v2_runner()
except Exception as exc:  # vLLM V2 modules unavailable
    mrv2, _MRV2_ERROR = None, exc


@pytest.fixture()
def mr():
    if mrv2 is None:
        pytest.skip(f"model_runner_v2 unavailable: {_MRV2_ERROR}")
    return mrv2


class _CtxM:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_mrv2_init_success(mr, monkeypatch):
    calls = []
    monkeypatch.setattr(
        mr.GPUModelRunner, "__init__", lambda self, cfg, dev: calls.append((cfg, dev))
    )
    monkeypatch.setattr(mr, "uses_dense_hybrid_cache", lambda mc: False)
    cfg = SimpleNamespace(
        use_v2_model_runner=True,
        compilation_config=SimpleNamespace(cudagraph_mode=mr.CUDAGraphMode.NONE),
        model_config=object(),
        kv_transfer_config=None,
    )
    runner = object.__new__(mr.GPUModelRunnerFLV2)
    mr.GPUModelRunnerFLV2.__init__(runner, cfg, torch.device("npu"))
    assert runner.use_ascend_hybrid_cache is False
    assert runner._ascend_manager_views == {}
    assert runner._ascend_block_planes == []
    assert calls and calls[0][0] is cfg


def test_mrv2_init_rejects_non_npu_device(mr):
    cfg = SimpleNamespace(
        use_v2_model_runner=True,
        compilation_config=SimpleNamespace(cudagraph_mode=mr.CUDAGraphMode.NONE),
    )
    runner = object.__new__(mr.GPUModelRunnerFLV2)
    with pytest.raises(ValueError):
        mr.GPUModelRunnerFLV2.__init__(runner, cfg, torch.device("cpu"))


def test_mrv2_init_rejects_cudagraph_mode(mr, monkeypatch):
    monkeypatch.setattr(mr, "uses_dense_hybrid_cache", lambda mc: False)
    cfg = SimpleNamespace(
        use_v2_model_runner=True,
        compilation_config=SimpleNamespace(cudagraph_mode=object()),
        model_config=object(),
        kv_transfer_config=None,
    )
    runner = object.__new__(mr.GPUModelRunnerFLV2)
    with pytest.raises(NotImplementedError):
        mr.GPUModelRunnerFLV2.__init__(runner, cfg, torch.device("npu"))


def test_mrv2_profile_cudagraph_memory_clears(mr, monkeypatch):
    cleared = []
    _install_module(
        monkeypatch,
        "vllm_fl.compilation.graph_params",
        {"clear_graph_params": lambda: cleared.append(True)},
    )
    monkeypatch.setattr(
        mr.GPUModelRunner, "profile_cudagraph_memory", lambda self: 42
    )

    plain = object.__new__(mr.GPUModelRunnerFLV2)
    plain.compilation_config = SimpleNamespace(cudagraph_mode=mr.CUDAGraphMode.NONE)
    assert mr.GPUModelRunnerFLV2.profile_cudagraph_memory(plain) == 42
    assert cleared == []

    full = object.__new__(mr.GPUModelRunnerFLV2)
    full.compilation_config = SimpleNamespace(
        cudagraph_mode=mr.CUDAGraphMode.FULL_DECODE_ONLY
    )
    full._ascend_manager_views = {"a": 1}
    full._ascend_block_planes = ["p"]
    full.kv_block_zeroer = "z"
    assert mr.GPUModelRunnerFLV2.profile_cudagraph_memory(full) == 42
    assert full._ascend_manager_views == {}
    assert full._ascend_block_planes == []
    assert full.kv_block_zeroer is None
    assert cleared == [True]


def _init_runner(mr, *, is_profiling, encoder_decoder=False, fast_prefill=False):
    init_calls = []
    spec = SimpleNamespace(
        block_size=16,
        uses_slot_mapping=True,
        dcp_sharded=False,
        max_num_blocks_per_req=lambda cfg, ln: 8,
        block_table_token_alignment=1,
    )
    group = SimpleNamespace(kv_cache_spec=spec, layer_names=["l0"])
    kv_cfg = SimpleNamespace(kv_cache_groups=[group], has_mamba_layers=False)

    model_state = SimpleNamespace(
        get_additional_cg_support=lambda: (0, 0),
        num_new_sampled_tokens_per_step=1,
        initialize_kv_cache=lambda *a: init_calls.append(a),
    )
    r = SimpleNamespace(
        parallel_config=SimpleNamespace(cp_kv_cache_interleave_size=1),
        model_state=model_state,
        max_model_len=128,
        is_encoder_decoder=encoder_decoder,
        scheduler_config=SimpleNamespace(max_num_encoder_input_tokens=64),
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(max_source_positions=512)
        ),
        speculator=None,
        jit_warmup_registry=_CtxM(),
        device=torch.device("cpu"),
        vllm_config=SimpleNamespace(
            attention_config=SimpleNamespace(hisparse_config=None),
            mamba_config=object(),
            aux_output_config=SimpleNamespace(enabled=not is_profiling),
        ),
        dcp_size=1,
        dcp_rank=0,
        max_num_reqs=2,
        max_num_tokens=16,
        supports_mm_inputs=False,
        pcp_manager_cls=None,
        decode_query_len=1,
        lora_capture_cases=None,
        vocab_size=100,
        req_states=object(),
        input_buffers=SimpleNamespace(query_start_loc=torch.zeros(1)),
        cache_config=SimpleNamespace(
            use_replayssm=False, kv_sharing_fast_prefill=fast_prefill
        ),
        model=object(),
        adaptive_verification=None,
        compilation_config=SimpleNamespace(
            cudagraph_mode=mr.CUDAGraphMode.NONE,
            resolve_cudagraph_mode_and_sizes=lambda *a, **k: "MODE",
        ),
        kernel_block_sizes=[16],
        attn_groups=[["g"]],
        use_ascend_hybrid_cache=False,
    )
    r._allocate_and_bind_kv_cache = lambda **kw: {
        "l0": SimpleNamespace(device=r.device)
    }
    return r, kv_cfg, init_calls


def _patch_init_deps(mr, monkeypatch, graph_calls):
    class _CG:
        def narrow(self, *a):
            return SimpleNamespace(
                min_cg_support=object(), min_cg_attn_backend=object()
            )

    class _Graph:
        def __init__(self, *a, **k):
            graph_calls.append((a, k))

        def release_graphs(self):
            return None

    monkeypatch.setattr(mr, "install_graph_metadata_bridge", lambda *a, **k: None)
    monkeypatch.setattr(
        mr, "init_attn_backend", lambda *a, **k: ([["g"]], _CG(), [16])
    )
    monkeypatch.setattr(
        mr, "maybe_create_adaptive_verification_manager", lambda **k: None
    )
    monkeypatch.setattr(
        mr, "AscendBlockTables", lambda **k: SimpleNamespace(kwargs=k)
    )
    monkeypatch.setattr(
        mr.pcp, "maybe_build_pcp_manager", lambda *a, **k: None
    )
    monkeypatch.setattr(mr, "maybe_build_ubatch_runner", lambda *a, **k: None)
    monkeypatch.setattr(mr, "initialize_mamba_ssu_backend", lambda *a, **k: None)
    monkeypatch.setattr(
        mr, "envs", SimpleNamespace(VLLM_USE_BREAKABLE_CUDAGRAPH=False)
    )
    monkeypatch.setattr(mr, "has_compiled_submodule", lambda m: False)
    monkeypatch.setattr(mr, "AscendModelGraphManager", _Graph)
    monkeypatch.setattr(mr, "check_attention_cp_compatibility", lambda *a: None)
    monkeypatch.setattr(mr, "NO_OP_KV_CONNECTOR", "NOOP")
    monkeypatch.setattr(mr, "get_kv_connector", lambda cfg, caches: "KC")
    monkeypatch.setattr(
        mr, "get_aux_output_connector", lambda *a: "AUX"
    )
    monkeypatch.setattr(
        mr, "FastPrefillHelper", lambda *a: SimpleNamespace(tokens=a[1])
    )
    monkeypatch.setattr(
        mr, "get_block_table_width", lambda n, b, token_alignment=1: n
    )






# ===========================================================================
# vllm_fl/distributed/kv_transfer/flagcx_connector.py
# ===========================================================================

CONNECTOR = "vllm_fl.distributed.kv_transfer.flagcx_connector"


class _FakeFlagcxLib:
    def __init__(self, recorder):
        self.recorder = recorder

    def flagcxP2pEngineCreate(self):
        self.recorder.append("create")
        return "ENG"


@pytest.fixture()
def fcx(monkeypatch):
    recorder = []
    wrapper = types.ModuleType("plugin.interservice.flagcx_wrapper")
    wrapper.FLAGCXLibrary = lambda path=None: _FakeFlagcxLib(recorder)
    _install_module(monkeypatch, "plugin", {"__path__": []})
    inter = _install_module(monkeypatch, "plugin.interservice", {"__path__": []})
    inter.flagcx_wrapper = wrapper
    monkeypatch.setitem(
        sys.modules, "plugin.interservice.flagcx_wrapper", wrapper
    )
    sys.modules.pop(CONNECTOR, None)
    try:
        module = importlib.import_module(CONNECTOR)
    except Exception as exc:  # pragma: no cover - environment guard
        sys.modules.pop(CONNECTOR, None)
        pytest.skip(f"flagcx connector not importable: {exc!r}")
    yield module
    sys.modules.pop(CONNECTOR, None)


def test_fcx_metadata_add_new_req_both_roles(fcx):
    md = fcx.FlagCXConnectorMetadata()
    md.add_new_req(
        "r1",
        [[1, 2]],
        {
            "transfer_id": "t1",
            "remote_host": "h",
            "remote_port": 7,
            "remote_tp_size": 2,
        },
    )
    assert md.reqs_to_recv["r1"].remote_tp_size == 2
    assert md.reqs_to_recv["r1"].remote_host == "h"
    md.add_new_req("r2", [[3]], {"transfer_id": "t2"}, load_remote_cache=False)
    assert md.reqs_to_send["r2"] == ("t2", [[3]])


def test_fcx_scheduler_request_finished_remote_prefill(fcx):
    s = object.__new__(fcx.FlagCXConnectorScheduler)
    s.kv_role = "kv_consumer"
    s._reqs_need_recv = {}
    params = {"transfer_id": "t", "do_remote_prefill": True}
    req = SimpleNamespace(request_id="r", kv_transfer_params=params)
    assert s.request_finished(req, ([],)) == (False, None)
    assert params["do_remote_prefill"] is False
    assert s._reqs_need_recv["r"] == (req, [])

    empty = SimpleNamespace(request_id="x", kv_transfer_params=None)
    assert s.request_finished(empty, ()) == (False, None)


def test_fcx_scheduler_sw_clipping_and_mamba_count(fcx):
    s = object.__new__(fcx.FlagCXConnectorScheduler)
    s._is_hma_required = True
    s.blocks_per_sw = [1, 0]
    assert s.get_sw_clipped_blocks([[1, 2, 3], [4, 5]]) == [[3], [4, 5]]
    assert s.get_sw_clipped_blocks([]) == []
    s._is_hma_required = False
    assert s.get_sw_clipped_blocks([[1, 2]]) == [[1, 2]]

    s._has_mamba = True
    assert s._get_remote_prefill_token_count(5) == 4
    assert s._get_remote_prefill_token_count(1) == 1
    s._has_mamba = False
    assert s._get_remote_prefill_token_count(5) == 5


def test_fcx_group_concurrent_contiguous(fcx):
    assert fcx.group_concurrent_contiguous([], []) == ([], [])
    src, dst = fcx.group_concurrent_contiguous([1, 2, 5], [10, 11, 14])
    assert src == [[1, 2], [5]]
    assert dst == [[10, 11], [14]]
    src2, dst2 = fcx.group_concurrent_contiguous([1, 2, 3], [4, 6, 7])
    assert src2 == [[1], [2, 3]]
    assert dst2 == [[4], [6, 7]]


def test_fcx_get_tp_ratio(fcx):
    assert fcx._get_tp_ratio(4, 2) == 2
    assert fcx._get_tp_ratio(2, 4) == -2
    assert fcx._get_tp_ratio(3, 3) == 1
    with pytest.raises(AssertionError):
        fcx._get_tp_ratio(3, 2)
    with pytest.raises(AssertionError):
        fcx._get_tp_ratio(2, 3)


def test_fcx_expand_transfer_regions(fcx):
    split = fcx._expand_transfer_regions(
        base_addrs=[100],
        block_lens=[10],
        kv_block_lens=[4],
        layer_names=["a"],
        layer_indices=[0],
        is_kv_layout_blocks_first=True,
    )
    assert [r.base_addr for r in split] == [100, 104]
    assert all(r.block_len == 10 for r in split)

    whole = fcx._expand_transfer_regions(
        base_addrs=[100],
        block_lens=[10],
        kv_block_lens=[4],
        layer_names=["a"],
        layer_indices=[0],
        is_kv_layout_blocks_first=False,
    )
    assert len(whole) == 1
    with pytest.raises(AssertionError):
        fcx._expand_transfer_regions(
            base_addrs=[100, 200],
            block_lens=[10],
            kv_block_lens=[4],
            layer_names=["a"],
            layer_indices=[0],
            is_kv_layout_blocks_first=False,
        )


def test_fcx_compute_sender_transfer_plan_matrix(fcx):
    plan = fcx._compute_sender_transfer_plan
    assert plan(0, 2, 0, 2, 64, 64, False) == (True, 0, 0, 64)
    assert plan(2, 4, 0, 2, 64, 128, True) == (True, 0, 0, 64)
    assert plan(1, 4, 0, 2, 64, 128, True) == (False, 0, 0, 64)
    assert plan(1, 4, 0, 2, 64, 128, False) == (True, 0, 64, 64)
    assert plan(0, 2, 0, 4, 128, 64, True) == (True, 0, 0, 128)
    assert plan(0, 2, 1, 4, 128, 64, False) == (True, 64, 0, 64)


def test_fcx_can_coalesce(fcx):
    assert fcx._can_coalesce_block_transfers(64, 64, 0, 0, 64) is True
    assert fcx._can_coalesce_block_transfers(64, 64, 4, 0, 64) is False
    assert fcx._can_coalesce_block_transfers(64, 64, 0, 4, 64) is False
    assert fcx._can_coalesce_block_transfers(64, 64, 0, 0, 32) is False
    assert fcx._can_coalesce_block_transfers(64, 128, 0, 0, 64) is False


def _region(fcx, name="a", index=0, base=0, block=64, kv=64, group=0):
    return fcx.TransferRegion(
        layer_name=name,
        layer_index=index,
        base_addr=base,
        block_len=block,
        kv_block_len=kv,
        group_index=group,
    )


def test_fcx_validate_asymmetric_region_lengths(fcx):
    v = fcx._validate_asymmetric_region_lengths
    assert v([_region(fcx)], [], 2, 2, False) is not None
    assert v([_region(fcx)], [_region(fcx)], 2, 2, True) is None
    msg = v([_region(fcx, kv=64)], [_region(fcx, kv=32)], 2, 2, False)
    assert "homogeneous TP" in msg
    assert v([_region(fcx, kv=64)], [_region(fcx, kv=64)], 2, 2, False) is None
    up = v([_region(fcx, kv=64)], [_region(fcx, kv=128)], 4, 2, False)
    assert up is None
    assert "destination" in v([_region(fcx, kv=64)], [_region(fcx, kv=64)], 4, 2, False)
    down = v([_region(fcx, kv=128)], [_region(fcx, kv=64)], 2, 4, False)
    assert down is None
    assert "source" in v([_region(fcx, kv=64)], [_region(fcx, kv=64)], 2, 4, False)


def test_fcx_align_transfer_regions(fcx):
    a = _region(fcx, name="a", index=0)
    b = _region(fcx, name="b", index=1)
    local = [a, b]
    remote = [_region(fcx, name="b", index=1), _region(fcx, name="a", index=0)]
    la, ra, err = fcx._align_transfer_regions(local, remote)
    assert err is None
    assert [r.layer_name for r in la] == ["a", "b"]
    assert [r.layer_name for r in ra] == ["a", "b"]

    _, _, err = fcx._align_transfer_regions([a], [])
    assert "no matching consumer" in err
    _, _, err = fcx._align_transfer_regions(
        [a], [_region(fcx, name="a", index=5)]
    )
    assert "layer index mismatch" in err
    _, _, err = fcx._align_transfer_regions(
        [a], [_region(fcx, name="a", index=0, group=3)]
    )
    assert "group index mismatch" in err


def test_fcx_side_channel_port(fcx, monkeypatch):
    cfg = SimpleNamespace(
        parallel_config=SimpleNamespace(
            data_parallel_rank=2, tensor_parallel_size=4
        )
    )
    monkeypatch.setenv("FLAGCX_BOOTSTRAP_PORT", "9000")
    assert fcx._get_side_channel_port(cfg) == 9008


def test_fcx_worker_get_conn_caches(fcx):
    w = object.__new__(fcx.FlagCXConnectorWorker)
    w.conn_lock = threading.Lock()
    w.session_conns = {}
    calls = []
    w.flagcx = SimpleNamespace(
        flagcxP2pGetConn=lambda eng, session: calls.append(session) or ("conn", session)
    )
    w.engine = "E"
    first = w._get_conn("host:1")
    second = w._get_conn("host:1")
    assert first == second == ("conn", "host:1")
    assert calls == ["host:1"]


def test_fcx_worker_logical_to_kernel_identity(fcx):
    w = object.__new__(fcx.FlagCXConnectorWorker)
    w._physical_blocks_per_logical_kv_block = 1
    ids = [[1, 2], [3]]
    assert w._logical_to_kernel_block_ids(ids) is ids
