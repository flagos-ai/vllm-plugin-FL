# Copyright (c) 2026 BAAI. All rights reserved.

"""CPU tests for the metax vendor MLA common helpers.

``vllm_fl/dispatch/backends/vendor/metax/impl/attention/mla/common.py``
is a metax (MACA) attention backend.  The whole module cannot import on
CPU machines (it pulls ``flash_attn``, ``flashinfer`` and the full vLLM
attention API), so the external boundaries are stubbed and the *real
source file* is loaded through ``spec_from_file_location`` — every line
under test executes unmodified:

* ``reorg_kvcache`` — the CP-gather cache reorganization: interleaved
  padded chunks per rank are un-padded and re-concatenated into
  sequence-major layout, verified against an independent index-based
  oracle over labelled tokens;
* ``MLACommonBackend`` metadata — name, kv-cache shape and stride
  permutations, head-size gate (only 576), the MLA flag;
* ``MLACommonMetadata.__post_init__`` — unsupported head dimensions
  are rejected;
* the three prefill-path gates — ``use_cudnn_prefill`` and
  ``use_trtllm_ragged_deepseek_prefill`` are hard-coded False on Maca
  (the log line fires once), ``use_flashinfer_prefill`` honours every
  config flag and the platform capability check.

Kernel launches are never reached: only the pure-logic surface above
is driven.
"""

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

_STUB_ROOTS = ("vllm", "vllm_fl", "flag_gems", "flash_attn", "flashinfer")
_MISSING = object()


@pytest.fixture(autouse=True)
def _fl_sys_modules():
    """Snapshot sys.modules around the stub install and restore exactly."""
    saved = {}
    for name in list(sys.modules):
        if name.split(".")[0] in _STUB_ROOTS:
            saved[name] = sys.modules.get(name, _MISSING)
    try:
        yield saved
    finally:
        for name in [
            name
            for name in list(sys.modules)
            if name.split(".")[0] in _STUB_ROOTS and name not in saved
        ]:
            del sys.modules[name]
        for name, module in saved.items():
            if module is _MISSING:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def _leaf(modules, name, **attrs):
    """Install ``name`` (and its parents) as stub modules with attrs."""
    parts = name.split(".")
    for length in range(1, len(parts) + 1):
        partial = ".".join(parts[:length])
        if partial not in modules:
            module = ModuleType(partial)
            module.__path__ = []
            modules[partial] = module
    for length in range(1, len(parts)):
        setattr(
            modules[".".join(parts[:length])],
            parts[length],
            modules[".".join(parts[: length + 1])],
        )
    for key, value in attrs.items():
        setattr(modules[name], key, value)
    return modules[name]


class _OnceLogger:
    """Logger with the *_once helpers vLLM loggers expose."""

    def __init__(self):
        self.records = []

    def info_once(self, msg, *args):
        self.records.append(("info", msg))

    def warning_once(self, msg, *args):
        self.records.append(("warning", msg))

    def __getattr__(self, item):
        import logging

        return getattr(logging.getLogger("stub"), item)


@pytest.fixture
def mla_common(monkeypatch):
    """Load the real metax mla/common.py under the stub vllm graph.

    Returns (module, once_logger, platform_stub).
    """
    modules = {}
    once = _OnceLogger()

    platform_stub = _leaf(
        modules,
        "vllm.platforms",
        current_platform=SimpleNamespace(
            is_device_capability_family=lambda family: False,
            vendor_name="metax",
        ),
    )

    # -- the names common.py binds at module import -------------------------
    _leaf(
        modules,
        "vllm",
        envs=SimpleNamespace(),
        _custom_ops=SimpleNamespace(),
    )
    _leaf(modules, "vllm.logger", init_logger=lambda name: once)
    _leaf(
        modules,
        "vllm.config",
        VllmConfig=type("VllmConfig", (), {"__init__": None}),
        get_current_vllm_config=lambda: None,
    )
    _leaf(modules, "vllm.distributed.parallel_state", get_dcp_group=lambda: None)
    _leaf(
        modules,
        "vllm.utils.math_utils",
        cdiv=lambda a, b: -(-a // b),
        round_down=lambda v, m: v,
    )
    _leaf(modules, "vllm.platforms")
    _leaf(modules, "vllm.v1.attention.backend")
    _leaf(modules, "vllm.v1.attention.backends.utils")
    _leaf(modules, "vllm.v1.kv_cache_interface")
    _leaf(
        modules,
        "vllm.model_executor.layers.attention.mla_attention",
        get_mla_dims=lambda *a: (1, 2, 3),
    )
    _leaf(
        modules, "vllm.v1.attention.ops.common", cp_lse_ag_out_rs=lambda *a, **k: None
    )
    _leaf(modules, "vllm.model_executor.layers.linear", LinearBase=object)
    _leaf(
        modules,
        "vllm.model_executor.determinism.batch_invariant",
        init_batch_invariance=lambda *a: None,
    )
    _leaf(
        modules,
        "vllm_fl.dispatch.backends.vendor.metax.impl.attention.ops.merge_attn_states",
        merge_attn_states=lambda *a, **k: None,
    )
    _leaf(
        modules,
        "vllm_fl.dispatch.backends.vendor.metax.impl.attention.utils.fa_utils",
        get_flash_attn_version=lambda *a, **k: 2,
    )
    # flash_attn / flashinfer external boundaries.
    _leaf(
        modules,
        "flash_attn",
        flash_attn_varlen_func=lambda *a, **k: None,
    )
    _leaf(
        modules,
        "flashinfer",
        BatchPrefillWithRaggedKVCacheWrapper=type(
            "BatchPrefillWithRaggedKVCacheWrapper", (), {"__init__": None}
        ),
    )

    # AttentionBackend abstract surface: the class under test subclasses it.
    class _AttentionBackend:
        @staticmethod
        def get_name():
            raise NotImplementedError

        @classmethod
        def get_supported_head_sizes(cls):
            return []

        @classmethod
        def supports_head_size(cls, head_size):
            supported = cls.get_supported_head_sizes()
            return (not supported) or head_size in supported

    backend_mod = modules["vllm.v1.attention.backend"]
    backend_mod.AttentionBackend = _AttentionBackend
    backend_mod.AttentionLayer = type("AttentionLayer", (), {"__init__": None})
    backend_mod.MLAAttentionImpl = type("MLAAttentionImpl", (), {"__init__": None})
    from typing import Generic as _Generic, TypeVar as _TypeVar

    _BuilderT = _TypeVar("_BuilderT")

    class _AttentionMetadataBuilder(_Generic[_BuilderT]):
        pass

    backend_mod.AttentionMetadataBuilder = _AttentionMetadataBuilder
    backend_mod._M = _BuilderT

    # MLAAttentionImpl must itself be a subscriptable generic (the file
    # derives MLACommonBaseImpl(MLAAttentionImpl[A], Generic[A])).
    _ImplT = _TypeVar("_ImplT")

    class _MLAAttentionImpl(_Generic[_ImplT]):
        pass

    backend_mod.MLAAttentionImpl = _MLAAttentionImpl
    backend_mod.AttentionImpl = object
    backend_mod.AttentionType = SimpleNamespace(DECODER="decoder", ENCODER="encoder")

    # Names bound from the other vllm leaves.
    modules["vllm.model_executor.layers.linear"].ColumnParallelLinear = type(
        "ColumnParallelLinear", (), {"__init__": None}
    )
    modules["vllm.model_executor.layers.linear"].UnquantizedLinearMethod = type(
        "UnquantizedLinearMethod", (), {"__init__": None}
    )
    modules[
        "vllm.model_executor.determinism.batch_invariant"
    ]._batch_invariant_MODE = "deterministic"
    utils_mod = modules["vllm.v1.attention.backends.utils"]
    for nm in (
        "CommonAttentionMetadata",
        "get_dcp_local_seq_lens",
        "get_per_layer_parameters",
        "infer_global_hyperparameters",
        "split_decodes_and_prefills",
    ):
        setattr(utils_mod, nm, lambda *a, **k: None)
    modules["vllm.v1.kv_cache_interface"].AttentionSpec = type(
        "AttentionSpec", (), {"__init__": None}
    )

    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    source = (
        Path(__file__).resolve().parents[4]
        / "vllm_fl/dispatch/backends/vendor/metax/impl/attention/mla/common.py"
    )
    import importlib.util

    # Load under the production dotted name so the file's relative imports
    # (..ops.merge_attn_states, ..utils.fa_utils) resolve against the stub
    # package graph installed above.
    fq_name = "vllm_fl.dispatch.backends.vendor.metax.impl.attention.mla.common"
    spec = importlib.util.spec_from_file_location(fq_name, source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[fq_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(fq_name, None)
    return module, once, platform_stub


# ---------------------------------------------------------------------------
# MLACommonBackend metadata surface
# ---------------------------------------------------------------------------


class TestMLACommonBackend:
    """Static metadata the dispatch manager reads from the backend."""

    def test_name_is_triton_mla(self, mla_common):
        module, _, _ = mla_common
        assert module.MLACommonBackend.get_name() == "TRITON_MLA"

    def test_builder_cls_is_the_common_builder(self, mla_common):
        module, _, _ = mla_common
        assert (
            module.MLACommonBackend.get_builder_cls() is module.MLACommonMetadataBuilder
        )

    def test_kv_cache_shape_drops_head_dimension(self, mla_common):
        """MLA latent cache is (blocks, block_size, head_size) — no heads."""
        module, _, _ = mla_common
        shape = module.MLACommonBackend.get_kv_cache_shape(
            num_blocks=128, block_size=64, num_kv_heads=1, head_size=576
        )
        assert shape == (128, 64, 576)

    def test_stride_orders(self, mla_common):
        """Stride permutation maps shape to memory layout."""
        module, _, _ = mla_common
        assert module.MLACommonBackend.get_kv_cache_stride_order() == (0, 1, 2)
        assert module.MLACommonBackend.get_kv_cache_stride_order(
            include_num_layers_dimension=True
        ) == (1, 0, 2, 3)

    def test_supported_head_sizes_only_576(self, mla_common):
        module, _, _ = mla_common
        assert module.MLACommonBackend.get_supported_head_sizes() == [576]
        assert module.MLACommonBackend.supports_head_size(576) is True
        assert module.MLACommonBackend.supports_head_size(128) is False

    def test_backend_is_mla(self, mla_common):
        module, _, _ = mla_common
        assert module.MLACommonBackend.is_mla() is True


class TestMLACommonMetadataPostInit:
    """__post_init__ rejects unsupported head dimensions."""

    def _metadata(self, module, head_dim):
        return module.MLACommonMetadata(
            num_reqs=1,
            max_query_len=1,
            max_seq_len=4,
            num_actual_tokens=4,
            query_start_loc=torch.tensor([0, 1]),
            slot_mapping=torch.zeros(1, dtype=torch.long),
            num_decodes=1,
            num_decode_tokens=1,
            num_prefills=0,
            head_dim=head_dim,
        )

    def test_unsupported_head_dim_raises(self, mla_common):
        module, _, _ = mla_common
        with pytest.raises(ValueError, match="Head dimension 128 is not supported"):
            self._metadata(module, 128)

    def test_supported_head_dim_passes(self, mla_common):
        module, _, _ = mla_common
        metadata = self._metadata(module, 576)
        assert metadata.head_dim == 576

    def test_none_head_dim_passes(self, mla_common):
        module, _, _ = mla_common
        metadata = self._metadata(module, None)
        assert metadata.head_dim is None


# ---------------------------------------------------------------------------
# Prefill-path gates
# ---------------------------------------------------------------------------


class TestPrefillGates:
    """Maca hard-disables cudnn/trtllm prefill; flashinfer follows config."""

    def test_cudnn_prefill_always_false(self, mla_common):
        module, once, _ = mla_common
        assert module.use_cudnn_prefill() is False
        assert any("cudnn prefill is not supported" in r[1] for r in once.records)

    def test_trtllm_prefill_always_false(self, mla_common):
        module, once, _ = mla_common
        assert module.use_trtllm_ragged_deepseek_prefill() is False
        assert any("TRT-LLM ragged" in r[1] for r in once.records)

    def test_flashinfer_prefill_requires_capability_100(self, mla_common, monkeypatch):
        module, _, platform_stub = mla_common
        # Config with every gate open — only the capability check remains.
        import sys as _sys

        import vllm.config as _unused  # noqa: F401  (ensures stub graph alive)

        cfg = SimpleNamespace(
            attention_config=SimpleNamespace(
                disable_flashinfer_prefill=False,
                use_cudnn_prefill=False,
                use_trtllm_ragged_deepseek_prefill=False,
            )
        )
        config_mod = _sys.modules["vllm.config"]
        monkeypatch.setattr(
            config_mod, "get_current_vllm_config", lambda: cfg, raising=False
        )
        platform_stub.current_platform.is_device_capability_family = lambda family: (
            family == 100
        )
        assert module.use_flashinfer_prefill() is True
        platform_stub.current_platform.is_device_capability_family = lambda family: (
            False
        )
        assert module.use_flashinfer_prefill() is False

    def test_flashinfer_prefill_disabled_by_config(self, mla_common, monkeypatch):
        module, _, platform_stub = mla_common
        import sys as _sys

        cfg = SimpleNamespace(
            attention_config=SimpleNamespace(
                disable_flashinfer_prefill=True,
                use_cudnn_prefill=False,
                use_trtllm_ragged_deepseek_prefill=False,
            )
        )
        monkeypatch.setattr(
            _sys.modules["vllm.config"],
            "get_current_vllm_config",
            lambda: cfg,
            raising=False,
        )
        platform_stub.current_platform.is_device_capability_family = lambda family: True
        assert module.use_flashinfer_prefill() is False


# ---------------------------------------------------------------------------
# reorg_kvcache — CP-gather cache reorganization with an independent oracle
# ---------------------------------------------------------------------------


def _oracle_reorg(kv_c, k_pe, padded_lens, context_lens, toks):
    """Independent oracle: emit kept tokens per (sequence, rank) in order."""
    kept_c, kept_pe = [], []
    src = 0
    for padded_len, per_rank in zip(padded_lens, context_lens):
        for rank, ctx in enumerate(per_rank):
            kept = max(0, ctx)  # chunk_idx=0, chunk_size >= ctx: no clamping
            if kept:
                base = rank * toks + src
                kept_c.append(kv_c[base : base + kept])
                kept_pe.append(k_pe[base : base + kept])
        src += padded_len
    return torch.cat(kept_c), torch.cat(kept_pe)


class TestReorgKvCache:
    """reorg_kvcache against a labelled-token oracle."""

    def test_two_sequences_two_ranks(self, mla_common):
        """Interleaved padded chunks unpad into sequence-major layout."""
        module, _, _ = mla_common
        toks = 8  # per-rank token capacity
        total = 4 * toks
        # Labels make every token unique: t = seq*100 + position.
        kv_c = torch.arange(total, dtype=torch.float32).reshape(total, 1)
        k_pe = (torch.arange(total, dtype=torch.float32) + 1000).reshape(total, 1)

        # Sequence 0: ranks see 3 and 5 context tokens (padded chunk = 5).
        # Sequence 1: ranks see 2 and 0 context tokens (padded chunk = 2).
        padded_lens = [5, 2]
        context_lens = [[3, 5], [2, 0]]

        got_c, got_pe = module.reorg_kvcache(
            kv_c,
            k_pe,
            padded_lens,
            context_lens,
            sum_seq_len=10,
            # Per-sequence total kept across ranks: max(3+5, 2+0) = 8.
            max_seq_len=8,
            chunk_size=1024,
            chunk_idx=0,
            toks=toks,
        )
        exp_c, exp_pe = _oracle_reorg(kv_c, k_pe, padded_lens, context_lens, toks)
        assert torch.equal(got_c, exp_c)
        assert torch.equal(got_pe, exp_pe)
        # Layout: all of sequence 0 (rank0's 3 then rank1's 5), then
        # sequence 1 (rank0's 2; rank1 contributed nothing).
        # Sequence 0: rank0's tokens [0:3], rank1's [8:13]. Sequence 1
        # starts after seq0's padded chunk: src_token_idx=5, rank0 keeps
        # [5:7] of rank0's region.
        assert got_c.flatten().tolist() == [0, 1, 2, 8, 9, 10, 11, 12, 5, 6]
        assert got_pe.flatten().tolist() == [
            1000,
            1001,
            1002,
            1008,
            1009,
            1010,
            1011,
            1012,
            1005,
            1006,
        ]

    def test_zero_context_ranks_skipped(self, mla_common):
        """Ranks with zero context contribute nothing."""
        module, _, _ = mla_common
        toks = 4
        kv_c = torch.arange(2 * toks, dtype=torch.float32).reshape(-1, 1)
        k_pe = kv_c + 100
        padded_lens = [4]
        context_lens = [[0, 3]]  # rank 0 has no context on this chunk
        got_c, got_pe = module.reorg_kvcache(
            kv_c,
            k_pe,
            padded_lens,
            context_lens,
            sum_seq_len=3,
            max_seq_len=3,
            chunk_size=1024,
            chunk_idx=0,
            toks=toks,
        )
        assert got_c.flatten().tolist() == [4, 5, 6]
        assert got_pe.flatten().tolist() == [104, 105, 106]

    def test_chunk_index_advances_the_window(self, mla_common):
        """chunk_idx*k offsets the per-rank context: only the tail remains."""
        module, _, _ = mla_common
        toks = 4
        kv_c = torch.arange(2 * toks, dtype=torch.float32).reshape(-1, 1)
        k_pe = kv_c + 100
        # rank context 5 with chunk_size 3 and chunk_idx 1: keep 5-3=2.
        padded_lens = [4]
        context_lens = [[5]]
        got_c, _ = module.reorg_kvcache(
            kv_c,
            k_pe,
            padded_lens,
            context_lens,
            sum_seq_len=2,
            max_seq_len=2,
            chunk_size=3,
            chunk_idx=1,
            toks=toks,
        )
        # src starts at 0; rank0 window is [0:2] of the rank's region.
        assert got_c.flatten().tolist() == [0, 1]

    def test_clamped_to_padded_chunk_len(self, mla_common):
        """A rank cannot keep more than its padded chunk length."""
        module, _, _ = mla_common
        toks = 4
        kv_c = torch.arange(2 * toks, dtype=torch.float32).reshape(-1, 1)
        k_pe = kv_c + 100
        padded_lens = [2]
        context_lens = [[7]]  # far larger than the padded chunk
        got_c, _ = module.reorg_kvcache(
            kv_c,
            k_pe,
            padded_lens,
            context_lens,
            sum_seq_len=2,
            max_seq_len=2,
            chunk_size=1024,
            chunk_idx=0,
            toks=toks,
        )
        assert got_c.flatten().tolist() == [0, 1]
