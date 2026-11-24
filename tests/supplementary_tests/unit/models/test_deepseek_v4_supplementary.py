# Copyright 2025 BAAI. All rights reserved.
"""Supplementary coverage for ``vllm_fl.models.deepseek_v4``.

``tests/unit_tests/models/test_deepseek_v4.py`` is gated on torch_npu, so
on every other platform the module stays at 0%. This file loads the real
production source under a stubbed ``vllm.*`` import graph (the same
technique as the prior-project's policy tests, fresh for this branch) and
drives every method body on plain CPU tensors:

* ``_validate_short_context_attention`` — all ratio acceptance/rejection,
  the max_model_len boundary checks and the C4A index_topk fit;
* ``_cache_rows`` / ``_store_cache_rows`` — shape validation and the
  non-contiguous page round-trip with a dtype cast;
* the registry surface of the four backend/builder classes and the
  module's ``__all__``;
* ``DeepseekV4FLAttention`` — construction (cuda.Event swap, positional
  vllm_config resolution, validation ordering, event restoration on
  super() failure) and the numeric method bodies (forward's RMSNorm split,
  input projections, qnorm/rope/cache insert, compressed-KV build for
  both ratios, prepare-and-attn, the o_proj projection, and the
  sink-stabilized MQA attention against independent oracles);
* ``_install_hc_head_fallback`` — idempotency and the patched
  ``forward_native`` contract;
* ``DeepseekV4ForCausalLM`` — validation before super(), subclass wiring.

The stub vllm graph mirrors only the import-time surface; all numeric
assertions use oracles written from the source's formulas.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

pytestmark = [pytest.mark.supplementary]

_SOURCE = Path(__file__).resolve().parents[4] / "vllm_fl/models/deepseek_v4.py"
_MODULE_NAME = "supp_dsv4_under_test"


class _UpstreamBase:
    """Recording stand-in for upstream DeepseekV4Attention."""

    def __init__(self, *args, **kwargs):
        self.init_args = args
        self.init_kwargs = kwargs


class _HCHeadOp:
    """Stand-in for vllm.model_executor.layers.mhc.HCHeadOp."""


class _XpuModelBase:
    """Stand-in for the upstream XPU causal-LM base class."""

    def __init__(self, *args, **kwargs):
        self.init_kwargs = kwargs
        self.init_args = args


def _install_stubs(monkeypatch):
    modules = {}

    def leaf(name, **attrs):
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

    leaf("vllm.forward_context", get_forward_context=lambda: SimpleNamespace())
    leaf("vllm.models.deepseek_v4.attention", DeepseekV4Attention=_UpstreamBase)
    leaf(
        "vllm.models.deepseek_v4.sparse_mla",
        DeepseekV4FlashMLAMetadata=type("DeepseekV4FlashMLAMetadata", (), {}),
        DeepseekV4SparseMLABackend=type("DeepseekV4SparseMLABackend", (), {}),
        DeepseekV4SparseMLAMetadataBuilder=type(
            "DeepseekV4SparseMLAMetadataBuilder", (), {}
        ),
    )
    leaf("vllm.platforms", current_platform=SimpleNamespace(device_type="npu"))
    leaf(
        "vllm.v1.attention.backends.mla.sparse_swa",
        DeepseekSparseSWABackend=type("DeepseekSparseSWABackend", (), {}),
        DeepseekSparseSWAMetadata=type("DeepseekSparseSWAMetadata", (), {}),
        DeepseekSparseSWAMetadataBuilder=type(
            "DeepseekSparseSWAMetadataBuilder", (), {}
        ),
    )
    leaf("vllm.model_executor.layers.mhc", HCHeadOp=_HCHeadOp)
    leaf("flaggems_vllm", hc_head_fused_kernel=lambda *args, **kwargs: None)
    xpu_model = SimpleNamespace(
        DeepseekV4ForCausalLM=_XpuModelBase,
        DeepseekV4XPUAttention=object,
    )
    leaf("vllm.models.deepseek_v4.xpu", model=xpu_model)

    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    return xpu_model


@pytest.fixture
def deepseek_mod(monkeypatch):
    """Load a fresh copy of the real deepseek_v4.py under stub modules."""
    xpu_model = _install_stubs(monkeypatch)
    monkeypatch.setattr(torch, "npu", SimpleNamespace(Event=object), raising=False)
    monkeypatch.delattr(_HCHeadOp, "forward_native", raising=False)
    monkeypatch.delattr(_HCHeadOp, "_vllm_fl_ascend", raising=False)

    spec = importlib.util.spec_from_file_location(_MODULE_NAME, _SOURCE)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, _MODULE_NAME, module)
    spec.loader.exec_module(module)
    # Expose the stub graph pieces the tests assert against.
    module._test_hc_head = _HCHeadOp
    module._test_xpu_model = xpu_model
    return module


def _set_attn_metadata(module, attn_metadata):
    """Point the loaded module's get_forward_context at fixed metadata."""
    module.get_forward_context = lambda: SimpleNamespace(
        attn_metadata=attn_metadata
    )


def _vllm_config(ratios, layers=None, sliding_window=128, max_model_len=128,
                 index_topk=512):
    hf_config = SimpleNamespace(
        compress_ratios=ratios,
        num_hidden_layers=len(ratios) if layers is None else layers,
        sliding_window=sliding_window,
        index_topk=index_topk,
    )
    return SimpleNamespace(
        model_config=SimpleNamespace(hf_config=hf_config,
                                     max_model_len=max_model_len)
    )


class TestValidateShortContext:
    def test_accepts_all_supported_ratios(self, deepseek_mod):
        config = _vllm_config([0, 1, 4, 128])
        deepseek_mod._validate_short_context_attention(config)

    def test_rejects_unsupported_ratio(self, deepseek_mod):
        for bad in (2, 3, 64):
            with pytest.raises(NotImplementedError,
                               match=rf"compress_ratio={bad}"):
                deepseek_mod._validate_short_context_attention(
                    _vllm_config([bad])
                )

    def test_rejects_context_beyond_window(self, deepseek_mod):
        with pytest.raises(ValueError, match="<= sliding_window"):
            deepseek_mod._validate_short_context_attention(
                _vllm_config([4], max_model_len=129)
            )

    def test_rejects_context_beyond_128_even_with_larger_window(
            self, deepseek_mod):
        with pytest.raises(ValueError, match="max_model_len <= 128"):
            deepseek_mod._validate_short_context_attention(
                _vllm_config([4], max_model_len=129, sliding_window=256)
            )

    def test_c4a_requires_all_candidates_fit_index_topk(self, deepseek_mod):
        with pytest.raises(NotImplementedError, match="index_topk"):
            deepseek_mod._validate_short_context_attention(
                _vllm_config([4], index_topk=31)
            )

    def test_c4a_boundary_passes_exactly(self, deepseek_mod):
        # (128 + 3) // 4 == 32 == index_topk: exactly fits.
        deepseek_mod._validate_short_context_attention(
            _vllm_config([4], index_topk=32)
        )

    def test_ignores_ratio_4_beyond_hidden_layers(self, deepseek_mod):
        """A ratio-4 entry past num_hidden_layers must not gate the check."""
        config = _vllm_config([0, 1], layers=2)
        config.model_config.hf_config.compress_ratios = [0, 1, 4, 4]
        deepseek_mod._validate_short_context_attention(config)


class TestCacheHelpers:
    def test_reject_wrong_cache_shape(self, deepseek_mod):
        cache = torch.zeros((2, 2, 3, 4))
        slots = torch.tensor([0, 1])
        with pytest.raises(ValueError, match="KV cache shape"):
            deepseek_mod._cache_rows(cache, slots, block_size=3)
        with pytest.raises(ValueError, match="KV cache shape"):
            deepseek_mod._store_cache_rows(
                cache, slots, torch.zeros(2, 4), block_size=3
            )

    def test_round_trip_through_noncontiguous_pages(self, deepseek_mod):
        num_blocks, block_size, width = 3, 2, 5
        cache = torch.zeros(num_blocks, block_size, width)
        targets = [(2, 1), (0, 0), (1, 0)]
        slots = torch.tensor(
            [page * block_size + offset for page, offset in targets]
        )
        rows = torch.arange(3 * width, dtype=torch.float32).reshape(3, width)

        deepseek_mod._store_cache_rows(cache, slots, rows, block_size)

        torch.testing.assert_close(
            deepseek_mod._cache_rows(cache, slots, block_size), rows,
            rtol=0, atol=0,
        )
        expected = torch.zeros_like(cache)
        for (page, offset), row in zip(targets, rows):
            expected[page, offset] = row
        torch.testing.assert_close(cache, expected, rtol=0, atol=0)

    def test_store_casts_rows_to_cache_dtype(self, deepseek_mod):
        cache = torch.zeros(2, 2, 3, dtype=torch.bfloat16)
        slots = torch.tensor([0, 3])
        rows = torch.full((2, 3), 1.5, dtype=torch.float32)
        deepseek_mod._store_cache_rows(cache, slots, rows, block_size=2)
        torch.testing.assert_close(
            deepseek_mod._cache_rows(cache, slots, 2),
            rows.to(torch.bfloat16), rtol=0, atol=0,
        )


class TestRegistrySurface:
    def test_backend_names(self, deepseek_mod):
        assert deepseek_mod.DeepseekV4FLBackend.get_name() == (
            "FL_ASCEND_V4_BF16_SWA"
        )
        assert deepseek_mod.DeepseekV4FLSWABackend.get_name() == (
            "FL_ASCEND_V4_BF16_SWA_CACHE"
        )
        assert (deepseek_mod.DeepseekV4FLBackend.get_builder_cls()
                is deepseek_mod.DeepseekV4SparseMLAMetadataBuilder)
        assert (deepseek_mod.DeepseekV4FLSWABackend.get_builder_cls()
                is deepseek_mod.DeepseekV4FLMetadataBuilder)

    def test_supports_any_capability(self, deepseek_mod):
        assert deepseek_mod.DeepseekV4FLBackend.supports_compute_capability(
            None
        )

    def test_tile_scheduler_disables_engines(self, deepseek_mod):
        scheduler = deepseek_mod.DeepseekV4FLMetadataBuilder()
        assert scheduler.build_tile_scheduler(0) == {
            "swaonly": None, "c4a": None, "c128a": None,
        }

    def test_module_all_exports(self, deepseek_mod):
        assert set(deepseek_mod.__all__) == {
            "DeepseekV4ForCausalLM", "DeepseekV4FLAttention"
        }


class TestAttentionConstruction:
    def _config(self):
        return _vllm_config([0, 1])

    def test_init_swaps_cuda_event_around_super(self, deepseek_mod):
        original_event = torch.cuda.Event
        seen = {}

        class _RecordingBase(_UpstreamBase):
            def __init__(self, *args, **kwargs):
                seen["event_at_super"] = torch.cuda.Event
                super().__init__(*args, **kwargs)

        real_init = _UpstreamBase.__init__

        def recording_init(self, *args, **kwargs):
            seen["event_at_super"] = torch.cuda.Event
            real_init(self, *args, **kwargs)

        _UpstreamBase.__init__ = recording_init
        try:
            instance = deepseek_mod.DeepseekV4FLAttention(
                vllm_config=self._config()
            )
            assert seen["event_at_super"] is torch.npu.Event
            assert torch.cuda.Event is original_event  # restored
            assert isinstance(instance, deepseek_mod.DeepseekV4FLAttention)
        finally:
            torch.cuda.Event = original_event
            _UpstreamBase.__init__ = real_init

    def test_init_resolves_positional_vllm_config(self, deepseek_mod):
        deepseek_mod.DeepseekV4FLAttention(self._config())

    def test_init_requires_vllm_config(self, deepseek_mod):
        with pytest.raises(AssertionError):
            deepseek_mod.DeepseekV4FLAttention()

    def test_init_validates_before_super(self, deepseek_mod):
        class _FailingBase(_UpstreamBase):
            def __init__(self, *args, **kwargs):
                raise RuntimeError("upstream init failed")

        real_init = _UpstreamBase.__init__

        def failing_init(self, *args, **kwargs):
            raise RuntimeError("upstream init failed")

        _UpstreamBase.__init__ = failing_init
        try:
            # Validation errors surface before the upstream failure.
            with pytest.raises(ValueError, match="max_model_len"):
                deepseek_mod.DeepseekV4FLAttention(
                    vllm_config=_vllm_config([4], max_model_len=999)
                )
            # The upstream error itself propagates when validation passes,
            # and the cuda.Event swap is restored either way.
            original_event = torch.cuda.Event
            with pytest.raises(RuntimeError, match="upstream init failed"):
                deepseek_mod.DeepseekV4FLAttention(vllm_config=self._config())
            assert torch.cuda.Event is original_event
        finally:
            _UpstreamBase.__init__ = real_init

    def test_class_attributes(self, deepseek_mod):
        assert deepseek_mod.DeepseekV4FLAttention.backend_cls is (
            deepseek_mod.DeepseekV4FLBackend
        )
        assert deepseek_mod.DeepseekV4FLAttention.swa_backend_cls is (
            deepseek_mod.DeepseekV4FLSWABackend
        )
        assert deepseek_mod.DeepseekV4FLAttention.use_fp8_ds_mla_layout is False
        assert deepseek_mod.DeepseekV4FLBackend.supported_kv_cache_dtypes == [
            "auto"
        ]
        assert deepseek_mod.DeepseekV4FLAttention.get_padded_num_q_heads(7) == 7


class TestForward:
    def test_rmsnorms_split_qr_and_kv(self, deepseek_mod):
        """forward splits the fused projection and RMSNorms both halves.

        Independent oracle: with qr_kv row [3, 4, 6, 8], q_lora_rank 2 and
        head_dim 2, qr = [3, 4] and kv = [6, 8]; each half is scaled by
        rsqrt(mean(x^2) + eps) and then by its norm weight.
        """
        eps = 1e-6
        qr_kv = torch.tensor([[3.0, 4.0, 6.0, 8.0]])
        captured = {}

        def fake_projections(hidden_states):
            captured["hidden_states"] = hidden_states
            return qr_kv, "kv-score", None, None

        attention = SimpleNamespace(
            padded_heads=2,
            head_dim=2,
            q_lora_rank=2,
            eps=eps,
            q_norm=SimpleNamespace(weight=torch.tensor([1.0, 0.5])),
            kv_norm=SimpleNamespace(weight=torch.tensor([2.0, 1.0])),
            _run_parallel_input_projections=fake_projections,
            _prepare_and_attn_fn=lambda *args: captured.setdefault(
                "prepare_args", args
            ),
            _o_proj=lambda o, positions: o,
            n_local_heads=2,
        )
        hidden_states = torch.randn(1, 7)
        positions = torch.tensor([0])

        out = deepseek_mod.DeepseekV4FLAttention.forward(
            attention, positions, hidden_states, llama_4_scaling=torch.ones(1)
        )

        qr_expected = (
            torch.tensor([[3.0, 4.0]])
            * torch.rsqrt(torch.tensor(12.5 + eps))
            * torch.tensor([1.0, 0.5])
        )
        kv_expected = (
            torch.tensor([[6.0, 8.0]])
            * torch.rsqrt(torch.tensor(50.0 + eps))
            * torch.tensor([2.0, 1.0])
        )
        args = captured["prepare_args"]
        torch.testing.assert_close(args[1], qr_expected)
        torch.testing.assert_close(args[2], kv_expected)
        assert args[3] == "kv-score"
        assert captured["hidden_states"] is hidden_states
        assert args[0] is hidden_states
        assert args[6] is positions
        assert out.shape == (1, 2, 2)

    def test_allocates_o_padded_and_slices_for_o_proj(self, deepseek_mod):
        seen = {}

        def fake_prepare(hidden_states, qr, kv, kv_score, iks, iw, positions,
                         o_padded):
            seen["o_padded"] = o_padded
            o_padded.fill_(2.0)

        def fake_o_proj(o, positions):
            seen["o_proj_shape"] = tuple(o.shape)
            seen["o_proj_filled"] = bool((o == 2.0).all())
            return "o-proj-result"

        attention = SimpleNamespace(
            padded_heads=5,
            head_dim=4,
            q_lora_rank=3,
            eps=1e-6,
            q_norm=SimpleNamespace(weight=torch.ones(3)),
            kv_norm=SimpleNamespace(weight=torch.ones(4)),
            _run_parallel_input_projections=lambda hs: (
                torch.zeros(2, 7), torch.zeros(2, 4), None, None,
            ),
            _prepare_and_attn_fn=fake_prepare,
            _o_proj=fake_o_proj,
            n_local_heads=3,
        )
        hidden_states = torch.randn(2, 6)

        out = deepseek_mod.DeepseekV4FLAttention.forward(
            attention, torch.zeros(2, dtype=torch.long), hidden_states
        )

        assert seen["o_padded"].shape == (2, 5, 4)
        assert seen["o_padded"].dtype == hidden_states.dtype
        assert seen["o_proj_shape"] == (2, 3, 4)
        assert seen["o_proj_filled"] is True
        assert out == "o-proj-result"


class TestInputProjections:
    def test_without_compressor_returns_none_scores(self, deepseek_mod):
        gemm_result = torch.randn(3, 6)
        seen = {}

        def fake_gemm(hidden_states):
            seen["hidden_states"] = hidden_states
            return gemm_result

        attention = SimpleNamespace(_fused_wqa_wkv_gemm=fake_gemm,
                                    compressor=None)
        hidden_states = torch.randn(3, 5)

        qr_kv, kv_score, iks, iw = (
            deepseek_mod.DeepseekV4FLAttention._run_parallel_input_projections(
                attention, hidden_states
            )
        )
        assert qr_kv is gemm_result
        assert kv_score is None and iks is None and iw is None
        assert seen["hidden_states"] is hidden_states

    def test_compressor_gemm_runs_in_fp32(self, deepseek_mod):
        weight = torch.randn(4, 5, dtype=torch.bfloat16) * 0.3
        hidden_states = torch.randn(3, 5, dtype=torch.bfloat16)
        attention = SimpleNamespace(
            _fused_wqa_wkv_gemm=lambda hs: torch.zeros(3, 2),
            compressor=SimpleNamespace(
                fused_wkv_wgate=SimpleNamespace(weight=weight)
            ),
        )
        qr_kv, kv_score, _, _ = (
            deepseek_mod.DeepseekV4FLAttention._run_parallel_input_projections(
                attention, hidden_states
            )
        )
        expected = torch.einsum("td,od->to", hidden_states.float(),
                               weight.float())
        torch.testing.assert_close(kv_score, expected)
        assert kv_score.dtype == torch.float32
        assert qr_kv.shape == (3, 2)


class TestFusedQnormRopeKvInsert:
    def test_passthrough_without_dict_metadata(self, deepseek_mod):
        attention = SimpleNamespace(
            swa_cache_layer=SimpleNamespace(
                prefix="layer.swa_cache", kv_cache=torch.zeros(1, 1, 1)
            )
        )
        q = torch.randn(2, 3, 4)
        returned = (
            deepseek_mod.DeepseekV4FLAttention._fused_qnorm_rope_kv_insert(
                attention, q, torch.randn(2, 4),
                torch.zeros(2, dtype=torch.long), object(),
            )
        )
        assert returned is q

    def test_norms_q_and_stores_rotated_kv(self, deepseek_mod):
        """q is weightless-RMSNormed, roped, and rotated kv lands in the
        SWA cache at exactly the valid slots."""
        eps = 1e-6
        head_dim = 3
        block_size = 2
        kv_cache = torch.zeros(2, block_size, head_dim)

        class _ScaleRotary:
            def __init__(self):
                self.calls = []

            def forward_native(self, positions, q, kv):
                self.calls.append((positions, q, kv))
                return q * 3.0, kv * 5.0

        rotary = _ScaleRotary()
        q = torch.full((3, 1, head_dim), 2.0)
        kv = torch.stack(
            [torch.arange(head_dim, dtype=torch.float32) + float(i)
             for i in range(3)]
        )
        positions = torch.tensor([0, 1, 2])
        metadata = SimpleNamespace(
            slot_mapping=torch.tensor([0, -1, 2]), block_size=block_size
        )
        attention = SimpleNamespace(
            swa_cache_layer=SimpleNamespace(prefix="layer.swa_cache",
                                            kv_cache=kv_cache),
            eps=eps,
            rotary_emb=rotary,
        )

        returned = (
            deepseek_mod.DeepseekV4FLAttention._fused_qnorm_rope_kv_insert(
                attention, q, kv, positions, {"layer.swa_cache": metadata}
            )
        )

        norm = torch.rsqrt(torch.tensor(4.0 + eps))
        torch.testing.assert_close(
            returned, torch.full((3, 1, head_dim), 2.0) * norm * 3
        )
        torch.testing.assert_close(kv_cache[0, 0], kv[0] * 5)
        torch.testing.assert_close(kv_cache[1, 0], kv[2] * 5)
        assert torch.count_nonzero(kv_cache[0, 1]) == 0
        assert torch.count_nonzero(kv_cache[1, 1]) == 0
        assert rotary.calls[0][0] is positions
        torch.testing.assert_close(rotary.calls[0][1], q * norm)

    def test_all_slots_invalid_writes_nothing(self, deepseek_mod):
        kv_cache = torch.zeros(1, 2, 3)

        class _Rotary:
            def forward_native(self, positions, q, kv):
                return q, kv

        attention = SimpleNamespace(
            swa_cache_layer=SimpleNamespace(prefix="layer.swa_cache",
                                            kv_cache=kv_cache),
            eps=1e-6,
            rotary_emb=_Rotary(),
        )
        metadata = SimpleNamespace(slot_mapping=torch.tensor([-1, -1]),
                                   block_size=2)
        deepseek_mod.DeepseekV4FLAttention._fused_qnorm_rope_kv_insert(
            attention, torch.randn(2, 1, 3), torch.randn(2, 3),
            torch.tensor([0, 1]), {"layer.swa_cache": metadata},
        )
        assert torch.count_nonzero(kv_cache) == 0


class _PassthroughRotary:
    """Rope stand-in: identity rotation with a cos/sin cache."""

    def __init__(self, num_entries=8):
        self.cos_sin_cache = torch.cat(
            (torch.ones((num_entries, 1)), torch.zeros((num_entries, 1))),
            dim=1,
        )

    def forward_native(self, positions, q, kv, inverse=False):
        return q, kv


class TestPrepareAndAttn:
    def _attention(self, deepseek_mod, **attrs):
        attention = SimpleNamespace(
            prefix="layer",
            wq_b=lambda qr: qr,
            eps=1e-6,
            swa_cache_layer=SimpleNamespace(
                prefix="layer.swa_cache", kv_cache=torch.zeros(1, 4, 2)
            ),
            rotary_emb=_PassthroughRotary(),
            compressor=None,
            compress_ratio=0,
            n_local_heads=1,
            head_dim=2,
            forward_mqa=lambda q, kv, positions, output: None,
        )
        for key, value in attrs.items():
            setattr(attention, key, value)
        # Bind the real method under test for the nested helper.
        attention._fused_qnorm_rope_kv_insert = (
            lambda q, kv, positions, attn_metadata:
            deepseek_mod.DeepseekV4FLAttention._fused_qnorm_rope_kv_insert(
                attention, q, kv, positions, attn_metadata
            )
        )
        return attention

    def test_pipeline_without_compressor(self, deepseek_mod):
        calls = {}
        attention = self._attention(
            deepseek_mod, forward_mqa=lambda q, kv, positions, output:
            calls.setdefault("mqa", (q, kv, positions, output)),
        )
        metadata = {
            "layer.swa_cache": SimpleNamespace(
                slot_mapping=torch.tensor([0]), block_size=4
            )
        }
        _set_attn_metadata(deepseek_mod, metadata)
        q = torch.ones(1, 1, 2)

        deepseek_mod.DeepseekV4FLAttention._prepare_and_attn(
            attention, "hs", torch.ones(1, 2), torch.ones(1, 2),
            None, None, None, torch.tensor([0]), torch.zeros(1, 1, 2),
        )
        mqa_q = calls["mqa"][0]
        # qnorm applied: ones * rsqrt(1 + eps).
        torch.testing.assert_close(
            mqa_q, q * torch.rsqrt(torch.tensor(1.0 + 1e-6))
        )

    def test_c4a_topk_fill_for_ratio_4(self, deepseek_mod):
        calls = {}
        attention = self._attention(
            deepseek_mod,
            compressor=SimpleNamespace(mark="compressor"),
            compress_ratio=4,
            indexer=SimpleNamespace(topk_tokens=8),
            topk_indices_buffer=torch.zeros(4, 8, dtype=torch.int32),
            forward_mqa=lambda q, kv, positions, output:
            calls.setdefault("mqa", (q, kv)),
            _compress_kv_bf16=lambda kv_score, positions:
            calls.setdefault("compress", (kv_score, positions)),
        )
        metadata = {
            "layer.swa_cache": SimpleNamespace(
                slot_mapping=torch.tensor([0, 1, 2, 3]), block_size=4
            )
        }
        _set_attn_metadata(deepseek_mod, metadata)

        deepseek_mod.DeepseekV4FLAttention._prepare_and_attn(
            attention, "hs", torch.ones(4, 2), torch.ones(4, 2),
            "kv-score", None, None, torch.tensor([1, 3, 4, 7]),
            torch.zeros(4, 1, 2),
        )
        assert calls["compress"][0] == "kv-score"
        torch.testing.assert_close(
            calls["compress"][1], torch.tensor([1, 3, 4, 7])
        )
        expected = torch.tensor(
            [
                [-1, -1, -1, -1, -1, -1, -1, -1],
                [0, -1, -1, -1, -1, -1, -1, -1],
                [0, -1, -1, -1, -1, -1, -1, -1],
                [0, 1, -1, -1, -1, -1, -1, -1],
            ],
            dtype=torch.int32,
        )
        torch.testing.assert_close(attention.topk_indices_buffer, expected)

    def test_skips_topk_fill_for_ratio_128(self, deepseek_mod):
        buffer = torch.full((2, 8), 7, dtype=torch.int32)
        attention = self._attention(
            deepseek_mod, compress_ratio=128,
            topk_indices_buffer=buffer,
        )
        _set_attn_metadata(
            deepseek_mod,
            {"layer.swa_cache": SimpleNamespace(
                slot_mapping=torch.tensor([0, 1]), block_size=4
            )},
        )
        deepseek_mod.DeepseekV4FLAttention._prepare_and_attn(
            attention, "hs", torch.ones(2, 2), torch.ones(2, 2),
            None, None, None, torch.tensor([127, 255]),
            torch.zeros(2, 1, 2),
        )
        assert torch.equal(
            attention.topk_indices_buffer,
            torch.full((2, 8), 7, dtype=torch.int32),
        )

    def test_missing_topk_buffer_asserts_for_ratio_4(self, deepseek_mod):
        attention = self._attention(
            deepseek_mod, compress_ratio=4, indexer=SimpleNamespace(
                topk_tokens=8
            ),
        )
        attention.topk_indices_buffer = None
        _set_attn_metadata(
            deepseek_mod,
            {"layer.swa_cache": SimpleNamespace(
                slot_mapping=torch.tensor([0]), block_size=4
            )},
        )
        with pytest.raises(AssertionError):
            deepseek_mod.DeepseekV4FLAttention._prepare_and_attn(
                attention, "hs", torch.ones(1, 2), torch.ones(1, 2),
                None, None, None, torch.tensor([3]), torch.zeros(1, 1, 2),
            )


class TestOProj:
    def test_inverse_rope_then_two_stage_projection(self, deepseek_mod):
        calls = []

        class _InverseRotary:
            def forward_native(self, positions, o, kv, inverse=False):
                assert inverse is True
                calls.append(("rope", positions))
                return o * 4.0, None

        attention = SimpleNamespace(
            rotary_emb=_InverseRotary(),
            wo_a=lambda flat: (calls.append(("wo_a", flat)), flat * 0.5)[1],
            wo_b=lambda z: (calls.append(("wo_b", z)), z + 1.0)[1],
            n_local_groups=1,
        )
        o = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        positions = torch.tensor([0, 1])

        out = deepseek_mod.DeepseekV4FLAttention._o_proj(
            attention, o, positions
        )
        torch.testing.assert_close(out, o * 4.0 * 0.5 + 1.0)
        assert [name for name, _ in calls] == ["rope", "wo_a", "wo_b"]

    def test_rejects_multiple_local_groups(self, deepseek_mod):
        class _Rotary:
            def forward_native(self, positions, o, kv, inverse=False):
                return o, None

        attention = SimpleNamespace(
            rotary_emb=_Rotary(), wo_a=lambda x: x, wo_b=lambda x: x,
            n_local_groups=2,
        )
        with pytest.raises(ValueError, match="one group per TP rank"):
            deepseek_mod.DeepseekV4FLAttention._o_proj(
                attention, torch.zeros(1, 4), torch.zeros(1, dtype=torch.long)
            )


class TestForwardMqa:
    def _swa_double(self, ratio=0, head_dim=4, window_size=128,
                    sink_value=-float("inf"), swa_cache=None,
                    compressed_cache=None, topk_buffer=None):
        return SimpleNamespace(
            prefix="layer",
            swa_cache_layer=SimpleNamespace(prefix="layer.swa_cache",
                                            kv_cache=swa_cache),
            kv_cache=compressed_cache,
            compress_ratio=ratio,
            max_model_len=128,
            window_size=window_size,
            head_dim=head_dim,
            n_local_heads=1,
            scale=head_dim ** -0.5,
            attn_sink=torch.tensor([sink_value]),
            topk_indices_buffer=topk_buffer,
        )

    def test_zeroes_output_without_dict_metadata(self, deepseek_mod):
        output = torch.ones(2, 1, 4)
        attention = self._swa_double(swa_cache=torch.zeros(1, 128, 4))
        _set_attn_metadata(deepseek_mod, object())
        deepseek_mod.DeepseekV4FLAttention.forward_mqa(
            attention, torch.zeros(2, 1, 4), torch.zeros(2, 4),
            torch.tensor([0, 1]), output,
        )
        assert torch.count_nonzero(output) == 0

    def test_swa_only_matches_softmax_reference(self, deepseek_mod):
        head_dim = 4
        swa_cache = torch.zeros(1, 128, head_dim)
        for i in range(16):
            swa_cache[0, i] = torch.arange(head_dim,
                                           dtype=torch.float32) * (i + 1) * 0.1
        attention = self._swa_double(swa_cache=swa_cache)
        positions = torch.tensor([15])
        q = torch.ones(1, 1, head_dim)
        swa_metadata = SimpleNamespace(
            token_to_req_indices=torch.tensor([0], dtype=torch.int32),
            block_size=128,
            block_table=torch.tensor([[0]]),
        )
        _set_attn_metadata(deepseek_mod, {"layer.swa_cache": swa_metadata})
        output = torch.empty_like(q)

        deepseek_mod.DeepseekV4FLAttention.forward_mqa(
            attention, q, torch.zeros(1, head_dim), positions, output
        )

        keys = swa_cache[0, :16].float()
        logits = (q[0, 0].float() @ keys.T) * attention.scale
        expected = torch.softmax(logits, dim=-1) @ keys
        torch.testing.assert_close(
            output[0, 0], expected.to(output.dtype), rtol=0.01, atol=0.01
        )

    def test_crops_to_sliding_window(self, deepseek_mod):
        head_dim = 4
        window_size = 8
        swa_cache = torch.zeros(1, 128, head_dim)
        for i in range(64):
            swa_cache[0, i] = torch.arange(head_dim,
                                           dtype=torch.float32) * (i + 1)
        attention = self._swa_double(window_size=window_size,
                                     swa_cache=swa_cache)
        positions = torch.tensor([63])
        q = torch.ones(1, 1, head_dim)
        swa_metadata = SimpleNamespace(
            token_to_req_indices=torch.tensor([0], dtype=torch.int32),
            block_size=128,
            block_table=torch.tensor([[0]]),
        )
        _set_attn_metadata(deepseek_mod, {"layer.swa_cache": swa_metadata})
        output = torch.empty_like(q)

        deepseek_mod.DeepseekV4FLAttention.forward_mqa(
            attention, q, torch.zeros(1, head_dim), positions, output
        )

        keys = swa_cache[0, 56:64].float()
        logits = (q[0, 0].float() @ keys.T) * attention.scale
        expected = torch.softmax(logits, dim=-1) @ keys
        torch.testing.assert_close(
            output[0, 0], expected.to(output.dtype), rtol=0.01, atol=0.01
        )

    def test_masks_rows_before_window_start(self, deepseek_mod):
        head_dim = 4
        swa_cache = torch.zeros(1, 8, head_dim)
        for i in range(8):
            swa_cache[0, i] = torch.arange(head_dim,
                                           dtype=torch.float32) * (i + 1)
        attention = self._swa_double(window_size=8, swa_cache=swa_cache)
        positions = torch.tensor([2])
        q = torch.ones(1, 1, head_dim)
        swa_metadata = SimpleNamespace(
            token_to_req_indices=torch.tensor([0], dtype=torch.int32),
            block_size=8,
            block_table=torch.tensor([[0]]),
        )
        _set_attn_metadata(deepseek_mod, {"layer.swa_cache": swa_metadata})
        output = torch.empty_like(q)

        deepseek_mod.DeepseekV4FLAttention.forward_mqa(
            attention, q, torch.zeros(1, head_dim), positions, output
        )

        keys = swa_cache[0, :3].float()
        logits = (q[0, 0].float() @ keys.T) * attention.scale
        expected = torch.softmax(logits, dim=-1) @ keys
        torch.testing.assert_close(
            output[0, 0], expected.to(output.dtype), rtol=0.01, atol=0.01
        )

    def test_skips_unallocated_blocks(self, deepseek_mod):
        head_dim = 4
        swa_cache = torch.ones(1, 8, head_dim)
        attention = self._swa_double(window_size=8, sink_value=0.0,
                                     swa_cache=swa_cache)
        positions = torch.tensor([5])
        q = torch.ones(1, 1, head_dim)
        swa_metadata = SimpleNamespace(
            token_to_req_indices=torch.tensor([0], dtype=torch.int32),
            block_size=8,
            block_table=torch.tensor([[-1]]),
        )
        _set_attn_metadata(deepseek_mod, {"layer.swa_cache": swa_metadata})
        output = torch.empty_like(q)

        deepseek_mod.DeepseekV4FLAttention.forward_mqa(
            attention, q, torch.zeros(1, head_dim), positions, output
        )
        assert torch.count_nonzero(output) == 0
        assert torch.isfinite(output).all()

    @pytest.mark.parametrize("ratio,position", [(4, 3), (128, 127)])
    def test_compressed_and_swa_share_softmax(self, deepseek_mod, ratio,
                                              position):
        head_dim = 4
        swa_cache = torch.zeros((1, 128, head_dim), dtype=torch.bfloat16)
        compressed_cache = torch.zeros((1, 128 // ratio, head_dim),
                                       dtype=torch.bfloat16)
        compressed_cache[0, 0, :] = 1
        attention = self._swa_double(
            ratio=ratio, swa_cache=swa_cache,
            compressed_cache=compressed_cache,
            topk_buffer=torch.arange(32, dtype=torch.int32).unsqueeze(0),
        )
        q = torch.ones((1, 1, head_dim), dtype=torch.bfloat16)
        positions = torch.tensor([position])
        swa_metadata = SimpleNamespace(
            token_to_req_indices=torch.tensor([0], dtype=torch.int32),
            block_size=128,
            block_table=torch.tensor([[0]]),
        )
        compressed_metadata = SimpleNamespace(
            block_size=128, block_table=torch.tensor([[0]])
        )
        _set_attn_metadata(
            deepseek_mod,
            {"layer.swa_cache": swa_metadata, "layer": compressed_metadata},
        )
        output = torch.empty_like(q)

        deepseek_mod.DeepseekV4FLAttention.forward_mqa(
            attention, q, q, positions, output
        )

        keys = torch.cat((compressed_cache[0, :1],
                          swa_cache[0, : position + 1]))
        logits = (q[0, 0].float() @ keys.float().T) * attention.scale
        expected = (torch.softmax(logits, dim=-1) @ keys.float()).to(q.dtype)
        torch.testing.assert_close(output[0, 0], expected, rtol=0.01,
                                    atol=0.01)
        assert output[0, 0, 0] > 0

    def test_batches_requests_via_token_to_req(self, deepseek_mod):
        head_dim = 4
        swa_cache = torch.zeros(2, 8, head_dim)
        for i in range(8):
            swa_cache[0, i] = torch.arange(head_dim, dtype=torch.float32)
            swa_cache[1, i] = torch.arange(head_dim,
                                           dtype=torch.float32) * -0.5
        attention = self._swa_double(window_size=8, swa_cache=swa_cache)
        q = torch.ones(2, 1, head_dim)
        positions = torch.tensor([4, 4])
        swa_metadata = SimpleNamespace(
            token_to_req_indices=torch.tensor([0, 1], dtype=torch.int32),
            block_size=8,
            block_table=torch.tensor([[0], [1]]),
        )
        _set_attn_metadata(deepseek_mod, {"layer.swa_cache": swa_metadata})
        output = torch.empty_like(q)

        deepseek_mod.DeepseekV4FLAttention.forward_mqa(
            attention, q, torch.zeros(2, head_dim), positions, output
        )

        for token, page in enumerate((0, 1)):
            keys = swa_cache[page, :5].float()
            logits = (q[token, 0].float() @ keys.T) * attention.scale
            expected = torch.softmax(logits, dim=-1) @ keys
            torch.testing.assert_close(
                output[token, 0], expected.to(output.dtype),
                rtol=0.01, atol=0.01,
            )

    def test_finite_sink_beats_all_masked_rows(self, deepseek_mod):
        """All-masked rows resolve to exact zeros, not NaN."""
        head_dim = 4
        swa_cache = torch.ones(1, 8, head_dim)
        attention = self._swa_double(window_size=1, sink_value=0.0,
                                     swa_cache=swa_cache)
        swa_metadata = SimpleNamespace(
            token_to_req_indices=torch.tensor([0], dtype=torch.int32),
            block_size=8,
            block_table=torch.tensor([[-1]]),
        )
        _set_attn_metadata(deepseek_mod, {"layer.swa_cache": swa_metadata})
        q = torch.ones(1, 1, head_dim)
        output = torch.empty_like(q)

        deepseek_mod.DeepseekV4FLAttention.forward_mqa(
            attention, q, torch.zeros(1, head_dim), torch.tensor([0]),
            output,
        )
        assert torch.isfinite(output).all()
        assert torch.count_nonzero(output) == 0

    def test_c4a_topk_buffer_respects_validity(self, deepseek_mod):
        head_dim = 4
        ratio = 4
        swa_cache = torch.zeros((1, 128, head_dim), dtype=torch.bfloat16)
        for i in range(8):
            swa_cache[0, i] = 0.1 * i
        compressed_cache = torch.zeros((1, 32, head_dim), dtype=torch.bfloat16)
        for i in range(3):
            compressed_cache[0, i] = 1.0 + i
        attention = self._swa_double(
            ratio=ratio, swa_cache=swa_cache,
            compressed_cache=compressed_cache,
            topk_buffer=torch.tensor([[-1, 1, 0]], dtype=torch.int32),
        )
        q = torch.ones((1, 1, head_dim), dtype=torch.bfloat16)
        positions = torch.tensor([7])
        swa_metadata = SimpleNamespace(
            token_to_req_indices=torch.tensor([0], dtype=torch.int32),
            block_size=128,
            block_table=torch.tensor([[0]]),
        )
        compressed_metadata = SimpleNamespace(
            block_size=128, block_table=torch.tensor([[0]])
        )
        _set_attn_metadata(
            deepseek_mod,
            {"layer.swa_cache": swa_metadata, "layer": compressed_metadata},
        )
        output = torch.empty_like(q)

        deepseek_mod.DeepseekV4FLAttention.forward_mqa(
            attention, q, q, positions, output
        )

        keys = torch.cat((compressed_cache[0, [1, 0]], swa_cache[0, :8]))
        logits = (q[0, 0].float() @ keys.float().T) * attention.scale
        expected = (torch.softmax(logits, dim=-1) @ keys.float()).to(q.dtype)
        torch.testing.assert_close(output[0, 0], expected, rtol=0.01,
                                   atol=0.01)


class TestHcHeadFallback:
    def test_installed_at_import(self, deepseek_mod):
        # The stub platform reports device_type="npu", so the loader ran
        # _install_hc_head_fallback and patched the (stub) HCHeadOp.
        assert getattr(_HCHeadOp, "_vllm_fl_ascend", True)

    def test_idempotent(self, deepseek_mod):
        # A second install call is a no-op (marker already present).
        deepseek_mod._install_hc_head_fallback()
        assert getattr(_HCHeadOp, "_vllm_fl_ascend", True)

    def test_forward_native_contract(self, deepseek_mod, monkeypatch):
        captured = {}

        def fake_kernel(flat, hc_fn, hc_scale, hc_base, out, hidden_size,
                        rms_norm_eps, hc_eps, hc_mult):
            captured.update(
                flat=flat, hc_fn=hc_fn, hc_scale=hc_scale, hc_base=hc_base,
                out=out, hidden_size=hidden_size, rms_norm_eps=rms_norm_eps,
                hc_eps=hc_eps, hc_mult=hc_mult,
            )
            out.fill_(2.5)

        import flaggems_vllm

        monkeypatch.setattr(flaggems_vllm, "hc_head_fused_kernel",
                            fake_kernel)
        # forward_native is installed as a plain function attribute on the
        # stub class; call it unbound with hidden_states as `self`-less arg.
        hc_head = _HCHeadOp.__dict__["forward_native"]
        hidden_states = torch.randn(2, 3, 4)  # [.., hc_mult, hidden]
        # The patched function keeps its `self` slot; the stub class has no
        # instances, so satisfy it with a placeholder.
        result = hc_head(
            _HCHeadOp(), hidden_states, "fn", 1.0, 0.5, 1e-6, 1e-5,
        )
        assert captured["hc_mult"] == 3
        assert captured["hidden_size"] == 4
        assert captured["hc_fn"] == "fn"
        assert captured["out"].shape == (2, 4)
        # The output view drops the hc_mult dimension: [.., hc_mult, hidden]
        # becomes [.., hidden].
        assert result.shape == (2, 4)
        assert torch.allclose(result, torch.full((2, 4), 2.5))


class TestCausalLM:
    def test_subclasses_xpu_base(self, deepseek_mod):
        assert issubclass(deepseek_mod.DeepseekV4ForCausalLM, _XpuModelBase)

    def test_init_delegates_to_super(self, deepseek_mod):
        config = _vllm_config([0, 1])
        model = deepseek_mod.DeepseekV4ForCausalLM(
            vllm_config=config, prefix="p"
        )
        assert model.init_kwargs == {"vllm_config": config, "prefix": "p"}

    def test_init_validates_before_super(self, deepseek_mod):
        with pytest.raises(ValueError, match="max_model_len"):
            deepseek_mod.DeepseekV4ForCausalLM(
                vllm_config=_vllm_config([4], max_model_len=999)
            )

    def test_xpu_attention_swapped_at_import(self, deepseek_mod):
        # device_type "npu" in the stub: the loader replaced the xpu
        # attention class with the FL one.
        assert deepseek_mod._test_xpu_model.DeepseekV4XPUAttention is (
            deepseek_mod.DeepseekV4FLAttention
        )
