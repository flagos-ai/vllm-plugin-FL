# Copyright (c) 2026 BAAI. All rights reserved.

"""Supplementary CPU tests for the deep paths of ModelRunnerFL.

This file targets the large branches not exercised by the core suite:
``ModelRunnerFL.__init__`` (the full constructor wiring matrix), the
``_update_states`` ngram-GPU correction closure, remaining branches of
``_prepare_inputs`` / ``_build_attention_metadata``, and the small helpers
that surround them.

``vllm_fl.worker.model_runner`` imports ``vllm_fl.compilation.graph`` whose
``Graph`` class raises ``NotImplementedError`` at class-body level unless the
current platform's ``device_type`` is cuda/npu/musa/gcu/ptpu/txda.  Every test
here therefore installs a delegating proxy on ``vllm.platforms.current_platform``
that fakes only the CUDA-ness attributes and forwards everything else to the
real platform, before the first import of the module under test.

The constructor is driven with real-shaped ``vllm_config`` namespaces and only
the GPU boundaries (the pipeline-parallel group lookup, the platform's
stream/event factory, pin_memory allocations, and the drafter classes) are
patched on the module under test.  All remaining methods are driven through
unbound invocation (``ModelRunnerFL.method(runner, ...)``) on hand-built
``SimpleNamespace`` runners, mirroring the pattern of the core suite.  Every
numeric expectation is computed with an independent plain-Python/torch oracle.
"""

from importlib import import_module
from types import MethodType, SimpleNamespace

import numpy as np
import pytest

try:  # pragma: no cover - dual-environment guard
    import torch

    import vllm  # noqa: F401

    HAS_VLLM = True
except ImportError:  # pragma: no cover - Windows machines without vllm
    HAS_VLLM = False

pytestmark = pytest.mark.skipif(not HAS_VLLM, reason="requires real vllm")

CPU_DEVICE = "cpu"


class _FakeCudaPlatform:
    """Delegate to the real platform; fake only CUDA-ness for import checks."""

    def __init__(self, real):
        object.__setattr__(self, "_real", real)

    @property
    def device_type(self):
        return "cuda"

    @property
    def dispatch_key(self):
        return "CUDA"

    @property
    def vendor_name(self):
        return None

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_real"), name)


@pytest.fixture
def model_runner_module(monkeypatch):
    """Import vllm_fl.worker.model_runner with CUDA-ness faked at import time.

    The compilation graph module validates ``device_type`` while its class
    body executes, so the proxy is installed before the first import and the
    real platform object is restored afterwards.  The module binds its own
    ``current_platform`` reference at import time, so per-test stream/event
    faking monkeypatches the module attribute instead.
    """
    pytest.importorskip("vllm", reason="requires real vllm")
    platforms = import_module("vllm.platforms")
    real = platforms.current_platform
    monkeypatch.setattr(platforms, "current_platform", _FakeCudaPlatform(real))
    module = import_module("vllm_fl.worker.model_runner")
    yield module
    monkeypatch.setattr(platforms, "current_platform", real)


@pytest.fixture
def patched_module(model_runner_module, monkeypatch):
    """Patch the constructor's GPU/distributed boundaries on the module.

    Only well-identified GPU-only boundaries are replaced: the pipeline-parallel
    group lookup (which would otherwise initialize the real distributed groups)
    and the platform's stream/event factory (which CPU-only torch builds cannot
    provide).  Everything else in ``__init__`` runs unmodified.
    """
    pp_group = SimpleNamespace(is_last_rank=True, is_first_rank=True, ranks=[0])
    monkeypatch.setattr(model_runner_module, "get_pp_group", lambda: pp_group)
    monkeypatch.setattr(
        model_runner_module,
        "current_platform",
        SimpleNamespace(
            torch_device_fn=SimpleNamespace(Stream=_FakeStream, Event=_FakeEvent)
        ),
    )
    # ``num_compute_units`` is resolved through the platform at call time and
    # raises on CPU-only hosts; pin it on the module's own binding.  The
    # upstream helper is ``@cache``-decorated, so patching the module-bound
    # name (not the helper itself) keeps the effect test-local.
    monkeypatch.setattr(model_runner_module, "num_compute_units", lambda device_id: 108)
    return model_runner_module


class _CpuSlotMappingKernel:
    """Pure-torch stand-in for the triton slot-mapping kernel boundary.

    Implements the documented mapping for the single-rank, non-interleaved
    configuration used by these tests: ``slot = block_number * block_size +
    offset_in_block``, with the ``PAD_SLOT_ID`` sentinel (-1) in the tail
    region between ``num_tokens`` and ``max_num_tokens``.  Everything but
    ``__call__`` is delegated to the real triton kernel object so metadata
    registrations inside ``BlockTable.__init__`` keep working.
    """

    def __init__(self, real):
        self._real = real

    def __getattr__(self, name):
        return getattr(self._real, name)

    def __call__(
        self,
        num_reqs,
        num_tokens,
        max_num_tokens,
        query_start_loc,
        positions,
        block_table,
        block_table_stride,
        slot_mapping,
        kv_cache_block_size,
        total_cp_world_size,
        total_cp_rank,
        cp_kv_cache_interleave_size,
    ):
        pad_id = -1
        for i in range(num_tokens, max_num_tokens):
            slot_mapping[i] = pad_id
        flat_block_table = block_table.reshape(-1)
        for req_idx in range(num_reqs):
            start = int(query_start_loc[req_idx])
            end = int(query_start_loc[req_idx + 1])
            for i in range(start, end):
                pos = int(positions[i])
                vbi = pos // kv_cache_block_size
                vbo = pos - vbi * kv_cache_block_size
                bn = int(flat_block_table[req_idx * block_table_stride + vbi])
                slot_mapping[i] = bn * kv_cache_block_size + vbo


@pytest.fixture
def cpu_slot_mapping_kernel(monkeypatch):
    """Replace the triton slot-mapping kernel with a pure-torch oracle."""
    block_table_module = import_module("vllm.v1.worker.block_table")
    real = block_table_module._COMPUTE_SLOT_MAPPING_KERNEL
    proxy = _CpuSlotMappingKernel(real)
    monkeypatch.setattr(block_table_module, "_COMPUTE_SLOT_MAPPING_KERNEL", proxy)
    return proxy


# ---------------------------------------------------------------------------
# CPU-safe fakes for the torch stream/event boundary.
# ---------------------------------------------------------------------------


class _FakeEvent:
    """Minimal stand-in for a torch device Event."""

    def __init__(self, blocking=False):
        self.blocking = blocking
        self.record_count = 0
        self.synchronized = False

    def record(self):
        self.record_count += 1

    def synchronize(self):
        self.synchronized = True


class _FakeStream:
    """Minimal stand-in for a torch device Stream."""

    def __init__(self):
        self.waited = []
        self.wait_event_count = 0

    def wait_stream(self, other):
        self.waited.append(other)

    def wait_event(self, event):
        self.wait_event_count += 1


# ---------------------------------------------------------------------------
# A real-shaped vllm_config namespace for driving ModelRunnerFL.__init__.
# ---------------------------------------------------------------------------


def _make_model_config(**overrides):
    """A ModelConfig-shaped namespace with the fields ``__init__`` reads.

    The real ModelConfig dataclass is not constructible without the model
    registry, so only the read fields are present, with defaults matching a
    small dense-attention text model.
    """
    config = SimpleNamespace(
        model="facebook/opt-125m",
        hf_text_config=SimpleNamespace(sliding_window=None),
        dtype=torch.float16,
        runner_type="generate",
        enable_prompt_embeds=False,
        is_multimodal_raw_input_only_model=False,
        is_moe=False,
        max_model_len=32,
        uses_alibi=False,
        disable_cascade_attn=False,
        is_mm_prefix_lm=False,
        uses_mrope=False,
        supports_multimodal_inputs=False,
        is_encoder_decoder=False,
        logprobs_mode="processed_logprobs",
        use_fp64_gumbel=False,
        logits_processors=None,
        get_num_attention_heads=lambda parallel_config: 8,
        get_inputs_embeds_size=lambda: 4,
        get_vocab_size=lambda: 32,
        rswa_window=None,
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def _make_spec_config(method, num_speculative_tokens=2, **overrides):
    """A SpeculativeConfig-shaped namespace with all dispatch predicates.

    ``__init__``'s drafter dispatch chain probes, in order: custom_class,
    ngram, uses_draft_model, use_ngram_gpu, use_gemma4_mtp, use_step3p5_mtp,
    use_dflash, suffix, use_eagle, medusa, extract_hidden_states.  Every
    predicate defaults to False and only ``method`` matches, so each config
    selects exactly the branch named by ``method``.
    """
    spec = SimpleNamespace(
        method=method,
        num_speculative_tokens=num_speculative_tokens,
        uses_draft_model=lambda: False,
        use_ngram_gpu=lambda: False,
        use_gemma4_mtp=lambda: False,
        use_step3p5_mtp=lambda: False,
        use_dflash=lambda: False,
        use_eagle=lambda: False,
        rejection_sample_method="rejection_sampler",
        draft_model_config=None,
    )
    for key, value in overrides.items():
        setattr(spec, key, value)
    return spec


def _make_vllm_config(**overrides):
    """Assemble the vllm_config namespace ``ModelRunnerFL.__init__`` reads."""
    config_mod = import_module("vllm.config")
    model_config = overrides.pop("model_config", None) or _make_model_config()
    speculative = overrides.pop("speculative_config", None)
    config = SimpleNamespace(
        model_config=model_config,
        cache_config=SimpleNamespace(
            cache_dtype="auto",
            block_size=16,
            calculate_kv_scales=False,
            mamba_cache_mode=None,
            use_replayssm=False,
            kv_sharing_fast_prefill=False,
        ),
        offload_config=config_mod.OffloadConfig(),
        compilation_config=SimpleNamespace(
            cudagraph_mode=config_mod.CUDAGraphMode.NONE,
            cudagraph_capture_sizes=None,
        ),
        lora_config=None,
        load_config=SimpleNamespace(load_format="dummy"),
        parallel_config=SimpleNamespace(
            data_parallel_size=1,
            decode_context_parallel_size=1,
            distributed_executor_backend="mp",
            cp_kv_cache_interleave_size=1,
        ),
        scheduler_config=SimpleNamespace(
            max_num_seqs=4,
            max_num_batched_tokens=64,
            max_num_encoder_input_tokens=0,
            async_scheduling=False,
        ),
        speculative_config=speculative,
        observability_config=SimpleNamespace(report_metrics=False),
        reasoning_config=None,
        num_speculative_tokens=(
            speculative.num_speculative_tokens if speculative else 0
        ),
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def _build_runner(module, vllm_config):
    """Instantiate ModelRunnerFL with the raw (unwrapped) constructor."""
    ModelRunnerFL = module.ModelRunnerFL
    runner = object.__new__(ModelRunnerFL)
    # ``JitWarmupRegistry.capture`` wraps ``__init__``; the raw function is
    # available through functools.wraps' ``__wrapped__``.
    ModelRunnerFL.__init__.__wrapped__(runner, vllm_config, torch.device(CPU_DEVICE))
    return runner


# ---------------------------------------------------------------------------
# __init__: baseline wiring and model-shape branches.
# ---------------------------------------------------------------------------


class TestInitBasics:
    """ModelRunnerFL.__init__ with the baseline (non-spec, non-mm) config."""

    def test_baseline_creates_input_batch_and_buffers(self, patched_module):
        """The baseline config yields a batch and the persistent buffers with
        the shapes dictated by max_num_seqs / max_num_batched_tokens."""
        runner = _build_runner(patched_module, _make_vllm_config())
        assert runner.input_batch is not None
        assert runner.max_num_reqs == 4
        # max_num_batched_tokens=64 sizes the token dim of the buffers.
        assert runner.input_ids.cpu.shape == (64,)
        assert runner.query_start_loc.cpu.shape == (5,)
        assert runner.seq_lens.shape == (4,)
        assert runner.num_scheduled_tokens.cpu.shape == (4,)
        assert runner.discard_request_mask.cpu.shape == (4,)
        assert runner.inputs_embeds.cpu.shape == (64, 4)
        # arange is cached in int64 to avoid long-context overflow.
        assert runner.arange_np.dtype == np.int64
        assert runner.arange_np.shape == (64,)
        assert runner._arange_scratch.dtype == np.int64
        assert runner.uniform_decode_query_len == 1
        assert runner.mm_budget is None
        assert not hasattr(runner, "mrope_positions")

    def test_baseline_config_field_defaults(self, patched_module):
        """Fields derived straight from the config namespace are propagated."""
        runner = _build_runner(patched_module, _make_vllm_config())
        assert runner.is_pooling_model is False
        assert runner.max_model_len == 32
        assert runner.num_query_heads == 8
        assert runner.inputs_embeds_size == 4
        assert runner.kv_cache_dtype == torch.float16
        assert runner.broadcast_pp_output is False
        assert runner.check_ep_fault is False
        assert runner.max_encoder_len == 0
        assert runner.speculative_config is None
        # Without speculative decoding, no drafter attribute is installed.
        assert not hasattr(runner, "drafter")
        assert runner.use_async_scheduling is False
        assert runner.use_async_spec_decode is False
        # num_sms is platform-dependent; only its presence is guaranteed.
        assert isinstance(runner.num_sms, int)
        assert runner.sampled_token_ids_pinned_cpu.shape == (4, 1)
        assert runner.reorder_batch_threshold is None
        assert runner._draft_token_ids is None
        assert runner.mamba_state_idx == {}
        assert runner._mamba_bufs is None
        assert runner.layerwise_nvtx_hooks_registered is False
        assert runner.execute_model_state is None
        assert runner.shared_kv_cache_layers == {}
        assert runner.kv_sharing_fast_prefill_eligible_layers == set()
        assert runner.kv_sharing_fast_prefill_logits_indices is None
        assert runner.intermediate_tensors is None
        assert runner.valid_sampled_token_count_gpu is None
        assert runner.cudagraph_batch_sizes == []
        # The constructor registers the offloader globally (there is no
        # self.offloader attribute); a fresh NoopOffloader is installed.
        offloader_mod = import_module("vllm.model_executor.offloader")
        assert isinstance(offloader_mod.get_offloader(), offloader_mod.NoopOffloader)
        assert runner.encoder_cache == {}
        assert runner.requests == {}
        assert runner.num_prompt_logprobs == {}

    def test_num_query_heads_from_model_config_override(self, patched_module):
        """An explicit num_query_heads / inputs_embeds_size wins over defaults."""
        config = _make_vllm_config()
        config.model_config.get_num_attention_heads = lambda parallel_config: 6
        config.model_config.get_inputs_embeds_size = lambda: 12
        runner = _build_runner(patched_module, config)
        assert runner.num_query_heads == 6
        assert runner.inputs_embeds_size == 12

    def test_vendor_attention_flags_default_false(self, patched_module):
        """A vendor_name of None leaves use_ascend_attention False."""
        runner = _build_runner(patched_module, _make_vllm_config())
        assert runner.use_ascend_attention is False
        assert runner.graph_runtime is None
        assert runner.use_ascend_hybrid_cache is False

    def test_kv_cache_dtype_string(self, patched_module):
        """A cache_dtype string is translated through the real dtype helper."""
        config = _make_vllm_config()
        config.cache_config.cache_dtype = "bfloat16"
        runner = _build_runner(patched_module, config)
        assert runner.kv_cache_dtype == torch.bfloat16

    def test_is_pooling_model_flag(self, patched_module):
        """runner_type=pooling flows into is_pooling_model."""
        config = _make_vllm_config()
        config.model_config.runner_type = "pooling"
        runner = _build_runner(patched_module, config)
        assert runner.is_pooling_model is True

    def test_calculate_kv_scales_getattr_fallback(self, patched_module):
        """A cache_config without calculate_kv_scales falls back to False."""
        config = _make_vllm_config()
        del config.cache_config.calculate_kv_scales
        runner = _build_runner(patched_module, config)
        assert runner.calculate_kv_scales is False

    def test_calculate_kv_scales_true(self, patched_module):
        """An explicit calculate_kv_scales=True is honored."""
        config = _make_vllm_config()
        config.cache_config.calculate_kv_scales = True
        runner = _build_runner(patched_module, config)
        assert runner.calculate_kv_scales is True

    def test_mrope_model_grows_positions_buffer(self, patched_module):
        """A multimodal (mrope) model allocates the [3, max_num_tokens+1]
        mrope_positions buffer with the non-contiguity dummy column."""
        config = _make_vllm_config()
        config.model_config.uses_mrope = True
        runner = _build_runner(patched_module, config)
        assert runner.uses_mrope is True
        assert runner.mrope_positions.cpu.shape == (3, 65)
        assert runner.mrope_positions.gpu.shape == (3, 65)

    def test_kv_sharing_fast_prefill_allocates_index_buffer(self, patched_module):
        """kv_sharing_fast_prefill allocates the per-token logits index buffer."""
        config = _make_vllm_config()
        config.cache_config.kv_sharing_fast_prefill = True
        runner = _build_runner(patched_module, config)
        assert runner.kv_sharing_fast_prefill_logits_indices.shape == (64,)
        assert runner.kv_sharing_fast_prefill_logits_indices.dtype == torch.int32

    def test_encoder_decoder_reads_max_encoder_len(self, patched_module):
        """An encoder-decoder model reads max_encoder_len from the scheduler
        config and grows the InputBatch token capacity accordingly."""
        config = _make_vllm_config()
        config.model_config.is_encoder_decoder = True
        config.scheduler_config.max_num_encoder_input_tokens = 48
        runner = _build_runner(patched_module, config)
        assert runner.max_encoder_len == 48
        # max(cdiv over max(max_model_len, max_encoder_len)) block rows.
        assert runner.input_batch.token_ids_cpu.shape[1] >= 48

    def test_async_scheduling_creates_stream_and_event(self, patched_module):
        """Async scheduling allocates the output-copy stream and the blocking
        prepare_inputs event from the platform's torch_device_fn."""
        config = _make_vllm_config()
        config.scheduler_config.async_scheduling = True
        runner = _build_runner(patched_module, config)
        assert isinstance(runner.async_output_copy_stream, _FakeStream)
        assert isinstance(runner.prepare_inputs_event, _FakeEvent)
        assert runner.prepare_inputs_event.blocking is True
        assert runner.use_async_scheduling is True

    def test_cudagraph_capture_sizes_sorted(self, patched_module):
        """Unsorted capture sizes are sorted ascending in the runner."""
        config_mod = import_module("vllm.config")
        config = _make_vllm_config()
        config.compilation_config.cudagraph_mode = config_mod.CUDAGraphMode.FULL
        config.compilation_config.cudagraph_capture_sizes = [8, 2, 4]
        runner = _build_runner(patched_module, config)
        assert runner.cudagraph_batch_sizes == [2, 4, 8]

    def test_cudagraph_sizes_empty_when_no_capture_list(self, patched_module):
        """No capture list means no cudagraph batch sizes even in FULL mode."""
        config_mod = import_module("vllm.config")
        config = _make_vllm_config()
        config.compilation_config.cudagraph_mode = config_mod.CUDAGraphMode.FULL
        runner = _build_runner(patched_module, config)
        assert runner.cudagraph_batch_sizes == []

    def test_input_batch_uses_placeholder_block_size(self, patched_module):
        """The InputBatch is created with the placeholder block size of 16."""
        runner = _build_runner(patched_module, _make_vllm_config())
        assert runner.input_batch.block_table[0].block_size == 16
        assert runner._init_block_sizes == [16]

    def test_input_batch_block_size_falls_back_to_default(self, patched_module):
        """A None block_size falls back to CacheConfig.DEFAULT_BLOCK_SIZE."""
        config = _make_vllm_config()
        config.cache_config.block_size = None
        runner = _build_runner(patched_module, config)
        default_bs = import_module("vllm.config").CacheConfig.DEFAULT_BLOCK_SIZE
        assert runner._init_block_sizes == [default_bs]

    def test_query_pos_and_arange_shapes(self, patched_module):
        """query_pos is sized max(max_num_reqs + 1, max_num_tokens)."""
        runner = _build_runner(patched_module, _make_vllm_config())
        assert runner.query_pos.cpu.shape == (64,)
        assert runner.query_pos.cpu.dtype == torch.int64

    def test_offloader_set_globally(self, patched_module):
        """The default OffloadConfig installs the NoopOffloader sentinel."""
        offloader_mod = import_module("vllm.model_executor.offloader")
        sentinel = SimpleNamespace(marker="before")
        offloader_mod.set_offloader(sentinel)
        try:
            _build_runner(patched_module, _make_vllm_config())
        finally:
            # Restore the real no-op offloader for other tests.
            offloader_mod.set_offloader(offloader_mod.NoopOffloader())
        # The constructor replaced our sentinel with a fresh offloader.
        current = offloader_mod.get_offloader()
        assert current is not sentinel
        assert isinstance(current, offloader_mod.NoopOffloader)


# ---------------------------------------------------------------------------
# __init__: spec-decode drafter dispatch.
# ---------------------------------------------------------------------------


class TestInitSpecDecode:
    """ModelRunnerFL.__init__ drafter selection across spec-decode methods."""

    def test_custom_class_proposer(self, patched_module, monkeypatch):
        """A custom class path constructs the proposer via create_custom_proposer."""
        created = []

        class _FakeCustom:
            def __init__(self, vllm_config):
                created.append(vllm_config)
                self.kv_cache_gid = 0

        monkeypatch.setattr(patched_module, "create_custom_proposer", _FakeCustom)
        config = _make_vllm_config(
            speculative_config=_make_spec_config("custom_class"),
        )
        runner = _build_runner(patched_module, config)
        assert isinstance(runner.drafter, _FakeCustom)
        assert created[0] is config

    def test_draft_model_proposer(self, patched_module, monkeypatch):
        """The draft-model path constructs DraftModelProposer with the config,
        device and runner."""
        created = []

        class _FakeDraft:
            def __init__(self, vllm_config, device, runner):
                created.append((vllm_config, device, runner))
                self.kv_cache_gid = 0

        monkeypatch.setattr(patched_module, "DraftModelProposer", _FakeDraft)
        config = _make_vllm_config(
            speculative_config=_make_spec_config(
                "draft", uses_draft_model=lambda: True
            ),
        )
        runner = _build_runner(patched_module, config)
        assert isinstance(runner.drafter, _FakeDraft)
        vllm_config_arg, device_arg, runner_arg = created[0]
        assert vllm_config_arg is config
        assert device_arg.type == "cpu"
        assert runner_arg is runner

    def test_eagle_proposer(self, patched_module, monkeypatch):
        """The eagle method selects EagleProposer on the module."""

        class _FakeEagle:
            def __init__(self, vllm_config, device, runner):
                self.kv_cache_gid = 0

        monkeypatch.setattr(patched_module, "EagleProposer", _FakeEagle)
        config = _make_vllm_config(
            speculative_config=_make_spec_config("eagle", use_eagle=lambda: True),
        )
        runner = _build_runner(patched_module, config)
        assert isinstance(runner.drafter, _FakeEagle)
        assert runner.use_aux_hidden_state_outputs is False

    def test_eagle3_propagates_aux_hidden_state(self, patched_module, monkeypatch):
        """eagle3 propagates the drafter's eagle3_use_aux_hidden_state flag."""

        class _FakeEagle:
            eagle3_use_aux_hidden_state = True

            def __init__(self, vllm_config, device, runner):
                self.kv_cache_gid = 0

        monkeypatch.setattr(patched_module, "EagleProposer", _FakeEagle)
        config = _make_vllm_config(
            speculative_config=_make_spec_config("eagle3", use_eagle=lambda: True),
        )
        runner = _build_runner(patched_module, config)
        assert isinstance(runner.drafter, _FakeEagle)
        assert runner.use_aux_hidden_state_outputs is True

    def test_suffix_proposer(self, patched_module, monkeypatch):
        """The suffix method constructs SuffixDecodingProposer."""

        class _FakeSuffix:
            def __init__(self, vllm_config):
                self.kv_cache_gid = 0

        monkeypatch.setattr(patched_module, "SuffixDecodingProposer", _FakeSuffix)
        config = _make_vllm_config(
            speculative_config=_make_spec_config("suffix"),
        )
        runner = _build_runner(patched_module, config)
        assert isinstance(runner.drafter, _FakeSuffix)

    def test_medusa_proposer(self, patched_module, monkeypatch):
        """The medusa method constructs MedusaProposer with keywords."""

        class _FakeMedusa:
            def __init__(self, vllm_config, device):
                self.kv_cache_gid = 0

        monkeypatch.setattr(patched_module, "MedusaProposer", _FakeMedusa)
        config = _make_vllm_config(
            speculative_config=_make_spec_config("medusa"),
        )
        runner = _build_runner(patched_module, config)
        assert isinstance(runner.drafter, _FakeMedusa)

    def test_extract_hidden_states_sets_aux_outputs(self, patched_module, monkeypatch):
        """extract_hidden_states constructs the proposer and turns on aux
        hidden-state outputs."""

        class _FakeExtract:
            def __init__(self, vllm_config, device):
                self.kv_cache_gid = 0

        monkeypatch.setattr(patched_module, "ExtractHiddenStatesProposer", _FakeExtract)
        config = _make_vllm_config(
            speculative_config=_make_spec_config("extract_hidden_states"),
        )
        runner = _build_runner(patched_module, config)
        assert isinstance(runner.drafter, _FakeExtract)
        assert runner.use_aux_hidden_state_outputs is True

    def test_gemma4_mtp_proposer(self, patched_module, monkeypatch):
        """use_gemma4_mtp constructs Gemma4Proposer."""

        class _FakeGemma4:
            def __init__(self, vllm_config, device, runner):
                self.kv_cache_gid = 0

        monkeypatch.setattr(patched_module, "Gemma4Proposer", _FakeGemma4)
        config = _make_vllm_config(
            speculative_config=_make_spec_config(
                "gemma4_mtp", use_gemma4_mtp=lambda: True
            ),
        )
        runner = _build_runner(patched_module, config)
        assert isinstance(runner.drafter, _FakeGemma4)

    def test_step3p5_mtp_proposer(self, patched_module, monkeypatch):
        """use_step3p5_mtp constructs Step3p5MTPProposer."""

        class _FakeStep3p5:
            def __init__(self, vllm_config, device, runner):
                self.kv_cache_gid = 0

        monkeypatch.setattr(patched_module, "Step3p5MTPProposer", _FakeStep3p5)
        config = _make_vllm_config(
            speculative_config=_make_spec_config(
                "step3p5_mtp", use_step3p5_mtp=lambda: True
            ),
        )
        runner = _build_runner(patched_module, config)
        assert isinstance(runner.drafter, _FakeStep3p5)

    def test_dflash_proposer_sets_aux_outputs(self, patched_module, monkeypatch):
        """use_dflash constructs DFlashProposer and enables aux outputs."""

        class _FakeDFlash:
            def __init__(self, vllm_config, device, runner):
                self.kv_cache_gid = 0

        monkeypatch.setattr(patched_module, "DFlashProposer", _FakeDFlash)
        config = _make_vllm_config(
            speculative_config=_make_spec_config("dflash", use_dflash=lambda: True),
        )
        runner = _build_runner(patched_module, config)
        assert isinstance(runner.drafter, _FakeDFlash)
        assert runner.use_aux_hidden_state_outputs is True

    def test_unknown_method_raises(self, patched_module):
        """An unrecognized spec method surfaces the constructor's ValueError."""
        config = _make_vllm_config(
            speculative_config=_make_spec_config("bogus"),
        )
        with pytest.raises(ValueError, match="Unknown speculative decoding method"):
            _build_runner(patched_module, config)

    def test_spec_decode_shared_fields(self, patched_module, monkeypatch):
        """Spec decode sets num_spec_tokens, the sampler and uniform query len."""

        class _FakeEagle:
            def __init__(self, vllm_config, device, runner):
                self.kv_cache_gid = 0

        monkeypatch.setattr(patched_module, "EagleProposer", _FakeEagle)
        monkeypatch.setattr(
            patched_module,
            "RejectionSampler",
            lambda sampler, spec, device: SimpleNamespace(fake=True),
        )
        config = _make_vllm_config(
            speculative_config=_make_spec_config("eagle", use_eagle=lambda: True),
        )
        runner = _build_runner(patched_module, config)
        assert runner.num_spec_tokens == 2
        assert runner.prev_num_spec_tokens == 2
        assert runner.uniform_decode_query_len == 3
        assert runner.rejection_sampler.fake is True
        assert runner.use_async_spec_decode is False
        # No draft model config: effective len mirrors max_model_len.
        assert runner.effective_drafter_max_model_len == 32

    def test_spec_decode_with_draft_model_len(self, patched_module, monkeypatch):
        """A draft model with its own max_model_len sets the effective len."""

        class _FakeEagle:
            def __init__(self, vllm_config, device, runner):
                self.kv_cache_gid = 0

        monkeypatch.setattr(patched_module, "EagleProposer", _FakeEagle)
        monkeypatch.setattr(
            patched_module,
            "RejectionSampler",
            lambda sampler, spec, device: SimpleNamespace(fake=True),
        )
        config = _make_vllm_config(
            speculative_config=_make_spec_config(
                "eagle",
                use_eagle=lambda: True,
                draft_model_config=SimpleNamespace(max_model_len=24),
            ),
        )
        runner = _build_runner(patched_module, config)
        assert runner.effective_drafter_max_model_len == 24

    def test_async_spec_decode_on_with_async_scheduling(
        self, patched_module, monkeypatch
    ):
        """Async scheduling plus spec decode turns on async spec decode."""

        class _FakeEagle:
            def __init__(self, vllm_config, device, runner):
                self.kv_cache_gid = 0

        monkeypatch.setattr(patched_module, "EagleProposer", _FakeEagle)
        monkeypatch.setattr(
            patched_module,
            "RejectionSampler",
            lambda sampler, spec, device: SimpleNamespace(fake=True),
        )
        config = _make_vllm_config(
            speculative_config=_make_spec_config("eagle", use_eagle=lambda: True),
        )
        config.scheduler_config.async_scheduling = True
        runner = _build_runner(patched_module, config)
        assert runner.use_async_spec_decode is True
        # Async + spec decode also allocates the valid-count buffers.
        assert runner.valid_sampled_token_count_cpu is not None
        assert runner.valid_sampled_token_count_event is not None

    def test_spec_decode_event_allocations(self, patched_module, monkeypatch):
        """Spec decode allocates the draft-token event, copy stream and CPU
        buffer sized (max_num_reqs, num_spec_tokens)."""

        class _FakeEagle:
            def __init__(self, vllm_config, device, runner):
                self.kv_cache_gid = 0

        monkeypatch.setattr(patched_module, "EagleProposer", _FakeEagle)
        monkeypatch.setattr(
            patched_module,
            "RejectionSampler",
            lambda sampler, spec, device: SimpleNamespace(fake=True),
        )
        config = _make_vllm_config(
            speculative_config=_make_spec_config("eagle", use_eagle=lambda: True),
        )
        runner = _build_runner(patched_module, config)
        assert runner.draft_token_ids_event is not None
        assert runner.num_accepted_tokens_event is not None
        assert isinstance(runner.draft_token_ids_copy_stream, _FakeStream)
        assert runner.draft_token_ids_cpu.shape == (4, 2)
        # Without async scheduling the valid-count buffers stay absent.
        assert runner.valid_sampled_token_count_event is None
        assert runner.valid_sampled_token_count_cpu is None


# ---------------------------------------------------------------------------
# __init__: ngram-GPU spec decode buffers.
# ---------------------------------------------------------------------------


class TestInitNgramGpu:
    """The ngram-GPU drafter branch and its pinned buffers."""

    @pytest.fixture(autouse=True)
    def _pin(self, monkeypatch):
        """Translate pin_memory=True allocations to host memory (CPU-only box)."""
        real_zeros = torch.zeros
        real_empty = torch.empty

        def _zeros(*args, **kwargs):
            if kwargs.get("pin_memory"):
                kwargs = dict(kwargs)
                kwargs["pin_memory"] = False
            return real_zeros(*args, **kwargs)

        def _empty(*args, **kwargs):
            if kwargs.get("pin_memory"):
                kwargs = dict(kwargs)
                kwargs["pin_memory"] = False
            return real_empty(*args, **kwargs)

        monkeypatch.setattr(torch, "zeros", _zeros)
        monkeypatch.setattr(torch, "empty", _empty)

    @staticmethod
    def _config(patched_module, monkeypatch):
        """A config + module patches for the ngram-GPU branch."""
        created_sampler = []
        created_proposer = []

        class _FakeSampler:
            def __init__(self, *args, **kwargs):
                created_sampler.append(args)

        class _FakeNgramGPU:
            def __init__(self, *args, **kwargs):
                created_proposer.append(args)

        monkeypatch.setattr(patched_module, "RejectionSampler", _FakeSampler)
        monkeypatch.setattr(patched_module, "NgramProposerGPU", _FakeNgramGPU)
        config = _make_vllm_config(
            speculative_config=_make_spec_config(
                "ngram_gpu", use_ngram_gpu=lambda: True
            ),
        )
        return config, created_sampler, created_proposer, _FakeSampler, _FakeNgramGPU

    def test_ngram_gpu_creates_buffers(self, patched_module, monkeypatch):
        """The ngram-GPU branch allocates the four bookkeeping tensors."""
        config, _, _, _, _ = self._config(patched_module, monkeypatch)
        runner = _build_runner(patched_module, config)
        assert runner._ngram_pinned_idx_buf.shape == (4,)
        assert runner._ngram_pinned_idx_buf.dtype == torch.long
        assert runner.token_ids_gpu_tensor.shape == (4, 32)
        assert runner.num_tokens_no_spec_gpu.shape == (4,)
        assert runner.num_tokens_no_spec_gpu.dtype == torch.int32
        assert runner._ngram_pinned_val_buf.shape == (4,)
        assert isinstance(runner._num_valid_draft_tokens_event, _FakeEvent)
        assert isinstance(runner._num_valid_draft_tokens_copy_stream, _FakeStream)

    def test_ngram_gpu_proposer_and_sampler_created(self, patched_module, monkeypatch):
        """Both the GPU proposer and the rejection sampler are constructed."""
        config, sampler_calls, proposer_calls, sampler_cls, proposer_cls = self._config(
            patched_module, monkeypatch
        )
        runner = _build_runner(patched_module, config)
        assert isinstance(runner.drafter, proposer_cls)
        assert len(sampler_calls) == 1
        assert len(proposer_calls) == 1
        assert isinstance(runner.rejection_sampler, sampler_cls)

    def test_ngram_gpu_async_extra_buffers(self, patched_module, monkeypatch):
        """Async + ngram-GPU adds the valid-count copy stream and CPU buffer."""
        config, _, _, _, _ = self._config(patched_module, monkeypatch)
        config.scheduler_config.async_scheduling = True
        runner = _build_runner(patched_module, config)
        assert isinstance(runner._num_valid_draft_tokens_copy_stream, _FakeStream)
        assert runner._num_valid_draft_tokens_cpu is not None
        assert isinstance(runner._num_valid_draft_tokens_event, _FakeEvent)


# ---------------------------------------------------------------------------
# __init__: multimodal budget.
# ---------------------------------------------------------------------------


class TestInitMultimodal:
    """supports_mm_inputs wiring for the multimodal budget."""

    def test_mm_budget_created_when_supported(self, patched_module, monkeypatch):
        """A model advertising multimodal inputs gets a MultiModalBudget."""
        created = []

        class _FakeBudget:
            def __init__(self, vllm_config, registry):
                created.append((vllm_config, registry))

        monkeypatch.setattr(patched_module, "MultiModalBudget", _FakeBudget)
        config = _make_vllm_config()
        config.model_config.supports_multimodal_inputs = True
        runner = _build_runner(patched_module, config)
        assert isinstance(runner.mm_budget, _FakeBudget)
        assert created[0][0] is config
        assert created[0][1] is patched_module.MULTIMODAL_REGISTRY

    def test_mm_budget_absent_when_unsupported(self, patched_module):
        """No multimodal support means no budget object."""
        config = _make_vllm_config()
        config.model_config.supports_multimodal_inputs = False
        runner = _build_runner(patched_module, config)
        assert runner.mm_budget is None

    def test_mm_budget_absent_with_getattr_fallback(self, patched_module, monkeypatch):
        """A model config lacking the field falls back to the registry check."""
        MULTIMODAL_REGISTRY = patched_module.MULTIMODAL_REGISTRY
        monkeypatch.setattr(
            MULTIMODAL_REGISTRY,
            "supports_multimodal_inputs",
            lambda model_config: False,
            raising=False,
        )
        config = _make_vllm_config()
        del config.model_config.supports_multimodal_inputs
        runner = _build_runner(patched_module, config)
        assert runner.mm_budget is None
        assert runner.supports_mm_inputs is False

    def test_mm_budget_via_registry_fallback(self, patched_module, monkeypatch):
        """The registry fallback path can also approve multimodal support."""

        class _FakeBudget:
            def __init__(self, vllm_config, registry):
                pass

        monkeypatch.setattr(patched_module, "MultiModalBudget", _FakeBudget)
        MULTIMODAL_REGISTRY = patched_module.MULTIMODAL_REGISTRY
        monkeypatch.setattr(
            MULTIMODAL_REGISTRY,
            "supports_multimodal_inputs",
            lambda model_config: True,
            raising=False,
        )
        config = _make_vllm_config()
        del config.model_config.supports_multimodal_inputs
        runner = _build_runner(patched_module, config)
        assert isinstance(runner.mm_budget, _FakeBudget)
        assert runner.supports_mm_inputs is True


# ---------------------------------------------------------------------------
# __init__: pipeline / data-parallel branches.
# ---------------------------------------------------------------------------


class TestInitParallel:
    """Pipeline-parallel and data-parallel derived flags."""

    def test_broadcast_pp_output_single_rank(self, patched_module):
        """A single-rank pipeline never broadcasts regardless of launcher."""
        config = _make_vllm_config()
        config.parallel_config.distributed_executor_backend = "external_launcher"
        runner = _build_runner(patched_module, config)
        assert runner.broadcast_pp_output is False

    def test_broadcast_pp_output_external_launcher_multi_rank(
        self, patched_module, monkeypatch
    ):
        """An external launcher on a multi-rank pipeline turns broadcasting on."""
        pp_group = SimpleNamespace(is_last_rank=True, is_first_rank=False, ranks=[0, 1])
        monkeypatch.setattr(patched_module, "get_pp_group", lambda: pp_group)
        config = _make_vllm_config()
        config.parallel_config.distributed_executor_backend = "external_launcher"
        runner = _build_runner(patched_module, config)
        assert runner.broadcast_pp_output is True

    def test_broadcast_pp_output_mp_backend_multi_rank(
        self, patched_module, monkeypatch
    ):
        """A non-external backend on a multi-rank pipeline stays off."""
        pp_group = SimpleNamespace(is_last_rank=True, is_first_rank=False, ranks=[0, 1])
        monkeypatch.setattr(patched_module, "get_pp_group", lambda: pp_group)
        runner = _build_runner(patched_module, _make_vllm_config())
        assert runner.broadcast_pp_output is False

    def test_check_ep_fault_dp_gt_1_and_is_moe(self, patched_module, monkeypatch):
        """dp > 1 with an MoE model reads the fault tolerance flag."""
        ep_manager = SimpleNamespace(support_fault_tolerance=True)
        monkeypatch.setattr(
            patched_module, "get_ep_all2all_manager", lambda: ep_manager
        )
        config = _make_vllm_config()
        config.parallel_config.data_parallel_size = 2
        config.model_config.is_moe = True
        runner = _build_runner(patched_module, config)
        assert runner.check_ep_fault is True

    def test_check_ep_fault_disabled_without_moe(self, patched_module):
        """dp > 1 without MoE leaves ep fault checking off."""
        config = _make_vllm_config()
        config.parallel_config.data_parallel_size = 2
        config.model_config.is_moe = False
        runner = _build_runner(patched_module, config)
        assert runner.check_ep_fault is False


# ---------------------------------------------------------------------------
# __init__: dcp world size > 1.
# ---------------------------------------------------------------------------


class TestInitDcp:
    """The dcp (decoupled context parallel) buffer branch."""

    def test_dcp_world_size_gt_1_creates_buffer(self, patched_module, monkeypatch):
        """dcp_world_size > 1 reads the group rank and allocates the buffer."""
        dcp_group = SimpleNamespace(rank_in_group=0)
        monkeypatch.setattr(patched_module, "get_dcp_group", lambda: dcp_group)
        config = _make_vllm_config()
        config.parallel_config.decode_context_parallel_size = 2
        runner = _build_runner(patched_module, config)
        assert runner.dcp_world_size == 2
        assert runner.dcp_rank == 0
        assert runner.dcp_local_seq_lens.cpu.shape == (4,)

    def test_dcp_world_size_1_no_buffer(self, patched_module):
        """A single-rank dcp group skips the buffer entirely."""
        runner = _build_runner(patched_module, _make_vllm_config())
        assert runner.dcp_world_size == 1
        assert runner.dcp_rank == 0
        assert not hasattr(runner, "dcp_local_seq_lens")


# ---------------------------------------------------------------------------
# _get_mamba_bufs.
# ---------------------------------------------------------------------------


class TestGetMambaBufs:
    """_get_mamba_bufs dispatch and the caching behavior."""

    @staticmethod
    def _runner(speculative_config=None):
        return SimpleNamespace(
            _mamba_bufs=None,
            speculative_config=speculative_config,
            model_config=SimpleNamespace(is_hybrid=True),
            cache_config=SimpleNamespace(mamba_cache_mode="align"),
            max_num_reqs=4,
            kv_cache_config=SimpleNamespace(),
            device=torch.device(CPU_DEVICE),
            _get_mamba_state_copy_funcs=lambda: {"mamba": "copy"},
            _make_buffer=lambda *a, **kw: SimpleNamespace(fake_buffer=True),
        )

    def test_align_mode_creates_and_caches(self, patched_module, monkeypatch):
        """The align cache mode creates MambaBuffers once and caches."""
        calls = []

        class _FakeBufs:
            @classmethod
            def create(cls, **kwargs):
                calls.append(kwargs)
                return SimpleNamespace(kwargs_record=kwargs)

        fake_mamba_utils = SimpleNamespace(MambaBuffers=_FakeBufs)
        monkeypatch.setattr(patched_module, "mamba_utils", fake_mamba_utils)
        runner = self._runner()
        first = patched_module.ModelRunnerFL._get_mamba_bufs(runner)
        second = patched_module.ModelRunnerFL._get_mamba_bufs(runner)
        assert first is second
        assert len(calls) == 1
        assert calls[0]["max_num_reqs"] == 4
        assert calls[0]["with_postprocess_align"] is False
        assert calls[0]["copy_funcs"] == {"mamba": "copy"}

    def test_align_with_spec_config_postprocess(self, patched_module, monkeypatch):
        """A spec config plus hybrid model enables postprocess alignment."""
        calls = []

        class _FakeBufs:
            @classmethod
            def create(cls, **kwargs):
                calls.append(kwargs)
                return SimpleNamespace(kwargs_record=kwargs)

        fake_mamba_utils = SimpleNamespace(MambaBuffers=_FakeBufs)
        monkeypatch.setattr(patched_module, "mamba_utils", fake_mamba_utils)
        runner = self._runner(
            speculative_config=SimpleNamespace(num_speculative_tokens=2)
        )
        patched_module.ModelRunnerFL._get_mamba_bufs(runner)
        assert calls[0]["with_postprocess_align"] is True

    def test_wrong_cache_mode_asserts(self, patched_module):
        """A non-align cache mode trips the method's assertion."""
        runner = self._runner()
        runner.cache_config.mamba_cache_mode = "streaming"
        with pytest.raises(AssertionError):
            patched_module.ModelRunnerFL._get_mamba_bufs(runner)


# ---------------------------------------------------------------------------
# Small helpers.
# ---------------------------------------------------------------------------


class TestLightHelpers:
    """One-shot helpers around the model runner lifecycle."""

    def test_reset_mm_cache(self, patched_module):
        """reset_mm_cache delegates to the mm budget and the late runner."""
        budget_calls = []
        lir_clears = []
        runner = SimpleNamespace(
            mm_budget=SimpleNamespace(reset_cache=lambda: budget_calls.append(1)),
            late_interaction_runner=SimpleNamespace(clear=lambda: lir_clears.append(1)),
        )
        patched_module.ModelRunnerFL.reset_mm_cache(runner)
        assert budget_calls == [1]
        assert lir_clears == [1]

    def test_reset_mm_cache_no_budget(self, patched_module):
        """A None mm_budget skips the reset; the late runner still clears."""
        lir_clears = []
        runner = SimpleNamespace(
            mm_budget=None,
            late_interaction_runner=SimpleNamespace(clear=lambda: lir_clears.append(1)),
        )
        patched_module.ModelRunnerFL.reset_mm_cache(runner)
        assert lir_clears == [1]

    def test_reset_encoder_cache(self, patched_module):
        """reset_encoder_cache clears the encoder cache and the late runner."""
        encoder_clears = []
        lir_clears = []
        cache = {"a": 1}

        class _Dict(dict):
            pass

        encoder_cache = _Dict(cache)
        encoder_cache.clear = lambda: encoder_clears.append(1)
        runner = SimpleNamespace(
            encoder_cache=encoder_cache,
            late_interaction_runner=SimpleNamespace(clear=lambda: lir_clears.append(1)),
        )
        patched_module.ModelRunnerFL.reset_encoder_cache(runner)
        assert encoder_clears == [1]
        assert lir_clears == [1]

    def test_post_kv_cache_wake_up(self, patched_module):
        """post_kv_cache_wake_up delegates to init_fp8_kv_scales."""
        calls = []
        runner = SimpleNamespace(init_fp8_kv_scales=lambda: calls.append(1))
        patched_module.ModelRunnerFL.post_kv_cache_wake_up(runner)
        assert calls == [1]

    def test_make_buffer(self, patched_module):
        """_make_buffer builds a CpuGpuBuffer with numpy view by default."""
        runner = SimpleNamespace(device=torch.device(CPU_DEVICE))
        buf = patched_module.ModelRunnerFL._make_buffer(runner, 3, 5, dtype=torch.int32)
        assert buf.cpu.shape == (3, 5)
        assert buf.cpu.dtype == torch.int32

    def test_make_buffer_without_numpy(self, patched_module):
        """numpy=False skips the numpy mirror (needed for bfloat16 embeds)."""
        runner = SimpleNamespace(device=torch.device(CPU_DEVICE))
        buf = patched_module.ModelRunnerFL._make_buffer(
            runner, 4, dtype=torch.bfloat16, numpy=False
        )
        assert buf.cpu.shape == (4,)
        assert buf.cpu.dtype == torch.bfloat16
        assert getattr(buf, "np", None) is None

    def test_init_device_properties(self, patched_module):
        """_init_device_properties records the compute-unit count."""
        runner = SimpleNamespace(device=torch.device(CPU_DEVICE))
        patched_module.ModelRunnerFL._init_device_properties(runner)
        assert isinstance(runner.num_sms, int)
        assert runner.num_sms >= 1

    def test_sync_device(self, patched_module, monkeypatch):
        """_sync_device routes through the module's accelerator sync."""
        calls = []
        monkeypatch.setattr(
            patched_module, "_accelerator_synchronize", lambda: calls.append(1)
        )
        runner = SimpleNamespace()
        patched_module.ModelRunnerFL._sync_device(runner)
        assert calls == [1]

    def test_on_request_state_removed_is_noop(self, patched_module):
        """_on_request_state_removed is a hook no-op."""
        runner = SimpleNamespace()
        result = patched_module.ModelRunnerFL._on_request_state_removed(
            runner, "r1", None
        )
        assert result is None

    def test_zero_block_ids_with_kv_zeroer(self, patched_module):
        """_zero_block_ids delegates to the KVBlockZeroer when present."""
        zeroed = []

        class _FakeZeroer:
            def zero_block_ids(self, ids):
                zeroed.append(list(ids))

        runner = SimpleNamespace(_kv_block_zeroer=_FakeZeroer())
        patched_module.ModelRunnerFL._zero_block_ids(runner, [1, 2, 3])
        assert zeroed == [[1, 2, 3]]

    def test_zero_block_ids_without_kv_zeroer(self, patched_module):
        """A missing _kv_block_zeroer keeps _zero_block_ids as a no-op."""
        runner = SimpleNamespace()
        # Must not raise.
        patched_module.ModelRunnerFL._zero_block_ids(runner, [1])

    def test_process_encoder_cache_scheduler_output(self, patched_module):
        """Free encoder hashes are popped from the encoder cache."""
        runner = SimpleNamespace(encoder_cache={"h1": 1, "h2": 2})
        sched = SimpleNamespace(free_encoder_mm_hashes=["h1"])
        patched_module.ModelRunnerFL._process_encoder_cache_scheduler_output(
            runner, sched
        )
        assert "h1" not in runner.encoder_cache
        assert "h2" in runner.encoder_cache

    def test_count_nans_per_row(self, patched_module):
        """NaN counting matches an independent row-by-row oracle."""
        t = torch.tensor([[1.0, float("nan")], [float("nan"), float("nan")]])
        counts = patched_module.count_nans_per_row(t)
        # Independent oracle: manual row scan.
        expected = [sum(1 for v in row if v != v) for row in t.tolist()]
        assert counts.tolist() == expected == [1, 2]

    def test_prepare_comm_buffer_returns_none(self, patched_module):
        """On vLLM 0.32 _prepare_comm_buffer is the ImportError fallback."""
        result = patched_module._prepare_comm_buffer()
        assert result is None


# ---------------------------------------------------------------------------
# _update_states: new-block zeroing, KV copies, generators, pooling.
# ---------------------------------------------------------------------------


def _update_states_sched(**fields):
    """A SchedulerOutput-shaped namespace for the _update_states extras."""
    base = dict(
        finished_req_ids=[],
        new_block_ids_to_zero=None,
        kv_cache_block_copies=None,
        free_encoder_mm_hashes=[],
        num_scheduled_tokens={},
        scheduled_new_reqs=[],
        scheduled_spec_decode_tokens={},
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=[],
            resumed_req_ids=set(),
            new_token_ids=[],
            all_token_ids={},
            new_block_ids=[],
            num_computed_tokens=[],
            num_output_tokens=[],
        ),
    )
    base.update(fields)
    return SimpleNamespace(**base)


class _NullLIR:
    """Recording stand-in for the LateInteractionRunner boundary."""

    def __init__(self):
        self.finished_calls = []
        self.registered = []

    def on_requests_finished(self, req_ids):
        self.finished_calls.append(list(req_ids))

    def register_request(self, req_id, pooling_params):
        self.registered.append((req_id, pooling_params))

    def clear(self):
        pass


class TestUpdateStatesExtras:
    """_update_states branches beyond the core suite's coverage."""

    @staticmethod
    def _runner(patched_module, *, device=None):
        """A SimpleNamespace runner with the update-states attributes."""
        return SimpleNamespace(
            requests={},
            num_prompt_logprobs={},
            late_interaction_runner=_NullLIR(),
            input_batch=SimpleNamespace(
                num_reqs=0,
                req_id_to_index={},
                prev_req_id_to_index=None,
                num_computed_tokens_cpu=np.zeros(8, dtype=np.int32),
                num_tokens_no_spec=np.zeros(8, dtype=np.int32),
                vocab_size=100,
                remove_request=lambda rid: None,
                condense=lambda: None,
                refresh_metadata=lambda: None,
                add_request=lambda r: None,
                update_req_spec_token_ids=lambda r, s: None,
                append_row=lambda ids, idx: None,
            ),
            _on_request_state_removed=lambda *a: None,
            _zero_block_ids=lambda ids: None,
            _process_encoder_cache_scheduler_output=lambda so: None,
            _may_reorder_batch=lambda so: None,
            speculative_config=None,
            use_async_scheduling=False,
            use_async_spec_decode=False,
            prev_num_draft_tokens=SimpleNamespace(np=np.zeros(8, dtype=np.int32)),
            is_pooling_model=False,
            device=device or torch.device(CPU_DEVICE),
            uses_mrope=False,
        )

    def test_new_block_ids_to_zero_delegates(self, patched_module):
        """new_block_ids_to_zero flows to _zero_block_ids unchanged."""
        zeroed = []
        runner = self._runner(patched_module)
        runner._zero_block_ids = lambda ids: zeroed.append(list(ids))
        sched = _update_states_sched(new_block_ids_to_zero=[[10, 11], [12]])
        patched_module.ModelRunnerFL._update_states(runner, sched)
        assert zeroed == [[[10, 11], [12]]]

    def test_kv_cache_block_copies_invoked(self, patched_module, monkeypatch):
        """kv_cache_block_copies routes through copy_kv_cache_blocks_inplace."""
        calls = []

        def _copy(kv, num_blocks, copies):
            calls.append((kv, num_blocks, copies))

        monkeypatch.setattr(patched_module, "copy_kv_cache_blocks_inplace", _copy)
        runner = self._runner(patched_module)
        runner.kv_caches = ["kv"]
        runner.kv_cache_config = SimpleNamespace(num_blocks=10)
        copies = [SimpleNamespace(src=1, tgt=2)]
        sched = _update_states_sched(kv_cache_block_copies=copies)
        patched_module.ModelRunnerFL._update_states(runner, sched)
        assert calls == [(["kv"], 10, copies)]

    def test_finished_request_pops_prompt_logprobs(self, patched_module):
        """A finished request also leaves num_prompt_logprobs."""
        runner = self._runner(patched_module)
        runner.requests["f1"] = SimpleNamespace(req_id="f1")
        runner.num_prompt_logprobs["f1"] = 5
        sched = _update_states_sched(finished_req_ids=["f1"])
        patched_module.ModelRunnerFL._update_states(runner, sched)
        assert "f1" not in runner.requests
        assert "f1" not in runner.num_prompt_logprobs
        assert runner.late_interaction_runner.finished_calls == [["f1"]]

    def test_random_seed_generator_cpu_device(self, patched_module):
        """A RANDOM_SEED request on the cpu device gets a cpu generator."""
        SamplingParams = import_module("vllm.sampling_params").SamplingParams
        runner = self._runner(patched_module)
        new_req = SimpleNamespace(
            req_id="new1",
            prompt_token_ids=[1, 2, 3],
            prompt_embeds=None,
            prompt_is_token_ids=True,
            mm_features=[],
            sampling_params=SamplingParams(seed=7),
            pooling_params=None,
            block_ids=([1],),
            num_computed_tokens=0,
            lora_request=None,
        )
        sched = _update_states_sched(
            num_scheduled_tokens={"new1": 3}, scheduled_new_reqs=[new_req]
        )
        patched_module.ModelRunnerFL._update_states(runner, sched)
        gen = runner.requests["new1"].generator
        assert isinstance(gen, torch.Generator)
        assert gen.device.type == "cpu"
        assert runner.late_interaction_runner.registered == [("new1", None)]

    def test_random_seed_generator_txda_device(self, patched_module):
        """The txda device branch routes the generator to cpu."""
        SamplingParams = import_module("vllm.sampling_params").SamplingParams
        runner = self._runner(patched_module)
        # ``device.type == "txda"`` selects the cpu generator unconditionally.
        runner.device = SimpleNamespace(type="txda")
        new_req = SimpleNamespace(
            req_id="new1",
            prompt_token_ids=[1, 2, 3],
            prompt_embeds=None,
            prompt_is_token_ids=True,
            mm_features=[],
            sampling_params=SamplingParams(seed=7),
            pooling_params=None,
            block_ids=([1],),
            num_computed_tokens=0,
            lora_request=None,
        )
        sched = _update_states_sched(
            num_scheduled_tokens={"new1": 3}, scheduled_new_reqs=[new_req]
        )
        patched_module.ModelRunnerFL._update_states(runner, sched)
        gen = runner.requests["new1"].generator
        assert isinstance(gen, torch.Generator)
        assert gen.device.type == "cpu"

    def test_greedy_request_gets_no_generator(self, patched_module):
        """A GREEDY sampling request records no generator."""
        SamplingParams = import_module("vllm.sampling_params").SamplingParams
        SamplingType = import_module("vllm.sampling_params").SamplingType
        runner = self._runner(patched_module)
        params = SamplingParams()
        params.sampling_type = SamplingType.GREEDY
        new_req = SimpleNamespace(
            req_id="new1",
            prompt_token_ids=[1, 2, 3],
            prompt_embeds=None,
            prompt_is_token_ids=True,
            mm_features=[],
            sampling_params=params,
            pooling_params=None,
            block_ids=([1],),
            num_computed_tokens=0,
            lora_request=None,
        )
        sched = _update_states_sched(
            num_scheduled_tokens={"new1": 3}, scheduled_new_reqs=[new_req]
        )
        patched_module.ModelRunnerFL._update_states(runner, sched)
        assert runner.requests["new1"].generator is None

    def test_pooling_request_applies_pooling_updates(self, patched_module):
        """A pooling request routes through the model's pooler updates."""
        applied = []

        class _FakePooler:
            def get_pooling_updates(self, task):
                return SimpleNamespace(
                    apply=lambda params: applied.append((task, params))
                )

        class _FakeModel:
            pooler = _FakePooler()

        runner = self._runner(patched_module)
        runner.is_pooling_model = True
        runner.get_model = lambda: _FakeModel()
        pooling_params = SimpleNamespace(task="classify")
        new_req = SimpleNamespace(
            req_id="new1",
            prompt_token_ids=[1, 2, 3],
            prompt_embeds=None,
            prompt_is_token_ids=True,
            mm_features=[],
            sampling_params=None,
            pooling_params=pooling_params,
            block_ids=([1],),
            num_computed_tokens=0,
            lora_request=None,
        )
        sched = _update_states_sched(
            num_scheduled_tokens={"new1": 3}, scheduled_new_reqs=[new_req]
        )
        patched_module.ModelRunnerFL._update_states(runner, sched)
        assert applied == [("classify", pooling_params)]
        assert runner.requests["new1"].generator is None

    def test_prompt_logprobs_neg_one_uses_vocab_size(self, patched_module):
        """prompt_logprobs=-1 records the batch vocab size."""
        SamplingParams = import_module("vllm.sampling_params").SamplingParams
        runner = self._runner(patched_module)
        params = SamplingParams()
        params.prompt_logprobs = -1
        new_req = SimpleNamespace(
            req_id="new1",
            prompt_token_ids=[1, 2, 3],
            prompt_embeds=None,
            prompt_is_token_ids=True,
            mm_features=[],
            sampling_params=params,
            pooling_params=None,
            block_ids=([1],),
            num_computed_tokens=0,
            lora_request=None,
        )
        sched = _update_states_sched(
            num_scheduled_tokens={"new1": 3}, scheduled_new_reqs=[new_req]
        )
        patched_module.ModelRunnerFL._update_states(runner, sched)
        assert runner.num_prompt_logprobs == {"new1": 100}

    def test_prompt_logprobs_explicit_value(self, patched_module):
        """An explicit prompt_logprobs value is recorded verbatim."""
        SamplingParams = import_module("vllm.sampling_params").SamplingParams
        runner = self._runner(patched_module)
        params = SamplingParams()
        params.prompt_logprobs = 5
        new_req = SimpleNamespace(
            req_id="new1",
            prompt_token_ids=[1, 2, 3],
            prompt_embeds=None,
            prompt_is_token_ids=True,
            mm_features=[],
            sampling_params=params,
            pooling_params=None,
            block_ids=([1],),
            num_computed_tokens=0,
            lora_request=None,
        )
        sched = _update_states_sched(
            num_scheduled_tokens={"new1": 3}, scheduled_new_reqs=[new_req]
        )
        patched_module.ModelRunnerFL._update_states(runner, sched)
        assert runner.num_prompt_logprobs == {"new1": 5}

    def test_mrope_new_request_initializes_positions(self, patched_module):
        """A mrope model request triggers _init_mrope_positions."""
        mrope_calls = []
        runner = self._runner(patched_module)
        runner.uses_mrope = True
        runner._init_mrope_positions = lambda req_state: mrope_calls.append(
            req_state.req_id
        )
        SamplingParams = import_module("vllm.sampling_params").SamplingParams
        params = SamplingParams()
        new_req = SimpleNamespace(
            req_id="new1",
            prompt_token_ids=[1, 2, 3],
            prompt_embeds=None,
            prompt_is_token_ids=True,
            mm_features=[],
            sampling_params=params,
            pooling_params=None,
            block_ids=([1],),
            num_computed_tokens=0,
            lora_request=None,
        )
        sched = _update_states_sched(
            num_scheduled_tokens={"new1": 3}, scheduled_new_reqs=[new_req]
        )
        patched_module.ModelRunnerFL._update_states(runner, sched)
        assert mrope_calls == ["new1"]

    def test_streaming_request_reuses_update_streaming(self, patched_module):
        """A request already in self.requests routes to the streaming path and
        the streamed state is re-added to the persistent batch."""
        streamed = []
        added = []
        runner = self._runner(patched_module)
        runner.requests["new1"] = SimpleNamespace(req_id="new1")
        runner._update_streaming_request = lambda rid, data: (
            streamed.append(rid) or SimpleNamespace(req_id=rid, tag="streamed")
        )
        runner.input_batch.add_request = lambda r: added.append(r)
        SamplingParams = import_module("vllm.sampling_params").SamplingParams
        params = SamplingParams()
        new_req = SimpleNamespace(
            req_id="new1",
            prompt_token_ids=[1, 2, 3],
            prompt_embeds=None,
            prompt_is_token_ids=True,
            mm_features=[],
            sampling_params=params,
            pooling_params=None,
            block_ids=([1],),
            num_computed_tokens=0,
            lora_request=None,
        )
        sched = _update_states_sched(
            num_scheduled_tokens={"new1": 3}, scheduled_new_reqs=[new_req]
        )
        patched_module.ModelRunnerFL._update_states(runner, sched)
        assert streamed == ["new1"]
        # The streamed state was carried through to the persistent batch.
        assert [getattr(r, "tag", None) for r in added] == ["streamed"]


# ---------------------------------------------------------------------------
# _update_states: ngram-GPU correction closure.
# ---------------------------------------------------------------------------


class TestCorrectSpecDecodeTokenCounts:
    """The deferred spec-decode correction closure returned by _update_states."""

    @staticmethod
    def _ngram_setup(patched_module, monkeypatch, *, install_stubs=True):
        """A runner configured for ngram-GPU async spec decode.

        ``install_stubs=False`` leaves the module's ngram helpers unpatched so
        a test can install its own recording doubles afterwards.
        """
        if install_stubs:
            monkeypatch.setattr(
                patched_module,
                "update_scheduler_for_invalid_drafts",
                lambda event, cpu, so, m: None,
            )
            monkeypatch.setattr(
                patched_module,
                "update_ngram_gpu_tensors_incremental",
                lambda *args, **kwargs: None,
            )

        class _ReqState:
            def __init__(self):
                self.req_id = "r1"
                self.prev_num_draft_len = 2
                self.output_token_ids = []
                self.num_computed_tokens = 5

        runner = SimpleNamespace(
            requests={"r1": _ReqState()},
            num_prompt_logprobs={},
            late_interaction_runner=_NullLIR(),
            input_batch=SimpleNamespace(
                num_reqs=1,
                req_id_to_index={"r1": 0},
                prev_req_id_to_index=None,
                num_computed_tokens_cpu=np.zeros(8, dtype=np.int32),
                num_tokens_no_spec=np.zeros(8, dtype=np.int32),
                num_prompt_tokens=np.full(8, 3, dtype=np.int32),
                token_ids_cpu=np.zeros((8, 64), dtype=np.int64),
                is_token_ids=np.ones((8, 64), dtype=bool),
                num_tokens_no_spec_row=None,
                remove_request=lambda rid: None,
                condense=lambda: None,
                refresh_metadata=lambda: None,
                add_request=lambda r: None,
                update_req_spec_token_ids=lambda r, s: None,
                append_row=lambda ids, idx: None,
            ),
            _on_request_state_removed=lambda *a: None,
            _zero_block_ids=lambda ids: None,
            _process_encoder_cache_scheduler_output=lambda so: None,
            _may_reorder_batch=lambda so: None,
            speculative_config=_make_spec_config(
                "ngram_gpu", use_ngram_gpu=lambda: True
            ),
            is_pooling_model=False,
            uses_mrope=False,
            use_async_scheduling=True,
            use_async_spec_decode=True,
            prev_num_draft_tokens=SimpleNamespace(np=np.zeros(8, dtype=np.int32)),
            token_ids_gpu_tensor="token-ids",
            num_tokens_no_spec_gpu=np.zeros(8, dtype=np.int32),
            _ngram_pinned_idx_buf=None,
            _ngram_pinned_val_buf=None,
            device=torch.device(CPU_DEVICE),
            _num_valid_draft_tokens_event=None,
            _num_valid_draft_tokens_cpu=None,
            prev_num_spec_tokens=2,
        )

        sched = _update_states_sched(
            num_scheduled_tokens={"r1": 3},
            scheduled_spec_decode_tokens={"r1": [10, 11]},
            scheduled_cached_reqs=SimpleNamespace(
                req_ids=["r1"],
                resumed_req_ids=set(),
                new_token_ids=[[9]],
                all_token_ids={"r1": [1, 2, 3]},
                new_block_ids=[None],
                num_computed_tokens=[5],
                num_output_tokens=[3],
            ),
        )
        return runner, sched

    def test_correction_applies_ngram_gpu_decrements(self, patched_module, monkeypatch):
        """A correction with valid counts adjusts the three counters."""
        runner, sched = self._ngram_setup(patched_module, monkeypatch)
        correct = patched_module.ModelRunnerFL._update_states(runner, sched)
        assert correct is not None
        runner.input_batch.prev_req_id_to_index = {"r1": 0}
        runner._get_valid_sampled_token_count = lambda: [4]
        # Independent oracle: num_accepted = 4 - 1 = 3; correction = 2 - 3
        # = -1, so num_computed_tokens goes 5 -> 6 (subtracting -1).
        correct()
        assert runner.requests["r1"].num_computed_tokens == 6
        assert runner.input_batch.num_computed_tokens_cpu[0] == 6
        # The optimistic increment (+2) stays: the ngram decrement only runs
        # when correction > 0.
        assert runner.input_batch.num_tokens_no_spec[0] == 2
        assert runner.num_tokens_no_spec_gpu[0] == 0

    def test_correction_positive_decrements_ngram_counters(
        self, patched_module, monkeypatch
    ):
        """A positive correction also decrements the ngram-GPU counters."""
        runner, sched = self._ngram_setup(patched_module, monkeypatch)
        # Make the optimistic count larger than what was accepted.
        runner.requests["r1"].prev_num_draft_len = 4
        correct = patched_module.ModelRunnerFL._update_states(runner, sched)
        runner.input_batch.prev_req_id_to_index = {"r1": 0}
        runner._get_valid_sampled_token_count = lambda: [2]
        # Independent oracle: num_accepted = 2 - 1 = 1; correction = 4 - 1
        # = 3, so num_computed 5 -> 2 and the ngram counters drop by 3.
        correct()
        assert runner.requests["r1"].num_computed_tokens == 2
        # num_tokens_no_spec was overwritten by the discard alignment at
        # num_prompt_tokens + num_output_tokens = 3 + 3 = 6, then the
        # positive correction (3) is subtracted: 6 - 3 = 3.
        assert runner.input_batch.num_tokens_no_spec[0] == 3
        assert runner.num_tokens_no_spec_gpu[0] == -3

    def test_correction_skipped_when_no_valid_counts(self, patched_module, monkeypatch):
        """An empty valid-count list leaves all counters untouched."""
        runner, sched = self._ngram_setup(patched_module, monkeypatch)
        correct = patched_module.ModelRunnerFL._update_states(runner, sched)
        runner.input_batch.prev_req_id_to_index = {"r1": 0}
        runner._get_valid_sampled_token_count = lambda: []
        correct()
        assert runner.requests["r1"].num_computed_tokens == 5

    def test_correction_skipped_without_prev_index_map(
        self, patched_module, monkeypatch
    ):
        """An empty prev_req_id_to_index aborts the correction early."""
        runner, sched = self._ngram_setup(patched_module, monkeypatch)
        correct = patched_module.ModelRunnerFL._update_states(runner, sched)
        runner.input_batch.prev_req_id_to_index = None
        runner._get_valid_sampled_token_count = lambda: [4]
        correct()
        assert runner.requests["r1"].num_computed_tokens == 5

    def test_correction_missing_prev_index_continues(self, patched_module, monkeypatch):
        """A req_id absent from prev_req_id_to_index is skipped gracefully."""
        runner, sched = self._ngram_setup(patched_module, monkeypatch)
        correct = patched_module.ModelRunnerFL._update_states(runner, sched)
        runner.input_batch.prev_req_id_to_index = {"other": 1}
        runner._get_valid_sampled_token_count = lambda: [4, 4]
        correct()
        assert runner.requests["r1"].num_computed_tokens == 5

    def test_correction_missing_current_index_skips_batch_row(
        self, patched_module, monkeypatch
    ):
        """A req_id absent from the current batch skips the batch-row update
        but still corrects the request state."""
        runner, sched = self._ngram_setup(patched_module, monkeypatch)
        correct = patched_module.ModelRunnerFL._update_states(runner, sched)
        runner.input_batch.prev_req_id_to_index = {"r1": 0}
        runner.input_batch.req_id_to_index = {}
        runner._get_valid_sampled_token_count = lambda: [4]
        correct()
        assert runner.requests["r1"].num_computed_tokens == 6
        # The batch row keeps its pre-correction scheduler value (5): the
        # req_id is no longer in the current index map.
        assert runner.input_batch.num_computed_tokens_cpu[0] == 5

    def test_no_correction_without_deferred_entries(self, patched_module, monkeypatch):
        """No optimistic entries means _update_states returns None."""
        runner, sched = self._ngram_setup(patched_module, monkeypatch)
        runner.requests["r1"].prev_num_draft_len = 0
        correct = patched_module.ModelRunnerFL._update_states(runner, sched)
        assert correct is None

    def test_update_scheduler_for_invalid_drafts_called(
        self, patched_module, monkeypatch
    ):
        """The ngram-GPU path forwards the valid-draft buffers to the
        scheduler-update kernel."""
        sched_calls = []

        def _record(event, cpu, so, m):
            sched_calls.append((event, cpu, so, m))

        monkeypatch.setattr(
            patched_module, "update_scheduler_for_invalid_drafts", _record
        )
        monkeypatch.setattr(
            patched_module,
            "update_ngram_gpu_tensors_incremental",
            lambda *args, **kwargs: None,
        )
        runner, sched = self._ngram_setup(
            patched_module, monkeypatch, install_stubs=False
        )
        runner._num_valid_draft_tokens_event = "evt"
        runner._num_valid_draft_tokens_cpu = "cpu-buf"
        patched_module.ModelRunnerFL._update_states(runner, sched)
        assert len(sched_calls) == 1
        event, cpu, so, mapping = sched_calls[0]
        assert event == "evt"
        assert cpu == "cpu-buf"
        assert so is sched

    def test_incremental_ngram_update_invoked_with_new_reqs(
        self, patched_module, monkeypatch
    ):
        """The incremental tensor update receives the runner's GPU tensors."""
        incr_calls = []

        def _incr(batch, token_ids, num_tokens, new_reqs, device, **kw):
            incr_calls.append((token_ids, num_tokens, list(new_reqs), device))

        monkeypatch.setattr(
            patched_module, "update_ngram_gpu_tensors_incremental", _incr
        )
        monkeypatch.setattr(
            patched_module,
            "update_scheduler_for_invalid_drafts",
            lambda event, cpu, so, m: None,
        )
        runner, sched = self._ngram_setup(
            patched_module, monkeypatch, install_stubs=False
        )
        patched_module.ModelRunnerFL._update_states(runner, sched)
        assert len(incr_calls) == 1
        token_ids, num_tokens, new_reqs, device = incr_calls[0]
        assert token_ids == "token-ids"
        assert num_tokens is runner.num_tokens_no_spec_gpu
        assert new_reqs == []
        assert device.type == "cpu"


# ---------------------------------------------------------------------------
# _update_states_after_model_execute: align branch.
# ---------------------------------------------------------------------------


class TestUpdateStatesAfterModelExecuteAlign:
    """The mamba align postprocess branch of _update_states_after_model_execute."""

    def test_align_branch_invokes_postprocess_and_records_event(
        self, patched_module, monkeypatch
    ):
        """The align cache mode calls postprocess_mamba_align_gpu with the
        runner's buffers and records the num_accepted event."""
        postprocess_calls = []

        def _postprocess(**kwargs):
            postprocess_calls.append(kwargs)
            return None

        fake_mamba_utils = SimpleNamespace(postprocess_mamba_align_gpu=_postprocess)
        monkeypatch.setattr(patched_module, "mamba_utils", fake_mamba_utils)
        CpuGpuBuffer = patched_module.CpuGpuBuffer
        device = torch.device(CPU_DEVICE)
        fake_bufs = SimpleNamespace(fake_mamba=True)
        copy_funcs = {"mamba": "copy"}
        runner = SimpleNamespace(
            speculative_config=SimpleNamespace(method="ngram"),
            model_config=SimpleNamespace(is_hybrid=True),
            num_accepted_tokens=CpuGpuBuffer(8, dtype=torch.int32, device=device),
            num_accepted_tokens_event=_FakeEvent(),
            cache_config=SimpleNamespace(mamba_cache_mode="align"),
            _get_mamba_bufs=lambda: fake_bufs,
            _get_mamba_state_copy_funcs=lambda: copy_funcs,
            input_batch=SimpleNamespace(
                num_accepted_tokens_cpu_tensor=torch.zeros(8, dtype=torch.int32)
            ),
            kv_cache_config=SimpleNamespace(),
            compilation_config=SimpleNamespace(
                static_forward_context=SimpleNamespace()
            ),
        )
        output_token_ids = torch.tensor([[3, 4, -1, -1], [5, 6, 7, -1]])
        patched_module.ModelRunnerFL._update_states_after_model_execute(
            runner, output_token_ids, SimpleNamespace()
        )
        assert len(postprocess_calls) == 1
        kwargs = postprocess_calls[0]
        assert kwargs["bufs"] is fake_bufs
        assert kwargs["num_reqs"] == 2
        assert kwargs["mamba_state_copy_funcs"] is copy_funcs
        assert kwargs["input_batch"] is runner.input_batch
        # Independent oracle: non-(-1) counts per row.
        expected = (output_token_ids != -1).sum(dim=1).to(torch.int32)
        assert torch.equal(runner.num_accepted_tokens.gpu[:2], expected)
        assert runner.num_accepted_tokens_event.record_count == 1


# ---------------------------------------------------------------------------
# Shared builders for real InputBatch / SchedulerOutput graphs.
# ---------------------------------------------------------------------------


def _make_input_batch(patched_module, *, num_groups=1):
    """Build a real InputBatch sized for the tests in this file."""
    InputBatch = import_module("vllm.v1.worker.gpu_input_batch").InputBatch
    return InputBatch(
        max_num_reqs=8,
        max_model_len=64,
        max_num_batched_tokens=64,
        device=torch.device(CPU_DEVICE),
        vocab_size=32,
        block_sizes=[16] * num_groups,
        max_num_blocks_per_req=[4] * num_groups,
    )


def _make_req_state(patched_module, req_id, prompt, block, **overrides):
    """Build a real CachedRequestState with greedy sampling params."""
    SamplingParams = import_module("vllm.sampling_params").SamplingParams
    SamplingType = import_module("vllm.sampling_params").SamplingType
    CachedRequestState = import_module(
        "vllm.v1.worker.gpu_input_batch"
    ).CachedRequestState

    params = SamplingParams()
    params.sampling_type = SamplingType.GREEDY
    state = CachedRequestState(
        req_id=req_id,
        prompt_token_ids=list(prompt),
        mm_features=[],
        sampling_params=params,
        generator=None,
        block_ids=(list(block),),
        num_computed_tokens=0,
        output_token_ids=[],
    )
    for key, value in overrides.items():
        setattr(state, key, value)
    return state


def _scheduler_output(
    patched_module,
    *,
    num_scheduled_tokens=None,
    total_num_scheduled_tokens=0,
    spec_tokens=None,
):
    """Build a real SchedulerOutput dataclass with the given contents."""
    CachedRequestData = import_module("vllm.v1.core.sched.output").CachedRequestData
    SchedulerOutput = import_module("vllm.v1.core.sched.output").SchedulerOutput
    return SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData(
            req_ids=[],
            resumed_req_ids=set(),
            new_token_ids=[],
            all_token_ids={},
            new_block_ids=[],
            num_computed_tokens=[],
            num_output_tokens=[],
        ),
        num_scheduled_tokens=dict(num_scheduled_tokens or {}),
        total_num_scheduled_tokens=total_num_scheduled_tokens,
        scheduled_spec_decode_tokens=dict(spec_tokens or {}),
        scheduled_encoder_inputs={},
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )


def _prepare_inputs_runner(patched_module, batch, requests):
    """A runner carrying every buffer _prepare_inputs touches."""
    device = torch.device(CPU_DEVICE)
    CpuGpuBuffer = patched_module.CpuGpuBuffer

    def buf(*size, dtype=torch.int32):
        return CpuGpuBuffer(*size, dtype=dtype, device=device)

    runner = SimpleNamespace(
        device=device,
        input_batch=batch,
        requests=dict(requests),
        num_prompt_logprobs={},
        enable_prompt_embeds=False,
        use_ascend_attention=False,
        use_async_spec_decode=False,
        use_async_scheduling=False,
        max_model_len=64,
        input_ids=buf(64),
        inputs_embeds=buf((64, 4), dtype=torch.float32),
        is_token_ids=buf(64, dtype=torch.bool),
        query_start_loc=buf(9),
        positions=torch.zeros(64, dtype=torch.int64, device=device),
        query_pos=buf(64),
        seq_lens=torch.zeros(8, dtype=torch.int32, device=device),
        optimistic_seq_lens_cpu=torch.zeros(8, dtype=torch.int32),
        num_computed_tokens=torch.zeros(8, dtype=torch.int32, device=device),
        prev_num_draft_tokens=buf(8),
        req_indices=buf(64, dtype=torch.int64),
        prev_positions=buf(8, dtype=torch.int64),
        num_scheduled_tokens=buf(8),
        num_decode_draft_tokens=buf(8),
        num_accepted_tokens=buf(8),
        encoder_seq_lens=buf(8),
        discard_request_mask=buf(8, dtype=torch.bool),
        _arange_scratch=np.zeros(64, dtype=np.int32),
        valid_sampled_token_count_gpu=None,
        num_accepted_tokens_event=None,
        lora_config=None,
        _kl_guard_on=lambda: False,
        _pp_recv_work=None,
        _draft_token_ids=None,
        prev_num_spec_tokens=0,
        uses_mrope=False,
        arange_np=np.arange(256, dtype=np.int32),
        vllm_config=SimpleNamespace(
            scheduler_config=SimpleNamespace(max_num_batched_tokens=64)
        ),
    )
    for name in (
        "_get_cumsum_and_arange",
        "_compute_prev_positions",
        "_calc_spec_decode_metadata",
        "_prepare_input_ids",
        "_get_encoder_seq_lens",
    ):
        setattr(
            runner,
            name,
            MethodType(getattr(patched_module.ModelRunnerFL, name), runner),
        )
    return runner


class TestPrepareInputsExtras:
    """Branches of _prepare_inputs not covered by the core suite."""

    @staticmethod
    def _batch_and_runner(patched_module):
        batch = _make_input_batch(patched_module)
        state = _make_req_state(patched_module, "r1", (5, 6, 7), (9,))
        batch.add_request(state)
        batch.refresh_metadata()
        runner = _prepare_inputs_runner(patched_module, batch, {"r1": state})
        sched = _scheduler_output(
            patched_module,
            num_scheduled_tokens={"r1": 1},
            total_num_scheduled_tokens=1,
        )
        return runner, sched, np.array([1], dtype=np.int32)

    def test_lora_set_active_loras_invoked(
        self, patched_module, cpu_slot_mapping_kernel
    ):
        """A lora config routes the batch through set_active_loras."""
        calls = []

        def _set_active_loras(batch, num_scheduled, num_sampled):
            calls.append((num_scheduled.tolist(), num_sampled.tolist()))

        runner, sched, num_scheduled = self._batch_and_runner(patched_module)
        runner.lora_config = SimpleNamespace(max_loras=1)
        runner.set_active_loras = _set_active_loras
        logits_indices, spec_md, max_sampled = (
            patched_module.ModelRunnerFL._prepare_inputs(runner, sched, num_scheduled)
        )
        assert calls == [([1], [1])]
        assert logits_indices is not None
        assert spec_md is None
        assert max_sampled == 1

    def test_no_lora_config_skips_set_active_loras(
        self, patched_module, cpu_slot_mapping_kernel
    ):
        """A None lora config skips the hot-swap path entirely."""
        calls = []
        runner, sched, num_scheduled = self._batch_and_runner(patched_module)
        runner.set_active_loras = lambda batch, num_scheduled, num_sampled: (
            calls.append(1)
        )
        patched_module.ModelRunnerFL._prepare_inputs(runner, sched, num_scheduled)
        assert calls == []

    def test_num_accepted_event_sync_non_async(
        self, patched_module, cpu_slot_mapping_kernel
    ):
        """A present num_accepted_tokens_event with sync scheduling pulls the
        CPU counts into the buffer and the GPU mirror."""
        runner, sched, num_scheduled = self._batch_and_runner(patched_module)
        runner.num_accepted_tokens_event = _FakeEvent()
        runner.input_batch.num_accepted_tokens_cpu[:1] = [3]
        patched_module.ModelRunnerFL._prepare_inputs(runner, sched, num_scheduled)
        assert runner.num_accepted_tokens_event.synchronized is True
        assert runner.num_accepted_tokens.np[0] == 3
        assert runner.num_accepted_tokens.np[1] == 1
        assert runner.num_accepted_tokens.gpu[0] == 3

    def test_num_accepted_async_remaps_prev_indices(
        self, patched_module, cpu_slot_mapping_kernel
    ):
        """Under async scheduling the CPU counts are remapped through
        prev_positions (a moved request keeps its accepted count)."""
        runner, sched, num_scheduled = self._batch_and_runner(patched_module)
        runner.num_accepted_tokens_event = _FakeEvent()
        runner.use_async_scheduling = True
        runner.cache_config = SimpleNamespace(mamba_cache_mode="align")
        runner.input_batch.prev_req_id_to_index = {"r1": 0}
        runner.prev_positions.np[:1] = [0]
        runner.input_batch.num_accepted_tokens_cpu[:1] = [3]
        runner.input_batch.num_accepted_tokens_cpu[1] = 9
        patched_module.ModelRunnerFL._prepare_inputs(runner, sched, num_scheduled)
        assert runner.num_accepted_tokens.np[0] == 3
        assert runner.num_accepted_tokens.gpu[0] == 3

    def test_num_accepted_async_new_request_defaults_one(
        self, patched_module, cpu_slot_mapping_kernel
    ):
        """A row with no previous position (new request) defaults to 1."""
        runner, sched, num_scheduled = self._batch_and_runner(patched_module)
        runner.num_accepted_tokens_event = _FakeEvent()
        runner.use_async_scheduling = True
        runner.cache_config = SimpleNamespace(mamba_cache_mode="align")
        # The prev map is non-empty (so the remap branch is taken) but lacks
        # "r1": _compute_prev_positions maps the new request to -1.
        runner.input_batch.prev_req_id_to_index = {"other": 0}
        runner.input_batch.num_accepted_tokens_cpu[:1] = [3]
        patched_module.ModelRunnerFL._prepare_inputs(runner, sched, num_scheduled)
        assert runner.num_accepted_tokens.np[0] == 1

    def test_num_accepted_defaults_without_event(
        self, patched_module, cpu_slot_mapping_kernel
    ):
        """Without an accepted-counts event the buffer defaults to one."""
        runner, sched, num_scheduled = self._batch_and_runner(patched_module)
        runner.input_batch.num_accepted_tokens_cpu[:1] = [3]
        patched_module.ModelRunnerFL._prepare_inputs(runner, sched, num_scheduled)
        assert runner.num_accepted_tokens.np[0] == 1
        assert runner.num_accepted_tokens.gpu[0] == 1


# ---------------------------------------------------------------------------
# _build_attention_metadata: extra branches.
# ---------------------------------------------------------------------------


class _RecordingBuilder:
    """A minimal AttentionMetadataBuilder stand-in recording build calls.

    ``AttentionMetadataBuilder`` is an ABC whose abstract ``__init__`` and
    ``build`` must be overridden; this subclass stores the common metadata it
    was built with so tests can assert what the production loop passed down
    without touching any real GPU backend.  The recorded metadata objects are
    the *real* ``CommonAttentionMetadata`` instances produced by the method
    under test.
    """

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        self.kv_cache_spec = kv_cache_spec
        self.layer_names = layer_names
        self.vllm_config = vllm_config
        self.device = device
        self.builds = []
        self.capture_builds = []
        self.updates = []

    # Mirrors AttentionMetadataBuilder.supports_update_block_table (False).
    supports_update_block_table = False

    def build(self, common_prefix_len, common_attn_metadata, **kwargs):
        self.builds.append((common_prefix_len, common_attn_metadata, kwargs))
        return {"cm": common_attn_metadata, "prefix_len": common_prefix_len}

    def build_for_cudagraph_capture(self, common_attn_metadata):
        self.capture_builds.append(common_attn_metadata)
        return {"cm": common_attn_metadata, "capture": True}

    def update_block_table(self, metadata, blk_table, slot_mapping):
        self.updates.append((metadata, blk_table, slot_mapping))
        return {"cm": metadata["cm"], "updated": True}


class _UpdateableBuilder(_RecordingBuilder):
    """Same recorder, but advertises block-table update support."""

    supports_update_block_table = True


def _full_spec():
    """A real FullAttentionSpec sized for the CPU test batch."""
    FullAttentionSpec = import_module("vllm.v1.kv_cache_interface").FullAttentionSpec
    return FullAttentionSpec(
        block_size=16, num_kv_heads=2, head_size=8, dtype=torch.float32
    )


def _encoder_only_spec():
    """A real EncoderOnlyAttentionSpec."""
    kv_interface = import_module("vllm.v1.kv_cache_interface")
    return kv_interface.EncoderOnlyAttentionSpec(
        block_size=16, num_kv_heads=2, head_size=8, dtype=torch.float32
    )


def _bam_runner(patched_module, batch, groups, kv_cache_specs, **overrides):
    """A runner wired for _build_attention_metadata (see the core suite)."""
    AttentionGroup = import_module("vllm.v1.worker.utils").AttentionGroup
    kv_interface = import_module("vllm.v1.kv_cache_interface")

    device = torch.device(CPU_DEVICE)
    attn_groups = []
    kv_cache_groups = []
    for gid, (group_list, spec) in enumerate(zip(groups, kv_cache_specs)):
        group_objects = []
        layer_names_all = []
        for layer_names, builder in group_list:
            group = AttentionGroup(
                backend=None,
                layer_names=list(layer_names),
                kv_cache_spec=spec,
                kv_cache_group_id=gid,
            )
            group.metadata_builders = [builder]
            group_objects.append(group)
            layer_names_all.extend(layer_names)
        attn_groups.append(group_objects)
        kv_cache_groups.append(
            kv_interface.KVCacheGroupSpec(
                layer_names=layer_names_all, kv_cache_spec=spec
            )
        )

    runner = SimpleNamespace(
        device=device,
        input_batch=batch,
        requests={},
        kv_cache_config=kv_interface.KVCacheConfig(
            num_blocks=8,
            kv_cache_tensors=[],
            kv_cache_groups=kv_cache_groups,
        ),
        attn_groups=attn_groups,
        max_model_len=64,
        optimistic_seq_lens_cpu=torch.zeros(8, dtype=torch.int32),
        query_start_loc=patched_module.CpuGpuBuffer(
            9, dtype=torch.int32, device=device
        ),
        seq_lens=torch.zeros(8, dtype=torch.int32, device=device),
        positions=torch.zeros(64, dtype=torch.int64, device=device),
        use_ascend_attention=False,
        use_async_spec_decode=False,
        dcp_world_size=1,
        dcp_rank=0,
        parallel_config=SimpleNamespace(cp_kv_cache_interleave_size=1),
        is_mm_prefix_lm=False,
        cache_config=SimpleNamespace(use_replayssm=False),
        model_config=SimpleNamespace(rswa_window=None),
        routed_experts_initialized=False,
        speculative_config=None,
        drafter=None,
        num_accepted_tokens=patched_module.CpuGpuBuffer(
            8, dtype=torch.int32, device=device
        ),
        num_decode_draft_tokens=patched_module.CpuGpuBuffer(
            8, dtype=torch.int32, device=device
        ),
        encoder_seq_lens=patched_module.CpuGpuBuffer(
            8, dtype=torch.int32, device=device
        ),
    )
    runner._get_encoder_seq_lens = MethodType(
        patched_module.ModelRunnerFL._get_encoder_seq_lens, runner
    )
    for key, value in overrides.items():
        setattr(runner, key, value)
    batch.block_table.commit_block_table(batch.num_reqs)
    return runner


def _fill_query_start_loc(runner, cu_tokens):
    """Write cu_num_tokens into the runner's query_start_loc buffer."""
    vals = [0, *cu_tokens]
    padded = vals + [cu_tokens[-1]] * (9 - len(vals))
    runner.query_start_loc.cpu[:] = torch.tensor(padded, dtype=torch.int32)
    runner.query_start_loc.copy_to_gpu()


class TestBuildAttentionMetadataExtras:
    """_build_attention_metadata branches beyond the core suite's coverage."""

    def _batch_with(self, patched_module, num_groups=1):
        """A batch of one request; ``num_groups`` block tables per request."""
        batch = _make_input_batch(patched_module, num_groups=num_groups)
        state = _make_req_state(patched_module, "r1", (5, 6, 7), (9,))
        if num_groups > 1:
            state.block_ids = ([9],) + ([9],) * (num_groups - 1)
        batch.add_request(state)
        batch.refresh_metadata()
        return batch, state

    def test_routed_experts_slot_mapping_copied(self, patched_module):
        """The routed-experts branch snapshots the attention slot mapping into
        the private device buffer."""
        batch, _ = self._batch_with(patched_module)
        builder = _RecordingBuilder(_full_spec(), ["layers.0"], None, None)
        runner = _bam_runner(
            patched_module, batch, [[(["layers.0"], builder)]], [_full_spec()]
        )
        _fill_query_start_loc(runner, [3])
        runner.optimistic_seq_lens_cpu[0] = 3
        slot_mappings = {0: torch.arange(64, dtype=torch.int64)}
        runner.routed_experts_initialized = True
        runner.routed_experts_capturer = SimpleNamespace(attn_gid=0)
        runner.routed_experts_slot_mapping_device = torch.full(
            (64,), -1, dtype=torch.int64
        )
        patched_module.ModelRunnerFL._build_attention_metadata(
            runner, 3, 1, 3, slot_mappings=slot_mappings
        )
        assert runner.routed_experts_slot_mapping_device[:3].tolist() == [0, 1, 2]

    def test_ascend_attention_state_attached(self, patched_module):
        """use_ascend_attention attaches the explicit attention state to the
        common metadata."""
        batch, _ = self._batch_with(patched_module)
        builder = _RecordingBuilder(_full_spec(), ["layers.0"], None, None)
        runner = _bam_runner(
            patched_module, batch, [[(["layers.0"], builder)]], [_full_spec()]
        )
        _fill_query_start_loc(runner, [3])
        runner.optimistic_seq_lens_cpu[0] = 3
        slot_mappings = {0: torch.arange(64, dtype=torch.int64)}
        runner.use_ascend_attention = True
        phase = SimpleNamespace(name="decode")
        patched_module.ModelRunnerFL._build_attention_metadata(
            runner,
            3,
            1,
            3,
            slot_mappings=slot_mappings,
            ascend_attn_state=phase,
        )
        _, cm, _ = builder.builds[0]
        assert cm.attn_state is phase

    def test_encoder_only_spec_zeros_block_table(self, patched_module):
        """An EncoderOnlyAttentionSpec group builds an all-zero block table."""
        spec = _encoder_only_spec()
        batch, _ = self._batch_with(patched_module)
        builder = _RecordingBuilder(spec, ["enc.0"], None, None)
        runner = _bam_runner(patched_module, batch, [[(["enc.0"], builder)]], [spec])
        _fill_query_start_loc(runner, [3])
        runner.optimistic_seq_lens_cpu[0] = 3
        slot_mappings = {0: torch.arange(64, dtype=torch.int64)}
        patched_module.ModelRunnerFL._build_attention_metadata(
            runner, 3, 1, 3, slot_mappings=slot_mappings
        )
        _, cm, _ = builder.builds[0]
        assert (cm.block_table_tensor == 0).all()
        assert cm.block_table_tensor.dtype == torch.int32

    def test_replayssm_base_cpu_attached(self, patched_module):
        """use_replayssm populates replayssm_decode_base_cpu on the cm."""
        batch, _ = self._batch_with(patched_module)
        builder = _RecordingBuilder(_full_spec(), ["layers.0"], None, None)
        runner = _bam_runner(
            patched_module, batch, [[(["layers.0"], builder)]], [_full_spec()]
        )
        _fill_query_start_loc(runner, [3])
        runner.optimistic_seq_lens_cpu[0] = 3
        slot_mappings = {0: torch.arange(64, dtype=torch.int64)}
        runner.cache_config = SimpleNamespace(use_replayssm=True)
        patched_module.ModelRunnerFL._build_attention_metadata(
            runner, 3, 1, 3, slot_mappings=slot_mappings
        )
        _, cm, _ = builder.builds[0]
        assert cm.replayssm_decode_base_cpu is not None

    def test_rswa_prefix_lens_attached(self, patched_module):
        """A non-None rswa_window exposes the prompt token lengths."""
        batch, _ = self._batch_with(patched_module)
        builder = _RecordingBuilder(_full_spec(), ["layers.0"], None, None)
        runner = _bam_runner(
            patched_module, batch, [[(["layers.0"], builder)]], [_full_spec()]
        )
        _fill_query_start_loc(runner, [3])
        runner.optimistic_seq_lens_cpu[0] = 3
        slot_mappings = {0: torch.arange(64, dtype=torch.int64)}
        runner.model_config = SimpleNamespace(rswa_window=512)
        patched_module.ModelRunnerFL._build_attention_metadata(
            runner, 3, 1, 3, slot_mappings=slot_mappings
        )
        _, cm, _ = builder.builds[0]
        assert cm.rswa_prefix_lens is not None

    def test_dcp_world_size_gt_1_populates_local_seq_lens(
        self, patched_module, monkeypatch
    ):
        """dcp_world_size > 1 fills dcp_local_seq_lens through the real helper."""
        calls = []

        def _fake_get_dcp_local_seq_lens(seq_lens, world_size, rank, interleave):
            calls.append((seq_lens.tolist(), world_size, rank))
            return seq_lens

        monkeypatch.setattr(
            patched_module, "get_dcp_local_seq_lens", _fake_get_dcp_local_seq_lens
        )
        batch, _ = self._batch_with(patched_module)
        builder = _RecordingBuilder(_full_spec(), ["layers.0"], None, None)
        runner = _bam_runner(
            patched_module, batch, [[(["layers.0"], builder)]], [_full_spec()]
        )
        _fill_query_start_loc(runner, [3])
        runner.optimistic_seq_lens_cpu[0] = 3
        slot_mappings = {0: torch.arange(64, dtype=torch.int64)}
        runner.dcp_world_size = 2
        runner.dcp_rank = 0
        runner.dcp_local_seq_lens = patched_module.CpuGpuBuffer(
            8, dtype=torch.int32, device=torch.device(CPU_DEVICE)
        )
        patched_module.ModelRunnerFL._build_attention_metadata(
            runner, 3, 1, 3, slot_mappings=slot_mappings
        )
        assert calls == [([3], 2, 0)]
        _, cm, _ = builder.builds[0]
        assert cm.dcp_local_seq_lens is not None
        assert cm.dcp_local_seq_lens_cpu is not None

    def test_kv_sharing_fast_prefill_cm_fields(self, patched_module):
        """kv_sharing_fast_prefill with logits indices sets the cm fields."""
        batch, _ = self._batch_with(patched_module)
        builder = _RecordingBuilder(_full_spec(), ["layers.0"], None, None)
        runner = _bam_runner(
            patched_module, batch, [[(["layers.0"], builder)]], [_full_spec()]
        )
        _fill_query_start_loc(runner, [3])
        runner.optimistic_seq_lens_cpu[0] = 3
        slot_mappings = {0: torch.arange(64, dtype=torch.int64)}
        runner.cache_config = SimpleNamespace(
            use_replayssm=False, kv_sharing_fast_prefill=True
        )
        runner._prepare_kv_sharing_fast_prefill = lambda li: li
        logits_indices = torch.tensor([2], dtype=torch.int64)
        patched_module.ModelRunnerFL._build_attention_metadata(
            runner,
            3,
            1,
            3,
            slot_mappings=slot_mappings,
            logits_indices=logits_indices,
            max_num_sampled_tokens=1,
        )
        _, cm, _ = builder.builds[0]
        assert cm.num_logits_indices == 1
        assert cm.max_logits_per_req == 1
        assert cm.logits_indices_padded is logits_indices

    def test_mm_prefix_doc_ranges_attached(self, patched_module):
        """is_mm_prefix_lm extracts image doc ranges from mm features."""
        batch, state = self._batch_with(patched_module)
        builder = _RecordingBuilder(_full_spec(), ["layers.0"], None, None)
        runner = _bam_runner(
            patched_module, batch, [[(["layers.0"], builder)]], [_full_spec()]
        )
        _fill_query_start_loc(runner, [3])
        runner.optimistic_seq_lens_cpu[0] = 3
        slot_mappings = {0: torch.arange(64, dtype=torch.int64)}
        runner.is_mm_prefix_lm = True
        runner.model_config = SimpleNamespace(
            rswa_window=None,
            hf_text_config=SimpleNamespace(sliding_window=None),
        )
        runner.model = SimpleNamespace(mm_prefix_clamp_sliding_window=False)
        runner.requests = {"r1": state}
        state.mm_features = [
            SimpleNamespace(
                modality="image",
                mm_position=SimpleNamespace(extract_embeds_range=lambda: [(2, 5)]),
            ),
            SimpleNamespace(
                modality="audio",
                mm_position=SimpleNamespace(extract_embeds_range=lambda: [(0, 9)]),
            ),
        ]
        patched_module.ModelRunnerFL._build_attention_metadata(
            runner, 3, 1, 3, slot_mappings=slot_mappings
        )
        _, cm, _ = builder.builds[0]
        # Audio is skipped; the image range survives (no sliding window).
        assert cm.mm_req_doc_ranges == {0: [(2, 5)]}

    def test_mm_prefix_doc_ranges_sliding_window_skip(self, patched_module):
        """Doc ranges wider than the sliding window are skipped unless the
        model clamps in-kernel."""
        batch, state = self._batch_with(patched_module)
        builder = _RecordingBuilder(_full_spec(), ["layers.0"], None, None)
        runner = _bam_runner(
            patched_module, batch, [[(["layers.0"], builder)]], [_full_spec()]
        )
        _fill_query_start_loc(runner, [3])
        runner.optimistic_seq_lens_cpu[0] = 3
        slot_mappings = {0: torch.arange(64, dtype=torch.int64)}
        runner.is_mm_prefix_lm = True
        runner.model_config = SimpleNamespace(
            rswa_window=None,
            hf_text_config=SimpleNamespace(sliding_window=2),
        )
        runner.model = SimpleNamespace(mm_prefix_clamp_sliding_window=False)
        runner.requests = {"r1": state}
        state.mm_features = [
            SimpleNamespace(
                modality="image",
                mm_position=SimpleNamespace(extract_embeds_range=lambda: [(0, 5)]),
            )
        ]
        patched_module.ModelRunnerFL._build_attention_metadata(
            runner, 3, 1, 3, slot_mappings=slot_mappings
        )
        _, cm, _ = builder.builds[0]
        # (0..5) spans 6 > 2 without clamping: dropped.
        assert cm.mm_req_doc_ranges == {0: []}

    def test_mm_prefix_doc_ranges_clamp_keeps_wide_range(self, patched_module):
        """With in-kernel clamping the wide range survives the sliding-window
        skip."""
        batch, state = self._batch_with(patched_module)
        builder = _RecordingBuilder(_full_spec(), ["layers.0"], None, None)
        runner = _bam_runner(
            patched_module, batch, [[(["layers.0"], builder)]], [_full_spec()]
        )
        _fill_query_start_loc(runner, [3])
        runner.optimistic_seq_lens_cpu[0] = 3
        slot_mappings = {0: torch.arange(64, dtype=torch.int64)}
        runner.is_mm_prefix_lm = True
        runner.model_config = SimpleNamespace(
            rswa_window=None,
            hf_text_config=SimpleNamespace(sliding_window=2),
        )
        runner.model = SimpleNamespace(mm_prefix_clamp_sliding_window=True)
        runner.requests = {"r1": state}
        state.mm_features = [
            SimpleNamespace(
                modality="image",
                mm_position=SimpleNamespace(extract_embeds_range=lambda: [(0, 5)]),
            )
        ]
        patched_module.ModelRunnerFL._build_attention_metadata(
            runner, 3, 1, 3, slot_mappings=slot_mappings
        )
        _, cm, _ = builder.builds[0]
        assert cm.mm_req_doc_ranges == {0: [(0, 5)]}

    def test_spec_decode_eagle_drafter_selects_matching_gid(
        self, patched_module, monkeypatch
    ):
        """An EagleProposer drafter receives the cm of its kv_cache_gid group."""
        batch, _ = self._batch_with(patched_module, num_groups=2)
        builders = [
            _RecordingBuilder(_full_spec(), ["layers.0"], None, None),
            _RecordingBuilder(_full_spec(), ["layers.1"], None, None),
        ]
        runner = _bam_runner(
            patched_module,
            batch,
            [
                [(["layers.0"], builders[0])],
                [(["layers.1"], builders[1])],
            ],
            [_full_spec(), _full_spec()],
        )
        _fill_query_start_loc(runner, [3])
        runner.optimistic_seq_lens_cpu[0] = 3
        slot_mappings = {
            0: torch.arange(64, dtype=torch.int64),
            1: torch.arange(64, dtype=torch.int64),
        }

        class _FakeEagle:
            kv_cache_gid = 1

        monkeypatch.setattr(patched_module, "EagleProposer", _FakeEagle)
        runner.speculative_config = SimpleNamespace(num_speculative_tokens=2)
        runner.drafter = _FakeEagle()
        _, spec_cm = patched_module.ModelRunnerFL._build_attention_metadata(
            runner, 3, 1, 3, slot_mappings=slot_mappings
        )
        assert spec_cm is not None
        # The drafter's gid=1 group's cm is selected, distinguishable through
        # the group-1 builder's recorded metadata.
        assert spec_cm is builders[1].builds[0][1]

    def test_step3p5_drafter_records_per_group_metadata(
        self, patched_module, monkeypatch
    ):
        """A Step3p5MTPProposer drafter gets set_per_group_attn_metadata calls
        for every group."""
        recorded = []

        class _FakeStep3p5:
            def set_per_group_attn_metadata(self, gid, block_table, slot_mapping):
                recorded.append(gid)

        # The source uses isinstance(self.drafter, Step3p5MTPProposer); patch
        # the module-level name so the fake satisfies the check.
        monkeypatch.setattr(patched_module, "Step3p5MTPProposer", _FakeStep3p5)

        batch, _ = self._batch_with(patched_module, num_groups=2)
        builders = [
            _RecordingBuilder(_full_spec(), ["layers.0"], None, None),
            _RecordingBuilder(_full_spec(), ["layers.1"], None, None),
        ]
        runner = _bam_runner(
            patched_module,
            batch,
            [
                [(["layers.0"], builders[0])],
                [(["layers.1"], builders[1])],
            ],
            [_full_spec(), _full_spec()],
        )
        _fill_query_start_loc(runner, [3])
        runner.optimistic_seq_lens_cpu[0] = 3
        slot_mappings = {
            0: torch.arange(64, dtype=torch.int64),
            1: torch.arange(64, dtype=torch.int64),
        }
        runner.speculative_config = SimpleNamespace(num_speculative_tokens=2)
        runner.drafter = _FakeStep3p5()
        patched_module.ModelRunnerFL._build_attention_metadata(
            runner, 3, 1, 3, slot_mappings=slot_mappings
        )
        assert recorded == [0, 1]

    def test_gemma4_drafter_records_per_group_block_table(
        self, patched_module, monkeypatch
    ):
        """A Gemma4Proposer drafter gets set_per_group_block_table calls."""
        recorded = []

        class _FakeGemma4:
            kv_cache_gid = 0

            def set_per_group_block_table(self, gid, block_table):
                recorded.append(gid)

        # The source uses isinstance(self.drafter, Gemma4Proposer); patch the
        # module-level name so the fake satisfies the check.
        monkeypatch.setattr(patched_module, "Gemma4Proposer", _FakeGemma4)

        batch, _ = self._batch_with(patched_module, num_groups=2)
        builders = [
            _RecordingBuilder(_full_spec(), ["layers.0"], None, None),
            _RecordingBuilder(_full_spec(), ["layers.1"], None, None),
        ]
        runner = _bam_runner(
            patched_module,
            batch,
            [
                [(["layers.0"], builders[0])],
                [(["layers.1"], builders[1])],
            ],
            [_full_spec(), _full_spec()],
        )
        _fill_query_start_loc(runner, [3])
        runner.optimistic_seq_lens_cpu[0] = 3
        slot_mappings = {
            0: torch.arange(64, dtype=torch.int64),
            1: torch.arange(64, dtype=torch.int64),
        }
        runner.speculative_config = SimpleNamespace(num_speculative_tokens=2)
        runner.drafter = _FakeGemma4()
        patched_module.ModelRunnerFL._build_attention_metadata(
            runner, 3, 1, 3, slot_mappings=slot_mappings
        )
        assert recorded == [0, 1]

    def test_unpadded_spec_cm_when_padded(self, patched_module):
        """When padded dims exceed the batch, the spec cm is unpadded."""
        batch, _ = self._batch_with(patched_module)
        builder = _RecordingBuilder(_full_spec(), ["layers.0"], None, None)
        runner = _bam_runner(
            patched_module, batch, [[(["layers.0"], builder)]], [_full_spec()]
        )
        _fill_query_start_loc(runner, [3])
        runner.optimistic_seq_lens_cpu[0] = 3
        slot_mappings = {0: torch.arange(64, dtype=torch.int64)}
        runner.speculative_config = SimpleNamespace(num_speculative_tokens=2)
        runner.drafter = SimpleNamespace()
        _, spec_cm = patched_module.ModelRunnerFL._build_attention_metadata(
            runner,
            3,
            1,
            3,
            num_tokens_padded=8,
            num_reqs_padded=4,
            slot_mappings=slot_mappings,
        )
        assert spec_cm is not None
        # unpadded() mirrors the real CommonAttentionMetadata.unpadded.
        assert spec_cm.num_reqs == 1
        assert spec_cm.num_actual_tokens == 3

    def test_updateable_builder_reuses_cached_metadata(self, patched_module):
        """Two groups sharing (spec, builder type) reuse the first build via
        update_block_table."""
        batch, _ = self._batch_with(patched_module, num_groups=2)
        builder0 = _UpdateableBuilder(_full_spec(), ["layers.0"], None, None)
        builder1 = _UpdateableBuilder(_full_spec(), ["layers.1"], None, None)
        runner = _bam_runner(
            patched_module,
            batch,
            [
                [(["layers.0"], builder0)],
                [(["layers.1"], builder1)],
            ],
            [_full_spec(), _full_spec()],
        )
        _fill_query_start_loc(runner, [3])
        runner.optimistic_seq_lens_cpu[0] = 3
        slot_mappings = {
            0: torch.arange(64, dtype=torch.int64),
            1: torch.arange(64, dtype=torch.int64),
        }
        attn_metadata, _ = patched_module.ModelRunnerFL._build_attention_metadata(
            runner, 3, 1, 3, slot_mappings=slot_mappings
        )
        assert builder0.builds, "first group must build"
        assert builder1.updates, "second group must reuse the cached metadata"
        assert not builder1.builds
        assert attn_metadata["layers.1"]["updated"] is True

    def test_updateable_builder_reuses_across_attn_groups(self, patched_module):
        """Two attention groups in one kv cache group also share the cache."""
        builder0 = _UpdateableBuilder(_full_spec(), ["layers.0"], None, None)
        builder1 = _UpdateableBuilder(_full_spec(), ["layers.1"], None, None)
        batch, _ = self._batch_with(patched_module)
        runner = _bam_runner(
            patched_module,
            batch,
            [[(["layers.0"], builder0), (["layers.1"], builder1)]],
            [_full_spec()],
        )
        _fill_query_start_loc(runner, [3])
        runner.optimistic_seq_lens_cpu[0] = 3
        slot_mappings = {0: torch.arange(64, dtype=torch.int64)}
        patched_module.ModelRunnerFL._build_attention_metadata(
            runner, 3, 1, 3, slot_mappings=slot_mappings
        )
        assert builder0.builds
        assert builder1.updates
        assert not builder1.builds


def _plain_batch_change_fixup(
    num_computed_tokens,
    num_accepted_tokens,
    prev_positions,
    valid_sampled_token_count,
    prev_num_draft_tokens,
    cpu_num_computed_tokens,
):
    """Plain-torch oracle for the async spec-decode batch-change fixup.

    Implements the same documented correction as vLLM's
    ``update_num_computed_tokens_for_batch_change`` without the
    ``@torch.compile`` wrapper (inductor codegen needs a C++ toolchain that
    CPU-only hosts may lack): requests that had drafts take
    ``prev_gpu + valid_count``, everything else takes the CPU value.
    """
    gather_indices = prev_positions.clamp(min=0)
    valid_counts = valid_sampled_token_count[gather_indices]
    prev_computed = num_computed_tokens[gather_indices]
    prev_drafts = prev_num_draft_tokens[gather_indices]
    participating = (prev_positions >= 0) & (prev_drafts > 0)
    corrected = prev_computed + valid_counts.int()
    n = prev_positions.shape[0]
    num_computed_tokens[:n].copy_(
        torch.where(participating, corrected, cpu_num_computed_tokens)
    )
    num_accepted_tokens.copy_(
        torch.where(participating, valid_counts, num_accepted_tokens)
    )


class TestInitAscendBranch:
    """The ascend vendor branch of __init__."""

    def test_ascend_vendor_builds_hybrid_cache_flag_and_graph_runtime(
        self, patched_module, monkeypatch
    ):
        """An ascend vendor_name wires the hybrid-cache flag and the graph
        runtime controller."""
        monkeypatch.setattr(
            patched_module, "current_platform", SimpleNamespace(vendor_name="ascend")
        )
        config = _make_vllm_config()
        # The Qwen3.5 GDN family opts into the dense hybrid cache layout.
        config.model_config.is_hybrid = True
        config.model_config.hf_text_config = SimpleNamespace(model_type="qwen3_5_text")
        runner = _build_runner(patched_module, config)
        assert runner.use_ascend_attention is True
        assert runner.use_ascend_hybrid_cache is True
        # A controller is installed (not None) alongside the state slot.
        assert runner.graph_runtime is not None
        assert runner._ascend_attn_state is None

    def test_ascend_vendor_non_hybrid_skips_cache_flag(
        self, patched_module, monkeypatch
    ):
        """A non-hybrid ascend model leaves the hybrid cache flag off."""
        monkeypatch.setattr(
            patched_module, "current_platform", SimpleNamespace(vendor_name="ascend")
        )
        config = _make_vllm_config()
        config.model_config.is_hybrid = False
        runner = _build_runner(patched_module, config)
        assert runner.use_ascend_attention is True
        assert runner.use_ascend_hybrid_cache is False
        assert runner.graph_runtime is not None


class TestRemainingDeepPaths:
    """Late additions covering the residual missing lines in the targets."""

    def test_bam_async_spec_decode_uses_gpu_tensors(self, patched_module):
        """use_async_spec_decode still builds metadata from the optimistic
        lens upper bound while the CPU mirrors are dropped."""
        batch = _make_input_batch(patched_module)
        state = _make_req_state(patched_module, "r1", (5, 6, 7), (9,))
        batch.add_request(state)
        batch.refresh_metadata()
        builder = _RecordingBuilder(_full_spec(), ["layers.0"], None, None)
        runner = _bam_runner(
            patched_module, batch, [[(["layers.0"], builder)]], [_full_spec()]
        )
        runner.use_async_spec_decode = True
        _fill_query_start_loc(runner, [3])
        runner.optimistic_seq_lens_cpu[0] = 7
        slot_mappings = {0: torch.arange(64, dtype=torch.int64)}
        attn_metadata, _ = patched_module.ModelRunnerFL._build_attention_metadata(
            runner, 3, 1, 3, slot_mappings=slot_mappings
        )
        cm = attn_metadata["layers.0"]["cm"]
        # The async branch ran: metadata is still built from the optimistic
        # upper bound, which the async path keeps (only the non-upper-bound
        # CPU mirrors are nulled out before the build).
        assert cm.seq_lens_cpu_upper_bound.tolist() == [7, 0, 0, 0][: cm.num_reqs]

    def test_bam_spec_decode_gdn_builder_gets_extra_args(self, patched_module):
        """A GDN-family builder under spec decode receives the accepted/draft
        token counts as extra build kwargs."""
        GDNAttentionMetadataBuilder = import_module(
            "vllm.v1.attention.backends.gdn_attn"
        ).GDNAttentionMetadataBuilder

        class _RecordingGDN(GDNAttentionMetadataBuilder):
            """Real-class recorder; the isinstance dispatch must select it."""

            def __init__(self):
                self.builds = []
                self.capture_builds = []
                self.updates = []
                self.supports_update_block_table = False

            def build(self, common_prefix_len, common_attn_metadata, **kwargs):
                self.builds.append((common_prefix_len, common_attn_metadata, kwargs))
                return {"cm": common_attn_metadata, "kwargs": kwargs}

        batch = _make_input_batch(patched_module)
        state = _make_req_state(patched_module, "r1", (5, 6, 7), (9,))
        batch.add_request(state)
        batch.refresh_metadata()
        builder = _RecordingGDN()
        runner = _bam_runner(
            patched_module, batch, [[(["layers.0"], builder)]], [_full_spec()]
        )
        runner.speculative_config = SimpleNamespace(num_speculative_tokens=2)
        runner.drafter = SimpleNamespace()
        _fill_query_start_loc(runner, [3])
        runner.optimistic_seq_lens_cpu[0] = 3
        slot_mappings = {0: torch.arange(64, dtype=torch.int64)}
        attn_metadata, _ = patched_module.ModelRunnerFL._build_attention_metadata(
            runner,
            3,
            1,
            3,
            use_spec_decode=True,
            slot_mappings=slot_mappings,
        )
        # The GDN build received the num_accepted_tokens /
        # num_decode_draft_tokens_cpu extras sliced to the padded batch.
        kwargs = attn_metadata["layers.0"]["kwargs"]
        assert kwargs["num_accepted_tokens"].tolist() == [0]
        assert kwargs["num_decode_draft_tokens_cpu"].tolist() == [0]

    def test_prepare_inputs_async_fixup_corrects_computed_tokens(
        self, patched_module, cpu_slot_mapping_kernel, monkeypatch
    ):
        """Async spec decode with a valid-count tensor routes through the
        batch-change fixup kernel instead of the plain CPU copy."""
        runner, sched, num_scheduled = TestPrepareInputsExtras._batch_and_runner(
            patched_module
        )
        runner.use_async_spec_decode = True
        runner.use_async_scheduling = True
        runner.cache_config = SimpleNamespace(mamba_cache_mode="align")
        runner.input_batch.prev_req_id_to_index = {"r1": 0}
        # prev_positions maps r1 to its previous row 0; it had 2 drafts and
        # the previous GPU value was 9 with a valid count of 3.
        runner.prev_positions.np[:1] = [0]
        runner.prev_positions.copy_to_gpu()
        runner.prev_num_draft_tokens.np[:1] = [2]
        runner.prev_num_draft_tokens.copy_to_gpu()
        runner.num_computed_tokens[0] = 9
        runner.valid_sampled_token_count_gpu = torch.tensor([3], dtype=torch.int32)
        # The real helper is @torch.compile-decorated; inductor codegen needs a
        # C++ toolchain absent on CPU-only hosts, so swap in a plain-torch
        # oracle implementing the same documented correction.
        monkeypatch.setattr(
            patched_module,
            "update_num_computed_tokens_for_batch_change",
            _plain_batch_change_fixup,
        )
        patched_module.ModelRunnerFL._prepare_inputs(runner, sched, num_scheduled)
        # Independent oracle: participating row -> prev_computed + valid count
        # = 9 + 3 = 12 (the CPU value of 0 is ignored for this row).
        assert runner.num_computed_tokens[0].item() == 12
        # num_accepted_tokens mirrors the valid count for participating rows.
        assert runner.num_accepted_tokens.gpu[0].item() == 3

    def test_prepare_inputs_mrope_rows_copied_and_drift_applied(
        self, patched_module, cpu_slot_mapping_kernel, monkeypatch
    ):
        """uses_mrope copies positions row-by-row and the async drift is
        folded into the GPU mirror."""
        runner, sched, num_scheduled = TestPrepareInputsExtras._batch_and_runner(
            patched_module
        )
        runner.uses_mrope = True
        runner.use_async_spec_decode = True
        runner.use_async_scheduling = True
        runner.cache_config = SimpleNamespace(mamba_cache_mode="align")
        runner.input_batch.prev_req_id_to_index = {"r1": 0}
        # mrope buffer: [3, max_num_tokens + 1]; the trailing column keeps
        # the rows non-contiguous exactly as the constructor allocates it.
        CpuGpuBuffer = patched_module.CpuGpuBuffer
        runner.mrope_positions = CpuGpuBuffer(
            3, 65, dtype=torch.int64, device=runner.device
        )
        # The mrope computation itself is exercised by the core suite; here
        # it is stubbed to a no-op so the copy/drift mechanics are isolated.
        runner._calc_mrope_positions = lambda scheduler_output: None
        # Write the pattern row r, col c = 100 * r + c into the CPU mirror.
        for r in range(3):
            for c in range(65):
                runner.mrope_positions.cpu[r, c] = 100 * r + c
        runner.mrope_positions.gpu.fill_(-1)
        # Wire the async fixup so the GPU num_computed_tokens (12) diverges
        # from the batch CPU value (3): prev row 0 had 2 drafts, GPU held 9,
        # the previous step accepted 3.
        runner.prev_positions.np[:1] = [0]
        runner.prev_positions.copy_to_gpu()
        runner.prev_num_draft_tokens.np[:1] = [2]
        runner.prev_num_draft_tokens.copy_to_gpu()
        runner.num_computed_tokens[0] = 9
        runner.valid_sampled_token_count_gpu = torch.tensor([3], dtype=torch.int32)
        runner.input_batch.num_computed_tokens_cpu[0] = 3
        monkeypatch.setattr(
            patched_module,
            "update_num_computed_tokens_for_batch_change",
            _plain_batch_change_fixup,
        )
        patched_module.ModelRunnerFL._prepare_inputs(runner, sched, num_scheduled)
        # Row-wise copies replaced the GPU sentinel with the CPU pattern for
        # the first total_num_scheduled_tokens = 1 column, then the drift
        # (12 - 3 = 9) was added to every row.
        assert runner.mrope_positions.gpu[0, 0].item() == 9
        assert runner.mrope_positions.gpu[1, 0].item() == 109
        assert runner.mrope_positions.gpu[2, 0].item() == 209
        # The trailing dummy column is untouched by the per-row copy.
        assert runner.mrope_positions.gpu[0, 1].item() == -1

    def test_prev_num_draft_tokens_written_for_prev_row(
        self, patched_module, monkeypatch
    ):
        """A known previous row records the optimistic draft count."""
        runner, sched = TestCorrectSpecDecodeTokenCounts._ngram_setup(
            patched_module, monkeypatch
        )
        runner.input_batch.prev_req_id_to_index = {"r1": 2}
        patched_module.ModelRunnerFL._update_states(runner, sched)
        # prev_num_draft_len = 2 lands in the previous row slot 2.
        assert runner.prev_num_draft_tokens.np[2] == 2

    def test_new_request_tracked_for_ngram_incremental(
        self, patched_module, monkeypatch
    ):
        """A brand-new request on the ngram-GPU path is forwarded to the
        incremental tensor update."""
        new_reqs = []

        def _incr(batch, token_ids, num_tokens, new_reqs_arg, device, **kw):
            new_reqs.extend(new_reqs_arg)

        monkeypatch.setattr(
            patched_module, "update_ngram_gpu_tensors_incremental", _incr
        )
        # The scheduler-update kernel needs the valid-draft event stream,
        # which only exists on real GPU runs.
        monkeypatch.setattr(
            patched_module,
            "update_scheduler_for_invalid_drafts",
            lambda event, cpu, so, m: None,
        )
        runner, sched = TestCorrectSpecDecodeTokenCounts._ngram_setup(
            patched_module, monkeypatch, install_stubs=False
        )
        new_req = SimpleNamespace(
            req_id="n1",
            prompt_token_ids=[1, 2],
            prompt_embeds=None,
            prompt_is_token_ids=False,
            mm_features=[],
            sampling_params=None,
            pooling_params=None,
            block_ids=([1],),
            num_computed_tokens=0,
            lora_request=None,
        )
        sched.scheduled_new_reqs = [new_req]
        sched.num_scheduled_tokens["n1"] = 2
        patched_module.ModelRunnerFL._update_states(runner, sched)
        assert [r.req_id for r in new_reqs] == ["n1"]

    def test_resumed_request_tracked_and_draft_len_reset(
        self, patched_module, monkeypatch
    ):
        """A request absent from the batch resets prev_num_draft_len, recovers
        its output ids, and is tracked for the incremental ngram copy."""
        new_reqs = []

        def _incr(batch, token_ids, num_tokens, new_reqs_arg, device, **kw):
            new_reqs.extend(new_reqs_arg)

        monkeypatch.setattr(
            patched_module, "update_ngram_gpu_tensors_incremental", _incr
        )
        # The scheduler-update kernel needs the valid-draft event stream,
        # which only exists on real GPU runs.
        monkeypatch.setattr(
            patched_module,
            "update_scheduler_for_invalid_drafts",
            lambda event, cpu, so, m: None,
        )
        runner, sched = TestCorrectSpecDecodeTokenCounts._ngram_setup(
            patched_module, monkeypatch, install_stubs=False
        )
        # The request is no longer in the batch: the unscheduled removal and
        # the resumed path both apply.
        runner.input_batch.req_id_to_index = {}
        patched_module.ModelRunnerFL._update_states(runner, sched)
        state = runner.requests["r1"]
        # prev_num_draft_len was reset by the req_index-is-None branch.
        assert state.prev_num_draft_len == 0
        # The async recovery restored the resumed output ids (3 tokens).
        assert state.output_token_ids == [1, 2, 3]
        # The resumed request was forwarded to the incremental update.
        assert [r.req_id for r in new_reqs] == ["r1"]

    def test_prompt_embeds_skip_branches(self, patched_module, cpu_slot_mapping_kernel):
        """The three req_prompt_embeds skip paths leave inputs_embeds
        untouched for their rows."""
        batch = _make_input_batch(patched_module)
        # r0: embeds fully consumed (start_pos beyond the available rows).
        state0 = _make_req_state(
            patched_module, "r0", (5, 6, 7), (9,), num_computed_tokens=3
        )
        state0.prompt_token_ids = None
        state0.prompt_embeds = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        # r1: plain token request, not in the req_prompt_embeds map at all.
        state1 = _make_req_state(patched_module, "r1", (5, 6), (9,))
        # r2: embeds request with zero scheduled tokens.
        state2 = _make_req_state(
            patched_module, "r2", (1, 2, 3), (9,), num_computed_tokens=0
        )
        state2.prompt_token_ids = None
        state2.prompt_embeds = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        for state in (state0, state1, state2):
            batch.add_request(state)
        batch.refresh_metadata()
        runner = _prepare_inputs_runner(
            patched_module,
            batch,
            {"r0": state0, "r1": state1, "r2": state2},
        )
        runner.enable_prompt_embeds = True
        sched = _scheduler_output(
            patched_module,
            num_scheduled_tokens={"r0": 1, "r1": 1, "r2": 0},
            total_num_scheduled_tokens=2,
        )
        runner.inputs_embeds.cpu.fill_(7.0)
        patched_module.ModelRunnerFL._prepare_inputs(
            runner, sched, np.array([1, 1, 0], dtype=np.int32)
        )
        # Neither skip branch wrote embeds: every scheduled row keeps the
        # sentinel value (the zero-scheduled row produced no rows at all).
        assert runner.inputs_embeds.gpu[:2].tolist() == [[7.0] * 4] * 2
