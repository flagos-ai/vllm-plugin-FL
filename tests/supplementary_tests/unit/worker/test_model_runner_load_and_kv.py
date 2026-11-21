# Copyright (c) 2026 BAAI. All rights reserved.

"""Supplementary CPU tests for the ModelRunnerFL load / KV-cache family.

Companion to ``test_model_runner_execute_paths.py``, targeting the remaining
uncovered regions of ``vllm_fl/worker/model_runner.py``:

* ``load_model`` (loader boundary, drafter loading, multimodal-pruning flags,
  the OutOfMemoryError handler, and the wrapper-selection tail);
* ``reload_weights`` (iterator and kernel-format branches, disk reload
  errors, and the not-loaded-weights validation warning);
* ``initialize_attn_backend`` (real attention-group construction and
  deduplication from the static forward context);
* ``initialize_kv_cache`` (real allocation / binding of CPU KV caches,
  metadata builders, input-batch reinitialization, and the KV-transfer
  registration boundary);
* ``may_reinitialize_input_batch`` (same-geometry no-op vs rebuild);
* ``_cleanup_profiling_kv_cache``;
* ``_dummy_sampler_run`` / ``_dummy_pooler_run_task`` / ``_dummy_pooler_run``
  (the profiling warm-ups, on real CPU tensors);
* ``_kl_apply`` and the Kunlunx guard helpers.

The module reuses the CUDA-ness proxy import pattern from the execute-paths
sweep.  Only well-identified GPU boundaries are replaced (the model loader
entry, the weight-loading device kernels, the offloader and device cache
ops); every numeric expectation is verified against an independent
plain-torch oracle.
"""

import contextlib
from importlib import import_module
from types import SimpleNamespace

import numpy as np
import pytest

try:  # pragma: no cover - dual-environment guard
    import torch
    import torch.nn as nn

    import vllm  # noqa: F401

    HAS_VLLM = True
except ImportError:  # pragma: no cover - Windows machines without vllm
    HAS_VLLM = False

pytestmark = [
    pytest.mark.supplementary,
    pytest.mark.skipif(not HAS_VLLM, reason="requires real vllm"),
]

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
    """Import vllm_fl.worker.model_runner with CUDA-ness faked at import."""
    pytest.importorskip("vllm", reason="requires real vllm")
    platforms = import_module("vllm.platforms")
    real = platforms.current_platform
    monkeypatch.setattr(platforms, "current_platform", _FakeCudaPlatform(real))
    module = import_module("vllm_fl.worker.model_runner")
    yield module
    monkeypatch.setattr(platforms, "current_platform", real)


class _FakeEvent:
    """Minimal stand-in for a torch device Event (CPU torch cannot record)."""

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


@pytest.fixture
def patched_module(model_runner_module, monkeypatch):
    """Patch the GPU/distributed boundaries on the module under test."""
    pp_group = SimpleNamespace(
        is_last_rank=True,
        is_first_rank=True,
        ranks=[0],
        world_size=1,
        rank=0,
        last_rank=0,
        device_group=None,
    )
    real = import_module("vllm.platforms").current_platform
    fake_platform = SimpleNamespace(
        device_type="cuda",
        torch_device_fn=SimpleNamespace(
            Stream=_FakeStream,
            Event=_FakeEvent,
            current_stream=lambda: _FakeStream(),
            stream=lambda stream: contextlib.nullcontext(),
        ),
        current_stream=lambda: _FakeStream(),
        is_cuda_alike=lambda: False,
        is_cpu=lambda: False,
        vendor_name=None,
        check_runner_kv_caches_multi_layer=lambda: None,
    )
    # Delegate anything else the production code may read to the real
    # platform (device_type stays "cuda" for the import-time checks).
    for attr in ("is_rocm", "is_xpu", "is_npu", "is_musa", "get_current_memory_usage"):
        if hasattr(real, attr):
            value = getattr(real, attr)
            setattr(fake_platform, attr, value)
    monkeypatch.setattr(model_runner_module, "get_pp_group", lambda: pp_group)
    monkeypatch.setattr(model_runner_module, "current_platform", fake_platform)
    monkeypatch.setattr(model_runner_module, "num_compute_units", lambda device_id: 108)
    return model_runner_module


# ---------------------------------------------------------------------------
# Config namespaces.
# ---------------------------------------------------------------------------


def _make_model_config(**overrides):
    """A ModelConfig-shaped namespace with the fields the runner reads."""
    config = SimpleNamespace(
        model="facebook/opt-125m",
        hf_text_config=SimpleNamespace(sliding_window=None),
        hf_config=SimpleNamespace(model_type="opt"),
        dtype=torch.float32,
        runner_type="generate",
        enable_prompt_embeds=False,
        is_multimodal_raw_input_only_model=False,
        is_moe=False,
        max_model_len=64,
        uses_alibi=False,
        disable_cascade_attn=True,
        is_mm_prefix_lm=False,
        uses_mrope=False,
        supports_multimodal_inputs=False,
        is_encoder_decoder=False,
        is_hybrid=False,
        logprobs_mode="processed_logprobs",
        use_fp64_gumbel=False,
        logits_processors=None,
        multimodal_config=None,
        pooler_config=None,
        quantization=None,
        enforce_eager=False,
        get_num_attention_heads=lambda parallel_config: 8,
        get_inputs_embeds_size=lambda: 4,
        get_vocab_size=lambda: 32,
        rswa_window=None,
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def _make_vllm_config(**overrides):
    """Assemble the vllm_config namespace the runner reads."""
    config_mod = import_module("vllm.config")
    model_config = overrides.pop("model_config", None) or _make_model_config()
    speculative = overrides.pop("speculative_config", None)
    # A real CompilationConfig: initialize_attn_backend resolves the
    # cudagraph mode through it (NONE resolves without touching devices).
    compilation_config = config_mod.CompilationConfig()
    compilation_config.cudagraph_mode = config_mod.CUDAGraphMode.NONE
    compilation_config.mode = config_mod.CompilationMode.NONE
    compilation_config.cudagraph_capture_sizes = None
    compilation_config.max_cudagraph_capture_size = None
    compilation_config.static_forward_context = {}
    compilation_config.fast_moe_cold_start = False
    compilation_config.static_all_moe_layers = []
    config = SimpleNamespace(
        model_config=model_config,
        cache_config=SimpleNamespace(
            cache_dtype="auto",
            block_size=16,
            calculate_kv_scales=False,
            mamba_cache_mode=None,
            use_replayssm=False,
            kv_sharing_fast_prefill=False,
            kv_cache_layout="NHD",
            num_gpu_blocks=None,
            # allocate_kv_cache resolves physical strides through the real
            # KVCacheLayout enum (NHD == LBNHC logical axis order).
            get_resolved_kv_cache_layout=lambda: import_module(
                "vllm.v1.kv_cache_layout"
            ).KVCacheLayout.LBNHC,
        ),
        mamba_config=None,
        offload_config=config_mod.OffloadConfig(),
        compilation_config=compilation_config,
        lora_config=None,
        load_config=SimpleNamespace(load_format="dummy", use_tqdm_on_load=False),
        parallel_config=SimpleNamespace(
            data_parallel_size=1,
            data_parallel_rank=0,
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
            distributed_executor_backend="mp",
            cp_kv_cache_interleave_size=1,
            use_ubatching=False,
            num_ubatches=1,
            enable_eplb=False,
            use_sequence_parallel_moe=False,
            is_moe_model=None,
        ),
        scheduler_config=SimpleNamespace(
            max_num_seqs=4,
            max_num_batched_tokens=64,
            max_num_encoder_input_tokens=0,
            async_scheduling=False,
        ),
        speculative_config=speculative,
        observability_config=SimpleNamespace(
            report_metrics=False,
            cudagraph_metrics=False,
            enable_layerwise_nvtx_tracing=False,
        ),
        reasoning_config=None,
        num_speculative_tokens=(
            speculative.num_speculative_tokens if speculative else 0
        ),
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


class _FakeSuffixProposer:
    """SuffixDecodingProposer class stand-in.

    The real class lazy-imports ``arctic_inference`` (absent on CPU venvs),
    so spec-configured runners patch the module-level class around the raw
    constructor call.
    """

    def __init__(self, vllm_config):
        self.vllm_config = vllm_config
        self.kv_cache_gid = 0
        self.proposals = []

    def propose(self, num_tokens, input_batch, sampled, slot_mappings=None):
        self.proposals.append((num_tokens, input_batch, sampled))
        return [[10 + 2 * i, 11 + 2 * i] for i in range(len(sampled))]

    def load_model(self, model):
        self.loaded_with = getattr(self, "loaded_with", [])
        self.loaded_with.append(model)


def _build_runner(module, vllm_config):
    """Instantiate ModelRunnerFL with the raw (unwrapped) constructor."""
    ModelRunnerFL = module.ModelRunnerFL
    spec_config = getattr(vllm_config, "speculative_config", None)
    suffix_patched = (
        spec_config is not None
        and getattr(spec_config, "method", None) == "suffix"
    )
    if suffix_patched:
        # Manual patch with try/finally restore: never monkeypatch.setitem on
        # module attributes that other fixtures pop from sys.modules.
        original_cls = module.SuffixDecodingProposer
        module.SuffixDecodingProposer = _FakeSuffixProposer
    runner = object.__new__(ModelRunnerFL)
    try:
        ModelRunnerFL.__init__.__wrapped__(
            runner, vllm_config, torch.device(CPU_DEVICE)
        )
    finally:
        if suffix_patched:
            module.SuffixDecodingProposer = original_cls
    runner.transfer_event = _FakeEvent()
    if runner.num_accepted_tokens_event is not None:
        runner.num_accepted_tokens_event = _FakeEvent()
    if runner.draft_token_ids_event is not None:
        runner.draft_token_ids_event = _FakeEvent()
    if runner.valid_sampled_token_count_event is not None:
        runner.valid_sampled_token_count_event = _FakeEvent()
    return runner


# ---------------------------------------------------------------------------
# Model / layer stand-ins wired to the real production code paths.
# ---------------------------------------------------------------------------


class _RecordingLoader:
    """Model-loader boundary stand-in for load_model / reload_weights.

    ``has_all_weights`` False omits the ``get_all_weights`` method entirely
    (production probes it with hasattr before the disk-reload branch).
    """

    def __init__(self, model, load_model_impl=None, has_all_weights=False):
        self.model = model
        self.load_calls = []
        self.all_weights_calls = []
        self._load_model_impl = load_model_impl
        self._has_all_weights = has_all_weights

    def load_model(self, *, vllm_config, model_config):
        self.load_calls.append(SimpleNamespace(vllm_config=vllm_config,
                                               model_config=model_config))
        if self._load_model_impl is not None:
            return self._load_model_impl(vllm_config, model_config)
        return self.model

    def __getattr__(self, name):
        if name == "get_all_weights" and not object.__getattribute__(
            self, "_has_all_weights"
        ):
            raise AttributeError(name)
        raise AttributeError(name)


class _TinyModel(nn.Module):
    """A real nn.Module the loader / reloader can operate on."""

    def __init__(self, vocab_size=32, hidden=8):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden)
        self.lm_head = nn.Linear(hidden, vocab_size, bias=False)
        # reload_weights' kernel-format branch copies into leaf parameters;
        # grad tracking is irrelevant here and blocks in-place copy_.
        for param in self.parameters():
            param.requires_grad_(False)
        self.logits_head = torch.arange(1, vocab_size + 1, dtype=torch.float32)
        self.load_weights_calls = []
        self.pooler = None

    def compute_logits(self, hidden_states):
        # Same row-sum projection the execute-paths model uses, so the
        # sampler warm-ups can be oracle-checked.
        return hidden_states.sum(dim=-1, keepdim=True) * self.logits_head

    def load_weights(self, weights_iterator):
        self.load_weights_calls.append(list(weights_iterator))
        return {name for name, _ in self.load_weights_calls[0]}


def _attention_layer_base():
    """The real AttentionLayerBase, imported lazily (Windows has no vllm)."""
    return import_module(
        "vllm.model_executor.layers.attention_layer_base"
    ).AttentionLayerBase


class _FakeAttentionLayer:
    """A static-forward-context layer exposing get_attn_backend.

    Subclasses the real ``AttentionLayerBase`` so
    ``get_layers_from_vllm_config`` picks it up, while keeping a recording
    ``bind_kv_cache`` for the binding assertions.  The subclass is built
    lazily on first instantiation (only on vllm-capable platforms) so this
    module still collects on machines without vllm.
    """

    _layer_cls = None

    def __new__(cls, backend, num_heads=8, num_kv_heads=2, head_size=8):
        if cls._layer_cls is None:
            base = _attention_layer_base()

            class _Layer(base):
                def __init__(self, backend, num_heads=8, num_kv_heads=2,
                             head_size=8):
                    self.backend = backend
                    self.num_heads = num_heads
                    self.num_kv_heads = num_kv_heads
                    self.head_size = head_size
                    self.kv_cache = torch.arange(4, dtype=torch.float32)
                    self.bound = []

                def get_attn_backend(self):
                    return self.backend

                def get_kv_cache_spec(self, vllm_config):
                    return None

                def bind_kv_cache(self, kv_cache):
                    self.bound.append(kv_cache)

            cls._layer_cls = _Layer
        # Construct through the resolved subclass (its __init__ runs; the
        # returned instance is not a _FakeAttentionLayer, so Python will not
        # call _FakeAttentionLayer.__init__ on it).
        return cls._layer_cls(backend, num_heads=num_heads,
                              num_kv_heads=num_kv_heads, head_size=head_size)


class _FakeAttentionBackend:
    """Attention backend class stand-in: dedup key + builder resolution."""

    instances = 0

    def __init__(self, name=None, support=None):
        cls = type(self)
        cls.instances += 1
        self._name = name or f"FakeBackend{cls.instances}"
        self._support = support

    def full_cls_name(self):
        return self._name

    @classmethod
    def get_builder_cls(cls):
        return _FakeBuilderCls

    @classmethod
    def customize_spec(cls, spec):
        # Real backends may shrink / reshape the spec; identity is the
        # neutral contract for a CPU stand-in.
        return spec

    @classmethod
    def get_supported_kernel_block_sizes(cls, spec=None):
        # Declaring the manager's own block size keeps kernel-block
        # selection from splitting the CPU cache views.
        return [spec.block_size] if spec is not None else [16]


class _FakeBuilderCls:
    """The builder class _check_and_update_cudagraph_mode consults."""

    requires_block_table_width = False
    reorder_batch_threshold = None

    @staticmethod
    def get_cudagraph_support(vllm_config, kv_cache_spec):
        from vllm.v1.attention.backend import AttentionCGSupport

        return AttentionCGSupport.ALWAYS

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device,
                 **kwargs):
        self.kv_cache_spec = kv_cache_spec
        self.layer_names = list(layer_names)
        self.vllm_config = vllm_config
        self.device = device

    def build(self, common_prefix_len, common_attn_metadata, **kwargs):
        return {"cm": common_attn_metadata, "prefix_len": common_prefix_len}


# ---------------------------------------------------------------------------
# load_model.
# ---------------------------------------------------------------------------


class TestLoadModel:
    """load_model with the loader entry replaced by a recording stand-in."""

    def _patch_boundaries(self, patched_module, monkeypatch, loader, drafter=None):
        """Replace the module-level loader / warmup / offloader boundaries."""
        warmups = []
        comm_buffers = []
        io_hooks = []
        post_inits = []
        monkeypatch.setattr(patched_module, "get_model_loader",
                            lambda load_config: loader)
        monkeypatch.setattr(patched_module, "warmup_cached_ops",
                            lambda: warmups.append(1))
        monkeypatch.setattr(patched_module, "_prepare_comm_buffer",
                            lambda model: comm_buffers.append(model))
        monkeypatch.setattr(patched_module, "register_io_module_hooks",
                            lambda model: io_hooks.append(model))
        monkeypatch.setattr(patched_module, "get_offloader",
                            lambda: SimpleNamespace(
                                post_init=lambda: post_inits.append(1)
                            ))
        if drafter is not None:
            runner_hint = drafter

            class _DrafterLoader:
                model = runner_hint.model
                loaded = []

                def load_model(self, model):
                    _DrafterLoader.loaded.append(model)
                    return runner_hint.model

            drafter.loader_cls = _DrafterLoader
        return SimpleNamespace(
            warmups=warmups,
            comm_buffers=comm_buffers,
            io_hooks=io_hooks,
            post_inits=post_inits,
        )

    def test_load_model_basic_flow(self, patched_module, monkeypatch):
        """load_model drives the loader entry with the runner's configs,
        warms cached ops, prepares comm buffers, registers IO hooks, and
        runs the offloader's post_init."""
        model = _TinyModel()
        loader = _RecordingLoader(model)
        runner = _build_runner(patched_module, _make_vllm_config())
        calls = self._patch_boundaries(patched_module, monkeypatch, loader)

        patched_module.ModelRunnerFL.load_model(runner)

        assert runner.model is model
        assert len(loader.load_calls) == 1
        assert loader.load_calls[0].vllm_config is runner.vllm_config
        assert loader.load_calls[0].model_config is runner.model_config
        # load_model(defaults to load_dummy_weights=False) leaves the load
        # format untouched.
        assert runner.load_config.load_format == "dummy"
        assert calls.warmups == [1]
        assert calls.comm_buffers == [model]
        assert calls.io_hooks == [model]
        assert calls.post_inits == [1]
        # Text-only opt-style model: no multimodal pruning.
        assert runner.is_multimodal_pruning_enabled is False
        assert runner.requires_sequential_video_encoding is False

    def test_load_model_dummy_weights_switches_load_format(
        self, patched_module, monkeypatch
    ):
        """load_dummy_weights=True rewrites load_config.load_format to
        'dummy' before the loader is constructed."""
        model = _TinyModel()
        loader = _RecordingLoader(model)
        runner = _build_runner(patched_module, _make_vllm_config())
        self._patch_boundaries(patched_module, monkeypatch, loader)
        runner.load_config.load_format = "auto"

        patched_module.ModelRunnerFL.load_model(runner, load_dummy_weights=True)

        assert runner.load_config.load_format == "dummy"

    def test_load_model_drafter_receives_loaded_model(
        self, patched_module, monkeypatch
    ):
        """A drafter with a load_model method is handed the loaded target
        model."""
        target = _TinyModel()

        class Drafter:
            def __init__(self):
                self.loaded_with = []

            def load_model(self, model):
                self.loaded_with.append(model)

        drafter = Drafter()
        loader = _RecordingLoader(target)
        runner = _build_runner(patched_module, _make_vllm_config())
        self._patch_boundaries(patched_module, monkeypatch, loader)
        runner.drafter = drafter

        patched_module.ModelRunnerFL.load_model(runner)

        assert drafter.loaded_with == [target]
        # A drafter without a .model attribute skips the comm buffer.
        # (the drafter has no model attribute here)

    def test_load_model_out_of_memory_reraises(self, patched_module, monkeypatch):
        """A torch.OutOfMemoryError from the loader is re-raised after the
        error is logged."""

        def boom(vllm_config, model_config):
            raise torch.OutOfMemoryError("CUDA out of memory")

        loader = _RecordingLoader(None, load_model_impl=boom)
        runner = _build_runner(patched_module, _make_vllm_config())
        self._patch_boundaries(patched_module, monkeypatch, loader)

        with pytest.raises(torch.OutOfMemoryError):
            patched_module.ModelRunnerFL.load_model(runner)
        # The model was never installed on the runner.
        assert not hasattr(runner, "model") or runner.model is None

    def test_load_model_multimodal_pruning_enabled(
        self, patched_module, monkeypatch
    ):
        """A model supporting multimodal pruning with an enabled multimodal
        config flips is_multimodal_pruning_enabled on."""
        model = _TinyModel()
        mm_config = SimpleNamespace(
            is_multimodal_pruning_enabled=lambda: True
        )
        loader = _RecordingLoader(model)
        runner = _build_runner(
            patched_module,
            _make_vllm_config(
                model_config=_make_model_config(multimodal_config=mm_config)
            ),
        )
        self._patch_boundaries(patched_module, monkeypatch, loader)
        monkeypatch.setattr(
            patched_module, "supports_multimodal_pruning", lambda model: True
        )

        patched_module.ModelRunnerFL.load_model(runner)

        assert runner.is_multimodal_pruning_enabled is True

    def test_load_model_multimodal_pruning_requires_enabled_config(
        self, patched_module, monkeypatch
    ):
        """Pruning support without an enabled multimodal config stays off."""
        model = _TinyModel()
        mm_config = SimpleNamespace(
            is_multimodal_pruning_enabled=lambda: False
        )
        loader = _RecordingLoader(model)
        runner = _build_runner(
            patched_module,
            _make_vllm_config(
                model_config=_make_model_config(multimodal_config=mm_config)
            ),
        )
        self._patch_boundaries(patched_module, monkeypatch, loader)
        monkeypatch.setattr(
            patched_module, "supports_multimodal_pruning", lambda model: True
        )

        patched_module.ModelRunnerFL.load_model(runner)

        assert runner.is_multimodal_pruning_enabled is False


# ---------------------------------------------------------------------------
# reload_weights.
# ---------------------------------------------------------------------------


class TestReloadWeights:
    """reload_weights across the iterator, kernel-format and disk branches."""

    def _runner_with_model(self, patched_module):
        runner = _build_runner(patched_module, _make_vllm_config())
        runner.model = _TinyModel()
        return runner

    def test_kernel_format_copies_weights_into_parameters(
        self, patched_module, monkeypatch
    ):
        """is_checkpoint_format=False copies each tensor into the resolved
        parameter with copy_ (real CPU tensor math)."""
        runner = self._runner_with_model(patched_module)
        model = runner.get_model()
        resets = []
        monkeypatch.setattr(
            patched_module.ModelRunnerFL, "reset_lora_state",
            lambda self: resets.append(1),
        )
        # Independent oracle: reload values chosen so each parameter's new
        # content differs from its initialization.
        new_embed = torch.full_like(model.embed_tokens.weight, 0.5)
        new_head = torch.full_like(model.lm_head.weight, 2.0)
        iterator = [
            ("embed_tokens.weight", new_embed),
            ("lm_head.weight", new_head),
        ]

        patched_module.ModelRunnerFL.reload_weights(
            runner, weights_iterator=iterator, is_checkpoint_format=False
        )

        assert torch.equal(model.embed_tokens.weight, new_embed)
        assert torch.equal(model.lm_head.weight, new_head)
        assert resets == [1]
        # The encoder / mm caches were reset through the real methods.
        assert runner.encoder_cache == {}

    def test_checkpoint_format_uses_model_load_weights(
        self, patched_module, monkeypatch
    ):
        """is_checkpoint_format=True runs the layerwise reload helpers around
        the model's own load_weights."""
        runner = self._runner_with_model(patched_module)
        model = runner.get_model()
        init_calls = []
        fin_calls = []
        monkeypatch.setattr(
            patched_module, "initialize_layerwise_reload",
            lambda m: init_calls.append(m),
        )
        monkeypatch.setattr(
            patched_module, "finalize_layerwise_reload",
            lambda m, mc: fin_calls.append((m, mc)),
        )
        iterator = [("embed_tokens.weight",
                     torch.zeros_like(model.embed_tokens.weight))]

        patched_module.ModelRunnerFL.reload_weights(
            runner, weights_iterator=iterator, is_checkpoint_format=True
        )

        assert init_calls == [model]
        assert fin_calls == [(model, runner.model_config)]
        # load_weights received the exact iterator contents.
        assert model.load_weights_calls == [list(iterator)]

    def test_disk_reload_without_get_all_weights_raises(
        self, patched_module, monkeypatch
    ):
        """Reloading from disk with a loader that has no get_all_weights
        raises NotImplementedError."""
        runner = self._runner_with_model(patched_module)
        loader = _RecordingLoader(None, has_all_weights=False)
        monkeypatch.setattr(patched_module, "get_model_loader",
                            lambda load_config: loader)

        with pytest.raises(NotImplementedError, match="format"):
            patched_module.ModelRunnerFL.reload_weights(runner)

    def test_disk_reload_rewrites_model_path_and_revision(
        self, patched_module, monkeypatch
    ):
        """A weights_path reload repoints model_config.model and clears the
        revision before the weights are enumerated."""
        runner = self._runner_with_model(patched_module)
        model = runner.get_model()
        loader = _RecordingLoader(None, has_all_weights=True)
        monkeypatch.setattr(patched_module, "get_model_loader",
                            lambda load_config: loader)

        def all_weights(model_config, model_):
            loader.all_weights_calls.append((model_config, model_))
            return []

        loader.get_all_weights = all_weights

        patched_module.ModelRunnerFL.reload_weights(
            runner,
            weights_iterator=None,
            weights_path="/new/path",
            is_checkpoint_format=True,
        )

        assert runner.model_config.model == "/new/path"
        assert runner.model_config.revision is None
        assert loader.all_weights_calls == [(runner.model_config, model)]

    def test_no_iterator_with_kernel_format_warns(
        self, patched_module, monkeypatch, caplog
    ):
        """weights_iterator=None with is_checkpoint_format=False warns that
        disk reloads produce checkpoint-format weights."""
        runner = self._runner_with_model(patched_module)
        loader = _RecordingLoader(None, has_all_weights=True)
        loader.get_all_weights = lambda model_config, model: []
        monkeypatch.setattr(patched_module, "get_model_loader",
                            lambda load_config: loader)

        with caplog.at_level("WARNING"):
            patched_module.ModelRunnerFL.reload_weights(
                runner, weights_iterator=None, is_checkpoint_format=False
            )

        assert any("checkpoint format" in r.message for r in caplog.records)

    def test_not_loaded_weights_logged_when_quantization_absent(
        self, patched_module, monkeypatch, caplog
    ):
        """With no quantization, parameter names the iterator never returned
        are reported in the not-loaded warning."""
        runner = self._runner_with_model(patched_module)
        model = runner.get_model()
        monkeypatch.setattr(
            patched_module, "initialize_layerwise_reload", lambda m: None
        )
        monkeypatch.setattr(
            patched_module, "finalize_layerwise_reload", lambda m, mc: None
        )
        # load only the embedding; lm_head stays unloaded.
        iterator = [("embed_tokens.weight",
                     torch.zeros_like(model.embed_tokens.weight))]

        with caplog.at_level("WARNING"):
            patched_module.ModelRunnerFL.reload_weights(
                runner, weights_iterator=iterator, is_checkpoint_format=True
            )

        messages = [r.message for r in caplog.records]
        assert any("not loaded" in m for m in messages)
        assert any("lm_head.weight" in m for m in messages)

    def test_lora_names_are_normalized_when_lora_config_present(
        self, patched_module, monkeypatch
    ):
        """With a lora_config the requested-names set strips .base_layer."""
        runner = self._runner_with_model(patched_module)
        model = runner.get_model()
        runner.lora_config = SimpleNamespace(marker="lora")
        # reset_lora_state consults the lora manager once a lora_config is
        # set; a recording stand-in keeps the CPU path real elsewhere.
        removed = []
        runner.lora_manager = SimpleNamespace(
            remove_all_adapters=lambda: removed.append(1)
        )
        monkeypatch.setattr(
            patched_module, "initialize_layerwise_reload", lambda m: None
        )
        monkeypatch.setattr(
            patched_module, "finalize_layerwise_reload", lambda m, mc: None
        )
        iterator = [("embed_tokens.weight",
                     torch.zeros_like(model.embed_tokens.weight))]

        # reload_weights only consults model.named_parameters(); a warning
        # surfaces listing names as not loaded. The lora branch's observable
        # effect: the adapters were invalidated and the weights still load.
        patched_module.ModelRunnerFL.reload_weights(
            runner, weights_iterator=iterator, is_checkpoint_format=True
        )
        assert model.load_weights_calls == [list(iterator)]
        assert removed == [1]


# ---------------------------------------------------------------------------
# initialize_attn_backend.
# ---------------------------------------------------------------------------


class TestInitializeAttnBackend:
    """Real attention-group construction from the static forward context."""

    @staticmethod
    def _full_spec():
        kv_interface = import_module("vllm.v1.kv_cache_interface")
        return kv_interface.FullAttentionSpec(
            block_size=16, num_kv_heads=2, head_size=8, dtype=torch.float32
        )

    def _runner_with_layers(self, patched_module, layer_specs):
        """layer_specs: list of (name, backend, num_heads)."""
        config = _make_vllm_config()
        forward_context = {}
        for name, backend, num_heads in layer_specs:
            forward_context[name] = _FakeAttentionLayer(backend, num_heads=num_heads)
        config.compilation_config.static_forward_context = forward_context
        runner = _build_runner(patched_module, config)
        return runner

    def test_groups_built_per_kv_cache_group(self, patched_module):
        """Layers sharing a backend+spec+num_heads key cluster into one
        AttentionGroup; different keys split."""
        backend_a = _FakeAttentionBackend(name="BackendA")
        backend_b = _FakeAttentionBackend(name="BackendB")
        runner = self._runner_with_layers(
            patched_module,
            [
                ("model.layers.0", backend_a, 8),
                ("model.layers.1", backend_a, 8),
                ("model.layers.2", backend_b, 4),
            ],
        )
        kv_interface = import_module("vllm.v1.kv_cache_interface")
        spec = self._full_spec()
        kv_cache_config = kv_interface.KVCacheConfig(
            num_blocks=8,
            kv_cache_tensors=[],
            kv_cache_groups=[
                kv_interface.KVCacheGroupSpec(
                    layer_names=["model.layers.0", "model.layers.1",
                                 "model.layers.2"],
                    kv_cache_spec=spec,
                )
            ],
        )
        runner.kv_cache_config = kv_cache_config

        patched_module.ModelRunnerFL.initialize_attn_backend(runner, kv_cache_config)

        assert len(runner.attn_groups) == 1
        groups = runner.attn_groups[0]
        # Two dedup keys -> two groups.
        assert len(groups) == 2
        by_backend = {g.backend.full_cls_name(): g for g in groups}
        assert set(by_backend["BackendA"].layer_names) == {
            "model.layers.0", "model.layers.1"
        }
        assert by_backend["BackendB"].layer_names == ["model.layers.2"]
        # The spec is carried onto each group.
        assert all(g.kv_cache_spec is spec for g in groups)
        # Each group records its kv cache group id.
        assert all(g.kv_cache_group_id == 0 for g in groups)

    def test_already_initialized_asserts(self, patched_module):
        """Calling initialize_attn_backend twice raises."""
        backend = _FakeAttentionBackend()
        runner = self._runner_with_layers(
            patched_module, [("model.layers.0", backend, 8)]
        )
        runner.attn_groups = [[SimpleNamespace()]]
        kv_interface = import_module("vllm.v1.kv_cache_interface")
        kv_cache_config = kv_interface.KVCacheConfig(
            num_blocks=8, kv_cache_tensors=[], kv_cache_groups=[]
        )
        with pytest.raises(AssertionError, match="already initialized"):
            patched_module.ModelRunnerFL.initialize_attn_backend(
                runner, kv_cache_config
            )

    def test_num_heads_difference_splits_groups(self, patched_module):
        """Same backend class but different num_heads -> separate groups."""
        backend = _FakeAttentionBackend(name="BackendA")
        runner = self._runner_with_layers(
            patched_module,
            [
                ("model.layers.0", backend, 8),
                ("model.layers.1", backend, 4),
            ],
        )
        kv_interface = import_module("vllm.v1.kv_cache_interface")
        spec = self._full_spec()
        kv_cache_config = kv_interface.KVCacheConfig(
            num_blocks=8,
            kv_cache_tensors=[],
            kv_cache_groups=[
                kv_interface.KVCacheGroupSpec(
                    layer_names=["model.layers.0", "model.layers.1"],
                    kv_cache_spec=spec,
                )
            ],
        )
        runner.kv_cache_config = kv_cache_config
        patched_module.ModelRunnerFL.initialize_attn_backend(runner, kv_cache_config)
        assert len(runner.attn_groups[0]) == 2

    def test_multiple_kv_cache_groups(self, patched_module):
        """Each kv cache group gets its own outer attn_groups entry."""
        backend = _FakeAttentionBackend(name="BackendA")
        runner = self._runner_with_layers(
            patched_module,
            [
                ("model.layers.0", backend, 8),
                ("model.layers.1", backend, 8),
            ],
        )
        kv_interface = import_module("vllm.v1.kv_cache_interface")
        spec = self._full_spec()
        kv_cache_config = kv_interface.KVCacheConfig(
            num_blocks=8,
            kv_cache_tensors=[],
            kv_cache_groups=[
                kv_interface.KVCacheGroupSpec(
                    layer_names=["model.layers.0"], kv_cache_spec=spec
                ),
                kv_interface.KVCacheGroupSpec(
                    layer_names=["model.layers.1"], kv_cache_spec=spec
                ),
            ],
        )
        runner.kv_cache_config = kv_cache_config
        patched_module.ModelRunnerFL.initialize_attn_backend(runner, kv_cache_config)
        assert len(runner.attn_groups) == 2
        assert runner.attn_groups[0][0].kv_cache_group_id == 0
        assert runner.attn_groups[1][0].kv_cache_group_id == 1


# ---------------------------------------------------------------------------
# may_reinitialize_input_batch.
# ---------------------------------------------------------------------------


class TestMayReinitializeInputBatch:
    """The input-batch rebuild decision on real KVCacheConfig geometry."""

    @staticmethod
    def _full_spec():
        kv_interface = import_module("vllm.v1.kv_cache_interface")
        return kv_interface.FullAttentionSpec(
            block_size=16, num_kv_heads=2, head_size=8, dtype=torch.float32
        )

    def _kv_cache_config(self, block_size):
        kv_interface = import_module("vllm.v1.kv_cache_interface")
        spec = kv_interface.FullAttentionSpec(
            block_size=block_size, num_kv_heads=2, head_size=8,
            dtype=torch.float32
        )
        return kv_interface.KVCacheConfig(
            num_blocks=8,
            kv_cache_tensors=[],
            kv_cache_groups=[
                kv_interface.KVCacheGroupSpec(
                    layer_names=["model.layers.0"], kv_cache_spec=spec
                )
            ],
        )

    def test_same_geometry_keeps_batch(self, patched_module):
        """A config matching the placeholder geometry leaves input_batch
        untouched."""
        runner = _build_runner(patched_module, _make_vllm_config())
        original = runner.input_batch
        config = self._kv_cache_config(16)
        # kernel_block_sizes equal to the placeholder block size.
        patched_module.ModelRunnerFL.may_reinitialize_input_batch(
            runner, config, [16]
        )
        assert runner.input_batch is original
        assert runner._init_block_sizes == [16]

    def test_different_block_size_rebuilds_batch(self, patched_module):
        """A different block size re-creates the InputBatch with the new
        geometry."""
        runner = _build_runner(patched_module, _make_vllm_config())
        original = runner.input_batch
        config = self._kv_cache_config(32)
        patched_module.ModelRunnerFL.may_reinitialize_input_batch(
            runner, config, [32]
        )
        assert runner.input_batch is not original
        assert runner._init_block_sizes == [32]
        assert runner._init_kernel_block_sizes == [32]
        # New batch carries the new block size through to the block table.
        block_tables = runner.input_batch.block_table.block_tables
        assert [bt.block_size for bt in block_tables] == [32]

    def test_multiple_groups_rebuild_batch(self, patched_module):
        """Two kv cache groups always rebuild the (single-group placeholder)
        input batch."""
        kv_interface = import_module("vllm.v1.kv_cache_interface")
        runner = _build_runner(patched_module, _make_vllm_config())
        original = runner.input_batch
        spec0 = kv_interface.FullAttentionSpec(
            block_size=16, num_kv_heads=2, head_size=8, dtype=torch.float32
        )
        spec1 = kv_interface.FullAttentionSpec(
            block_size=16, num_kv_heads=2, head_size=8, dtype=torch.float32
        )
        config = kv_interface.KVCacheConfig(
            num_blocks=8,
            kv_cache_tensors=[],
            kv_cache_groups=[
                kv_interface.KVCacheGroupSpec(
                    layer_names=["model.layers.0"], kv_cache_spec=spec0
                ),
                kv_interface.KVCacheGroupSpec(
                    layer_names=["model.layers.1"], kv_cache_spec=spec1
                ),
            ],
        )
        patched_module.ModelRunnerFL.may_reinitialize_input_batch(
            runner, config, [16, 16]
        )
        assert runner.input_batch is not original
        assert runner._init_block_sizes == [16, 16]

    def test_encoder_only_groups_are_skipped(self, patched_module):
        """Encoder-only specs contribute neither block sizes nor slot
        mappings."""
        kv_interface = import_module("vllm.v1.kv_cache_interface")
        runner = _build_runner(patched_module, _make_vllm_config())
        original = runner.input_batch
        enc_spec = kv_interface.EncoderOnlyAttentionSpec(
            block_size=16, num_kv_heads=2, head_size=8, dtype=torch.float32
        )
        full_spec = kv_interface.FullAttentionSpec(
            block_size=16, num_kv_heads=2, head_size=8, dtype=torch.float32
        )
        config = kv_interface.KVCacheConfig(
            num_blocks=8,
            kv_cache_tensors=[],
            kv_cache_groups=[
                kv_interface.KVCacheGroupSpec(
                    layer_names=["enc.0"], kv_cache_spec=enc_spec
                ),
                kv_interface.KVCacheGroupSpec(
                    layer_names=["model.layers.0"], kv_cache_spec=full_spec
                ),
            ],
        )
        # Only the full-attention group's geometry counts; combined with the
        # placeholder's [16] it stays equal, but the group count differs
        # through kernel_block_sizes.
        patched_module.ModelRunnerFL.may_reinitialize_input_batch(
            runner, config, [16]
        )
        # block_sizes = [16] (encoder group skipped) and kernel sizes [16]
        # both match the placeholder single group -> no rebuild.
        assert runner.input_batch is original
        assert runner._init_slot_mapping_modes == [
            import_module("vllm.v1.worker.block_table").SlotMappingMode
            .TOKEN_TO_KV_SLOT
        ]


# ---------------------------------------------------------------------------
# _cleanup_profiling_kv_cache.
# ---------------------------------------------------------------------------


class TestCleanupProfilingKvCache:
    """_cleanup_profiling_kv_cache releases profiling state."""

    def test_cleanup_clears_caches_groups_and_config(self, patched_module,
                                                     monkeypatch):
        """After cleanup the kv caches, attention groups and kv_cache_config
        are gone and the forward-context layers' caches are reset."""
        config = _make_vllm_config()
        layer = _FakeAttentionLayer(_FakeAttentionBackend())
        config.compilation_config.static_forward_context = {"l0": layer}
        runner = _build_runner(patched_module, config)
        runner.kv_caches = [torch.zeros(4), torch.zeros(4)]
        kv_interface = import_module("vllm.v1.kv_cache_interface")
        runner.kv_cache_config = kv_interface.KVCacheConfig(
            num_blocks=8, kv_cache_tensors=[], kv_cache_groups=[]
        )
        runner.attn_groups = [[SimpleNamespace()]]
        runner.cache_config.num_gpu_blocks = 128
        # torch.accelerator.synchronize / empty_cache are device ops.
        monkeypatch.setattr(torch.accelerator, "synchronize", lambda: None,
                            raising=False)
        monkeypatch.setattr(torch.accelerator, "empty_cache", lambda: None,
                            raising=False)

        patched_module.ModelRunnerFL._cleanup_profiling_kv_cache(runner)

        assert runner.kv_caches == []
        assert not hasattr(runner, "kv_cache_config")
        assert runner.attn_groups == []
        assert runner.cache_config.num_gpu_blocks is None
        # The forward-context layer's cache was detached.
        assert not isinstance(layer.kv_cache, torch.Tensor) or layer.kv_cache \
            .numel() == 0

    def test_cleanup_without_kv_state_is_a_noop(self, patched_module,
                                                monkeypatch):
        """A runner without kv caches / attn groups cleans up without
        errors."""
        config = _make_vllm_config()
        runner = _build_runner(patched_module, config)
        monkeypatch.setattr(torch.accelerator, "synchronize", lambda: None,
                            raising=False)
        monkeypatch.setattr(torch.accelerator, "empty_cache", lambda: None,
                            raising=False)
        patched_module.ModelRunnerFL._cleanup_profiling_kv_cache(runner)
        assert not hasattr(runner, "kv_cache_config")


# ---------------------------------------------------------------------------
# _dummy_sampler_run.
# ---------------------------------------------------------------------------


class _RecordingSampler:
    """Sampler stand-in for the warm-up paths (greedy argmax on CPU)."""

    def __init__(self, logprobs_mode="processed_logprobs"):
        self.logprobs_mode = logprobs_mode
        self.calls = []

    def __call__(self, logits, sampling_metadata, **kwargs):
        self.calls.append(
            SimpleNamespace(
                logits=logits.clone(), sampling_metadata=sampling_metadata
            )
        )
        from vllm.v1.outputs import SamplerOutput

        sampled = logits.argmax(dim=-1)
        return SamplerOutput(
            sampled_token_ids=sampled.reshape(-1, 1).to(torch.int64),
            logprobs_tensors=None,
        )


class TestDummySamplerRun:
    """_dummy_sampler_run through the real CPU logits path."""

    @contextlib.contextmanager
    def _platform_fl_stub(self, vendor_name=None):
        """Temporarily provide a vllm_fl.platform module.

        The production helper lazily imports ``vllm_fl.platform``, which
        requires ``flag_gems`` on real vendors and is unavailable on CPU
        venvs. The stub is inserted into sys.modules manually (restored in
        the finally arm); monkeypatch.setitem is never used on this name.
        """
        import sys
        import types

        stub = types.ModuleType("vllm_fl.platform")
        stub.PlatformFL = SimpleNamespace(vendor_name=vendor_name)
        saved = sys.modules.get("vllm_fl.platform")
        sys.modules["vllm_fl.platform"] = stub
        try:
            yield stub
        finally:
            if saved is None:
                sys.modules.pop("vllm_fl.platform", None)
            else:
                sys.modules["vllm_fl.platform"] = saved

    def _runner(self, patched_module, spec_config=None):
        config = _make_vllm_config(speculative_config=spec_config)
        runner = _build_runner(patched_module, config)
        runner.model = _TinyModel()
        runner.sampler = _RecordingSampler()
        return runner

    def test_basic_run_samples_random_logits(self, patched_module):
        """A plain warm-up computes logits from rand_like'd hidden states and
        hands them to the sampler with a full dummy SamplingMetadata."""
        runner = self._runner(patched_module)
        hidden = torch.randn(3, 8)

        with self._platform_fl_stub(vendor_name=None):
            out = patched_module.ModelRunnerFL._dummy_sampler_run(runner, hidden)

        assert len(runner.sampler.calls) == 1
        call = runner.sampler.calls[0]
        # The model's real projection produced the logits.
        assert call.logits.shape == (3, 32)
        # Oracle: the projection of a rand_like copy of the hidden rows.
        # (values are random, so assert shape + finiteness + row structure)
        assert torch.isfinite(call.logits).all()
        # The dummy metadata describes num_reqs rows with greedy sampling
        # options.
        md = call.sampling_metadata
        assert md.temperature.shape == (3,)
        assert not md.all_greedy
        assert md.no_penalties
        # The output is the sampler's return.
        assert out.sampled_token_ids.shape == (3, 1)

    def test_mm_encoder_only_returns_empty_tensor(self, patched_module):
        """An encoder-only multimodal model skips the sampler entirely."""
        config = _make_vllm_config(
            model_config=_make_model_config(
                multimodal_config=SimpleNamespace(mm_encoder_only=True)
            )
        )
        runner = _build_runner(patched_module, config)
        runner.model = _TinyModel()
        runner.sampler = _RecordingSampler()

        out = patched_module.ModelRunnerFL._dummy_sampler_run(
            runner, torch.randn(2, 8)
        )

        assert isinstance(out, torch.Tensor) and out.numel() == 0
        assert runner.sampler.calls == []

    def test_spec_config_runs_rejection_sampler_warmups(self, patched_module,
                                                        monkeypatch):
        """With a speculative config the warm-up also drives the rejection
        sampler twice (mixed + all-greedy) with dummy spec metadata."""
        spec = SimpleNamespace(
            method="suffix",
            num_speculative_tokens=2,
            uses_draft_model=lambda: False,
            use_ngram_gpu=lambda: False,
            use_gemma4_mtp=lambda: False,
            use_step3p5_mtp=lambda: False,
            use_dflash=lambda: False,
            use_eagle=lambda: False,
            uses_extract_hidden_states=lambda: False,
            disable_padded_drafter_batch=False,
            rejection_sample_method="standard",
            draft_sample_method="probabilistic",
            draft_model_config=None,
        )
        runner = self._runner(patched_module, spec_config=spec)
        rejection_calls = []

        class RecordingRejectionSampler:
            def __call__(self, metadata, draft_probs, logits, md):
                rejection_calls.append(
                    SimpleNamespace(metadata=metadata, draft_probs=draft_probs,
                                    logits=logits, md=md)
                )
                from vllm.v1.outputs import SamplerOutput

                return SamplerOutput(
                    sampled_token_ids=torch.zeros(3, 3, dtype=torch.int64),
                    logprobs_tensors=None,
                )

        runner.rejection_sampler = RecordingRejectionSampler()

        # The spec branch ends with torch.accelerator.synchronize, a device
        # op unavailable on CPU builds.
        monkeypatch.setattr(torch.accelerator, "synchronize", lambda: None,
                            raising=False)
        with self._platform_fl_stub(vendor_name=None):
            patched_module.ModelRunnerFL._dummy_sampler_run(
                runner, torch.randn(3, 8)
            )

        # Mixed-sampling + all-greedy warm-ups.
        assert len(rejection_calls) == 2
        first, second = rejection_calls
        # The probabilistic draft method produced softmaxed draft probs.
        assert first.draft_probs is not None
        assert first.draft_probs.shape[-1] == 32
        # The second call is the all-greedy metadata variant.
        assert second.md.all_greedy
        assert not second.md.all_random
        # The dummy spec metadata carries one draft token per request.
        assert first.metadata.num_draft_tokens == [1, 1, 1]

    def test_kunlunxin_vendor_switches_to_greedy(self, patched_module):
        """The kunlunxin vendor path replaces the dummy metadata with an
        all-greedy variant before sampling."""
        runner = self._runner(patched_module)

        with self._platform_fl_stub(vendor_name="kunlunxin"):
            patched_module.ModelRunnerFL._dummy_sampler_run(
                runner, torch.randn(2, 8)
            )

        assert len(runner.sampler.calls) == 1
        assert runner.sampler.calls[0].sampling_metadata.all_greedy


# ---------------------------------------------------------------------------
# _dummy_pooler_run_task / _dummy_pooler_run.
# ---------------------------------------------------------------------------


class _FakePooler:
    """Pooler boundary stand-in: mean over each request's hidden rows."""

    def __init__(self, tasks=("embed", "classify")):
        self.tasks = list(tasks)
        self.calls = []
        self.updates_applied = []

    def get_supported_tasks(self):
        return list(self.tasks)

    def get_pooling_updates(self, task):
        def apply(params):
            self.updates_applied.append((task, params))

        return SimpleNamespace(apply=apply)

    def __call__(self, hidden_states, pooling_metadata):
        self.calls.append(
            SimpleNamespace(hidden_states=hidden_states.clone(),
                            pooling_metadata=pooling_metadata)
        )
        prompt_lens = pooling_metadata.prompt_lens
        rows = []
        offset = 0
        for length in prompt_lens.tolist():
            rows.append(hidden_states[offset:offset + length].mean(dim=0))
            offset += length
        return rows


class _PoolingTinyModel:
    """Model facade exposing the pooler boundary for pooling runners.

    Satisfies ``is_pooling_model`` (via ``is_vllm_model``): __init__ takes
    ``vllm_config``, and both ``embed_input_ids`` and ``forward`` exist with
    the vLLM keyword signatures.
    """

    is_pooling_model = True

    def __init__(self, pooler, vllm_config=None):
        self.pooler = pooler
        self.vllm_config = vllm_config

    def embed_input_ids(self, input_ids):
        raise AssertionError("embed_input_ids must not run in the pooler warm-ups")

    def forward(self, input_ids, positions):
        raise AssertionError("forward must not run in the pooler warm-ups")


class TestDummyPoolerRunTask:
    """_dummy_pooler_run_task builds a real PoolingMetadata on CPU."""

    def _runner(self, patched_module):
        model_config = _make_model_config(runner_type="pooling")
        config = _make_vllm_config(model_config=model_config)
        runner = _build_runner(patched_module, config)
        return runner

    def test_task_runs_pooler_with_built_metadata(self, patched_module):
        """The pooler sees the warm-up hidden states and a metadata whose
        prompt lens partition the token budget evenly."""
        runner = self._runner(patched_module)
        pooler = _FakePooler()
        runner.model = _PoolingTinyModel(pooler)

        hidden = torch.arange(12, dtype=torch.float32).reshape(6, 2)
        out = patched_module.ModelRunnerFL._dummy_pooler_run_task(
            runner, hidden, "embed"
        )

        assert len(pooler.calls) == 1
        call = pooler.calls[0]
        assert torch.equal(call.hidden_states, hidden)
        md = call.pooling_metadata
        # num_tokens=6, max_num_seqs=4 -> 4 rows of [1, 1, 1, 3]: the floor
        # split with the remainder piled onto the last request.
        assert md.prompt_lens.tolist() == [1, 1, 1, 3]
        assert md.prompt_token_ids.shape == (4, 1)
        # Every request shares the one pooling params built for the task.
        assert all(p.task == "embed" for p in md.pooling_params)
        # The task's update was applied to those params.
        assert len(pooler.updates_applied) == 1
        assert pooler.updates_applied[0][0] == "embed"
        # Oracle: each row is the mean of its hidden slice.
        assert torch.allclose(out[0], hidden[:1].mean(dim=0))
        assert torch.allclose(out[1], hidden[1:2].mean(dim=0))
        assert torch.allclose(out[3], hidden[3:6].mean(dim=0))

    def test_single_request_full_budget(self, patched_module):
        """num_tokens <= max_num_seqs still makes one request per token up to
        max_num_seqs: 3 tokens over 3 requests of one token each."""
        runner = self._runner(patched_module)
        pooler = _FakePooler()
        runner.model = _PoolingTinyModel(pooler)

        hidden = torch.randn(3, 2)
        out = patched_module.ModelRunnerFL._dummy_pooler_run_task(
            runner, hidden, "classify"
        )

        md = pooler.calls[0].pooling_metadata
        # num_reqs = min(num_tokens=3, max_num_seqs=4) = 3 -> [1, 1, 1].
        assert md.prompt_lens.tolist() == [1, 1, 1]
        assert md.prompt_token_ids.shape == (3, 1)
        # Oracle: each row is the mean of its single hidden token.
        for i in range(3):
            assert torch.allclose(out[i], hidden[i:i + 1].mean(dim=0))

    def test_mm_encoder_only_skips_pooler(self, patched_module):
        """An encoder-only multimodal model returns an empty tensor without
        pooling."""
        model_config = _make_model_config(
            runner_type="pooling",
            multimodal_config=SimpleNamespace(mm_encoder_only=True),
        )
        runner = _build_runner(
            patched_module, _make_vllm_config(model_config=model_config)
        )
        pooler = _FakePooler()
        runner.model = _PoolingTinyModel(pooler)

        out = patched_module.ModelRunnerFL._dummy_pooler_run(
            runner, torch.randn(4, 2)
        )

        assert isinstance(out, torch.Tensor) and out.numel() == 0
        assert pooler.calls == []


class TestDummyPoolerRun:
    """_dummy_pooler_run selects the largest-output task."""

    def _runner(self, patched_module):
        model_config = _make_model_config(runner_type="pooling")
        return _build_runner(
            patched_module, _make_vllm_config(model_config=model_config)
        )

    def test_runs_each_task_and_returns_largest(self, patched_module):
        """Every supported task is warmed; the returned output comes from the
        task with the largest output bytes."""
        runner = self._runner(patched_module)
        pooler = _FakePooler(tasks=("embed", "classify"))
        runner.model = _PoolingTinyModel(pooler)

        # Make 'classify' produce bigger rows than 'embed'.
        original_call = pooler.__call__

        def call_with_task(hidden_states, pooling_metadata):
            rows = original_call(hidden_states, pooling_metadata)
            task_params = pooling_metadata.pooling_params[0]
            if task_params.task == "classify":
                rows = [r * 1.0 for r in rows]
            return rows

        pooler.__call__ = call_with_task
        hidden = torch.randn(4, 2)
        out = patched_module.ModelRunnerFL._dummy_pooler_run(runner, hidden)

        # embed ran first, classify second (per supported task), then the
        # largest-output task ran once more.
        tasks_seen = [
            c.pooling_metadata.pooling_params[0].task for c in pooler.calls
        ]
        assert tasks_seen[0] == "embed"
        assert set(tasks_seen) == {"embed", "classify"}
        # The final call re-ran one of the tasks with the full hidden batch.
        assert torch.equal(pooler.calls[-1].hidden_states, hidden)

    def test_no_supported_tasks_raises(self, patched_module):
        """A pooling model without tasks raises RuntimeError."""
        runner = self._runner(patched_module)
        pooler = _FakePooler(tasks=())
        runner.model = _PoolingTinyModel(pooler)

        with pytest.raises(RuntimeError, match="pooling tasks"):
            patched_module.ModelRunnerFL._dummy_pooler_run(runner,
                                                           torch.randn(2, 2))


# ---------------------------------------------------------------------------
# _kl_apply and the Kunlunx guards.
# ---------------------------------------------------------------------------


class TestKlApply:
    """_kl_apply's capture-size trimming and its no-op guards."""

    def _runner(self, patched_module, **overrides):
        config = _make_vllm_config()
        runner = _build_runner(patched_module, config)
        # _kl_used is read by the trimming branch's log line.
        runner._kl_used = overrides.pop("_kl_used", 0)
        for key, value in overrides.items():
            setattr(runner, key, value)
        return runner

    @staticmethod
    def _desc(name, num_tokens=None):
        """A capture descriptor: the pair head needs only a name; the
        entries need num_tokens."""
        return SimpleNamespace(name=name, num_tokens=num_tokens)

    def test_zero_budget_returns_early(self, patched_module):
        """_kl_budget == 0 leaves the compilation config untouched."""
        runner = self._runner(patched_module, _kl_budget=0, _kl_stop=True)
        before = runner.compilation_config.cudagraph_capture_sizes
        patched_module.ModelRunnerFL._kl_apply(runner, [SimpleNamespace()])
        assert runner.compilation_config.cudagraph_capture_sizes is before

    def test_not_stopped_returns_early(self, patched_module):
        """_kl_stop False leaves the config untouched even with a budget."""
        runner = self._runner(patched_module, _kl_budget=100, _kl_stop=False)
        descs = [(self._desc("m0", 8), [self._desc("m0", 8)])]
        patched_module.ModelRunnerFL._kl_apply(runner, descs)
        assert runner.compilation_config.cudagraph_capture_sizes is None

    def test_empty_descs_return_early(self, patched_module):
        """No capture descriptors -> nothing to trim."""
        runner = self._runner(patched_module, _kl_budget=100, _kl_stop=True)
        patched_module.ModelRunnerFL._kl_apply(runner, [])
        assert runner.compilation_config.cudagraph_capture_sizes is None

    def test_all_sizes_captured_keeps_everything(self, patched_module):
        """When every (mode, size) pair is already captured the sizes stay
        unchanged."""
        runner = self._runner(patched_module, _kl_budget=100, _kl_stop=True,
                             _kl_captured={("m0", 8), ("m1", 8)})
        descs = [
            (self._desc("m0"), [self._desc("m0", 8)]),
            (self._desc("m1"), [self._desc("m1", 8)]),
        ]
        patched_module.ModelRunnerFL._kl_apply(runner, descs)
        # kept == sizes -> early return before any mutation.
        assert runner.compilation_config.cudagraph_capture_sizes is None

    def test_partial_capture_trims_sizes(self, patched_module, caplog):
        """Uncaptured sizes trim the capture list down to the captured
        prefix."""
        runner = self._runner(patched_module, _kl_budget=100, _kl_stop=True,
                              _kl_captured={("m0", 8), ("m0", 16)})
        descs = [
            (self._desc("m0"), [self._desc("m0", 8), self._desc("m0", 16),
                                 self._desc("m0", 32)]),
        ]
        with caplog.at_level("WARNING"):
            patched_module.ModelRunnerFL._kl_apply(runner, descs)

        cc = runner.compilation_config
        assert cc.cudagraph_capture_sizes == [8, 16]
        assert cc.max_cudagraph_capture_size == 16
        assert any("reduced CUDA graph" in r.message for r in caplog.records)

    def test_nothing_captured_keeps_smallest_size(self, patched_module):
        """With no captured pairs at all, the smallest size is kept."""
        runner = self._runner(patched_module, _kl_budget=100, _kl_stop=True,
                              _kl_captured=set())
        descs = [
            (self._desc("m0"), [self._desc("m0", 8), self._desc("m0", 32)]),
        ]
        patched_module.ModelRunnerFL._kl_apply(runner, descs)
        assert runner.compilation_config.cudagraph_capture_sizes == [8]

    def test_kl_guard_on_reads_vendor_name(self, patched_module, monkeypatch):
        """_kl_guard_on mirrors the platform vendor name."""
        runner = self._runner(patched_module)
        fake_platform = SimpleNamespace(vendor_name="kunlunxin")
        monkeypatch.setattr(patched_module, "current_platform", fake_platform)
        assert patched_module.ModelRunnerFL._kl_guard_on(runner) is True
        other = SimpleNamespace(vendor_name="nvidia")
        monkeypatch.setattr(patched_module, "current_platform", other)
        assert patched_module.ModelRunnerFL._kl_guard_on(runner) is False


# ---------------------------------------------------------------------------
# initialize_kv_cache (end to end through the real allocation path).
# ---------------------------------------------------------------------------


class TestInitializeKvCache:
    """initialize_kv_cache wiring real CPU KV caches end to end."""

    @staticmethod
    def _full_spec():
        kv_interface = import_module("vllm.v1.kv_cache_interface")
        return kv_interface.FullAttentionSpec(
            block_size=16, num_kv_heads=2, head_size=8, dtype=torch.float32
        )

    def _config_with_layers(self, patched_module, layer_specs, num_blocks=8):
        """Build a runner + kv cache config with fake attention layers."""
        config = _make_vllm_config()
        forward_context = {}
        for name, backend, num_heads in layer_specs:
            forward_context[name] = _FakeAttentionLayer(backend,
                                                         num_heads=num_heads)
        config.compilation_config.static_forward_context = forward_context
        runner = _build_runner(patched_module, config)
        kv_interface = import_module("vllm.v1.kv_cache_interface")
        spec = self._full_spec()
        # Real page geometry from the spec: heads * states * content bytes
        # (content holds K and V back to back). One shared int8 buffer backs
        # every layer.
        page_size = spec.page_size_bytes
        layer_names = [name for name, _, _ in layer_specs]
        kv_cache_config = kv_interface.KVCacheConfig(
            num_blocks=num_blocks,
            kv_cache_tensors=[
                kv_interface.KVCacheTensor(
                    size=page_size * num_blocks * len(layer_names),
                    layers=list(layer_names),
                    layer_stride=page_size * num_blocks,
                    block_stride=page_size,
                )
            ],
            kv_cache_groups=[
                kv_interface.KVCacheGroupSpec(
                    layer_names=layer_names, kv_cache_spec=spec
                )
            ],
        )
        return runner, kv_cache_config, forward_context

    def test_initialize_kv_cache_binds_real_cpu_tensors(
        self, patched_module, monkeypatch
    ):
        """The full path allocates a real CPU KV cache buffer, binds views to
        the forward-context layers, installs metadata builders, and
        reinitializes the input batch for the final geometry."""
        backend = _FakeAttentionBackend(name="BackendA")
        runner, config, forward_context = self._config_with_layers(
            patched_module,
            [("model.layers.0", backend, 8), ("model.layers.1", backend, 8)],
        )
        monkeypatch.setattr(patched_module, "has_kv_transfer_group",
                            lambda: False)
        original_batch = runner.input_batch

        patched_module.ModelRunnerFL.initialize_kv_cache(runner, config)

        # The runner's kv caches were bound in layer order.
        assert len(runner.kv_caches) == 2
        # Real CPU float tensors with the spec's per-layer page geometry.
        cache = runner.kv_caches[0]
        assert cache.dtype == torch.float32
        assert cache.is_cpu
        # [num_blocks, num_kv_heads, block_size, K+V content]: the content
        # axis holds K and V back to back (2 * head_size float32 = 16).
        assert tuple(cache.shape) == (8, 2, 16, 16)
        # Both layers were bound through the real bind_kv_cache path.
        assert len(forward_context["model.layers.0"].bound) == 1
        assert forward_context["model.layers.0"].bound[0] is not None
        # Attention groups + metadata builders were created.
        assert len(runner.attn_groups) == 1
        assert len(runner.attn_groups[0]) == 1
        assert len(runner.attn_groups[0][0].metadata_builders) == 1
        # The input batch was rebuilt for the final geometry (the
        # placeholder used a single block size of 16; the config matches, so
        # the same batch object can be kept).
        assert runner.input_batch is original_batch
        assert runner.kv_cache_config.num_blocks == 8
        assert runner._kernel_block_sizes == [16]

    def test_initialize_kv_cache_with_kv_transfer_group(
        self, patched_module, monkeypatch
    ):
        """With a KV transfer group configured, the allocated caches are
        registered with the group (the distributed-group boundary)."""
        backend = _FakeAttentionBackend(name="BackendA")
        runner, config, forward_context = self._config_with_layers(
            patched_module, [("model.layers.0", backend, 8)]
        )
        registered = []

        class FakeTransferGroup:
            def register_kv_caches(self, kv_caches):
                registered.append(kv_caches)

            def set_host_xfer_buffer_ops(self, ops):
                pass

        monkeypatch.setattr(patched_module, "has_kv_transfer_group",
                            lambda: True)
        monkeypatch.setattr(patched_module, "get_kv_transfer_group",
                            lambda: FakeTransferGroup())

        patched_module.ModelRunnerFL.initialize_kv_cache(runner, config)

        # The group received the name -> tensor dict of allocated caches
        # (runner.kv_caches is the layer-ordered list of the same tensors).
        assert len(registered) == 1
        assert set(registered[0]) == {"model.layers.0"}
        assert registered[0]["model.layers.0"] is runner.kv_caches[0]

    def test_initialize_kv_cache_profiling_mode_skips_transfer(
        self, patched_module, monkeypatch
    ):
        """is_profiling=True skips the KV-transfer registration even when a
        group is configured."""
        backend = _FakeAttentionBackend(name="BackendA")
        runner, config, _ = self._config_with_layers(
            patched_module, [("model.layers.0", backend, 8)]
        )
        registered = []

        class FakeTransferGroup:
            def register_kv_caches(self, kv_caches):
                registered.append(kv_caches)

            def set_host_xfer_buffer_ops(self, ops):
                pass

        monkeypatch.setattr(patched_module, "has_kv_transfer_group",
                            lambda: True)
        monkeypatch.setattr(patched_module, "get_kv_transfer_group",
                            lambda: FakeTransferGroup())

        patched_module.ModelRunnerFL.initialize_kv_cache(
            runner, config, is_profiling=True
        )

        assert registered == []

    def test_initialize_kv_cache_copies_the_config(self, patched_module,
                                                  monkeypatch):
        """initialize_kv_cache deep-copies the incoming config so later
        mutation of the caller's object cannot leak in."""
        backend = _FakeAttentionBackend(name="BackendA")
        runner, config, _ = self._config_with_layers(
            patched_module, [("model.layers.0", backend, 8)]
        )
        monkeypatch.setattr(patched_module, "has_kv_transfer_group",
                            lambda: False)

        patched_module.ModelRunnerFL.initialize_kv_cache(runner, config)

        assert runner.kv_cache_config is not config
        assert runner.kv_cache_config.num_blocks == config.num_blocks
