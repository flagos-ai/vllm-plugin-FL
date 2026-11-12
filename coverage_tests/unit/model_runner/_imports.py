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

"""Shared construction recipe for CPU-side ModelRunnerFL instances.

Reuses the W4 worker-area shim (uvloop / flag_gems / xgrammar placeholders,
vllm 0.24-line symbol backfills, CUDA-like platform import window) and adds
the local-only patch layer that makes a real ModelRunnerFL constructible on
this CPU-only dev host. On the A100 CI image every step is a no-op guarded
by a device/probe check:

* pinned memory — ``PIN_MEMORY`` is True on the UnspecifiedPlatform this
  host resolves but ``torch.zeros(pin_memory=True)`` fails without CUDA,
  so the module constants and the CpuGpuBuffer default are forced to
  False only while a runner is being built.
* ``num_compute_units`` — local platform raises NotImplementedError; the
  runner's SM count is a device probe (CONTRACT patch boundary) pinned
  to a deterministic 8.
* ``InputBatch`` — local vllm 0.33 dropped ``kernel_block_sizes`` in favor
  of ``max_num_blocks_per_req``; the wrapper translates the CI signature
  the plugin calls into the local one.
* ``build_logitsprocs`` — same pin-memory forcing for the logits
  processor constructor path.
* ``MULTIMODAL_REGISTRY`` — local 0.33 registry lacks
  ``supports_multimodal_inputs``; a probe-shaped stand-in returns False
  (the text-only configuration every test here uses).
* gloo process group — ``get_dcp_group``/``get_pp_group`` assert unless a
  one-rank gloo group is initialized; ``make_runner`` sets one up per
  process (idempotent) and leaves teardown to the test.
* ``VLLM_FL_COMMON_ATTENTION_METADATA=1`` — opts into the
  CommonAttentionMetadataGraphRunner so ``_prepare_inputs`` takes the
  graph-present path (as it does on CUDA CI) instead of the local-only
  Triton ``compute_slot_mapping`` branch; the graph pool probe is given
  a CPU platform stand-in during the construction window only.

All patches are applied by ``make_runner`` inside a context manager that
restores the previous module attributes on exit, so no state outlives a
test.
"""

from __future__ import annotations

import inspect
import os
from contextlib import ExitStack, contextmanager, nullcontext as _nullcontext
from functools import wraps
from types import SimpleNamespace
from unittest.mock import patch

import torch

from ..._distributed import single_rank_gloo
from ..worker import _imports as _worker_imports  # noqa: F401  (import shims)

_worker_imports.ensure_importable_env()


def _backfill_common_attention_metadata_fields():
    """Accept the CI's CommonAttentionMetadata kwargs on local vllm 0.33.

    The CI image's vllm (0.24-line) dataclass has ``_seq_lens_cpu`` and
    ``_num_computed_tokens_cpu`` fields the plugin passes positionally by
    keyword; vllm 0.33 dropped both (they are now derived from
    ``seq_lens``). On the CI image the fields exist and this is a no-op.
    Locally the constructor wrapper stores them as plain attributes so the
    plugin's call resolves and tests can assert what was passed through.
    """
    from vllm.v1.attention.backend import CommonAttentionMetadata

    params = inspect.signature(CommonAttentionMetadata.__init__).parameters
    if "_seq_lens_cpu" in params and "_num_computed_tokens_cpu" in params:
        return
    original_init = CommonAttentionMetadata.__init__

    def _init(self, *args, **kwargs):
        extra = {}
        for name in ("_seq_lens_cpu", "_num_computed_tokens_cpu"):
            if name in kwargs and name not in params:
                extra[name] = kwargs.pop(name)
        original_init(self, *args, **kwargs)
        for name, value in extra.items():
            setattr(self, name, value)

    CommonAttentionMetadata.__init__ = _init


_backfill_common_attention_metadata_fields()

# Shared handle on the module under test, for tests that patch its
# input symbols (rotary embeddings, platforms) at the module boundary.
import vllm_fl.worker.model_runner as mr_module  # noqa: E402, F401


class _Patch:
    """Deterministic set/restore of a module attribute."""

    def __init__(self, module, name, value):
        self.module, self.name, self.value = module, name, value

    def __enter__(self):
        self.saved = getattr(self.module, self.name)
        setattr(self.module, self.name, self.value)
        return self

    def __exit__(self, *exc):
        setattr(self.module, self.name, self.saved)
        return False


def _model_config(**overrides):
    """SimpleNamespace model_config with the generate-runner defaults."""
    from types import SimpleNamespace as NS

    values = dict(
        dtype=torch.float16,
        max_model_len=128,
        runner_type="generate",
        enable_prompt_embeds=False,
        is_multimodal_raw_input_only_model=False,
        is_encoder_decoder=False,
        is_mm_prefix_lm=False,
        disable_cascade_attn=False,
        uses_alibi=False,
        uses_mrope=False,
        uses_xdrope_dim=0,
        logprobs_mode="raw_logprobs",
        use_fp64_gumbel=False,
        logits_processors=None,
        get_vocab_size=lambda: 100,
        get_inputs_embeds_size=lambda: 16,
        get_num_attention_heads=lambda pc: 4,
        is_hybrid=False,
        hf_text_config=NS(sliding_window=None),
    )
    values.update(overrides)
    return NS(**values)


def _vllm_config(model_config=None, **overrides):
    """SimpleNamespace vllm_config mirroring the real field set."""
    from types import SimpleNamespace as NS

    from vllm.config import CUDAGraphMode
    from vllm.config.compilation import CompilationConfig
    from vllm.config.offload import OffloadConfig

    values = dict(
        model_config=model_config or _model_config(),
        cache_config=NS(
            cache_dtype="auto",
            calculate_kv_scales=False,
            block_size=16,
            kv_sharing_fast_prefill=False,
            mamba_cache_mode="all",
        ),
        offload_config=OffloadConfig(),
        compilation_config=CompilationConfig(
            cudagraph_capture_sizes=[], cudagraph_mode=CUDAGraphMode.NONE
        ),
        lora_config=None,
        load_config=None,
        parallel_config=NS(
            distributed_executor_backend="mp",
            decode_context_parallel_size=1,
            cp_kv_cache_interleave_size=1,
            tensor_parallel_size=1,
        ),
        scheduler_config=NS(
            max_num_batched_tokens=64,
            max_num_seqs=4,
            max_num_encoder_input_tokens=0,
            async_scheduling=False,
            enable_chunked_prefill=True,
        ),
        speculative_config=None,
        observability_config=None,
        reasoning_config=None,
        num_speculative_tokens=0,
    )
    values.update(overrides)
    return NS(**values)


@contextmanager
def _local_cpu_patches(mr):
    """Install the local-only CPU construction patches for model_runner.

    Each patch is bound to the module object so restoring is exact; on the
    CI image the guarded branches (pin memory available, CUDA device
    probes, CI InputBatch signature) never trigger the compat wrappers.
    """
    import vllm.utils.torch_utils as tu
    import vllm.v1.utils as vu
    import vllm.v1.worker.gpu_input_batch as gib

    import vllm_fl.worker.common_attention_metadata as cam

    # InputBatch reads PIN_MEMORY at construction time and forwards it to
    # the block table, so patching the constant in its module is enough.
    pin_patches = [_Patch(mod, "PIN_MEMORY", False) for mod in (tu, vu, gib)]

    class _CpuOnlyCGB(mr.CpuGpuBuffer):
        def __init__(self, *args, **kwargs):
            kwargs["pin_memory"] = False
            super().__init__(*args, **kwargs)

    orig_input_batch = mr.InputBatch
    batch_params = inspect.signature(orig_input_batch).parameters

    @wraps(orig_input_batch)
    def _input_batch_compat(*args, **kwargs):
        if "kernel_block_sizes" not in batch_params:
            kwargs.pop("kernel_block_sizes", None)
            if kwargs.get("max_num_blocks_per_req") is None:
                kwargs["max_num_blocks_per_req"] = [
                    (kwargs["max_model_len"] + block - 1) // block
                    for block in kwargs["block_sizes"]
                ]
        return orig_input_batch(*args, **kwargs)

    orig_blp = mr.build_logitsprocs

    def _blp_compat(vllm_config, device, is_pin_memory, is_pooling_model, *a, **kw):
        return orig_blp(vllm_config, device, False, is_pooling_model, *a, **kw)

    class _CamCpuPlatform:
        device_type = "cpu"
        torch_device_fn = SimpleNamespace(graph=None)

        def get_global_graph_pool(self):
            return None

    import torch as _torch_mod

    class _MrCpuPlatform:
        """CPU stand-in for the runner module's platform during construction.

        Only the attributes __init__ reads: streams/events through
        torch_device_fn (inert stand-ins; torch.cuda.Stream cannot be
        constructed on this CPU-only host) and device_type for the
        sampled-id dtype selection. It replaces mr.current_platform for
        the construction window only and is restored afterwards.
        """

        device_type = "cpu"

        class _Stream:
            def wait_stream(self, other):
                pass

        class torch_device_fn:
            Stream = staticmethod(lambda *a, **k: _MrCpuPlatform._Stream())
            Event = staticmethod(_torch_mod.Event)
            current_stream = staticmethod(lambda *a, **k: _MrCpuPlatform._Stream())
            stream = staticmethod(lambda s: _nullcontext())
            graph = None

    saved_cam_platform = cam.current_platform
    saved_mr_platform = mr.current_platform
    saved_is_pin = mr.is_pin_memory_available
    saved_mr_compute_units = mr.num_compute_units
    import vllm.utils.platform_utils as putils

    saved_numb = putils.num_compute_units
    saved_mmr = mr.MULTIMODAL_REGISTRY
    saved_cgb = mr.CpuGpuBuffer

    for p in pin_patches:
        p.__enter__()
    mr.CpuGpuBuffer = _CpuOnlyCGB
    mr.InputBatch = _input_batch_compat
    mr.build_logitsprocs = _blp_compat
    mr.is_pin_memory_available = lambda: False
    cam.current_platform = _CamCpuPlatform()
    mr.current_platform = _MrCpuPlatform()
    mr.MULTIMODAL_REGISTRY = SimpleNamespace(
        supports_multimodal_inputs=lambda mc: False
    )
    try:
        # num_compute_units is lru-cached; clear so the pinned value of a
        # prior runner build does not leak back through the module global.
        putils.num_compute_units.cache_clear()
        try:
            mr.num_compute_units = lambda idx: 8
            yield
        finally:
            putils.num_compute_units.cache_clear()
    finally:
        mr.MULTIMODAL_REGISTRY = saved_mmr
        cam.current_platform = saved_cam_platform
        mr.current_platform = saved_mr_platform
        mr.is_pin_memory_available = saved_is_pin
        mr.num_compute_units = saved_mr_compute_units
        mr.build_logitsprocs = orig_blp
        mr.InputBatch = orig_input_batch
        mr.CpuGpuBuffer = saved_cgb
        for p in reversed(pin_patches):
            p.__exit__(None, None, None)
        putils.num_compute_units = saved_numb


@contextmanager
def _runner_patches(mr):
    defaults = {
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
        "VLLM_FL_COMMON_ATTENTION_METADATA": "1",
    }
    env = {key: os.environ.get(key, value) for key, value in defaults.items()}
    with patch.dict(os.environ, env), _local_cpu_patches(mr):
        yield


def make_scheduler_output(**kwargs):
    """Build real scheduler output for a step without shared prefix blocks."""
    from vllm.v1.core.sched.output import SchedulerOutput

    if "num_common_prefix_blocks" in inspect.signature(SchedulerOutput).parameters:
        kwargs.setdefault("num_common_prefix_blocks", [0])
    return SchedulerOutput(**kwargs)


def make_runner(vllm_config=None):
    """Construct a real ModelRunnerFL on CPU; return (runner, patches-cm).

    The returned context manager keeps the local patches active so the
    test can exercise buffer/GPU-copy paths; exiting it restores every
    module attribute touched.
    """
    import vllm_fl.worker.model_runner as mr

    config = vllm_config or _vllm_config()
    with ExitStack() as patches:
        patches.enter_context(single_rank_gloo("tp", "pp", "dcp"))
        patches.enter_context(_runner_patches(mr))
        runner = mr.ModelRunnerFL(vllm_config=config, device=torch.device("cpu"))
        return runner, patches.pop_all()
