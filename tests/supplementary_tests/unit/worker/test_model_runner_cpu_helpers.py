# Copyright (c) 2026 BAAI. All rights reserved.

"""Supplementary CPU tests for the light helpers of ModelRunnerFL.

``vllm_fl.worker.model_runner`` imports ``vllm_fl.compilation.graph`` whose
``Graph`` class raises ``NotImplementedError`` at class-body level unless the
current platform's ``device_type`` is cuda/npu/musa/gcu/ptpu/txda.  Every test
here therefore installs a delegating proxy on ``vllm.platforms.current_platform``
that fakes only the CUDA-ness attributes and forwards everything else to the
real platform, before the first import of the module under test.

Methods are driven through unbound invocation
(``ModelRunnerFL.method(runner, ...)``) on hand-built ``SimpleNamespace``
runners, mirroring the pattern of the core suite.  Only external boundaries
(torch stream/event APIs, GPU-only torch calls) are replaced; the logic under
test runs unmodified and every numeric expectation is computed with an
independent plain-Python/torch oracle.
"""

from contextlib import nullcontext
from importlib import import_module
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

try:  # pragma: no cover - dual-environment guard
    import vllm  # noqa: F401

    HAS_VLLM = True
except ImportError:  # pragma: no cover - Windows machines without vllm
    HAS_VLLM = False

pytestmark = [
    pytest.mark.supplementary,
    pytest.mark.skipif(not HAS_VLLM, reason="requires real vllm"),
]


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


# ---------------------------------------------------------------------------
# CPU-safe fakes for the torch stream/event boundary (CPU-only builds cannot
# instantiate torch.cuda.Stream / torch.cuda.Event).
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

    def wait_stream(self, other):
        self.waited.append(other)


def _fake_torch_device_fn():
    """A torch_device_fn namespace backed by the CPU-safe fakes."""
    return SimpleNamespace(
        Event=_FakeEvent,
        Stream=_FakeStream,
        current_stream=lambda: "default-stream",
        stream=lambda s: nullcontext(),
    )


# ---------------------------------------------------------------------------
# Module-level helpers.
# ---------------------------------------------------------------------------


class TestAcceleratorSynchronize:
    """_accelerator_synchronize dispatches on the platform device type."""

    def test_musa_platform_calls_torch_musa(self, model_runner_module, monkeypatch):
        """A musa device_type synchronizes through the torch_musa module."""
        import sys
        import types as types_mod

        module = model_runner_module
        calls = []
        fake_musa = types_mod.ModuleType("torch_musa")
        fake_musa.synchronize = lambda: calls.append("musa")
        monkeypatch.setitem(sys.modules, "torch_musa", fake_musa)
        # The module binds its own current_platform at import time, so a
        # musa-typed stand-in is swapped in wholesale (device_type is a
        # read-only property on the real platform classes).
        monkeypatch.setattr(
            module, "current_platform", SimpleNamespace(device_type="musa")
        )

        module._accelerator_synchronize()
        assert calls == ["musa"]

    def test_other_platforms_use_torch_accelerator(
        self, model_runner_module, monkeypatch
    ):
        """Every other device type goes through torch.accelerator.synchronize."""
        import torch

        module = model_runner_module
        calls = []
        monkeypatch.setattr(
            module, "current_platform", SimpleNamespace(device_type="cuda")
        )
        monkeypatch.setattr(
            torch.accelerator, "synchronize", lambda: calls.append("acc")
        )
        module._accelerator_synchronize()
        assert calls == ["acc"]


class TestDeviceGraphCapture:
    """_device_graph_capture creates/enters a stream via the platform API."""

    def test_creates_stream_when_no_context_given(
        self, model_runner_module, monkeypatch
    ):
        """Without a caller-supplied context a new Stream is built per device."""
        module = model_runner_module
        capture_stream = _FakeStream()
        created = []

        def make_stream(device=None):
            created.append(device)
            return capture_stream

        device_api = SimpleNamespace(
            Stream=make_stream,
            current_stream=lambda: "curr",
            stream=lambda s: nullcontext(),
        )
        monkeypatch.setattr(
            module, "current_platform", SimpleNamespace(torch_device_fn=device_api)
        )

        device = object()
        with module._device_graph_capture(device) as ctx:
            assert ctx.stream is capture_stream
        # The Stream constructor saw exactly the requested device and the
        # capture stream waited on the current stream before being entered.
        assert created == [device]
        assert capture_stream.waited == ["curr"]

    def test_reuses_supplied_context(self, model_runner_module, monkeypatch):
        """A supplied GraphCaptureContext is entered without a new Stream."""
        from vllm.distributed.parallel_state import GraphCaptureContext

        module = model_runner_module
        capture_stream = _FakeStream()
        supplied = GraphCaptureContext(capture_stream)
        made = []
        device_api = SimpleNamespace(
            Stream=lambda device=None: made.append("made"),
            current_stream=lambda: "curr",
            stream=lambda s: nullcontext(),
        )
        monkeypatch.setattr(
            module, "current_platform", SimpleNamespace(torch_device_fn=device_api)
        )

        with module._device_graph_capture(
            object(), graph_capture_context=supplied
        ) as ctx:
            assert ctx is supplied
        assert made == []
        assert capture_stream.waited == ["curr"]


class TestGetParameterForReload:
    """_get_parameter_for_reload resolves checkpoint names onto modules."""

    def test_resolves_parameter_on_plain_module(self, model_runner_module):
        """A dotted name resolves through get_submodule + get_parameter."""
        import torch
        import torch.nn as nn

        module = model_runner_module

        class Leaf(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(3))

        class Root(nn.Module):
            def __init__(self):
                super().__init__()
                self.leaf = Leaf()

        root = Root()
        param = module._get_parameter_for_reload(root, "leaf.weight")
        assert param is root.leaf.weight

    def test_unwraps_lora_base_layer(self, model_runner_module):
        """A BaseLayerWithLoRA module is unwrapped before parameter lookup."""
        import torch
        import torch.nn as nn

        module = model_runner_module

        class Leaf(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(2))

        class FakeLora(module.BaseLayerWithLoRA):
            def __init__(self, base):
                nn.Module.__init__(self)
                self.base_layer = base

        class Root(nn.Module):
            def __init__(self, lora):
                super().__init__()
                self.wrapped = lora

        leaf = Leaf()
        root = Root(FakeLora(leaf))
        param = module._get_parameter_for_reload(root, "wrapped.weight")
        # The checkpoint name addresses the base layer's parameter even
        # though the module tree holds it inside a LoRA wrapper.
        assert param is leaf.weight

    def test_top_level_name_uses_whole_model(self, model_runner_module):
        """A parameter without a dot resolves on the model root itself."""
        import torch
        import torch.nn as nn

        module = model_runner_module

        class Root(nn.Module):
            def __init__(self):
                super().__init__()
                self.bias = nn.Parameter(torch.zeros(5))

        root = Root()
        assert module._get_parameter_for_reload(root, "bias") is root.bias


# ---------------------------------------------------------------------------
# AsyncGPUModelRunnerOutput / AsyncGPUPoolingModelRunnerOutput
# ---------------------------------------------------------------------------


class TestAsyncGPUModelRunnerOutputGetOutput:
    """AsyncGPUModelRunnerOutput.get_output host-copy semantics."""

    def _make(self, module, output, sampled, invalid, logprobs=None, num_nans=None):
        """Build the async wrapper with the CPU-safe stream/event fakes."""
        return module.AsyncGPUModelRunnerOutput(
            model_runner_output=output,
            sampled_token_ids=sampled,
            logprobs_tensors=logprobs,
            invalid_req_indices=invalid,
            async_output_copy_stream=_FakeStream(),
            vocab_size=100,
            num_nans=num_nans,
        )

    def test_single_token_step_drops_invalid_rows(
        self, model_runner_module, monkeypatch
    ):
        """max_gen_len == 1 clears the sampled list of invalid requests."""
        import torch

        from vllm.v1.outputs import ModelRunnerOutput

        module = model_runner_module
        monkeypatch.setattr(
            module,
            "current_platform",
            SimpleNamespace(torch_device_fn=_fake_torch_device_fn()),
        )
        output = ModelRunnerOutput(
            req_ids=["r0", "r1"], req_id_to_index={"r0": 0, "r1": 1}
        )
        out = self._make(module, output, torch.tensor([[5], [7]]), invalid=[1])
        got = out.get_output()
        # Row 1 was invalidated, so its sampled list is emptied; row 0
        # survives untouched.
        assert got.sampled_token_ids == [[5], []]
        assert got.logprobs is None

    def test_single_token_step_parses_logprobs(self, model_runner_module, monkeypatch):
        """max_gen_len == 1 converts CPU logprob tensors to LogprobsLists."""
        import torch

        from vllm.v1.outputs import LogprobsTensors, ModelRunnerOutput

        module = model_runner_module
        monkeypatch.setattr(
            module,
            "current_platform",
            SimpleNamespace(torch_device_fn=_fake_torch_device_fn()),
        )
        lp = LogprobsTensors(
            logprob_token_ids=torch.tensor([[11], [13]]),
            logprobs=torch.tensor([[0.1], [0.3]]),
            selected_token_ranks=torch.tensor([0, 1]),
            cu_num_generated_tokens=[0, 1, 2],
        )
        output = ModelRunnerOutput(
            req_ids=["r0", "r1"], req_id_to_index={"r0": 0, "r1": 1}
        )
        out = self._make(
            module, output, torch.tensor([[5], [7]]), invalid=[], logprobs=lp
        )
        got = out.get_output()
        # Independent oracle: the per-request rows keep their token ids.
        assert got.logprobs.logprob_token_ids.tolist() == [[11], [13]]

    def test_multi_token_step_uses_rejection_sampler(
        self, model_runner_module, monkeypatch
    ):
        """max_gen_len > 1 filters placeholder tokens via parse_output."""
        import torch

        from vllm.v1.outputs import ModelRunnerOutput

        module = model_runner_module
        monkeypatch.setattr(
            module,
            "current_platform",
            SimpleNamespace(torch_device_fn=_fake_torch_device_fn()),
        )
        output = ModelRunnerOutput(req_ids=["r0"], req_id_to_index={"r0": 0})
        # Placeholder (-1) tokens are rejected drafts and must disappear.
        out = self._make(module, output, torch.tensor([[3, 4, -1, -1]]), invalid=[])
        got = out.get_output()
        assert got.sampled_token_ids == [[3, 4]]

    def test_multi_token_step_filters_logprobs(self, model_runner_module, monkeypatch):
        """max_gen_len > 1 keeps logprobs only for accepted tokens."""
        import torch

        from vllm.v1.outputs import LogprobsTensors, ModelRunnerOutput

        module = model_runner_module
        monkeypatch.setattr(
            module,
            "current_platform",
            SimpleNamespace(torch_device_fn=_fake_torch_device_fn()),
        )
        lp = LogprobsTensors(
            logprob_token_ids=torch.tensor([[1], [2], [3], [4]]),
            logprobs=torch.tensor([[0.5], [0.6], [0.7], [0.8]]),
            selected_token_ranks=torch.tensor([0, 0, 0, 0]),
        )
        output = ModelRunnerOutput(req_ids=["r0"], req_id_to_index={"r0": 0})
        out = self._make(
            module,
            output,
            torch.tensor([[3, 4, -1, -1]]),
            invalid=[],
            logprobs=lp,
        )
        got = out.get_output()
        # Independent oracle: rejected positions (-1) drop their logprob rows.
        assert got.logprobs.logprob_token_ids.reshape(-1).tolist() == [1, 2]

    def test_num_nans_reported_per_request(self, model_runner_module, monkeypatch):
        """num_nans lands in num_nans_in_logits keyed by request id."""
        import torch

        from vllm.v1.outputs import ModelRunnerOutput

        module = model_runner_module
        monkeypatch.setattr(
            module,
            "current_platform",
            SimpleNamespace(torch_device_fn=_fake_torch_device_fn()),
        )
        output = ModelRunnerOutput(req_ids=["r0"], req_id_to_index={"r0": 0})
        out = self._make(
            module, output, torch.tensor([[9]]), invalid=[], num_nans=torch.tensor([2])
        )
        got = out.get_output()
        assert got.num_nans_in_logits == {"r0": 2}

    def test_nan_logits_raise_when_env_set(self, model_runner_module, monkeypatch):
        """VLLM_RAISE_ON_LOGIT_NANS turns any NaN count into a RuntimeError."""
        import torch

        from vllm.v1.outputs import ModelRunnerOutput

        module = model_runner_module
        monkeypatch.setattr(
            module,
            "current_platform",
            SimpleNamespace(torch_device_fn=_fake_torch_device_fn()),
        )
        monkeypatch.setattr(module.envs, "VLLM_RAISE_ON_LOGIT_NANS", True)
        output = ModelRunnerOutput(req_ids=["r0"], req_id_to_index={"r0": 0})
        out = self._make(
            module, output, torch.tensor([[9]]), invalid=[], num_nans=torch.tensor([1])
        )
        with pytest.raises(RuntimeError, match="NaNs detected"):
            out.get_output()

    def test_ep_fault_raises_after_sync(self, model_runner_module, monkeypatch):
        """A queried EP all2all fault surfaces as RuntimeError with the mask."""
        import torch

        from vllm.v1.outputs import ModelRunnerOutput

        module = model_runner_module
        monkeypatch.setattr(
            module,
            "current_platform",
            SimpleNamespace(torch_device_fn=_fake_torch_device_fn()),
        )
        manager = SimpleNamespace(
            query_fault=lambda: torch.tensor(True),
            query_active_mask=lambda: torch.tensor([0, 1, 0]),
        )
        monkeypatch.setattr(module, "get_ep_all2all_manager", lambda: manager)
        output = ModelRunnerOutput(req_ids=["r0"], req_id_to_index={"r0": 0})
        out = module.AsyncGPUModelRunnerOutput(
            model_runner_output=output,
            sampled_token_ids=torch.tensor([[7]]),
            logprobs_tensors=None,
            invalid_req_indices=[],
            async_output_copy_stream=_FakeStream(),
            vocab_size=100,
            check_ep_fault=True,
        )
        assert out._has_fault.item() is True
        with pytest.raises(RuntimeError, match="Fault detected in EP all2all"):
            out.get_output()


class TestAsyncGPUPoolingModelRunnerOutput:
    """AsyncGPUPoolingModelRunnerOutput.get_output host-copy semantics."""

    def test_returns_model_runner_output_after_sync(
        self, model_runner_module, monkeypatch
    ):
        """get_output synchronizes the copy and returns the stored output."""
        import torch

        from vllm.v1.outputs import ModelRunnerOutput

        module = model_runner_module
        monkeypatch.setattr(
            module,
            "current_platform",
            SimpleNamespace(torch_device_fn=_fake_torch_device_fn()),
        )
        output = ModelRunnerOutput(req_ids=["r0"], req_id_to_index={"r0": 0})
        out = module.AsyncGPUPoolingModelRunnerOutput(
            model_runner_output=output,
            raw_pooler_output=torch.ones(2, 2),
            finished_mask=[True, False],
            async_output_copy_stream=_FakeStream(),
        )
        got = out.get_output()
        assert got is output
        assert out.async_copy_ready_event.synchronized is True
        # The pooler output was copied through _copy_pooler_output_to_cpu:
        # row 0 kept, row 1 not finished -> None.
        assert len(output.pooler_output) == 2
        torch.testing.assert_close(output.pooler_output[0], torch.ones(2))
        assert output.pooler_output[1] is None


# ---------------------------------------------------------------------------
# _copy_pooler_output_to_cpu
# ---------------------------------------------------------------------------


class TestCopyPoolerOutputToCpu:
    """_copy_pooler_output_to_cpu tensor/list dispatch and masks."""

    def test_tensor_all_finished_copies_every_row(self, model_runner_module):
        """A tensor with every request finished is moved to CPU row-wise."""
        import torch

        module = model_runner_module
        raw = torch.arange(12, dtype=torch.float32).reshape(3, 4) + 0.5
        out = module._copy_pooler_output_to_cpu(raw, [True, True, True])
        assert len(out) == 3
        for i in range(3):
            torch.testing.assert_close(out[i], raw[i])

    def test_tensor_none_finished_returns_all_none(self, model_runner_module):
        """No finished request yields an all-None list of num_reqs entries."""
        import torch

        module = model_runner_module
        out = module._copy_pooler_output_to_cpu(torch.zeros(2, 4), [False, False])
        assert out == [None, None]

    def test_tensor_partial_finished_selects_rows(self, model_runner_module):
        """Only finished rows are returned, at their original positions."""
        import torch

        module = model_runner_module
        raw = torch.arange(8, dtype=torch.float32).reshape(4, 2)
        out = module._copy_pooler_output_to_cpu(raw, [False, True, False, True])
        # Independent oracle: rows 1 and 3 are the finished ones.
        assert out[0] is None and out[2] is None
        torch.testing.assert_close(out[1], raw[1])
        torch.testing.assert_close(out[3], raw[3])

    def test_tensor_size_mismatch_raises(self, model_runner_module):
        """A tensor whose batch size differs from the mask is rejected."""
        import torch

        module = model_runner_module
        with pytest.raises(ValueError, match="batch size does not match"):
            module._copy_pooler_output_to_cpu(torch.zeros(2, 4), [True])

    def test_list_size_mismatch_raises(self, model_runner_module):
        """A list whose batch size differs from the mask is rejected."""
        import torch

        module = model_runner_module
        with pytest.raises(ValueError, match="batch size does not match"):
            module._copy_pooler_output_to_cpu([torch.ones(2)], [True, True])

    def test_list_keeps_finished_entries(self, model_runner_module):
        """List input keeps only finished, non-None entries per request."""
        import torch

        module = model_runner_module
        rows = [torch.full((2,), float(i)) for i in range(3)]
        out = module._copy_pooler_output_to_cpu(rows, [True, False, True])
        torch.testing.assert_close(out[0], rows[0])
        assert out[1] is None
        torch.testing.assert_close(out[2], rows[2])

    def test_list_skips_none_data_for_finished_request(self, model_runner_module):
        """A finished request whose pooled output is None stays None."""
        import torch

        module = model_runner_module
        rows = [torch.ones(2), None]
        out = module._copy_pooler_output_to_cpu(rows, [True, True])
        torch.testing.assert_close(out[0], rows[0])
        assert out[1] is None


# ---------------------------------------------------------------------------
# ModelRunnerFL light methods driven through unbound invocation.
# ---------------------------------------------------------------------------


class TestUpdateMaxModelLen:
    """update_max_model_len propagates the new length to the drafter bound."""

    def test_without_speculative_config(self, model_runner_module):
        """No speculative config: only max_model_len is updated."""
        module = model_runner_module
        runner = SimpleNamespace(max_model_len=128, speculative_config=None)
        module.ModelRunnerFL.update_max_model_len(runner, 256)
        assert runner.max_model_len == 256

    def test_draft_config_without_len_inherits(self, model_runner_module):
        """A drafter without an explicit length inherits the new length."""
        module = model_runner_module
        runner = SimpleNamespace(
            max_model_len=128,
            speculative_config=SimpleNamespace(
                draft_model_config=SimpleNamespace(max_model_len=None)
            ),
            effective_drafter_max_model_len=None,
        )
        module.ModelRunnerFL.update_max_model_len(runner, 512)
        assert runner.max_model_len == 512
        assert runner.effective_drafter_max_model_len == 512

    def test_no_draft_model_config_inherits(self, model_runner_module):
        """speculative_config present but no draft config also inherits."""
        module = model_runner_module
        runner = SimpleNamespace(
            max_model_len=128,
            speculative_config=SimpleNamespace(draft_model_config=None),
            effective_drafter_max_model_len=None,
        )
        module.ModelRunnerFL.update_max_model_len(runner, 300)
        assert runner.effective_drafter_max_model_len == 300

    def test_draft_model_config_with_explicit_len_wins(self, model_runner_module):
        """A drafter carrying its own length keeps it (early branch)."""
        module = model_runner_module
        runner = SimpleNamespace(
            max_model_len=128,
            speculative_config=SimpleNamespace(
                draft_model_config=SimpleNamespace(max_model_len=64)
            ),
            effective_drafter_max_model_len=None,
        )
        module.ModelRunnerFL.update_max_model_len(runner, 512)
        assert runner.max_model_len == 512
        assert runner.effective_drafter_max_model_len is None


class TestInitFp8KvScales:
    """init_fp8_kv_scales zeroes caches and resets attention scales."""

    def _attn(self, module, k_attr, v_attr):
        """Build an Attention subclass holding scale tensors."""
        import torch
        import torch.nn as nn

        class FakeAttn(module.Attention):
            def __init__(self):
                nn.Module.__init__(self)
                setattr(self, k_attr, torch.tensor(0.0))
                setattr(self, v_attr, torch.tensor(0.0))

        return FakeAttn()

    def _runner(self, module, cache_dtype, kv_caches, context):
        return SimpleNamespace(
            cache_config=SimpleNamespace(cache_dtype=cache_dtype),
            kv_caches=kv_caches,
            compilation_config=SimpleNamespace(static_forward_context=context),
        )

    def test_skips_unquantized_cache(self, model_runner_module):
        """A non-fp8 cache dtype returns without touching any state."""
        import torch

        module = model_runner_module
        tensor_cache = torch.ones(2, 3)
        list_cache = [torch.ones(2), torch.ones(4)]
        attn = self._attn(module, "_k_scale", "_v_scale")
        runner = self._runner(
            module, "auto", [tensor_cache, list_cache, None], {"attn": attn}
        )
        module.ModelRunnerFL.init_fp8_kv_scales(runner)
        # Nothing was zeroed or reset.
        assert torch.equal(tensor_cache, torch.ones(2, 3))
        assert torch.equal(list_cache[0], torch.ones(2))
        assert attn._k_scale.item() == 0.0

    def test_zeroes_tensor_and_list_caches(self, model_runner_module):
        """Tensor and per-layer-list caches are zeroed in place."""
        import torch

        module = model_runner_module
        tensor_cache = torch.ones(2, 3)
        list_cache = [torch.ones(2), torch.ones(4)]
        attn = self._attn(module, "_k_scale", "_v_scale")
        runner = self._runner(
            module, "fp8", [tensor_cache, list_cache, None], {"attn": attn}
        )
        module.ModelRunnerFL.init_fp8_kv_scales(runner)
        assert torch.equal(tensor_cache, torch.zeros(2, 3))
        for t in list_cache:
            assert torch.equal(t, torch.zeros_like(t))
        # Scales are reset to 1.0 on both attribute spellings.
        assert attn._k_scale.item() == 1.0
        assert attn._v_scale.item() == 1.0

    def test_resets_k_scale_without_underscore(self, model_runner_module):
        """The k_scale/v_scale spellings (MLA) are reset as well."""
        module = model_runner_module
        attn = self._attn(module, "k_scale", "v_scale")
        runner = self._runner(module, "fp8", [], {"mla": attn})
        module.ModelRunnerFL.init_fp8_kv_scales(runner)
        assert attn.k_scale.item() == 1.0
        assert attn.v_scale.item() == 1.0

    def test_ignores_non_tensor_and_foreign_modules(self, model_runner_module):
        """Non-tensor scale attrs and non-attention modules are untouched."""
        import torch.nn as nn

        module = model_runner_module

        class NotAttention(nn.Module):
            def __init__(self):
                super().__init__()
                self._k_scale = "sentinel"

        foreign = NotAttention()
        attn = self._attn(module, "_k_scale", "_v_scale")
        runner = self._runner(module, "fp8_e4m3", [], {"other": foreign, "attn": attn})
        module.ModelRunnerFL.init_fp8_kv_scales(runner)
        assert foreign._k_scale == "sentinel"
        assert attn._k_scale.item() == 1.0


class TestGetPositions:
    """_get_positions routes int/tensor indexing by uses_mrope."""

    def _runner(self, module, uses_mrope):
        import torch

        runner = SimpleNamespace(uses_mrope=uses_mrope)
        runner.positions = torch.arange(100)
        runner.mrope_positions = SimpleNamespace(gpu=torch.arange(300).reshape(3, 100))
        return runner

    def test_int_without_mrope_slices_positions(self, model_runner_module):
        """Plain RoPE int path: first N positions."""
        import torch

        module = model_runner_module
        runner = self._runner(module, uses_mrope=False)
        out = module.ModelRunnerFL._get_positions(runner, 7)
        torch.testing.assert_close(out, torch.arange(7))

    def test_int_with_mrope_slices_3d(self, model_runner_module):
        """MRoPE int path: gpu[:, :N] keeps the 3D layout."""
        import torch

        module = model_runner_module
        runner = self._runner(module, uses_mrope=True)
        out = module.ModelRunnerFL._get_positions(runner, 4)
        expected = runner.mrope_positions.gpu[:, :4]
        assert out.shape == (3, 4)
        torch.testing.assert_close(out, expected)

    def test_tensor_without_mrope_gathers(self, model_runner_module):
        """Plain RoPE tensor path: positions[indices]."""
        import torch

        module = model_runner_module
        runner = self._runner(module, uses_mrope=False)
        idx = torch.tensor([3, 0, 9])
        out = module.ModelRunnerFL._get_positions(runner, idx)
        torch.testing.assert_close(out, runner.positions[idx])

    def test_tensor_with_mrope_gathers_3d(self, model_runner_module):
        """MRoPE tensor path: gpu[:, indices]."""
        import torch

        module = model_runner_module
        runner = self._runner(module, uses_mrope=True)
        idx = torch.tensor([5, 2])
        out = module.ModelRunnerFL._get_positions(runner, idx)
        expected = runner.mrope_positions.gpu[:, idx]
        assert out.shape == (3, 2)
        torch.testing.assert_close(out, expected)


class TestJitWarmupRegistryProperty:
    """jit_warmup_registry lazily creates and caches the registry."""

    def _runner(self):
        runner = SimpleNamespace()
        runner.__dict__["_jit_warmup_registry"] = None
        runner.vllm_config = "cfg-sentinel"
        return runner

    def test_getter_builds_registry_once(self, model_runner_module):
        """First access builds a JitWarmupRegistry, second returns it."""
        module = model_runner_module
        runner = self._runner()
        prop = module.ModelRunnerFL.__dict__["jit_warmup_registry"]
        reg1 = prop.fget(runner)
        assert isinstance(reg1, module.JitWarmupRegistry)
        assert reg1.vllm_config == "cfg-sentinel"
        reg2 = prop.fget(runner)
        assert reg2 is reg1
        assert runner.__dict__["_jit_warmup_registry"] is reg1

    def test_setter_round_trip(self, model_runner_module):
        """The setter stores any registry object in __dict__."""
        module = model_runner_module
        runner = self._runner()
        prop = module.ModelRunnerFL.__dict__["jit_warmup_registry"]
        sentinel = object()
        prop.fset(runner, sentinel)
        assert runner.__dict__["_jit_warmup_registry"] is sentinel
        assert prop.fget(runner) is sentinel


class TestGetOrCreateAsyncOutputCopyStream:
    """_get_or_create_async_output_copy_stream lazily builds the stream."""

    def test_creates_once_and_caches(self, model_runner_module, monkeypatch):
        """The stream is created through the platform API, then reused."""
        module = model_runner_module
        made = []
        device_api = SimpleNamespace(Stream=lambda: made.append("new") or object())
        monkeypatch.setattr(
            module, "current_platform", SimpleNamespace(torch_device_fn=device_api)
        )
        runner = SimpleNamespace(async_output_copy_stream=None)
        s1 = module.ModelRunnerFL._get_or_create_async_output_copy_stream(runner)
        assert made == ["new"]
        assert runner.async_output_copy_stream is s1
        s2 = module.ModelRunnerFL._get_or_create_async_output_copy_stream(runner)
        assert s2 is s1
        assert made == ["new"]


class TestMayReorderBatch:
    """_may_reorder_batch gates reordering on groups and threshold."""

    def test_no_kv_cache_groups_returns_early(self, model_runner_module, monkeypatch):
        """Attention-free models with zero groups skip reordering."""
        module = model_runner_module
        calls = []
        monkeypatch.setattr(
            module,
            "reorder_batch_to_split_decodes_and_prefills",
            lambda *a, **k: calls.append("reorder"),
        )
        runner = SimpleNamespace(
            kv_cache_config=SimpleNamespace(kv_cache_groups=[]),
            reorder_batch_threshold=8,
            input_batch=object(),
        )
        module.ModelRunnerFL._may_reorder_batch(runner, SimpleNamespace())
        assert calls == []

    def test_no_threshold_skips_reorder(self, model_runner_module, monkeypatch):
        """A None threshold never reorders the batch."""
        module = model_runner_module
        calls = []
        monkeypatch.setattr(
            module,
            "reorder_batch_to_split_decodes_and_prefills",
            lambda *a, **k: calls.append((a, k)),
        )
        runner = SimpleNamespace(
            kv_cache_config=SimpleNamespace(kv_cache_groups=[object()]),
            reorder_batch_threshold=None,
            input_batch="input-batch",
        )
        module.ModelRunnerFL._may_reorder_batch(runner, SimpleNamespace())
        assert calls == []

    def test_threshold_triggers_reorder_with_args(
        self, model_runner_module, monkeypatch
    ):
        """With groups and a threshold, reorder runs with the threshold."""
        module = model_runner_module
        calls = []
        monkeypatch.setattr(
            module,
            "reorder_batch_to_split_decodes_and_prefills",
            lambda *a, **k: calls.append((a, k)),
        )
        runner = SimpleNamespace(
            kv_cache_config=SimpleNamespace(kv_cache_groups=[object()]),
            reorder_batch_threshold=4,
            input_batch="input-batch",
        )
        sched = SimpleNamespace()
        module.ModelRunnerFL._may_reorder_batch(runner, sched)
        assert len(calls) == 1
        args, kwargs = calls[0]
        assert args[0] == "input-batch"
        assert args[1] is sched
        assert kwargs["decode_threshold"] == 4


class TestGetModelAndGetDraftModel:
    """get_model / get_draft_model unwrap graph wrappers."""

    def _wrapped_graph(self, module, inner):
        """A real GraphWrapper instance whose runnable is ``inner``."""
        wrapped = object.__new__(module.GraphWrapper)
        wrapped.runnable = inner
        return wrapped

    def test_missing_model_raises(self, model_runner_module):
        """get_model refuses before the model is initialized."""
        module = model_runner_module
        runner = SimpleNamespace()
        with pytest.raises(ValueError, match="Cannot get model"):
            module.ModelRunnerFL.get_model(runner)

    def test_plain_model_returned_as_is(self, model_runner_module):
        """An unwrapped model object is returned directly."""
        module = model_runner_module
        model = object()
        runner = SimpleNamespace(model=model)
        assert module.ModelRunnerFL.get_model(runner) is model

    def test_graph_wrapper_unwrapped(self, model_runner_module):
        """A GraphWrapper yields its runnable through unwrap()."""
        import torch.nn as nn

        module = model_runner_module

        class Dummy(nn.Module):
            def forward(self, x):
                return x

        inner = Dummy()
        runner = SimpleNamespace(model=self._wrapped_graph(module, inner))
        assert module.ModelRunnerFL.get_model(runner) is inner

    def test_nested_wrappers_recurse_to_innermost(self, model_runner_module):
        """CUDAGraphWrapper/Breakable unwrap recurse through nested wrappers."""
        import torch.nn as nn

        module = model_runner_module

        class Dummy(nn.Module):
            def forward(self, x):
                return x

        inner = Dummy()
        graph = self._wrapped_graph(module, inner)
        for wrapper_cls in (module.CUDAGraphWrapper, module.BreakableCUDAGraphWrapper):
            wrapped = object.__new__(wrapper_cls)
            wrapped.runnable = graph
            runner = SimpleNamespace(model=wrapped)
            assert module.ModelRunnerFL.get_model(runner) is inner

    def test_draft_model_missing_returns_none(self, model_runner_module):
        """No drafter attribute means no draft model."""
        module = model_runner_module
        runner = SimpleNamespace()
        assert module.ModelRunnerFL.get_draft_model(runner) is None

    def test_draft_model_plain(self, model_runner_module):
        """A drafter holding a plain model returns that model."""
        module = model_runner_module
        model = object()
        runner = SimpleNamespace(drafter=SimpleNamespace(model=model))
        assert module.ModelRunnerFL.get_draft_model(runner) is model

    def test_draft_model_none_model(self, model_runner_module):
        """A drafter with model=None propagates None."""
        module = model_runner_module
        runner = SimpleNamespace(drafter=SimpleNamespace(model=None))
        assert module.ModelRunnerFL.get_draft_model(runner) is None

    def test_draft_model_unwraps_graph_wrapper(self, model_runner_module):
        """A drafter model hidden in a GraphWrapper is unwrapped."""
        import torch.nn as nn

        module = model_runner_module

        class Dummy(nn.Module):
            def forward(self, x):
                return x

        inner = Dummy()
        runner = SimpleNamespace(
            drafter=SimpleNamespace(model=self._wrapped_graph(module, inner))
        )
        assert module.ModelRunnerFL.get_draft_model(runner) is inner


class TestGetSupportedTasks:
    """get_supported_generation_tasks / get_supported_tasks routing."""

    def _runner(self, module, model):
        """A bare ModelRunnerFL instance carrying just the model."""
        runner = object.__new__(module.ModelRunnerFL)
        runner.model = model
        return runner

    def _gen_model(self, extra=None):
        """A model satisfying the VllmModelForTextGeneration protocol."""
        import torch.nn as nn

        attrs = dict(extra or {})

        class GenModel(nn.Module):
            def __init__(self, vllm_config=None, **kwargs):
                super().__init__()
                for key, value in attrs.items():
                    setattr(self, key, value)

            def embed_input_ids(self, input_ids):
                return input_ids

            def forward(self, input_ids, positions, **kwargs):
                return input_ids

            def compute_logits(self, hidden_states):
                return hidden_states

        return GenModel()

    def _pooling_model(self):
        """A model satisfying the VllmModelForPooling protocol."""
        import torch.nn as nn

        class PoolModel(nn.Module):
            is_pooling_model = True

            def __init__(self, vllm_config=None, **kwargs):
                super().__init__()
                self.pooler = SimpleNamespace(
                    get_supported_tasks=lambda: ["embed", "classify"]
                )

            def embed_input_ids(self, input_ids):
                return input_ids

            def forward(self, input_ids, positions, **kwargs):
                return None

        return PoolModel()

    def test_generate_only_model(self, model_runner_module):
        """A plain generation model reports exactly ['generate']."""
        module = model_runner_module
        runner = self._runner(module, self._gen_model())
        tasks = module.ModelRunnerFL.get_supported_generation_tasks(runner)
        assert tasks == ["generate"]

    def test_transcription_only_model(self, model_runner_module):
        """supports_transcription_only short-circuits to transcription."""
        module = model_runner_module
        model = self._gen_model(
            {"supports_transcription": True, "supports_transcription_only": True}
        )
        runner = self._runner(module, model)
        tasks = module.ModelRunnerFL.get_supported_generation_tasks(runner)
        assert tasks == ["transcription"]

    def test_generate_plus_transcription(self, model_runner_module):
        """A transcribing model that also generates reports both."""
        module = model_runner_module
        model = self._gen_model(
            {"supports_transcription": True, "supports_transcription_only": False}
        )
        runner = self._runner(module, model)
        tasks = module.ModelRunnerFL.get_supported_generation_tasks(runner)
        assert tasks == ["generate", "transcription"]

    def test_generate_plus_realtime(self, model_runner_module):
        """A realtime-capable model appends the realtime task."""
        module = model_runner_module
        model = self._gen_model({"supports_realtime": True})
        runner = self._runner(module, model)
        tasks = module.ModelRunnerFL.get_supported_generation_tasks(runner)
        assert tasks == ["generate", "realtime"]

    def test_get_supported_tasks_generate_runner(self, model_runner_module):
        """runner_type 'generate' extends with generation tasks only."""
        module = model_runner_module
        runner = self._runner(module, self._gen_model())
        runner.model_config = SimpleNamespace(runner_type="generate")
        tasks = module.ModelRunnerFL.get_supported_tasks(runner)
        assert tasks == ("generate",)

    def test_get_supported_tasks_pooling_runner(self, model_runner_module):
        """runner_type 'pooling' extends with pooling tasks only."""
        module = model_runner_module
        runner = self._runner(module, self._pooling_model())
        runner.model_config = SimpleNamespace(runner_type="pooling")
        tasks = module.ModelRunnerFL.get_supported_tasks(runner)
        assert tasks == ("embed", "classify")


class TestSyncAndGatherIntermediateTensors:
    """sync_and_gather_intermediate_tensors PP-handoff semantics."""

    def test_sync_self_copies_and_slices(self, model_runner_module):
        """sync_self=True copies the received tensors, then slices both."""
        import torch

        from vllm.sequence import IntermediateTensors

        module = model_runner_module
        held = IntermediateTensors({"hidden": torch.arange(10, dtype=torch.float32)})
        incoming = IntermediateTensors({"hidden": torch.full((10,), 7.0)})
        runner = SimpleNamespace(intermediate_tensors=held)
        out = module.ModelRunnerFL.sync_and_gather_intermediate_tensors(
            runner, num_tokens=4, intermediate_tensors=incoming, sync_self=True
        )
        # The first 4 slots were overwritten with 7.0 by the sync...
        torch.testing.assert_close(held.tensors["hidden"][:4], torch.full((4,), 7.0))
        # ...and the returned view is exactly the first 4 tokens.
        torch.testing.assert_close(out.tensors["hidden"], torch.full((4,), 7.0))

    def test_without_sync_self_only_slices(self, model_runner_module):
        """sync_self=False returns the held tensors sliced, unchanged."""
        import torch

        from vllm.sequence import IntermediateTensors

        module = model_runner_module
        held = IntermediateTensors({"hidden": torch.arange(8, dtype=torch.float32)})
        runner = SimpleNamespace(intermediate_tensors=held)
        out = module.ModelRunnerFL.sync_and_gather_intermediate_tensors(
            runner, num_tokens=3, intermediate_tensors=None, sync_self=False
        )
        torch.testing.assert_close(
            out.tensors["hidden"], torch.arange(3, dtype=torch.float32)
        )


class TestEplbStep:
    """eplb_step gating on enable_eplb and suppression."""

    def _runner(self, module, enable=True, suppressed=False, state=None):
        return SimpleNamespace(
            parallel_config=SimpleNamespace(
                enable_eplb=enable,
                eplb_config=SimpleNamespace(log_balancedness=True),
            ),
            eep_eplb_suppressed=suppressed,
            eplb_state=state,
            _moe_model=object(),
        )

    def test_disabled_eplb_returns_early(self, model_runner_module):
        """enable_eplb=False never touches the state."""
        module = model_runner_module
        state = SimpleNamespace(step=lambda *a, **k: pytest.fail("step must not run"))
        runner = self._runner(module, enable=False, state=state)
        module.ModelRunnerFL.eplb_step(runner)
        module.ModelRunnerFL.eplb_step(runner, is_dummy=True, is_profile=True)

    def test_suppressed_eplb_returns_early(self, model_runner_module):
        """eep_eplb_suppressed=True skips the step as well."""
        module = model_runner_module
        state = SimpleNamespace(step=lambda *a, **k: pytest.fail("step must not run"))
        runner = self._runner(module, suppressed=True, state=state)
        module.ModelRunnerFL.eplb_step(runner)

    def test_step_called_with_flags(self, model_runner_module):
        """The EplbState step receives the dummy/profile flags and stats."""
        module = model_runner_module
        calls = []
        state = SimpleNamespace(step=lambda *a, **k: calls.append((a, k)))
        runner = self._runner(module, state=state)
        module.ModelRunnerFL.eplb_step(runner, is_dummy=True, is_profile=False)
        assert calls == [((True, False), {"log_stats": True})]
        module.ModelRunnerFL.eplb_step(runner)
        assert calls[-1] == ((False, False), {"log_stats": True})


class TestPrepareKvSharingFastPrefill:
    """_prepare_kv_sharing_fast_prefill buffer fill + dispatch padding."""

    def test_pads_leftover_indices_with_last(self, model_runner_module):
        """Leftover indices past num_logits are filled with the last index."""
        import torch

        module = model_runner_module
        buffer = torch.full((8,), -99, dtype=torch.int64)
        dispatched = []

        def dispatch(num_logits, invalid_modes=None):
            dispatched.append((num_logits, invalid_modes))
            # Pad to the next multiple of 4 (simulated batch descriptor).
            padded = (num_logits + 3) // 4 * 4
            return None, SimpleNamespace(num_tokens=padded)

        runner = SimpleNamespace(
            kv_sharing_fast_prefill_logits_indices=buffer,
            cudagraph_dispatcher=SimpleNamespace(dispatch=dispatch),
        )
        logits_indices = torch.tensor([10, 11, 12])
        out = module.ModelRunnerFL._prepare_kv_sharing_fast_prefill(
            runner, logits_indices
        )
        # Buffer head holds the indices, tail repeats the last one.
        assert buffer[:3].tolist() == [10, 11, 12]
        assert buffer[3:].tolist() == [12, 12, 12, 12, 12]
        # Returned slice covers the padded token count.
        assert out.shape[0] == 4
        assert out.tolist() == [10, 11, 12, 12]
        assert dispatched == [(3, {module.CUDAGraphMode.FULL})]


class TestComputePrevPositions:
    """_compute_prev_positions builds the current->previous position map."""

    def test_no_prev_mapping_fills_minus_one(self, model_runner_module):
        """An empty prev mapping fills every slot with -1."""
        import numpy as np

        module = model_runner_module
        arr = np.zeros(3, dtype=np.int64)
        runner = SimpleNamespace(
            input_batch=SimpleNamespace(
                prev_req_id_to_index={}, req_ids=["a", "b", "c"]
            ),
            prev_positions=SimpleNamespace(np=arr),
        )
        module.ModelRunnerFL._compute_prev_positions(runner, 3)
        assert arr.tolist() == [-1, -1, -1]

    def test_maps_request_ids_to_prev_indices(self, model_runner_module):
        """Known ids map to their previous index; unknown ids to -1."""
        import numpy as np

        module = model_runner_module
        arr = np.zeros(3, dtype=np.int64)
        runner = SimpleNamespace(
            input_batch=SimpleNamespace(
                prev_req_id_to_index={"a": 7, "c": 2},
                req_ids=["a", "zzz", "c"],
            ),
            prev_positions=SimpleNamespace(np=arr),
        )
        module.ModelRunnerFL._compute_prev_positions(runner, 3)
        assert arr.tolist() == [7, -1, 2]


class TestGetCumsumAndArange:
    """_get_cumsum_and_arange vectorized batched arange."""

    def _runner(self, module):
        import numpy as np

        return SimpleNamespace(arange_np=np.arange(64, dtype=np.int64))

    @pytest.mark.parametrize(
        "num_tokens",
        [[2, 5, 3], [1, 1, 1, 1], [7], [4, 0, 3]],
    )
    def test_matches_concatenated_arange_oracle(self, model_runner_module, num_tokens):
        """Output equals concatenate(arange(n) for n in num_tokens)."""
        import numpy as np

        module = model_runner_module
        runner = self._runner(module)
        num_tokens = np.array(num_tokens)
        total = int(num_tokens.sum())
        out = np.zeros(total, dtype=np.int64)
        cu = module.ModelRunnerFL._get_cumsum_and_arange(runner, num_tokens, out)
        # Independent oracle: per-sequence aranges concatenated.
        expected = np.concatenate([np.arange(int(n)) for n in num_tokens])
        np.testing.assert_array_equal(out, expected)
        np.testing.assert_array_equal(cu, np.cumsum(num_tokens))

    def test_cumsum_dtype_forwarded(self, model_runner_module):
        """cumsum_dtype controls the returned dtype."""
        import numpy as np

        module = model_runner_module
        runner = self._runner(module)
        out = np.zeros(5, dtype=np.int64)
        cu = module.ModelRunnerFL._get_cumsum_and_arange(
            runner, np.array([2, 3]), out, cumsum_dtype=np.int32
        )
        assert cu.dtype == np.int32


class TestGetEncoderSeqLens:
    """_get_encoder_seq_lens cross-attention encoder length computation."""

    def _spec(self, module):
        import torch

        return module.CrossAttentionSpec(
            block_size=16, num_kv_heads=2, head_size=64, dtype=torch.float16
        )

    def _buffer(self, n):
        import numpy as np
        import torch

        class FakeBuffer:
            def __init__(self, size):
                self.np = np.zeros(size, dtype=np.int32)
                self.gpu = torch.zeros(size, dtype=torch.int32)
                self.copied = None

            def copy_to_gpu(self, count):
                self.copied = count

        return FakeBuffer(n)

    def test_non_cross_spec_returns_nones(self, model_runner_module):
        """A full-attention spec short-circuits to (None, None)."""
        import torch

        module = model_runner_module
        spec = module.FullAttentionSpec(
            block_size=16, num_kv_heads=2, head_size=64, dtype=torch.float16
        )
        got = module.ModelRunnerFL._get_encoder_seq_lens(
            SimpleNamespace(), {"r0": 4}, spec, 1
        )
        assert got == (None, None)

    def test_sums_encoder_token_lengths(self, model_runner_module):
        """Each request's length is the sum of its mm_position lengths."""
        from vllm.multimodal.inputs import PlaceholderRange

        module = model_runner_module
        buf = self._buffer(4)
        feats_r0 = [
            SimpleNamespace(mm_position=PlaceholderRange(offset=0, length=5)),
            SimpleNamespace(mm_position=PlaceholderRange(offset=5, length=7)),
        ]
        feats_r1 = [
            SimpleNamespace(mm_position=PlaceholderRange(offset=0, length=3)),
        ]
        runner = SimpleNamespace(
            input_batch=SimpleNamespace(req_id_to_index={"r0": 0, "r1": 1, "pad": 2}),
            requests={
                "r0": SimpleNamespace(mm_features=feats_r0),
                "r1": SimpleNamespace(mm_features=feats_r1),
            },
            encoder_seq_lens=buf,
            model_config=SimpleNamespace(hf_config={}),
            max_encoder_len=999,
        )
        enc, enc_cpu = module.ModelRunnerFL._get_encoder_seq_lens(
            runner, {"r0": 5, "r1": 3}, self._spec(module), 3
        )
        # Independent oracle: r0 has 5+7 encoder tokens, r1 has 3, pads 0.
        assert enc_cpu.tolist() == [12, 3, 0]
        assert buf.copied == 3
        assert enc.shape[0] == 3

    def test_none_mm_features_zeroes_entry(self, model_runner_module):
        """A request without mm features gets an encoder length of 0."""
        module = model_runner_module
        buf = self._buffer(2)
        runner = SimpleNamespace(
            input_batch=SimpleNamespace(req_id_to_index={"r0": 0, "r1": 1}),
            requests={
                "r0": SimpleNamespace(mm_features=None),
                "r1": SimpleNamespace(
                    mm_features=[SimpleNamespace(mm_position=SimpleNamespace(length=9))]
                ),
            },
            encoder_seq_lens=buf,
            model_config=SimpleNamespace(hf_config={}),
            max_encoder_len=999,
        )
        _, enc_cpu = module.ModelRunnerFL._get_encoder_seq_lens(
            runner, {"r0": 1, "r1": 2}, self._spec(module), 2
        )
        assert enc_cpu.tolist() == [0, 9]

    def test_cudagraph_capture_uses_hf_max_source_positions(self, model_runner_module):
        """for_cudagraph_capture overrides lengths with max_source_positions."""
        module = model_runner_module
        buf = self._buffer(2)
        runner = SimpleNamespace(
            input_batch=SimpleNamespace(req_id_to_index={"r0": 0, "r1": 1}),
            requests={
                "r0": SimpleNamespace(mm_features=None),
                "r1": SimpleNamespace(
                    mm_features=[SimpleNamespace(mm_position=SimpleNamespace(length=4))]
                ),
            },
            encoder_seq_lens=buf,
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(max_source_positions=33)
            ),
            max_encoder_len=999,
        )
        _, enc_cpu = module.ModelRunnerFL._get_encoder_seq_lens(
            runner, {"r0": 1, "r1": 2}, self._spec(module), 2, True
        )
        assert enc_cpu.tolist() == [33, 33]

    def test_cudagraph_capture_falls_back_to_max_encoder_len(self, model_runner_module):
        """Without hf_config.max_source_positions, max_encoder_len is used."""
        module = model_runner_module
        buf = self._buffer(2)
        runner = SimpleNamespace(
            input_batch=SimpleNamespace(req_id_to_index={"r0": 0, "r1": 1}),
            requests={
                "r0": SimpleNamespace(mm_features=None),
                "r1": SimpleNamespace(mm_features=None),
            },
            encoder_seq_lens=buf,
            model_config=SimpleNamespace(hf_config={}),
            max_encoder_len=21,
        )
        _, enc_cpu = module.ModelRunnerFL._get_encoder_seq_lens(
            runner, {"r0": 1, "r1": 2}, self._spec(module), 2, True
        )
        assert enc_cpu.tolist() == [21, 21]


class TestComputeCascadeAttnPrefixLens:
    """_compute_cascade_attn_prefix_lens group traversal and gating."""

    def _attn_group(self, module, spec, builder):
        return SimpleNamespace(
            kv_cache_spec=spec,
            get_metadata_builder=lambda ubatch_id=0: builder,
        )

    def _full_spec(self, module):
        import torch

        return module.FullAttentionSpec(
            block_size=16, num_kv_heads=2, head_size=64, dtype=torch.float16
        )

    def test_encoder_only_group_contributes_zero(self, model_runner_module):
        """EncoderOnlyAttentionSpec groups report prefix len 0 (no cascade)."""
        import numpy as np
        import torch

        module = model_runner_module
        enc_spec = module.EncoderOnlyAttentionSpec(
            block_size=16, num_kv_heads=2, head_size=64, dtype=torch.float16
        )
        builder = SimpleNamespace(use_cascade_attention=lambda **k: True)
        runner = object.__new__(module.ModelRunnerFL)
        runner.kv_cache_config = SimpleNamespace(kv_cache_groups=[object()])
        runner.attn_groups = [[self._attn_group(module, enc_spec, builder)]]
        out = module.ModelRunnerFL._compute_cascade_attn_prefix_lens(
            runner,
            num_scheduled_tokens=np.array([4]),
            num_computed_tokens=np.array([0]),
            num_common_prefix_blocks=[0],
        )
        # Zero prefix len never enables cascade attention.
        assert out is None

    def test_positive_prefix_len_enables_cascade(self, model_runner_module):
        """Any positive prefix len flips the return to the 2D list."""
        import numpy as np

        module = model_runner_module
        seen = []
        builder = SimpleNamespace(
            use_cascade_attention=lambda **kwargs: seen.append(kwargs) or True
        )
        runner = object.__new__(module.ModelRunnerFL)
        runner.kv_cache_config = SimpleNamespace(kv_cache_groups=[object()])
        runner.attn_groups = [
            [self._attn_group(module, self._full_spec(module), builder)]
        ]
        runner.num_query_heads = 32
        runner.num_sms = 80
        runner.dcp_world_size = 1
        runner.use_alibi = False

        out = module.ModelRunnerFL._compute_cascade_attn_prefix_lens(
            runner,
            num_scheduled_tokens=np.array([4, 4]),
            num_computed_tokens=np.array([40, 40]),
            num_common_prefix_blocks=[2],
        )
        assert out == [[32]]
        assert seen, "builder must have been consulted"

    def test_all_zero_prefix_lens_return_none(self, model_runner_module):
        """Zero prefix lens everywhere keep cascade attention disabled."""
        import numpy as np

        module = model_runner_module
        builder = SimpleNamespace(use_cascade_attention=lambda **k: True)
        groups = [
            self._attn_group(module, self._full_spec(module), builder) for _ in range(2)
        ]
        runner = object.__new__(module.ModelRunnerFL)
        runner.kv_cache_config = SimpleNamespace(kv_cache_groups=[object()])
        runner.attn_groups = [groups]
        out = module.ModelRunnerFL._compute_cascade_attn_prefix_lens(
            runner,
            num_scheduled_tokens=np.array([4]),
            num_computed_tokens=np.array([40]),
            num_common_prefix_blocks=[0],
        )
        assert out is None

    def test_two_kv_cache_groups_indexed_separately(self, model_runner_module):
        """Each kv_cache_group reads its own num_common_prefix_blocks slot."""
        import numpy as np

        module = model_runner_module
        builders = [
            SimpleNamespace(use_cascade_attention=lambda **k: True) for _ in range(2)
        ]
        runner = object.__new__(module.ModelRunnerFL)
        runner.kv_cache_config = SimpleNamespace(kv_cache_groups=[object(), object()])
        runner.attn_groups = [
            [self._attn_group(module, self._full_spec(module), builders[0])],
            [self._attn_group(module, self._full_spec(module), builders[1])],
        ]
        runner.num_query_heads = 32
        runner.num_sms = 80
        runner.dcp_world_size = 1
        runner.use_alibi = False

        out = module.ModelRunnerFL._compute_cascade_attn_prefix_lens(
            runner,
            num_scheduled_tokens=np.array([4]),
            num_computed_tokens=np.array([64]),
            num_common_prefix_blocks=[1, 3],  # 16 vs 48 tokens
        )
        # Independent oracle: group 0 -> 16, group 1 -> 48 tokens.
        assert out == [[16], [48]]


class TestComputeCascadeAttnPrefixLen:
    """_compute_cascade_attn_prefix_len truncation rules."""

    def _spec(self, module, cls=None, **extra):
        import torch

        if cls is None:
            cls = module.FullAttentionSpec
        kwargs = dict(block_size=16, num_kv_heads=2, head_size=64)
        kwargs.update(extra)
        kwargs.setdefault("dtype", torch.float16)
        return cls(**kwargs)

    def _runner(self, module):
        return SimpleNamespace(
            num_query_heads=32,
            use_alibi=False,
            num_sms=80,
            dcp_world_size=1,
        )

    def test_zero_common_prefix_returns_zero(self, model_runner_module):
        """No shared blocks short-circuits to 0 without consulting builder."""
        import numpy as np

        module = model_runner_module
        builder = SimpleNamespace(
            use_cascade_attention=lambda **k: pytest.fail("must not be consulted")
        )
        out = module.ModelRunnerFL._compute_cascade_attn_prefix_len(
            self._runner(module),
            num_scheduled_tokens=np.array([4, 4]),
            num_computed_tokens=np.array([8, 8]),
            num_common_prefix_blocks=0,
            kv_cache_spec=self._spec(module),
            attn_metadata_builder=builder,
        )
        assert out == 0

    def test_prefix_capped_by_min_computed_and_block_aligned(self, model_runner_module):
        """Prefix len = min(blocks*size, min computed), floored to blocks."""
        import numpy as np

        module = model_runner_module
        seen = []
        builder = SimpleNamespace(
            use_cascade_attention=lambda **kwargs: seen.append(kwargs) or True
        )
        # blocks*16 = 64 tokens, but min computed is 33 -> floor to 32.
        out = module.ModelRunnerFL._compute_cascade_attn_prefix_len(
            self._runner(module),
            num_scheduled_tokens=np.array([4, 4]),
            num_computed_tokens=np.array([33, 50]),
            num_common_prefix_blocks=4,
            kv_cache_spec=self._spec(module),
            attn_metadata_builder=builder,
        )
        assert out == 32
        assert seen[0]["common_prefix_len"] == 32
        assert seen[0]["num_query_heads"] == 32
        assert seen[0]["num_kv_heads"] == 2
        assert seen[0]["use_sliding_window"] is False
        assert seen[0]["use_local_attention"] is False

    def test_sliding_window_flag_from_spec(self, model_runner_module):
        """SlidingWindowSpec sets use_sliding_window in the builder call."""
        import numpy as np

        module = model_runner_module
        seen = []
        builder = SimpleNamespace(
            use_cascade_attention=lambda **kwargs: seen.append(kwargs) or True
        )
        out = module.ModelRunnerFL._compute_cascade_attn_prefix_len(
            self._runner(module),
            num_scheduled_tokens=np.array([4]),
            num_computed_tokens=np.array([64]),
            num_common_prefix_blocks=2,
            kv_cache_spec=self._spec(
                module, module.SlidingWindowSpec, sliding_window=64
            ),
            attn_metadata_builder=builder,
        )
        assert out == 32
        assert seen[0]["use_sliding_window"] is True

    def test_sliding_window_flag_from_full_spec(self, model_runner_module):
        """FullAttentionSpec with sliding_window also sets the flag."""
        import numpy as np

        module = model_runner_module
        seen = []
        builder = SimpleNamespace(
            use_cascade_attention=lambda **kwargs: seen.append(kwargs) or True
        )
        out = module.ModelRunnerFL._compute_cascade_attn_prefix_len(
            self._runner(module),
            num_scheduled_tokens=np.array([4]),
            num_computed_tokens=np.array([64]),
            num_common_prefix_blocks=2,
            kv_cache_spec=self._spec(module, sliding_window=64),
            attn_metadata_builder=builder,
        )
        assert out == 32
        assert seen[0]["use_sliding_window"] is True

    def test_local_attention_flag_from_full_spec_chunk(self, model_runner_module):
        """FullAttentionSpec with attention_chunk_size sets the local flag."""
        import numpy as np

        module = model_runner_module
        seen = []
        builder = SimpleNamespace(
            use_cascade_attention=lambda **kwargs: seen.append(kwargs) or False
        )
        out = module.ModelRunnerFL._compute_cascade_attn_prefix_len(
            self._runner(module),
            num_scheduled_tokens=np.array([4]),
            num_computed_tokens=np.array([64]),
            num_common_prefix_blocks=2,
            kv_cache_spec=self._spec(module, attention_chunk_size=128),
            attn_metadata_builder=builder,
        )
        # Builder declined -> result forced to 0 even with a valid prefix.
        assert out == 0
        assert seen[0]["use_local_attention"] is True

    def test_local_attention_flag_from_chunked_spec(self, model_runner_module):
        """ChunkedLocalAttentionSpec also sets use_local_attention."""
        import numpy as np

        module = model_runner_module
        seen = []
        builder = SimpleNamespace(
            use_cascade_attention=lambda **kwargs: seen.append(kwargs) or True
        )
        out = module.ModelRunnerFL._compute_cascade_attn_prefix_len(
            self._runner(module),
            num_scheduled_tokens=np.array([4]),
            num_computed_tokens=np.array([64]),
            num_common_prefix_blocks=1,
            kv_cache_spec=self._spec(
                module, module.ChunkedLocalAttentionSpec, attention_chunk_size=64
            ),
            attn_metadata_builder=builder,
        )
        assert out == 16
        assert seen[0]["use_local_attention"] is True


class TestCalcMropePositions:
    """_calc_mrope_positions copies prompt positions and extends decodes."""

    def _runner(self, module, cpu_buf, np_buf):
        return SimpleNamespace(
            input_batch=SimpleNamespace(
                req_ids=["r0", "r1"],
                num_computed_tokens_cpu=[0, 4],
            ),
            requests={
                "r0": SimpleNamespace(
                    mrope_positions=cpu_buf[:, :8].clone(),
                    mrope_position_delta=None,
                    prompt_token_ids=list(range(8)),
                    prompt_embeds=None,
                ),
                "r1": SimpleNamespace(
                    mrope_positions=cpu_buf[:, :10].clone(),
                    mrope_position_delta=2,
                    prompt_token_ids=list(range(10)),
                    prompt_embeds=None,
                ),
            },
            mrope_positions=SimpleNamespace(cpu=cpu_buf, np=np_buf),
        )

    def test_prompt_copy_and_completion_extension(self, model_runner_module):
        """Prompt slices are copied; decode positions advance by delta."""
        import numpy as np
        import torch

        module = model_runner_module
        cpu_buf = torch.zeros(3, 32, dtype=torch.int64)
        np_buf = np.zeros((3, 32), dtype=np.int64)
        # Give the precomputed per-request prompt positions recognizable
        # per-axis patterns: axis i row j = (i+1)*100 + index.
        cpu_buf[0, :8] = torch.arange(100, 108)
        cpu_buf[1, :10] = torch.arange(200, 210)
        cpu_buf[2, :8] = torch.arange(300, 308)
        runner = self._runner(module, cpu_buf, np_buf)
        sched = SimpleNamespace(num_scheduled_tokens={"r0": 5, "r1": 4})
        module.ModelRunnerFL._calc_mrope_positions(runner, sched)

        # r0: computed=0, scheduled=5, prompt_len=8 -> prompt-only step,
        # the 5 scheduled prompt tokens are copied to dst columns 0..4 from
        # src columns 0..4 of r0's precomputed positions.
        torch.testing.assert_close(cpu_buf[0, :5], torch.arange(100, 105))
        # r1: computed=4, scheduled=4, prompt_len=10 -> computed+scheduled
        # (8) <= prompt_len (10) so all 4 are prompt tokens; they land at
        # dst columns 5..8 (after r0's 5 tokens) from src columns 4..7 of
        # r1's precomputed positions.
        torch.testing.assert_close(cpu_buf[1, 5:9], torch.arange(204, 208))
        # The decoder axes mirror the same layout for r0's slice.
        torch.testing.assert_close(cpu_buf[2, :5], torch.arange(300, 305))
        # Pure-prompt batches never touch the decode-position scratch array.
        assert not np_buf.any()

    def test_completion_positions_advance_with_delta(self, model_runner_module):
        """Decode positions are context_len + mrope_position_delta."""
        import numpy as np
        import torch

        module = model_runner_module
        cpu_buf = torch.zeros(3, 32, dtype=torch.int64)
        np_buf = np.zeros((3, 32), dtype=np.int64)
        runner = self._runner(module, cpu_buf, np_buf)
        # Make both requests pure decodes: computed+scheduled > prompt_len.
        runner.input_batch.num_computed_tokens_cpu = [8, 10]
        runner.requests["r0"].mrope_position_delta = 0
        sched = SimpleNamespace(num_scheduled_tokens={"r0": 2, "r1": 3})
        module.ModelRunnerFL._calc_mrope_positions(runner, sched)
        # Independent oracle (MRotaryEmbedding semantics): each axis row
        # gets context_len + delta + k for the k-th new token.
        # r0: context = 8, delta 0 -> positions 8, 9 on every axis.
        for axis in range(3):
            assert np_buf[axis, 0:2].tolist() == [8, 9]
        # r1: context = 10, delta 2 -> positions 12, 13, 14 on every axis.
        for axis in range(3):
            assert np_buf[axis, 2:5].tolist() == [12, 13, 14]


class TestInitMropePositions:
    """_init_mrope_positions input selection and model delegation."""

    def _req_state(self, prompt_token_ids=None, prompt_embeds=None, features=()):
        return SimpleNamespace(
            prompt_token_ids=prompt_token_ids,
            prompt_embeds=prompt_embeds,
            mm_features=list(features),
            mrope_positions=None,
            mrope_position_delta=None,
        )

    def test_prompt_token_ids_passed_through(self, model_runner_module):
        """The prompt's token ids are forwarded to the model helper."""
        import torch

        module = model_runner_module
        seen = []

        def helper(input_tokens, features):
            seen.append((list(input_tokens), features))
            return torch.arange(15).reshape(3, 5), 3

        model = SimpleNamespace(supports_mrope=True, get_mrope_input_positions=helper)
        runner = SimpleNamespace(get_model=lambda: model)
        req = self._req_state(prompt_token_ids=[1, 2, 3, 4, 5])
        module.ModelRunnerFL._init_mrope_positions(runner, req)
        assert seen[0][0] == [1, 2, 3, 4, 5]
        assert req.mrope_positions.shape == (3, 5)
        assert req.mrope_position_delta == 3

    def test_prompt_embeds_synthesizes_range(self, model_runner_module):
        """Embeds-only inputs pass list(range(seq_len)) instead."""
        import torch

        module = model_runner_module
        seen = []

        def helper(input_tokens, features):
            seen.append((list(input_tokens), features))
            return torch.zeros(3, 4), 0

        model = SimpleNamespace(supports_mrope=True, get_mrope_input_positions=helper)
        runner = SimpleNamespace(get_model=lambda: model)
        embeds = torch.zeros(4, 7)
        req = self._req_state(prompt_embeds=embeds)
        module.ModelRunnerFL._init_mrope_positions(runner, req)
        assert seen[0][0] == [0, 1, 2, 3]

    def test_neither_prompt_source_raises(self, model_runner_module):
        """No token ids and no embeds is a contract violation."""
        module = model_runner_module
        model = SimpleNamespace(
            supports_mrope=True,
            get_mrope_input_positions=lambda *a: (object(), 0),
        )
        runner = SimpleNamespace(get_model=lambda: model)
        req = self._req_state()
        with pytest.raises(ValueError, match="M-RoPE requires"):
            module.ModelRunnerFL._init_mrope_positions(runner, req)

    def test_prompt_embeds_modality_filtered_from_features(self, model_runner_module):
        """prompt_embeds passthrough features are excluded from the call."""
        module = model_runner_module
        seen = []

        def helper(input_tokens, features):
            seen.append((input_tokens, features))
            return None, 0

        model = SimpleNamespace(supports_mrope=True, get_mrope_input_positions=helper)
        runner = SimpleNamespace(get_model=lambda: model)
        keep = SimpleNamespace(modality="image")
        drop = SimpleNamespace(modality="prompt_embeds")
        req = self._req_state(prompt_token_ids=[9, 8], features=[keep, drop])
        module.ModelRunnerFL._init_mrope_positions(runner, req)
        assert seen[0][1] == [keep]


class TestExtractMmKwargs:
    """_extract_mm_kwargs batching of scheduled multimodal features."""

    def test_empty_scheduler_output_returns_empty(self, model_runner_module):
        """A falsy scheduler output short-circuits to {}."""
        module = model_runner_module
        runner = SimpleNamespace(is_multimodal_raw_input_only_model=True)
        assert module.ModelRunnerFL._extract_mm_kwargs(runner, None) == {}

    def test_non_raw_input_model_returns_empty(self, model_runner_module):
        """Non raw-input-only models skip extraction entirely."""
        module = model_runner_module
        sched = SimpleNamespace(scheduled_new_reqs=[SimpleNamespace(mm_features=[])])
        runner = SimpleNamespace(is_multimodal_raw_input_only_model=False)
        assert module.ModelRunnerFL._extract_mm_kwargs(runner, sched) == {}

    def test_batches_features_with_data(self, model_runner_module):
        """Features carrying data are grouped and batched per modality."""
        import torch

        from vllm.multimodal.inputs import MultiModalKwargsItem

        module = model_runner_module
        # Two identical shared items batch together (a shared field reduces
        # the batch to the shared data); a different-modality item forms
        # its own group under the same key name.
        image_item = MultiModalKwargsItem.dummy(nbytes=1)
        audio_item = MultiModalKwargsItem.dummy(nbytes=1)
        reqs = [
            SimpleNamespace(
                mm_features=[
                    SimpleNamespace(modality="image", data=image_item),
                    SimpleNamespace(modality="audio", data=audio_item),
                ]
            ),
            SimpleNamespace(
                mm_features=[SimpleNamespace(modality="image", data=image_item)]
            ),
        ]
        # A feature without data must be skipped.
        reqs.append(
            SimpleNamespace(mm_features=[SimpleNamespace(modality="image", data=None)])
        )
        sched = SimpleNamespace(scheduled_new_reqs=reqs)
        runner = SimpleNamespace(is_multimodal_raw_input_only_model=True, device=None)
        out = module.ModelRunnerFL._extract_mm_kwargs(runner, sched)
        # Independent oracle: every modality groups under the "dummy" key
        # (both batches share that key), the shared field keeps the data.
        assert set(out.keys()) == {"dummy"}
        assert isinstance(out["dummy"], torch.Tensor)
        assert out["dummy"].shape == (1,)

    def test_none_data_features_are_skipped(self, model_runner_module):
        """A batch whose features all lack data produces no kwargs."""
        module = model_runner_module
        sched = SimpleNamespace(
            scheduled_new_reqs=[
                SimpleNamespace(
                    mm_features=[SimpleNamespace(modality="image", data=None)]
                )
            ]
        )
        runner = SimpleNamespace(is_multimodal_raw_input_only_model=True, device=None)
        assert module.ModelRunnerFL._extract_mm_kwargs(runner, sched) == {}


class TestDummyMmKwargs:
    """_dummy_mm_kwargs gating on model kind and budget."""

    def test_non_raw_input_model_returns_empty(self, model_runner_module):
        """Non raw-input-only models never build dummy multimodal kwargs."""
        module = model_runner_module
        runner = SimpleNamespace(is_multimodal_raw_input_only_model=False)
        assert module.ModelRunnerFL._dummy_mm_kwargs(runner, 2) == {}

    def test_no_tower_modalities_returns_empty(self, model_runner_module):
        """An empty mm_max_toks_per_item (embed-only) skips dummy data."""
        module = model_runner_module
        runner = SimpleNamespace(
            is_multimodal_raw_input_only_model=True,
            mm_budget=SimpleNamespace(mm_max_toks_per_item={}),
        )
        assert module.ModelRunnerFL._dummy_mm_kwargs(runner, 2) == {}

    def test_uses_modality_with_max_tokens(self, model_runner_module):
        """The dummy batch is built for the heaviest-tower modality."""
        module = model_runner_module
        budget = SimpleNamespace(
            mm_max_toks_per_item={"image": 256, "video": 1024},
            get_modality_with_max_tokens=lambda: "video",
        )
        calls = []

        def dummy_batch(modality, num_seqs):
            calls.append((modality, num_seqs))
            return {"dummy": object()}

        runner = SimpleNamespace(
            is_multimodal_raw_input_only_model=True,
            mm_budget=budget,
            _get_mm_dummy_batch=dummy_batch,
        )
        out = module.ModelRunnerFL._dummy_mm_kwargs(runner, 3)
        assert calls == [("video", 3)]
        assert "dummy" in out


class TestGetMambaStateCopyFuncs:
    """_get_mamba_state_copy_funcs modern/legacy and caching paths."""

    def _config(self, module):
        import torch

        from vllm.v1.kv_cache_interface import (
            KVCacheConfig,
            KVCacheGroupSpec,
            MambaSpec,
        )

        spec = MambaSpec(
            block_size=16,
            shapes=((4, 8), (2, 6)),
            dtypes=(torch.float32, torch.float32),
        )
        group = KVCacheGroupSpec(layer_names=["mamba.0"], kv_cache_spec=spec)
        return KVCacheConfig(num_blocks=4, kv_cache_tensors=[], kv_cache_groups=[group])

    def _runner(self, module, cfg, model):
        return SimpleNamespace(
            kv_cache_config=cfg, model=model, _mamba_state_copy_funcs=None
        )

    def test_modern_api_validated_and_cached(self, model_runner_module):
        """get_mamba_state_copy_funcs output is validated then cached."""
        from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum

        module = model_runner_module
        cfg = self._config(module)
        mamba_type = MambaAttentionBackendEnum.MAMBA2
        funcs = {mamba_type: [object(), object()]}
        model = SimpleNamespace(get_mamba_state_copy_funcs=lambda types: funcs)
        runner = self._runner(module, cfg, model)
        out = module.ModelRunnerFL._get_mamba_state_copy_funcs(runner)
        assert out is funcs
        assert runner._mamba_state_copy_funcs is funcs
        # Second call returns the cached dict without model involvement.
        model.get_mamba_state_copy_funcs = lambda types: pytest.fail(
            "cached path must not re-query"
        )
        assert module.ModelRunnerFL._get_mamba_state_copy_funcs(runner) is funcs

    def test_legacy_api_broadcast_to_all_types(self, model_runner_module):
        """The legacy single func is replicated for every mamba type."""
        from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum

        module = model_runner_module
        cfg = self._config(module)
        sentinel = object()
        model = SimpleNamespace(get_mamba_state_copy_func=lambda: sentinel)
        runner = self._runner(module, cfg, model)
        out = module.ModelRunnerFL._get_mamba_state_copy_funcs(runner)
        assert set(out.keys()) == {MambaAttentionBackendEnum.MAMBA2}
        assert out[MambaAttentionBackendEnum.MAMBA2] is sentinel


class TestDetermineBatchExecutionAndPadding:
    """_determine_batch_execution_and_padding dispatch + stats wiring."""

    def _runner(self, module, dispatcher, parallel, metrics=False):
        return SimpleNamespace(
            _is_uniform_decode=module.ModelRunnerFL._is_uniform_decode,
            uniform_decode_query_len=1,
            model_config=SimpleNamespace(is_encoder_decoder=False),
            input_batch=SimpleNamespace(lora_id_to_lora_request={}),
            cudagraph_dispatcher=dispatcher,
            parallel_config=parallel,
            vllm_config=SimpleNamespace(
                parallel_config=parallel,
                observability_config=SimpleNamespace(cudagraph_metrics=metrics),
            ),
        )

    def _dispatch(self, mode, num_tokens):
        dispatcher = MagicMock()
        dispatcher.dispatch.return_value = (
            mode,
            SimpleNamespace(num_tokens=num_tokens),
        )
        return dispatcher

    def test_no_padding_without_dp_and_stats_disabled(self, model_runner_module):
        """TP>1 no longer pads; observability stats stay None when off."""
        import numpy as np

        from vllm.config import CUDAGraphMode

        module = model_runner_module
        dispatcher = self._dispatch(CUDAGraphMode.NONE, 10)
        parallel = SimpleNamespace(
            data_parallel_size=1, data_parallel_rank=0, tensor_parallel_size=4
        )
        runner = self._runner(module, dispatcher, parallel)
        result = module.ModelRunnerFL._determine_batch_execution_and_padding(
            runner,
            num_tokens=10,
            num_reqs=5,
            num_scheduled_tokens_np=np.array([2, 2, 2, 2, 2]),
            max_num_scheduled_tokens=2,
            use_cascade_attn=False,
        )
        mode, batch_desc, should_ubatch, dp_tokens, stats = result
        assert mode == CUDAGraphMode.NONE
        assert batch_desc.num_tokens == 10
        assert should_ubatch is False
        assert dp_tokens is None
        assert stats is None
        assert dispatcher.dispatch.call_args.kwargs["num_tokens"] == 10

    def test_cudagraph_stats_emitted_when_enabled(self, model_runner_module):
        """observability_config.cudagraph_metrics produces a CUDAGraphStat."""
        import numpy as np

        from vllm.config import CUDAGraphMode

        module = model_runner_module
        dispatcher = self._dispatch(CUDAGraphMode.NONE, 16)
        parallel = SimpleNamespace(data_parallel_size=1, data_parallel_rank=0)
        runner = self._runner(module, dispatcher, parallel, metrics=True)
        result = module.ModelRunnerFL._determine_batch_execution_and_padding(
            runner,
            num_tokens=10,
            num_reqs=5,
            num_scheduled_tokens_np=np.array([2, 2, 2, 2, 2]),
            max_num_scheduled_tokens=2,
            use_cascade_attn=False,
        )
        stats = result[4]
        assert stats is not None
        assert stats.num_unpadded_tokens == 10
        assert stats.num_padded_tokens == 16
        assert stats.num_paddings == 6

    def test_cascade_attention_disables_full_graph(self, model_runner_module):
        """use_cascade_attn=True feeds invalid_modes={FULL} to dispatch."""
        import numpy as np

        from vllm.config import CUDAGraphMode

        module = model_runner_module
        dispatcher = self._dispatch(CUDAGraphMode.NONE, 10)
        parallel = SimpleNamespace(data_parallel_size=1, data_parallel_rank=0)
        runner = self._runner(module, dispatcher, parallel)
        module.ModelRunnerFL._determine_batch_execution_and_padding(
            runner,
            num_tokens=10,
            num_reqs=5,
            num_scheduled_tokens_np=np.array([2, 2, 2, 2, 2]),
            max_num_scheduled_tokens=2,
            use_cascade_attn=True,
        )
        assert dispatcher.dispatch.call_args.kwargs["invalid_modes"] == {
            CUDAGraphMode.FULL
        }

    def test_force_eager_restricts_to_none_mode(self, model_runner_module):
        """force_eager passes valid_modes={CUDAGraphMode.NONE}."""
        import numpy as np

        from vllm.config import CUDAGraphMode

        module = model_runner_module
        dispatcher = self._dispatch(CUDAGraphMode.NONE, 10)
        parallel = SimpleNamespace(data_parallel_size=1, data_parallel_rank=0)
        runner = self._runner(module, dispatcher, parallel)
        module.ModelRunnerFL._determine_batch_execution_and_padding(
            runner,
            num_tokens=10,
            num_reqs=5,
            num_scheduled_tokens_np=np.array([2, 2, 2, 2, 2]),
            max_num_scheduled_tokens=2,
            use_cascade_attn=False,
            force_eager=True,
        )
        assert dispatcher.dispatch.call_args.kwargs["valid_modes"] == {
            CUDAGraphMode.NONE
        }

    def test_uniform_decode_forwarded(self, model_runner_module):
        """force_uniform_decode reaches the dispatcher untouched."""
        import numpy as np

        from vllm.config import CUDAGraphMode

        module = model_runner_module
        dispatcher = self._dispatch(CUDAGraphMode.NONE, 10)
        parallel = SimpleNamespace(data_parallel_size=1, data_parallel_rank=0)
        runner = self._runner(module, dispatcher, parallel)
        module.ModelRunnerFL._determine_batch_execution_and_padding(
            runner,
            num_tokens=10,
            num_reqs=5,
            num_scheduled_tokens_np=np.array([2, 2, 2, 2, 2]),
            max_num_scheduled_tokens=2,
            use_cascade_attn=False,
            force_uniform_decode=True,
        )
        assert dispatcher.dispatch.call_args.kwargs["uniform_decode"] is True

    def test_lora_state_computed_from_input_batch(self, model_runner_module):
        """has_lora/num_active_loras come from input_batch when not forced."""
        import numpy as np

        from vllm.config import CUDAGraphMode

        module = model_runner_module
        dispatcher = self._dispatch(CUDAGraphMode.NONE, 10)
        parallel = SimpleNamespace(data_parallel_size=1, data_parallel_rank=0)
        runner = self._runner(module, dispatcher, parallel)
        runner.input_batch = SimpleNamespace(
            lora_id_to_lora_request={"lora-1": object(), "lora-2": object()}
        )
        module.ModelRunnerFL._determine_batch_execution_and_padding(
            runner,
            num_tokens=10,
            num_reqs=5,
            num_scheduled_tokens_np=np.array([2, 2, 2, 2, 2]),
            max_num_scheduled_tokens=2,
            use_cascade_attn=False,
        )
        assert dispatcher.dispatch.call_args.kwargs["has_lora"] is True
        assert dispatcher.dispatch.call_args.kwargs["num_active_loras"] == 2

    def test_encoder_decoder_with_encoder_reqs_disables_full(self, model_runner_module):
        """has_encoder_output only when encoder-decoder and encoder reqs."""
        import numpy as np

        from vllm.config import CUDAGraphMode

        module = model_runner_module
        dispatcher = self._dispatch(CUDAGraphMode.NONE, 10)
        parallel = SimpleNamespace(data_parallel_size=1, data_parallel_rank=0)
        runner = self._runner(module, dispatcher, parallel)
        runner.model_config = SimpleNamespace(is_encoder_decoder=True)
        module.ModelRunnerFL._determine_batch_execution_and_padding(
            runner,
            num_tokens=10,
            num_reqs=5,
            num_scheduled_tokens_np=np.array([2, 2, 2, 2, 2]),
            max_num_scheduled_tokens=2,
            use_cascade_attn=False,
            num_encoder_reqs=2,
        )
        assert dispatcher.dispatch.call_args.kwargs["invalid_modes"] == {
            CUDAGraphMode.FULL
        }
