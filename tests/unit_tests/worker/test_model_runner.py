# Copyright (c) 2025 BAAI. All rights reserved.

"""
Tests for model runner module.

This module follows a layered testing strategy:
- Layer 1: Pure functions and data classes (no external dependencies)
- Layer 2: Methods with mocked dependencies
- Layer 3: Integration tests (in functional_tests/, requires GPU)

Note: These tests require vllm >= 0.13.0 with full installation.
"""

import inspect
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from vllm.config import CUDAGraphMode

# =============================================================================
# Test Utilities - Check availability before importing
# =============================================================================


def has_vllm_model_runner():
    """Check if vllm model runner dependencies are available."""
    try:
        from vllm_fl.worker.model_runner import ModelRunnerFL  # noqa: F401

        return True
    except (ImportError, AttributeError):
        return False


# Skip all tests if vllm model runner is not available
pytestmark = pytest.mark.skipif(
    not has_vllm_model_runner(), reason="vllm_fl.worker.model_runner not available"
)


class TestAsyncGPUModelRunnerOutputScopes:
    def test_scopes_preserve_async_copy_and_materialization_order(
        self,
        monkeypatch,
    ):
        import vllm_fl.worker.model_runner as model_runner

        scopes = []
        ready_event = MagicMock()
        default_stream = object()
        copy_stream = MagicMock()
        stream_context = MagicMock(return_value=nullcontext())
        device_fn = SimpleNamespace(
            current_stream=MagicMock(return_value=default_stream),
            stream=stream_context,
        )
        sampled_token_ids = MagicMock()
        sampled_token_ids.to.return_value = torch.tensor([[3], [4]])
        output = SimpleNamespace(sampled_token_ids=None, logprobs=None)

        monkeypatch.setattr(
            model_runner,
            "current_platform",
            SimpleNamespace(torch_device_fn=device_fn),
        )
        monkeypatch.setattr(
            model_runner.torch,
            "Event",
            MagicMock(return_value=ready_event),
        )
        monkeypatch.setattr(
            model_runner,
            "record_function_or_nullcontext",
            lambda name: scopes.append(name) or nullcontext(),
        )

        async_output = model_runner.AsyncGPUModelRunnerOutput(
            model_runner_output=output,
            sampled_token_ids=sampled_token_ids,
            logprobs_tensors=None,
            invalid_req_indices=[1],
            async_output_copy_stream=copy_stream,
            vocab_size=8,
        )
        result = async_output.get_output()

        assert result is output
        assert result.sampled_token_ids == [[3], []]
        assert result.logprobs is None
        copy_stream.wait_stream.assert_called_once_with(default_stream)
        sampled_token_ids.to.assert_called_once_with("cpu", non_blocking=True)
        ready_event.record.assert_called_once_with()
        ready_event.synchronize.assert_called_once_with()
        stream_context.assert_called_once_with(copy_stream)
        assert scopes == [
            "async_output: submit_wait_stream",
            "async_output: submit_d2h",
            "async_output: event_record",
            "async_output: get_event_sync",
            "async_output: tolist",
        ]


class TestAscendSamplerSelection:
    @pytest.mark.parametrize(
        ("device_type", "selected_name", "rejected_name"),
        [
            ("npu", "AscendSampler", "Sampler"),
            ("cuda", "Sampler", "AscendSampler"),
        ],
    )
    def test_model_runner_selects_sampler_for_platform(
        self,
        monkeypatch,
        device_type,
        selected_name,
        rejected_name,
    ):
        import vllm_fl.worker.model_runner as model_runner

        selected = MagicMock(return_value=object())
        rejected = MagicMock(side_effect=AssertionError("wrong sampler"))
        monkeypatch.setattr(
            model_runner,
            "current_platform",
            SimpleNamespace(device_type=device_type),
        )
        monkeypatch.setattr(model_runner, selected_name, selected)
        monkeypatch.setattr(model_runner, rejected_name, rejected)

        result = model_runner._make_sampler(
            "processed_logprobs",
            enable_global_stream_random_sample=True,
        )

        assert result is selected.return_value
        if device_type == "npu":
            selected.assert_called_once_with(
                logprobs_mode="processed_logprobs",
                enable_global_stream_random_sample=True,
            )
        else:
            selected.assert_called_once_with(logprobs_mode="processed_logprobs")
        rejected.assert_not_called()


class TestProfileDummySampler:
    def test_selects_terminal_states_then_profiles_logits_without_sampling(self):
        import vllm_fl.worker.model_runner as model_runner

        runner = object.__new__(model_runner.ModelRunnerFL)
        runner.max_num_tokens = 10
        runner.max_num_reqs = 4
        hidden_states = torch.arange(30, dtype=torch.float32).reshape(10, 3)
        logits = torch.randn((4, 128))
        compute_logits = MagicMock(return_value=logits)
        runner.model = SimpleNamespace(compute_logits=compute_logits)
        runner.sampler = MagicMock(
            side_effect=AssertionError("profile must not execute sampler")
        )
        runner.rejection_sampler = MagicMock(
            side_effect=AssertionError("profile must not execute speculative rejection")
        )

        result = runner._dummy_sampler_run(hidden_states)

        assert result is logits
        # max_num_tokens=10 is split as [2, 2, 2, 4], so the sampler must
        # select the terminal state for each request at [1, 3, 5, 9].
        torch.testing.assert_close(
            compute_logits.call_args.args[0], hidden_states[[1, 3, 5, 9]]
        )
        runner.sampler.assert_not_called()
        runner.rejection_sampler.assert_not_called()

    @pytest.mark.parametrize("dp_size", [1, 2, 3, 4, 8])
    def test_idle_dummy_propagates_graph_padding_metadata(
        self,
        monkeypatch,
        dp_size,
    ):
        import vllm_fl.worker.model_runner as model_runner

        class PaddingObserved(Exception):
            pass

        runner = object.__new__(model_runner.ModelRunnerFL)
        runner.vllm_config = SimpleNamespace(
            model_config=SimpleNamespace(multimodal_config=None),
            parallel_config=SimpleNamespace(num_ubatches=1),
        )
        runner.max_num_tokens = 8
        runner.scheduler_config = SimpleNamespace(max_num_seqs=8)
        runner.uniform_decode_query_len = 1

        tokens_across_dp = torch.ones(dp_size, dtype=torch.int32)
        runner._determine_batch_execution_and_padding = MagicMock(
            return_value=(
                CUDAGraphMode.FULL,
                SimpleNamespace(num_tokens=4, num_reqs=4),
                False,
                tokens_across_dp,
                None,
            )
        )

        observed = {}

        def observe_padding(
            should_ubatch,
            num_scheduled_tokens,
            num_tokens_padded,
            num_reqs_padded,
            num_ubatches,
        ):
            observed["should_ubatch"] = should_ubatch
            observed["num_scheduled_tokens"] = num_scheduled_tokens.copy()
            observed["num_tokens_padded"] = num_tokens_padded
            observed["num_reqs_padded"] = num_reqs_padded
            observed["num_ubatches"] = num_ubatches
            raise PaddingObserved

        monkeypatch.setattr(
            model_runner,
            "maybe_create_ubatch_slices",
            observe_padding,
        )

        with pytest.raises(PaddingObserved):
            runner._dummy_run(
                num_tokens=1,
                cudagraph_runtime_mode=CUDAGraphMode.FULL,
                uniform_decode=True,
            )

        # The matching Ascend contract reaches this branch for the one-request
        # idle dummy. Graph padding expands it to one token per padded request.
        assert observed["num_scheduled_tokens"].tolist() == [1, 1, 1, 1]
        assert observed["num_reqs_padded"] == 4
        # The dummy rank may pad its own scheduled-request metadata, but must
        # not overwrite the complete vector produced by the DP collective.
        assert tokens_across_dp.tolist() == [1] * dp_size

    def test_dummy_run_returns_full_hidden_states_without_device_index_tensor(self):
        import vllm_fl.worker.model_runner as model_runner

        source = inspect.getsource(model_runner.ModelRunnerFL._dummy_run)

        assert "logit_indices_device" not in source
        assert "return hidden_states, hidden_states" in source

    def test_dummy_run_never_mutates_complete_dp_token_vector(self):
        import vllm_fl.worker.model_runner as model_runner

        source = inspect.getsource(model_runner.ModelRunnerFL._dummy_run)

        assert "num_tokens_across_dp[:]" not in source

    @staticmethod
    def _run_dummy_ubatch_forward_context_case(
        monkeypatch,
        *,
        device_type,
        should_ubatch,
    ):
        import vllm_fl.worker.model_runner as model_runner

        runner = object.__new__(model_runner.ModelRunnerFL)
        runner.vllm_config = SimpleNamespace(
            model_config=SimpleNamespace(multimodal_config=None),
            parallel_config=SimpleNamespace(num_ubatches=2),
        )
        runner.max_num_tokens = 8
        runner.scheduler_config = SimpleNamespace(max_num_seqs=8)
        runner.uniform_decode_query_len = 1
        runner.lora_config = None
        runner.supports_mm_inputs = False
        runner.enable_prompt_embeds = False
        runner.uses_mrope = False
        runner.uses_xdrope_dim = 0
        runner.input_ids = SimpleNamespace(gpu=torch.arange(8))
        runner.positions = torch.arange(8)
        runner.model = object()
        runner.use_aux_hidden_state_outputs = False
        runner.speculative_config = None

        original_dp_tokens = torch.tensor([8, 8], dtype=torch.int32)
        runner._determine_batch_execution_and_padding = MagicMock(
            return_value=(
                CUDAGraphMode.NONE,
                SimpleNamespace(num_tokens=8, num_reqs=8),
                should_ubatch,
                original_dp_tokens,
                None,
            )
        )
        ubatch_slices = (
            [SimpleNamespace(num_tokens=4), SimpleNamespace(num_tokens=4)]
            if should_ubatch
            else None
        )
        monkeypatch.setattr(
            model_runner,
            "maybe_create_ubatch_slices",
            MagicMock(return_value=(ubatch_slices, ubatch_slices)),
        )
        monkeypatch.setattr(
            model_runner,
            "current_platform",
            SimpleNamespace(device_type=device_type),
        )
        monkeypatch.setattr(
            model_runner,
            "get_pp_group",
            lambda: SimpleNamespace(is_first_rank=True),
        )
        forward_context = MagicMock(side_effect=lambda *args, **kwargs: nullcontext())
        monkeypatch.setattr(
            model_runner,
            "_set_model_forward_context",
            forward_context,
        )

        runner._get_slot_mappings = MagicMock(return_value=(None, None))
        runner.synchronize_input_prep = MagicMock(return_value=nullcontext())
        runner.maybe_dummy_run_with_lora = MagicMock(return_value=nullcontext())
        runner._init_model_kwargs = MagicMock(return_value={})
        runner.maybe_randomize_inputs = MagicMock(return_value=nullcontext())
        runner._model_forward = MagicMock(return_value=torch.zeros((8, 4)))
        runner._register_layerwise_nvtx_hooks = MagicMock()

        runner._dummy_run(
            num_tokens=8,
            cudagraph_runtime_mode=CUDAGraphMode.NONE,
            uniform_decode=True,
            skip_eplb=True,
        )

        return original_dp_tokens, ubatch_slices, forward_context.call_args.kwargs

    def test_non_npu_ubatch_derives_forward_dp_vector_without_mutation(
        self,
        monkeypatch,
    ):
        original, ubatch_slices, context_kwargs = (
            self._run_dummy_ubatch_forward_context_case(
                monkeypatch,
                device_type="cuda",
                should_ubatch=True,
            )
        )

        forwarded = context_kwargs["num_tokens_across_dp"]
        assert context_kwargs["num_tokens"] == 4
        assert forwarded.tolist() == [4, 4]
        assert forwarded is not original
        assert original.tolist() == [8, 8]
        assert context_kwargs["ubatch_slices"] is ubatch_slices
        assert [slice_.num_tokens for slice_ in ubatch_slices] == [4, 4]

    def test_npu_without_ubatching_reuses_original_dp_vector(
        self,
        monkeypatch,
    ):
        original, ubatch_slices, context_kwargs = (
            self._run_dummy_ubatch_forward_context_case(
                monkeypatch,
                device_type="npu",
                should_ubatch=False,
            )
        )

        assert ubatch_slices is None
        assert context_kwargs["num_tokens_across_dp"] is original
        assert original.tolist() == [8, 8]
        assert context_kwargs["ubatch_slices"] is None

    def test_dummy_model_invocation_uses_forward_lifecycle_helper(self):
        import vllm_fl.worker.model_runner as model_runner

        source = inspect.getsource(model_runner.ModelRunnerFL._dummy_run)

        assert "outputs = self._model_forward(" in source
        assert "outputs = self.model(" not in source


class TestGlobalStreamRandomSample:
    @pytest.mark.parametrize(
        (
            "device_type",
            "additional_config",
            "batch_invariant",
            "expected",
        ),
        [
            ("npu", None, False, True),
            ("npu", {}, False, True),
            (
                "npu",
                {"enable_global_stream_random_sample": False},
                False,
                False,
            ),
            (
                "npu",
                {"enable_global_stream_random_sample": True},
                False,
                True,
            ),
            (
                "npu",
                {"enable_global_stream_random_sample": True},
                True,
                False,
            ),
            (
                "cuda",
                {"enable_global_stream_random_sample": True},
                False,
                False,
            ),
        ],
    )
    def test_config_is_npu_opt_in_and_batch_invariant_forces_off(
        self,
        monkeypatch,
        device_type,
        additional_config,
        batch_invariant,
        expected,
    ):
        import vllm_fl.worker.model_runner as model_runner

        monkeypatch.setattr(
            model_runner,
            "current_platform",
            SimpleNamespace(device_type=device_type),
        )
        monkeypatch.setattr(
            model_runner.envs,
            "VLLM_BATCH_INVARIANT",
            batch_invariant,
        )

        assert (
            model_runner._global_stream_random_sample_enabled(
                SimpleNamespace(additional_config=additional_config)
            )
            is expected
        )

    def test_global_stream_and_async_exponential_are_mutually_exclusive(self):
        import vllm_fl.worker.model_runner as model_runner

        with pytest.raises(ValueError, match="mutually exclusive"):
            model_runner._validate_sampling_optimization_flags(True, True)

        model_runner._validate_sampling_optimization_flags(True, False)
        model_runner._validate_sampling_optimization_flags(False, True)
        model_runner._validate_sampling_optimization_flags(False, False)


class TestHybridSamplingStreamReturnEdge:
    @pytest.mark.parametrize(
        (
            "device_type",
            "additional_config",
            "batch_invariant",
            "expected",
        ),
        [
            ("npu", None, False, True),
            ("npu", {}, False, True),
            (
                "npu",
                {"enable_hybrid_sampling_stream_return_edge": False},
                False,
                False,
            ),
            (
                "npu",
                {"enable_hybrid_sampling_stream_return_edge": True},
                False,
                True,
            ),
            (
                "npu",
                {
                    "enable_hybrid_sampling_stream_return_edge": True,
                    "enable_global_stream_random_sample": False,
                },
                False,
                False,
            ),
            (
                "npu",
                {"enable_hybrid_sampling_stream_return_edge": True},
                True,
                False,
            ),
            (
                "cuda",
                {"enable_hybrid_sampling_stream_return_edge": True},
                False,
                False,
            ),
        ],
    )
    def test_config_defaults_on_for_npu_global_stream(
        self,
        monkeypatch,
        device_type,
        additional_config,
        batch_invariant,
        expected,
    ):
        import vllm_fl.worker.model_runner as model_runner

        monkeypatch.setattr(
            model_runner,
            "current_platform",
            SimpleNamespace(device_type=device_type),
        )
        monkeypatch.setattr(
            model_runner.envs,
            "VLLM_BATCH_INVARIANT",
            batch_invariant,
        )

        assert (
            model_runner._hybrid_sampling_stream_return_edge_enabled(
                SimpleNamespace(additional_config=additional_config)
            )
            is expected
        )

    @staticmethod
    def _runner(
        *,
        enabled=True,
        global_stream=True,
        is_hybrid=True,
        has_mamba=True,
    ):
        from vllm_fl.worker.model_runner import ModelRunnerFL

        runner = object.__new__(ModelRunnerFL)
        runner.enable_hybrid_sampling_stream_return_edge = enabled
        runner.enable_global_stream_random_sample = global_stream
        runner.model_config = SimpleNamespace(is_hybrid=is_hybrid)
        runner.kv_cache_config = SimpleNamespace(has_mamba_layers=has_mamba)
        runner.hybrid_sampling_done_event = None
        return runner

    @pytest.mark.parametrize(
        ("enabled", "global_stream", "is_hybrid", "has_mamba", "expected"),
        [
            (True, True, True, True, True),
            (False, True, True, True, False),
            (True, False, True, True, False),
            (True, True, False, True, False),
            (True, True, True, False, False),
        ],
    )
    def test_runtime_requires_hybrid_mamba_and_global_stream(
        self,
        enabled,
        global_stream,
        is_hybrid,
        has_mamba,
        expected,
    ):
        runner = self._runner(
            enabled=enabled,
            global_stream=global_stream,
            is_hybrid=is_hybrid,
            has_mamba=has_mamba,
        )

        assert runner._hybrid_sampling_stream_return_edge_active() is expected

    def test_event_is_persistent_and_record_wait_precedes_state_update(
        self,
        monkeypatch,
    ):
        import vllm_fl.worker.model_runner as model_runner

        runner = self._runner()
        calls = []
        event = MagicMock()
        event.record.side_effect = lambda: calls.append("record")
        sampling_stream = MagicMock()
        sampling_stream.wait_event.side_effect = lambda value: (
            calls.append("wait") if value is event else calls.append("wrong_event")
        )
        event_factory = MagicMock(side_effect=lambda: calls.append("create") or event)
        stream_context = MagicMock(return_value=nullcontext())
        monkeypatch.setattr(model_runner.torch.npu, "Event", event_factory)
        monkeypatch.setattr(model_runner.torch.npu, "stream", stream_context)
        monkeypatch.setattr(
            model_runner,
            "_global_random_sample_stream",
            MagicMock(
                side_effect=lambda: calls.append("get_stream") or sampling_stream
            ),
        )
        monkeypatch.setattr(
            model_runner,
            "record_function_or_nullcontext",
            lambda _name: nullcontext(),
        )
        runner._update_states_after_model_execute = MagicMock(
            side_effect=lambda *_args: calls.append("update")
        )

        output_token_ids = object()
        scheduler_output = object()
        for _ in range(2):
            runner._record_hybrid_sampling_done_event()
            runner._update_states_after_sampling(
                output_token_ids,
                scheduler_output,
            )

        assert calls == [
            "create",
            "record",
            "get_stream",
            "wait",
            "update",
            "record",
            "get_stream",
            "wait",
            "update",
        ]
        assert runner.hybrid_sampling_done_event is event
        assert event_factory.call_count == 1
        assert event.record.call_count == 2
        assert sampling_stream.wait_event.call_count == 2
        assert stream_context.call_args_list == [
            ((sampling_stream,),),
            ((sampling_stream,),),
        ]
        assert runner._update_states_after_model_execute.call_count == 2

    def test_fallback_does_not_create_or_wait_on_event(self, monkeypatch):
        import vllm_fl.worker.model_runner as model_runner

        runner = self._runner(enabled=False)
        monkeypatch.setattr(
            model_runner.torch.npu,
            "Event",
            MagicMock(side_effect=AssertionError("event must not be created")),
        )
        monkeypatch.setattr(
            model_runner,
            "_global_random_sample_stream",
            MagicMock(side_effect=AssertionError("stream must not be used")),
        )
        runner._update_states_after_model_execute = MagicMock()

        output_token_ids = object()
        scheduler_output = object()
        runner._record_hybrid_sampling_done_event()
        runner._update_states_after_sampling(
            output_token_ids,
            scheduler_output,
        )

        assert runner.hybrid_sampling_done_event is None
        runner._update_states_after_model_execute.assert_called_once_with(
            output_token_ids,
            scheduler_output,
        )


class TestCompactDiscardIndices:
    @pytest.mark.parametrize(
        ("device_type", "additional_config", "speculative_config", "expected"),
        [
            ("npu", None, None, True),
            ("npu", {}, None, True),
            ("npu", {"enable_compact_discard_indices": True}, None, True),
            ("npu", {"enable_compact_discard_indices": False}, None, False),
            ("npu", {"enable_compact_discard_indices": True}, object(), False),
            ("cuda", {"enable_compact_discard_indices": True}, None, False),
        ],
    )
    def test_config_defaults_on_for_npu_only(
        self,
        monkeypatch,
        device_type,
        additional_config,
        speculative_config,
        expected,
    ):
        import vllm_fl.worker.model_runner as model_runner

        monkeypatch.setattr(
            model_runner,
            "current_platform",
            SimpleNamespace(device_type=device_type),
        )

        assert (
            model_runner._compact_discard_indices_enabled(
                SimpleNamespace(
                    additional_config=additional_config,
                    speculative_config=speculative_config,
                )
            )
            is expected
        )

    @staticmethod
    def _buffer(size, dtype):
        np_dtype = np.bool_ if dtype == torch.bool else np.int64
        return SimpleNamespace(
            np=np.zeros(size, dtype=np_dtype),
            copy_to_gpu=MagicMock(),
        )

    @classmethod
    def _runner(cls, *, compact):
        from vllm_fl.worker.model_runner import ModelRunnerFL

        runner = object.__new__(ModelRunnerFL)
        runner.enable_compact_discard_indices = compact
        runner.discard_request_mask = cls._buffer(8, torch.bool)
        runner.discard_request_indices = cls._buffer(8, torch.int64)
        runner.num_discarded_requests = 0
        runner.input_batch = SimpleNamespace(num_reqs=4)
        return runner

    def test_compact_path_copies_only_actual_indices(self):
        from vllm_fl.worker.model_runner import ModelRunnerFL

        runner = self._runner(compact=True)
        mask = np.array([False, True, False, True], dtype=np.bool_)

        ModelRunnerFL._record_discarded_requests(runner, mask, num_reqs=4)

        assert runner.num_discarded_requests == 2
        np.testing.assert_array_equal(
            runner.discard_request_indices.np[:2],
            np.array([1, 3], dtype=np.int64),
        )
        runner.discard_request_indices.copy_to_gpu.assert_called_once_with(2)
        runner.discard_request_mask.copy_to_gpu.assert_not_called()
        assert not ModelRunnerFL._is_all_reqs_chunked_prefill(runner)

    def test_compact_empty_batch_uses_zero_length_copy(self):
        from vllm_fl.worker.model_runner import ModelRunnerFL

        runner = self._runner(compact=True)

        ModelRunnerFL._record_discarded_requests(
            runner,
            np.zeros(4, dtype=np.bool_),
            num_reqs=4,
        )

        assert runner.num_discarded_requests == 0
        runner.discard_request_indices.copy_to_gpu.assert_called_once_with(0)
        runner.discard_request_mask.copy_to_gpu.assert_not_called()
        assert not ModelRunnerFL._is_all_reqs_chunked_prefill(runner)

    def test_dense_fallback_preserves_speculative_decode_contract(self):
        from vllm_fl.worker.model_runner import ModelRunnerFL

        runner = self._runner(compact=False)
        mask = np.ones(4, dtype=np.bool_)

        ModelRunnerFL._record_discarded_requests(runner, mask, num_reqs=4)

        assert runner.num_discarded_requests == 4
        np.testing.assert_array_equal(runner.discard_request_mask.np[:4], mask)
        runner.discard_request_mask.copy_to_gpu.assert_called_once_with(4)
        runner.discard_request_indices.copy_to_gpu.assert_not_called()
        assert ModelRunnerFL._is_all_reqs_chunked_prefill(runner)


class TestContiguousMropeCopy:
    @pytest.mark.parametrize(
        ("device_type", "additional_config", "expected"),
        [
            ("npu", None, True),
            ("npu", {}, True),
            ("npu", {"enable_contiguous_mrope_copy": True}, True),
            ("npu", {"enable_contiguous_mrope_copy": False}, False),
            ("cuda", {"enable_contiguous_mrope_copy": True}, False),
        ],
    )
    def test_config_defaults_on_for_npu_only(
        self,
        monkeypatch,
        device_type,
        additional_config,
        expected,
    ):
        import vllm_fl.worker.model_runner as model_runner

        monkeypatch.setattr(
            model_runner,
            "current_platform",
            SimpleNamespace(device_type=device_type),
        )

        assert (
            model_runner._contiguous_mrope_copy_enabled(
                SimpleNamespace(additional_config=additional_config)
            )
            is expected
        )

    @staticmethod
    def _runner(*, enabled, cpu, gpu):
        from vllm_fl.worker.model_runner import ModelRunnerFL

        runner = object.__new__(ModelRunnerFL)
        runner.enable_contiguous_mrope_copy = enabled
        runner.mrope_positions = SimpleNamespace(cpu=cpu, gpu=gpu)
        return runner

    def test_enabled_copies_complete_backing_buffer(self):
        from vllm_fl.worker.model_runner import ModelRunnerFL

        cpu = object()
        gpu = MagicMock()
        runner = self._runner(enabled=True, cpu=cpu, gpu=gpu)

        ModelRunnerFL._copy_mrope_positions_to_gpu(runner, num_tokens=2)

        gpu.copy_.assert_called_once_with(cpu, non_blocking=True)
        gpu.__getitem__.assert_not_called()

    def test_disabled_copies_only_requested_column_prefix(self):
        from vllm_fl.worker.model_runner import ModelRunnerFL

        cpu = MagicMock()
        gpu = MagicMock()
        cpu_prefix = object()
        gpu_prefix = MagicMock()
        cpu.__getitem__.return_value = cpu_prefix
        gpu.__getitem__.return_value = gpu_prefix
        runner = self._runner(enabled=False, cpu=cpu, gpu=gpu)

        ModelRunnerFL._copy_mrope_positions_to_gpu(runner, num_tokens=2)

        prefix = (slice(None), slice(None, 2))
        cpu.__getitem__.assert_called_once_with(prefix)
        gpu.__getitem__.assert_called_once_with(prefix)
        gpu_prefix.copy_.assert_called_once_with(
            cpu_prefix,
            non_blocking=True,
        )
        gpu.copy_.assert_not_called()

    @pytest.mark.parametrize("enabled", [True, False])
    def test_active_prefix_values_are_identical(self, enabled):
        from vllm_fl.worker.model_runner import ModelRunnerFL

        cpu = torch.arange(18, dtype=torch.int64).reshape(3, 6)
        gpu = torch.full_like(cpu, -1)
        runner = self._runner(enabled=enabled, cpu=cpu, gpu=gpu)

        ModelRunnerFL._copy_mrope_positions_to_gpu(runner, num_tokens=2)

        torch.testing.assert_close(gpu[:, :2], cpu[:, :2])
        if enabled:
            torch.testing.assert_close(gpu, cpu)
        else:
            torch.testing.assert_close(gpu[:, 2:], torch.full((3, 4), -1))


class TestInt32SlotMappingSource:
    @pytest.mark.parametrize(
        ("device_type", "additional_config", "speculative_config", "expected"),
        [
            ("npu", None, None, True),
            ("npu", {}, None, True),
            (
                "npu",
                {"enable_int32_slot_mapping_source": True},
                None,
                True,
            ),
            (
                "npu",
                {"enable_int32_slot_mapping_source": False},
                None,
                False,
            ),
            (
                "npu",
                {"enable_int32_slot_mapping_source": True},
                object(),
                False,
            ),
            (
                "cuda",
                {"enable_int32_slot_mapping_source": True},
                None,
                False,
            ),
        ],
    )
    def test_config_defaults_on_for_npu_non_spec_only(
        self,
        monkeypatch,
        device_type,
        additional_config,
        speculative_config,
        expected,
    ):
        import vllm_fl.worker.model_runner as model_runner

        monkeypatch.setattr(
            model_runner,
            "current_platform",
            SimpleNamespace(device_type=device_type),
        )

        assert (
            model_runner._int32_slot_mapping_source_enabled(
                SimpleNamespace(
                    additional_config=additional_config,
                    speculative_config=speculative_config,
                )
            )
            is expected
        )

    @staticmethod
    def _int64_input_batch(num_groups=4, size=16):
        class _BlockTableGroups:
            def __init__(self, block_tables):
                self.block_tables = block_tables

            def __getitem__(self, index):
                return self.block_tables[index]

        block_tables = []
        for _ in range(num_groups):
            slot_mapping = SimpleNamespace(
                cpu=torch.zeros(size, dtype=torch.int64),
                gpu=torch.zeros(size, dtype=torch.int64),
            )
            block_tables.append(SimpleNamespace(slot_mapping=slot_mapping))
        return SimpleNamespace(block_table=_BlockTableGroups(block_tables))

    def test_replaces_all_groups_once_with_independent_int32_buffers(self):
        import vllm_fl.worker.model_runner as model_runner

        input_batch = self._int64_input_batch()

        model_runner._replace_slot_mapping_sources_with_int32(
            input_batch,
            device=torch.device("cpu"),
            pin_memory=False,
        )

        buffers = [table.slot_mapping for table in input_batch.block_table.block_tables]
        object_ids = [id(buffer) for buffer in buffers]
        gpu_addresses = [buffer.gpu.data_ptr() for buffer in buffers]
        assert len(set(gpu_addresses)) == 4
        for buffer in buffers:
            assert buffer.cpu.shape == (16,)
            assert buffer.gpu.shape == (16,)
            assert buffer.cpu.dtype == torch.int32
            assert buffer.gpu.dtype == torch.int32
            assert buffer.np.dtype == np.int32

        # Duplicate setup must not invalidate graph-captured source addresses.
        model_runner._replace_slot_mapping_sources_with_int32(
            input_batch,
            device=torch.device("cpu"),
            pin_memory=False,
        )
        assert [
            id(table.slot_mapping) for table in input_batch.block_table.block_tables
        ] == object_ids
        assert [
            table.slot_mapping.gpu.data_ptr()
            for table in input_batch.block_table.block_tables
        ] == gpu_addresses

    def test_direct_sources_keep_addresses_and_padding_across_steps(self):
        from vllm_fl.worker import model_runner as model_runner_module

        input_batch = self._int64_input_batch()
        model_runner_module._replace_slot_mapping_sources_with_int32(
            input_batch,
            device=torch.device("cpu"),
            pin_memory=False,
        )
        runner = SimpleNamespace(
            kv_cache_config=SimpleNamespace(
                kv_cache_groups=[
                    SimpleNamespace(
                        kv_cache_spec=object(),
                        layer_names=[f"model.layers.{gid}.self_attn"],
                    )
                    for gid in range(4)
                ]
            ),
            input_batch=input_batch,
            device=torch.device("cpu"),
        )
        for gid, table in enumerate(input_batch.block_table.block_tables):
            table.slot_mapping.gpu.copy_(
                torch.arange(16, dtype=torch.int32) + gid * 100
            )

        with patch.object(
            model_runner_module,
            "current_platform",
            SimpleNamespace(device_type="npu"),
        ):
            first_by_group, _ = model_runner_module.ModelRunnerFL._get_slot_mappings(
                runner,
                num_tokens_padded=8,
                num_reqs_padded=4,
                num_tokens_unpadded=6,
            )
            first_addresses = {
                gid: mapping.data_ptr() for gid, mapping in first_by_group.items()
            }
            for gid, table in enumerate(input_batch.block_table.block_tables):
                table.slot_mapping.gpu[:8].copy_(
                    torch.arange(8, dtype=torch.int32) + gid * 1000
                )
            second_by_group, _ = model_runner_module.ModelRunnerFL._get_slot_mappings(
                runner,
                num_tokens_padded=8,
                num_reqs_padded=4,
                num_tokens_unpadded=7,
            )

        assert len(set(first_addresses.values())) == 4
        assert not hasattr(runner, "_ascend_slot_mapping_buffers")
        for gid, mapping in second_by_group.items():
            assert mapping.dtype == torch.int32
            assert mapping.data_ptr() == first_addresses[gid]
            torch.testing.assert_close(
                mapping,
                torch.tensor(
                    [
                        gid * 1000,
                        gid * 1000 + 1,
                        gid * 1000 + 2,
                        gid * 1000 + 3,
                        gid * 1000 + 4,
                        gid * 1000 + 5,
                        gid * 1000 + 6,
                        -1,
                    ],
                    dtype=torch.int32,
                ),
            )

    def test_kv_cache_reinitialization_replaces_rebuilt_sources(self):
        from vllm_fl.worker import model_runner as model_runner_module

        runner = object.__new__(model_runner_module.ModelRunnerFL)
        runner.max_model_len = 4096
        runner.max_encoder_len = 0
        runner.max_num_reqs = 32
        runner.max_num_tokens = 8192
        runner.device = torch.device("cpu")
        runner.pin_memory = False
        runner.is_pooling_model = False
        runner.cache_config = SimpleNamespace(enable_prefix_caching=True)
        runner.model_config = SimpleNamespace(
            get_vocab_size=MagicMock(return_value=1024)
        )
        runner.vllm_config = SimpleNamespace(speculative_config=None)
        runner._init_block_sizes = [16]
        runner._init_kernel_block_sizes = [16]
        runner.enable_int32_slot_mapping_source = True
        runner.input_batch = SimpleNamespace(
            logitsprocs=object(),
            logitsprocs_need_output_token_ids=False,
        )
        kv_cache_config = SimpleNamespace(
            kv_cache_groups=[
                SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=32))
            ]
        )
        rebuilt_batch = self._int64_input_batch(num_groups=1)

        with (
            patch.object(
                model_runner_module,
                "get_total_cp_world_size",
                return_value=1,
            ),
            patch.object(
                model_runner_module,
                "InputBatch",
                return_value=rebuilt_batch,
            ) as input_batch_ctor,
            patch.object(
                model_runner_module,
                "_replace_slot_mapping_sources_with_int32",
            ) as replace_sources,
        ):
            model_runner_module.ModelRunnerFL.may_reinitialize_input_batch(
                runner,
                kv_cache_config,
                kernel_block_sizes=[32],
            )

        assert runner.input_batch is rebuilt_batch
        input_batch_ctor.assert_called_once()
        replace_sources.assert_called_once_with(
            rebuilt_batch,
            device=torch.device("cpu"),
            pin_memory=False,
        )


class TestAsyncExponential:
    @pytest.mark.parametrize(
        (
            "device_type",
            "additional_config",
            "batch_invariant",
            "expected",
        ),
        [
            ("npu", None, False, False),
            ("npu", {}, False, False),
            ("npu", {"enable_async_exponential": False}, False, False),
            ("npu", {"enable_async_exponential": True}, False, True),
            ("npu", {"enable_async_exponential": True}, True, False),
            ("cuda", {"enable_async_exponential": True}, False, False),
        ],
    )
    def test_config_is_npu_opt_in_and_batch_invariant_forces_off(
        self,
        monkeypatch,
        device_type,
        additional_config,
        batch_invariant,
        expected,
    ):
        import vllm_fl.worker.model_runner as model_runner

        monkeypatch.setattr(
            model_runner,
            "current_platform",
            SimpleNamespace(device_type=device_type),
        )
        monkeypatch.setattr(
            model_runner.envs,
            "VLLM_BATCH_INVARIANT",
            batch_invariant,
        )

        assert (
            model_runner._async_exponential_enabled(
                SimpleNamespace(additional_config=additional_config)
            )
            is expected
        )

    @staticmethod
    def _runner(*, enabled=True, pooling=False, all_greedy=False):
        from vllm_fl.dispatch.backends.vendor.ascend.impl.sampler import (
            AscendSampler,
        )
        from vllm_fl.worker.model_runner import ModelRunnerFL

        runner = object.__new__(ModelRunnerFL)
        runner.enable_async_exponential = enabled
        runner.is_pooling_model = pooling
        runner.device = torch.device("cpu")
        runner.model_config = SimpleNamespace(
            get_vocab_size=MagicMock(return_value=248320)
        )
        runner.input_batch = SimpleNamespace(
            sampling_metadata=SimpleNamespace(
                all_greedy=all_greedy,
                generators={1: object()},
            )
        )
        runner.sampler = AscendSampler()
        runner.sampler.prepare_async_exponential = MagicMock()
        runner.sampler.discard_async_exponential = MagicMock(return_value=True)
        return runner

    def test_prepare_uses_actual_logits_indices_and_model_vocab(self):
        from vllm_fl.worker.model_runner import ModelRunnerFL

        runner = self._runner()
        logits_indices = torch.tensor([0, 4, 9, 12, 20, 21, 30])

        prepared = ModelRunnerFL._prepare_async_exponential(
            runner,
            logits_indices,
            None,
        )

        assert prepared
        runner.sampler.prepare_async_exponential.assert_called_once_with(
            batch_size=logits_indices.numel(),
            vocab_size=248320,
            generators=runner.input_batch.sampling_metadata.generators,
            device=torch.device("cpu"),
        )
        runner.model_config.get_vocab_size.assert_called_once_with()

    @pytest.mark.parametrize(
        ("enabled", "pooling", "all_greedy", "batch_size", "spec_decode"),
        [
            (False, False, False, 3, False),
            (True, True, False, 3, False),
            (True, False, True, 3, False),
            (True, False, False, 0, False),
            (True, False, False, 3, True),
        ],
    )
    def test_prepare_skips_unsupported_batches(
        self,
        enabled,
        pooling,
        all_greedy,
        batch_size,
        spec_decode,
    ):
        from vllm_fl.worker.model_runner import ModelRunnerFL

        runner = self._runner(
            enabled=enabled,
            pooling=pooling,
            all_greedy=all_greedy,
        )

        prepared = ModelRunnerFL._prepare_async_exponential(
            runner,
            torch.arange(batch_size),
            object() if spec_decode else None,
        )

        assert not prepared
        runner.sampler.prepare_async_exponential.assert_not_called()

    def test_error_boundary_discards_pending_q_and_reraises(self):
        from vllm_fl.worker.model_runner import ModelRunnerFL

        runner = self._runner()

        with (
            pytest.raises(RuntimeError, match="forward failed"),
            ModelRunnerFL._discard_async_exponential_on_error(runner),
        ):
            raise RuntimeError("forward failed")

        runner.sampler.discard_async_exponential.assert_called_once_with()

    def test_success_boundary_does_not_discard_pending_q(self):
        from vllm_fl.worker.model_runner import ModelRunnerFL

        runner = self._runner()

        with ModelRunnerFL._discard_async_exponential_on_error(runner):
            pass

        runner.sampler.discard_async_exponential.assert_not_called()

    def test_cleanup_failure_does_not_mask_original_error(self):
        from vllm_fl.worker.model_runner import ModelRunnerFL

        runner = self._runner()
        runner.sampler.discard_async_exponential.side_effect = RuntimeError(
            "cleanup failed"
        )

        with (
            pytest.raises(RuntimeError, match="compute logits failed"),
            ModelRunnerFL._discard_async_exponential_on_error(runner),
        ):
            raise RuntimeError("compute logits failed")


class _QueryStartLocBuffer:
    def __init__(self, size):
        self.np = np.zeros(size, dtype=np.int32)
        self.cpu = torch.from_numpy(self.np)
        self.gpu = self.cpu
        self.copy_to_gpu = MagicMock()


class TestAscendFIAQueryStartLoc:
    def _runner(self, values, *, max_num_reqs=32):
        from vllm_fl.worker.model_runner import ModelRunnerFL

        runner = object.__new__(ModelRunnerFL)
        runner.max_num_reqs = max_num_reqs
        runner.query_start_loc = _QueryStartLocBuffer(max_num_reqs + 2)
        runner.query_start_loc.np[: len(values)] = values
        runner.arange_np = np.arange(max_num_reqs + 2, dtype=np.int32)
        runner.uniform_decode_query_len = 1
        runner.compilation_config = SimpleNamespace(
            cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY
        )
        runner._has_gdn = False
        return runner

    def test_uniform_c30_is_padded_to_c32_with_virtual_requests(self):
        runner = self._runner(np.arange(31, dtype=np.int32))

        padded_reqs = runner._pad_query_start_loc_for_fia(
            num_tokens_padded=32,
            num_reqs_padded=32,
            num_reqs=30,
            cudagraph_runtime_mode=CUDAGraphMode.FULL,
            batch_desc_num_reqs=32,
        )

        assert padded_reqs == 32
        np.testing.assert_array_equal(
            runner.query_start_loc.np[:33],
            np.arange(33, dtype=np.int32),
        )
        runner.query_start_loc.copy_to_gpu.assert_called_once()

    def test_mixed_batch_appends_one_dummy_request(self):
        runner = self._runner([0, 2, 5])

        padded_reqs = runner._pad_query_start_loc_for_fia(
            num_tokens_padded=8,
            num_reqs_padded=2,
            num_reqs=2,
            cudagraph_runtime_mode=CUDAGraphMode.FULL,
            batch_desc_num_reqs=2,
        )

        assert padded_reqs == 3
        np.testing.assert_array_equal(
            runner.query_start_loc.np[:4],
            np.array([0, 2, 5, 8], dtype=np.int32),
        )

    def test_non_full_mode_does_not_request_fia_padding(self):
        from vllm_fl.worker.model_runner import (
            _should_pad_fia_query_start_loc,
        )

        assert not _should_pad_fia_query_start_loc("npu", CUDAGraphMode.NONE)
        assert not _should_pad_fia_query_start_loc("cuda", CUDAGraphMode.FULL)
        assert _should_pad_fia_query_start_loc("npu", CUDAGraphMode.FULL)

    def test_gdn_builder_receives_unpadded_query_boundaries(
        self,
        monkeypatch,
    ):
        import vllm_fl.worker.model_runner as model_runner

        class FakeGDNBuilder:
            pass

        monkeypatch.setattr(
            model_runner,
            "GDNAttentionMetadataBuilder",
            FakeGDNBuilder,
        )
        runner = self._runner(np.arange(33, dtype=np.int32))
        runner._has_gdn = True
        runner.gdn_query_start_loc = _QueryStartLocBuffer(33)
        runner.gdn_query_start_loc.np[:31] = np.arange(31, dtype=np.int32)
        runner.gdn_query_start_loc.np[31:] = 30
        common = SimpleNamespace(
            num_reqs=32,
            query_start_loc_cpu=runner.query_start_loc.cpu[:33],
            query_start_loc=runner.query_start_loc.gpu[:33],
        )

        gdn_common = runner._common_attention_metadata_for_builder(
            common,
            FakeGDNBuilder(),
        )
        fia_common = runner._common_attention_metadata_for_builder(
            common,
            object(),
        )

        assert gdn_common is not common
        torch.testing.assert_close(
            gdn_common.query_start_loc_cpu,
            torch.tensor(list(range(31)) + [30, 30], dtype=torch.int32),
        )
        assert fia_common is common
        assert int(fia_common.query_start_loc_cpu[-1]) == 32


# =============================================================================
# Layer 1: ExecuteModelState Data Structure Tests
# =============================================================================


class TestExecuteModelState:
    """Test ExecuteModelState NamedTuple behavior and contract."""

    def test_fields_match_expected_contract(self):
        """Verify ExecuteModelState has exact fields required by execute_model pipeline."""
        from vllm_fl.worker.model_runner import ExecuteModelState

        expected_fields = (
            "scheduler_output",
            "logits",
            "spec_decode_metadata",
            "spec_decode_common_attn_metadata",
            "hidden_states",
            "sample_hidden_states",
            "aux_hidden_states",
            "ec_connector_output",
            "cudagraph_stats",
            "slot_mappings",
        )
        assert ExecuteModelState._fields == expected_fields, (
            "ExecuteModelState fields changed - this may break execute_model consumers"
        )

    def test_immutability_prevents_accidental_mutation(self):
        """Ensure state cannot be mutated after creation (important for pipeline safety)."""
        from vllm_fl.worker.model_runner import ExecuteModelState

        state = ExecuteModelState(
            scheduler_output=MagicMock(),
            logits=torch.randn(4, 1000),
            spec_decode_metadata=None,
            spec_decode_common_attn_metadata=None,
            hidden_states=torch.randn(4, 512),
            sample_hidden_states=torch.randn(4, 512),
            aux_hidden_states=None,
            ec_connector_output=None,
            cudagraph_stats=None,
            slot_mappings=None,
        )

        with pytest.raises(AttributeError):
            state.logits = torch.randn(4, 1000)

    def test_unpacking_for_downstream_processing(self):
        """Test that state can be unpacked correctly for downstream use."""
        from vllm_fl.worker.model_runner import ExecuteModelState

        mock_scheduler = MagicMock()
        mock_logits = torch.randn(4, 1000)

        state = ExecuteModelState(
            scheduler_output=mock_scheduler,
            logits=mock_logits,
            spec_decode_metadata=None,
            spec_decode_common_attn_metadata=None,
            hidden_states=None,
            sample_hidden_states=None,
            aux_hidden_states=None,
            ec_connector_output=None,
            cudagraph_stats=None,
            slot_mappings=None,
        )

        # Simulate downstream unpacking
        scheduler, logits, *rest = state
        assert scheduler is mock_scheduler
        assert torch.equal(logits, mock_logits)


# =============================================================================
# Layer 2: _get_cumsum_and_arange Algorithm Tests
# =============================================================================


class TestGetCumsumAndArange:
    """Test _get_cumsum_and_arange method - critical for batch processing."""

    @pytest.fixture
    def mock_model_runner(self):
        """Create a minimal mock of ModelRunnerFL for testing."""
        from vllm_fl.worker.model_runner import ModelRunnerFL

        mock_runner = MagicMock(spec=ModelRunnerFL)
        mock_runner.arange_np = np.arange(10000)
        mock_runner._get_cumsum_and_arange = (
            ModelRunnerFL._get_cumsum_and_arange.__get__(mock_runner, ModelRunnerFL)
        )
        return mock_runner

    def test_multi_sequence_batch(self, mock_model_runner):
        """Test cumsum and per-sequence arange for typical multi-sequence batch."""
        num_tokens = np.array([2, 5, 3])
        arange_out = np.zeros(10, dtype=np.int64)

        cu_num_tokens = mock_model_runner._get_cumsum_and_arange(num_tokens, arange_out)

        # Cumsum: [2, 7, 10] - used for indexing into flattened batch
        np.testing.assert_array_equal(cu_num_tokens, np.array([2, 7, 10]))

        # Arange: per-sequence position indices [0,1 | 0,1,2,3,4 | 0,1,2]
        expected_arange = np.array([0, 1, 0, 1, 2, 3, 4, 0, 1, 2])
        np.testing.assert_array_equal(arange_out, expected_arange)

    def test_single_sequence(self, mock_model_runner):
        """Test with single sequence (common in generation phase)."""
        num_tokens = np.array([5])
        arange_out = np.zeros(5, dtype=np.int64)

        cu_num_tokens = mock_model_runner._get_cumsum_and_arange(num_tokens, arange_out)

        np.testing.assert_array_equal(cu_num_tokens, np.array([5]))
        np.testing.assert_array_equal(arange_out, np.array([0, 1, 2, 3, 4]))

    def test_all_single_token_sequences(self, mock_model_runner):
        """Test batch where each sequence has 1 token (decode phase)."""
        num_tokens = np.array([1, 1, 1, 1])
        arange_out = np.zeros(4, dtype=np.int64)

        cu_num_tokens = mock_model_runner._get_cumsum_and_arange(num_tokens, arange_out)

        np.testing.assert_array_equal(cu_num_tokens, np.array([1, 2, 3, 4]))
        np.testing.assert_array_equal(arange_out, np.array([0, 0, 0, 0]))

    def test_large_sequences(self, mock_model_runner):
        """Test with larger sequences to verify correct boundary handling."""
        num_tokens = np.array([10, 20, 30])
        arange_out = np.zeros(60, dtype=np.int64)

        cu_num_tokens = mock_model_runner._get_cumsum_and_arange(num_tokens, arange_out)

        assert cu_num_tokens[-1] == 60
        # Verify boundaries: first seq 0-9, second seq 0-19, third seq 0-29
        np.testing.assert_array_equal(arange_out[:10], np.arange(10))
        np.testing.assert_array_equal(arange_out[10:30], np.arange(20))
        np.testing.assert_array_equal(arange_out[30:60], np.arange(30))

    def test_dtype_preservation(self, mock_model_runner):
        """Test that dtype is correctly applied to cumsum output."""
        num_tokens = np.array([2, 3])
        arange_out = np.zeros(5, dtype=np.int64)

        cu_num_tokens = mock_model_runner._get_cumsum_and_arange(
            num_tokens, arange_out, cumsum_dtype=np.int64
        )

        assert cu_num_tokens.dtype == np.int64


class TestAscendSlotMappingBuffers:
    def test_int32_conversion_buffer_has_stable_address_across_steps(self):
        from vllm_fl.worker import model_runner as model_runner_module

        source = torch.arange(16, dtype=torch.int64)
        runner = SimpleNamespace(
            kv_cache_config=SimpleNamespace(
                kv_cache_groups=[
                    SimpleNamespace(
                        kv_cache_spec=object(),
                        layer_names=["model.layers.0.self_attn"],
                    )
                ]
            ),
            input_batch=SimpleNamespace(
                block_table=[SimpleNamespace(slot_mapping=SimpleNamespace(gpu=source))]
            ),
            device=torch.device("cpu"),
        )

        with patch.object(
            model_runner_module,
            "current_platform",
            SimpleNamespace(device_type="npu"),
        ):
            first_by_group, _ = model_runner_module.ModelRunnerFL._get_slot_mappings(
                runner,
                num_tokens_padded=8,
                num_reqs_padded=1,
                num_tokens_unpadded=6,
            )
            first = first_by_group[0]
            first_address = first.data_ptr()

            source[:8].add_(100)
            second_by_group, _ = model_runner_module.ModelRunnerFL._get_slot_mappings(
                runner,
                num_tokens_padded=8,
                num_reqs_padded=1,
                num_tokens_unpadded=7,
            )
            second = second_by_group[0]

        assert first.dtype == torch.int32
        assert second.data_ptr() == first_address
        torch.testing.assert_close(
            second,
            torch.tensor(
                [100, 101, 102, 103, 104, 105, 106, -1],
                dtype=torch.int32,
            ),
        )


class TestDenseMambaStateViews:
    def test_dense_mamba_views_are_non_overlapping(self):
        from vllm_fl.worker.model_runner import _make_dense_mamba_state_views

        num_blocks = 4
        shapes = [(3, 8), (2, 8, 8)]
        dtypes = [torch.bfloat16, torch.float32]
        required_bytes = sum(
            num_blocks
            * int(np.prod(shape))
            * torch.empty((), dtype=dtype).element_size()
            for shape, dtype in zip(shapes, dtypes)
        )
        raw = torch.empty(required_bytes + 64, dtype=torch.uint8)

        conv_state, ssm_state = _make_dense_mamba_state_views(
            raw, num_blocks, shapes, dtypes
        )

        assert conv_state.shape == (num_blocks, *shapes[0])
        assert ssm_state.shape == (num_blocks, *shapes[1])
        assert conv_state.stride(0) == int(np.prod(shapes[0]))
        assert ssm_state.stride(0) == int(np.prod(shapes[1]))

        conv_start = conv_state.storage_offset() * conv_state.element_size()
        conv_end = conv_start + conv_state.numel() * conv_state.element_size()
        ssm_start = ssm_state.storage_offset() * ssm_state.element_size()
        ssm_end = ssm_start + ssm_state.numel() * ssm_state.element_size()
        assert conv_start == 0
        assert conv_end == ssm_start
        assert ssm_end == required_bytes


class TestAscendHybridKVCacheLayout:
    ATTN_LAYER = "model.layers.0.self_attn"
    MAMBA_LAYER = "model.layers.1.linear_attn"

    @staticmethod
    def _qwen_tp2_config():
        from vllm.v1.kv_cache_interface import (
            FullAttentionSpec,
            KVCacheConfig,
            KVCacheGroupSpec,
            KVCacheTensor,
            MambaSpec,
        )

        # Qwen3.6-35B-A3B TP2 evidence from the v0.20.2rc1 service:
        #   conv=24,576 bytes, SSM/K=1,048,576 bytes,
        #   attention K+V=2,097,152 bytes, pool page=2,121,728 bytes.
        block_size = 4096
        pool_page_bytes = 2_121_728
        attn_spec = FullAttentionSpec(
            block_size=block_size,
            num_kv_heads=1,
            head_size=128,
            dtype=torch.bfloat16,
            page_size_padded=pool_page_bytes,
        )
        mamba_spec = MambaSpec(
            block_size=block_size,
            shapes=((12_288,), (524_288,)),
            dtypes=(torch.bfloat16, torch.bfloat16),
            page_size_padded=pool_page_bytes,
        )
        num_blocks = 10
        config = KVCacheConfig(
            num_blocks=num_blocks,
            kv_cache_tensors=[
                KVCacheTensor(
                    size=num_blocks * pool_page_bytes,
                    shared_by=[
                        TestAscendHybridKVCacheLayout.ATTN_LAYER,
                        TestAscendHybridKVCacheLayout.MAMBA_LAYER,
                    ],
                )
            ],
            kv_cache_groups=[
                KVCacheGroupSpec(
                    layer_names=[TestAscendHybridKVCacheLayout.ATTN_LAYER],
                    kv_cache_spec=attn_spec,
                ),
                KVCacheGroupSpec(
                    layer_names=[TestAscendHybridKVCacheLayout.MAMBA_LAYER],
                    kv_cache_spec=mamba_spec,
                ),
            ],
        )
        return config, attn_spec, mamba_spec

    @staticmethod
    def _runner(config, attn_spec, mamba_spec):
        from vllm_fl.worker.model_runner import ModelRunnerFL

        class FakeAscendBackend:
            @staticmethod
            def get_kv_cache_shape(
                num_blocks,
                block_size,
                num_kv_heads,
                head_size,
                cache_dtype_str="auto",
            ):
                del cache_dtype_str
                return (
                    2,
                    num_blocks,
                    block_size,
                    num_kv_heads,
                    head_size,
                )

        runner = object.__new__(ModelRunnerFL)
        runner.device = torch.device("cpu")
        runner.runner_only_attn_layers = set()
        runner.cache_config = SimpleNamespace(cache_dtype="auto")
        runner.kv_cache_config = config
        runner.attn_groups = [
            [
                SimpleNamespace(
                    kv_cache_spec=attn_spec,
                    backend=FakeAscendBackend,
                    kv_cache_group_id=0,
                    layer_names=[TestAscendHybridKVCacheLayout.ATTN_LAYER],
                )
            ],
            [
                SimpleNamespace(
                    kv_cache_spec=mamba_spec,
                    backend=object(),
                    kv_cache_group_id=1,
                    layer_names=[TestAscendHybridKVCacheLayout.MAMBA_LAYER],
                )
            ],
        ]
        return runner

    def test_qwen_tp2_pool_is_allocated_once_and_views_match_boundaries(self):
        import vllm_fl.worker.model_runner as model_runner

        config, attn_spec, mamba_spec = self._qwen_tp2_config()
        runner = self._runner(config, attn_spec, mamba_spec)
        real_zeros = torch.zeros

        with patch.object(
            model_runner.torch,
            "zeros",
            side_effect=real_zeros,
        ) as zeros:
            raw_by_layer = runner._allocate_kv_cache_tensors(config)

        assert zeros.call_count == 1
        raw = raw_by_layer[self.ATTN_LAYER]
        assert raw_by_layer[self.MAMBA_LAYER] is raw
        assert raw.nbytes == 10 * 2_121_728
        assert runner.hybrid_with_attn_and_mamba

        with patch.object(
            model_runner,
            "current_platform",
            SimpleNamespace(device_type="npu"),
        ):
            caches = runner._reshape_kv_cache_tensors(
                raw_by_layer,
                kernel_block_sizes=[128, 4096],
            )

        attn_cache = caches[self.ATTN_LAYER]
        conv_state, ssm_state = caches[self.MAMBA_LAYER]
        conv_bytes = 10 * 24_576
        state_or_k_bytes = 10 * 1_048_576

        assert attn_cache.shape == (2, 320, 128, 1, 128)
        assert attn_cache.is_contiguous()
        assert conv_state.shape == (10, 12_288)
        assert ssm_state.shape == (10, 524_288)

        raw_ptr = raw.data_ptr()
        assert conv_state.data_ptr() == raw_ptr
        assert ssm_state.data_ptr() == raw_ptr + conv_bytes
        assert attn_cache[0].data_ptr() == raw_ptr + conv_bytes
        assert attn_cache[1].data_ptr() == (raw_ptr + conv_bytes + state_or_k_bytes)
        assert attn_cache[1].data_ptr() + state_or_k_bytes == raw_ptr + raw.nbytes
        assert (
            conv_state.untyped_storage().data_ptr()
            == ssm_state.untyped_storage().data_ptr()
            == attn_cache.untyped_storage().data_ptr()
            == raw.untyped_storage().data_ptr()
        )

    def test_layer_spec_resolution_supports_uniform_wrappers(self):
        from vllm.v1.kv_cache_interface import (
            KVCacheConfig,
            KVCacheGroupSpec,
            UniformTypeKVCacheSpecs,
        )

        from vllm_fl.worker.model_runner import ModelRunnerFL

        config, attn_spec, _ = self._qwen_tp2_config()
        wrapped = UniformTypeKVCacheSpecs(
            block_size=attn_spec.block_size,
            kv_cache_specs={self.ATTN_LAYER: attn_spec},
        )
        uniform_config = KVCacheConfig(
            num_blocks=config.num_blocks,
            kv_cache_tensors=config.kv_cache_tensors,
            kv_cache_groups=[
                KVCacheGroupSpec(
                    layer_names=[self.ATTN_LAYER],
                    kv_cache_spec=wrapped,
                )
            ],
        )

        resolved = ModelRunnerFL._get_layer_kv_cache_specs(uniform_config)

        assert resolved == {self.ATTN_LAYER: attn_spec}

    def test_nonhybrid_allocation_path_remains_one_tensor_per_plan(self):
        from vllm.v1.kv_cache_interface import (
            KVCacheConfig,
            KVCacheGroupSpec,
            KVCacheTensor,
        )

        config, attn_spec, _ = self._qwen_tp2_config()
        second_attn = "model.layers.2.self_attn"
        nonhybrid_config = KVCacheConfig(
            num_blocks=1,
            kv_cache_tensors=[
                KVCacheTensor(
                    size=attn_spec.page_size_bytes,
                    shared_by=[self.ATTN_LAYER, second_attn],
                )
            ],
            kv_cache_groups=[
                KVCacheGroupSpec(
                    layer_names=[self.ATTN_LAYER, second_attn],
                    kv_cache_spec=attn_spec,
                )
            ],
        )
        runner = self._runner(config, attn_spec, None)
        runner.kv_cache_config = nonhybrid_config

        raw_by_layer = runner._allocate_kv_cache_tensors(nonhybrid_config)

        assert not runner.hybrid_with_attn_and_mamba
        assert raw_by_layer[self.ATTN_LAYER] is raw_by_layer[second_attn]


class TestReserveModularMoeWorkspace:
    def test_reserves_maximum_modular_kernel_requirement(self):
        from vllm_fl.worker.model_runner import _reserve_modular_moe_workspace

        class FakeExperts:
            @staticmethod
            def workspace_shapes(
                tokens,
                intermediate_dim,
                hidden_dim,
                top_k,
                global_num_experts,
                local_num_experts,
                expert_tokens_meta,
                activation,
            ):
                assert (global_num_experts, local_num_experts) == (256, 128)
                assert expert_tokens_meta is None
                assert activation == "silu"
                return (
                    (tokens, top_k, intermediate_dim // 2),
                    (tokens, top_k, max(intermediate_dim, hidden_dim)),
                    (tokens, hidden_dim),
                )

            @staticmethod
            def workspace_dtype(in_dtype):
                assert in_dtype == torch.bfloat16
                return torch.bfloat16

        moe_layer = torch.nn.Module()
        moe_layer.quant_method = MagicMock()
        moe_layer.quant_method.moe_kernel.is_monolithic = False
        moe_layer.quant_method.moe_kernel.impl.fused_experts = FakeExperts()
        moe_layer.moe_config = MagicMock(
            hidden_dim=1024,
            experts_per_token=8,
            num_experts=256,
            in_dtype=torch.bfloat16,
        )
        moe_layer.global_num_experts = 256
        moe_layer.w13_weight = torch.empty(128, 512, 1024, device="meta")
        moe_layer.activation = "silu"

        model = torch.nn.Module()
        model.add_module("moe", moe_layer)
        workspace_manager = MagicMock()

        num_kernels, reserved_bytes = _reserve_modular_moe_workspace(
            model, max_num_tokens=64, workspace_manager=workspace_manager
        )

        assert num_kernels == 1
        assert reserved_bytes == (64 * 8 * 256 + 64 * 8 * 1024) * 2
        workspace_manager.get_simultaneous.assert_called_once_with(
            ((64 * 8 * 256,), torch.bfloat16),
            ((64, 8, 1024), torch.bfloat16),
        )

    def test_skips_monolithic_moe_kernel(self):
        from vllm_fl.worker.model_runner import _reserve_modular_moe_workspace

        moe_layer = torch.nn.Module()
        moe_layer.quant_method = MagicMock()
        moe_layer.quant_method.moe_kernel.is_monolithic = True
        model = torch.nn.Module()
        model.add_module("moe", moe_layer)
        workspace_manager = MagicMock()

        result = _reserve_modular_moe_workspace(
            model, max_num_tokens=64, workspace_manager=workspace_manager
        )

        assert result == (0, 0)
        workspace_manager.get_simultaneous.assert_not_called()


# =============================================================================
# Layer 2: _pad_for_sequence_parallelism Logic Tests
# =============================================================================


class TestPadForSequenceParallelism:
    """Test sequence parallelism padding logic."""

    @pytest.fixture
    def mock_model_runner(self):
        """Create mock model runner for padding tests."""
        from vllm_fl.worker.model_runner import ModelRunnerFL

        mock_runner = MagicMock(spec=ModelRunnerFL)
        mock_runner.vllm_config = MagicMock()
        mock_runner.vllm_config.parallel_config = MagicMock()
        mock_runner.compilation_config = MagicMock()
        mock_runner.compilation_config.pass_config = MagicMock()
        mock_runner._pad_for_sequence_parallelism = (
            ModelRunnerFL._pad_for_sequence_parallelism.__get__(
                mock_runner, ModelRunnerFL
            )
        )
        return mock_runner

    def test_no_padding_when_sp_disabled(self, mock_model_runner):
        """SP disabled should return original token count."""
        mock_model_runner.vllm_config.parallel_config.tensor_parallel_size = 4
        mock_model_runner.compilation_config.pass_config.enable_sp = False

        assert mock_model_runner._pad_for_sequence_parallelism(10) == 10

    def test_no_padding_when_tp_size_1(self, mock_model_runner):
        """TP size 1 means no parallelism, no padding needed."""
        mock_model_runner.vllm_config.parallel_config.tensor_parallel_size = 1
        mock_model_runner.compilation_config.pass_config.enable_sp = True

        assert mock_model_runner._pad_for_sequence_parallelism(10) == 10

    @pytest.mark.parametrize(
        "num_tokens,tp_size,expected",
        [
            (10, 4, 12),  # 10 -> ceil to multiple of 4
            (8, 4, 8),  # 8 already multiple of 4
            (10, 8, 16),  # 10 -> ceil to multiple of 8
            (1, 4, 4),  # 1 -> ceil to multiple of 4
            (15, 8, 16),  # 15 -> ceil to multiple of 8
        ],
    )
    def test_padding_calculation(
        self, mock_model_runner, num_tokens, tp_size, expected
    ):
        """Verify padding rounds up to next multiple of tp_size."""
        mock_model_runner.vllm_config.parallel_config.tensor_parallel_size = tp_size
        mock_model_runner.compilation_config.pass_config.enable_sp = True

        result = mock_model_runner._pad_for_sequence_parallelism(num_tokens)

        assert result == expected
        assert result % tp_size == 0  # Must be divisible


class TestAscendDPMetadataSynchronization:
    """Regression coverage for Ascend's CPU DP metadata protocol."""

    @staticmethod
    def _make_runner(model_runner, dp_size, dp_rank):
        runner = object.__new__(model_runner.ModelRunnerFL)
        runner.parallel_config = SimpleNamespace(
            data_parallel_size=dp_size,
            data_parallel_rank=dp_rank,
        )
        runner.vllm_config = SimpleNamespace(
            parallel_config=runner.parallel_config,
        )
        return runner

    @staticmethod
    def _modes_for_scenario(dp_size, scenario):
        if scenario == "all-full":
            return [CUDAGraphMode.FULL] * dp_size
        if scenario == "all-none":
            return [CUDAGraphMode.NONE] * dp_size
        if scenario == "full-then-none":
            return [CUDAGraphMode.FULL] * max(dp_size - 1, 0) + [CUDAGraphMode.NONE]
        assert scenario == "alternating-full-none"
        return [
            CUDAGraphMode.FULL if rank % 2 == 0 else CUDAGraphMode.NONE
            for rank in range(dp_size)
        ]

    @pytest.mark.parametrize(
        ("dp_size", "dp_rank"),
        [(size, rank) for size in (1, 2, 3, 4, 8) for rank in range(size)],
    )
    @pytest.mark.parametrize(
        "mode_scenario",
        ["all-full", "full-then-none", "alternating-full-none", "all-none"],
    )
    @pytest.mark.parametrize(
        "padding_force",
        ["none", "sequence-parallel", "draft-model"],
    )
    def test_metadata_matrix_is_rank_invariant(
        self,
        monkeypatch,
        dp_size,
        dp_rank,
        mode_scenario,
        padding_force,
    ):
        """Every rank derives the same vector from the same packed metadata."""
        import vllm_fl.worker.model_runner as model_runner

        runner = self._make_runner(model_runner, dp_size, dp_rank)
        tokens_by_rank = [1 if rank % 2 == 0 else rank + 4 for rank in range(dp_size)]
        modes_by_rank = self._modes_for_scenario(dp_size, mode_scenario)
        cpu_group = object()
        monkeypatch.setattr(
            model_runner,
            "get_dp_group",
            lambda: SimpleNamespace(
                cpu_group=cpu_group,
                world_size=dp_size,
                rank_in_group=dp_rank,
            ),
            raising=False,
        )

        all_reduce_calls = []

        def fake_all_reduce(packed_tensor, *, group):
            assert group is cpu_group
            assert packed_tensor.device.type == "cpu"
            assert packed_tensor.dtype == torch.int32
            assert tuple(packed_tensor.shape) == (2, dp_size)
            assert int(packed_tensor[0, dp_rank]) == tokens_by_rank[dp_rank]
            assert int(packed_tensor[1, dp_rank]) == modes_by_rank[dp_rank].value
            packed_tensor[0].copy_(torch.tensor(tokens_by_rank, dtype=torch.int32))
            packed_tensor[1].copy_(
                torch.tensor(
                    [mode.value for mode in modes_by_rank],
                    dtype=torch.int32,
                )
            )
            all_reduce_calls.append(True)

        monkeypatch.setattr(
            model_runner.torch.distributed, "all_reduce", fake_all_reduce
        )

        force_dp_padding = padding_force == "sequence-parallel"
        is_draft_model = padding_force == "draft-model"
        max_tokens, token_vector, synced_mode = runner._sync_metadata_across_dp(
            num_tokens=tokens_by_rank[dp_rank],
            cudagraph_mode=modes_by_rank[dp_rank],
            force_dp_padding=force_dp_padding,
            is_draft_model=is_draft_model,
        )

        assert max_tokens == max(tokens_by_rank)
        if dp_size == 1:
            assert synced_mode is modes_by_rank[0]
            assert token_vector is None
            assert not all_reduce_calls
            return

        expected_mode = CUDAGraphMode(min(mode.value for mode in modes_by_rank))
        should_pad = (
            expected_mode != CUDAGraphMode.NONE or force_dp_padding or is_draft_model
        )
        expected_vector = (
            [max(tokens_by_rank)] * dp_size if should_pad else tokens_by_rank
        )
        assert synced_mode is expected_mode
        assert token_vector.tolist() == expected_vector
        assert all_reduce_calls == [True]

    @pytest.mark.parametrize(
        ("dp_size", "dp_rank"),
        [(size, rank) for size in (2, 3, 4, 8) for rank in range(size)],
    )
    @pytest.mark.parametrize("mismatch", ["world-size", "rank"])
    def test_rejects_active_dp_group_topology_mismatch(
        self,
        monkeypatch,
        dp_size,
        dp_rank,
        mismatch,
    ):
        import vllm_fl.worker.model_runner as model_runner

        runner = self._make_runner(model_runner, dp_size, dp_rank)
        group_world_size = dp_size + 1 if mismatch == "world-size" else dp_size
        group_rank = (dp_rank + 1) % dp_size if mismatch == "rank" else dp_rank
        monkeypatch.setattr(
            model_runner,
            "get_dp_group",
            lambda: SimpleNamespace(
                cpu_group=object(),
                world_size=group_world_size,
                rank_in_group=group_rank,
            ),
            raising=False,
        )
        all_reduce = MagicMock()
        monkeypatch.setattr(model_runner.torch.distributed, "all_reduce", all_reduce)

        with pytest.raises(RuntimeError, match="DP topology mismatch"):
            runner._sync_metadata_across_dp(
                num_tokens=dp_rank + 1,
                cudagraph_mode=CUDAGraphMode.NONE,
            )

        all_reduce.assert_not_called()

    @pytest.mark.parametrize(
        ("dp_size", "dp_rank"),
        [(size, rank) for size in (2, 3, 4, 8) for rank in range(size)],
    )
    @pytest.mark.parametrize(
        "mode_scenario",
        ["all-full", "full-then-none", "alternating-full-none", "all-none"],
    )
    @pytest.mark.parametrize("enable_sp", [False, True])
    def test_determine_redispatches_synced_mode_on_every_dp_rank(
        self,
        monkeypatch,
        dp_size,
        dp_rank,
        mode_scenario,
        enable_sp,
    ):
        import vllm_fl.worker.model_runner as model_runner

        runner = self._make_runner(model_runner, dp_size, dp_rank)
        runner.parallel_config.tensor_parallel_size = 2
        runner.uniform_decode_query_len = 1
        runner.model_config = SimpleNamespace(is_encoder_decoder=False)
        runner.input_batch = SimpleNamespace(lora_id_to_lora_request={})
        runner.compilation_config = SimpleNamespace(
            pass_config=SimpleNamespace(enable_sp=enable_sp)
        )
        runner.vllm_config.observability_config = SimpleNamespace(
            cudagraph_metrics=False
        )
        runner._pad_for_sequence_parallelism = MagicMock(return_value=8)

        modes_by_rank = self._modes_for_scenario(dp_size, mode_scenario)
        local_mode = modes_by_rank[dp_rank]
        synced_mode = CUDAGraphMode(min(mode.value for mode in modes_by_rank))
        raw_tokens = [8 + rank for rank in range(dp_size)]
        should_pad = synced_mode != CUDAGraphMode.NONE or enable_sp
        synced_tokens = torch.tensor(
            [max(raw_tokens)] * dp_size if should_pad else raw_tokens,
            dtype=torch.int32,
        )
        first_descriptor = model_runner.BatchDescriptor(num_tokens=8)
        synced_descriptor = model_runner.BatchDescriptor(
            num_tokens=int(synced_tokens[dp_rank].item())
        )
        runner.cudagraph_dispatcher = SimpleNamespace(
            dispatch=MagicMock(
                side_effect=[
                    (local_mode, first_descriptor),
                    (synced_mode, synced_descriptor),
                ]
            )
        )
        runner._sync_metadata_across_dp = MagicMock(
            return_value=(max(raw_tokens), synced_tokens, synced_mode)
        )

        monkeypatch.setattr(
            model_runner,
            "current_platform",
            SimpleNamespace(device_type="npu"),
        )
        upstream_coordinate = MagicMock()
        monkeypatch.setattr(
            model_runner,
            "coordinate_batch_across_dp",
            upstream_coordinate,
        )

        mode, descriptor, should_ubatch, tokens_across_dp, _ = (
            runner._determine_batch_execution_and_padding(
                num_tokens=5,
                num_reqs=5,
                num_scheduled_tokens_np=np.ones(5, dtype=np.int32),
                max_num_scheduled_tokens=1,
                use_cascade_attn=False,
            )
        )

        # The helper receives the first dispatch's padded count, but its force
        # flag is rank-invariant and never derived from ``local_mode``.
        runner._sync_metadata_across_dp.assert_called_once_with(
            num_tokens=8,
            cudagraph_mode=local_mode,
            force_dp_padding=enable_sp,
        )
        upstream_coordinate.assert_not_called()
        assert runner.cudagraph_dispatcher.dispatch.call_args_list[1].kwargs[
            "valid_modes"
        ] == {synced_mode}
        assert mode is synced_mode
        assert descriptor is synced_descriptor
        assert should_ubatch is False
        assert tokens_across_dp is synced_tokens

    def test_non_npu_determine_keeps_upstream_dp_coordination(
        self,
        monkeypatch,
    ):
        import vllm_fl.worker.model_runner as model_runner

        runner = self._make_runner(model_runner, dp_size=2, dp_rank=1)
        runner.parallel_config.tensor_parallel_size = 1
        runner.parallel_config.num_ubatches = 1
        runner.uniform_decode_query_len = 1
        runner.model_config = SimpleNamespace(is_encoder_decoder=False)
        runner.input_batch = SimpleNamespace(lora_id_to_lora_request={})
        runner.compilation_config = SimpleNamespace(
            pass_config=SimpleNamespace(enable_sp=False)
        )
        runner.vllm_config.observability_config = SimpleNamespace(
            cudagraph_metrics=False
        )
        runner._pad_for_sequence_parallelism = MagicMock(return_value=3)
        descriptor = model_runner.BatchDescriptor(num_tokens=3)
        runner.cudagraph_dispatcher = SimpleNamespace(
            dispatch=MagicMock(return_value=(CUDAGraphMode.NONE, descriptor))
        )
        runner._sync_metadata_across_dp = MagicMock()
        upstream_coordinate = MagicMock(
            return_value=(False, torch.tensor([2, 3]), CUDAGraphMode.NONE.value)
        )

        monkeypatch.setattr(
            model_runner,
            "current_platform",
            SimpleNamespace(device_type="cuda"),
        )
        monkeypatch.setattr(
            model_runner,
            "coordinate_batch_across_dp",
            upstream_coordinate,
        )

        runner._determine_batch_execution_and_padding(
            num_tokens=3,
            num_reqs=3,
            num_scheduled_tokens_np=np.ones(3, dtype=np.int32),
            max_num_scheduled_tokens=1,
            use_cascade_attn=False,
        )

        runner._sync_metadata_across_dp.assert_not_called()
        upstream_coordinate.assert_called_once_with(
            num_tokens_unpadded=3,
            parallel_config=runner.parallel_config,
            allow_microbatching=True,
            num_tokens_padded=3,
            uniform_decode=True,
            cudagraph_mode=CUDAGraphMode.NONE.value,
        )


# =============================================================================
# Layer 2: _get_positions Routing Tests
# =============================================================================


class TestGetPositions:
    """Test position retrieval for different position encoding schemes."""

    @pytest.fixture
    def mock_model_runner(self):
        """Create mock model runner for position tests."""
        from vllm_fl.worker.model_runner import ModelRunnerFL

        mock_runner = MagicMock(spec=ModelRunnerFL)

        # Standard positions buffer (used directly, not .gpu)
        mock_runner.positions = torch.arange(100)

        # MRoPE positions (3D for temporal, height, width)
        mock_runner.mrope_positions = MagicMock()
        mock_runner.mrope_positions.gpu = torch.arange(300).reshape(3, 100)

        # XDRoPE positions (2D)
        mock_runner.xdrope_positions = MagicMock()
        mock_runner.xdrope_positions.gpu = torch.arange(200).reshape(2, 100)

        mock_runner.uses_mrope = False
        mock_runner.uses_xdrope_dim = 0

        mock_runner._get_positions = ModelRunnerFL._get_positions.__get__(
            mock_runner, ModelRunnerFL
        )
        return mock_runner

    def test_standard_positions_with_int(self, mock_model_runner):
        """Standard RoPE: integer returns first N positions."""
        result = mock_model_runner._get_positions(10)
        torch.testing.assert_close(result, torch.arange(10))

    def test_standard_positions_with_indices(self, mock_model_runner):
        """Standard RoPE: tensor indices for selective position lookup."""
        indices = torch.tensor([0, 5, 10, 15])
        result = mock_model_runner._get_positions(indices)
        expected = mock_model_runner.positions[indices]
        torch.testing.assert_close(result, expected)

    def test_mrope_returns_3d_positions(self, mock_model_runner):
        """MRoPE (Qwen2-VL): returns [3, num_tokens] positions."""
        mock_model_runner.uses_mrope = True

        result = mock_model_runner._get_positions(10)

        expected = mock_model_runner.mrope_positions.gpu[:, :10]
        assert result.shape == (3, 10)
        torch.testing.assert_close(result, expected)

    def test_xdrope_returns_2d_positions(self, mock_model_runner):
        """XDRoPE: returns [2, num_tokens] positions."""
        mock_model_runner.uses_xdrope_dim = 64

        result = mock_model_runner._get_positions(10)

        expected = mock_model_runner.xdrope_positions.gpu[:, :10]
        assert result.shape == (2, 10)
        torch.testing.assert_close(result, expected)


class TestGraphRuntimeDelegation:
    def test_model_forward_delegates_graph_lifecycle(self):
        from vllm_fl.worker.model_runner import ModelRunnerFL

        model_output = object()
        runner = MagicMock(spec=ModelRunnerFL)
        runner.model = MagicMock(return_value=model_output)
        runner.graph_runtime = MagicMock()
        runner.vllm_config = MagicMock()
        runner._model_forward = ModelRunnerFL._model_forward.__get__(
            runner, ModelRunnerFL
        )
        positions = torch.zeros((1,), dtype=torch.int64)

        result = runner._model_forward(positions=positions)

        assert result is model_output
        runner.model.assert_called_once_with(
            input_ids=None,
            positions=positions,
            intermediate_tensors=None,
            inputs_embeds=None,
        )
        runner.graph_runtime.after_model_forward.assert_called_once_with(
            runner.vllm_config
        )
