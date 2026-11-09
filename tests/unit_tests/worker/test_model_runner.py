# Copyright (c) 2025 BAAI. All rights reserved.

"""
Tests for model runner module.

This module follows a layered testing strategy:
- Layer 1: Pure functions and data classes (no external dependencies)
- Layer 2: Methods with mocked dependencies
- Layer 3: Integration tests (in functional_tests/, requires GPU)

Note: These tests require vllm >= 0.13.0 with full installation.
"""

import ast
import importlib.util
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

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


# Full-runner tests retain their dependency guard; RNG ownership tests run independently.
requires_model_runner = pytest.mark.skipif(
    not has_vllm_model_runner(), reason="vllm_fl.worker.model_runner not available"
)


@requires_model_runner
def test_musa_piecewise_capture_sizes_respect_live_graph_limit(monkeypatch):
    from vllm.config import CUDAGraphMode

    from vllm_fl.worker import model_runner as model_runner_module
    from vllm_fl.worker.model_runner import ModelRunnerFL

    capture_sizes = [1, 2, 4] + list(range(8, 256, 8)) + list(range(256, 513, 16))
    compilation_config = SimpleNamespace(
        cudagraph_mode=CUDAGraphMode.PIECEWISE,
        cudagraph_capture_sizes=capture_sizes,
        max_cudagraph_capture_size=512,
    )
    dispatcher = MagicMock()
    dispatcher.cudagraph_mode = CUDAGraphMode.PIECEWISE
    dispatcher.cudagraph_keys = {
        CUDAGraphMode.PIECEWISE: set(),
        CUDAGraphMode.FULL: set(),
    }
    dispatcher.get_capture_descs.return_value = [
        (CUDAGraphMode.PIECEWISE, [object()] * len(capture_sizes))
    ]
    runner = SimpleNamespace(
        compilation_config=compilation_config,
        parallel_config=SimpleNamespace(tensor_parallel_size=2),
        cudagraph_dispatcher=dispatcher,
        uniform_decode_query_len=1,
    )

    monkeypatch.setattr(
        model_runner_module,
        "current_platform",
        SimpleNamespace(device_type="musa"),
    )
    monkeypatch.setattr(
        model_runner_module.GraphWrapper,
        "_all_instances",
        [SimpleNamespace(runtime_mode=CUDAGraphMode.PIECEWISE) for _ in range(86)],
    )

    ModelRunnerFL._limit_musa_piecewise_capture_sizes(runner)

    assert compilation_config.cudagraph_capture_sizes == capture_sizes[:23]
    assert compilation_config.max_cudagraph_capture_size == 160
    dispatcher.initialize_cudagraph_keys.assert_called_once_with(
        CUDAGraphMode.PIECEWISE, 1
    )


@requires_model_runner
@pytest.mark.parametrize("disable_full", [False, True])
@pytest.mark.parametrize("dp_size", [1, 2])
def test_full_graph_exclusion_survives_dp_redispatch(
    monkeypatch, disable_full, dp_size
):
    from vllm.config import CUDAGraphMode

    from vllm_fl.worker import model_runner as module

    dispatcher = MagicMock()
    mode = CUDAGraphMode.NONE if disable_full else CUDAGraphMode.FULL
    dispatcher.dispatch.return_value = (mode, SimpleNamespace(num_tokens=1))
    parallel = SimpleNamespace(data_parallel_size=dp_size, data_parallel_rank=0)
    runner = SimpleNamespace(
        _is_uniform_decode=module.ModelRunnerFL._is_uniform_decode,
        uniform_decode_query_len=1,
        model_config=SimpleNamespace(is_encoder_decoder=False),
        input_batch=SimpleNamespace(lora_id_to_lora_request={}),
        _pad_for_sequence_parallelism=lambda tokens: tokens,
        cudagraph_dispatcher=dispatcher,
        compilation_config=SimpleNamespace(
            pass_config=SimpleNamespace(enable_sp=False)
        ),
        parallel_config=parallel,
        vllm_config=SimpleNamespace(
            parallel_config=parallel,
            observability_config=SimpleNamespace(cudagraph_metrics=False),
        ),
    )
    monkeypatch.setattr(
        module,
        "coordinate_batch_across_dp",
        lambda **kwargs: (False, torch.ones(dp_size, dtype=torch.int32), mode.value),
    )
    result = module.ModelRunnerFL._determine_batch_execution_and_padding(
        runner,
        num_tokens=1,
        num_reqs=1,
        num_scheduled_tokens_np=np.array([1]),
        max_num_scheduled_tokens=1,
        use_cascade_attn=False,
        disable_full=disable_full,
    )
    assert result[0] == mode
    assert dispatcher.dispatch.call_count == dp_size
    for call in dispatcher.dispatch.call_args_list:
        assert call.kwargs["invalid_modes"] == (
            {CUDAGraphMode.FULL} if disable_full else None
        )


# =============================================================================
# Layer 1: ExecuteModelState Data Structure Tests
# =============================================================================


@requires_model_runner
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


@requires_model_runner
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


# =============================================================================
# Layer 2: _pad_for_sequence_parallelism Logic Tests
# =============================================================================


@requires_model_runner
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


# =============================================================================
# Layer 2: _get_positions Routing Tests
# =============================================================================


@requires_model_runner
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


class _OffsetGenerator:
    def __init__(self, offset=0):
        self.offset = offset
        self.restored = []

    def get_offset(self):
        return self.offset

    def set_offset(self, offset):
        self.restored.append(offset)
        self.offset = offset

    def draw(self, increment):
        before = self.offset
        self.offset += increment
        return before


class TestSamplingRngOwnership:
    """Request-local RNG ownership, including the production sampling block."""

    @pytest.fixture(scope="class")
    def preserve(self):
        path = Path(__file__).resolve().parents[3] / "vllm_fl/worker/sampling_rng.py"
        spec = importlib.util.spec_from_file_location("sampling_rng_under_test", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.preserve_discarded_generator_offsets

    @pytest.mark.parametrize("increment", [0, 4, 256, 31040, 124160])
    def test_discarded_request_restores_exact_entry_offset(self, preserve, increment):
        generator = _OffsetGenerator(72)
        with preserve({0: generator}, [0]):
            assert generator.draw(increment) == 72
        assert generator.offset == 72
        assert generator.restored == [72]
        assert generator.draw(increment) == 72

    def test_mixed_valid_discarded_and_unseeded_rows(self, preserve):
        (discarded, valid) = (_OffsetGenerator(8), _OffsetGenerator(20))
        with preserve({0: discarded, 1: valid}, [0, 2]):
            discarded.draw(31040)
            valid.draw(124160)
        assert discarded.offset == 8
        assert valid.offset == 124180
        assert valid.restored == []

    def test_no_discard_leaves_valid_generator_untouched(self, preserve):
        generator = _OffsetGenerator(12)
        with preserve({0: generator}, []):
            generator.draw(31040)
        assert generator.offset == 31052
        assert generator.restored == []

    def test_global_rng_rows_need_no_offset_api(self, preserve):
        with preserve({}, [0, 1]):
            pass

    def test_sampler_error_restores_discarded_offset_without_retry(self, preserve):
        generator = _OffsetGenerator(28)
        calls = []
        with (
            pytest.raises(RuntimeError, match="sampler failed"),
            preserve({0: generator}, [0]),
        ):
            calls.append(1)
            generator.draw(31040)
            raise RuntimeError("sampler failed")
        assert calls == [1]
        assert generator.offset == 28

    def test_original_generator_is_restored_if_mapping_changes(self, preserve):
        (original, replacement) = (_OffsetGenerator(16), _OffsetGenerator(40))
        generators = {0: original}
        with preserve(generators, [0]):
            original.draw(31040)
            generators[0] = replacement
        assert original.offset == 16
        assert replacement.offset == 40
        assert replacement.restored == []

    def test_repeated_calls_do_not_reuse_another_requests_state(self, preserve):
        for offset in [0, 100, 200]:
            generator = _OffsetGenerator(offset)
            with preserve({0: generator}, [0]):
                generator.draw(31040)
            assert generator.offset == offset

    def test_snapshot_failure_prevents_sampling(self, preserve):
        class Broken(_OffsetGenerator):
            def get_offset(self):
                raise RuntimeError("offset unavailable")

        sampled = False
        with (
            pytest.raises(RuntimeError, match="offset unavailable"),
            preserve({0: Broken()}, [0]),
        ):
            sampled = True
        assert not sampled

    @pytest.mark.parametrize("fail", [False, True])
    def test_real_runner_sampling_block_restores_only_discarded_row(
        self, preserve, fail
    ):
        source = (
            Path(__file__).resolve().parents[3] / "vllm_fl/worker/model_runner.py"
        ).read_text()
        tree = ast.parse(source)
        sample = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "sample_tokens"
        )
        blocks = [
            node
            for node in ast.walk(sample)
            if isinstance(node, ast.With)
            and "preserve_discarded_generator_offsets" in ast.unparse(node)
        ]
        assert len(blocks) == 1
        (discarded, valid) = (_OffsetGenerator(24), _OffsetGenerator(44))
        calls = []

        def sampler(*args):
            calls.append(args)
            discarded.draw(31040)
            valid.draw(31040)
            if fail:
                raise RuntimeError("sampler failed")
            return "sampled"

        owner = SimpleNamespace(
            input_batch=SimpleNamespace(
                num_reqs=2, generators={0: discarded, 1: valid}
            ),
            discard_request_mask=SimpleNamespace(np=np.array([True, False])),
            _sample=sampler,
        )
        env = dict(
            self=owner,
            logits=object(),
            spec_decode_metadata=None,
            record_function_or_nullcontext=lambda *args: nullcontext(),
            preserve_discarded_generator_offsets=preserve,
            np=np,
        )
        code = compile(
            ast.fix_missing_locations(ast.Module(body=blocks, type_ignores=[])),
            "real-runner-sampling-block",
            "exec",
        )
        if fail:
            with pytest.raises(RuntimeError, match="sampler failed"):
                exec(code, env)
        else:
            exec(code, env)
            assert env["sampler_output"] == "sampled"
        assert len(calls) == 1
        assert discarded.offset == 24
        assert valid.offset == 31084
        bookkeeping = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "_bookkeeping_sync"
        )
        assert not any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and (node.func.attr == "set_offset")
            for node in ast.walk(bookkeeping)
        ), "Bookkeeping must not rewind a restored generator a second time"
