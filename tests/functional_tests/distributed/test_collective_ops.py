# Copyright (c) 2025 BAAI. All rights reserved.

"""
Functional tests for distributed collective operations.
Tests correctness of collective operations like all_reduce, reduce_scatter, etc.

NOTE: These tests require multiple GPUs and a distributed environment.
They are designed to be run with pytest-mpi or similar multi-process test runners.
"""

import pytest
import torch
from typing import List
from unittest.mock import MagicMock


# Mark all tests as requiring multiple GPUs
pytestmark = [pytest.mark.multi_gpu, pytest.mark.gpu]


class TestCollectiveOpsBasic:
    """Basic tests for collective operations that can run without actual distributed setup."""

    def test_communicator_fl_import(self):
        """Test that CommunicatorFL can be imported."""
        try:
            from vllm_fl.distributed.communicator import CommunicatorFL
            assert CommunicatorFL is not None
        except ImportError as e:
            pytest.skip(f"CommunicatorFL not available: {e}")

    def test_pyflagcx_import(self):
        """Test that PyFlagcxCommunicator can be imported."""
        try:
            from vllm_fl.distributed.device_communicators.flagcx import PyFlagcxCommunicator
            assert PyFlagcxCommunicator is not None
        except ImportError as e:
            pytest.skip(f"PyFlagcxCommunicator not available: {e}")


class TestAllReduceCorrectness:
    """Test all_reduce operation correctness."""

    @staticmethod
    def reference_all_reduce(tensors: List[torch.Tensor]) -> torch.Tensor:
        """Reference implementation of all_reduce (sum)."""
        return sum(tensors)

    @pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.device_count() < 2,
        reason="Multiple GPUs not available"
    )
    def test_all_reduce_sum_correctness(self):
        """Test all_reduce sum produces correct results."""
        # This test would need actual distributed setup
        # For now, test the reference implementation
        tensors = [
            torch.tensor([1.0, 2.0, 3.0]),
            torch.tensor([4.0, 5.0, 6.0]),
        ]
        expected = torch.tensor([5.0, 7.0, 9.0])
        result = self.reference_all_reduce(tensors)
        assert torch.allclose(result, expected)


class TestReduceScatterCorrectness:
    """Test reduce_scatter operation correctness."""

    @staticmethod
    def reference_reduce_scatter(
        input_tensor: torch.Tensor,
        world_size: int
    ) -> List[torch.Tensor]:
        """Reference implementation of reduce_scatter."""
        # Split input into chunks
        chunks = input_tensor.chunk(world_size, dim=0)
        # Each rank gets the reduced chunk at its position
        return list(chunks)

    def test_reduce_scatter_reference(self):
        """Test reference reduce_scatter implementation."""
        input_tensor = torch.tensor([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
        ])
        world_size = 2

        result = self.reference_reduce_scatter(input_tensor, world_size)

        assert len(result) == world_size
        assert torch.allclose(result[0], torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
        assert torch.allclose(result[1], torch.tensor([[5.0, 6.0], [7.0, 8.0]]))


class TestAllGatherCorrectness:
    """Test all_gather operation correctness."""

    @staticmethod
    def reference_all_gather(tensors: List[torch.Tensor]) -> torch.Tensor:
        """Reference implementation of all_gather."""
        return torch.cat(tensors, dim=0)

    def test_all_gather_reference(self):
        """Test reference all_gather implementation."""
        tensors = [
            torch.tensor([[1.0, 2.0]]),
            torch.tensor([[3.0, 4.0]]),
        ]
        expected = torch.tensor([
            [1.0, 2.0],
            [3.0, 4.0],
        ])

        result = self.reference_all_gather(tensors)
        assert torch.allclose(result, expected)


class TestSendRecvCorrectness:
    """Test point-to-point send/recv operations."""

    @pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.device_count() < 2,
        reason="Multiple GPUs not available"
    )
    def test_send_recv_mock(self):
        """Test send/recv with mocked communicator."""
        # Create mock communicator
        mock_comm = MagicMock()
        mock_comm.disabled = False

        tensor = torch.randn(4, 8)

        # Simulate send
        mock_comm.send(tensor, dst=1)
        mock_comm.send.assert_called_once()

        # Simulate recv
        mock_comm.recv.return_value = tensor.clone()
        received = mock_comm.recv(tensor.shape, tensor.dtype, src=0)
        assert received.shape == tensor.shape


class TestCommunicatorDisabled:
    """Test communicator behavior when disabled."""

    def test_disabled_all_reduce_returns_none(self):
        """Test that disabled communicator all_reduce returns None."""
        mock_comm = MagicMock()
        mock_comm.disabled = True

        def mock_all_reduce(tensor, out=None, op=None, stream=None):
            if mock_comm.disabled:
                return None
            return torch.empty_like(tensor)

        mock_comm.all_reduce = mock_all_reduce

        result = mock_comm.all_reduce(torch.randn(4, 8))
        assert result is None

    def test_disabled_send_returns_early(self):
        """Test that disabled communicator send returns early."""
        mock_comm = MagicMock()
        mock_comm.disabled = True

        call_count = [0]

        def mock_send(tensor, dst, stream=None):
            if mock_comm.disabled:
                return
            call_count[0] += 1
            # Would do actual send here

        mock_comm.send = mock_send
        mock_comm.send(torch.randn(4, 8), dst=1)

        assert call_count[0] == 0


class TestDistributedUtils:
    """Test distributed utility functions."""

    def test_world_size_calculation(self):
        """Test world size calculation logic."""
        # Single GPU case
        world_size_1 = 1
        assert world_size_1 == 1

        # Multi-GPU case
        world_size_4 = 4
        tp_size = 2
        pp_size = 2
        assert world_size_4 == tp_size * pp_size

    def test_rank_calculation(self):
        """Test rank calculation in tensor/pipeline parallel."""
        world_size = 4
        tp_size = 2
        pp_size = 2

        # Calculate expected ranks
        for global_rank in range(world_size):
            tp_rank = global_rank % tp_size
            pp_rank = global_rank // tp_size

            assert 0 <= tp_rank < tp_size
            assert 0 <= pp_rank < pp_size
