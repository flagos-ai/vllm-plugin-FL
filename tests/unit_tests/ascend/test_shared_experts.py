from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
    SharedExperts,
    SharedExpertsOrder,
)

from vllm_fl.dispatch.backends.vendor.ascend.impl.shared_experts import (
    AscendSharedExperts,
)


def make_shared_experts(enabled: bool = True) -> AscendSharedExperts:
    shared = object.__new__(AscendSharedExperts)
    shared._ascend_multistream_enabled = enabled
    shared._stream = MagicMock() if enabled else None
    shared._moe_config = SimpleNamespace(
        disable_inplace=True,
        moe_parallel_config=SimpleNamespace(
            enable_eplb=False,
            all2all_backend="allgather_reducescatter",
            use_fi_nvl_two_sided_kernels=False,
        ),
    )
    shared._quant_method = SimpleNamespace(mk_owns_shared_expert=False)
    return shared


def test_multistream_switch_selects_auxiliary_order():
    enabled = make_shared_experts(True)
    disabled = make_shared_experts(False)
    hidden_states = MagicMock()

    assert (
        enabled._determine_shared_experts_order(hidden_states)
        == SharedExpertsOrder.MULTI_STREAM_OVERLAPPED
    )
    assert (
        disabled._determine_shared_experts_order(hidden_states)
        == SharedExpertsOrder.NO_OVERLAP
    )


def test_internal_kernel_order_takes_precedence():
    shared = make_shared_experts(True)
    shared._quant_method.mk_owns_shared_expert = True

    assert (
        shared._determine_shared_experts_order(MagicMock())
        == SharedExpertsOrder.MK_INTERNAL_OVERLAPPED
    )


@patch("torch.npu.current_stream")
def test_sync_records_input_and_orders_aux_stream(mock_current_stream):
    shared = make_shared_experts(True)
    shared_input = MagicMock()
    current = MagicMock()
    mock_current_stream.return_value = current

    shared.maybe_sync_shared_experts_stream(shared_input)

    shared_input.record_stream.assert_called_once_with(shared._stream)
    shared._stream.wait_stream.assert_called_once_with(current)


@patch("torch.npu.current_stream")
@patch("torch.npu.stream")
def test_aux_stream_executes_layer_and_joins_default(
    mock_stream_context, mock_current_stream
):
    shared = make_shared_experts(True)
    shared._layer = MagicMock(return_value=torch.tensor([1.0]))
    shared_input = torch.tensor([2.0])
    current = MagicMock()
    mock_current_stream.return_value = current
    mock_stream_context.return_value = nullcontext()

    output = shared._run_in_aux_stream(shared_input)

    torch.testing.assert_close(output, torch.tensor([1.0]))
    shared._layer.assert_called_once_with(shared_input)
    mock_stream_context.assert_called_once_with(shared._stream)
    current.wait_stream.assert_called_once_with(shared._stream)


@patch.object(SharedExperts, "__init__", return_value=None)
@patch("torch.npu.Stream")
@patch(
    "vllm_fl.dispatch.backends.vendor.ascend.impl.shared_experts."
    "_SHARED_EXPERTS_STREAM",
    None,
)
def test_constructor_reuses_one_stream_for_all_layers(mock_stream, _mock_parent):
    enabled = AscendSharedExperts(multistream_enabled=True)
    enabled_again = AscendSharedExperts(multistream_enabled=True)
    disabled = AscendSharedExperts(multistream_enabled=False)

    assert enabled._stream is mock_stream.return_value
    assert enabled_again._stream is enabled._stream
    assert disabled._stream is None
    mock_stream.assert_called_once_with()
