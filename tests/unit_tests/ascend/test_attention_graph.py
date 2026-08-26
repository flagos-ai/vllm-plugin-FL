from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from vllm.config import CUDAGraphMode


def _import_attention():
    try:
        from vllm_fl.dispatch.backends.vendor.ascend.impl import attention
    except (AttributeError, ImportError) as exc:
        pytest.skip(f"Ascend attention dependencies are unavailable: {exc}")
    return attention


def _import_gdn():
    try:
        from vllm_fl.dispatch.backends.vendor.ascend.impl import gdn
    except (AttributeError, ImportError) as exc:
        pytest.skip(f"Ascend GDN dependencies are unavailable: {exc}")
    return gdn


def _config(
    *,
    pa_shape_list=None,
    mode=CUDAGraphMode.FULL_DECODE_ONLY,
    speculative_config=None,
):
    additional_config = {}
    if pa_shape_list is not None:
        additional_config["pa_shape_list"] = pa_shape_list
    return SimpleNamespace(
        additional_config=additional_config,
        compilation_config=SimpleNamespace(cudagraph_mode=mode),
        speculative_config=speculative_config,
    )


@pytest.mark.parametrize(
    ("config", "runtime_shape", "expected"),
    [
        (_config(), 32, False),
        (_config(pa_shape_list=[]), 32, False),
        (_config(pa_shape_list=[16, 32]), 32, True),
        (_config(pa_shape_list=[16, 32]), 8, False),
        (_config(pa_shape_list=[32], mode=CUDAGraphMode.FULL), 32, False),
        (_config(pa_shape_list=[32], speculative_config=object()), 32, False),
    ],
)
def test_paged_attention_selector_matches_target_version(
    config,
    runtime_shape,
    expected,
):
    from vllm_fl.dispatch.backends.vendor.ascend.impl.attention_utils import (
        using_paged_attention,
    )

    assert using_paged_attention(runtime_shape, config) is expected


def test_paged_attention_is_disabled_on_a5(monkeypatch):
    from vllm_fl.dispatch.backends.vendor.ascend.impl import attention_utils

    monkeypatch.setattr(attention_utils, "_is_ascend_a5", lambda: True)

    assert not attention_utils.using_paged_attention(
        32,
        _config(pa_shape_list=[32]),
    )


def test_decode_dispatch_defaults_to_fia(monkeypatch):
    attention = _import_attention()
    impl = object.__new__(attention.AscendAttentionBackendImpl)
    impl.vllm_config = _config()
    impl.sliding_window = None
    impl.forward_fused_infer_attention = MagicMock(return_value="fia")
    impl.forward_paged_attention = MagicMock(return_value="pa")
    metadata = SimpleNamespace(
        attn_state=attention.AscendAttentionState.DecodeOnly,
    )
    query = SimpleNamespace(shape=(32, 1, 1))

    result = impl.forward_impl(
        query, object(), object(), (), metadata, object(), "layer.0"
    )

    assert result == "fia"
    impl.forward_fused_infer_attention.assert_called_once()
    impl.forward_paged_attention.assert_not_called()


def test_decode_dispatch_uses_pa_for_explicit_shape(monkeypatch):
    attention = _import_attention()
    impl = object.__new__(attention.AscendAttentionBackendImpl)
    impl.vllm_config = _config(pa_shape_list=[32])
    impl.sliding_window = None
    impl.forward_fused_infer_attention = MagicMock(return_value="fia")
    impl.forward_paged_attention = MagicMock(return_value="pa")
    metadata = SimpleNamespace(
        attn_state=attention.AscendAttentionState.DecodeOnly,
    )
    query = SimpleNamespace(shape=(32, 1, 1))

    result = impl.forward_impl(
        query, object(), object(), (), metadata, object(), "layer.0"
    )

    assert result == "pa"
    impl.forward_paged_attention.assert_called_once()
    impl.forward_fused_infer_attention.assert_not_called()


def test_decode_fia_keeps_target_attention_mask(monkeypatch):
    attention = _import_attention()
    builder = object.__new__(attention.AscendAttentionMetadataBuilder)
    builder.model_config = SimpleNamespace(
        runner_type="generate",
        use_mla=False,
    )
    mask = object()
    mask_builder = MagicMock()
    mask_builder.get_splitfuse_attn_mask.return_value = mask
    monkeypatch.setattr(builder, "_get_mask_builder", lambda: mask_builder)

    result = builder._make_attention_mask(
        attention.AscendAttentionState.DecodeOnly
    )

    assert result is mask
    mask_builder.get_splitfuse_attn_mask.assert_called_once_with()


def test_fia_graph_update_refreshes_runtime_metadata(monkeypatch):
    attention = _import_attention()
    event = MagicMock()
    handle = object()
    output = object()
    softmax_lse = object()
    params = (
        "layer.0",
        object(),
        object(),
        object(),
        object(),
        None,
        128,
        2,
        4,
        0.5,
        output,
        softmax_lse,
        3,
        2**31 - 1,
        2**31 - 1,
    )
    graph_params = SimpleNamespace(
        attention_params={32: [params]},
        handles={32: [handle]},
        events={32: [event]},
        workspaces={32: "workspace"},
    )
    metadata = attention.AscendMetadata(
        seq_lens_list=[101, 202],
        actual_seq_lengths_q=[1, 2],
        block_tables="runtime-block-table",
    )
    forward_context = SimpleNamespace(attn_metadata={"layer.0": metadata})
    out = MagicMock()
    monkeypatch.setattr(attention, "get_ascend_graph_params", lambda: graph_params)
    monkeypatch.setattr(attention, "using_paged_attention", lambda *_: False)
    monkeypatch.setattr(attention.torch.npu, "stream", lambda _: nullcontext())
    monkeypatch.setattr(
        attention.torch.npu,
        "graph_task_update_begin",
        MagicMock(),
    )
    monkeypatch.setattr(
        attention.torch.npu,
        "graph_task_update_end",
        MagicMock(),
    )
    monkeypatch.setattr(
        attention.torch_npu.npu_fused_infer_attention_score,
        "out",
        out,
    )

    attention.AscendAttentionBackendImpl.update_graph_params(
        object(), forward_context, 32, _config()
    )

    assert out.call_count == 1
    kwargs = out.call_args.kwargs
    assert kwargs["block_table"] == "runtime-block-table"
    assert kwargs["actual_seq_lengths"] == [1, 2]
    assert kwargs["actual_seq_lengths_kv"] == [101, 202]
    assert kwargs["workspace"] == "workspace"
    event.record.assert_called_once()


def test_graph_update_uses_zip_semantics_for_inconsistent_task_state(
    monkeypatch,
):
    attention = _import_attention()
    graph_params = SimpleNamespace(
        attention_params={32: [("layer.0",)]},
        handles={32: []},
        events={32: [object()]},
        workspaces={32: None},
    )
    monkeypatch.setattr(attention, "get_ascend_graph_params", lambda: graph_params)
    monkeypatch.setattr(attention.torch.npu, "stream", lambda _: nullcontext())

    attention.AscendAttentionBackendImpl.update_graph_params(
        object(),
        SimpleNamespace(attn_metadata={}),
        32,
        _config(),
    )


@pytest.mark.parametrize(
    "runtime_metadata",
    [
        object(),
        {"linear_attn": object()},
        {"layer.7": object()},
    ],
)
def test_graph_update_skips_unknown_runtime_metadata(
    monkeypatch, runtime_metadata
):
    attention = _import_attention()
    param = (
        "layer.7",
        object(),
        object(),
        object(),
        object(),
        None,
        128,
        2,
        4,
        0.5,
        object(),
        object(),
        3,
        2**31 - 1,
        2**31 - 1,
    )
    graph_params = SimpleNamespace(
        attention_params={32: [param]},
        handles={32: [object()]},
        events={32: [object()]},
        workspaces={32: None},
    )
    monkeypatch.setattr(attention, "get_ascend_graph_params", lambda: graph_params)
    monkeypatch.setattr(attention, "using_paged_attention", lambda *_: False)
    monkeypatch.setattr(attention.torch.npu, "stream", lambda _: nullcontext())
    begin = MagicMock()
    monkeypatch.setattr(attention.torch.npu, "graph_task_update_begin", begin)

    attention.AscendAttentionBackendImpl.update_graph_params(
        object(),
        SimpleNamespace(attn_metadata=runtime_metadata),
        32,
        _config(),
    )

    begin.assert_not_called()


@pytest.mark.parametrize(
    "runtime_metadata",
    [object(), {}, {"model.layers.0.self_attn": object()}],
)
def test_gdn_graph_update_skips_unknown_runtime_metadata(
    monkeypatch, runtime_metadata
):
    gdn = _import_gdn()
    layer_name = "model.layers.0.self_attn"
    graph_params = SimpleNamespace(
        conv1d_params={
            32: [
                (
                    object(),
                    MagicMock(size=lambda _dim: 32),
                    object(),
                    object(),
                    None,
                    1,
                    -1,
                    1,
                    "non_spec_decode",
                    layer_name,
                    (),
                    (),
                    (),
                    1,
                )
            ]
        },
        conv1d_handles={32: [object()]},
        conv1d_events={32: [MagicMock()]},
    )
    begin = MagicMock()
    monkeypatch.setattr(gdn, "get_graph_params", lambda: graph_params)
    monkeypatch.setattr(gdn.torch.npu, "stream", lambda _: nullcontext())
    monkeypatch.setattr(gdn.torch.npu, "graph_task_update_begin", begin)

    gdn.update_conv1d_graph_params(
        object(),
        SimpleNamespace(attn_metadata=runtime_metadata),
        32,
        _config(),
    )

    begin.assert_not_called()


def test_fia_graph_update_uses_layer_names_with_interleaved_gdn(
    monkeypatch,
):
    attention = _import_attention()

    def fia_param(layer_name):
        return (
            layer_name,
            object(),
            object(),
            object(),
            object(),
            None,
            128,
            2,
            4,
            0.5,
            object(),
            object(),
            3,
            2**31 - 1,
            2**31 - 1,
        )

    graph_params = SimpleNamespace(
        attention_params={32: [fia_param("fa.1"), fia_param("fa.0")]},
        handles={32: [object(), object()]},
        events={32: [MagicMock(), MagicMock()]},
        workspaces={32: "workspace"},
    )
    metadata = {
        "gdn.0": object(),
        "fa.0": attention.AscendMetadata(
            seq_lens_list=[10],
            actual_seq_lengths_q=[1],
            block_tables="bt-0",
        ),
        "fa.1": attention.AscendMetadata(
            seq_lens_list=[20],
            actual_seq_lengths_q=[1],
            block_tables="bt-1",
        ),
    }
    out = MagicMock()
    monkeypatch.setattr(attention, "get_ascend_graph_params", lambda: graph_params)
    monkeypatch.setattr(attention, "using_paged_attention", lambda *_: False)
    monkeypatch.setattr(attention.torch.npu, "stream", lambda _: nullcontext())
    monkeypatch.setattr(attention.torch.npu, "graph_task_update_begin", MagicMock())
    monkeypatch.setattr(attention.torch.npu, "graph_task_update_end", MagicMock())
    monkeypatch.setattr(
        attention.torch_npu.npu_fused_infer_attention_score,
        "out",
        out,
    )

    attention.AscendAttentionBackendImpl.update_graph_params(
        object(),
        SimpleNamespace(attn_metadata=metadata),
        32,
        _config(),
    )

    assert [call.kwargs["block_table"] for call in out.call_args_list] == [
        "bt-1",
        "bt-0",
    ]


def test_pa_graph_update_regression(monkeypatch):
    attention = _import_attention()
    output = object()
    param = (
        "fa.0",
        object(),
        object(),
        object(),
        2,
        4,
        0.5,
        "captured-block-table",
        output,
    )
    event = MagicMock()
    graph_params = SimpleNamespace(
        attention_params={32: [param]},
        handles={32: [object()]},
        events={32: [event]},
        workspaces={32: "captured-workspace"},
    )
    metadata = attention.AscendMetadata(seq_lens=torch.tensor([17]))
    workspace_getter = MagicMock(return_value="runtime-workspace")
    pa = MagicMock()
    monkeypatch.setattr(attention, "get_ascend_graph_params", lambda: graph_params)
    monkeypatch.setattr(attention, "using_paged_attention", lambda *_: True)
    monkeypatch.setattr(attention.torch.npu, "stream", lambda _: nullcontext())
    monkeypatch.setattr(attention.torch.npu, "graph_task_update_begin", MagicMock())
    monkeypatch.setattr(attention.torch.npu, "graph_task_update_end", MagicMock())
    monkeypatch.setattr(
        attention.torch_npu,
        "_npu_paged_attention_get_workspace",
        workspace_getter,
    )
    monkeypatch.setattr(attention.torch_npu, "_npu_paged_attention", pa)

    attention.AscendAttentionBackendImpl.update_graph_params(
        object(),
        SimpleNamespace(attn_metadata={"fa.0": metadata}),
        32,
        _config(pa_shape_list=[32]),
    )

    workspace_getter.assert_called_once()
    assert pa.call_args.kwargs["workspace"] == "runtime-workspace"
    torch.testing.assert_close(
        pa.call_args.kwargs["context_lens"],
        torch.tensor([17]),
    )
    event.record.assert_called_once()


def test_fia_capture_registers_event_params_handle_and_reuses_workspace(
    monkeypatch,
):
    attention = _import_attention()
    impl = object.__new__(attention.AscendAttentionBackendImpl)
    impl.num_kv_heads = 2
    impl.num_heads = 4
    impl.scale = 0.5
    impl.sliding_window = None
    query = torch.empty(2, 4, 8)
    key_cache = torch.empty(2, 128, 2, 8)
    value_cache = torch.empty_like(key_cache)
    block_table = torch.zeros(2, 1, dtype=torch.int32)
    output = torch.empty_like(query)
    metadata = attention.AscendMetadata(
        actual_seq_lengths_q=[1, 2],
        seq_lens_list=[10, 20],
        block_tables=block_table,
        attn_mask=torch.ones(1, dtype=torch.int8),
        causal=True,
    )
    graph_params = SimpleNamespace(
        attention_params={2: []},
        handles={2: []},
        events={2: []},
        workspaces={2: None},
    )
    workspace = torch.empty(1)
    workspace_getter = MagicMock(return_value=workspace)
    event = MagicMock()
    stream = object()
    calls = MagicMock()
    begin = MagicMock()
    run = MagicMock()
    end = MagicMock(return_value="handle")
    calls.attach_mock(begin, "begin")
    calls.attach_mock(run, "run")
    calls.attach_mock(end, "end")

    monkeypatch.setattr(
        impl,
        "_get_fia_params",
        lambda *_: (key_cache, value_cache, 128, block_table, [10, 20]),
    )
    monkeypatch.setattr(attention, "get_ascend_graph_params", lambda: graph_params)
    monkeypatch.setattr(attention, "weak_ref_tensors", lambda value: value)
    monkeypatch.setattr(
        attention,
        "update_ascend_graph_params_workspace",
        lambda num_tokens, value: graph_params.workspaces.__setitem__(
            num_tokens, value
        ),
    )
    monkeypatch.setattr(
        attention.torch_npu,
        "_npu_fused_infer_attention_score_get_max_workspace",
        workspace_getter,
    )
    monkeypatch.setattr(
        attention.torch_npu.npu_fused_infer_attention_score,
        "out",
        run,
    )
    monkeypatch.setattr(attention.torch.npu, "current_stream", lambda: stream)
    monkeypatch.setattr(attention.torch.npu, "ExternalEvent", lambda: event)
    monkeypatch.setattr(attention.torch.npu, "graph_task_group_begin", begin)
    monkeypatch.setattr(attention.torch.npu, "graph_task_group_end", end)

    impl.full_graph_fused_infer_attention(
        query, query, query, metadata, output, "fa.0"
    )
    impl.full_graph_fused_infer_attention(
        query, query, query, metadata, output, "fa.1"
    )

    workspace_getter.assert_called_once()
    assert graph_params.workspaces[2] is workspace
    assert [params[0] for params in graph_params.attention_params[2]] == [
        "fa.0",
        "fa.1",
    ]
    assert graph_params.handles[2] == ["handle", "handle"]
    assert graph_params.events[2] == [event, event]
    assert [call[0] for call in calls.mock_calls] == [
        "begin", "run", "end", "begin", "run", "end"
    ]
    assert event.wait.call_count == 2
    assert event.reset.call_count == 2


def test_pa_capture_uses_max_workspace_sequence_length():
    from vllm_fl.worker.model_runner import (
        SEQ_LEN_WITH_MAX_PA_WORKSPACE,
        _ascend_graph_capture_seq_len,
    )

    assert SEQ_LEN_WITH_MAX_PA_WORKSPACE == 6144
    assert _ascend_graph_capture_seq_len(
        num_tokens=32,
        vllm_config=_config(pa_shape_list=[32]),
        is_graph_capturing=True,
        default_seq_len=1,
    ) == 6144
    assert _ascend_graph_capture_seq_len(
        num_tokens=32,
        vllm_config=_config(),
        is_graph_capturing=True,
        default_seq_len=1,
    ) == 1
