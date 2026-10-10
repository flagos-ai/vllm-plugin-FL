# Copyright (c) 2026 Kunlunxin, Inc. All rights reserved.

import pytest
import torch

from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphWrapper
from vllm.config import CompilationConfig, CompilationMode, CUDAGraphMode, VllmConfig
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.platforms import current_platform

from vllm_fl.dispatch.backends.vendor.kunlunxin.patch import (
    patch_breakable_full_only,
    patch_breakable_private_pools,
)

pytestmark = pytest.mark.skipif(
    getattr(current_platform, "vendor_name", None) != "kunlunxin",
    reason="Real Kunlunxin graph replay",
)


def test_native_piecewise_boundary_refreshes_metadata(monkeypatch):
    from types import SimpleNamespace

    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
        QwenGatedDeltaNetAttention,
    )

    from vllm_fl.dispatch.backends.vendor.kunlunxin.impl.attention import (
        KunlunxinAttentionBackendImpl,
    )
    from vllm_fl.dispatch.backends.vendor.kunlunxin.patch import (
        patch_native_piecewise_boundaries,
    )

    monkeypatch.setenv("VLLM_USE_BREAKABLE_CUDAGRAPH", "1")
    config = VllmConfig(
        compilation_config=CompilationConfig(
            mode=CompilationMode.NONE, cudagraph_mode=CUDAGraphMode.PIECEWISE
        )
    )
    layer = SimpleNamespace(
        layer_name="test.attention", kv_cache=torch.empty(0, device="cuda")
    )
    config.compilation_config.static_forward_context[layer.layer_name] = layer
    eager_calls = []

    def attention(self, layer, query, key, value, kv_cache, attn_metadata, output=None):
        assert not torch.cuda.is_current_stream_capturing()
        eager_calls.append(attn_metadata["offset"])
        output.copy_(query + attn_metadata["offset"])
        return output

    monkeypatch.setattr(KunlunxinAttentionBackendImpl, "forward", attention)
    monkeypatch.setattr(
        QwenGatedDeltaNetAttention,
        "_forward_core",
        QwenGatedDeltaNetAttention._forward_core,
    )
    patch_native_piecewise_boundaries()
    impl = object.__new__(KunlunxinAttentionBackendImpl)
    value = torch.ones((4, 16), device="cuda")
    attn_output, output = torch.empty_like(value), torch.empty_like(value)
    outer_calls = []

    def run():
        outer_calls.append(True)
        impl.forward(
            layer,
            value,
            value,
            value,
            layer.kv_cache,
            {"offset": -999},
            output=attn_output,
        )
        output.copy_(attn_output * 3)
        return output

    wrapper = BreakableCUDAGraphWrapper(run, config)
    stream = torch.cuda.Stream()
    descriptor = BatchDescriptor(num_tokens=4, uniform=False)
    for step in (1, 2, 5):
        value.fill_(step)
        output.fill_(float("nan"))
        torch.cuda.synchronize()
        with (
            torch.cuda.stream(stream),
            set_forward_context(
                {layer.layer_name: {"offset": step + 7}},
                config,
                cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE,
                batch_descriptor=descriptor,
                slot_mapping={},
            ),
        ):
            wrapper()
            if step == 1:
                wrapper()
        torch.cuda.synchronize()
        torch.testing.assert_close(
            output.cpu(), torch.full((4, 16), (2 * step + 7) * 3, dtype=torch.float32)
        )
    assert len(outer_calls) == 1
    assert eager_calls == [8, 8, 9, 12]
    assert wrapper.entries[descriptor].capture.num_eager_breaks == 1
    assert wrapper.entries[descriptor].capture.num_graphs > 0
    wrapper.clear_graphs()


def test_compile_wrapper_selects_actual_graph_owner(monkeypatch):
    from vllm.compilation.decorators import support_torch_compile
    from vllm.config import set_current_vllm_config

    from vllm_fl.compilation.graph import GraphWrapper

    monkeypatch.delenv("VLLM_USE_BREAKABLE_CUDAGRAPH", raising=False)
    config = VllmConfig(
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            cudagraph_mode=CUDAGraphMode.PIECEWISE,
            cudagraph_capture_sizes=[4, 8],
        )
    )

    @support_torch_compile(dynamic_arg_dims={"value": 0})
    class CompiledModel(torch.nn.Module):
        def __init__(self, vllm_config: VllmConfig):
            super().__init__()

        def forward(self, value: torch.Tensor):
            return value * 3 + 7

    patch_breakable_private_pools()
    patch_breakable_full_only()
    existing = set(GraphWrapper._all_instances)
    stream = torch.cuda.Stream()
    with set_current_vllm_config(config):
        model = CompiledModel(vllm_config=config)
        wrapper = BreakableCUDAGraphWrapper(model, config)
        value = torch.ones((8, 16), device="cuda")
        # Real vLLM compilation warmup, not a mocked compiled flag.
        with set_forward_context(None, config):
            model(value)
        for size in (4, 8):
            value = torch.ones((size, 16), device="cuda")
            descriptor = BatchDescriptor(num_tokens=size, uniform=False)
            torch.cuda.synchronize()
            with (
                torch.cuda.stream(stream),
                set_forward_context(
                    None,
                    config,
                    cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE,
                    batch_descriptor=descriptor,
                ),
            ):
                wrapper(value)
            for step in (2, 5):
                value.fill_(step)
                torch.cuda.synchronize()
                with (
                    torch.cuda.stream(stream),
                    set_forward_context(
                        None,
                        config,
                        cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE,
                        batch_descriptor=descriptor,
                    ),
                ):
                    output = wrapper(value)
                torch.cuda.synchronize()
                torch.testing.assert_close(
                    output.cpu(),
                    torch.full((size, 16), step * 3 + 7, dtype=torch.float32),
                )
                output.fill_(float("nan"))
        inner = set(GraphWrapper._all_instances) - existing
        if inner:
            assert not wrapper.entries, (
                "outer wrapper nested capture around compiled graph"
            )
            assert any(instance.concrete_graph_entries for instance in inner)
            assert all(
                entry.graph is not None
                for instance in inner
                for entry in instance.concrete_graph_entries.values()
            )
        else:
            # This image's torch.compile can return the original callable.
            # vLLM still sets compiled=True: native replay must remain active.
            assert model.compiled
            assert getattr(model, "_compiled_bytecode", None) is None
            assert not hasattr(model._compiled_callable, "_torchdynamo_orig_callable")
            assert len(wrapper.entries) == 2
            assert all(
                entry.capture.num_graphs > 0 for entry in wrapper.entries.values()
            )
        for instance in inner:
            instance.clear_graphs()
        wrapper.clear_graphs()


@pytest.mark.parametrize("mode", [CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL])
def test_native_breakable_replays_refreshed_inputs(mode, monkeypatch):
    monkeypatch.setenv("VLLM_USE_BREAKABLE_CUDAGRAPH", "1")
    config = VllmConfig(
        compilation_config=CompilationConfig(
            mode=CompilationMode.NONE, cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE
        )
    )
    patch_breakable_private_pools()
    patch_breakable_full_only()
    calls = []
    outputs = {}

    def run(value):
        calls.append(value.shape[0])
        # Retain owned output buffers while the production wrapper uses weak refs.
        outputs[value.shape[0]].copy_(value * 3 + 7)
        return outputs[value.shape[0]]

    wrapper = BreakableCUDAGraphWrapper(run, config)
    stream = torch.cuda.Stream()
    for size in (4, 8):
        value = torch.ones((size, 16), device="cuda")
        outputs[size] = torch.empty_like(value)
        descriptor = BatchDescriptor(
            num_tokens=size, uniform=mode == CUDAGraphMode.FULL
        )
        torch.cuda.synchronize()
        with (
            torch.cuda.stream(stream),
            set_forward_context(
                None, config, cudagraph_runtime_mode=mode, batch_descriptor=descriptor
            ),
        ):
            result = wrapper(value)
        torch.cuda.synchronize()
        captured_calls = len(calls)
        for step in (2, 5):
            value.fill_(step)
            result.fill_(float("nan"))
            torch.cuda.synchronize()
            with (
                torch.cuda.stream(stream),
                set_forward_context(
                    None,
                    config,
                    cudagraph_runtime_mode=mode,
                    batch_descriptor=descriptor,
                ),
            ):
                actual = wrapper(value)
            torch.cuda.synchronize()
            assert len(calls) == captured_calls, (
                "runnable ran eagerly instead of replay"
            )
            expected = torch.full((size, 16), step * 3 + 7, dtype=torch.float32)
            torch.testing.assert_close(actual.cpu(), expected)
        assert wrapper.entries[descriptor].capture.num_graphs > 0
    assert len(wrapper.entries) == 2
    pools = [entry.capture.pool for entry in wrapper.entries.values()]
    if mode == CUDAGraphMode.PIECEWISE:
        assert all(pool is not None for pool in pools)
        assert pools[0] != pools[1]
    else:
        assert pools == [None, None]
    wrapper.clear_graphs()


@pytest.mark.parametrize("interleaved", [False, True])
@pytest.mark.parametrize("size", [4, 8])
def test_full_attention_replay_isolates_padding_and_null_block(interleaved, size):
    from vllm_fl.dispatch.backends.vendor.kunlunxin.impl.attention import (
        KunlunxinAttentionBackendImpl,
        KunlunxinMetadata,
    )

    config = VllmConfig(
        compilation_config=CompilationConfig(
            mode=CompilationMode.NONE, cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY
        )
    )
    patch_breakable_private_pools()
    patch_breakable_full_only()
    heads, dim, block_size, blocks = 2, 128, 128, size + 2
    generator = torch.Generator().manual_seed(230)
    base = torch.randn((2, blocks, heads, block_size, dim), generator=generator).to(
        torch.bfloat16
    )
    cache = torch.empty(base.shape, device="cuda", dtype=base.dtype)
    if interleaved:
        elements = heads * block_size * dim
        cache = cache.as_strided(
            base.shape, (elements, 2 * elements, block_size * dim, dim, 1)
        )
    cache.copy_(base)
    inputs_cpu = [
        torch.randn((size, heads, dim), generator=generator).to(base.dtype)
        for _ in range(3)
    ]
    query, key, value = [tensor.cuda() for tensor in inputs_cpu]
    output = torch.empty_like(query)
    slots = torch.empty(size, dtype=torch.int64, device="cuda")
    lengths = torch.empty(size, dtype=torch.int32, device="cuda")
    host_lengths = torch.empty(size, dtype=torch.int32)
    tables = torch.empty((size, 1), dtype=torch.int32, device="cuda")
    metadata = KunlunxinMetadata(
        seq_lens_tensor=lengths,
        seq_lens_tensor_host=host_lengths,
        max_decode_seq_len=2,
        block_tables=tables,
        num_prefills=0,
        num_prefill_tokens=0,
        num_decode_tokens=size,
        slot_mapping=slots,
        enable_kv_scales_calculation=False,
        max_prefill_seq_len=0,
        num_actual_tokens=size,
        use_cuda_graph=True,
    )
    impl = KunlunxinAttentionBackendImpl(
        heads, dim, dim**-0.5, heads, None, None, "auto"
    )
    calls = []

    def run(q, k, v):
        calls.append(True)
        return impl.forward(None, q, k, v, cache, metadata, output=output)

    wrapper = BreakableCUDAGraphWrapper(run, config)
    stream = torch.cuda.Stream()
    descriptor = BatchDescriptor(num_tokens=size, uniform=True)
    for step, (live, poison) in enumerate(
        ((size, 0.0), (size - 1, float("nan")), (1, float("inf")), (size, 0.0))
    ):
        cache.copy_(base)
        q_cpu, k_cpu, v_cpu = [tensor.clone() + step / 8 for tensor in inputs_cpu]
        for cpu, tensor in zip((q_cpu, k_cpu, v_cpu), (query, key, value)):
            cpu[live:] = poison
            tensor.copy_(cpu)
        slots_cpu = torch.full((size,), -1, dtype=torch.int64)
        slots_cpu[:live] = torch.arange(1, live + 1) * block_size + 1
        slots.copy_(slots_cpu)
        lens_cpu = torch.ones(size, dtype=torch.int32)
        lens_cpu[:live] = 2
        lengths.copy_(lens_cpu)
        host_lengths.copy_(lens_cpu)
        tables_cpu = torch.zeros((size, 1), dtype=torch.int32)
        tables_cpu[:live, 0] = torch.arange(1, live + 1)
        tables.copy_(tables_cpu)
        output.fill_(float("nan"))
        if step == 0:
            with set_forward_context(
                metadata,
                config,
                cudagraph_runtime_mode=CUDAGraphMode.FULL,
                batch_descriptor=descriptor,
            ):
                run(query, key, value)
            calls.clear()
            cache.copy_(base)
        torch.cuda.synchronize()
        with (
            torch.cuda.stream(stream),
            set_forward_context(
                metadata,
                config,
                cudagraph_runtime_mode=CUDAGraphMode.FULL,
                batch_descriptor=descriptor,
            ),
        ):
            actual = wrapper(query, key, value)
            if step == 0:
                # Capture records kernels; initial capture output is not a replay.
                actual = wrapper(query, key, value)
        torch.cuda.synchronize()
        assert len(calls) == 1, "expected real FULL replay, not eager execution"
        expected_cache = base.clone()
        expected = torch.zeros_like(q_cpu).float()
        for row in range(live):
            block = row + 1
            expected_cache[0, block, :, 1] = k_cpu[row]
            expected_cache[1, block, :, 1] = v_cpu[row]
            k_hist = expected_cache[0, block, :, :2].float()
            v_hist = expected_cache[1, block, :, :2].float()
            scores = (q_cpu[row].float().unsqueeze(1) * k_hist).sum(-1) * dim**-0.5
            expected[row] = (scores.softmax(-1).unsqueeze(-1) * v_hist).sum(1)
        for row in range(live, size):
            expected_cache[:, 0, :, row % block_size] = 0
        torch.testing.assert_close(cache.cpu(), expected_cache, rtol=0, atol=0)
        torch.testing.assert_close(
            actual.cpu().view_as(expected).float(), expected, rtol=0.025, atol=0.025
        )
    assert wrapper.entries[descriptor].capture.num_graphs > 0
    wrapper.clear_graphs()


@pytest.mark.parametrize("size", [4, 8])
def test_full_recurrent_replay_padding_does_not_mutate_live_state(size):
    from vllm_fl.dispatch.backends.vendor.kunlunxin.impl.fla.fused_recurrent import (
        fused_recurrent_gated_delta_rule,
    )

    config = VllmConfig(
        compilation_config=CompilationConfig(
            mode=CompilationMode.NONE, cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY
        )
    )
    generator = torch.Generator().manual_seed(322)
    heads, dim = 2, 128
    base = torch.randn((size + 2, heads, dim, dim), generator=generator) * 0.01
    state = base.cuda()
    cpu_inputs = [
        (torch.randn((1, size, heads, dim), generator=generator) * 0.1).to(
            torch.bfloat16
        )
        for _ in range(3)
    ]
    q, k, v = [tensor.cuda() for tensor in cpu_inputs]
    g = torch.full((1, size, heads), -0.2, device="cuda")
    beta_value = torch.tensor(0.3, dtype=torch.bfloat16).float().item()
    beta = torch.full((1, size, heads), beta_value, device="cuda", dtype=torch.bfloat16)
    indices = torch.arange(1, size + 1, dtype=torch.int32, device="cuda")
    lod = torch.arange(size + 1, dtype=torch.int32, device="cuda")
    output = torch.empty_like(v)
    calls = []

    def run():
        calls.append(True)
        result, _ = fused_recurrent_gated_delta_rule(
            q,
            k,
            v,
            g,
            beta,
            scale=dim**-0.5,
            initial_state=state,
            inplace_final_state=True,
            cu_seqlens=lod,
            ssm_state_indices=indices,
            use_qk_l2norm_in_kernel=False,
        )
        output.copy_(result)
        return output

    run()
    calls.clear()
    wrapper = BreakableCUDAGraphWrapper(run, config)
    descriptor = BatchDescriptor(num_tokens=size, uniform=True)
    stream = torch.cuda.Stream()
    for step, (live, poison) in enumerate(
        ((size, 0.0), (size - 1, float("nan")), (1, float("inf")), (size, 0.0))
    ):
        state.copy_(base)
        actual_inputs = [tensor.clone() for tensor in cpu_inputs]
        for cpu, device in zip(actual_inputs, (q, k, v)):
            cpu[:, :live] += step / 128
            cpu[:, live:] = poison
            device.copy_(cpu)
        index_cpu = torch.zeros(size, dtype=torch.int32)
        index_cpu[:live] = torch.arange(1, live + 1)
        indices.copy_(index_cpu)
        output.fill_(float("nan"))
        torch.cuda.synchronize()
        with (
            torch.cuda.stream(stream),
            set_forward_context(
                None,
                config,
                cudagraph_runtime_mode=CUDAGraphMode.FULL,
                batch_descriptor=descriptor,
            ),
        ):
            wrapper()
            if step == 0:
                wrapper()
        torch.cuda.synchronize()
        assert len(calls) == 1
        expected_state = base.clone()
        expected_output = torch.empty((live, heads, dim))
        for row in range(live):
            # Vendor state is transposed [V,K]. This reference is independent
            # CPU PyTorch, with the recurrence written explicitly.
            qc, kc, vc = [tensor[0, row].float() for tensor in actual_inputs]
            history = base[row + 1] * torch.exp(torch.tensor(-0.2))
            residual = (vc - (history * kc.unsqueeze(-2)).sum(-1)) * beta_value
            history = history + residual.unsqueeze(-1) * kc.unsqueeze(-2)
            expected_state[row + 1] = history
            expected_output[row] = (history * qc.unsqueeze(-2)).sum(-1) * dim**-0.5
        # State zero is vLLM's reserved scratch slot, never a live request.
        torch.testing.assert_close(
            state[1:].cpu(), expected_state[1:], rtol=0.025, atol=0.003
        )
        torch.testing.assert_close(
            output[0, :live].cpu().float(), expected_output, rtol=0.025, atol=0.003
        )
    wrapper.clear_graphs()
