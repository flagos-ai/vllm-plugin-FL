# Copyright (c) 2026 BAAI. All rights reserved.

"""HY V4 EMPTY preserves real upstream dispatch, cache semantics and sinks."""

import inspect
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from tests.utils.device_utils import get_backend, requires_cuda_hopper

pytestmark = pytest.mark.skipif(get_backend() != "nvidia", reason="NVIDIA HY V4 EMPTY")


@pytest.fixture(params=[False, True])
def gems_mode(request):
    import flag_gems

    from vllm_fl.utils import get_flag_gems_whitelist_blacklist

    _, blacklist = get_flag_gems_whitelist_blacklist()
    context = flag_gems.use_gems(exclude=blacklist) if request.param else nullcontext()
    with context:
        yield request.param


@pytest.fixture
def installed(monkeypatch):
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.custom_op import op_registry_oot
    from vllm.model_executor.layers import sparse_attn_indexer
    from vllm.models.hy_v4.nvidia import attention, flashmla_sparse

    from vllm_fl.ops import nvidia_hy4

    config = VllmConfig()
    config.model_config = SimpleNamespace(
        architectures=["HYV4ForCausalLM"],
        is_moe=True,
        dtype=torch.bfloat16,
        hf_text_config=SimpleNamespace(learnable_sink=True),
    )
    config.compilation_config.custom_ops = ["all"]
    monkeypatch.delitem(op_registry_oot, "SparseAttnIndexer", raising=False)
    for obj, name in (
        (sparse_attn_indexer, "fp8_fp4_mqa_logits"),
        (sparse_attn_indexer, "fp8_fp4_paged_mqa_logits"),
        (attention, "per_token_group_quant_fp8"),
        (flashmla_sparse, "flash_mla_sparse_fwd"),
        (flashmla_sparse, "HYV4FlashMLASparseBackend"),
    ):
        monkeypatch.setattr(obj, name, getattr(obj, name))
    with set_current_vllm_config(config):
        yield config, nvidia_hy4
    op_registry_oot.pop("SparseAttnIndexer", None)


@pytest.mark.usefixtures("mock_hopper_capability")
def test_real_upstream_indexer_constructor_and_capabilities(installed):
    from vllm.model_executor.layers.sparse_attn_indexer import SparseAttnIndexer
    from vllm.models.hy_v4.nvidia import flashmla_sparse
    from vllm.utils.import_utils import has_deep_gemm
    from vllm.v1.attention.ops.flashmla import is_flashmla_sparse_supported

    from vllm_fl.dispatch.backends.flaggems.impl.dsv41_mla import flash_mla_sparse_fwd

    config, provider = installed
    before = has_deep_gemm(), is_flashmla_sparse_supported()
    assert provider.install_hy4_empty_adapters(config)
    assert provider.install_hy4_empty_adapters(config)
    assert flashmla_sparse.flash_mla_sparse_fwd is flash_mla_sparse_fwd
    cache = SimpleNamespace(prefix="test.indexer", kv_cache=torch.empty(0))
    layer = SparseAttnIndexer(
        cache,
        128,
        "ue8m0",
        2048,
        128,
        4096,
        4096,
        torch.empty(4, 2048, dtype=torch.int32),
    )
    from vllm_fl.ops.nvidia_sparse_indexer import NvidiaSparseAttnIndexer

    assert type(layer) is NvidiaSparseAttnIndexer
    assert type(layer).forward_cuda is SparseAttnIndexer.forward_cuda
    assert type(layer).forward_native is SparseAttnIndexer.forward_native
    assert layer._forward_method.__func__ is SparseAttnIndexer.forward_cuda
    assert type(layer).name == SparseAttnIndexer.name
    assert inspect.signature(type(layer).__init__) == inspect.signature(
        SparseAttnIndexer.__init__
    )
    assert (layer.dcp_world_size, layer.dcp_rank, layer.use_pcp) == (1, 0, False)
    assert (has_deep_gemm(), is_flashmla_sparse_supported()) == before


@pytest.mark.usefixtures("mock_hopper_capability")
def test_user_provider_and_native_preserved(installed, monkeypatch):
    from vllm.model_executor.custom_op import op_registry_oot
    from vllm.model_executor.layers.sparse_attn_indexer import SparseAttnIndexer

    config, provider = installed

    class UserIndexer(SparseAttnIndexer):
        pass

    monkeypatch.setitem(op_registry_oot, "SparseAttnIndexer", UserIndexer)
    assert not provider.install_hy4_empty_adapters(config)
    assert op_registry_oot["SparseAttnIndexer"] is UserIndexer
    monkeypatch.setattr(provider, "has_native_vllm_kernels", lambda: True)
    config.cache_config.cache_dtype = "fp8"
    assert not provider.install_hy4_empty_adapters(config)


@pytest.mark.parametrize(
    "architecture", ["DeepseekV41ForCausalLM", "Qwen4ExpForCausalLM"]
)
def test_other_models_unchanged(installed, architecture):
    config, provider = installed
    config.model_config.architectures = [architecture]
    assert not provider.install_hy4_empty_adapters(config)


@pytest.mark.parametrize("dtype", ["fp8", "fp8_ds_mla", "nvfp4_ds_mla"])
@pytest.mark.usefixtures("mock_hopper_capability")
def test_model_rejects_unimplemented_cache_writer(installed, dtype):
    config, provider = installed
    config.cache_config.cache_dtype = dtype
    with pytest.raises(NotImplementedError, match="656-byte FP8 cache writer"):
        provider.install_hy4_empty_adapters(config)


@pytest.mark.parametrize("major", [8, 10, None])
def test_model_rejects_unsupported_hardware(installed, monkeypatch, major):
    from vllm.platforms import current_platform
    from vllm.platforms.interface import DeviceCapability

    config, provider = installed
    capability = None if major is None else DeviceCapability(major, 0)
    monkeypatch.setattr(
        current_platform, "get_device_capability", lambda device_id=0: capability
    )
    with pytest.raises(NotImplementedError, match="Hopper SM90"):
        provider.install_hy4_empty_adapters(config)


def _values(shape, seed):
    return torch.randn(shape, generator=torch.Generator().manual_seed(seed)).to(
        torch.bfloat16
    )


def _fp8_reference(x, eps):
    scale = torch.exp2(
        torch.ceil(
            torch.log2(x.float().abs().amax(-1, keepdim=True).clamp_min(eps) / 448)
        )
    )
    q = (x.float() / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    return q, scale


@pytest.mark.gpu
@requires_cuda_hopper
@pytest.mark.parametrize("rows", [1, 32, 96])
def test_query_quantization_matches_fp8_reference(installed, rows, gems_mode):
    config, provider = installed
    assert provider.install_hy4_empty_adapters(config)
    from vllm.models.hy_v4.nvidia import attention

    x = _values((rows, 128), 7)
    expected_q, expected_scale = _fp8_reference(x, 1e-10)
    q, scales = attention.per_token_group_quant_fp8(x.cuda(), 128, use_ue8m0=True)
    torch.testing.assert_close(q.float().cpu(), expected_q.float(), rtol=0, atol=0)
    torch.testing.assert_close(scales.cpu(), expected_scale, rtol=0, atol=0)


@pytest.mark.gpu
@requires_cuda_hopper
@pytest.mark.parametrize("graph", [False, True])
@pytest.mark.parametrize("shared_pages", [False, True])
def test_indexer_cache_insert_gather_and_slot_replay(
    installed, graph, gems_mode, shared_pages
):
    config, provider = installed
    assert provider.install_hy4_empty_adapters(config)
    from vllm import _custom_ops as ops

    cpu_keys = _values((4, 128), 8)
    keys = cpu_keys.cuda()
    slots = torch.tensor([0, 63, 64, -1], dtype=torch.int64, device="cuda")
    if shared_pages:
        from tests.unit_tests.ops.test_nvidia_indexer_cache_layout import (
            _allocated_cache,
        )

        cache, raw, untouched = _allocated_cache("hy", 2)
        assert not cache.is_contiguous() and cache.storage_offset() > 0
    else:
        cache = torch.zeros(3, 64, 132, dtype=torch.uint8, device="cuda")
    num_pages = cache.shape[0]
    num_slots = num_pages * 64

    def clear_cache():
        # Production zeroes the contiguous backing allocation. Initialize
        # only our test layer's contiguous pages so other layers keep their
        # sentinels; a global Gems zero_ on the strided view ignores its gaps.
        for page in cache.unbind(0):
            assert page.is_contiguous()
            page.zero_()
        assert not cache.cpu().any()
        if shared_pages:
            assert torch.all(raw.cpu()[untouched] == 165)

    clear_cache()

    def insert():
        ops.indexer_k_quant_and_cache(keys, cache, slots, 128, "ue8m0")

    insert()
    if graph:
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            insert()
        stream.synchronize()
        captured = torch.cuda.CUDAGraph()
        with torch.cuda.graph(captured):
            insert()
        slots.copy_(torch.tensor([129, 130, 128, -1], device="cuda"))
        keys.copy_((cpu_keys * 2).cuda())
        clear_cache()
        captured.replay()
        cpu_keys = cpu_keys * 2
        target_slots = [129, 130, 128]
    else:
        target_slots = [0, 63, 64]
    expected_q, expected_scales = _fp8_reference(cpu_keys, 1e-4)
    flat = cache.cpu().reshape(num_pages, -1)
    values = flat[:, : 64 * 128].reshape(num_slots, 128).view(torch.float8_e4m3fn)
    scales = flat[:, 64 * 128 :].contiguous().view(torch.float32).reshape(num_slots)
    for row, slot in enumerate(target_slots):
        torch.testing.assert_close(
            values[slot].float(), expected_q[row].float(), rtol=0, atol=0
        )
        torch.testing.assert_close(
            scales[slot], expected_scales[row, 0], rtol=0, atol=0
        )
    unused = sorted(set(range(num_slots)) - set(target_slots))
    assert not values[unused].float().any()
    assert not scales[unused].any()
    gathered = torch.empty(num_slots, 128, dtype=torch.float8_e4m3fn, device="cuda")
    gathered_scales = torch.empty(num_slots, 4, dtype=torch.uint8, device="cuda")
    ops.cp_gather_indexer_k_quant_cache(
        cache,
        gathered,
        gathered_scales,
        torch.arange(num_pages, dtype=torch.int32, device="cuda").unsqueeze(0),
        torch.tensor([0, num_slots], dtype=torch.int32, device="cuda"),
    )
    torch.testing.assert_close(gathered.float().cpu(), values.float(), rtol=0, atol=0)
    torch.testing.assert_close(
        gathered_scales.cpu().view(torch.float32).flatten(), scales, rtol=0, atol=0
    )
    if shared_pages:
        assert torch.all(raw.cpu()[untouched] == 165)


def _attention_reference(q, kv, indices, sink, length=None):
    q, kv, indices, sink = (
        q.float().cpu(),
        kv.float().cpu().squeeze(1),
        indices.cpu().squeeze(1).long(),
        sink.float().cpu(),
    )
    valid = (indices >= 0) & (indices < kv.shape[0])
    if length is not None:
        valid &= torch.arange(indices.shape[1])[None] < length.cpu()[:, None]
    selected = kv[indices.clamp(0, kv.shape[0] - 1)]
    scores = torch.einsum("thd,tkd->thk", q, selected) * (576**-0.5)
    scores.masked_fill_(~valid[:, None], -float("inf"))
    lse = scores.logsumexp(-1)
    probs = torch.softmax(scores, -1)
    output = torch.einsum("thk,tkd->thd", probs, selected[..., :512])
    output *= torch.sigmoid(lse - sink[None])[..., None]
    return output, lse


@pytest.mark.gpu
@requires_cuda_hopper
@pytest.mark.parametrize("heads", [64, 128])
@pytest.mark.parametrize("graph", [False, True])
def test_bf16_mla_sink_prefill_decode_and_graph(heads, graph, gems_mode):
    from vllm_fl.dispatch.backends.flaggems.impl.dsv41_mla import flash_mla_sparse_fwd

    q = (_values((3, heads, 576), 9) * 0.25).cuda()
    kv = (_values((96, 1, 576), 10) * 0.25).cuda()
    indices = (
        torch.arange(128, dtype=torch.int32, device="cuda").repeat(3, 1).unsqueeze(1)
    )
    indices[:, :, 5] = -1
    sink = torch.linspace(-2, 2, heads, device="cuda")
    sink[0], sink[1] = -float("inf"), float("inf")
    length = torch.tensor([32, 67, 96], dtype=torch.int32, device="cuda")
    output = torch.empty(3, heads, 512, dtype=torch.bfloat16, device="cuda")

    def run():
        return flash_mla_sparse_fwd(
            q, kv, indices, 576**-0.5, attn_sink=sink, topk_length=length, out=output
        )

    result, _, lse = run()
    assert result.data_ptr() == output.data_ptr()
    if graph:
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            run()
        stream.synchronize()
        captured = torch.cuda.CUDAGraph()
        with torch.cuda.graph(captured):
            result, _, lse = run()
        q.mul_(0.5)
        sink[2:].add_(0.75)
        indices[:, :, :4] = -1
        length.copy_(torch.tensor([48, 71, 93], device="cuda"))
        captured.replay()
    expected, expected_lse = _attention_reference(q, kv, indices, sink, length)
    torch.testing.assert_close(result.float().cpu(), expected, rtol=0.015, atol=0.002)
    torch.testing.assert_close(lse.cpu(), expected_lse, rtol=2e-4, atol=2e-4)
    assert not result[:, 1].float().cpu().any()


@pytest.mark.gpu
@requires_cuda_hopper
@pytest.mark.parametrize("heads", [4, 64])
def test_upstream_sink_impl_padding_and_live_graph(
    installed, tmp_path, heads, gems_mode
):
    from vllm.distributed import init_distributed_environment, initialize_model_parallel
    from vllm.distributed.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
    )
    from vllm.models.hy_v4.nvidia.flashmla_sparse import HYV4FlashMLASparseImpl
    from vllm.v1.worker.workspace import init_workspace_manager

    from vllm_fl.ops.nvidia_hy4_mla import NvidiaHYV4FlashMLASparseImpl

    config, provider = installed
    assert provider.install_hy4_empty_adapters(config)
    config.scheduler_config.max_num_batched_tokens = 4
    owned_group = not torch.distributed.is_initialized()
    try:
        if owned_group:
            init_distributed_environment(
                world_size=1,
                rank=0,
                local_rank=0,
                distributed_init_method=(tmp_path / "dist").as_uri(),
                backend="nccl",
            )
            initialize_model_parallel(
                tensor_model_parallel_size=1, pipeline_model_parallel_size=1
            )
        init_workspace_manager(torch.device("cuda:0"))
        sink = torch.linspace(-2, 2, heads, dtype=torch.float32, device="cuda")
        impl = NvidiaHYV4FlashMLASparseImpl(
            num_heads=heads,
            head_size=576,
            scale=576**-0.5,
            num_kv_heads=1,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="auto",
            logits_soft_cap=None,
            attn_type="decoder",
            kv_sharing_target_layer_name=None,
            q_lora_rank=2048,
            kv_lora_rank=512,
            qk_nope_head_dim=192,
            qk_rope_head_dim=64,
            qk_head_dim=256,
            v_head_dim=256,
            kv_b_proj=None,
            sinks=sink,
        )
        assert not impl.supports_dense_mha_prefill
        assert type(impl).__init__ is HYV4FlashMLASparseImpl.__init__
        assert (
            type(impl)._bf16_flash_mla_kernel
            is HYV4FlashMLASparseImpl._bf16_flash_mla_kernel
        )
        assert type(impl).forward_mqa is HYV4FlashMLASparseImpl.forward_mqa
        q = (_values((2, heads, 576), 27) * 0.25).cuda()
        cache = (_values((2, 64, 576), 28) * 0.25).cuda()
        indices = torch.arange(128, dtype=torch.int32, device="cuda").repeat(2, 1)
        indices[:, 8] = -1
        padded = impl._sinks_for_query(q, 1, 64)
        torch.testing.assert_close(padded[:heads], sink)
        assert torch.isneginf(padded[heads:]).all()

        concatenated = torch.empty(2, 64, 576, dtype=torch.bfloat16, device="cuda")

        def run():
            from vllm import _custom_ops as ops

            ops.concat_mla_q(q[..., :512], q[..., 512:], concatenated)
            return impl._bf16_flash_mla_kernel(
                concatenated, cache, indices, actual_num_heads=heads
            )

        run()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            run()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output, lse = run()
        sink.add_(2)
        q.mul_(0.5)
        indices[:, :3] = -1
        graph.replay()
        torch.testing.assert_close(
            concatenated[:, :heads].cpu(), q.cpu(), rtol=0, atol=0
        )
        assert not concatenated[:, heads:].cpu().any()
        expected, expected_lse = _attention_reference(
            q, cache.reshape(-1, 1, 576), indices[:, None], sink
        )
        torch.testing.assert_close(
            output.float().cpu(), expected, rtol=0.015, atol=0.002
        )
        torch.testing.assert_close(lse.cpu(), expected_lse, rtol=2e-4, atol=2e-4)
    finally:
        if owned_group:
            destroy_model_parallel()
            destroy_distributed_environment()
