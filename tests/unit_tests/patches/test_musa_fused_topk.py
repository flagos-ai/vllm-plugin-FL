# Copyright (c) 2026 BAAI. All rights reserved.

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import vllm
from vllm.config import CompilationMode, CUDAGraphMode

from vllm_fl.dispatch.backends.vendor.musa import patch as musa_patch
from vllm_fl.models import cosyvoice3


class NativeSampler:
    def __init__(self):
        self.native = Mock(return_value=("native", None))
        self.forward = self.forward_native
        self.use_fp64_gumbel = False
        self.logprobs_mode = "raw_logprobs"

    def forward_native(self, *args):
        return self.native(*args)


def tensor_metadata(shape, dtype, device):
    return SimpleNamespace(
        shape=shape,
        ndim=len(shape),
        dtype=dtype,
        device=device,
        is_contiguous=lambda: True,
    )


def make_config(marker="hf"):
    hf_config = SimpleNamespace(
        architectures=["Qwen2ForCausalLM"],
        model_type="qwen2",
        vocab_size=6761,
        hidden_size=896,
        num_hidden_layers=24,
        num_attention_heads=14,
        num_key_value_heads=2,
    )
    additional_config = {}
    if marker == "hf":
        hf_config.flagos_profile = "cosyvoice3-0.5b"
    elif marker == "additional":
        additional_config["flagos_profile"] = "cosyvoice3-0.5b"
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=hf_config,
            dtype=torch.bfloat16,
            enable_prompt_embeds=True,
            quantization=None,
            enforce_eager=False,
        ),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=1
        ),
        compilation_config=SimpleNamespace(cudagraph_mode=None, mode=None),
        additional_config=additional_config,
        attention_config=SimpleNamespace(backend=None),
        speculative_config=None,
        kv_transfer_config=None,
    )


@pytest.fixture
def platform(monkeypatch):
    monkeypatch.setattr(vllm, "__version__", "0.28.0")
    monkeypatch.setattr(torch, "__version__", "2.11.0.post2+musa5.2.0")
    monkeypatch.delenv("VLLM_USE_V2_MODEL_RUNNER", raising=False)
    monkeypatch.delenv("VLLM_USE_BREAKABLE_CUDAGRAPH", raising=False)
    return SimpleNamespace(
        device_type="musa",
        vendor_name="mthreads",
        torch_device_fn=SimpleNamespace(get_device_name=Mock(return_value="MTT S5000")),
    )


@pytest.fixture
def sampler_case(monkeypatch, platform):
    for key in (
        "USE_FLAGGEMS",
        "VLLM_FL_PREFER",
        "VLLM_FL_PREFER_ENABLED",
        "VLLM_FL_FLAGOS_WHITELIST",
        "VLLM_FL_FLAGOS_BLACKLIST",
        "VLLM_FL_FLAGOS_BLACKLIST_APPEND",
    ):
        monkeypatch.delenv(key, raising=False)
    sampler = NativeSampler()
    fused = Mock(return_value="fused")
    noise = Mock()
    noise.exponential_.return_value = noise
    monkeypatch.setattr(torch, "empty_like", Mock(return_value=noise))
    monkeypatch.setitem(
        sys.modules, "flaggems_vllm", SimpleNamespace(topk_sample_from_logits=fused)
    )
    device = SimpleNamespace(type="musa")
    logits = tensor_metadata((1, 6761), torch.float32, device)
    k = tensor_metadata((1,), torch.int32, device)
    return sampler, logits, k, fused, noise


def bind(config, platform, sampler):
    return cosyvoice3.configure_sampler(
        config, platform, 0, SimpleNamespace(topk_topp_sampler=sampler)
    )


@pytest.mark.parametrize("first", ["cosyvoice", "qwen2"])
def test_musa_fused_topk_is_local_to_sampler(sampler_case, platform, first):
    sampler, logits, k, fused, noise = sampler_case
    other = NativeSampler()
    native_forward = NativeSampler.forward_native
    cosyvoice = make_config("additional")
    qwen2 = make_config(None)
    if first == "cosyvoice":
        assert bind(cosyvoice, platform, sampler)
        assert not bind(qwen2, platform, other)
    else:
        assert not bind(qwen2, platform, other)
        assert bind(cosyvoice, platform, sampler)
    assert NativeSampler.forward_native is native_forward
    assert sampler.forward(logits, {}, k, None) == ("fused", None)
    assert other.forward(logits, {}, k, None) == ("native", None)
    sampler.native.assert_not_called()
    noise.exponential_.assert_called_once_with()
    fused.assert_called_once_with(logits, k, noise)


@pytest.mark.parametrize(
    "unsupported",
    [
        "top_p",
        "request_seed",
        "fp64",
        "processed_logits",
        "processed_logprobs",
        "wide_vocab",
        "non_contiguous",
        "cpu",
        "invalid_k",
        "no_top_k",
    ],
)
def test_musa_fused_topk_retains_native_path(sampler_case, platform, unsupported):
    sampler, logits, k, fused, noise = sampler_case
    assert bind(make_config(), platform, sampler)
    generators = {}
    p = None
    if unsupported == "top_p":
        p = object()
    elif unsupported == "request_seed":
        generators = {0: object()}
    elif unsupported == "fp64":
        sampler.use_fp64_gumbel = True
    elif unsupported in ("processed_logits", "processed_logprobs"):
        sampler.logprobs_mode = unsupported
    elif unsupported == "wide_vocab":
        logits.shape = (1, 8193)
    elif unsupported == "non_contiguous":
        logits.is_contiguous = lambda: False
    elif unsupported == "cpu":
        logits.device = torch.device("cpu")
    elif unsupported == "invalid_k":
        k.shape = (2,)
    elif unsupported == "no_top_k":
        k = None

    assert sampler.forward(logits, generators, k, p) == ("native", None)
    sampler.native.assert_called_once_with(logits, generators, k, p)
    fused.assert_not_called()
    noise.exponential_.assert_not_called()


@pytest.mark.parametrize(
    "disable", ["model", "global", "blacklist", "shared", "missing", "import_error"]
)
def test_musa_fused_topk_stays_native(sampler_case, platform, disable, monkeypatch):
    from vllm_fl.dispatch.backends.vendor.musa.impl import topk_sampling

    sampler, logits, k, fused, noise = sampler_case
    config = make_config()
    load_provider = Mock(wraps=topk_sampling._load_fused_topk)
    monkeypatch.setattr(topk_sampling, "_load_fused_topk", load_provider)
    if disable == "model":
        config.additional_config["flagos_fused_topk"] = False
    elif disable == "global":
        monkeypatch.setenv("USE_FLAGGEMS", "0")
    elif disable == "blacklist":
        monkeypatch.setenv("VLLM_FL_FLAGOS_BLACKLIST", "topk_sample_from_logits")
    elif disable == "shared":
        monkeypatch.delitem(sys.modules, "flaggems_vllm")
    elif disable == "missing":
        monkeypatch.setitem(sys.modules, "flaggems_vllm", SimpleNamespace())
    elif disable == "import_error":
        load_provider.side_effect = OSError("optional provider cannot load")

    assert not bind(config, platform, sampler)
    assert load_provider.called == (disable in ("missing", "import_error"))
    assert sampler.forward == sampler.forward_native
    assert sampler.forward(logits, {}, k, None) == ("native", None)
    fused.assert_not_called()
    noise.exponential_.assert_not_called()


def test_private_engine_can_load_provider(sampler_case, platform, monkeypatch):
    from vllm_fl.dispatch.backends.vendor.musa.impl import topk_sampling

    sampler, _, _, fused, _ = sampler_case
    monkeypatch.delitem(sys.modules, "flaggems_vllm")
    monkeypatch.setattr(
        cosyvoice3.multiprocessing,
        "current_process",
        lambda: SimpleNamespace(name="EngineCore"),
    )
    load_provider = Mock(return_value=fused)
    monkeypatch.setattr(topk_sampling, "_load_fused_topk", load_provider)
    assert bind(make_config(), platform, sampler)
    load_provider.assert_called_once_with()


def test_existing_native_musa_compatibility_ignores_retired_switch(monkeypatch):
    from vllm.v1.sample.ops import topk_topp_sampler

    monkeypatch.setenv("VLLM_FL_MUSA_FUSED_TOPK", "1")
    native_forward = topk_topp_sampler.TopKTopPSampler.forward_native
    monkeypatch.setattr(topk_topp_sampler, "apply_top_k_top_p", Mock())
    musa_patch.patch_topk_topp_sampler()
    assert (
        topk_topp_sampler.apply_top_k_top_p
        is topk_topp_sampler.apply_top_k_top_p_pytorch
    )
    assert topk_topp_sampler.TopKTopPSampler.forward_native is native_forward


@pytest.mark.parametrize("marker", ["hf", "additional", None])
def test_mixed_graph_default_requires_identity(platform, marker):
    config = make_config(marker)
    cosyvoice3.apply_graph_default(config, platform)
    expected = CUDAGraphMode.FULL_AND_PIECEWISE if marker else None
    assert config.compilation_config.cudagraph_mode == expected


@pytest.mark.parametrize(
    "mode",
    [CUDAGraphMode.NONE, CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL_DECODE_ONLY],
)
def test_mixed_graph_default_preserves_explicit_mode(platform, mode):
    config = make_config()
    config.compilation_config.cudagraph_mode = mode
    cosyvoice3.apply_graph_default(config, platform)
    assert config.compilation_config.cudagraph_mode == mode


@pytest.mark.parametrize(
    "change",
    ["eager", "no_compile", "other_backend", "other_device", "tp", "fp16", "runtime"],
)
def test_mixed_graph_default_preserves_unsupported_configs(
    platform, change, monkeypatch
):
    config = make_config()
    if change == "eager":
        config.model_config.enforce_eager = True
    elif change == "no_compile":
        config.compilation_config.mode = CompilationMode.NONE
    elif change == "other_backend":
        config.attention_config.backend = "FLASH_ATTN"
    elif change == "other_device":
        platform.torch_device_fn.get_device_name.return_value = "MTT S4000"
    elif change == "tp":
        config.parallel_config.tensor_parallel_size = 2
    elif change == "fp16":
        config.model_config.dtype = torch.float16
    elif change == "runtime":
        monkeypatch.setattr(vllm, "__version__", "0.24.0")
    cosyvoice3.apply_graph_default(config, platform)
    assert config.compilation_config.cudagraph_mode is None
    assert not cosyvoice3.supports_mixed_graph(config, platform)
