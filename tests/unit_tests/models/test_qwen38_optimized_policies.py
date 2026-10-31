# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for the optimized paths; GPU arithmetic stays in pinned libraries."""

import importlib.util
import sys
from bisect import bisect_right
from types import ModuleType, SimpleNamespace as NS

import pytest
import torch
from _qwen38_cpu import ROOT, declarations, sampling_metadata

VENDOR = ROOT / "vllm_fl/dispatch/backends/vendor/metax/impl/qwen3_8_flash_next"
moe = declarations(VENDOR / "mctlass_moe.py", "qwen_review_moe")
rows = declarations(
    ROOT / "vllm_fl/models/qwen3_8_flash_next/common/qsa_packing.py",
    "qwen_review_packing",
    names={"qsa_pack_row_groups"},
)


def tensor(shape, dtype=torch.bfloat16):
    return NS(
        shape=shape,
        ndim=len(shape),
        dtype=dtype,
        device=torch.device("cuda:0"),
        is_contiguous=lambda: True,
    )


@pytest.mark.parametrize(
    "m,expected",
    [
        (1, False),
        (30, False),
        (31, True),
        (64, True),
        (65, True),
        (1023, True),
        (1024, True),
        (2048, True),
        (2049, False),
    ],
)
def test_closed_moe_bounds_and_quantized_fallback(m, expected):
    args = (
        tensor((m, 2560)),
        tensor((512, 160, 2560)),
        tensor((512, 2560, 80)),
        10,
        "MetaX C550",
        "silu",
    )
    assert moe.moe_eligible(*args, False, False, False, None, 512, False) is expected
    assert not moe.moe_eligible(*args, True, False, False, None, 512, False)
    assert not moe.moe_eligible(*args, False, True, False, None, 512, False)


@pytest.mark.parametrize("m,stages", [(64, 1), (1024, 2)])
def test_closed_moe_preserves_small_m_g2_fallback(monkeypatch, m, stages):
    calls = []
    provider = ModuleType("flaggems_vllm.runtime.backend._metax.ops.mctlass_bf16")
    provider.get_plan = lambda *args, **kwargs: (
        calls.append((args, kwargs)) or NS(block_m=32)
    )
    monkeypatch.setitem(sys.modules, provider.__name__, provider)
    x, w1, w2, c1, act, c2 = (
        tensor((m, 2560)),
        object(),
        object(),
        object(),
        object(),
        object(),
    )
    a, b = moe.prepare_moe_stages(x, w1, w2, c1, act, c2, eligible=True)
    assert a is not None and (b is not None) == (stages == 2)
    assert len(calls) == stages and calls[0][1] == {"topk": 10}
    if stages == 2:
        assert calls[1][0] == (act, w2, c2) and calls[1][1] == {
            "topk": 1,
            "weighted": True,
        }


def test_linear_preserves_native_fallback_and_bias(monkeypatch):
    calls = []

    class Native:
        def apply(self, layer, x, bias=None):
            calls.append((layer, x, bias))
            return "native"

    module = declarations(
        VENDOR / "mctlass_linear.py",
        "qwen_review_linear",
        {"UnquantizedLinearMethod": Native},
    )
    obj = module.Qwen38MctlassLinearMethod()
    layer = NS(weight=object())
    x = object()
    monkeypatch.setattr(module, "linear_or_none", lambda *args: "closed")
    assert obj.apply(layer, x) == "closed" and not calls
    assert obj.apply(layer, x, bias=7) == "native" and calls[-1] == (layer, x, 7)
    monkeypatch.setattr(module, "linear_or_none", lambda *args: None)
    assert obj.apply(layer, x) == "native"


def test_sampling_contract_tracks_batch_metadata_and_is_idempotent():
    metadata = NS(top_k=torch.tensor([20, 20]))
    batch = NS(
        num_reqs=2,
        top_k_cpu_tensor=torch.tensor([20, 20]),
        sampling_metadata=metadata,
        _make_sampling_metadata=lambda: metadata,
    )
    sampling_metadata.install_sampling_contract(batch)
    installed = batch._make_sampling_metadata
    sampling_metadata.install_sampling_contract(batch)
    assert batch._make_sampling_metadata is installed
    assert batch._make_sampling_metadata().top_k._fl_all_topk20
    batch.top_k_cpu_tensor[1] = 7
    assert not batch._make_sampling_metadata().top_k._fl_all_topk20
    batch.num_reqs = 1
    assert batch._make_sampling_metadata().top_k._fl_all_topk20


def test_packing_covers_rows_without_crossing_request_or_kv_group():
    starts = [0, 5, 6, 10]
    lengths = [8, 12, 21]
    groups = rows.qsa_pack_row_groups(starts, lengths, 12)
    assert sorted(x for group in groups for x in group if x >= 0) == list(range(12))
    for group in groups:
        real = [x for x in group if 0 <= x < starts[-1]]
        owners = [bisect_right(starts, x) - 1 for x in real]
        positions = [
            lengths[o] - (starts[o + 1] - starts[o]) + x - starts[o]
            for x, o in zip(real, owners)
        ]
        assert len(set(owners)) <= 1
        assert len({p // 4 for p in positions}) <= 1


def test_native_rng_context_restores_after_an_exception():
    native = NS()
    seen = []
    module = None

    def sample(*args):
        seen.append(module._POSITIVE_NATIVE_NOISE.get())
        raise RuntimeError("native failure")

    native.random_sample = sample
    native.sample_with_exponential_noise = lambda *args: "native_noise"
    native.apply_top_k_top_p_pytorch = lambda *args: "native_filter"
    module = declarations(
        ROOT / "vllm_fl/dispatch/backends/vendor/metax/patches/topk_topp_sampler.py",
        "qwen_review_sampling",
        {"topk_topp_sampler": native, "fl_envs": NS(VLLM_FL_USE_FLAGGEMS_VLLM=True)},
    )
    with pytest.raises(RuntimeError, match="native failure"):
        module._random_sample(tensor((64, 248320), torch.float32), {})
    assert seen == [True] and not module._POSITIVE_NATIVE_NOISE.get()
    assert module._sample_with_exponential_noise(object(), object()) == "native_noise"


def test_packing_metadata_cache_is_scoped_and_cleared_on_error(monkeypatch):
    path = ROOT / "vllm_fl/models/qwen3_8_flash_next/common/qsa_packing.py"
    spec = importlib.util.spec_from_file_location("qwen_review_pack_cache", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = []

    class Rows:
        def to(self, *args, **kwargs):
            return self

    def make(*args, **kwargs):
        calls.append(1)
        return Rows()

    # Only the metadata upload is substituted; the real grouping and ContextVar run.
    monkeypatch.setattr(module, "torch", NS(tensor=make, int32=torch.int32))
    starts = torch.tensor([0, 3])
    lengths = torch.tensor([5])
    with pytest.raises(RuntimeError, match="scope"):
        with module.qsa_metadata_build_scope():
            a = module._cached_pack_rows(starts, lengths, 3, "cpu")
            assert module._cached_pack_rows(starts, lengths, 3, "cpu") is a
            raise RuntimeError("scope")
    assert module._QSA_PACK_ROWS_CACHE.get() is None
    with module.qsa_metadata_build_scope():
        assert module._cached_pack_rows(starts, lengths, 3, "cpu") is not a
    assert len(calls) == 2


def test_runner_import_resolves_real_ple_state_without_gpu_initialization():
    package = ModuleType("qwen_review_worker_package")
    package.__path__ = [str(ROOT / "vllm_fl/worker")]
    sys.modules[package.__name__] = package
    from importlib import import_module

    module = import_module(package.__name__ + ".qwen3_8_flash_next")
    assert module._mrv1_prepare.__module__.endswith("qwen3_8_flash_next_ple")
    assert module.install_sampling_contract.__module__.endswith(
        "qwen3_8_flash_next_sampling"
    )
    assert not torch.cuda.is_initialized()
