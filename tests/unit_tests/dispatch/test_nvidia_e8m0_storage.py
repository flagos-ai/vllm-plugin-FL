# Copyright (c) 2026 BAAI. All rights reserved.

"""NVIDIA's real Gems policy must preserve Humming scale storage operations."""

import pytest
import torch
import torch.nn.functional as F

from tests.utils.device_utils import get_backend

from vllm_fl.dispatch.config.utils import load_platform_config

pytestmark = pytest.mark.skipif(
    get_backend() != "nvidia", reason="NVIDIA FlagGems E8M0 storage regression"
)

_E8M0 = torch.float8_e8m0fnu
_DTYPES = [_E8M0, torch.bfloat16]


@pytest.fixture(params=["nvidia", "nvidia_hopper"])
def gems_policy(request, monkeypatch):
    import flag_gems

    def fail_registration(error):
        raise RuntimeError(f"FlagGems registration failed: {error}")

    # A failed Gems registration must not produce a native-only false pass.
    monkeypatch.setattr(flag_gems.runtime.error, "register_error", fail_registration)
    monkeypatch.delenv("VLLM_FL_HOPPER_LONG_CONTEXT_OPT", raising=False)
    config = load_platform_config(request.param)
    return flag_gems, config["flagos_blacklist"]


def _scales(dtype):
    # Valid nonzero E8M0 powers of two. Build their exact bytes so
    # the fixture does not depend on the dtype conversion under test.
    raw = torch.tensor(
        [[126, 127, 128, 129], [130, 129, 128, 127], [126, 128, 130, 132]],
        dtype=torch.uint8,
    )
    if dtype == _E8M0:
        return raw.cuda().view(_E8M0)
    return torch.tensor(
        [[0.5, 1.0, 2.0, 4.0], [8.0, 4.0, 2.0, 1.0], [0.5, 2.0, 8.0, 32.0]],
        dtype=dtype,
        device="cuda",
    )


def _assert_storage_equal(actual, expected):
    assert actual.dtype == expected.dtype
    if actual.dtype == _E8M0:
        torch.testing.assert_close(
            actual.view(torch.uint8), expected.view(torch.uint8), rtol=0, atol=0
        )
    else:
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_nvidia_policy_keeps_native_e8m0_storage(gems_policy):
    _, blacklist = gems_policy
    for name in ("pad", "constant_pad_nd", "fill_scalar_", "index"):
        assert blacklist.count(name) == 1


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("value", [1.0, 2.0])
@pytest.mark.parametrize("direct_aten", [False, True])
def test_nvidia_e8m0_and_bf16_padding(gems_policy, dtype, value, direct_aten):
    flag_gems, blacklist = gems_policy
    source = _scales(dtype)

    def pad():
        if direct_aten:
            return torch.ops.aten.constant_pad_nd.default(source, [1, 2, 0, 1], value)
        return F.pad(source, (1, 2, 0, 1), value=value)

    expected = pad()
    with flag_gems.use_gems(exclude=blacklist):
        actual = pad()
    _assert_storage_equal(actual, expected)
    if dtype == _E8M0:
        raw = actual.view(torch.uint8)
        assert torch.all(raw[-1] == (127 if value == 1.0 else 128))
        _assert_storage_equal(actual[:3, 1:5], source)


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("value", [1.0, 2.0])
def test_nvidia_e8m0_and_bf16_scalar_fill(gems_policy, dtype, value):
    flag_gems, blacklist = gems_policy
    actual = torch.empty((3, 7), dtype=dtype, device="cuda")
    expected = torch.empty_like(actual).fill_(value)
    with flag_gems.use_gems(exclude=blacklist):
        actual.fill_(value)
    _assert_storage_equal(actual, expected)
    if dtype == _E8M0:
        assert torch.all(actual.view(torch.uint8) == (127 if value == 1.0 else 128))


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", _DTYPES)
def test_nvidia_e8m0_and_bf16_index_and_strided_copy(gems_policy, dtype):
    flag_gems, blacklist = gems_policy
    source = _scales(dtype)
    indices = torch.tensor([2, 0], device="cuda")
    expected = torch.empty((2, 8), dtype=dtype, device="cuda").fill_(2.0)
    expected[:, ::2].copy_(source[indices])
    actual = torch.empty_like(expected).fill_(2.0)
    with flag_gems.use_gems(exclude=blacklist):
        selected = source[indices]
        actual[:, ::2].copy_(selected)
    _assert_storage_equal(actual, expected)
    _assert_storage_equal(selected, source[indices])


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", _DTYPES)
def test_nvidia_scale_storage_cuda_graph_replays_changed_inputs(gems_policy, dtype):
    flag_gems, blacklist = gems_policy
    source = _scales(dtype)
    work = torch.empty((3, 8), dtype=dtype, device="cuda")
    indices = torch.tensor([2, 0], device="cuda")

    def run():
        work.fill_(2.0)
        work[:, ::2].copy_(source)
        return F.pad(work[indices], (1, 2), value=1.0)

    expected = run()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with flag_gems.use_gems(exclude=blacklist):
        with torch.cuda.stream(stream):
            run()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = run()
        graph.replay()
    _assert_storage_equal(actual, expected)
    # Replay uses updated data, including changed indexing, rather than merely
    # checking that the same captured result can be returned twice.
    if dtype == _E8M0:
        source.view(torch.uint8).add_(1)
    else:
        source.mul_(2.0)
    indices.copy_(torch.tensor([1, 2], device="cuda"))
    expected = run()
    graph.replay()
    _assert_storage_equal(actual, expected)
