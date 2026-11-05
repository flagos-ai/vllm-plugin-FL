# Copyright (c) 2026 BAAI. All rights reserved.

"""Native GCU MoE sum values, layouts and graph replay at the prefill boundary."""

import hashlib
import importlib.metadata
import inspect
from pathlib import Path

import pytest
import torch

pytestmark = [pytest.mark.gpu, pytest.mark.flaggems]

_TOPK = 8
_HIDDEN_SIZE = 2048


def _provider_identity(export, provider):
    # This observes only the actual selected source file, not a package tree.
    source = inspect.getsourcefile(inspect.unwrap(provider))
    assert source is not None, "The selected MoE provider must have a source file"
    path = Path(source)
    before = path.stat()
    with path.open("rb") as stream:
        raw = stream.read()
        eof = stream.read(1) == b""
    after = path.stat()
    return {
        "export": export,
        "module": provider.__module__,
        "qualname": provider.__qualname__,
        "flag_gems_version": importlib.metadata.version("flag-gems"),
        "flaggems_vllm_version": importlib.metadata.version("flaggems-vllm"),
        "source_file": str(path),
        "source_bytes": len(raw),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "source_read_eof": eof,
        "source_stat_stable": (
            before.st_dev == after.st_dev
            and before.st_ino == after.st_ino
            and before.st_size == after.st_size == len(raw)
            and before.st_mtime_ns == after.st_mtime_ns
        ),
        "source_mtime_ns_before": before.st_mtime_ns,
        "source_mtime_ns_after": after.st_mtime_ns,
    }


def _values(num_tokens, dtype, revision, topk=_TOPK):
    generator = torch.Generator(device="cpu").manual_seed(3817 + revision)
    # Dyadic values make all FP32 partial sums exact. For FP32, add small
    # 2**-16 terms to a 1.0 base so reduced-precision accumulation loses bits.
    # BF16/FP16 cases use 1/16 terms whose sums are exact in their output dtype.
    values = torch.randint(
        -16,
        17,
        (num_tokens, topk, _HIDDEN_SIZE),
        generator=generator,
        dtype=torch.int16,
        device="cpu",
    ).to(dtype)
    if dtype == torch.float32:
        values.div_(65536)
        values[:, 0, :].add_(1)
    else:
        values.div_(16)
    return values


def _buffers(values, device, input_strided, output_strided):
    num_tokens = values.shape[0]
    if input_strided:
        input_storage = torch.full(
            (num_tokens, values.shape[1], _HIDDEN_SIZE * 2),
            float("nan"),
            dtype=values.dtype,
            device=device,
        )
        inp = input_storage[..., ::2]
        inp.copy_(values)
    else:
        inp = values.to(device)
        input_storage = inp
    output_storage = torch.full(
        (num_tokens, _HIDDEN_SIZE * (2 if output_strided else 1)),
        float("nan"),
        dtype=values.dtype,
        device=device,
    )
    out = output_storage[..., ::2] if output_strided else output_storage
    return inp, out, input_storage, output_storage


def _assert_result(inp, out, values, device, input_storage, output_storage, identity):
    assert out.device == inp.device == device
    assert out.dtype == inp.dtype == values.dtype
    assert tuple(out.shape) == (values.shape[0], _HIDDEN_SIZE)
    assert (
        id(out),
        inp.data_ptr(),
        out.data_ptr(),
        tuple(inp.stride()),
        tuple(out.stride()),
    ) == identity
    expected = values.to(torch.float32).sum(dim=1).to(values.dtype)
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)
    torch.testing.assert_close(inp.cpu(), values, rtol=0, atol=0)
    if input_storage is not inp:
        assert bool(torch.isnan(input_storage[..., 1::2].cpu()).all())
    if output_storage is not out:
        assert bool(torch.isnan(output_storage[..., 1::2].cpu()).all())


def _check_moe_sum_gcu(
    backend,
    device,
    monkeypatch,
    record_property,
    num_tokens,
    dtype,
    input_strided,
    output_strided,
    *,
    use_graph=False,
    topk=_TOPK,
):
    if backend != "enflame":
        pytest.skip("Requires the Enflame GCU backend")
    assert device.type == "gcu", "Enflame tests must use an actual GCU device"
    assert torch.gcu.is_available(), "The selected GCU runtime is unavailable"

    # On GCU missing dependencies or native failures must fail, not skip.
    import flag_gems
    import flaggems_vllm

    from vllm_fl.dispatch.backends.flaggems.impl.fused_moe import moe_sum_flaggems

    calls = []

    def observe(export, provider):
        def call(*args, **kwargs):
            if not calls:
                identity = _provider_identity(export, provider)
                record_property("gcu_moe_sum_provider", identity)
                assert identity["source_read_eof"]
                assert identity["source_stat_stable"]
            calls.append((export, provider.__module__))
            return provider(*args, **kwargs)

        return call

    for package in (flag_gems, flaggems_vllm):
        monkeypatch.setattr(
            package,
            "moe_sum",
            observe(f"{package.__name__}.moe_sum", package.moe_sum),
        )

    record_property(
        "gcu_moe_sum_case",
        {
            "adapter": "moe_sum_flaggems",
            "device": str(device),
            "num_tokens": num_tokens,
            "topk": topk,
            "hidden_size": _HIDDEN_SIZE,
            "dtype": str(dtype),
            "input_hidden_stride": 2 if input_strided else 1,
            "output_hidden_stride": 2 if output_strided else 1,
            "graph": use_graph,
            "reference": "CPU FP32 sum; exact dyadic inputs, FP32 low-bit terms",
            "rtol": 0,
            "atol": 0,
        },
    )
    initial_values = _values(num_tokens, dtype, revision=0, topk=topk)
    inp, out, input_storage, output_storage = _buffers(
        initial_values, device, input_strided, output_strided
    )
    identity = (
        id(out),
        inp.data_ptr(),
        out.data_ptr(),
        tuple(inp.stride()),
        tuple(out.stride()),
    )

    def apply():
        assert moe_sum_flaggems(inp, out) is None

    def check(values):
        _assert_result(
            inp, out, values, device, input_storage, output_storage, identity
        )

    if use_graph:
        apply()  # Finish native compilation/autotuning before capture.
        torch.gcu.synchronize()
        check(initial_values)
        graph = torch.gcu.GCUGraph()
        with torch.gcu.graph(graph):
            apply()
        for revision in (1, 2):
            replay_values = _values(num_tokens, dtype, revision, topk=topk)
            inp.copy_(replay_values)
            out.fill_(float("nan"))
            graph.replay()
            torch.gcu.synchronize()
            check(replay_values)
        record_property("gcu_moe_sum_graph_replays", 2)
    else:
        apply()
        torch.gcu.synchronize()
        check(initial_values)

    expected_export = (
        "flaggems_vllm.moe_sum"
        if input_strided or output_strided
        else "flag_gems.moe_sum"
    )
    assert calls, "The production adapter did not invoke an observed provider"
    assert {export for export, _ in calls} == {expected_export}
    if expected_export == "flag_gems.moe_sum":
        assert all(
            module.startswith(("_enflame.", "flag_gems.runtime.backend._enflame."))
            for _, module in calls
        )
    else:
        assert {module for _, module in calls} == {"flaggems_vllm.ops.moe_sum"}


@pytest.mark.parametrize(
    "num_tokens,dtype,input_strided,output_strided",
    [
        pytest.param(2040, torch.bfloat16, False, False, id="bf16-last-valid-mt8"),
        pytest.param(2041, torch.bfloat16, False, False, id="bf16-first-invalid-mt8"),
        pytest.param(2048, torch.bfloat16, False, False, id="bf16-profile-prefill"),
        pytest.param(4096, torch.bfloat16, False, False, id="bf16-larger-prefill"),
        pytest.param(2048, torch.float16, False, False, id="fp16-profile-prefill"),
        pytest.param(2048, torch.float32, False, False, id="fp32-profile-prefill"),
        pytest.param(2048, torch.bfloat16, True, False, id="bf16-strided-input"),
        pytest.param(2048, torch.bfloat16, False, True, id="bf16-strided-output"),
    ],
)
def test_moe_sum_gcu_values_and_layouts(
    backend,
    device,
    monkeypatch,
    record_property,
    num_tokens,
    dtype,
    input_strided,
    output_strided,
):
    _check_moe_sum_gcu(
        backend,
        device,
        monkeypatch,
        record_property,
        num_tokens,
        dtype,
        input_strided,
        output_strided,
    )


@pytest.mark.parametrize(
    "dtype,strided",
    [
        pytest.param(torch.bfloat16, False, id="bf16-official-contiguous"),
        pytest.param(torch.float32, True, id="fp32-fused-strided-both"),
    ],
)
def test_moe_sum_gcu_graph(
    backend, device, monkeypatch, record_property, dtype, strided
):
    _check_moe_sum_gcu(
        backend,
        device,
        monkeypatch,
        record_property,
        2048,
        dtype,
        strided,
        strided,
        use_graph=True,
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
def test_moe_sum_gcu_small_odd_topk(
    backend, device, monkeypatch, record_property, dtype
):
    _check_moe_sum_gcu(
        backend,
        device,
        monkeypatch,
        record_property,
        7,
        dtype,
        False,
        False,
        topk=7,
    )
