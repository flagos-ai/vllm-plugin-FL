# Copyright (c) 2026 BAAI. All rights reserved.

"""Validate the production GCU RoPE adapter against independent CPU math."""

import pytest
import torch

pytestmark = [pytest.mark.gpu, pytest.mark.flaggems]


def _rotary_reference(value, cos, sin, positions, interleaved):
    """Compute the reference entirely on CPU before invoking the provider."""
    assert value.device.type == cos.device.type == sin.device.type == "cpu"
    assert positions.device.type == "cpu"
    value = value.float()
    cos, sin = cos[positions].float()[:, None], sin[positions].float()[:, None]
    if interleaved:
        first, second = value[..., ::2], value[..., 1::2]
        return torch.stack(
            (first * cos - second * sin, second * cos + first * sin), dim=-1
        ).flatten(-2)
    first, second = value.chunk(2, dim=-1)
    return torch.cat((first * cos - second * sin, second * cos + first * sin), dim=-1)


def _assert_output(actual, expected, device, dtype, label):
    assert actual.shape == expected.shape, label
    assert actual.dtype == dtype, label
    assert actual.device == device, label
    tolerance = 1e-2 if dtype == torch.bfloat16 else 1e-3
    torch.testing.assert_close(
        actual.cpu(),
        expected,
        rtol=tolerance,
        atol=tolerance,
        msg=lambda message: f"{label}\n{message}",
    )


def _check_rotary_gcu_values_and_mutation(
    backend,
    device,
    dtype,
    interleaved,
    inplace,
    packed_views,
    record_property,
    *,
    use_graph=False,
):
    if backend != "enflame":
        pytest.skip("Requires the Enflame GCU backend")
    assert device.type == "gcu", "Enflame tests must use an actual GCU device"
    assert torch.gcu.is_available(), "The selected GCU runtime is unavailable"

    # Missing dependencies on GCU are failures, not successful skips. Test the
    # production adapter, retaining an independent CPU reference rather than
    # treating the known-broken raw provider's inplace path as a reference.
    from flaggems_vllm import apply_rotary_pos_emb

    from vllm_fl.dispatch.backends.flaggems.impl.rotary import rotary_embedding_flaggems

    label = (
        f"rotary_embedding_flaggems via flaggems_vllm dtype={dtype} "
        f"interleaved={interleaved} inplace={inplace} packed_views={packed_views} "
        f"graph={use_graph}"
    )
    record_property(
        "gcu_rotary_provider",
        {
            "export": "flaggems_vllm.apply_rotary_pos_emb",
            "adapter": "rotary_embedding_flaggems",
            "graph": use_graph,
            "module": apply_rotary_pos_emb.__module__,
            "qualname": apply_rotary_pos_emb.__qualname__,
            "device": str(device),
            "dtype": str(dtype),
            "interleaved": interleaved,
            "inplace": inplace,
            "packed_views": packed_views,
        },
    )
    generator = torch.Generator(device="cpu").manual_seed(17)
    original = torch.randn((7, 6, 160), generator=generator, device="cpu").to(dtype)
    angles = torch.randn((16, 40), generator=generator, device="cpu")
    cos_cpu, sin_cpu = angles.cos().to(dtype), angles.sin().to(dtype)
    positions_cpu = torch.tensor([3, 0, 7, 4, 9, 1, 5], device="cpu")
    expected = [
        _rotary_reference(value, cos_cpu, sin_cpu, positions_cpu, interleaved).to(dtype)
        for value in (original[:, :4, :80], original[:, 4:, :80])
    ]

    packed = original.to(device).clone()
    query, key = packed[:, :4, :80], packed[:, 4:, :80]
    if not packed_views:
        query, key = query.contiguous(), key.contiguous()
    before = [query.clone(), key.clone()]
    original_strides = [query.stride(), key.stride()]
    cos, sin = cos_cpu.to(device), sin_cpu.to(device)
    positions = positions_cpu.to(device)

    def apply():
        return rotary_embedding_flaggems(
            None,
            query,
            key,
            cos,
            sin,
            position_ids=positions,
            rotary_interleaved=interleaved,
            inplace=inplace,
        )

    if use_graph:
        # Compile before capture, then reset the same buffers before each replay.
        # Since outputs alias these buffers, a no-op replay cannot pass the
        # value assertions by retaining output from warmup or capture.
        apply()
        query.copy_(before[0])
        key.copy_(before[1])
        torch.gcu.synchronize()
        graph = torch.gcu.GCUGraph()
        with torch.gcu.graph(graph):
            actual = apply()
        for _ in range(2):
            query.copy_(before[0])
            key.copy_(before[1])
            graph.replay()
        torch.gcu.synchronize()
        record_property("gcu_rotary_graph_replays", 2)
    else:
        actual = apply()
    assert isinstance(actual, tuple) and len(actual) == 2, label
    for name, output, value, saved, reference, stride in zip(
        ("query", "key"), actual, (query, key), before, expected, original_strides
    ):
        tensor_label = f"{label} tensor={name}"
        _assert_output(output, reference, device, dtype, tensor_label)
        assert value.stride() == stride, tensor_label
        if inplace:
            assert output.data_ptr() == value.data_ptr(), tensor_label
            assert output.stride() == stride, tensor_label
            _assert_output(value, reference, device, dtype, tensor_label)
        else:
            assert output.data_ptr() != value.data_ptr(), tensor_label
            torch.testing.assert_close(
                value,
                saved,
                rtol=0,
                atol=0,
                msg=lambda message, label=tensor_label: (
                    f"{label} input mutated\n{message}"
                ),
            )
    if packed_views and inplace:
        torch.testing.assert_close(
            packed[:, :, 80:].cpu(),
            original[:, :, 80:],
            rtol=0,
            atol=0,
            msg=lambda message: f"{label} non-RoPE tail mutated\n{message}",
        )
    else:
        torch.testing.assert_close(
            packed.cpu(),
            original,
            rtol=0,
            atol=0,
            msg=lambda message: f"{label} packed input mutated\n{message}",
        )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize(
    "interleaved", [False, True], ids=["split-half", "interleaved"]
)
@pytest.mark.parametrize("inplace", [False, True], ids=["out-of-place", "inplace"])
@pytest.mark.parametrize("packed_views", [False, True], ids=["contiguous", "packed"])
def test_flaggems_vllm_rotary_gcu_values_and_mutation(
    backend, device, dtype, interleaved, inplace, packed_views, record_property
):
    _check_rotary_gcu_values_and_mutation(
        backend, device, dtype, interleaved, inplace, packed_views, record_property
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("packed_views", [False, True], ids=["contiguous", "packed"])
def test_rotary_gcu_split_half_inplace_graph(
    backend, device, dtype, packed_views, record_property
):
    _check_rotary_gcu_values_and_mutation(
        backend,
        device,
        dtype,
        False,
        True,
        packed_views,
        record_property,
        use_graph=True,
    )
