# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math

import flaggems_vllm
import pytest
import torch

from vllm_fl.third_party.attn_res_pr737.attn_res import attn_res as pr737_attn_res


def gems_assert_close(actual, expected, dtype, atol):
    # Identical comparator and tolerance to the upstream PR737 tests.
    flaggems_vllm.testing.assert_close(actual, expected, dtype, atol=atol)


EPS = 1e-5
HIDDEN_SIZE = 7168
MAX_BLOCKS = 8


def _is_hopper() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0)


requires_hopper = pytest.mark.skipif(not _is_hopper(), reason="requires CUDA SM90")


def _randn_with_row_padding(
    *shape: int,
    padding: int = 0,
    offset: int = 0,
    scale: float = 1.0,
) -> torch.Tensor:
    if not 0 <= offset <= padding:
        raise ValueError("offset must be in [0, padding]")
    storage = torch.randn(
        *shape[:-1],
        shape[-1] + padding,
        device=flaggems_vllm.device,
        dtype=torch.bfloat16,
    )
    storage.mul_(scale)
    return storage[..., offset : offset + shape[-1]]


def _make_inputs(
    num_tokens: int,
    has_delta: bool,
    apply_output_norm: bool,
    row_padding: int = 0,
    row_offset: int = 0,
    hidden_size: int = HIDDEN_SIZE,
) -> tuple[torch.Tensor, ...]:
    prefix = _randn_with_row_padding(
        num_tokens,
        hidden_size,
        padding=row_padding,
        offset=row_offset,
    )
    delta = (
        _randn_with_row_padding(
            num_tokens,
            hidden_size,
            padding=row_padding,
            offset=row_offset,
        )
        if has_delta
        else None
    )
    blocks = _randn_with_row_padding(
        num_tokens,
        MAX_BLOCKS,
        hidden_size,
        padding=row_padding,
        offset=row_offset,
    )
    norm_weight = 1 + _randn_with_row_padding(hidden_size, scale=0.1)
    qk_weight = _randn_with_row_padding(
        hidden_size,
        scale=1 / math.sqrt(hidden_size),
    )
    output_norm_weight = (
        1 + _randn_with_row_padding(hidden_size, scale=0.1)
        if apply_output_norm
        else None
    )
    return prefix, delta, blocks, norm_weight, qk_weight, output_norm_weight


def _reference(
    prefix: torch.Tensor,
    delta: torch.Tensor | None,
    blocks: torch.Tensor,
    norm_weight: torch.Tensor,
    qk_weight: torch.Tensor,
    output_norm_weight: torch.Tensor | None,
    num_blocks: int,
    block_write_idx: int,
    eps: float,
    output_norm_eps: float,
) -> torch.Tensor:
    if delta is not None:
        updated_prefix = (prefix.float() + delta.float()).to(prefix.dtype)
        prefix.copy_(updated_prefix)
    else:
        updated_prefix = prefix

    if block_write_idx >= 0:
        blocks[:, block_write_idx].copy_(updated_prefix)

    values = torch.cat(
        (blocks[:, :num_blocks], updated_prefix.unsqueeze(1)),
        dim=1,
    ).float()
    reciprocal_std = torch.rsqrt(values.square().mean(dim=-1) + eps)
    logits = (
        values * reciprocal_std.unsqueeze(-1) * norm_weight.float() * qk_weight.float()
    ).sum(dim=-1)
    probabilities = logits.softmax(dim=-1)
    output = (probabilities.unsqueeze(-1) * values).sum(dim=1)
    if output_norm_weight is not None:
        output = (
            output
            * torch.rsqrt(output.square().mean(dim=-1, keepdim=True) + output_norm_eps)
            * output_norm_weight.float()
        )
    return output.to(prefix.dtype)


def _run_and_check(
    num_tokens: int,
    num_blocks: int,
    block_write_idx: int,
    has_delta: bool,
    apply_output_norm: bool,
    row_padding: int = 0,
    row_offset: int = 0,
    hidden_size: int = HIDDEN_SIZE,
) -> None:
    torch.manual_seed(2026)
    args = _make_inputs(
        num_tokens,
        has_delta,
        apply_output_norm,
        row_padding,
        row_offset,
        hidden_size,
    )
    prefix, delta, blocks, norm_weight, qk_weight, output_norm_weight = args
    expected_prefix = prefix.clone()
    expected_blocks = blocks.clone()
    expected = _reference(
        expected_prefix,
        delta,
        expected_blocks,
        norm_weight,
        qk_weight,
        output_norm_weight,
        num_blocks,
        block_write_idx,
        EPS,
        2 * EPS,
    )

    actual = pr737_attn_res(
        prefix,
        delta,
        blocks,
        norm_weight,
        qk_weight,
        output_norm_weight,
        num_blocks,
        block_write_idx,
        EPS,
        2 * EPS,
    )

    gems_assert_close(actual, expected, torch.bfloat16, atol=4e-2)
    torch.testing.assert_close(prefix, expected_prefix, atol=0, rtol=0)
    torch.testing.assert_close(blocks, expected_blocks, atol=0, rtol=0)
    assert actual.is_contiguous()


pytestmark = [pytest.mark.gpu, requires_hopper]


ATTN_RES_CASES = [
    pytest.param(1, 0, 0, False, True, HIDDEN_SIZE, 0, 0, id="block0-write"),
    pytest.param(3, 4, 4, True, True, HIDDEN_SIZE, 0, 0, id="write"),
    pytest.param(1, 2, 2, True, True, HIDDEN_SIZE, 0, 0, id="write-b2"),
    pytest.param(17, 5, -1, False, True, HIDDEN_SIZE, 0, 0, id="post"),
    pytest.param(64, 1, -1, False, True, HIDDEN_SIZE, 0, 0, id="post-n64-b1"),
    pytest.param(64, 4, -1, True, True, HIDDEN_SIZE, 0, 0, id="common"),
    pytest.param(64, 3, -1, True, True, HIDDEN_SIZE, 0, 0, id="common-b3"),
    pytest.param(
        5,
        6,
        -1,
        True,
        True,
        HIDDEN_SIZE,
        0,
        0,
        id="common-partial-source-tile",
    ),
    pytest.param(256, 4, -1, True, True, HIDDEN_SIZE, 0, 0, id="persistent-common"),
    pytest.param(256, 3, 3, True, True, HIDDEN_SIZE, 0, 0, id="persistent-write"),
    pytest.param(256, 3, -1, False, True, HIDDEN_SIZE, 0, 0, id="post-b3"),
    pytest.param(
        1,
        1,
        -1,
        False,
        False,
        HIDDEN_SIZE,
        0,
        0,
        id="post-without-output-norm",
    ),
    pytest.param(5, 4, 2, True, True, HIDDEN_SIZE, 0, 0, id="rewrite-attended-block"),
    pytest.param(
        5, 2, 1, True, True, HIDDEN_SIZE, 0, 0, id="rewrite-attended-block-b2"
    ),
    pytest.param(
        256,
        4,
        -1,
        False,
        False,
        HIDDEN_SIZE,
        0,
        0,
        id="read-only-without-output-norm",
    ),
    pytest.param(512, 8, -1, True, False, HIDDEN_SIZE, 0, 0, id="final"),
    pytest.param(4096, 8, -1, True, True, HIDDEN_SIZE, 0, 0, id="prefill-n4096-common"),
    pytest.param(4096, 7, 7, True, True, HIDDEN_SIZE, 0, 0, id="prefill-n4096-write"),
    pytest.param(4096, 8, -1, True, False, HIDDEN_SIZE, 0, 0, id="prefill-n4096-final"),
    pytest.param(1, 0, -1, False, True, HIDDEN_SIZE, 0, 0, id="post-b0"),
    pytest.param(1, 2, -1, False, True, HIDDEN_SIZE, 0, 0, id="post-b2"),
    pytest.param(64, 3, -1, False, True, HIDDEN_SIZE, 0, 0, id="post-n64-b3"),
    pytest.param(1, 7, -1, False, True, HIDDEN_SIZE, 0, 0, id="post-b7"),
    pytest.param(3, 2, -1, False, True, 1024, 0, 0, id="hidden-1024-post"),
    pytest.param(3, 4, -1, True, True, 1024, 0, 0, id="hidden-1024-common"),
    pytest.param(3, 2, -1, False, True, 4096, 0, 0, id="hidden-4096-post"),
    pytest.param(3, 4, -1, True, True, 4096, 0, 0, id="hidden-4096-common"),
    pytest.param(7, 3, 3, True, True, HIDDEN_SIZE, 11, 0, id="padded-write"),
    pytest.param(7, 3, -1, False, True, HIDDEN_SIZE, 11, 0, id="padded-post"),
    pytest.param(
        7,
        3,
        -1,
        False,
        True,
        HIDDEN_SIZE,
        16,
        1,
        id="misaligned-row-base",
    ),
    pytest.param(0, MAX_BLOCKS, -1, True, False, HIDDEN_SIZE, 0, 0, id="empty"),
]


@pytest.mark.parametrize(
    (
        "num_tokens",
        "num_blocks",
        "block_write_idx",
        "has_delta",
        "apply_output_norm",
        "hidden_size",
        "row_padding",
        "row_offset",
    ),
    ATTN_RES_CASES,
)
def test_attn_res(
    num_tokens,
    num_blocks,
    block_write_idx,
    has_delta,
    apply_output_norm,
    hidden_size,
    row_padding,
    row_offset,
):
    _run_and_check(
        num_tokens,
        num_blocks,
        block_write_idx,
        has_delta,
        apply_output_norm,
        row_padding=row_padding,
        row_offset=row_offset,
        hidden_size=hidden_size,
    )


@pytest.mark.parametrize(
    "case",
    [
        "unsupported-dtype",
        "negative-num-blocks",
        "too-many-blocks",
        "negative-write-index",
        "large-write-index",
        "noncontiguous-hidden-dimension",
        "overlapping-prefix",
        "overlapping-delta",
        "overlapping-blocks",
        "mutating-alias",
        "autograd-input",
    ],
)
def test_attn_res_invalid_inputs(case):
    args = list(_make_inputs(2, True, True))
    num_blocks, block_write_idx = 1, -1
    error, message = ValueError, None

    if case == "unsupported-dtype":
        args[0] = args[0].float()
        error, message = NotImplementedError, "bfloat16"
    elif case == "negative-num-blocks":
        num_blocks, message = -1, "num_blocks"
    elif case == "too-many-blocks":
        num_blocks, message = MAX_BLOCKS + 1, "num_blocks"
    elif case == "negative-write-index":
        block_write_idx, message = -2, "block_write_idx"
    elif case == "large-write-index":
        block_write_idx, message = MAX_BLOCKS, "block_write_idx"
    elif case == "noncontiguous-hidden-dimension":
        args[0] = torch.empty(
            (2, HIDDEN_SIZE, 2),
            device="cuda",
            dtype=torch.bfloat16,
        )[:, :, 0]
        message = "last dimension of prefix"
    elif case.startswith("overlapping-"):
        tensor_name = case[len("overlapping-") :]
        tensor_index = {"prefix": 0, "delta": 1, "blocks": 2}[tensor_name]
        if tensor_name == "blocks":
            args[tensor_index] = args[tensor_index][:, :1, :].expand(-1, MAX_BLOCKS, -1)
        else:
            args[tensor_index] = args[tensor_index][:1].expand(2, -1)
        message = rf"{tensor_name} must have a non-overlapping row-major layout"
    elif case == "mutating-alias":
        args[1] = args[0]
        message = "prefix must not overlap delta when mutated"
    elif case == "autograd-input":
        args[0].requires_grad_(True)
        error, message = NotImplementedError, "forward-only"

    with pytest.raises(error, match=message):
        pr737_attn_res(
            *args,
            num_blocks,
            block_write_idx,
            EPS,
            EPS,
        )


@pytest.fixture(scope="module")
def installed_provider():
    import importlib
    import os
    from types import SimpleNamespace

    from vllm_fl.patches.attn_res_provider import (
        attn_res_provider_hits,
        patch_attn_res_provider,
    )

    config = SimpleNamespace(
        architecture="KimiK3ForConditionalGeneration",
        dtype=torch.bfloat16,
        hf_text_config=SimpleNamespace(attn_res_block_size=6, num_hidden_layers=48),
    )
    # Keep the installed provider for the module; never restore just one binding.
    old = os.environ.get("VLLM_FL_ATTN_RES_PROVIDER")
    os.environ["VLLM_FL_ATTN_RES_PROVIDER"] = "1"
    try:
        patch_attn_res_provider(config, torch.device("cuda"))
    finally:
        if old is None:
            os.environ.pop("VLLM_FL_ATTN_RES_PROVIDER", None)
        else:
            os.environ["VLLM_FL_ATTN_RES_PROVIDER"] = old
    model = importlib.import_module("vllm.models.kimi_k3.nvidia.model")
    ops = importlib.import_module("vllm.models.kimi_k3.nvidia.ops")
    native = importlib.import_module("vllm.models.kimi_k3.nvidia.ops.attn_res")
    assert model.attn_res._fl_attn_res_pr737
    assert model.attn_res is ops.attn_res is native.attn_res
    return model.attn_res, attn_res_provider_hits


@pytest.mark.parametrize(
    "num_blocks,write_idx,has_delta,output_norm",
    [
        (0, 0, False, True),
        (3, 3, True, True),
        (3, -1, False, True),
        (4, -1, True, True),
        (8, -1, True, False),
    ],
    ids=["block0-write", "write", "post", "common", "final"],
)
@torch.inference_mode()
def test_installed_provider_cuda_graph(
    installed_provider, num_blocks, write_idx, has_delta, output_norm
):
    dispatch, hits = installed_provider
    args = _make_inputs(64, has_delta, output_norm)
    prefix, delta, blocks, nw, qw, ow = args
    initial_prefix, initial_blocks = prefix.clone(), blocks.clone()
    expected_prefix, expected_blocks = prefix.clone(), blocks.clone()
    call_args = (*args, num_blocks, write_idx, EPS, 2 * EPS)
    # Warm the genuine provider outside capture, including post autotuning.
    warm_stream = torch.cuda.Stream()
    warm_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warm_stream):
        dispatch(*call_args)
    torch.cuda.current_stream().wait_stream(warm_stream)
    prefix.copy_(initial_prefix)
    blocks.copy_(initial_blocks)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = dispatch(*call_args)
    captured_hits = hits()
    assert any(key.endswith(":capture") for key in captured_hits)
    prefix.copy_(initial_prefix)
    blocks.copy_(initial_blocks)
    for _ in range(2):
        expected = _reference(
            expected_prefix,
            delta,
            expected_blocks,
            nw,
            qw,
            ow,
            num_blocks,
            write_idx,
            EPS,
            2 * EPS,
        )
        graph.replay()
        torch.cuda.synchronize()
        gems_assert_close(actual, expected, torch.bfloat16, atol=4e-2)
        torch.testing.assert_close(prefix, expected_prefix, atol=0, rtol=0)
        torch.testing.assert_close(blocks, expected_blocks, atol=0, rtol=0)
    # Replays are GPU execution, and must not fall back to Python/native ops.
    assert hits() == captured_hits


@pytest.mark.parametrize("output_norm", [True, False], ids=["common", "final"])
@torch.inference_mode()
def test_installed_provider_torch_compile(installed_provider, output_norm):
    dispatch, hits = installed_provider
    args = _make_inputs(3, True, output_norm)
    prefix, delta, blocks, nw, qw, ow = args
    expected_prefix, expected_blocks = prefix.clone(), blocks.clone()
    expected = _reference(
        expected_prefix,
        delta,
        expected_blocks,
        nw,
        qw,
        ow,
        8,
        -1,
        EPS,
        2 * EPS,
    )
    before = sum(hits().values())
    compiled = torch.compile(dispatch, fullgraph=True, backend="inductor")
    actual = compiled(*args, 8, -1, EPS, 2 * EPS)
    torch.cuda.synchronize()
    assert sum(hits().values()) > before
    gems_assert_close(actual, expected, torch.bfloat16, atol=4e-2)
    torch.testing.assert_close(prefix, expected_prefix, atol=0, rtol=0)
    torch.testing.assert_close(blocks, expected_blocks, atol=0, rtol=0)


# Integration regression: model weights may retain requires_grad=True while
# vLLM runs under inference_mode. Reject actual autograd use, not these weights.
def test_inference_parameter_weights(installed_provider):
    args = list(_make_inputs(64, True, True))
    for index in (3, 4, 5):
        args[index] = torch.nn.Parameter(args[index])
    call_args = (*args, 4, 4, EPS, 2 * EPS)
    with torch.enable_grad(), pytest.raises(NotImplementedError, match="forward-only"):
        pr737_attn_res(*call_args)
    prefix, delta, blocks, nw, qw, ow = args
    initial_prefix, initial_blocks = prefix.clone(), blocks.clone()
    expected_prefix, expected_blocks = prefix.clone(), blocks.clone()
    with torch.no_grad():
        expected = _reference(
            expected_prefix,
            delta,
            expected_blocks,
            nw,
            qw,
            ow,
            4,
            4,
            EPS,
            2 * EPS,
        )
        raw_actual = pr737_attn_res(*call_args)
        gems_assert_close(raw_actual, expected, torch.bfloat16, atol=4e-2)
        torch.testing.assert_close(prefix, expected_prefix, atol=0, rtol=0)
        torch.testing.assert_close(blocks, expected_blocks, atol=0, rtol=0)
        prefix.copy_(initial_prefix)
        blocks.copy_(initial_blocks)
        dispatch, hits = installed_provider
        before = sum(hits().values())
        actual = dispatch(*call_args)
        torch.cuda.synchronize()
        assert sum(hits().values()) > before
        gems_assert_close(actual, expected, torch.bfloat16, atol=4e-2)
        torch.testing.assert_close(prefix, expected_prefix, atol=0, rtol=0)
        torch.testing.assert_close(blocks, expected_blocks, atol=0, rtol=0)
