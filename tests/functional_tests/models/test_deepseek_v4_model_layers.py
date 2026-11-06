# SPDX-License-Identifier: Apache-2.0
"""NPU-only compression and HC layer checks without full models."""

import pytest
import torch
import torch.nn.functional as F

from vllm.platforms import current_platform

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(current_platform.device_type != "npu", reason="Requires Ascend"),
]


@pytest.fixture(scope="module")
def runtime_context():
    from tests.utils.device_utils import ascend_single_rank_runtime
    from vllm.config import (
        CompilationConfig,
        CompilationMode,
        VllmConfig,
    )

    def config_factory():
        return VllmConfig(
            compilation_config=CompilationConfig(
                mode=CompilationMode.NONE, custom_ops=["all"]
            )
        )

    with ascend_single_rank_runtime(config_factory) as config:
        yield config


@pytest.fixture(autouse=True)
def inference_and_dispatch(runtime_context):
    from vllm_fl.dispatch import get_default_manager, with_strict_mode

    with torch.inference_mode(), with_strict_mode():
        assert (
            get_default_manager().get_selected_impl_id("deepseek_v4_compressor")
            == "vendor.ascend"
        )
        for name in ("hc_pre", "hc_post", "hc_head"):
            assert (
                get_default_manager().get_selected_impl_id("deepseek_v4_" + name)
                == "reference.torch"
            )
        yield


def _call(name, *args, **kwargs):
    from vllm_fl.dispatch import call_op

    return call_op("deepseek_v4_" + name, *args, **kwargs)


def _data(shape, scale=0.03, dtype=torch.float32, offset=0):
    size = 1
    for dim in shape:
        size *= dim
    return ((torch.arange(size, device="npu").reshape(shape) + offset) % 31 - 15).to(
        dtype
    ) * scale


def _hc_parameters(hc=3, d=8):
    features = 2 * hc + hc * hc
    return (
        _data((features, hc * d), 0.021),
        torch.tensor([0.7, 1.1, 0.6], device="npu"),
        _data((features,), 0.04, offset=7),
    )


def _pre_oracle(x, fn, scale, base, iterations, rms_eps=1e-6, hc_eps=1e-4):
    hc = x.shape[1]
    flattened = x.float().flatten(1)
    mix = (
        F.linear(flattened, fn.float())
        * (flattened.square().mean(-1, keepdim=True) + rms_eps).rsqrt()
    )
    before = torch.sigmoid(mix[:, :hc] * scale[0] + base[:hc]) + hc_eps
    after = torch.sigmoid(mix[:, hc : 2 * hc] * scale[1] + base[hc : 2 * hc]) * 2
    logits = mix[:, 2 * hc :].reshape(-1, hc, hc) * scale[2] + base[2 * hc :].reshape(
        hc, hc
    )
    matrix = logits.softmax(-1) + hc_eps
    matrix = matrix / (matrix.sum(-2, keepdim=True) + hc_eps)
    for _ in range(iterations - 1):
        matrix = matrix / (matrix.sum(-1, keepdim=True) + hc_eps)
        matrix = matrix / (matrix.sum(-2, keepdim=True) + hc_eps)
    combined = sum(before[:, i, None] * x[:, i].float() for i in range(hc))
    return combined.to(x.dtype), after, matrix


def _post_oracle(x, residual, post, comb):
    # Explicit source-channel → destination-channel sum catches transposition.
    destinations = []
    for dest in range(residual.shape[1]):
        mixed = sum(
            comb[:, source, dest, None] * residual[:, source].float()
            for source in range(residual.shape[1])
        )
        destinations.append(mixed + post[:, dest, None] * x.float())
    return torch.stack(destinations, dim=1).to(residual.dtype)


def _rms(x, weight):
    return (
        x.float()
        * (x.float().square().mean(-1, keepdim=True) + 1e-6).rsqrt()
        * weight.float()
    ).to(x.dtype)


def _compressor(ratio):
    from vllm.model_executor.model_loader.weight_utils import default_weight_loader

    from vllm_fl.models.deepseek_v4 import DeepseekV4Compressor

    module = DeepseekV4Compressor(
        8, 4, ratio, prefix="model.layers.0.self_attn.compressor"
    ).to("npu")
    for index, name in enumerate(("wkv.weight", "wgate.weight", "ape", "norm.weight")):
        param = dict(module.named_parameters())[name]
        value = _data(tuple(param.shape), 0.017, param.dtype, offset=index * 5)
        if name == "norm.weight":
            value = value.abs() + 0.7
        loader = getattr(param, "weight_loader", default_weight_loader)
        loader(param, value)
        torch.testing.assert_close(param, value)
    assert set(dict(module.named_parameters())) == {
        "wkv.weight",
        "wgate.weight",
        "ape",
        "norm.weight",
    }
    assert callable(module.wkv.weight.weight_loader)
    assert callable(module.wgate.weight.weight_loader)
    assert module.ape.dtype == torch.float32
    assert module.ape.shape == (ratio, (2 if ratio == 4 else 1) * 4)
    return module


def _compression_oracle(module, x, ends, ratio, rope):
    d, coff = 4, 2 if ratio == 4 else 1
    kv = F.linear(x.float(), module.wkv.weight.float())
    gates = F.linear(x.float(), module.wgate.weight.float())
    outputs = []
    for end in ends:
        values, scores = [], []
        for position in range(max(0, end - ratio * coff + 1), end + 1):
            half = int(coff == 2 and position >= end - ratio + 1)
            values.append(kv[position, half * d : (half + 1) * d])
            scores.append(
                (gates[position] + module.ape[position % ratio])[
                    half * d : (half + 1) * d
                ]
            )
        weights = torch.stack(scores).softmax(0)
        value = (torch.stack(values) * weights).sum(0)
        value = (
            value * (value.square().mean() + 1e-6).rsqrt() * module.norm.weight.float()
        )
        cosine, sine = rope[(end // ratio) * ratio]
        outputs.append(
            torch.stack(
                [
                    value[0],
                    value[1],
                    value[2] * cosine - value[3] * sine,
                    value[3] * cosine + value[2] * sine,
                ]
            )
        )
    return torch.stack(outputs).to(x.dtype)


@pytest.mark.parametrize("ratio", [4, 128])
@pytest.mark.parametrize("chunked", [False, True])
def test_compressor_actual_module_prefill_decode(ratio, chunked):
    module = _compressor(ratio)
    n, coff = 2 * ratio, 2 if ratio == 4 else 1
    x = _data((n, 8), 0.04, torch.bfloat16)
    positions = torch.arange(n, device="npu")
    slots = torch.cat([positions[:ratio] + ratio, positions[ratio:] - ratio])
    state = torch.zeros(2, ratio, 2 * coff * 4, device="npu")
    table = torch.tensor([[1, 0]], dtype=torch.int32, device="npu")
    angle = torch.arange(n, device="npu").float() * 0.013
    rope = torch.stack([angle.cos(), angle.sin()], -1)
    outputs = []
    slices = (
        [(0, ratio - 1, []), (ratio - 1, ratio, [ratio - 1]), (ratio, n, [n - 1])]
        if chunked
        else [(0, n, [ratio - 1, n - 1])]
    )
    for begin, finish, completed in slices:
        ends = torch.tensor(completed, dtype=torch.int64, device="npu")
        result = module(
            x[begin:finish],
            positions[begin:finish],
            slots[begin:finish],
            state,
            table,
            torch.zeros_like(ends),
            ends,
            rope,
        )
        outputs.append(result)
    actual = torch.cat(outputs)
    expected = _compression_oracle(module, x, [ratio - 1, n - 1], ratio, rope)
    torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.01)
    # Independently check the complete persistent state on CPU, including the
    # physical page permutation. Finite outputs alone cannot detect lost writes
    # during the chunks that return no compressed rows.
    cpu_x = x.cpu().float()
    cpu_kv = F.linear(cpu_x, module.wkv.weight.cpu().float())
    cpu_gate = F.linear(cpu_x, module.wgate.weight.cpu().float())
    cpu_ape = module.ape.cpu()
    expected_state = torch.zeros(state.shape, dtype=torch.float32)
    for index, slot in enumerate(slots.cpu().tolist()):
        expected_state[slot // ratio, slot % ratio] = torch.cat(
            [cpu_kv[index], cpu_gate[index] + cpu_ape[index % ratio]]
        )
    torch.testing.assert_close(state.cpu(), expected_state)


def test_full_compressor_fake_has_no_kernel_launch(monkeypatch):
    from torch._subclasses.fake_tensor import FakeTensorMode
    from triton.runtime.jit import JITFunction

    from vllm_fl.dispatch.backends.vendor.ascend.impl.compressor import (
        _compressor_op,
        deepseek_v4_compressor,
    )

    module = _compressor(4)
    x = _data((2, 8), dtype=torch.bfloat16)
    state = torch.empty(2, 4, 16, dtype=torch.float32, device="npu")
    positions = torch.arange(2, device="npu")
    slots = torch.tensor([0, 1], device="npu")
    table = torch.tensor([[0, 1]], dtype=torch.int32, device="npu")
    ends = torch.tensor([3, 7, -1], device="npu")
    requests = torch.tensor([0, 0, -1], device="npu")
    rope = torch.empty(8, 2, dtype=torch.float32, device="npu")
    args = (
        x,
        module.wkv.weight,
        module.wgate.weight,
        module.ape,
        state,
        positions,
        slots,
        table,
        requests,
        ends,
        module.norm.weight,
        rope,
    )

    def forbidden_launch(*args, **kwargs):
        pytest.fail("FakeTensor evaluation launched a Triton kernel")

    monkeypatch.setattr(JITFunction, "run", forbidden_launch)
    with FakeTensorMode() as mode:
        fake_args = tuple(mode.from_tensor(tensor) for tensor in args)
        result = deepseek_v4_compressor(*fake_args)
        assert result.shape == (3, 4)
        assert result.dtype == x.dtype
        assert result.device == x.device
        assert isinstance(result, torch._subclasses.fake_tensor.FakeTensor)

    # Mutation must be present in the dispatcher schema, where functionalization
    # and graph dead-code elimination can see it, rather than only in a comment.
    schema = _compressor_op._opoverload._schema
    mutated = {
        argument.name
        for argument in schema.arguments
        if argument.alias_info is not None and argument.alias_info.is_write
    }
    assert mutated == {"state_cache"}


def test_full_compressor_write_only_compile_preserves_state():
    from vllm_fl.dispatch.backends.vendor.ascend.impl.compressor import (
        deepseek_v4_compressor,
    )

    module = _compressor(4)
    x = _data((3, 8), 0.04, torch.bfloat16)
    state = torch.full((2, 4, 16), -9.0, device="npu", dtype=torch.float32)
    positions = torch.tensor([0, 3, 7], device="npu")
    slots = torch.tensor([0, 3, 7], device="npu")
    table = torch.tensor([[0, 1]], device="npu", dtype=torch.int32)
    empty = torch.empty(0, device="npu", dtype=torch.int64)
    rope = torch.ones(8, 2, device="npu", dtype=torch.float32)

    def write_only(hidden, cache):
        # Intentionally discard the zero-row result. The state mutation is the
        # only observable effect of the custom op and must survive compilation.
        deepseek_v4_compressor(
            hidden,
            module.wkv.weight,
            module.wgate.weight,
            module.ape,
            cache,
            positions,
            slots,
            table,
            empty,
            empty,
            module.norm.weight,
            rope,
        )
        return cache.clone()

    compiled = torch.compile(write_only, backend="aot_eager", fullgraph=True)
    expected = torch.full(state.shape, -9.0, dtype=torch.float32)
    cpu_ape = module.ape.cpu()
    cpu_positions = positions.cpu().tolist()
    cpu_slots = slots.cpu().tolist()
    for hidden in (x, x * 2):
        cpu_hidden = hidden.cpu().float()
        cpu_kv = F.linear(cpu_hidden, module.wkv.weight.cpu().float())
        cpu_gate = F.linear(cpu_hidden, module.wgate.weight.cpu().float())
        for index, slot in enumerate(cpu_slots):
            expected[slot // 4, slot % 4] = torch.cat(
                [cpu_kv[index], cpu_gate[index] + cpu_ape[cpu_positions[index] % 4]]
            )
        output = compiled(hidden, state)
        torch.npu.synchronize()
        torch.testing.assert_close(output.cpu(), expected)
        torch.testing.assert_close(state.cpu(), expected)


@pytest.mark.parametrize("iterations", [1, 4])
def test_hc_pre_independent_equations(iterations):
    x = _data((2, 3, 8), 0.08, torch.bfloat16)
    fn, scale, base = _hc_parameters()
    actual = _call("hc_pre", x, fn, scale, base, 1e-6, 1e-4, iterations)
    expected = _pre_oracle(x, fn, scale, base, iterations)
    assert actual[1].shape == (2, 3)
    for output, oracle in zip(actual, expected):
        torch.testing.assert_close(output, oracle, rtol=0.01, atol=0.01)


def test_hc_post_nontrivial_comb_orientation():
    residual = _data((2, 3, 8), 0.1, torch.bfloat16)
    x = _data((2, 8), 0.07, torch.bfloat16)
    post = torch.tensor([[0.2, 0.8, 1.4], [1.1, 0.3, 0.6]], device="npu")
    comb = torch.tensor(
        [[[0.1, 0.2, 0.7], [0.6, 0.1, 0.3], [0.2, 0.5, 0.3]]] * 2, device="npu"
    )
    actual = _call("hc_post", x, residual, post, comb)
    expected = _post_oracle(x, residual, post, comb)
    torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.01)
    assert not torch.equal(
        expected, _post_oracle(x, residual, post, comb.transpose(1, 2))
    )


def test_hc_head_independent_equation():
    x = _data((2, 3, 8), 0.1, torch.bfloat16)
    fn, scale, base = (
        _data((3, 24), 0.017),
        torch.tensor([0.7], device="npu"),
        _data((3,), 0.03),
    )
    actual = _call("hc_head", x, fn, scale, base, 1e-6, 1e-4)
    flat = x.float().flatten(1)
    mixes = F.linear(flat, fn) * (flat.square().mean(-1, keepdim=True) + 1e-6).rsqrt()
    pre = (mixes * scale + base).sigmoid() + 1e-4
    expected = sum(pre[:, i, None] * x[:, i].float() for i in range(3)).to(x.dtype)
    torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.01)


class _Attention(torch.nn.Module):
    def forward(self, *, positions, hidden_states, llama_4_scaling=None):
        self.last_positions = positions
        self.last_scaling = llama_4_scaling
        return (hidden_states.float() * 0.7 + positions[:, None].float() * 0.02).to(
            hidden_states.dtype
        )


class _MLP(torch.nn.Module):
    def forward(self, x, input_ids=None):
        self.last_ids = input_ids
        extra = 0 if input_ids is None else input_ids[:, None].float() * 0.01
        return (F.silu(x.float()) * 0.6 + extra).to(x.dtype)


def _decoder():
    from vllm_fl.models.deepseek_v4 import DeepseekV4DecoderLayer

    attention, mlp = _Attention(), _MLP()
    layer = DeepseekV4DecoderLayer(8, 3, 4, 1e-4, 1e-6, attention, mlp).to("npu")
    for prefix in ("hc_attn", "hc_ffn"):
        for suffix, value in zip(("fn", "scale", "base"), _hc_parameters()):
            getattr(layer, prefix + "_" + suffix).copy_(value)
    layer.input_layernorm.weight.copy_(torch.linspace(0.6, 1.3, 8, device="npu"))
    layer.post_attention_layernorm.weight.copy_(
        torch.linspace(1.4, 0.7, 8, device="npu")
    )
    return layer


def _decoder_oracle(layer, positions, x, ids):
    before, post, comb = _pre_oracle(
        x, layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base, 4
    )
    normalized = _rms(before, layer.input_layernorm.weight)
    attention = (normalized.float() * 0.7 + positions[:, None].float() * 0.02).to(
        x.dtype
    )
    residual = _post_oracle(attention, x, post, comb)
    before, post, comb = _pre_oracle(
        residual, layer.hc_ffn_fn, layer.hc_ffn_scale, layer.hc_ffn_base, 4
    )
    normalized = _rms(before, layer.post_attention_layernorm.weight)
    extra = 0 if ids is None else ids[:, None].float() * 0.01
    mlp = (F.silu(normalized.float()) * 0.6 + extra).to(x.dtype)
    return _post_oracle(mlp, residual, post, comb)


@pytest.mark.parametrize("with_ids", [False, True])
def test_actual_decoder_norms_and_argument_forwarding(with_ids):
    layer = _decoder()
    x = _data((2, 3, 8), 0.08, torch.bfloat16)
    positions = torch.tensor([3, 7], device="npu")
    ids = torch.tensor([11, 17], device="npu") if with_ids else None
    scaling = torch.tensor([0.8, 1.2], device="npu")
    actual = layer(positions, x, input_ids=ids, llama_4_scaling=scaling)
    expected = _decoder_oracle(layer, positions, x, ids)
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)
    assert layer.self_attn.last_positions is positions
    assert layer.self_attn.last_scaling is scaling
    assert layer.mlp.last_ids is ids


def test_zero_token_decoder():
    x = torch.empty(0, 8, device="npu", dtype=torch.bfloat16)
    output = _decoder()(
        torch.empty(0, device="npu", dtype=torch.int64), x.reshape(0, 3, 8)
    )
    assert output.shape == (0, 3, 8)
