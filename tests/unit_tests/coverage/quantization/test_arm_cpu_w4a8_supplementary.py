# Copyright 2025 BAAI. All rights reserved.
"""Supplementary coverage for ``vllm_fl.quantization.arm_cpu_w4a8``.

The core test (``tests/unit_tests/quantization/test_arm_cpu_w4a8.py``) is
gated on a real ARM64 host with the FlagGems ARM backend, so on every other
platform the whole module stays at 0%. This file installs the patch for
real and drives the patched functions on plain CPU tensors, stubbing only
genuine external boundaries: ``flag_gems.quantized_linear`` (the ARM
kernels never exist off-target). Real vLLM 0.32 is present in the run
environment, so its actual W4A8 scheme/kernel/config classes are used —
with a small vLLM-0.24-shaped compatibility shim pinned per test where
this vLLM drifted (kernel chooser hardware probing, ``replace_parameter``
strictness, TP rank probing).

Covered branches:
* installer idempotency and the corrupt-marker detection;
* the patched ``_get_scheme_from_parts`` packed-path selection and its
  fall-through to the original selector for other formats;
* the patched scheme ``__init__`` (packed flag + pack factor);
* the patched ``create_weights`` packed path: parameter shapes/dtypes,
  the channel-scale row-parallel layout, and the group-size rejection;
  plus the delegation to stock vLLM for unpacked checkpoints;
* the patched ``process_weights_after_loading``: packed-int32 unpacking
  with an independent code oracle, the repack call and scale dropping, the
  weight_shape rejection, and the delegation arms (non-G128, zero-point);
* the patched ``apply_weights``: bf16 conversion, bias add, dtype cast
  back, and the delegation to stock vLLM without the packed shape.
"""

from __future__ import annotations

import pytest
import torch

import vllm_fl.quantization.arm_cpu_w4a8 as arm_module
from vllm_fl.quantization.arm_cpu_w4a8 import install_arm_cpu_packed_w4a8

pytestmark = [pytest.mark.supplementary]


def _install_fake_flag_gems(monkeypatch):
    """Install the ``flag_gems.quantized_linear`` boundary module.

    The ARM operators only exist on an ARM64 accelerator host, so both
    lazy ``from flag_gems.quantized_linear import ...`` statements in the
    module under test need a stand-in here.
    """
    import sys
    from types import ModuleType

    if "flag_gems.quantized_linear" not in sys.modules:
        flag_gems = ModuleType("flag_gems")
        quantized_linear = ModuleType("flag_gems.quantized_linear")
        quantized_linear.pack_rhs_qsi4c128p = lambda weight, scales: weight
        quantized_linear.w4a8_g128_linear = lambda *args, **kwargs: args[0]
        flag_gems.quantized_linear = quantized_linear
        sys.modules.setdefault("flag_gems", flag_gems)
        sys.modules["flag_gems.quantized_linear"] = quantized_linear
    return sys.modules["flag_gems.quantized_linear"]


@pytest.fixture(autouse=True)
def _guard_fake_flag_gems():
    """Track the fake flag_gems.quantized_linear boundary precisely.

    The real vllm tree must stay loaded between tests (re-importing it
    re-registers custom torch operators and raises), so only the fake
    boundary module this file installs is removed at teardown unless it
    pre-existed.
    """
    import sys

    had_root = "flag_gems" in sys.modules
    had_ql = "flag_gems.quantized_linear" in sys.modules
    prior_root = sys.modules.get("flag_gems")
    prior_ql = sys.modules.get("flag_gems.quantized_linear")
    try:
        yield
    finally:
        if had_ql:
            sys.modules["flag_gems.quantized_linear"] = prior_ql
        else:
            sys.modules.pop("flag_gems.quantized_linear", None)
        if had_root:
            sys.modules["flag_gems"] = prior_root
        else:
            sys.modules.pop("flag_gems", None)


def _patch_real_vllm_w4a8_compat(monkeypatch):
    """Pin the vLLM-0.24-shaped API the ARM adapter was written against.

    Real vLLM 0.32 drifted from that API: ``choose_mp_linear_kernel`` now
    probes hardware (a CPU host picks a WNA16 kernel instead of the
    dynamic-4bit one) and ``replace_parameter`` became an in-place strided
    copy that rejects freshly packed tensors. Both sit outside the logic
    under test, so pin the 0.24 shapes for the duration of the fixture.
    The parameter classes additionally consult the TP process group, so pin
    rank 0 / world 1 as the core suite does.
    """
    import vllm.model_executor.kernels.linear as linear_kernels
    import vllm.model_executor.layers.quantization.utils as quant_utils
    import vllm.model_executor.parameter as vllm_parameter
    from vllm.model_executor.kernels.linear.mixed_precision.dynamic_4bit \
        import Dynamic4bitLinearKernel

    real_config = linear_kernels.MPLinearLayerConfig

    def config_compat(*args, **kwargs):
        kwargs.pop("has_g_idx", None)
        return real_config(*args, **kwargs)

    monkeypatch.setattr(linear_kernels, "MPLinearLayerConfig", config_compat)

    real_init = Dynamic4bitLinearKernel.__init__

    def init_compat(self, c, w_q_param_name, w_s_param_name,
                    w_zp_param_name=None, w_gidx_param_name=None):
        real_init(self, c, w_q_param_name, w_s_param_name, w_zp_param_name)

    monkeypatch.setattr(Dynamic4bitLinearKernel, "__init__", init_compat)
    monkeypatch.setattr(
        linear_kernels,
        "choose_mp_linear_kernel",
        lambda config, compute_capability=None: Dynamic4bitLinearKernel,
    )
    monkeypatch.setattr(
        quant_utils,
        "replace_parameter",
        lambda mod, name, new: setattr(mod, name, new),
    )
    monkeypatch.setattr(vllm_parameter, "get_tensor_model_parallel_rank",
                        lambda: 0)
    monkeypatch.setattr(vllm_parameter,
                        "get_tensor_model_parallel_world_size", lambda: 1)


@pytest.fixture
def arm_w4a8(monkeypatch):
    """Install the ARM W4A8 patch per test; undo every mutation after."""
    _install_fake_flag_gems(monkeypatch)
    _patch_real_vllm_w4a8_compat(monkeypatch)

    from vllm.model_executor.kernels.linear.mixed_precision.dynamic_4bit \
        import Dynamic4bitLinearKernel
    from vllm.model_executor.layers.quantization.compressed_tensors \
        import compressed_tensors as config_module
    from vllm.model_executor.layers.quantization.compressed_tensors.schemes \
        import compressed_tensors_w4a8_int as scheme_module

    scheme_cls = scheme_module.CompressedTensorsW4A8Int
    config_cls = config_module.CompressedTensorsConfig
    kernel_cls = Dynamic4bitLinearKernel
    owners = {
        "scheme_init": (scheme_cls, "__init__"),
        "scheme_create_weights": (scheme_cls, "create_weights"),
        "config_get_scheme": (config_cls, "_get_scheme_from_parts"),
        "kernel_process": (kernel_cls, "process_weights_after_loading"),
        "kernel_apply": (kernel_cls, "apply_weights"),
    }
    saved = {name: getattr(owner, attr) for name, (owner, attr) in owners.items()}

    def install(**replacements):
        for name, replacement in replacements.items():
            owner, attr = owners[name]
            setattr(owner, attr, replacement)
        return install_arm_cpu_packed_w4a8()

    try:
        yield {
            "scheme_cls": scheme_cls,
            "config_cls": config_cls,
            "kernel_cls": kernel_cls,
            "install": install,
        }
    finally:
        for name, (owner, attr) in owners.items():
            setattr(owner, attr, saved[name])
        for owner, attribute in (
            (scheme_cls, "_vllm_fl_arm_packed_w4a8"),
            (scheme_cls, "_vllm_fl_arm_original_init"),
            (scheme_cls, "_vllm_fl_arm_original_create_weights"),
            (config_cls, "_vllm_fl_arm_original_get_scheme"),
            (kernel_cls, "_vllm_fl_arm_w4a8"),
            (kernel_cls, "_vllm_fl_arm_original_process"),
            (kernel_cls, "_vllm_fl_arm_original_apply"),
        ):
            if attribute in vars(owner):
                delattr(owner, attribute)


def _quants():
    from compressed_tensors.quantization import (
        QuantizationArgs,
        QuantizationStrategy,
        QuantizationType,
    )

    weight = QuantizationArgs(
        num_bits=4,
        type=QuantizationType.INT,
        strategy=QuantizationStrategy.GROUP,
        symmetric=True,
        dynamic=False,
        group_size=128,
    )
    activation = QuantizationArgs(
        num_bits=8,
        type=QuantizationType.INT,
        strategy=QuantizationStrategy.TOKEN,
        symmetric=True,
        dynamic=True,
    )
    return weight, activation


def _packed_parts(n=32, k=256, group_size=128):
    """Packed G128 W4A8 layer data with an independent code oracle.

    The packed checkpoint stores 8 nibbles per int32 word (little-endian
    bytes, low nibble first), so ``weight_packed`` is [n, k//8]; the
    expected unpacked codes are derived directly from the words here,
    independent of any unpack implementation the module imports.
    """
    g = torch.Generator().manual_seed(0)
    words = torch.randint(-(2**31), 2**31 - 1, (n, k // 8),
                           dtype=torch.int32, generator=g)
    codes = torch.zeros((n, k), dtype=torch.int8)
    word_bytes = words.view(torch.uint8).view(n, k // 8, 4)
    for byte_slot in range(4):
        byte = word_bytes[:, :, byte_slot].to(torch.int16)
        low = ((byte & 0xF) - 8).to(torch.int8)
        high = ((byte >> 4) - 8).to(torch.int8)
        col_low = torch.arange(0, k, 8) + byte_slot * 2
        codes[:, col_low] = low
        codes[:, col_low + 1] = high
    scales = torch.rand((n, k // group_size), dtype=torch.float32,
                        generator=g)
    return words, codes, scales


class TestInstaller:
    def test_second_install_returns_false(self, arm_w4a8):
        assert arm_w4a8["install"]() is True
        assert install_arm_cpu_packed_w4a8() is False

    def test_mismatched_marker_raises(self, arm_w4a8):
        arm_w4a8["install"]()
        arm_w4a8["scheme_cls"]._vllm_fl_arm_packed_w4a8 = False
        try:
            with pytest.raises(RuntimeError, match="incomplete ARM W4A8"):
                install_arm_cpu_packed_w4a8()
        finally:
            arm_w4a8["scheme_cls"]._vllm_fl_arm_packed_w4a8 = True

    def test_marker_attributes_installed(self, arm_w4a8):
        arm_w4a8["install"]()
        assert arm_w4a8["scheme_cls"]._vllm_fl_arm_packed_w4a8 is True
        assert arm_w4a8["kernel_cls"]._vllm_fl_arm_w4a8 is True
        # The originals the installer captured are the pre-install attrs.
        assert callable(arm_w4a8["scheme_cls"]._vllm_fl_arm_original_init)
        assert callable(arm_w4a8["config_cls"]._vllm_fl_arm_original_get_scheme)
        assert callable(arm_w4a8["kernel_cls"]._vllm_fl_arm_original_process)
        assert callable(arm_w4a8["kernel_cls"]._vllm_fl_arm_original_apply)


class TestSchemeSelection:
    def test_packed_dynamic_token_selects_flagged_scheme(self, arm_w4a8):
        arm_w4a8["install"]()
        config = object.__new__(arm_w4a8["config_cls"])
        config.quant_format = "pack-quantized"
        weight, activation = _quants()

        scheme = config._get_scheme_from_parts(
            weight, activation, format="pack-quantized", layer_name="model.fc"
        )

        assert scheme._vllm_fl_checkpoint_packed is True
        assert scheme._vllm_fl_pack_factor == 8
        assert scheme.group_size == 128

    def test_other_format_falls_back_to_original(self, arm_w4a8):
        recorded = {}

        def fake_original(self, *args, **kwargs):
            recorded["args"] = (args, kwargs)
            return "sentinel"

        assert arm_w4a8["install"](config_get_scheme=fake_original) is True
        config = object.__new__(arm_w4a8["config_cls"])
        config.quant_format = "dense"
        weight, activation = _quants()

        result = config._get_scheme_from_parts(
            weight, activation, format="dense", layer_name="model.fc"
        )

        assert result == "sentinel"
        args, kwargs = recorded["args"]
        assert args == (weight, activation)
        assert kwargs == {
            "output_quant": None,
            "format": "dense",
            "layer_name": "model.fc",
        }

    def test_stock_scheme_marks_unpackaged(self, arm_w4a8):
        arm_w4a8["install"]()
        scheme = arm_w4a8["scheme_cls"](
            strategy="group",
            num_bits=4,
            group_size=128,
            is_static_input_scheme=False,
            input_symmetric=True,
        )
        assert scheme._vllm_fl_checkpoint_packed is False
        assert scheme._vllm_fl_pack_factor == 8


class TestCreateWeights:
    @staticmethod
    def _create(scheme, layer, *, input_size, input_size_per_partition):
        scheme.create_weights(
            layer,
            output_size=32,
            input_size=input_size,
            output_partition_sizes=[32],
            input_size_per_partition=input_size_per_partition,
            params_dtype=torch.bfloat16,
            weight_loader=lambda *args, **kwargs: None,
        )

    def test_packed_parameters_shapes_and_dtypes(self, arm_w4a8):
        arm_w4a8["install"]()
        scheme = arm_w4a8["scheme_cls"](
            strategy="group",
            num_bits=4,
            group_size=128,
            is_static_input_scheme=False,
            input_symmetric=True,
            packed=True,
        )
        layer = torch.nn.Module()
        self._create(scheme, layer, input_size=256, input_size_per_partition=256)

        assert layer.weight_packed.shape == (32, 256 // 8)
        assert layer.weight_packed.dtype == torch.int32
        assert layer.weight_scale.shape == (32, 256 // 128)
        assert layer.weight_shape.shape == (2,)
        assert layer.weight_shape.dtype == torch.int64
        assert scheme.kernel._vllm_fl_arm_packed_checkpoint is True

    def test_channel_scale_for_row_parallel_full_groups(self, arm_w4a8):
        arm_w4a8["install"]()
        scheme = arm_w4a8["scheme_cls"](
            strategy="channel",
            num_bits=4,
            group_size=None,
            is_static_input_scheme=False,
            input_symmetric=True,
            packed=True,
        )
        assert scheme.group_size == -1
        layer = torch.nn.Module()
        # Row parallel: effective group = per-partition input (128).
        self._create(scheme, layer, input_size=512, input_size_per_partition=128)
        assert layer.weight_packed.shape == (32, 128 // 8)
        assert layer.weight_scale.shape == (32, 1)  # one per channel
        assert getattr(layer.weight_scale, "input_dim", None) is None

    def test_group_partition_mismatch_rejected(self, arm_w4a8):
        arm_w4a8["install"]()
        scheme = arm_w4a8["scheme_cls"](
            strategy="group",
            num_bits=4,
            group_size=128,
            is_static_input_scheme=False,
            input_symmetric=True,
            packed=True,
        )
        with pytest.raises(ValueError, match="not divisible"):
            self._create(
                scheme, torch.nn.Module(),
                input_size=130, input_size_per_partition=130,
            )

    def test_unpacked_scheme_uses_original_create(self, arm_w4a8):
        recorded = {}

        def fake_original(self, *args, **kwargs):
            recorded["args"] = args
            recorded["kwargs"] = kwargs

        assert arm_w4a8["install"](scheme_create_weights=fake_original) is True
        scheme = arm_w4a8["scheme_cls"](
            strategy="group",
            num_bits=4,
            group_size=128,
            is_static_input_scheme=False,
            input_symmetric=True,
        )
        layer = torch.nn.Module()
        layer.bias = None
        scheme.create_weights(
            layer,
            output_size=8,
            input_size=64,
            output_partition_sizes=[8],
            input_size_per_partition=64,
            params_dtype=torch.bfloat16,
            weight_loader=lambda *args, **kwargs: None,
        )

        assert recorded["args"][0] is layer
        assert not hasattr(layer, "weight_packed")


class TestProcessWeightsAfterLoading:
    @staticmethod
    def _loaded(arm_w4a8, *, weight_shape=None):
        arm_w4a8["install"]()
        scheme = arm_w4a8["scheme_cls"](
            strategy="group",
            num_bits=4,
            group_size=128,
            is_static_input_scheme=False,
            input_symmetric=True,
            packed=True,
        )
        n, k = 32, 256
        layer = torch.nn.Module()
        scheme.create_weights(
            layer,
            output_size=n,
            input_size=k,
            output_partition_sizes=[n],
            input_size_per_partition=k,
            params_dtype=torch.bfloat16,
            weight_loader=lambda *args, **kwargs: None,
        )
        words, codes, scales = _packed_parts(n, k)
        layer.weight_packed.data.copy_(words)
        layer.weight_scale.data.copy_(scales.to(torch.bfloat16))
        layer.weight_shape.data.copy_(torch.tensor(weight_shape or (n, k)))
        return scheme, layer, words, codes, scales

    def test_packed_path_repacks_and_nulls_scale(self, arm_w4a8, monkeypatch):
        packed_calls = []

        def fake_pack(weight, scales):
            packed_calls.append((weight.clone(), scales.clone()))
            return torch.zeros(weight.shape[0], weight.shape[1] // 2,
                               dtype=torch.uint8)

        ql = _install_fake_flag_gems(monkeypatch)
        monkeypatch.setattr(ql, "pack_rhs_qsi4c128p", fake_pack)
        scheme, layer, words, codes, scales = self._loaded(arm_w4a8)
        kernel = scheme.kernel

        kernel.process_weights_after_loading(layer)

        unpacked, packed_scales = packed_calls[0]
        assert unpacked.shape == (32, 256)
        assert unpacked.dtype == torch.int8
        # Independent code oracle: the unpacked matrix equals the codes
        # derived from the packed words in the test (not via the module).
        assert torch.equal(unpacked, codes)
        assert torch.equal(packed_scales, scales.to(torch.bfloat16))
        assert getattr(layer, kernel.w_s_name) is None
        assert layer.weight_shape is None
        assert kernel._vllm_fl_arm_w4a8_shape == (32, 256)

    def test_bad_weight_shape_rejected(self, arm_w4a8, monkeypatch):
        ql = _install_fake_flag_gems(monkeypatch)
        monkeypatch.setattr(
            ql,
            "pack_rhs_qsi4c128p",
            lambda *args: pytest.fail("packer must not run for invalid shapes"),
        )
        scheme, layer, *_ = self._loaded(arm_w4a8, weight_shape=(32, 999))
        with pytest.raises(RuntimeError, match="weight_shape"):
            scheme.kernel.process_weights_after_loading(layer)

    def test_non_g128_group_delegates_to_original(self, arm_w4a8):
        recorded = []

        def fake_original(self, layer):
            recorded.append(layer)

        arm_w4a8["install"](kernel_process=fake_original)
        scheme, layer, *_ = self._loaded(arm_w4a8)
        # Force the kernel's config group_size to something other than 128
        # so the patched process delegates to the original (returns None).
        kernel = scheme.kernel
        kernel.config.group_size = 64
        assert kernel.process_weights_after_loading(layer) is None
        assert recorded[0] is layer
        # Nothing was repacked or nullified on the delegation path.
        assert kernel.w_s_name in layer._parameters or getattr(
            layer, kernel.w_s_name, "unset"
        ) != "unset"

    def test_zero_points_delegates_to_original(self, arm_w4a8):
        recorded = []

        def fake_original(self, layer):
            recorded.append(layer)

        arm_w4a8["install"](kernel_process=fake_original)
        scheme, layer, *_ = self._loaded(arm_w4a8)
        kernel = scheme.kernel
        kernel.config.zero_points = True
        assert kernel.process_weights_after_loading(layer) is None
        assert recorded[0] is layer

    def test_unpacked_checkpoint_skips_shape_unwrap(self, arm_w4a8,
                                                    monkeypatch):
        """An int8 [n, k] checkpoint (not packed int32) skips the unpack."""
        ql = _install_fake_flag_gems(monkeypatch)
        packed_calls = []

        def fake_pack(weight, scales):
            packed_calls.append(weight.shape)
            return torch.zeros(weight.shape[0], weight.shape[1] // 2,
                               dtype=torch.uint8)

        monkeypatch.setattr(ql, "pack_rhs_qsi4c128p", fake_pack)
        arm_w4a8["install"]()
        scheme = arm_w4a8["scheme_cls"](
            strategy="group",
            num_bits=4,
            group_size=128,
            is_static_input_scheme=False,
            input_symmetric=True,
            packed=True,
        )
        n, k = 16, 256
        layer = torch.nn.Module()
        scheme.create_weights(
            layer,
            output_size=n,
            input_size=k,
            output_partition_sizes=[n],
            input_size_per_partition=k,
            params_dtype=torch.bfloat16,
            weight_loader=lambda *args, **kwargs: None,
        )
        # Load the int8 unpacked layout with matching scales.
        layer.weight_packed.data = torch.randint(
            -8, 7, (n, k), dtype=torch.int8
        )
        layer.weight_scale.data = torch.rand(n, k // 128).to(torch.bfloat16)
        scheme.kernel.process_weights_after_loading(layer)
        # The [n, k] int8 matrix went straight to the packer.
        assert packed_calls == [(16, 256)]


class TestApplyWeights:
    @staticmethod
    def _loaded(arm_w4a8):
        arm_w4a8["install"]()
        scheme = arm_w4a8["scheme_cls"](
            strategy="group",
            num_bits=4,
            group_size=128,
            is_static_input_scheme=False,
            input_symmetric=True,
            packed=True,
        )
        n, k = 32, 256
        layer = torch.nn.Module()
        scheme.create_weights(
            layer,
            output_size=n,
            input_size=k,
            output_partition_sizes=[n],
            input_size_per_partition=k,
            params_dtype=torch.bfloat16,
            weight_loader=lambda *args, **kwargs: None,
        )
        words, codes, scales = _packed_parts(n, k)
        layer.weight_packed.data.copy_(words)
        layer.weight_scale.data.copy_(scales.to(torch.bfloat16))
        layer.weight_shape.data.copy_(torch.tensor((n, k)))
        return scheme, layer

    def test_apply_converts_dtype_and_adds_bias(self, arm_w4a8, monkeypatch):
        apply_calls = []

        def fake_linear(x, weight, n, k):
            apply_calls.append((x.clone(), weight, n, k))
            return torch.full((x.shape[0], n), 1.5, dtype=torch.bfloat16)

        ql = _install_fake_flag_gems(monkeypatch)
        monkeypatch.setattr(ql, "w4a8_g128_linear", fake_linear)
        scheme, layer = self._loaded(arm_w4a8)
        kernel = scheme.kernel
        kernel.process_weights_after_loading(layer)

        x = torch.ones((2, 256), dtype=torch.float32)
        bias = torch.full((32,), 0.25, dtype=torch.float32)
        out = kernel.apply_weights(layer, x, bias)

        assert out.dtype == torch.float32
        assert out.shape == (2, 32)
        # 1.5 + 0.25 computed then cast back to the source dtype.
        assert torch.allclose(out, torch.full((2, 32), 1.75, dtype=torch.float32))
        called_x, called_weight, n, k = apply_calls[0]
        assert called_x.dtype == torch.bfloat16
        assert called_weight is layer.weight_packed
        assert (n, k) == (32, 256)

    def test_bf16_input_passes_through_unconverted(self, arm_w4a8, monkeypatch):
        apply_calls = []

        def fake_linear(x, weight, n, k):
            apply_calls.append(x)
            return torch.zeros((x.shape[0], n), dtype=torch.bfloat16)

        ql = _install_fake_flag_gems(monkeypatch)
        monkeypatch.setattr(ql, "w4a8_g128_linear", fake_linear)
        scheme, layer = self._loaded(arm_w4a8)
        kernel = scheme.kernel
        kernel.process_weights_after_loading(layer)

        x = torch.ones((1, 256), dtype=torch.bfloat16)
        out = kernel.apply_weights(layer, x, None)
        assert out.dtype == torch.bfloat16
        assert apply_calls[0] is x  # same object: no conversion was needed

    def test_apply_without_shape_uses_original(self, arm_w4a8):
        recorded = []

        def fake_original(self, layer, x, bias=None):
            recorded.append((layer, x, bias))
            return "stock"

        arm_w4a8["install"](kernel_apply=fake_original)
        scheme, layer = self._loaded(arm_w4a8)
        kernel = scheme.kernel
        # Do NOT process weights: no _vllm_fl_arm_w4a8_shape is present.
        x = torch.ones((1, 256), dtype=torch.bfloat16)
        assert kernel.apply_weights(layer, x, None) == "stock"
        assert recorded[0][0] is layer
