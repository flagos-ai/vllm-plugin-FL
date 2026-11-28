# Copyright (c) 2026 BAAI. All rights reserved.

"""Edge tests for the W8A8/W4A8 quantization family.

Two groups of gaps left by the core suite (tests/unit_tests/quantization/):

* ``arm_cpu_w4a8.py`` (0/120 executed): ``install_arm_cpu_packed_w4a8``
  monkey-patches vLLM's W4A8 scheme, its config scheme selector and the
  ``Dynamic4bitLinearKernel`` load/apply hooks. The core test
  (test_arm_cpu_w4a8.py) is skipped unless the host is a real ARM64 machine
  with the FlagGems ARM backend, so on CPU dev hosts none of the patched
  code runs. The tests below install the patch for real and drive the
  patched functions on plain CPU tensors, stubbing only genuine external
  boundaries: the ``flag_gems.quantized_linear`` operators (the ARM
  kernels never exist off-target, and the flag_gems package cannot even
  import on a CPU-only host), and, on hosts without vLLM, the
  vLLM/compressed_tensors import graph the installer reaches into. On
  hosts with real vLLM 0.32 the installer additionally needs a small
  vLLM-0.24-shaped compatibility shim (the module's target API drifted:
  ``MPLinearLayerConfig`` lost ``has_g_idx``, the kernel constructor lost
  ``w_gidx_param_name``, the kernel chooser now probes hardware, and
  ``replace_parameter`` became an in-place strided copy), applied by the
  ``arm_w4a8`` fixture before the install and reversed after it.
* The small W8A8 gaps: the reference module's error branches
  (unpack/dynamic-quant/linear validation) and the dynamic-linear
  kernel's weight-layout rejection paths, all pure-torch and previously
  only covered through their happy paths.

State restoration: the ARM installer's own class-attribute monkey-patches
are reversed by the ``arm_w4a8`` fixture teardown (the exact inverse of
the core test's tearDownClass), and every ``sys.modules`` change goes
through the directory conftest's ``fl_sys_modules`` snapshot.
"""

import enum
import sys
import types
from typing import Any

import pytest
import torch


def _make_module(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__path__ = []
    return module


def _install_fake_flag_gems() -> None:
    """Install a fake ``flag_gems.quantized_linear`` boundary module.

    The real flag_gems package aborts import on a CPU-only host ("No device
    were detected") and its ARM operators only exist on an ARM64 accelerator
    host anyway, so the module-under-test's two lazy
    ``from flag_gems.quantized_linear import ...`` statements always need a
    stand-in here. Installed at file scope so the directory conftest's
    first snapshot captures it and re-instates it around every test.
    """
    if "flag_gems.quantized_linear" in sys.modules:
        return
    flag_gems = _make_module("flag_gems")
    quantized_linear = _make_module("flag_gems.quantized_linear")
    quantized_linear.pack_rhs_qsi4c128p = lambda weight, scales: weight
    quantized_linear.w4a8_g128_linear = lambda *args, **kwargs: args[0]
    flag_gems.quantized_linear = quantized_linear
    sys.modules.setdefault("flag_gems", flag_gems)
    sys.modules["flag_gems.quantized_linear"] = quantized_linear


_install_fake_flag_gems()


def _vllm_available() -> bool:
    """Report whether a real vllm distribution is present on this host."""
    import importlib.util

    try:
        return importlib.util.find_spec("vllm") is not None
    except (ImportError, ValueError):  # pragma: no cover
        return False


class _ArmW4A8Stubs:
    """Install the external graph ``install_arm_cpu_packed_w4a8`` imports.

    Only used on hosts without real vLLM (CPU-only Windows): the module
    under test imports vLLM lazily inside the installer, so the stubs mirror
    just the constructor contracts the installer and the patched functions
    touch — the vLLM-0.24 shapes the module was written against. With real
    vLLM the class is never instantiated and everything below runs against
    the real classes.
    """

    def __init__(self) -> None:
        self._installed: list[str] = []
        self._previous: dict[str, Any] = {}

        # compressed_tensors graph.
        ct_root = _make_module("compressed_tensors")
        ct_config = _make_module("compressed_tensors.config")
        ct_config.CompressionFormat = enum.Enum(
            "CompressionFormat", {"pack_quantized": "pack-quantized"}
        )
        ct_quant = _make_module("compressed_tensors.quantization")

        class _Args:
            def __init__(self, **kwargs) -> None:
                for key, value in kwargs.items():
                    setattr(self, key, value)

        ct_quant.QuantizationArgs = _Args

        helpers = _make_module("compressed_tensors.compressors.pack_quantized.helpers")

        def unpack_from_int32(value, num_bits, shape, packed_dim=1):
            """Reference inverse of compressed-tensors int32 packing."""
            assert value.dtype is torch.int32 and num_bits == 4
            if packed_dim == 0:
                value = value.transpose(0, 1)
            total = 1
            for dim in shape:
                total *= dim
            per_word = 32 // num_bits
            codes = []
            for word in value.reshape(-1).tolist():
                word &= 0xFFFFFFFF
                for slot in range(per_word):
                    codes.append(((word >> (slot * num_bits)) & 0xF) - 8)
            return torch.tensor(codes[:total], dtype=torch.int8).reshape(shape)

        helpers.unpack_from_int32 = unpack_from_int32

        # vllm graph.
        vllm = _make_module("vllm")
        me = _make_module("vllm.model_executor")
        layers = _make_module("vllm.model_executor.layers")
        quant = _make_module("vllm.model_executor.layers.quantization")
        ct_pkg = _make_module(
            "vllm.model_executor.layers.quantization.compressed_tensors"
        )
        ct_module = _make_module(
            "vllm.model_executor.layers.quantization.compressed_tensors."
            "compressed_tensors"
        )
        schemes = _make_module(
            "vllm.model_executor.layers.quantization.compressed_tensors.schemes"
        )
        scheme_module = _make_module(
            "vllm.model_executor.layers.quantization.compressed_tensors."
            "schemes.compressed_tensors_w4a8_int"
        )
        utils = _make_module("vllm.model_executor.layers.quantization.utils")
        parameter = _make_module("vllm.model_executor.parameter")
        kernels = _make_module("vllm.model_executor.kernels")
        linear_kernels = _make_module("vllm.model_executor.kernels.linear")
        mixed = _make_module("vllm.model_executor.kernels.linear.mixed_precision")
        dynamic_4bit = _make_module(
            "vllm.model_executor.kernels.linear.mixed_precision.dynamic_4bit"
        )

        class _Parameter(torch.nn.Parameter):
            def __new__(cls, data, **kwargs):
                return torch.nn.Parameter._make_subclass(cls, data, False)

            def __init__(self, data, **kwargs) -> None:
                self.weight_loader = kwargs.get("weight_loader")
                self.input_dim = kwargs.get("input_dim")
                self.output_dim = kwargs.get("output_dim")
                self.packed_factor = kwargs.get("packed_factor")
                self.packed_dim = kwargs.get("packed_dim")

        parameter.BasevLLMParameter = _Parameter
        parameter.ChannelQuantScaleParameter = _Parameter
        parameter.GroupQuantScaleParameter = _Parameter
        parameter.PackedvLLMParameter = _Parameter

        class _Config:
            def __init__(self, **kwargs) -> None:
                self.__dict__.update(kwargs)

        class _Kernel:
            def __init__(
                self,
                c,
                w_q_param_name,
                w_s_param_name,
                w_zp_param_name=None,
                w_gidx_param_name=None,
            ) -> None:
                self.config = c
                self.w_q_name = w_q_param_name
                self.w_s_name = w_s_param_name
                self.w_zp_name = w_zp_param_name

        class _Dynamic4bit(_Kernel):
            def process_weights_after_loading(self, layer):
                raise AssertionError("replaced by the installer")

            def apply_weights(self, layer, x, bias=None):
                raise AssertionError("replaced by the installer")

        linear_kernels.MPLinearLayerConfig = _Config
        linear_kernels.choose_mp_linear_kernel = lambda config: _Dynamic4bit
        linear_kernels.Int8ScaledMMLinearKernel = type(
            "Int8ScaledMMLinearKernel", (), {}
        )
        linear_kernels.Int8ScaledMMLinearLayerConfig = type(
            "Int8ScaledMMLinearLayerConfig", (), {}
        )
        dynamic_4bit.Dynamic4bitLinearKernel = _Dynamic4bit

        class _ConfigCls:
            quant_format = "pack-quantized"

            @staticmethod
            def _is_dynamic_token_w4a8_int(weight_quant, input_quant):
                return (
                    weight_quant is not None
                    and weight_quant.num_bits == 4
                    and weight_quant.type == "int"
                    and weight_quant.strategy in ("group", "channel")
                    and weight_quant.group_size == 128
                    and input_quant is not None
                    and input_quant.num_bits == 8
                    and input_quant.type == "int"
                    and input_quant.strategy == "token"
                    and input_quant.symmetric
                    and input_quant.dynamic
                )

            def _get_scheme_from_parts(self, *args, **kwargs):
                raise AssertionError("replaced by the installer")

        ct_module.CompressedTensorsConfig = _ConfigCls

        class _W4A8Int:
            def __init__(
                self,
                strategy,
                num_bits,
                group_size=None,
                is_static_input_scheme=False,
                input_symmetric=True,
            ) -> None:
                self.strategy = strategy
                self.group_size = -1 if group_size is None else group_size
                self.is_static_input_scheme = is_static_input_scheme
                self.input_symmetric = input_symmetric
                self.quant_type = "uint4"

            def create_weights(self, *args, **kwargs):
                raise AssertionError("replaced by the installer")

        scheme_module.CompressedTensorsW4A8Int = _W4A8Int
        utils.replace_parameter = lambda layer, name, param: setattr(layer, name, param)

        entries = {
            "compressed_tensors": ct_root,
            "compressed_tensors.config": ct_config,
            "compressed_tensors.quantization": ct_quant,
            "compressed_tensors.compressors": _make_module(
                "compressed_tensors.compressors"
            ),
            "compressed_tensors.compressors.pack_quantized": _make_module(
                "compressed_tensors.compressors.pack_quantized"
            ),
            "compressed_tensors.compressors.pack_quantized.helpers": helpers,
            "vllm": vllm,
            "vllm.model_executor": me,
            "vllm.model_executor.layers": layers,
            "vllm.model_executor.layers.quantization": quant,
            "vllm.model_executor.layers.quantization.compressed_tensors": (ct_pkg),
            "vllm.model_executor.layers.quantization.compressed_tensors."
            "compressed_tensors": ct_module,
            "vllm.model_executor.layers.quantization.compressed_tensors."
            "schemes": schemes,
            "vllm.model_executor.layers.quantization.compressed_tensors."
            "schemes.compressed_tensors_w4a8_int": scheme_module,
            "vllm.model_executor.layers.quantization.utils": utils,
            "vllm.model_executor.parameter": parameter,
            "vllm.model_executor.kernels": kernels,
            "vllm.model_executor.kernels.linear": linear_kernels,
            "vllm.model_executor.kernels.linear.mixed_precision": mixed,
            "vllm.model_executor.kernels.linear.mixed_precision.dynamic_4bit": (
                dynamic_4bit
            ),
        }
        for name, module in entries.items():
            self._previous[name] = sys.modules.get(name)
            sys.modules[name] = module
            self._installed.append(name)

    def restore(self) -> None:
        """Undo every sys.modules replacement made above."""
        for name in self._installed:
            previous = self._previous[name]
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
        self._installed.clear()
        self._previous.clear()


_REAL_VLLM = _vllm_available()

if not _REAL_VLLM:  # pragma: no cover - exercised only without vllm
    _stubs = _ArmW4A8Stubs()

import vllm_fl.quantization.arm_cpu_w4a8 as arm_module  # noqa: E402
from vllm_fl.quantization.w8a8.linear import (  # noqa: E402
    FLW8A8DynamicLinearKernel,
)

if not _REAL_VLLM:  # pragma: no cover - order-independence for later files
    # arm_cpu_w4a8 and w8a8.linear hold every vllm symbol they need as
    # attributes of their own namespaces, so the imports above stay
    # functional once the stub tree is removed. Leaving the stubs installed
    # would keep a fake "vllm" in sys.modules: any later file's
    # ``import vllm`` probe then succeeds and HAS_VLLM-gated files import
    # against stale partial stubs instead of skipping on vllm-less hosts.
    _stubs.restore()

# ---------------------------------------------------------------------------
# Reference-module edges (pure torch; runs everywhere)
# ---------------------------------------------------------------------------


class TestReferenceUnpackEdges:
    """unpack_uint8b128_int32 validation branches."""

    def test_rejects_wrong_rank_or_dtype(self):
        """A 1D input or a non-int32 dtype is rejected."""
        from vllm_fl.quantization.w8a8.reference import unpack_uint8b128_int32

        with pytest.raises(ValueError, match="2D int32"):
            unpack_uint8b128_int32(torch.zeros(4, dtype=torch.int32))
        with pytest.raises(ValueError, match="2D int32"):
            unpack_uint8b128_int32(torch.zeros(2, 2, dtype=torch.int8))

    def test_rejects_oversized_in_features(self):
        """in_features beyond the packed width is rejected."""
        from vllm_fl.quantization.w8a8.reference import unpack_uint8b128_int32

        packed = torch.zeros((2, 1), dtype=torch.int32)
        with pytest.raises(ValueError, match="incompatible"):
            unpack_uint8b128_int32(packed, in_features=8)
        # Negative in_features is equally invalid.
        with pytest.raises(ValueError, match="incompatible"):
            unpack_uint8b128_int32(packed, in_features=-1)

    def test_in_features_truncates_and_offset_restores_int8(self):
        """Truncation keeps the first in_features codes; codes are byte-128."""
        from vllm_fl.quantization.w8a8.reference import unpack_uint8b128_int32

        # One int32 word per row holds 4 offset-binary codes; write codes
        # 130, 129, 128, 127 (little-endian byte order) into word 0.
        word = 130 | (129 << 8) | (128 << 16) | (127 << 24)
        packed = torch.tensor([[word]], dtype=torch.int32)
        full = unpack_uint8b128_int32(packed)
        assert full.tolist() == [[2, 1, 0, -1]]
        truncated = unpack_uint8b128_int32(packed, in_features=2)
        assert truncated.tolist() == [[2, 1]]


class TestReferenceDynamicQuantEdges:
    """dynamic_per_token_quant_int8 validation and dtype handling."""

    def test_rejects_non_2d_or_integer_input(self):
        """Non-matrix and integer inputs are rejected with distinct errors."""
        from vllm_fl.quantization.w8a8.reference import (
            dynamic_per_token_quant_int8,
        )

        with pytest.raises(ValueError, match="2D"):
            dynamic_per_token_quant_int8(torch.ones(4))
        with pytest.raises(TypeError, match="floating point"):
            dynamic_per_token_quant_int8(torch.ones(2, 2, dtype=torch.int32))


class TestReferenceLinearEdges:
    """w8a8_linear_reference validation branches."""

    def test_rejects_bad_weight_and_input_shapes(self):
        """Weight must be 2D int8; x's last dim must match weight K."""
        from vllm_fl.quantization.w8a8.reference import w8a8_linear_reference

        x = torch.ones((1, 4))
        with pytest.raises(ValueError, match="weight must be"):
            w8a8_linear_reference(x, torch.ones(4, dtype=torch.float32), torch.ones(4))
        with pytest.raises(ValueError, match="incompatible input dimensions"):
            w8a8_linear_reference(
                x, torch.ones((2, 3), dtype=torch.int8), torch.ones(2)
            )

    def test_rejects_wrong_bias_width(self):
        """A bias not matching the output channels is rejected."""
        from vllm_fl.quantization.w8a8.reference import w8a8_linear_reference

        with pytest.raises(ValueError, match="one value per output channel"):
            w8a8_linear_reference(
                torch.ones((1, 4)),
                torch.ones((2, 4), dtype=torch.int8),
                torch.ones(2),
                bias=torch.ones(3),
            )

    def test_bias_added_in_float_before_cast(self):
        """Bias is added to the scaled accumulator, not to the int8 codes."""
        from vllm_fl.quantization.w8a8.reference import w8a8_linear_reference

        # Row [127, -127] quantizes exactly to itself (scale 1.0); a weight
        # column of ones and scale 1.0 make the accumulator [0]; the bias
        # then decides the output.
        x = torch.tensor([[127.0, -127.0]])
        weight = torch.ones((1, 2), dtype=torch.int8)
        scale = torch.ones(1)
        bias = torch.tensor([0.5])
        out = w8a8_linear_reference(x, weight, scale, bias)
        assert torch.equal(out, torch.tensor([[0.5]]))
        assert out.dtype == torch.float32


# ---------------------------------------------------------------------------
# Dynamic-linear kernel weight-layout edges
# ---------------------------------------------------------------------------


class TestLinearKernelLayoutValidation:
    """FLW8A8DynamicLinearKernel.process_weights_after_loading guards."""

    @staticmethod
    def _kernel():
        kernel = object.__new__(FLW8A8DynamicLinearKernel)
        kernel.layer_param_names = [
            "weight",
            "weight_scale",
            "input_scale",
            "input_zero_point",
            "azp_adj",
        ]
        return kernel

    @staticmethod
    def _layer(weight, scale):
        layer = torch.nn.Module()
        layer.register_parameter(
            "weight", torch.nn.Parameter(weight, requires_grad=False)
        )
        layer.register_parameter(
            "weight_scale", torch.nn.Parameter(scale, requires_grad=False)
        )
        return layer

    def test_rejects_non_2d_or_non_int8_weight(self):
        """A 1D or float weight fails the checkpoint contract check."""
        kernel = self._kernel()
        layer = self._layer(torch.zeros(4), torch.ones(4))
        with pytest.raises(ValueError, match="2D int8"):
            kernel.process_weights_after_loading(layer)

        layer = self._layer(torch.zeros(2, 2), torch.ones(2))
        with pytest.raises(ValueError, match="2D int8"):
            kernel.process_weights_after_loading(layer)

    def test_rejects_scale_count_mismatch(self):
        """One scale per output channel is enforced before the transpose."""
        kernel = self._kernel()
        layer = self._layer(torch.zeros((3, 4), dtype=torch.int8), torch.ones(2))
        with pytest.raises(ValueError, match="one scale per output channel"):
            kernel.process_weights_after_loading(layer)


# ---------------------------------------------------------------------------
# ARM CPU packed W4A8 (the 0/120 module)
# ---------------------------------------------------------------------------


def _patch_real_vllm_w4a8_compat(monkeypatch):
    """Pin the vLLM-0.24-shaped API the ARM adapter was written against.

    Real vLLM 0.32 drifted from that API: ``MPLinearLayerConfig`` lost
    ``has_g_idx``, the kernel constructor lost ``w_gidx_param_name``,
    ``choose_mp_linear_kernel`` now probes hardware (a CPU host picks a
    WNA16 kernel instead of the dynamic-4bit one), and ``replace_parameter``
    became an in-place strided copy that rejects freshly packed tensors.
    All of these sit outside the logic under test, so pin the 0.24 shapes
    for the duration of the fixture. The parameter classes additionally
    consult the TP process group, so pin rank 0 / world 1 as the core
    suite does.
    """
    import vllm.model_executor.kernels.linear as linear_kernels
    import vllm.model_executor.layers.quantization.utils as quant_utils
    import vllm.model_executor.parameter as vllm_parameter
    from vllm.model_executor.kernels.linear.mixed_precision.dynamic_4bit import (
        Dynamic4bitLinearKernel,
    )

    real_config = linear_kernels.MPLinearLayerConfig

    def config_compat(*args, **kwargs):
        kwargs.pop("has_g_idx", None)
        return real_config(*args, **kwargs)

    monkeypatch.setattr(linear_kernels, "MPLinearLayerConfig", config_compat)
    monkeypatch.setattr(
        linear_kernels,
        "choose_mp_linear_kernel",
        lambda config, compute_capability=None: Dynamic4bitLinearKernel,
    )

    real_init = Dynamic4bitLinearKernel.__init__

    def init_compat(
        self,
        c,
        w_q_param_name,
        w_s_param_name,
        w_zp_param_name=None,
        w_gidx_param_name=None,
    ):
        real_init(self, c, w_q_param_name, w_s_param_name, w_zp_param_name)

    monkeypatch.setattr(Dynamic4bitLinearKernel, "__init__", init_compat)
    monkeypatch.setattr(
        quant_utils,
        "replace_parameter",
        lambda mod, name, new: setattr(mod, name, new),
    )
    monkeypatch.setattr(vllm_parameter, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        vllm_parameter, "get_tensor_model_parallel_world_size", lambda: 1
    )


@pytest.fixture
def arm_w4a8(fl_sys_modules, monkeypatch):
    """Yield a controller that installs the ARM W4A8 patch per test.

    ``install_arm_cpu_packed_w4a8`` mutates class attributes on the real
    (or stubbed) vLLM scheme/kernel/config classes. The controller's
    ``install()`` runs the installer once; ``install(**replacements)``
    first points one of the five patched attributes at a caller-supplied
    function so the installer records it as the "original" — the cleanest
    way to observe the delegation branches without touching the module
    under test. Teardown restores the pre-install attributes and deletes
    the installer's markers, the exact inverse of the core test's
    tearDownClass, so later tests see the pristine classes.
    """
    if _REAL_VLLM:
        _patch_real_vllm_w4a8_compat(monkeypatch)
    else:
        # install_arm_cpu_packed_w4a8 imports its vLLM graph lazily, and this
        # fixture's imports below read the same modules, so the stub tree the
        # file-scope block removed after import is re-installed per test and
        # removed again in teardown — nothing leaks to later files.
        stubs = _ArmW4A8Stubs()
    from vllm.model_executor.kernels.linear.mixed_precision.dynamic_4bit import (  # type: ignore[import-not-found]
        Dynamic4bitLinearKernel,
    )
    from vllm.model_executor.layers.quantization.compressed_tensors import (  # type: ignore[import-not-found]
        compressed_tensors as config_module,
    )
    from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (  # type: ignore[import-not-found]
        compressed_tensors_w4a8_int as scheme_module,
    )

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
        """Run the installer, optionally pre-replacing class attributes."""
        for name, replacement in replacements.items():
            owner, attr = owners[name]
            setattr(owner, attr, replacement)
        return arm_module.install_arm_cpu_packed_w4a8()

    try:
        yield {
            "scheme_cls": scheme_cls,
            "config_cls": config_cls,
            "kernel_cls": kernel_cls,
            "install": install,
        }
    finally:
        if not _REAL_VLLM:
            stubs.restore()
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


def _packed_parts(n=32, k=256, group_size=128):
    """Build packed G128 W4A8 layer data with an independent code oracle.

    ``n`` output channels, ``k`` input features. The packed checkpoint
    stores 8 nibbles per int32 word (little-endian bytes, low nibble
    first), so ``weight_packed`` is [n, k//8]; the expected unpacked codes
    are derived here directly from the words, independent of any unpack
    implementation the module imports.
    """
    torch.manual_seed(0)
    words = torch.randint(-(2**31), 2**31 - 1, (n, k // 8), dtype=torch.int32)
    codes = torch.zeros((n, k), dtype=torch.int8)
    word_bytes = words.view(torch.uint8).view(n, k // 8, 4)
    for byte_slot in range(4):
        byte = word_bytes[:, :, byte_slot].to(torch.int16)
        low = ((byte & 0xF) - 8).to(torch.int8)
        high = ((byte >> 4) - 8).to(torch.int8)
        col_low = torch.arange(0, k, 8) + byte_slot * 2
        codes[:, col_low] = low
        codes[:, col_low + 1] = high
    scales = torch.rand((n, k // group_size), dtype=torch.float32)
    return words, codes, scales


class TestArmCpuPackedW4A8Install:
    """Installer idempotency and marker bookkeeping."""

    def test_second_install_returns_false(self, arm_w4a8):
        """A second call is a no-op returning False."""
        assert arm_w4a8["install"]() is True
        assert arm_module.install_arm_cpu_packed_w4a8() is False

    def test_mismatched_marker_raises(self, arm_w4a8):
        """A lone marker (scheme without kernel) is reported as corrupt."""
        arm_w4a8["install"]()
        arm_w4a8["scheme_cls"]._vllm_fl_arm_packed_w4a8 = False
        try:
            with pytest.raises(RuntimeError, match="incomplete ARM W4A8"):
                arm_module.install_arm_cpu_packed_w4a8()
        finally:
            arm_w4a8["scheme_cls"]._vllm_fl_arm_packed_w4a8 = True


class TestArmCpuPackedW4A8SchemeSelection:
    """The patched config scheme selector and scheme __init__."""

    @staticmethod
    def _quants():
        """Weight/activation quant args selecting the packed W4 G128 path."""
        if _REAL_VLLM:
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
        else:
            from compressed_tensors.quantization import QuantizationArgs

            weight = QuantizationArgs(
                num_bits=4,
                type="int",
                strategy="group",
                symmetric=True,
                dynamic=False,
                group_size=128,
            )
            activation = QuantizationArgs(
                num_bits=8,
                type="int",
                strategy="token",
                symmetric=True,
                dynamic=True,
            )
        return weight, activation

    def test_packed_dynamic_token_selects_flagged_scheme(self, arm_w4a8):
        """Pack-quantized + dynamic token W4A8 routes to the patched scheme."""
        arm_w4a8["install"]()
        config = object.__new__(arm_w4a8["config_cls"])
        config.quant_format = "pack-quantized"
        weight, activation = self._quants()

        scheme = config._get_scheme_from_parts(
            weight, activation, format="pack-quantized", layer_name="model.fc"
        )

        assert scheme._vllm_fl_checkpoint_packed is True
        assert scheme._vllm_fl_pack_factor == 8
        assert scheme.group_size == 128

    def test_other_format_falls_back_to_original(self, arm_w4a8):
        """A non-matching format calls through to the original selector."""
        recorded = {}

        def fake_original(self, *args, **kwargs):
            recorded["args"] = (args, kwargs)
            return "sentinel"

        assert arm_w4a8["install"](config_get_scheme=fake_original) is True
        config = object.__new__(arm_w4a8["config_cls"])
        config.quant_format = "dense"
        weight, activation = self._quants()

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
        """Constructing the scheme without ``packed`` records False."""
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


class TestArmCpuPackedW4A8CreateWeights:
    """The patched scheme create_weights (packed checkpoint path)."""

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

    def test_channel_scale_for_row_parallel_full_groups(self, arm_w4a8):
        """group_size -1 + row parallel yields a per-channel scale param."""
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
        self._create(scheme, layer, input_size=512, input_size_per_partition=128)
        # Row parallel: effective group = per-partition input (128), so the
        # scale has one column per channel.
        assert layer.weight_packed.shape == (32, 128 // 8)
        assert layer.weight_scale.shape == (32, 1)
        assert layer.weight_shape.dtype == torch.int64
        # Channel scale params do not carry an input_dim; group ones do.
        assert getattr(layer.weight_scale, "input_dim", None) is None
        assert scheme.kernel._vllm_fl_arm_packed_checkpoint is True

    def test_group_partition_mismatch_rejected(self, arm_w4a8):
        """A partition not divisible by the group size raises ValueError."""
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
                scheme,
                torch.nn.Module(),
                input_size=130,
                input_size_per_partition=130,
            )

    def test_unpacked_scheme_uses_original_create(self, arm_w4a8):
        """A stock (unpacked) scheme delegates to vLLM's create_weights."""
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
        assert recorded["args"][1:6] == (8, 64, [8], 64, torch.bfloat16)
        assert callable(recorded["args"][6])
        assert not hasattr(layer, "weight_packed")


class TestArmCpuPackedW4A8LoadApply:
    """The patched kernel process_weights_after_loading / apply_weights."""

    @staticmethod
    def _loaded(arm_w4a8, *, weight_shape=None):
        """Create a packed scheme + layer with deterministic packed data."""
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
        """Packed G128 loads unpack, repack via flag_gems, and null scales."""
        packed_calls = []

        def fake_pack(weight, scales):
            packed_calls.append((weight.clone(), scales.clone()))
            return torch.zeros(weight.shape[0], weight.shape[1] // 2, dtype=torch.uint8)

        import flag_gems.quantized_linear as ql

        monkeypatch.setattr(ql, "pack_rhs_qsi4c128p", fake_pack)
        scheme, layer, words, codes, scales = self._loaded(arm_w4a8)
        kernel = scheme.kernel

        kernel.process_weights_after_loading(layer)

        # The unpacked int8 [n, k] matrix was handed to the packer in bf16.
        unpacked, packed_scales = packed_calls[0]
        assert unpacked.shape == (32, 256)
        assert unpacked.dtype == torch.int8
        assert torch.equal(unpacked, codes)
        assert torch.equal(packed_scales, scales.to(torch.bfloat16))
        # Scales are dropped and the runtime shape is recorded.
        assert getattr(layer, kernel.w_s_name) is None
        assert layer.weight_shape is None
        assert kernel._vllm_fl_arm_w4a8_shape == (32, 256)
        assert torch.count_nonzero(layer.weight_packed) == 0

    def test_bad_weight_shape_rejected(self, arm_w4a8, monkeypatch):
        """A weight_shape inconsistent with K raises RuntimeError."""
        import flag_gems.quantized_linear as ql

        monkeypatch.setattr(
            ql,
            "pack_rhs_qsi4c128p",
            lambda *args: pytest.fail("packer must not run for invalid shapes"),
        )
        scheme, layer, *_ = self._loaded(arm_w4a8, weight_shape=(32, 999))
        with pytest.raises(RuntimeError, match="weight_shape"):
            scheme.kernel.process_weights_after_loading(layer)

    def test_apply_converts_dtype_and_adds_bias(self, arm_w4a8, monkeypatch):
        """apply_weights converts to bf16, calls the ARM op and casts back."""
        apply_calls = []

        def fake_linear(x, weight, n, k):
            apply_calls.append((x.clone(), weight, n, k))
            return torch.full((x.shape[0], n), 1.5, dtype=torch.bfloat16)

        import flag_gems.quantized_linear as ql

        monkeypatch.setattr(ql, "w4a8_g128_linear", fake_linear)
        scheme, layer, words, codes, scales = self._loaded(arm_w4a8)
        kernel = scheme.kernel
        kernel.process_weights_after_loading(layer)

        x = torch.ones((2, 256), dtype=torch.float32)
        bias = torch.full((32,), 0.25, dtype=torch.float32)
        out = kernel.apply_weights(layer, x, bias)

        assert out.dtype == torch.float32
        assert out.shape == (2, 32)
        # 1.5 + 0.25 computed in bf16 then cast back to the source dtype.
        assert torch.allclose(out, torch.full((2, 32), 1.75, dtype=torch.float32))
        called_x, called_weight, n, k = apply_calls[0]
        assert called_x.dtype == torch.bfloat16
        assert called_weight is layer.weight_packed
        assert (n, k) == (32, 256)

    def test_apply_without_shape_uses_original(self, arm_w4a8):
        """A kernel without the packed runtime shape calls stock vLLM."""
        recorded = []

        def fake_original(self, layer, x, bias=None):
            recorded.append((layer, x, bias))
            return "stock"

        arm_w4a8["install"](kernel_apply=fake_original)
        scheme, layer, *_ = self._loaded(arm_w4a8)
        kernel = scheme.kernel
        # Do NOT process weights: no _vllm_fl_arm_w4a8_shape is present.
        x = torch.ones((1, 256), dtype=torch.bfloat16)
        assert kernel.apply_weights(layer, x, None) == "stock"
        assert recorded[0][0] is layer
