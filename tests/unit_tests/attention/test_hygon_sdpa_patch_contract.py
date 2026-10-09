"""Dependency-stub source contracts; real HCU numerics are tested separately."""

import __future__

import argparse
import importlib.machinery
import importlib.util
import os
import sys
import unittest
import zipfile
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

UPSTREAM_WHEEL = os.environ.get("VLLM_SDPA_SOURCE_WHEEL")
PRODUCTION_SOURCE = Path(__file__).resolve().parents[3] / "vllm_fl/attention/utils.py"


def read_upstream_wrapper_source():
    if UPSTREAM_WHEEL:
        with zipfile.ZipFile(UPSTREAM_WHEEL) as archive:
            member = "vllm/v1/attention/ops/vit_attn_wrappers.py"
            assert archive.getinfo(member).file_size <= 262144
            return archive.read(member).decode("utf-8")
    spec = importlib.machinery.PathFinder.find_spec("vllm", sys.path)
    if spec and spec.submodule_search_locations:
        for location in spec.submodule_search_locations:
            source = Path(location) / "v1/attention/ops/vit_attn_wrappers.py"
            if source.is_file():
                assert source.stat().st_size <= 262144
                return source.read_text(encoding="utf-8")
    raise unittest.SkipTest("vLLM source unavailable; supply VLLM_SDPA_SOURCE_WHEEL")


class Tensor:
    """Track layout metadata only; this object does not perform tensor numerics."""

    def __init__(self, shape, dtype="bf16", device="hcu:0", contiguous=False):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.device = device
        self.is_contiguous = contiguous

    def contiguous(self):
        return Tensor(self.shape, self.dtype, self.device, True)

    def __getitem__(self, index):
        if not isinstance(index, tuple):
            index = (index,)
        if Ellipsis in index:
            position = index.index(Ellipsis)
            fill = (slice(None),) * (len(self.shape) - len(index) + 1)
            index = index[:position] + fill + index[position + 1 :]
        index += (slice(None),) * (len(self.shape) - len(index))
        assert len(index) == len(self.shape)
        assert all(isinstance(part, slice) for part in index)
        shape = tuple(
            len(range(*part.indices(size))) for part, size in zip(index, self.shape)
        )
        contiguous = self.is_contiguous and shape == self.shape
        return Tensor(shape, self.dtype, self.device, contiguous)


class CumulativeLengths:
    def __init__(self, values):
        self.values = values

    def __getitem__(self, index):
        return CumulativeLengths(self.values[index])

    def __sub__(self, other):
        return CumulativeLengths([a - b for a, b in zip(self.values, other.values)])

    def tolist(self):
        return self.values


class SdpaSourceContracts(unittest.TestCase):
    """Load the whole actual modules with stubbed external dependencies.

    No implementation function is copied into this test. The Torch library
    registry and tensor operations are contract stubs. These tests establish
    Python binding, layout, arguments and scope, not Torch/GPU correctness.
    """

    @classmethod
    def setUpClass(cls):
        cls.wrapper_source = read_upstream_wrapper_source()

    def setUp(self):
        self.calls = []
        self.pad_calls = []
        self.logs = []
        self.platform = SimpleNamespace(
            vendor_name="hygon",
            dispatch_key="CUDA",
            is_rocm=lambda: False,
        )
        self.platform.is_cuda = lambda: self.platform.vendor_name == "nvidia"

        def rearrange(tensor, pattern):
            assert pattern.strip() in ("b s h d -> b h s d", "b h s d -> b s h d")
            shape = tensor.shape
            return Tensor(
                (shape[0], shape[2], shape[1], shape[3]), tensor.dtype, tensor.device
            )

        def sdpa(q, k, v, **kwargs):
            self.assertTrue(all(t.is_contiguous for t in (q, k, v)))
            self.calls.append({"shapes": (q.shape, k.shape, v.shape), **kwargs})
            return Tensor((*q.shape[:-1], v.shape[-1]), q.dtype, q.device, True)

        def pad(tensor, padding, mode="constant", value=None):
            self.assertTrue(tensor.is_contiguous)
            self.assertEqual(len(padding), 2)
            self.pad_calls.append(
                {
                    "shape": tensor.shape,
                    "padding": tuple(padding),
                    "mode": mode,
                    "value": 0 if value is None else value,
                    "dtype": tensor.dtype,
                    "device": tensor.device,
                }
            )
            shape = (*tensor.shape[:-1], tensor.shape[-1] + sum(padding))
            # A metadata stub, deliberately requiring post-pad contiguity.
            return Tensor(shape, tensor.dtype, tensor.device)

        def split(tensor, lengths, dim):
            self.assertEqual(dim, 1)
            self.assertEqual(sum(lengths), tensor.shape[dim])
            return tuple(
                Tensor(
                    (tensor.shape[0], length, *tensor.shape[2:]),
                    tensor.dtype,
                    tensor.device,
                )
                for length in lengths
            )

        def cat(tensors, dim):
            self.assertEqual(dim, 1)
            first = tensors[0]
            return Tensor(
                (first.shape[0], sum(t.shape[1] for t in tensors), *first.shape[2:]),
                first.dtype,
                first.device,
            )

        modules = {}
        for name in [
            "torch",
            "torch.nn",
            "torch.nn.functional",
            "einops",
            "vllm",
            "vllm.logger",
            "vllm._aiter_ops",
            "vllm.utils",
            "vllm.utils.torch_utils",
            "vllm.utils.gpu_sync_debug",
            "vllm.platforms",
            "vllm.v1",
            "vllm.v1.attention",
            "vllm.v1.attention.ops",
            "vllm.v1.attention.ops.vit_attn_wrappers",
            "vllm.v1.attention.backends",
            "vllm.v1.attention.backends.registry",
            "vllm.model_executor",
            "vllm.model_executor.layers",
            "vllm.model_executor.layers.attention",
            "vllm.model_executor.layers.attention.mm_encoder_attention",
            "vllm_fl",
            "vllm_fl.attention",
        ]:
            module = ModuleType(name)
            module.__path__ = []
            modules[name] = module
        self.torch = modules["torch"]
        self.torch.ops = SimpleNamespace(vllm=SimpleNamespace())
        self.torch.split = split
        self.torch.cat = cat
        modules["torch.nn.functional"].scaled_dot_product_attention = sdpa
        modules["torch.nn.functional"].pad = pad
        modules["einops"].rearrange = rearrange
        modules["vllm.platforms"].current_platform = self.platform
        modules["vllm._aiter_ops"].rocm_aiter_ops = SimpleNamespace()
        modules["vllm.utils.gpu_sync_debug"].gpu_sync_allowed = nullcontext
        modules["vllm.logger"].init_logger = lambda name: SimpleNamespace(
            info_once=self.logs.append
        )
        modules[
            "vllm.v1.attention.backends.registry"
        ].AttentionBackendEnum = SimpleNamespace(
            FLASH_ATTN="flash", ROCM_AITER_FA="aiter"
        )
        self.mm = modules["vllm.model_executor.layers.attention.mm_encoder_attention"]
        self.mm.MMEncoderAttention = type(
            "MMEncoderAttention", (), {"forward_cuda": lambda *args: None}
        )
        self.vit = modules["vllm.v1.attention.ops.vit_attn_wrappers"]

        def register(op_name, op_func, **kwargs):
            # A stub for the external Torch registry, retaining callable identity.
            setattr(self.torch.ops.vllm, op_name, op_func)

        modules["vllm.utils.torch_utils"].direct_register_custom_op = register
        for name, module in list(modules.items()):
            if "." in name:
                parent, attribute = name.rsplit(".", 1)
                setattr(modules[parent], attribute, module)
        self.module_patch = patch.dict(sys.modules, modules)
        self.module_patch.start()
        self.addCleanup(self.module_patch.stop)
        exec(
            compile(
                self.wrapper_source,
                "<actual-vllm-vit-module>",
                "exec",
                flags=__future__.annotations.compiler_flag,
                dont_inherit=True,
            ),
            self.vit.__dict__,
        )
        self.original_apply = self.vit.apply_sdpa
        self.original_wrapper = self.vit.torch_sdpa_wrapper
        self.registered_function = self.torch.ops.vllm.torch_sdpa_wrapper
        self.dispatch_key = self.platform.dispatch_key
        spec = importlib.util.spec_from_file_location(
            "vllm_fl.attention.utils", PRODUCTION_SOURCE
        )
        plugin = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = plugin
        spec.loader.exec_module(plugin)
        self.patch_attention = plugin.patch_mm_encoder_attention

    def tensors(
        self, sequence=64, kv_heads=8, dtype="bf16", device="hcu:0", dim=128, v_dim=None
    ):
        dimensions = (dim, dim, dim if v_dim is None else v_dim)
        return tuple(
            Tensor((2, sequence, heads, size), dtype, device)
            for heads, size in zip((8, kv_heads, kv_heads), dimensions)
        )

    def test_patch_reaches_pre_registered_callable(self):
        self.patch_attention()
        active_helper = self.vit.apply_sdpa
        helper_calls = []

        def forwarding_helper(*args, **kwargs):
            helper_calls.append(kwargs)
            return active_helper(*args, **kwargs)

        self.vit.apply_sdpa = forwarding_helper
        self.vit.torch_sdpa_wrapper = lambda *a, **kw: self.fail(
            "reassigned module name was called"
        )
        self.assertIs(self.registered_function, self.original_wrapper)
        output = self.vit.vit_torch_sdpa_wrapper(*self.tensors(dim=72), scale=0.37)
        self.assertEqual(output.shape, (2, 64, 8, 72))
        self.assertIs(self.registered_function.__globals__, self.vit.__dict__)
        self.assertEqual(len(self.pad_calls), 3)
        self.assertEqual(helper_calls, [{"scale": 0.37, "enable_gqa": False}])
        self.assertEqual(self.dispatch_key, "CUDA")

    def test_original_unpatched_layout_violates_contiguity_contract(self):
        with self.assertRaises(AssertionError):
            self.torch.ops.vllm.torch_sdpa_wrapper(*self.tensors())
        self.patch_attention()
        self.torch.ops.vllm.torch_sdpa_wrapper(*self.tensors())
        self.assertEqual(len(self.calls), 1)

    def test_scale_and_gqa_are_forwarded_without_head_rewrite(self):
        self.patch_attention()
        for scale, gqa, heads in [(None, False, 8), (0.37, False, 8), (0.37, True, 4)]:
            with self.subTest(scale=scale, gqa=gqa):
                self.torch.ops.vllm.torch_sdpa_wrapper(
                    *self.tensors(kv_heads=heads), scale=scale, enable_gqa=gqa
                )
                call = self.calls[-1]
                self.assertEqual(call["scale"], scale)
                self.assertEqual(call["enable_gqa"], gqa)
                self.assertEqual(call["dropout_p"], 0.0)
                self.assertEqual(
                    call["shapes"],
                    ((2, 8, 64, 128), (2, heads, 64, 128), (2, heads, 64, 128)),
                )

    def test_cu_seqlens_preserves_three_segments_and_output_layout(self):
        self.patch_attention()
        output = self.torch.ops.vllm.torch_sdpa_wrapper(
            *self.tensors(), cu_seqlens=CumulativeLengths([0, 16, 40, 64]), scale=0.125
        )
        self.assertEqual([call["shapes"][0][2] for call in self.calls], [16, 24, 24])
        self.assertEqual([call["scale"] for call in self.calls], [0.125] * 3)
        self.assertEqual(output.shape, (2, 64, 8, 128))

    def test_dtype_device_and_inputs_are_preserved(self):
        self.patch_attention()
        for dtype, device in [("bf16", "hcu:0"), ("fp32", "hcu:1")]:
            for dim in (72, 128):
                with self.subTest(dtype=dtype, device=device, dim=dim):
                    tensors = self.tensors(dtype=dtype, device=device, dim=dim)
                    before = [
                        (t.shape, t.dtype, t.device, t.is_contiguous) for t in tensors
                    ]
                    output = self.torch.ops.vllm.torch_sdpa_wrapper(*tensors)
                    self.assertEqual((output.dtype, output.device), (dtype, device))
                    self.assertEqual(output.shape, (2, 64, 8, dim))
                    self.assertEqual(
                        before,
                        [
                            (t.shape, t.dtype, t.device, t.is_contiguous)
                            for t in tensors
                        ],
                    )

    def test_exact_padding_marker_is_idempotent(self):
        self.patch_attention()
        active = self.vit.apply_sdpa
        self.patch_attention()
        self.assertIs(self.vit.apply_sdpa, active)
        self.assertEqual(active._fl_hygon_sdpa_layout, "bhsd-pad-d72")
        self.assertEqual(len(self.logs), 1)

    def test_before_layout_marker_does_not_block_after_patch(self):
        self.original_apply._fl_hygon_sdpa_layout = "bshd"
        self.patch_attention()
        self.assertIsNot(self.vit.apply_sdpa, self.original_apply)
        self.assertEqual(self.vit.apply_sdpa._fl_hygon_sdpa_layout, "bhsd-pad-d72")

    def test_d72_padding_uses_zero_constant_and_trims_output(self):
        self.patch_attention()
        output = self.torch.ops.vllm.torch_sdpa_wrapper(*self.tensors(dim=72))
        self.assertEqual(output.shape, (2, 64, 8, 72))
        self.assertEqual(len(self.pad_calls), 3)
        for call in self.pad_calls:
            self.assertEqual(call["shape"], (2, 8, 64, 72))
            self.assertEqual(call["padding"], (0, 56))
            self.assertEqual((call["mode"], call["value"]), ("constant", 0))
        self.assertEqual(self.calls[0]["shapes"], ((2, 8, 64, 128),) * 3)

    def test_d72_none_scale_uses_original_dimension(self):
        self.patch_attention()
        self.torch.ops.vllm.torch_sdpa_wrapper(*self.tensors(dim=72))
        self.assertEqual(self.calls[-1]["scale"], 72**-0.5)
        self.assertNotEqual(self.calls[-1]["scale"], 128**-0.5)

    def test_d72_explicit_scale_is_preserved_including_zero(self):
        self.patch_attention()
        for scale in (0.0, 0.37, -0.5, 72**-0.5):
            with self.subTest(scale=scale):
                self.torch.ops.vllm.torch_sdpa_wrapper(
                    *self.tensors(dim=72), scale=scale
                )
                self.assertEqual(self.calls[-1]["scale"], scale)
                self.assertEqual(self.calls[-1]["dropout_p"], 0.0)

    def test_non_d72_or_unequal_value_dimension_does_not_pad(self):
        self.patch_attention()
        for dim, value_dim in ((64, 64), (96, 96), (128, 128), (72, 96), (128, 72)):
            with self.subTest(dim=dim, value_dim=value_dim):
                output = self.torch.ops.vllm.torch_sdpa_wrapper(
                    *self.tensors(dim=dim, v_dim=value_dim)
                )
                self.assertEqual(self.pad_calls, [])
                self.assertIsNone(self.calls[-1]["scale"])
                self.assertEqual(
                    self.calls[-1]["shapes"],
                    ((2, 8, 64, dim), (2, 8, 64, dim), (2, 8, 64, value_dim)),
                )
                self.assertEqual(output.shape, (2, 64, 8, value_dim))

    def test_d72_gqa_and_cu_seqlens_keep_heads_segments_and_trim(self):
        self.patch_attention()
        output = self.torch.ops.vllm.torch_sdpa_wrapper(
            *self.tensors(dim=72, kv_heads=4),
            cu_seqlens=CumulativeLengths([0, 16, 40, 64]),
            scale=0.125,
            enable_gqa=True,
        )
        self.assertEqual(output.shape, (2, 64, 8, 72))
        self.assertEqual(len(self.pad_calls), 9)
        for call, length in zip(self.calls, (16, 24, 24)):
            self.assertEqual(
                call["shapes"],
                ((2, 8, length, 128), (2, 4, length, 128), (2, 4, length, 128)),
            )
            self.assertTrue(call["enable_gqa"])
            self.assertEqual(call["scale"], 0.125)
            self.assertEqual(call["dropout_p"], 0.0)

    def test_old_contiguous_marker_does_not_block_padding_patch(self):
        self.original_apply._fl_hygon_sdpa_layout = "bhsd"
        self.patch_attention()
        self.assertIsNot(self.vit.apply_sdpa, self.original_apply)
        self.assertEqual(self.vit.apply_sdpa._fl_hygon_sdpa_layout, "bhsd-pad-d72")

    def test_other_vendors_do_not_change_sdpa_helper(self):
        for vendor in ["cpu", "nvidia", "metax", "ascend"]:
            with self.subTest(vendor=vendor):
                self.platform.vendor_name = vendor
                self.patch_attention()
                self.assertIs(self.vit.apply_sdpa, self.original_apply)
        self.assertEqual(self.logs, [])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-wheel", type=Path)
    parser.add_argument("--source", type=Path)
    options, unittest_args = parser.parse_known_args()
    if options.upstream_wheel:
        UPSTREAM_WHEEL = str(options.upstream_wheel)
    if options.source:
        PRODUCTION_SOURCE = options.source
    print(
        "SCOPE: complete real modules with dependency stubs; not ordinary installed import, Torch dispatch or numerical acceptance."
    )
    unittest.main(argv=[sys.argv[0], *unittest_args])
