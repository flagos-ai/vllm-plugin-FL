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

"""PlatformFL identity, vendor branches and device-fn delegation (E1).

Vendor identity is device-probe-derived state, so each test drives it
through instance attributes (``vendor_name``/``device_type``/``dist_backend``)
on the platform class and restores them via ``monkeypatch`` — the CONTRACT R2
sanctioned boundary for device probes. All expected values below were
verified against the source and the nvidia.yaml platform config.
"""

from __future__ import annotations

import pytest
import torch

# The shim import is load-bearing: it must precede vllm_fl/vllm
# imports on hosts without triton. Do not alphabetize.
# isort: split
from . import _import_shim  # noqa: F401  (inert placeholders on hosts without triton)

# isort: split
import vllm_fl.platform as fl_platform
from vllm_fl.platform import PlatformFL


@pytest.fixture
def restore_identity(monkeypatch):
    """Snapshot vendor/device identity so tests can mutate it freely."""
    saved = (
        PlatformFL.vendor_name,
        PlatformFL.device_type,
        PlatformFL.device_name,
        PlatformFL.dist_backend,
        PlatformFL.torch_device_fn,
    )
    yield monkeypatch
    (
        PlatformFL.vendor_name,
        PlatformFL.device_type,
        PlatformFL.device_name,
        PlatformFL.dist_backend,
        PlatformFL.torch_device_fn,
    ) = saved


class TestIdentityAttributes:
    """Class-level identity derived from the CUDA-shaped device probe."""

    def test_cuda_identity_attributes(self):
        """A CUDA probe yields the NVIDIA/NCCL identity vLLM expects."""
        assert PlatformFL.vendor_name == "nvidia"
        assert PlatformFL.device_type == "cuda"
        assert PlatformFL.device_name == "nvidia"
        assert PlatformFL.dispatch_key == "CUDA"
        assert PlatformFL.ray_device_key == "GPU"

    def test_cuda_dist_backend_and_compile_backend(self):
        """CUDA uses NCCL and routes small compile helpers through inductor."""
        assert PlatformFL.dist_backend == "nccl"
        assert PlatformFL.simple_compile_backend == "inductor"

    def test_device_control_env_var_validated_for_nvidia(self):
        """NVIDIA maps to the validated CUDA_VISIBLE_DEVICES control variable."""
        assert PlatformFL.device_control_env_var == "CUDA_VISIBLE_DEVICES"

    def test_static_capability_and_wrapper_paths(self):
        """Static capability getters return their documented values."""
        assert PlatformFL.is_integrated_gpu() is False
        assert PlatformFL.opaque_attention_op() is True
        assert PlatformFL.support_hybrid_kv_cache() is True
        assert PlatformFL.check_if_supports_dtype(torch.float16) is None
        assert (
            PlatformFL.get_punica_wrapper()
            == "vllm.lora.punica_wrapper.punica_gpu.PunicaWrapperGPU"
        )
        assert (
            PlatformFL.get_static_graph_wrapper_cls()
            == "vllm_fl.compilation.graph.GraphWrapper"
        )

    def test_supported_dtypes_are_bf16_fp16_fp32(self):
        """The accelerator dtype set is exactly bf16/fp16/fp32, in order."""
        assert PlatformFL().supported_dtypes == [
            torch.bfloat16,
            torch.float16,
            torch.float32,
        ]

    def test_support_static_graph_mode_vendors(self, monkeypatch):
        """Every validated vendor opts into the static-graph path."""
        for vendor in (
            "nvidia",
            "ascend",
            "metax",
            "hygon",
            "mthreads",
            "iluvatar",
            "thead",
            "gcu",
            "enflame",
            "kunlunxin",
            "tsingmicro",
            "biren",
            "sunrise",
        ):
            monkeypatch.setattr(PlatformFL, "vendor_name", vendor)
            assert PlatformFL.support_static_graph_mode() is True, vendor

    def test_is_pin_memory_available_matrix(self):
        """Pin memory is offered for accelerator device types only."""
        saved = PlatformFL.device_type
        try:
            for device_type, want in (
                ("cuda", True),
                ("xpu", True),
                ("npu", True),
                ("musa", True),
                ("gcu", True),
                ("cpu", False),
            ):
                PlatformFL.device_type = device_type
                assert PlatformFL.is_pin_memory_available() is want, device_type
        finally:
            PlatformFL.device_type = saved

    def test_get_compile_backend_eager_for_npu(self):
        """Ascend NPU selects upstream eager FX; others inherit inductor."""
        saved = PlatformFL.device_type
        try:
            PlatformFL.device_type = "npu"
            assert PlatformFL.get_compile_backend() == "eager"
            PlatformFL.device_type = "cuda"
            assert PlatformFL.get_compile_backend() == "inductor"
        finally:
            PlatformFL.device_type = saved


class TestVendorClassification:
    """Instance-level vendor classification branches."""

    @pytest.mark.parametrize(
        ("vendor_name", "device_type", "expected"),
        [
            ("nvidia", "cuda", True),
            ("metax", "cuda", True),
            ("mthreads", "musa", True),
            ("gcu", "gcu", True),
            ("enflame", "gcu", True),
            ("iluvatar", "cuda", False),
            ("hygon", "cuda", False),
            ("ascend", "npu", False),
        ],
    )
    def test_is_cuda_alike_matrix(
        self, restore_identity, vendor_name, device_type, expected
    ):
        """is_cuda_alike classifies each vendor exactly per the boundary table."""
        instance = PlatformFL()
        instance.vendor_name = vendor_name
        instance.device_type = device_type
        assert instance.is_cuda_alike() is expected

    @pytest.mark.parametrize(
        ("vendor_name", "is_rocm", "is_cuda"),
        [
            ("nvidia", False, True),
            ("amd", True, False),
            ("hygon", True, False),
            ("ascend", False, False),
        ],
    )
    def test_is_rocm_and_is_cuda(self, restore_identity, vendor_name, is_rocm, is_cuda):
        """ROCm vendors are explicit; is_cuda requires nvidia+cuda together."""
        instance = PlatformFL()
        instance.vendor_name = vendor_name
        instance.device_type = "cuda" if vendor_name != "ascend" else "npu"
        assert instance.is_rocm() is is_rocm
        assert instance.is_cuda() is is_cuda

    def test_is_musa_and_is_gcu_false_without_vendor_runtime(self):
        """Without torch.musa/torch.gcu attributes both probes report False."""
        instance = PlatformFL()
        assert instance.is_musa() is False
        assert instance.is_gcu() is False

    @pytest.mark.parametrize(
        ("vendor_name", "expected"),
        [
            ("nvidia", True),
            ("amd", True),
            ("hygon", True),
            ("thead", True),
            ("iluvatar", True),
            ("ascend", False),
            ("metax", False),
            ("sunrise", False),
            ("kunlunxin", False),
        ],
    )
    def test_use_custom_op_collectives_vendors(
        self, restore_identity, vendor_name, expected
    ):
        """HIP-shaped and ctypes communicators stay opaque to Dynamo."""
        PlatformFL.vendor_name = vendor_name
        assert PlatformFL.use_custom_op_collectives() is expected

    def test_use_custom_op_collectives_ptpu_without_config(self, restore_identity):
        """PTPU with no resolved config defaults to the eager direct path."""
        PlatformFL.device_type = "ptpu"
        assert PlatformFL.use_custom_op_collectives() is False

    def test_use_custom_allreduce_matrix(self, restore_identity):
        """Custom all-reduce is off for hygon and for the flagcx backend."""
        PlatformFL.vendor_name = "nvidia"
        PlatformFL.dist_backend = "nccl"
        assert PlatformFL.use_custom_allreduce() is True

        PlatformFL.vendor_name = "hygon"
        assert PlatformFL.use_custom_allreduce() is False

        PlatformFL.vendor_name = "nvidia"
        PlatformFL.dist_backend = "flagcx"
        assert PlatformFL.use_custom_allreduce() is False

    def test_get_device_communicator_cls_matrix(self, restore_identity):
        """The communicator class follows dist_backend then vendor."""
        PlatformFL.dist_backend = "flagcx"
        assert (
            PlatformFL.get_device_communicator_cls()
            == "vllm_fl.distributed.communicator.CommunicatorFL"
        )
        PlatformFL.dist_backend = "nccl"
        PlatformFL.vendor_name = "thead"
        assert (
            PlatformFL.get_device_communicator_cls()
            == "vllm_fl.distributed.nccl_communicator.NcclCommunicatorFL"
        )
        PlatformFL.vendor_name = "ascend"
        assert (
            PlatformFL.get_device_communicator_cls()
            == "vllm_fl.distributed.device_communicators.npu_communicator.NPUCommunicator"
        )
        PlatformFL.vendor_name = "nvidia"
        assert (
            PlatformFL.get_device_communicator_cls()
            == "vllm.distributed.device_communicators.cuda_communicator.CudaCommunicator"
        )

    def test_dist_backend_dict_contract(self):
        """The static vendor-to-collective mapping keeps its documented shape."""
        assert fl_platform.dist_backend_dict == {
            "npu": "hccl",
            "cuda": "nccl",
            "musa": "mccl",
            "gcu": "eccl",
        }
