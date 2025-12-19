import pytest
import torch

from vllm_fl.platform import PlatformFL


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for PlatformFL CUDA tests",
)
class TestPlatformFLCUDA:

    @classmethod
    def setup_class(cls):
        cls.platform = PlatformFL()

    def test_is_cuda_alike(self):
        assert self.platform.is_cuda_alike() is True

    def test_supported_dtypes(self):
        dtypes = self.platform.supported_dtypes
        assert torch.float16 in dtypes
        assert torch.bfloat16 in dtypes
        assert torch.float32 in dtypes

    def test_check_if_supports_dtype_no_error(self):
        PlatformFL.check_if_supports_dtype(torch.float16)
        PlatformFL.check_if_supports_dtype(torch.float32)

    def test_set_device_and_empty_cache(self):
        device = torch.device("cuda:0")
        PlatformFL.set_device(device)
        PlatformFL.empty_cache()

        x = torch.randn(4, 4, device=device)
        assert x.device.type == "cuda"

    def test_get_current_memory_usage(self):
        device = torch.device("cuda:0")
        PlatformFL.set_device(device)

        _ = torch.randn(512, 512, device=device)

        mem = PlatformFL.get_current_memory_usage(device)
        assert isinstance(mem, (int, float))
        assert mem >= 0

    def test_get_device_name(self):
        name = PlatformFL.get_device_name()
        assert isinstance(name, str)
        assert len(name) > 0

    def test_is_pin_memory_available(self):
        assert PlatformFL.is_pin_memory_available() is True

    def test_get_attn_backend_cls(self):
        backend = PlatformFL.get_attn_backend_cls(
            selected_backend=None,
            head_size=64,
            dtype=torch.float16,
            kv_cache_dtype=None,
            block_size=16,
            use_v1=True,
            use_mla=False,
            has_sink=False,
            use_sparse=False,
        )
        assert backend == "vllm_fl.attention.attention.AttentionFLBackend"

    def test_support_flags(self):
        assert PlatformFL.support_static_graph_mode() is True
        assert PlatformFL.support_hybrid_kv_cache() is False
        assert PlatformFL.opaque_attention_op() is True

