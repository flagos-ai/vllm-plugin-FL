from vllm.attention.backends.registry import (
    AttentionBackendEnum,
    register_backend,
)

def register_attention():
    register_backend(
            backend=AttentionBackendEnum.TRITON_ATTN,
            class_path="vllm_fl.attention.attention.AttentionFLBackend",
            is_mamba=False,
        )
