"""Short-context dense-attention fallback for GLM DSA on Ascend."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from copy import copy
from typing import Any


def prepare_glm_dsa_dense_fallback(model_config: Any) -> bool:
    """Disable MLA for GLM DSA when dense attention is semantically equivalent."""

    hf_config = getattr(model_config, "hf_text_config", None)
    if hf_config is None:
        hf_config = getattr(model_config, "hf_config", None)
    if getattr(hf_config, "model_type", None) != "glm_moe_dsa":
        return False

    index_topk = getattr(hf_config, "index_topk", None)
    max_model_len = getattr(model_config, "max_model_len", None)
    if index_topk is None or max_model_len is None:
        raise ValueError("GLM DSA dense fallback requires index_topk and max_model_len")
    if max_model_len > index_topk:
        raise ValueError(
            "Ascend GLM DSA currently uses dense attention and requires "
            f"--max-model-len <= index_topk ({index_topk}), got {max_model_len}."
        )

    arch_config = getattr(model_config, "model_arch_config", None)
    if arch_config is None:
        raise ValueError("GLM DSA dense fallback requires model_arch_config")
    original_arch = (
        arch_config.is_deepseek_mla,
        arch_config.head_size,
        arch_config.total_num_kv_heads,
    )
    try:
        arch_config.is_deepseek_mla = False
        arch_config.head_size = hf_config.qk_nope_head_dim + hf_config.qk_rope_head_dim
        arch_config.total_num_kv_heads = hf_config.num_attention_heads
        if getattr(model_config, "use_mla", arch_config.is_deepseek_mla):
            raise RuntimeError("Failed to select dense attention for GLM DSA")
    except BaseException:
        (
            arch_config.is_deepseek_mla,
            arch_config.head_size,
            arch_config.total_num_kv_heads,
        ) = original_arch
        raise
    return True


@contextmanager
def use_glm_dsa_dense_hf_config(model_config: Any) -> Iterator[Any]:
    """Build GLM with a private HF config that does not expose ``index_topk``.

    Transformers 5 exposes ``index_topk`` on both the instance and its shared
    config class. A shallow copy with a private subclass avoids mutating either
    location while making vLLM construct its dense attention path.
    """

    original_hf_config = model_config.hf_config
    original_hf_text_config = model_config.hf_text_config
    if original_hf_text_config is not original_hf_config:
        raise ValueError("GLM DSA dense fallback requires a text-only HF config")

    class _DenseGlmConfig(type(original_hf_config)):
        def __getattribute__(self, name: str) -> Any:
            if name == "index_topk":
                raise AttributeError(name)
            return super().__getattribute__(name)

    dense_hf_config = copy(original_hf_config)
    dense_hf_config.__class__ = _DenseGlmConfig
    if hasattr(dense_hf_config, "index_topk"):
        raise RuntimeError("Unable to hide GLM DSA index_topk from vLLM")

    model_config.hf_config = dense_hf_config
    model_config.hf_text_config = dense_hf_config
    try:
        yield dense_hf_config
    finally:
        try:
            model_config.hf_config = original_hf_config
        finally:
            model_config.hf_text_config = original_hf_text_config


__all__ = [
    "prepare_glm_dsa_dense_fallback",
    "use_glm_dsa_dense_hf_config",
]
