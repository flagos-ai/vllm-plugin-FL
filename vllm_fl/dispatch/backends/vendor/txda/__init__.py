# Copyright (c) 2026 BAAI. All rights reserved.

"""
Txda (Tsingmicro) backend for vllm-plugin-FL dispatch.
"""

from .txda import TxdaBackend

__all__ = ["TxdaBackend"]

# NOTE: do NOT import `.impl.patchs` here. This package's patch.py module is
# imported by PlatformFL.import_kernels() while `import vllm` is still in
# progress (vllm._custom_ops partially initialized); the patchs modules import
# vllm.model_executor and would fail with circular imports. patchs is applied
# from register_ops.py, which the dispatch system loads only after vllm is
# fully imported.
