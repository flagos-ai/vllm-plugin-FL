# SPDX-License-Identifier: Apache-2.0
"""Vendored Qwen3.8-Flash-Next material and explicit ABI adapters.

The package is split into three ownership layers:

* :mod:`official` contains source copied from the pinned Qwen3.8 image.
* :mod:`vllm024` contains the small adapter/dispatch contract for vLLM 0.24.
* the model's ``common`` and ``gpu`` packages contain FlagOS fallbacks and
  semantic fixes.  They never import a newer vLLM installation at runtime.

The official source modules are deliberately not imported from this package's
``__init__``.  A caller must select an ABI-compatible implementation
explicitly; this prevents the newer official metadata graph from being loaded
as a side effect of model registration.
"""
