# SPDX-License-Identifier: Apache-2.0
"""Pinned official QSA/HC kernel sources, kept under the plugin namespace.

The QSA modules are source-vendored from the official Qwen3.8 image and are
the runtime implementation selected by the vLLM 0.24 model adapter.  The
package intentionally does not re-export them; callers choose the exact
kernel entry point through the small compatibility facade.  See
``PROVENANCE.md`` for source hashes and ABI boundaries.
"""
