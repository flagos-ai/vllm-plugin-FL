# Copyright (c) 2026 BAAI. All rights reserved.

"""FlagCX stream adapter patch for Sunrise/PTPU."""

from __future__ import annotations

import ctypes
import importlib
import logging
import os
import sys

logger = logging.getLogger(__name__)


def patch_flagcx_stream_adapter() -> None:
    """Cache ``flagcxStream_t`` per raw stream pointer; ``adaptor_stream_free`` is a no-op."""
    try:
        from vllm.platforms import current_platform

        if getattr(current_platform, "device_type", None) != "ptpu":
            return

        flagcx_path = os.getenv("FLAGCX_PATH")
        if flagcx_path and os.path.isdir(flagcx_path) and flagcx_path not in sys.path:
            sys.path.append(flagcx_path)

        flagcx_wrapper = importlib.import_module("plugin.interservice.flagcx_wrapper")
        FLAGCXLibrary = flagcx_wrapper.FLAGCXLibrary
        flagcxStream_t = flagcx_wrapper.flagcxStream_t

        if getattr(FLAGCXLibrary, "_sunrise_stream_patch_applied", False):
            return

        def _to_void_p(raw_stream_ptr):
            if isinstance(raw_stream_ptr, ctypes.c_void_p):
                return raw_stream_ptr
            if raw_stream_ptr is None:
                raise ValueError("Stream pointer is None.")
            return ctypes.c_void_p(int(raw_stream_ptr))

        def _extract_raw_stream_ptr(old_stream):
            if isinstance(old_stream, (int, ctypes.c_void_p)):
                return old_stream

            for attr in ("ptpu_stream", "cuda_stream"):
                stream_ptr = getattr(old_stream, attr, None)
                if stream_ptr is not None:
                    return stream_ptr

            stream_fn = getattr(old_stream, "stream", None)
            if callable(stream_fn):
                stream_ptr = stream_fn()
                if stream_ptr is not None:
                    return stream_ptr

            raise AttributeError(
                "Unsupported stream object: expected a raw pointer or one of "
                "`ptpu_stream`, `cuda_stream`, or callable `stream()`."
            )

        def _raw_key(raw_stream_ptr):
            if isinstance(raw_stream_ptr, ctypes.c_void_p):
                return int(raw_stream_ptr.value or 0)
            return int(raw_stream_ptr)

        def _adaptor_stream_copy(self, old_stream):
            raw_stream_ptr = _extract_raw_stream_ptr(old_stream)
            key = _raw_key(raw_stream_ptr)
            cache = self.__dict__.setdefault("_sunrise_flagcx_stream_cache", {})
            cached = cache.get(key)
            if cached is not None:
                return cached

            new_stream = flagcxStream_t()
            self.FLAGCX_CHECK(
                self.devHandle.contents.streamCopy(
                    ctypes.byref(new_stream), _to_void_p(raw_stream_ptr)
                )
            )
            cache[key] = new_stream
            return new_stream

        def _adaptor_stream_free(self, stream):
            return

        FLAGCXLibrary.adaptor_stream_copy = _adaptor_stream_copy
        FLAGCXLibrary.adaptor_stream_free = _adaptor_stream_free
        FLAGCXLibrary._sunrise_stream_patch_applied = True
        logger.info("Patched FlagCX stream adapter for Sunrise/PTPU")
    except Exception as e:
        logger.warning("Failed to patch FlagCX stream adapter for Sunrise: %s", e)
