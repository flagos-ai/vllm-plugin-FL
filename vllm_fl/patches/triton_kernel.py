# Copyright (c) 2026 BAAI. All rights reserved.

"""Reusable Triton kernel launch override utilities."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class KernelLaunchMetaProxy:
    """Override selected launch parameters while preserving the kernel API."""

    def __init__(self, kernel: Any, launch_overrides: Mapping[str, Any]) -> None:
        if not launch_overrides:
            raise ValueError("At least one kernel launch override is required.")
        self._kernel = kernel
        # Keep a plain dict on this hot path. The public property returns a copy.
        self._launch_overrides = dict(launch_overrides)

    @property
    def launch_overrides(self) -> dict[str, Any]:
        return self._launch_overrides.copy()

    def __getitem__(self, grid: Any) -> Any:
        launch = self._kernel[grid]

        def launch_with_overrides(*args: Any, **kwargs: Any) -> Any:
            kwargs.update(self._launch_overrides)
            return launch(*args, **kwargs)

        return launch_with_overrides

    def __getattr__(self, name: str) -> Any:
        return getattr(self._kernel, name)


def patch_kernel_launch_meta(
    module: Any,
    kernel_name: str,
    launch_overrides: Mapping[str, Any],
) -> None:
    """Apply launch overrides to a kernel attribute exactly once."""
    kernel = getattr(module, kernel_name)
    if isinstance(kernel, KernelLaunchMetaProxy):
        if kernel.launch_overrides != dict(launch_overrides):
            raise RuntimeError(
                f"Kernel {kernel_name} already has launch overrides "
                f"{kernel.launch_overrides}, expected {dict(launch_overrides)}."
            )
        return
    setattr(module, kernel_name, KernelLaunchMetaProxy(kernel, launch_overrides))
