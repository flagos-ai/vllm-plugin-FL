# Copyright (c) 2026 BAAI. All rights reserved.

"""
Template backend implementation.

This is a template for creating vendor-specific backend implementations.
Replace 'Template' with your vendor name and implement the required operators.
"""

from __future__ import annotations

from typing import Optional, Union

import torch

from vllm_fl.dispatch.backends.base import Backend


class TemplateBackend(Backend):
    """
    Template backend for operator implementations.

    Replace this with your vendor-specific backend implementation.
    Inherit from Backend and implement the required operator methods.
    """

    _available: Optional[bool] = None

    @property
    def name(self) -> str:
        """Return the backend name (e.g., 'my_vendor')."""
        return "template"

    @property
    def vendor(self) -> Optional[str]:
        """Return the vendor name (e.g., 'my_vendor')."""
        return "template"

    def is_available(self) -> bool:
        """
        Check if the vendor hardware and libraries are available.

        Implement this method to detect if your vendor's hardware/software
        is present and functional.
        """
        if TemplateBackend._available is None:
            try:
                # TODO: Add your vendor-specific availability check here
                # Example:
                # import your_vendor_library
                # if your_vendor_library.is_available():
                #     TemplateBackend._available = True
                # else:
                #     TemplateBackend._available = False

                TemplateBackend._available = False
            except (ImportError, AttributeError):
                TemplateBackend._available = False
        return TemplateBackend._available

    # ==================== Operator Implementations ====================
    # Implement the operators your backend supports below.
    # You can refer to the base.py file for the full list of available operators.

    # Example operator implementation:
    # def silu_and_mul(self, x: torch.Tensor) -> torch.Tensor:
    #     """
    #     SiLU activation followed by element-wise multiplication.
    #
    #     Args:
    #         x: Input tensor of shape [..., 2*d]
    #
    #     Returns:
    #         Output tensor of shape [..., d]
    #     """
    #     from .impl.activation import silu_and_mul_template
    #     return silu_and_mul_template(x)
