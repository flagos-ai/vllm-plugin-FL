# Copyright (c) 2026 BAAI. All rights reserved.

"""Sunrise INT8 kernel implementations for the vLLM-native W8A8 path."""

from .scaled_int8_quant import dynamic_scaled_int8_quant, scaled_int8_quant

__all__ = ["dynamic_scaled_int8_quant", "scaled_int8_quant"]
