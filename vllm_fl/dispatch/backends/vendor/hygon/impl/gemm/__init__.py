"""Hygon GEMM implementations."""

from .bf16_fp32_gemm import linear_bf16_fp32_hygon

__all__ = ["linear_bf16_fp32_hygon"]
