# Copyright (c) 2026 BAAI. All rights reserved.
"""LPU ops-fallback patches (individual files, one per model-domain fallback).

Each file provides an ``apply_lpu_*_patch()`` function that replaces a
Triton/CUDA kernel or operation that does not run on LPU with a pure-PyTorch
fallback.
Kept as separate files because each is model domain-specific and independently
large — they are not always needed together.  ``dsv4_weights.py`` additionally
carries the LPU weight loader that its patch installs, since the two are one
concern.
"""
