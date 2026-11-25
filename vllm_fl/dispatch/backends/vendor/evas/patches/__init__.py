# Copyright (c) 2026 BAAI. All rights reserved.
"""Evas LPU compatibility patches.

Stock vLLM does not run on LPU: its Triton/CUDA kernels fail to compile or
misbehave there, and a few driver capabilities the framework relies on are
missing.  Every LPU stand-in for that is installed from here as an idempotent,
LPU-gated monkey-patch.
"""
