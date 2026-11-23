# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Platform-local pure-PyTorch machinery for the evas backend.

Holds helpers that are not ops and so must not live in the FlagGems
``_evas/{ops,fused}`` tree: ``e4m3`` and ``packed_cache`` (moved from
``fused/`` — torch-lpu has no fp8 compute/cast and prices every sub-interval
access of a big buffer as the whole backing) and ``double_die`` (backend ABI,
copied so a maintainer rename of a FlagGems-internal path cannot break the
plugin at import time; the two copies stay independent — recheck the four
carried helpers when the FlagGems file changes).

"""
