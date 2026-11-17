# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Normalization shared by the TP-local attention variants."""

import triton
import triton.language as tl


@triton.jit
def sink_factor(max_log, sum_exp, sink):
    # The sink has zero value. Combine its denominator in the same shifted
    # domain as the accumulated PV; never exponentiate absolute logits.
    finite_sink = tl.where(sink == float("inf"), 0.0, sink)
    shift = tl.maximum(max_log, finite_sink)
    shift = tl.where(shift == float("-inf"), 0.0, shift)
    data_scale = tl.math.exp(max_log - shift)
    denominator = sum_exp * data_scale + tl.math.exp(finite_sink - shift)
    denominator = tl.where(denominator > 0.0, denominator, 1.0)
    return tl.where(
        (sum_exp > 0.0) & (sink != float("inf")),
        data_scale / denominator,
        0.0,
    )
