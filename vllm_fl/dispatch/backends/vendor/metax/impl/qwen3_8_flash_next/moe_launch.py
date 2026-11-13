# SPDX-License-Identifier: Apache-2.0
"""Model launch policy: Qwen3.8 TP8 BF16 on MetaX C550 only.
No operator implementation changes. Both selected stages use BM16, so the
assignment must be built using gemm1_config before either GEMM executes.
Other shapes, tiny M1 and the regressing M2048 retain the complete old config.
"""

MIN_ROWS = 8
MAX_ROWS = 1024
GEMM1 = {
    "BLOCK_SIZE_M": 16,
    "BLOCK_SIZE_N": 32,
    "BLOCK_SIZE_K": 64,
    "GROUP_SIZE_M": 1,
    "num_warps": 4,
    "num_stages": 2,
}
GEMM2 = {
    "BLOCK_SIZE_M": 16,
    "BLOCK_SIZE_N": 256,
    "BLOCK_SIZE_K": 32,
    "GROUP_SIZE_M": 1,
    "num_warps": 4,
    "num_stages": 1,
}


def select_stage_configs(
    base_config,
    *,
    num_tokens,
    w1_shape,
    w2_shape,
    topk,
    dtype,
    vendor,
    device_name,
    activation="silu",
    quantized=False,
    has_lora=False,
    has_bias=False,
):
    eligible = (
        vendor == "metax"
        and "C550" in device_name
        and str(dtype) in ("bfloat16", "torch.bfloat16")
        and MIN_ROWS <= num_tokens <= MAX_ROWS
        and tuple(w1_shape) == (512, 160, 2560)
        and tuple(w2_shape) == (512, 2560, 80)
        and topk == 10
        and activation == "silu"
        and not quantized
        and not has_lora
        and not has_bias
    )
    if not eligible:
        return dict(base_config), dict(base_config), False
    gemm1, gemm2 = dict(base_config), dict(base_config)
    gemm1.update(GEMM1)
    gemm2.update(GEMM2)
    return gemm1, gemm2, True
