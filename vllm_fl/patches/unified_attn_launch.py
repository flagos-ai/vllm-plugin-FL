# Copyright (c) 2026 BAAI. All rights reserved.
"""FlagOS S2 赛道二：重标定 vLLM unified_attention 的 launch 参数。

为什么需要它
------------
vLLM 0.24.0 的 prefill 与 decode 共用 ``unified_attention``，而 prefill 的
``BLOCK_M`` 被固定为 16。MiniCPM5-2B 是 16 q-head / 2 KV-head（GQA 8:1），
``BLOCK_Q = 16 // 8 = 2`` —— 每个 program 只算 **2 个 query 位置**，
q_len=2048 的 chunked prefill 要开 1025 个 q-block，实测只有 4.5 TFLOPS。
上游的 ``tuned_large_head`` 同方向补丁被
``head_size == 256 and is_device_capability_family(100)`` 两重门控卡死，
MiniCPM5-2B（head_dim=128，非 sm100）完全吃不到。

改什么
------
只改 launch 参数，不改任何语义、不改数据布局：

* prefill（``max_seqlen_q > 1 and head_size <= 128``）：
  ``BLOCK_M=256 / num_warps=16 / num_stages=2 / TILE_SIZE_PREFILL=128``
* decode（确认走 3D 内核时）：``num_warps=2 / num_stages=1``

实测（天数 BI-V150，MiniCPM5-2B，官方 benchmark_throughput_serve.py）：
4k  [4096,1024,64,256]  1871.19 -> 3231.34 tok/s  (+72.7%)
16k [16384,1024,64,128]  895.78 -> 2536.07 tok/s  (+183.1%)
精度 math_500 Level 3 (105 题) 96.2% -> 98.1%（合格线 0.95）

为什么不直接改 vllm 的文件
--------------------------
提交包里只有 vllm-plugin-FL，vllm 的 wheel 应保持原样。这里沿用
``vllm_fl.patches.moe_sum`` 的思路做**运行时适配**，只是适配点更深：
要改的是函数内的局部变量，外部包一层 wrapper 够不到。
因此取 ``unified_attention`` 的源码做三处定点替换后在原模块命名空间里
重新 exec —— 等价于打补丁，但**磁盘上的 vllm 一个字节都不动**，
也不需要 vendor 那 334 行第三方代码。

安全性
------
* 三处锚点必须全部命中（或全部已是打过的状态），否则整体放弃、回退原生，
  不会出现"半打补丁"的中间态。vllm 版本变动导致锚点失效时只会失去加速，
  不会改变正确性。
* 幂等：重复调用直接返回。
* ``FLAGOS_P1=0`` 可整体关闭，回到原生行为（A/B 用）。
* 各 ``P1_*`` 环境变量与磁盘版补丁完全一致，默认即实测最优点。

数值
----
BLOCK_M / num_warps / num_stages 只改 query 行分组与并行方式，不动 KV 归约
顺序，逐位一致。TILE=128 改变 KV 归约分块，max|diff| 0.00024~0.00391，
均小于 1 个 bf16 ULP@1.0（0.0078）。
"""

import inspect
import logging
import os
import textwrap

logger = logging.getLogger(__name__)

_MARK = "_flagos_p1_unified_attn"

OLD_BLOCK = """    BLOCK_M = (
        16 if num_queries_per_kv <= 16 else triton.next_power_of_2(num_queries_per_kv)
    )
    BLOCK_Q = BLOCK_M // num_queries_per_kv"""

NEW_BLOCK = """    # --- FLAGOS_S2_P1: prefill attention BLOCK_M 修正（默认开启，P1_BLOCK_M=0 可关）---
    _p1_env = __import__("os").environ
    _p1_prefill = max_seqlen_q > 1 and head_size <= 128
    _p1_bm = int(_p1_env.get("P1_BLOCK_M", "256"))
    _p1_bmd = int(_p1_env.get("P1_BLOCK_M_DEC", "0"))
    if _p1_prefill and _p1_bm > 0:
        BLOCK_M = _p1_bm
    elif (not _p1_prefill) and _p1_bmd > 0:
        BLOCK_M = _p1_bmd
    else:
        BLOCK_M = (
            16 if num_queries_per_kv <= 16
            else triton.next_power_of_2(num_queries_per_kv)
        )
    BLOCK_Q = BLOCK_M // num_queries_per_kv"""

OLD_WARP = """    if tuned_large_head:
        BLOCK_M = 32
        BLOCK_Q = BLOCK_M // num_queries_per_kv
        launch_num_warps = 8
        launch_num_stages = 2"""

NEW_WARP = OLD_WARP + """
    _p1_w = int(_p1_env.get("P1_WARPS", "16"))
    _p1_s = int(_p1_env.get("P1_STAGES", "2"))
    _p1_wd = int(_p1_env.get("P1_WARPS_DEC", "2"))
    _p1_sd = int(_p1_env.get("P1_STAGES_DEC", "1"))
    # decode 只在确实走 3D 内核时才调参；2D decode 的最优值不同，不动它。
    _p1_decode3d = (
        not _p1_prefill
        and seq_threshold_3D is not None
        and num_par_softmax_segments is not None
        and softmax_segm_output is not None
        and softmax_segm_max is not None
        and softmax_segm_expsum is not None
        and num_seqs <= seq_threshold_3D
    )
    _p1_dec2d = int(_p1_env.get("P1_DEC_2D", "0"))
    _p1_decode_any = (not _p1_prefill) and (_p1_decode3d or _p1_dec2d)
    if _p1_prefill:
        if _p1_w > 0:
            launch_num_warps = _p1_w
        if _p1_s > 0:
            launch_num_stages = _p1_s
    elif _p1_decode_any:
        if _p1_wd > 0:
            launch_num_warps = _p1_wd
        if _p1_sd > 0:
            launch_num_stages = _p1_sd"""

OLD_TILE = """    if tuned_large_head:
        TILE_SIZE_PREFILL = 128"""

NEW_TILE = OLD_TILE + """
    # --- FLAGOS_S2_P1: prefill KV tile 宽度 ---
    # BM=256/W=16/S=2 下实测 T=128 在 kv=2048~16384 全更快（1.23~1.31x），
    # max|diff| <= 0.00391（< 1 个 bf16 ULP@1.0），故默认门控为 0（全长度启用）。
    _p1_t = int(_p1_env.get("P1_TILE", "128"))
    _p1_tmin = int(_p1_env.get("P1_TILE_MIN_KV", "0"))
    if _p1_prefill and _p1_t > 0:
        TILE_SIZE_PREFILL = _p1_t if max_seqlen_k >= _p1_tmin else 32
    _p1_td = int(_p1_env.get("P1_TILE_DEC", "0"))
    if not _p1_prefill and _p1_td > 0:
        TILE_SIZE_DECODE = _p1_td"""

_PATCHES = ((OLD_BLOCK, NEW_BLOCK), (OLD_WARP, NEW_WARP), (OLD_TILE, NEW_TILE))

# ---- MACA（沐曦 C500）锚点候选集 -------------------------------------------
# MACA 栈的 vLLM wheel / vendor 文本没有 tuned_large_head 块，TILE 走
# _get_tile_size()，且 kernel 调用不带 launch_kwargs 机制，因此：
#   槽1 与 V150 相同（BLOCK_M 块逐字一致）；
#   槽2 挂在 TILE_SIZE_DECODE 赋值之后，定义 prefill/decode 的 warps/stages/tile
#       覆盖与 **_p1_extra 字典（decode 只在确实走 3D 内核时生效）；
#   槽3/槽4 给 2D/3D 两个 kernel 调用追加 **_p1_extra（空字典 = 无操作）。
# 全部命中才生效，否则整体回退原生（与 V150 集同一安全语义）。

V_OLD_TILEDEC = """    TILE_SIZE_DECODE = _get_tile_size(
        head_size,
        sliding_window_val,
        q.element_size(),
        is_prefill=False,
    )"""

V_NEW_TILEDEC = V_OLD_TILEDEC + """
    # --- FLAGOS_S2_P1: metax 变体（launch 参数经 **_p1_extra 注入）---
    _p1_w = int(_p1_env.get("P1_WARPS", "16"))
    _p1_s = int(_p1_env.get("P1_STAGES", "2"))
    _p1_t = int(_p1_env.get("P1_TILE", "128"))
    _p1_tmin = int(_p1_env.get("P1_TILE_MIN_KV", "0"))
    _p1_wd = int(_p1_env.get("P1_WARPS_DEC", "2"))
    _p1_sd = int(_p1_env.get("P1_STAGES_DEC", "1"))
    _p1_dec2d = int(_p1_env.get("P1_DEC_2D", "0"))
    _p1_extra2d = {}
    _p1_extra3d = {}
    _p1_decode3d = (
        seq_threshold_3D is not None
        and num_par_softmax_segments is not None
        and softmax_segm_output is not None
        and softmax_segm_max is not None
        and softmax_segm_expsum is not None
        and num_seqs <= seq_threshold_3D
    )
    if _p1_prefill:
        _p1_extra2d = {"num_warps": _p1_w, "num_stages": _p1_s}
        if _p1_t > 0 and max_seqlen_k >= _p1_tmin:
            TILE_SIZE_PREFILL = _p1_t
    elif _p1_decode3d:
        _p1_extra3d = {"num_warps": _p1_wd, "num_stages": _p1_sd}
    elif _p1_dec2d:
        _p1_extra2d = {"num_warps": _p1_wd, "num_stages": _p1_sd}"""

V_OLD_L2D = """            num_seqs=num_seqs,
            BLOCK_M=BLOCK_M,
            USE_FP8=output_scale is not None,
        )"""

V_NEW_L2D = """            num_seqs=num_seqs,
            BLOCK_M=BLOCK_M,
            USE_FP8=output_scale is not None,
            **_p1_extra2d,
        )"""

V_OLD_L3D = """            num_seqs=num_seqs,
            BLOCK_M=BLOCK_M,
            NUM_SEGMENTS_PER_SEQ=num_par_softmax_segments,
        )"""

V_NEW_L3D = """            num_seqs=num_seqs,
            BLOCK_M=BLOCK_M,
            NUM_SEGMENTS_PER_SEQ=num_par_softmax_segments,
            **_p1_extra3d,
        )"""

_PATCHES_METAX = (
    (OLD_BLOCK, NEW_BLOCK),
    (V_OLD_TILEDEC, V_NEW_TILEDEC),
    (V_OLD_L2D, V_NEW_L2D),
    (V_OLD_L3D, V_NEW_L3D),
)

# 候选集按序尝试，第一个全部命中的生效。V150 官方 wheel 在前（含
# tuned_large_head / launch_kwargs 机制），MACA 变体在后。
_PATCH_SETS = (_PATCHES, _PATCHES_METAX)


def _apply(src, patch_set):
    """对单个锚点候选集做全有全无替换。

    返回 (status, 源码)：status ∈ {"applied", "already", "miss"}。
    """
    out = src
    applied = 0
    for old, new in patch_set:
        if new in out:
            continue  # 该处已是打好补丁的状态，跳过
        if old not in out:
            return "miss", src
        out = out.replace(old, new, 1)
        applied += 1
    return ("applied" if applied else "already"), out


def _rebuild(module):
    """在原模块命名空间里重建打过补丁的 unified_attention。

    依次尝试 _PATCH_SETS 中的锚点候选集（V150 官方 wheel / MACA 变体），
    第一个全部命中的集合生效；全部不命中时回退原生（调用方拿到 None），
    并可经 P1_DUMP_SRC 导出实际源码供离线迭代锚点。
    """
    try:
        src = textwrap.dedent(inspect.getsource(module.unified_attention))
    except (OSError, TypeError) as exc:
        logger.warning("P1: 无法读取 unified_attention 源码，保持原生：%s", exc)
        return None

    for patch_set in _PATCH_SETS:
        status, out = _apply(src, patch_set)
        if status == "miss":
            continue
        if status == "already":
            return module.unified_attention  # 全部锚点处均已是打好补丁的状态
        namespace = dict(module.__dict__)
        namespace.setdefault("__builtins__", __builtins__)
        try:
            exec(compile(out, module.__file__, "exec"), namespace)
        except Exception as exc:  # 任何意外都只回退原生，绝不影响启动
            logger.warning("P1: 候选集重建失败，尝试下一候选集：%s", exc)
            continue
        return namespace["unified_attention"]

    dump = os.environ.get("P1_DUMP_SRC", "")
    if dump:
        try:
            with open(dump, "w", encoding="utf-8") as _f:
                _f.write(src)
            logger.warning("P1: 锚点全部未命中，unified_attention 源码已导出：%s", dump)
        except Exception:
            pass
    logger.warning("P1: 补丁锚点未命中（vllm 版本可能已变），保持原生")
    return None


def patch_vllm_unified_attention() -> bool:
    """把调优后的 unified_attention 装进 vLLM 的 Triton attention 后端。"""
    import os

    if os.environ.get("FLAGOS_P1", "1") == "0":
        logger.info("P1: FLAGOS_P1=0，保持原生 unified_attention")
        return False

    try:
        import vllm.v1.attention.backends.triton_attn as backend
        import vllm.v1.attention.ops.triton_unified_attention as ops
    except Exception as exc:
        # register_model() 会在参数解析、EngineCore、Worker 等多个阶段/进程被调用，
        # 早期阶段 import 可能因依赖未就绪而失败；Worker 进程会再次调用本函数。
        logger.warning("P1: 目标模块不可用，保持原生：%s", exc)
        return False

    if getattr(ops.unified_attention, _MARK, False):
        return False

    new_fn = _rebuild(ops)
    if new_fn is None:
        return False

    new_fn.__dict__[_MARK] = True
    ops.unified_attention = new_fn
    # triton_attn 用的是 from-import，名字绑在自己的模块全局里，必须一起换。
    backend.unified_attention = new_fn
    logger.info(
        "P1: unified_attention launch 参数已重标定 "
        "(prefill BM=256/W=16/S=2/T=128, decode W=2/S=1)"
    )
    return True


__all__ = ["patch_vllm_unified_attention"]
