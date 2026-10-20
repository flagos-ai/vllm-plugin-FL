# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors

"""Per-apply expert launch context. No thread-local pairing or function replacement."""

import torch
import triton.language as tl

from . import moe_gemv
from vllm_fl.minimax_m3_config import load, thaw

_ALLOWED = {
    "compute_type",
    "use_fp8_w8a8",
    "use_int8_w8a8",
    "use_int8_w8a16",
    "use_int4_w4a16",
    "per_channel_quant",
    "block_shape",
    "B_bias",
}
_COUNTS = {"small_gemv": 0, "small_dot": 0, "g2": 0, "realign": 0, "fallback": 0}


def snapshot():
    return dict(_COUNTS)


class ExpertLaunchContext:
    def __init__(
        self,
        ids,
        base,
        m,
        topk,
        experts,
        expert_map,
        assignment,
        prepare,
        prepare_kwargs,
    ):
        self.ids, self.base, self.m, self.topk, self.experts, self.expert_map = (
            ids,
            base,
            m,
            topk,
            experts,
            expert_map,
        )
        self.assignment = assignment
        self.prepare = prepare
        self.prepare_kwargs = prepare_kwargs
        self.base_g1 = False
        self.small_g1 = False
        self.choice = load().moe["small"].get(str(m))
        self.small_enabled = (
            self.choice is not None
            and topk == 4
            and experts == 128
            and expert_map is None
            and base.get("BLOCK_SIZE_M") == 16
            and tuple(ids.shape) == (m, 4)
        )

    def plain(self, a, kw, contiguous=False):
        return (
            len(a) == 12
            and not (set(kw) - _ALLOWED)
            and all(
                getattr(x, "dtype", None) == torch.bfloat16
                and (not contiguous or x.is_contiguous())
                for x in a[:3]
            )
            and a[3] is None
            and a[4] is None
            and kw.get("compute_type") == tl.bfloat16
            and kw.get("B_bias") is None
            and kw.get("block_shape") is None
            and not any(
                kw.get(k, False)
                for k in (
                    "use_fp8_w8a8",
                    "use_int8_w8a8",
                    "use_int8_w8a16",
                    "use_int4_w4a16",
                    "per_channel_quant",
                )
            )
        )

    def invoke(self, stage, native, *a, **kw):
        A, B, C = a[:3]
        m = self.m
        small_stage = None
        if (
            self.small_enabled
            and self.plain(a, kw, True)
            and A.device == B.device == C.device == self.ids.device
            and not (m <= 8 and a[6] is not None)
            and not (m == 16 and a[6] is None)
        ):
            if (
                tuple(B.shape) == (128, 384, 6144)
                and tuple(A.shape) == (m, 6144)
                and tuple(C.shape) == (m, 4, 384)
                and a[10] == 4
                and a[9] is False
            ):
                small_stage = 1
                self.small_g1 = True
            elif (
                tuple(B.shape) == (128, 6144, 192)
                and tuple(A.shape) == (m * 4, 192)
                and tuple(C.shape) == (m, 4, 6144)
                and a[10] == 1
                and a[9] is True
                and self.small_g1
                and a[5] is not None
                and a[5].dtype == torch.float32
                and tuple(a[5].shape) == (m, 4)
                and a[5].is_contiguous()
                and a[5].device == A.device
            ):
                small_stage = 2
        if small_stage is not None:
            assert small_stage == stage
            cfg = thaw(self.choice["config" + str(stage)])
            if cfg.get("kind") == "naive_gemv":
                _COUNTS["small_gemv"] += 1
                return moe_gemv.launch(A, B, C, a[5], a[6:9], stage, cfg)
            changed = list(a)
            changed[11] = cfg
            a = tuple(changed)
            _COUNTS["small_dot"] += 1
        if self.plain(a, kw):
            if tuple(B.shape) == (128, 384, 6144) and a[10] == 4 and a[9] is False:
                self.base_g1 = True
            elif (
                self.base_g1
                and tuple(B.shape) == (128, 6144, 192)
                and a[10] == 1
                and a[9] is True
                and self.topk == 4
                and self.experts == 128
                and self.expert_map is None
                and A.shape == (m * 4, 192)
                and C.shape == (m, 4, 6144)
                and a[5] is not None
                and a[5].dtype == torch.float32
                and a[5].shape == (m, 4)
                and A.device == B.device == C.device == a[5].device == self.ids.device
            ):
                cfg = dict(a[11])
                cfg.update(thaw(load().moe["g2_configs"].get(str(m), {})))
                if cfg != a[11]:
                    changed = list(a)
                    changed[11] = cfg
                    if cfg["BLOCK_SIZE_M"] != self.base["BLOCK_SIZE_M"]:
                        changed[6:9] = self.prepare(
                            self.ids,
                            cfg,
                            m,
                            self.topk,
                            self.experts,
                            self.expert_map,
                            **self.prepare_kwargs,
                        )
                        _COUNTS["realign"] += 1
                    a = tuple(changed)
                    _COUNTS["g2"] += 1
        _COUNTS["fallback"] += int(small_stage is None)
        return native(*a, **kw)
