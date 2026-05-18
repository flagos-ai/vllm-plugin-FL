# Copyright (c) 2025 BAAI. All rights reserved.


def __getattr__(name):
    if name in ("FusedMoEFL", "SharedFusedMoEFL", "UnquantizedFusedMoEMethodFL"):
        from vllm_fl.ops.fused_moe.layer import (
            FusedMoEFL,
            SharedFusedMoEFL,
            UnquantizedFusedMoEMethodFL,
        )
        globals().update({
            "FusedMoEFL": FusedMoEFL,
            "SharedFusedMoEFL": SharedFusedMoEFL,
            "UnquantizedFusedMoEMethodFL": UnquantizedFusedMoEMethodFL,
        })
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["FusedMoEFL", "SharedFusedMoEFL", "UnquantizedFusedMoEMethodFL"]
