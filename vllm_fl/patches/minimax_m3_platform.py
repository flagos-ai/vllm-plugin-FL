# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors

"""Strict deployment platform using the accepted MetaX backend."""

from vllm_fl.platform import PlatformFL


class MiniMaxPlatformFL(PlatformFL):
    @classmethod
    def check_and_update_config(cls, vllm_config):
        model = vllm_config.model_config
        if model is not None and not set(model.hf_config.architectures or []) <= {
            "MiniMaxM3SparseForConditionalGeneration",
            "MiniMaxM3SparseForCausalLM",
            "MiniMaxM3MTP",
        }:
            raise RuntimeError("This deployment profile is restricted to MiniMax M3")
        super().check_and_update_config(vllm_config)
        vllm_config.parallel_config.worker_cls = (
            "vllm_fl.worker.minimax_m3.MiniMaxWorkerFL"
        )
