# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Constructor hooks for the pinned vLLM 0.32 HYV4 model; no core-file edits."""

import torch

from vllm.models.hy_v4.nvidia import model as hy4_base


def initialize_backbone(self, *, vllm_config, prefix="", decoder_cls):
    torch.nn.Module.__init__(self)
    config = hy4_base._normalize_hyv4_config(vllm_config.model_config.hf_text_config)
    cache_config = vllm_config.cache_config
    quant_config = vllm_config.quant_config
    parallel_config = vllm_config.parallel_config
    eplb_config = parallel_config.eplb_config
    self.num_redundant_experts = eplb_config.num_redundant_experts
    self.device = hy4_base.current_platform.device_type
    self.vocab_size = config.vocab_size
    self.is_sparse = hasattr(config, "index_topk")
    if self.is_sparse:
        self.topk_indices_buffer = torch.empty(
            vllm_config.scheduler_config.max_num_batched_tokens,
            config.index_topk,
            dtype=torch.int32,
            device=self.device,
        )
    else:
        self.topk_indices_buffer = None
    self.config = config
    self.quant_config = quant_config
    self.enable_ihc = getattr(config, "enable_ihc", False)
    if hy4_base.get_pp_group().is_first_rank or (
        self.config.tie_word_embeddings and hy4_base.get_pp_group().is_last_rank
    ):
        self.embed_tokens = hy4_base.VocabParallelEmbedding(
            config.vocab_size, config.hidden_size
        )
    else:
        self.embed_tokens = hy4_base.PPMissingLayer()
    self.start_layer, self.end_layer, self.layers = hy4_base.make_layers(
        config.num_hidden_layers,
        lambda prefix: decoder_cls(
            config=config,
            vllm_config=vllm_config,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=prefix,
            topk_indices_buffer=self.topk_indices_buffer,
        ),
        prefix=f"{prefix}.layers",
    )
    if hy4_base.get_pp_group().is_last_rank:
        self.norm = hy4_base.RMSNorm(config.hidden_size, config.rms_norm_eps)
    else:
        self.norm = hy4_base.PPMissingLayer()
    if self.enable_ihc:
        if hy4_base.get_pp_group().is_last_rank:
            self.hc_head = hy4_base.HYV4HCHeadLayer(
                config,
                hidden_size=config.hidden_size,
                hc_mult=config.hc_mult,
                hc_eps=config.hc_eps,
                prefix=f"{prefix}.hc_head",
            )
        else:
            self.hc_head = hy4_base.PPMissingLayer()
        self.make_empty_intermediate_tensors = (
            hy4_base.make_empty_intermediate_tensors_factory(
                ["hidden_states"], config.hc_mult * config.hidden_size
            )
        )
    else:
        self.make_empty_intermediate_tensors = (
            hy4_base.make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], config.hidden_size
            )
        )
    self.expert_weights: hy4_base.MutableSequence[hy4_base.Sequence[torch.Tensor]] = []
    self.num_expert_groups = 1
    self.moe_layers: list[hy4_base.nn.Module] = []
    example_layer: hy4_base.HYV4MoEFused | None = None
    for layer in self.layers:
        if isinstance(layer, hy4_base.PPMissingLayer):
            continue
        assert isinstance(layer, decoder_cls)
        if layer.block_type == "moe":
            assert isinstance(layer.mlp, hy4_base.HYV4MoEFused)
            example_layer = layer.mlp
            self.moe_layers.append(layer.mlp.experts)
    if example_layer is None:
        self.num_moe_layers = 0
        self.num_logical_experts = 0
        self.num_physical_experts = 0
        self.num_local_physical_experts = 0
        self.num_routed_experts = 0
        self.num_redundant_experts = 0
        return
    self.num_moe_layers = len(self.moe_layers)
    self.num_logical_experts = example_layer.n_logical_experts
    self.num_physical_experts = example_layer.n_physical_experts
    self.num_local_physical_experts = example_layer.n_local_physical_experts
    self.num_routed_experts = example_layer.n_routed_experts
    self.num_redundant_experts = example_layer.n_redundant_experts


def initialize_lm(self, *, vllm_config, prefix="", model_cls):
    torch.nn.Module.__init__(self)
    config = vllm_config.model_config.hf_text_config
    quant_config = vllm_config.quant_config
    self.config = config
    self.quant_config = quant_config
    if quant_config is not None:
        quant_config.packed_modules_mapping = self.packed_modules_mapping
    parallel_config = vllm_config.parallel_config
    eplb_config = parallel_config.eplb_config
    self.num_redundant_experts = eplb_config.num_redundant_experts
    self.model = model_cls(
        vllm_config=vllm_config, prefix=hy4_base.maybe_prefix(prefix, "model")
    )
    if hy4_base.get_pp_group().is_last_rank:
        self.lm_head = hy4_base.ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=hy4_base.maybe_prefix(prefix, "lm_head"),
        )
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
    else:
        self.lm_head = hy4_base.PPMissingLayer()
    self.logits_processor = hy4_base.LogitsProcessor(config.vocab_size)
    self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors
