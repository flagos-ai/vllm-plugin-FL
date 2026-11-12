# Copyright (c) 2026 BAAI. All rights reserved.

"""CPU checks for the runner's vLLM 0.32 allocation and state-copy contracts."""

import __future__

import ast
from contextlib import nullcontext
from copy import copy
from pathlib import Path
from types import SimpleNamespace
from typing import Any, NamedTuple, cast
from unittest.mock import Mock

import numpy as np
import torch


def _runner_class(methods, **namespace):
    path = Path(__file__).resolve().parents[3] / "vllm_fl/worker/model_runner.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "ModelRunnerFL"
    )
    cls.bases = []
    cls.decorator_list = []
    cls.body = [
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in methods
    ]
    for method in cls.body:
        method.decorator_list = []
    scope = {"Any": Any, "cast": cast, "nullcontext": nullcontext, **namespace}
    exec(
        compile(
            ast.Module(body=[cls], type_ignores=[]),
            str(path),
            "exec",
            flags=__future__.annotations.compiler_flag,
        ),
        scope,
    )
    runner_class = scope["ModelRunnerFL"]
    # AST contracts do not run __init__; emulate its non-Ascend defaults.
    runner_class.use_ascend_attention = False
    runner_class.use_ascend_hybrid_cache = False
    return runner_class


def test_allocator_binds_kernel_views_and_preserves_shared_manager_views():
    backing = torch.arange(16, dtype=torch.float32).view(4, 1, 2, 2)
    allocation = {"model.layers.0.attn": backing}

    def allocate(config, device, layout, kernel_sizes):
        assert device == torch.device("cpu")
        assert layout == "LBNHC" and kernel_sizes == [1]
        return allocation.copy()

    main_group, draft_group = object(), object()

    bind_mock = Mock()

    runner = _runner_class(
        {"initialize_kv_cache_tensors"},
        allocate_kv_cache=allocate,
        bind_kv_cache=bind_mock,
        logger=Mock(),
    )()
    runner.device = torch.device("cpu")
    runner.cache_config = SimpleNamespace(
        get_resolved_kv_cache_layout=lambda: "LBNHC",
        cache_dtype="auto",
    )
    runner.shared_kv_cache_layers = {"model.layers.1.attn": "model.layers.0.attn"}
    runner.model_config = SimpleNamespace(
        hf_config=SimpleNamespace(model_type="qwen3_5")
    )
    runner.attn_groups = [[main_group]]
    runner.speculative_config = object()
    runner.drafter = SimpleNamespace(draft_attn_groups=[draft_group])
    runner.compilation_config = SimpleNamespace(static_forward_context={})
    runner.kv_caches = []
    runner.use_uniform_kv_cache = lambda groups: False
    config = SimpleNamespace(kv_cache_groups=["cache-group"])
    caches = runner.initialize_kv_cache_tensors(config, [1])
    assert caches["model.layers.0.attn"] is caches["model.layers.1.attn"]
    bind_mock.assert_called_once()

    # Like allocation, model construction registers JIT providers. Keep the
    # native 0.32 registry active around the actual loader call: registrations
    # silently disappear outside it, and dummy forward cannot discover every
    # specialization needed during graph capture.
    source = Path(__file__).resolve().parents[3] / "vllm_fl/worker/model_runner.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    load_model = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "load_model"
    )
    capture_contexts = [
        node
        for node in ast.walk(load_model)
        if isinstance(node, ast.With)
        and any(
            ast.unparse(item.context_expr) == "self.jit_warmup_registry.activate()"
            for item in node.items
        )
    ]
    assert any(
        isinstance(node, ast.Call)
        and ast.unparse(node.func) == "model_loader.load_model"
        for context in capture_contexts
        for node in ast.walk(context)
    )


def test_spec_customization_needs_no_removed_backend_layout_methods():
    class Spec:
        pass

    class Attention:
        kv_sharing_target_layer_name = None

        def get_kv_cache_spec(self, config):
            return raw_spec

        def get_attn_backend(self):
            return backend

    raw_spec, customized_spec = Spec(), Spec()
    backend = SimpleNamespace(customize_spec=lambda spec: customized_spec)
    runner = _runner_class(
        {"get_kv_cache_spec"},
        Attention=Attention,
        AttentionSpec=Spec,
        AttentionLayerBase=object,
        has_ec_transfer=lambda: False,
        get_layers_from_vllm_config=lambda *args: {"layer.0": Attention()},
        customize_attention_spec=lambda owner, spec: owner.customize_spec(spec),
    )()
    runner.vllm_config = object()
    assert runner.get_kv_cache_spec() == {"layer.0": customized_spec}


def test_mamba_copy_functions_are_resolved_by_state_type_once():
    groups = [SimpleNamespace(mamba_type="gdn"), SimpleNamespace(mamba_type="mamba2")]
    copy_functions = {"gdn": object(), "mamba2": object()}
    validate = Mock()
    runner = _runner_class(
        {"_get_mamba_state_copy_funcs"},
        mamba_utils=SimpleNamespace(
            get_mamba_groups=lambda config: groups,
            validate_mamba_state_copy_funcs=validate,
        ),
    )()
    runner._mamba_state_copy_funcs = None
    runner.kv_cache_config = object()
    provider = Mock(return_value=copy_functions)
    runner.model = SimpleNamespace(get_mamba_state_copy_funcs=provider)
    assert runner._get_mamba_state_copy_funcs() is copy_functions
    assert runner._get_mamba_state_copy_funcs() is copy_functions
    provider.assert_called_once_with({"gdn", "mamba2"})
    validate.assert_called_once_with(groups, copy_functions)


def test_kv_zeroer_receives_manager_block_count():
    zeroer = Mock()
    runner = _runner_class({"_init_kv_zero_meta"}, KVBlockZeroer=zeroer)()
    runner.device = torch.device("cpu")
    runner._kv_cache_spec_attn_group_iterator = lambda: iter(())
    runner._kernel_block_sizes = [16]
    runner.runner_only_attn_layers = set()
    runner.compilation_config = SimpleNamespace(static_forward_context={})
    runner.kv_cache_config = SimpleNamespace(num_blocks=7)
    runner._init_kv_zero_meta()
    assert zeroer.call_args.kwargs["num_blocks"] == 7
    assert "cache_dtype" not in zeroer.call_args.kwargs


def test_mm_dummy_batch_uses_processor_owned_profile_inputs():
    items = [("video", object()), ("video", object())]
    batch = {"pixel_values_videos": torch.zeros((2, 3, 4))}
    provider = Mock(return_value=items)

    def group(inputs, *, device, pin_memory):
        assert inputs is items
        assert device == torch.device("cpu") and pin_memory is False
        yield "video", 2, batch

    runner = _runner_class(
        {"_get_mm_dummy_batch"},
        group_and_batch_mm_kwargs=group,
        PIN_MEMORY=False,
    )()
    runner.device = torch.device("cpu")
    runner.mm_budget = SimpleNamespace(get_dummy_encoder_profile_inputs=provider)
    # No legacy get_dummy_mm_inputs method exists on the registry.
    runner.mm_registry = object()
    assert runner._get_mm_dummy_batch("video", 2) is batch
    provider.assert_called_once_with("video", 2)


def test_default_mm_profile_still_executes_encoder_and_lm():
    batch = {"pixel_values": torch.zeros((2, 3, 4))}
    encoded = [torch.zeros((3, 4)), torch.zeros((3, 4))]
    sanity_check = Mock()
    runner = _runner_class(
        {"profile_run"},
        logger=Mock(),
        sanity_check_mm_encoder_outputs=sanity_check,
        get_pp_group=lambda: SimpleNamespace(is_last_rank=False),
        gc=SimpleNamespace(collect=Mock()),
    )()
    runner.supports_mm_inputs = True
    runner.model_config = SimpleNamespace(
        multimodal_config=SimpleNamespace(skip_mm_profiling=False)
    )
    runner.mm_budget = SimpleNamespace(
        get_encoder_budget=lambda: 6,
        mm_max_toks_per_item={"image": 3},
        get_modality_with_max_tokens=lambda: "image",
        mm_max_items_per_batch={"image": 2},
    )
    runner._get_mm_dummy_batch = Mock(return_value=batch)
    runner.model = SimpleNamespace(embed_multimodal=Mock(return_value=encoded))
    runner.encoder_cache = {}
    runner.max_num_tokens = 4

    def dummy_run(tokens, *, is_profile):
        assert tokens == 4 and is_profile
        assert list(runner.encoder_cache.values()) == encoded
        return torch.zeros((4, 4)), torch.zeros((4, 4))

    runner._dummy_run = dummy_run
    runner._sync_device = Mock()
    runner.profile_run()
    runner._get_mm_dummy_batch.assert_called_once_with("image", 2)
    runner.model.embed_multimodal.assert_called_once_with(**batch)
    sanity_check.assert_called_once_with(encoded, expected_num_items=2)
    assert runner.encoder_cache == {}


def test_padding_mask_reuses_buffer_without_stale_padding():
    runner = _runner_class({"_prepare_padding_mask"}, torch=torch)()
    runner.is_padding = torch.zeros(8, dtype=torch.bool)
    mask = runner._prepare_padding_mask(3, 6)
    assert mask.tolist() == [False, False, False, True, True, True]
    assert mask.data_ptr() == runner.is_padding.data_ptr()
    # A later, larger real batch must clear padding left by the previous step.
    assert runner._prepare_padding_mask(6, 8).tolist() == [False] * 6 + [True] * 2
    # Dummy warmup skips all experts; memory profiling exercises all experts.
    assert runner._prepare_padding_mask(0, 4).all()
    assert not runner._prepare_padding_mask(4, 4).any()


def test_zero_draft_spec_batch_keeps_decode_state_offset_and_masks_prefill():
    class Buffer:
        def __init__(self, size, dtype=torch.int32):
            self.cpu = torch.zeros(size, dtype=dtype)
            self.np = self.cpu.numpy()
            self.gpu = self.cpu.clone()

        def copy_to_gpu(self, count=None):
            self.gpu[:count].copy_(self.cpu[:count])

    runner = _runner_class(
        {
            "_prepare_inputs",
            "_get_cumsum_and_arange",
            "_compute_prev_positions",
            "_kl_guard_on",
        },
        torch=torch,
        np=np,
        current_platform=SimpleNamespace(vendor_name="metax"),
    )()
    token_ids = torch.arange(24, dtype=torch.int32).view(3, 8)
    computed = torch.tensor([5, 2, 0, 0], dtype=torch.int32)
    req_ids = ["decode", "partial_prefill", "new_prefill"]
    runner.input_batch = SimpleNamespace(
        num_reqs=3,
        req_ids=req_ids,
        req_id_to_index={req: i for i, req in enumerate(req_ids)},
        prev_req_id_to_index=None,
        token_ids_cpu=token_ids.numpy(),
        token_ids_cpu_tensor=token_ids,
        num_computed_tokens_cpu=computed.numpy(),
        num_computed_tokens_cpu_tensor=computed,
        num_prompt_tokens=np.array([5, 5, 2, 0]),
        req_prompt_embeds={},
        block_table=SimpleNamespace(
            commit_block_table=Mock(), compute_slot_mapping=Mock()
        ),
    )
    runner.requests = {
        req: SimpleNamespace(num_tokens=length)
        for req, length in zip(req_ids, [6, 5, 2])
    }
    runner.arange_np = np.arange(8)
    runner.query_pos = Buffer(8, torch.int64)
    runner.input_ids = Buffer(8)
    runner.query_start_loc = Buffer(5)
    runner.optimistic_seq_lens_cpu = torch.zeros(4, dtype=torch.int32)
    runner.prev_positions = Buffer(4, torch.int64)
    runner.discard_request_mask = Buffer(4, torch.bool)
    runner.num_accepted_tokens = Buffer(4)
    runner.num_accepted_tokens_event = None
    runner.num_computed_tokens = torch.zeros(4, dtype=torch.int32)
    runner.req_indices = Buffer(8, torch.int64)
    runner.num_scheduled_tokens = Buffer(4)
    runner.positions = torch.zeros(8, dtype=torch.int64)
    runner.seq_lens = torch.zeros(4, dtype=torch.int32)
    runner.num_decode_draft_tokens = Buffer(4)
    runner.uses_mrope = False
    runner.uses_xdrope_dim = 0
    runner.enable_prompt_embeds = False
    runner.use_async_spec_decode = False
    runner.mamba_prev_last_scheduled_idx = None
    runner.speculative_config = object()
    runner.lora_config = None
    runner._prepare_input_ids = Mock()
    metadata = SimpleNamespace(logits_indices=torch.tensor([0, 1, 3]))
    runner._calc_spec_decode_metadata = Mock(return_value=metadata)
    scheduler = SimpleNamespace(
        total_num_scheduled_tokens=4, scheduled_spec_decode_tokens={}
    )
    logits_indices, spec_metadata, max_sampled = runner._prepare_inputs(
        scheduler, np.array([1, 1, 2], dtype=np.int32)
    )
    assert spec_metadata is metadata and max_sampled == 1
    assert logits_indices.tolist() == [0, 1, 3]
    assert runner.num_decode_draft_tokens.gpu.tolist() == [0, -1, -1, -1]
    assert runner.discard_request_mask.gpu.tolist() == [False, True, False, False]
    assert runner.seq_lens.tolist() == [6, 3, 2, 0]
    assert runner.input_ids.cpu[:4].tolist() == [5, 10, 16, 17]
    num_drafts, cumulative = runner._calc_spec_decode_metadata.call_args.args
    assert num_drafts.tolist() == [0, 0, 0]
    assert cumulative.tolist() == [1, 2, 4]


def test_prompt_logprobs_accept_five_field_tensor_result():
    class Logprobs(NamedTuple):
        logprob_token_ids: torch.Tensor
        logprobs: torch.Tensor
        selected_token_ranks: torch.Tensor
        cu_num_generated_tokens: object = None
        cu_num_generated_tokens_tensor: object = None

        @staticmethod
        def empty_cpu(rows, columns):
            return Logprobs(
                torch.empty((rows, columns), dtype=torch.int32),
                torch.empty((rows, columns)),
                torch.empty(rows, dtype=torch.int32),
            )

    result = Logprobs(
        torch.tensor([[2, 1], [3, 0]], dtype=torch.int32),
        torch.tensor([[-0.2, -1.0], [-0.3, -1.5]]),
        torch.tensor([1, 2], dtype=torch.int32),
    )
    runner = _runner_class(
        {"_get_prompt_logprobs_dict"},
        torch=torch,
        LogprobsTensors=Logprobs,
        async_tensor_h2d=lambda ids, **kwargs: torch.tensor(ids),
    )()
    runner.device = torch.device("cpu")
    runner.num_prompt_logprobs = {"r": 1}
    runner.requests = {
        "r": SimpleNamespace(
            prompt_token_ids=[1, 2, 3],
            num_computed_tokens=0,
            in_progress_prompt_logprobs_cpu=None,
        )
    }
    runner.model_config = SimpleNamespace(logprobs_mode="raw_logits")
    runner.input_batch = SimpleNamespace(req_id_to_index={"r": 0})
    runner.query_start_loc = SimpleNamespace(np=np.array([0, 3]))
    runner.model = SimpleNamespace(compute_logits=lambda hidden: hidden)
    runner.sampler = SimpleNamespace(gather_logprobs=Mock(return_value=result))
    runner._sync_device = Mock()
    output = runner._get_prompt_logprobs_dict(torch.zeros((3, 4)), {"r": 3})["r"]
    assert torch.equal(output.logprob_token_ids, result.logprob_token_ids)
    assert torch.equal(output.logprobs, result.logprobs)
    assert torch.equal(output.selected_token_ranks, result.selected_token_ranks)
    assert not runner.num_prompt_logprobs
    assert runner.requests["r"].in_progress_prompt_logprobs_cpu is None
    runner._sync_device.assert_called_once()


def test_metadata_uses_kernel_tables_but_preserves_manager_physical_slots():
    class UnusedSpec:
        pass

    class Builder:
        supports_update_block_table = True

        def __init__(self):
            self.update_block_table = Mock(
                side_effect=lambda previous, table, slots: SimpleNamespace(
                    block_table=table.clone(), slot_mapping=slots
                )
            )

    class Group:
        def __init__(self, layer):
            self.layer_names = [layer]
            self.builder = Builder()
            self.map_to_kernel_block_table = Mock(
                side_effect=lambda table, ubatch: (
                    table.unsqueeze(-1) * 33 + torch.arange(33)
                ).flatten(1)
            )
            self.build_metadata = Mock(side_effect=self._build)
            self.build_metadata_for_cudagraph_capture = Mock(side_effect=self._build)

        def get_metadata_builder(self, ubatch=0):
            return self.builder

        def _build(self, common, ubatch, **kwargs):
            return SimpleNamespace(
                block_table=self.map_to_kernel_block_table(
                    common.block_table_tensor, ubatch
                ),
                slot_mapping=common.slot_mapping,
            )

    tables = [torch.tensor([[2, 7]]), torch.tensor([[5, 9]])]
    positions = torch.arange(525, 530, dtype=torch.int64)
    physical_slots = [
        table[0, positions // 528] * 528 + positions % 528 for table in tables
    ]
    block_tables = [
        SimpleNamespace(
            get_device_tensor=lambda count, table=table: table[:count],
            slot_mapping=SimpleNamespace(gpu=slots),
        )
        for table, slots in zip(tables, physical_slots)
    ]
    groups = [Group("layer.0"), Group("layer.1")]
    spec = object()  # Same spec and builder type exercise cached metadata updates.
    runner = _runner_class(
        {"_get_slot_mappings", "_build_attention_metadata"},
        torch=torch,
        copy=copy,
        EncoderOnlyAttentionSpec=UnusedSpec,
        UniformTypeKVCacheSpecs=UnusedSpec,
        CommonAttentionMetadata=SimpleNamespace,
        NULL_BLOCK_ID=0,
    )()
    runner.device = torch.device("cpu")
    runner.kv_cache_config = SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(kv_cache_spec=spec, layer_names=group.layer_names)
            for group in groups
        ]
    )
    runner.attn_groups = [[group] for group in groups]
    runner.input_batch = SimpleNamespace(
        block_table=block_tables,
        num_computed_tokens_cpu_tensor=torch.tensor([525]),
        num_prompt_tokens_cpu_tensor=torch.tensor([530]),
    )
    runner.routed_experts_initialized = False
    runner.optimistic_seq_lens_cpu = torch.tensor([530])
    runner.seq_lens = torch.tensor([530])
    runner.query_start_loc = SimpleNamespace(
        gpu=torch.tensor([0, 5]), cpu=torch.tensor([0, 5])
    )
    runner.positions = positions
    runner.use_async_spec_decode = False
    runner.is_mm_prefix_lm = False
    runner.model_config = SimpleNamespace(rswa_window=None)
    runner.cache_config = SimpleNamespace(
        use_replayssm=False, kv_sharing_fast_prefill=False
    )
    runner.dcp_world_size = 1
    runner.speculative_config = None
    runner._get_encoder_seq_lens = lambda *args, **kwargs: (None, None)
    runner.max_model_len = 1024

    slots_by_group, slots_by_layer = runner._get_slot_mappings(5, 1, 5)
    for capturing in (False, True):
        metadata, draft_metadata = runner._build_attention_metadata(
            5, 1, 5, slot_mappings=slots_by_group, for_cudagraph_capture=capturing
        )
        assert draft_metadata is None
        for gid, group in enumerate(groups):
            layer = group.layer_names[0]
            table = metadata[layer].block_table
            assert table.shape == (1, 66)
            expected = (tables[gid].unsqueeze(-1) * 33 + torch.arange(33)).flatten(1)
            assert torch.equal(table, expected)
            # Kernel block addressing must retrieve exactly the slots written
            # through the manager table, including the 528-token boundary.
            read_slots = table[0, positions // 16] * 16 + positions % 16
            assert torch.equal(read_slots, slots_by_layer[layer])
            assert metadata[layer].slot_mapping is slots_by_group[gid]
    groups[0].build_metadata.assert_called_once()
    groups[1].build_metadata.assert_not_called()
    groups[1].builder.update_block_table.assert_called_once()
    for group in groups:
        group.build_metadata_for_cudagraph_capture.assert_called_once()
