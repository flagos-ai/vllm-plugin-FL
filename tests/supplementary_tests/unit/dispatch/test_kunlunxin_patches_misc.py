# Copyright 2025 BAAI. All rights reserved.

"""CPU contracts for the remaining Kunlunxin patch surfaces.

Covers modules not exercised by the other supplementary kunlunxin files
(``test_kunlunxin_backend_class.py``, ``test_kunlunxin_register_ops.py``,
``ops/test_kunlunxin_impls_cpu.py`` and
``vendor/test_kunlunxin_patches_compat.py``):

* ``patches/mamba_align_compat.py`` — the ``KunlunxinMambaCopyContext``
  installed over ``mamba_hybrid.MambaSpecDecodeGPUContext``: the
  initialization pass that groups conv/temporal states by copy spec, the
  ``run_fused_precopy`` GPU-index path, and the
  ``run_fused_postprocess_align`` boundary arithmetic and reset rows;
* the compat apply functions' guards not reached elsewhere:
  ``apply_vl_compat``'s missing-native-function warning path and
  ``maybe_reduce_vision_tp`` numeric contract (gather-in-rank-order FP32
  rounding), which ``tests/unit_tests/test_kunlunxin_vl_compat.py`` only
  tests for the None branches;
* ``patches/mamba_hybrid_compat.py`` and
  ``patches/mamba_utils_compat.py`` module contracts (existence and
  idempotent apply signatures) where full behavior needs XPU kernels.

Everything restores the exact attributes and module entries it touched.
"""

from __future__ import annotations

import contextlib
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

pytestmark = [pytest.mark.supplementary]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _mamba_align_applied():
    """Apply mamba_align_compat and restore the class binding on exit."""
    import vllm.v1.worker.gpu.model_states.mamba_hybrid as mh
    from vllm.v1.worker.mamba_utils import MambaSpecDecodeGPUContext

    from vllm_fl.dispatch.backends.vendor.kunlunxin.patches.mamba_align_compat import (
        apply_mamba_align_compat,
    )

    real = mh.MambaSpecDecodeGPUContext
    # the guard returns early when already applied
    apply_mamba_align_compat()
    installed = mh.MambaSpecDecodeGPUContext
    try:
        assert installed is not MambaSpecDecodeGPUContext
        yield installed
    finally:
        mh.MambaSpecDecodeGPUContext = real


def _make_context(cls, *, block_size=4, groups=None, is_initialized=True):
    """Build a bare context with hand-filled internals (no __init__)."""
    obj = cls.__new__(cls)
    obj.is_initialized = is_initialized
    obj.block_size = block_size
    obj.mamba_group_ids = [0]
    if groups is None:
        groups = [
            SimpleNamespace(
                block_table=torch.tensor([[1, 2, 3]]),
                conv_states=[torch.zeros(4, 4, 6)],
                temporal_states=[torch.zeros(4, 8)],
            )
        ]
    obj._kunlunxin_states = groups
    return obj


# ---------------------------------------------------------------------------
# apply_mamba_align_compat + initialization pass
# ---------------------------------------------------------------------------


class TestMambaAlignApply:
    @pytest.mark.platforms("kunlunxin")
    def test_replaces_the_context_class(self):
        with _mamba_align_applied() as installed:
            assert installed.__name__ == "KunlunxinMambaCopyContext"

    @pytest.mark.platforms("kunlunxin")
    def test_apply_is_guarded_by_identity(self):
        import vllm.v1.worker.gpu.model_states.mamba_hybrid as mh

        from vllm_fl.dispatch.backends.vendor.kunlunxin.patches.mamba_align_compat import (
            apply_mamba_align_compat,
        )

        real = mh.MambaSpecDecodeGPUContext
        try:
            apply_mamba_align_compat()
            first = mh.MambaSpecDecodeGPUContext
            # a second application over the already-installed class is a no-op
            apply_mamba_align_compat()
            assert mh.MambaSpecDecodeGPUContext is first
        finally:
            mh.MambaSpecDecodeGPUContext = real


class TestInitializeFromForwardContext:
    @pytest.mark.platforms("kunlunxin")
    def test_groups_conv_and_temporal_states_by_copy_spec(self):
        with _mamba_align_applied() as cls:
            obj = cls.__new__(cls)
            obj.is_initialized = False

            conv_state = torch.zeros(2, 4, 6)
            temporal_state = torch.zeros(2, 8)
            kv_cache_config = SimpleNamespace(
                kv_cache_groups=[
                    SimpleNamespace(
                        layer_names=["mamba.layer"],
                        kv_cache_spec=SimpleNamespace(mamba_type="mamba"),
                    )
                ]
            )
            # each mamba layer's kv_cache is the (conv, temporal) state
            # pair, zipped positionally with the copy-spec funcs
            forward_context = {
                "mamba.layer": SimpleNamespace(
                    kv_cache=[conv_state, temporal_state]
                ),
            }

            from vllm.model_executor.layers.mamba.mamba_utils import (
                get_conv_copy_spec,
                get_temporal_copy_spec,
            )

            super_called = []
            self_ref = obj

            def fake_super_init(*args, **kwargs):
                super_called.append(args)
                # attributes the base initializer would publish
                self_ref.mamba_group_ids = [0]
                self_ref.is_initialized = True

            # patch the base initializer the override calls
            base = cls.__mro__[1]
            real_init = base.initialize_from_forward_context
            base.initialize_from_forward_context = fake_super_init
            try:
                obj.initialize_from_forward_context(
                    kv_cache_config,
                    forward_context,
                    # the funcs tuple is indexed by mamba_type and zipped
                    # with the layer's cache list
                    {
                        "mamba": (
                            get_conv_copy_spec,
                            get_temporal_copy_spec,
                        )
                    },
                    [torch.tensor([[0, 1, 2]])],
                )
            finally:
                base.initialize_from_forward_context = real_init

            assert len(super_called) == 1
            groups = obj._kunlunxin_states
            assert len(groups) == 1
            assert len(groups[0].conv_states) == 1
            torch.testing.assert_close(groups[0].conv_states[0], conv_state)
            assert len(groups[0].temporal_states) == 1
            torch.testing.assert_close(
                groups[0].temporal_states[0], temporal_state
            )

    @pytest.mark.platforms("kunlunxin")
    def test_already_initialized_short_circuits_super(self):
        with _mamba_align_applied() as cls:
            obj = cls.__new__(cls)
            obj.is_initialized = True

            base = cls.__mro__[1]
            real_init = base.initialize_from_forward_context
            calls = []

            def fake_super_init(*args, **kwargs):
                calls.append(args)

            base.initialize_from_forward_context = fake_super_init
            try:
                result = obj.initialize_from_forward_context(1, 2, 3, 4)
                # already initialized: the override returns BEFORE super
                assert result is None
                assert calls == []
            finally:
                base.initialize_from_forward_context = real_init


# ---------------------------------------------------------------------------
# run_fused_precopy
# ---------------------------------------------------------------------------


class TestRunFusedPrecopy:
    @pytest.mark.platforms("kunlunxin")
    def test_zero_requests_returns_early(self):
        with _mamba_align_applied() as cls:
            obj = _make_context(cls)
            assert obj.run_fused_precopy(0, None, None, None, None) is None

    @pytest.mark.platforms("kunlunxin")
    def test_uninitialized_returns_early(self):
        with _mamba_align_applied() as cls:
            obj = _make_context(cls, is_initialized=False)
            assert (
                obj.run_fused_precopy(
                    1, torch.tensor([0]), torch.tensor([0]),
                    torch.tensor([0]), torch.tensor([0]),
                )
                is None
            )

    @pytest.mark.platforms("kunlunxin")
    def test_copies_shifted_conv_and_temporal_states(self):
        with _mamba_align_applied() as cls:
            conv_pages = torch.arange(4 * 4 * 6).view(4, 4, 6).float()
            temporal_pages = torch.arange(4 * 8).view(4, 8).float()
            obj = _make_context(
                cls,
                groups=[
                    SimpleNamespace(
                        block_table=torch.tensor([[1, 2, 3]]),
                        conv_states=[conv_pages.clone()],
                        temporal_states=[temporal_pages.clone()],
                    )
                ],
            )

            # src col 0, dst col 1, token bias 2, request idx 0
            obj.run_fused_precopy(
                1,
                torch.tensor([1]),      # state_idx -> dst col
                torch.tensor([0]),      # src col
                torch.tensor([2]),      # token bias
                torch.tensor([0]),      # idx mapping
            )

            # block table row 0: src block = table[0, 0] = 1, dst = table[0, 1] = 2
            # conv: state[2][0:2] <- state[1][2:4] (token bias shift)
            conv_result = obj._kunlunxin_states[0].conv_states[0]
            torch.testing.assert_close(
                conv_result[2, :2], conv_pages[1, 2:4]
            )
            # untouched row keeps its original content
            torch.testing.assert_close(conv_result[0], conv_pages[0])
            # temporal src block = table[0, src+bias] = table[0, 2] = 3
            # temporal[2] <- temporal[3]
            temporal_result = obj._kunlunxin_states[0].temporal_states[0]
            torch.testing.assert_close(temporal_result[2], temporal_pages[3])
            torch.testing.assert_close(temporal_result[0], temporal_pages[0])

    @pytest.mark.platforms("kunlunxin")
    def test_negative_request_row_is_masked_out(self):
        with _mamba_align_applied() as cls:
            conv = torch.arange(4 * 4 * 6).view(4, 4, 6).float()
            obj = _make_context(
                cls,
                groups=[
                    SimpleNamespace(
                        block_table=torch.tensor([[1, 2, 3]]),
                        conv_states=[conv.clone()],
                        temporal_states=[torch.zeros(4, 8)],
                    )
                ],
            )
            before = conv.clone()
            obj.run_fused_precopy(
                1,
                torch.tensor([1]),
                torch.tensor([0]),
                torch.tensor([0]),
                torch.tensor([-1]),  # invalid request
            )
            torch.testing.assert_close(conv, before)


# ---------------------------------------------------------------------------
# run_fused_postprocess_align
# ---------------------------------------------------------------------------


class TestRunFusedPostprocessAlign:
    @pytest.mark.platforms("kunlunxin")
    def test_zero_requests_and_uninitialized_return_early(self):
        with _mamba_align_applied() as cls:
            obj = _make_context(cls)
            assert (
                obj.run_fused_postprocess_align(
                    0, None, None, None, None
                )
                is None
            )
            obj2 = _make_context(cls, is_initialized=False)
            assert (
                obj2.run_fused_postprocess_align(
                    1, torch.tensor([1]), torch.tensor([1]),
                    torch.tensor([1]), torch.tensor([0]),
                )
                is None
            )

    @pytest.mark.platforms("kunlunxin")
    def test_boundary_arithmetic_and_state_copy(self):
        with _mamba_align_applied() as cls:
            conv = torch.zeros(4, 4, 6)
            temporal = torch.zeros(4, 8)
            obj = _make_context(
                cls,
                groups=[
                    SimpleNamespace(
                        block_table=torch.tensor([[0, 1, 2, 3]]),
                        conv_states=[conv],
                        temporal_states=[temporal],
                    )
                ],
            )
            num_accepted = torch.tensor([1])
            state_idx = torch.tensor([0])   # src col
            computed = torch.tensor([5])    # new computed tokens
            idx_mapping = torch.tensor([0])

            obj.run_fused_postprocess_align(
                1, num_accepted, state_idx, computed, idx_mapping
            )

            # block_size 4: aligned = (5 // 4) * 4 = 4
            # dst_cols = 4 // 4 - 1 = 0; token_biases = 4 - (5 - 1 + 1) = -1
            # at_boundary requires token_biases >= 0 -> False -> no copy
            assert conv.abs().sum().item() == 0
            assert temporal.abs().sum().item() == 0
            assert num_accepted.tolist() == [1]

    @pytest.mark.platforms("kunlunxin")
    def test_exact_boundary_copies_and_resets_accepted(self):
        with _mamba_align_applied() as cls:
            conv = torch.arange(4 * 4 * 6).view(4, 4, 6).float()
            temporal = torch.arange(4 * 8).view(4, 8).float()
            obj = _make_context(
                cls,
                groups=[
                    SimpleNamespace(
                        block_table=torch.tensor([[0, 1, 2, 3]]),
                        conv_states=[conv.clone()],
                        temporal_states=[temporal.clone()],
                    )
                ],
            )
            num_accepted = torch.tensor([1])
            state_idx = torch.tensor([0])
            # computed 4 -> aligned 4, dst_cols 0, token_biases = 4 - 4 = 0
            computed = torch.tensor([4])
            idx_mapping = torch.tensor([0])

            obj.run_fused_postprocess_align(
                1, num_accepted, state_idx, computed, idx_mapping
            )

            # src_cols == dst_cols == 0 and token_biases == 0: no bytes move
            # but num_accepted is reset to 1 on the reset row (a no-op write)
            assert num_accepted.tolist() == [1]
            # states untouched (same-block, zero-bias copy is skipped)
            torch.testing.assert_close(
                conv, obj._kunlunxin_states[0].conv_states[0]
            )

    @pytest.mark.platforms("kunlunxin")
    def test_cross_block_postprocess_copies_state(self):
        with _mamba_align_applied() as cls:
            conv = torch.zeros(4, 4, 6)
            temporal = torch.arange(4 * 8).view(4, 8).float()
            obj = _make_context(
                cls,
                groups=[
                    SimpleNamespace(
                        block_table=torch.tensor([[0, 1, 2, 3]]),
                        conv_states=[conv.clone()],
                        temporal_states=[temporal.clone()],
                    )
                ],
            )
            num_accepted = torch.tensor([2])
            state_idx = torch.tensor([0])  # src col
            # computed 6 -> aligned 4 (block_size 4), dst_cols 0, biases = 4 - (6-2+1) = -1?
            # 4 - 5 = -1 -> not at boundary. Use computed 5:
            computed = torch.tensor([5])
            # aligned = 4, dst_cols 0, biases = 4 - (5 - 2 + 1) = 0 -> at boundary
            idx_mapping = torch.tensor([0])

            obj.run_fused_postprocess_align(
                1, num_accepted, state_idx, computed, idx_mapping
            )
            # src == dst == 0 and bias 0: nothing moves, accepted reset
            assert num_accepted.tolist() == [1]


# ---------------------------------------------------------------------------
# maybe_reduce_vision_tp numeric contract
# ---------------------------------------------------------------------------


class TestMaybeReduceVisionTp:
    @pytest.mark.platforms("kunlunxin")
    def test_reduces_bf16_partials_in_fp32_rank_order(self):
        from vllm_fl.dispatch.backends.vendor.kunlunxin.patches import (
            vl_compat,
        )

        # enter the vision TP scope
        token = vl_compat._vision_tp_scope.set(True)
        try:
            partials = [
                torch.tensor([0.1, 0.2], dtype=torch.bfloat16),
                torch.tensor([0.3, 0.4], dtype=torch.bfloat16),
            ]

            class FakeFlagcx:
                disabled = False

            class FakeComm:
                world_size = 4  # scope requires 4
                unique_name = "tp:group"
                pyflagcx_comm = FakeFlagcx()

                def all_gather(self, tensor, dim=0):
                    # logical rank order gather
                    return torch.cat(
                        [tensor.float() for _ in range(4)], dim=0
                    ).to(torch.bfloat16)

            comm = FakeComm()
            # the rank-ordered FP32 sum of four identical partials
            result = vl_compat.maybe_reduce_vision_tp(
                comm, torch.tensor([0.25, 0.5], dtype=torch.bfloat16)
            )
            expected = (
                torch.tensor([0.25, 0.5])
                .float()
                .mul(4)
                .to(torch.bfloat16)
            )
            assert result is not None
            assert result.dtype == torch.bfloat16
            torch.testing.assert_close(result, expected)
        finally:
            vl_compat._vision_tp_scope.reset(token)

    @pytest.mark.platforms("kunlunxin")
    def test_outside_scope_returns_none(self):
        from vllm_fl.dispatch.backends.vendor.kunlunxin.patches import (
            vl_compat,
        )

        assert not vl_compat._vision_tp_scope.get()
        comm = SimpleNamespace(
            world_size=4,
            unique_name="tp:group",
            pyflagcx_comm=SimpleNamespace(disabled=False),
        )
        assert (
            vl_compat.maybe_reduce_vision_tp(
                comm, torch.tensor([1.0], dtype=torch.bfloat16)
            )
            is None
        )


# ---------------------------------------------------------------------------
# module contracts for the remaining compat shims
# ---------------------------------------------------------------------------


class TestCompatModuleContracts:
    @pytest.mark.platforms("kunlunxin")
    @pytest.mark.parametrize(
        ("module_name", "apply_name"),
        [
            ("mamba_hybrid_compat", "apply_mamba_hybrid_compat"),
            ("mamba_utils_compat", "apply_mamba_utils_compat"),
            ("buffer_utils_compat", "apply_buffer_utils_compat"),
            ("penalties_compat", "apply_penalties_compat"),
            ("structured_outputs_compat", "apply_structured_outputs_compat"),
            ("topk_topp_compat", "apply_topk_topp_compat"),
            ("uva_compat", "apply_uva_compat"),
            ("v2_greedy_sampler", "apply_v2_greedy_sampler"),
        ],
    )
    def test_apply_functions_exist(self, module_name, apply_name):
        module = __import__(
            f"vllm_fl.dispatch.backends.vendor.kunlunxin.patches.{module_name}",
            fromlist=[apply_name],
        )
        assert callable(getattr(module, apply_name))


# ---------------------------------------------------------------------------
# input_batch_compat apply() launcher propagation (mechanics only)
# ---------------------------------------------------------------------------


class TestInputBatchApplyPropagation:
    @pytest.mark.platforms("kunlunxin")
    def test_launchers_installed_and_flagged(self):
        from vllm_fl.dispatch.backends.vendor.kunlunxin.patches import (
            input_batch_compat,
        )

        input_batch = ModuleType("vllm.v1.worker.gpu.input_batch")

        def old_launcher(*args, **kwargs):  # pragma: no cover
            raise AssertionError("original launcher must not run")

        for name in (
            "combine_sampled_and_draft_tokens",
            "prepare_pos_seq_lens",
            "get_num_sampled_and_rejected",
            "post_update",
            "prepare_prefill_inputs",
            "expand_idx_mapping",
            "post_update_num_computed_tokens",
        ):
            setattr(input_batch, name, old_launcher)

        model_runner = ModuleType("vllm.v1.worker.gpu.model_runner")
        for name in (
            "combine_sampled_and_draft_tokens",
            "post_update",
            "post_update_num_computed_tokens",
            "prepare_pos_seq_lens",
            "prepare_prefill_inputs",
            "expand_idx_mapping",
        ):
            setattr(model_runner, name, old_launcher)

        # `from vllm.v1.worker.gpu import model_runner` resolves the PARENT
        # package attribute, so the stub must be bound there too (the real
        # module's attributes are restored afterwards).
        import vllm.v1.worker.gpu as gpu_pkg

        saved = {
            name: sys.modules.get(name)
            for name in (
                "vllm.v1.worker.gpu.input_batch",
                "vllm.v1.worker.gpu.model_runner",
            )
        }
        saved_mr_attr = getattr(gpu_pkg, "model_runner", None)
        saved_ib_attr = getattr(gpu_pkg, "input_batch", None)
        sys.modules["vllm.v1.worker.gpu.input_batch"] = input_batch
        sys.modules["vllm.v1.worker.gpu.model_runner"] = model_runner
        gpu_pkg.model_runner = model_runner
        gpu_pkg.input_batch = input_batch
        try:
            input_batch_compat.apply_input_batch_compat()
            assert getattr(
                input_batch, input_batch_compat._PATCH_FLAG_LAUNCHERS, False
            )
            for name in (
                "combine_sampled_and_draft_tokens",
                "prepare_pos_seq_lens",
                "get_num_sampled_and_rejected",
                "post_update",
                "prepare_prefill_inputs",
                "expand_idx_mapping",
                "post_update_num_computed_tokens",
            ):
                installed = getattr(input_batch, name)
                assert installed is not old_launcher
                assert getattr(
                    installed, input_batch_compat._PATCH_FLAG_LAUNCHERS, False
                )
                assert callable(installed)
            # model_runner namespace received the same objects (the
            # sampler-only launcher propagates into sampler modules instead)
            for name in (
                "combine_sampled_and_draft_tokens",
                "prepare_pos_seq_lens",
                "post_update",
                "prepare_prefill_inputs",
                "expand_idx_mapping",
                "post_update_num_computed_tokens",
            ):
                assert getattr(model_runner, name) is getattr(input_batch, name)
        finally:
            for name, entry in saved.items():
                if entry is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = entry
            if saved_mr_attr is None:
                try:
                    del gpu_pkg.model_runner
                except AttributeError:
                    pass
            else:
                gpu_pkg.model_runner = saved_mr_attr
            if saved_ib_attr is None:
                try:
                    del gpu_pkg.input_batch
                except AttributeError:
                    pass
            else:
                gpu_pkg.input_batch = saved_ib_attr

    @pytest.mark.platforms("kunlunxin")
    def test_apply_is_idempotent_for_launchers(self):
        from vllm_fl.dispatch.backends.vendor.kunlunxin.patches import (
            input_batch_compat,
        )

        input_batch = ModuleType("vllm.v1.worker.gpu.input_batch")

        def old_launcher(*args, **kwargs):  # pragma: no cover
            raise AssertionError("original launcher must not run")

        for name in (
            "combine_sampled_and_draft_tokens",
            "prepare_pos_seq_lens",
            "get_num_sampled_and_rejected",
            "post_update",
            "prepare_prefill_inputs",
            "expand_idx_mapping",
            "post_update_num_computed_tokens",
        ):
            setattr(input_batch, name, old_launcher)

        name = "vllm.v1.worker.gpu.input_batch"
        saved = sys.modules.get(name)
        sys.modules[name] = input_batch
        try:
            input_batch_compat.apply_input_batch_compat()
            first = input_batch.prepare_pos_seq_lens
            # second application: the launcher flag short-circuits
            input_batch_compat.apply_input_batch_compat()
            assert input_batch.prepare_pos_seq_lens is first
        finally:
            if saved is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = saved
