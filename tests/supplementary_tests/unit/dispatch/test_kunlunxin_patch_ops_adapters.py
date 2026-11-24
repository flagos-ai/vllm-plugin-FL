# Copyright 2025 BAAI. All rights reserved.

"""CPU contracts for the Kunlunxin operator-adapter patches in ``patch.py``.

Covers the non-graph patches of ``vllm_fl/.../kunlunxin/patch.py``:

* ``patch_block_table_slot_mapping`` — the numpy-free torch
  ``compute_slot_mapping`` replacement installed on the REAL
  ``BlockTable`` class, including CP>1 interleaved ownership masking;
* ``patch_attention_backend_registry`` — registering the Kunlunxin backend
  under ``AttentionBackendEnum.CUSTOM``;
* ``patch_sampler_rng`` — the TP-consistent broadcast wrapper around
  ``random_sample``;
* ``patch_causal_conv1d`` — NCW↔NWC adapters installed on both the causal
  conv1d module and the GDN linear-attn module, with the
  ``resolve_op``-resolved Kunlunxin kernel;
* ``patch_fla_ops`` — replacing chunk / fused-recurrent entry points with
  the Kunlunxin implementations;
* ``patch_fused_gdn_gating`` — the ``fused_post_conv_prep_kunlunxin``
  numeric oracle (split/reshape/l2norm/softplus gating, output_g_exp) and
  the ``fused_sigmoid_gating_delta_rule_update_kunlunxin`` gating math,
  KDA rejection, qk-l2norm pre-normalization, and ssm-state slicing;
* ``patch_ssm_cache_update`` — source rewrite of
  ``QwenGatedDeltaNetAttention._forward_core`` (the zero-mask guard and
  the ssm cache write call sites).

Every patched attribute or ``sys.modules`` entry is restored on exit.
"""

from __future__ import annotations

import contextlib
import sys
from types import ModuleType

import pytest
import torch
import torch.nn.functional as F

pytestmark = [pytest.mark.supplementary]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _xtorch_ops_stub(**ops):
    """Install an ``xtorch_ops`` stub module with the given callables."""
    stub = ModuleType("xtorch_ops")
    for name, fn in ops.items():
        setattr(stub, name, fn)
    saved = sys.modules.get("xtorch_ops")
    sys.modules["xtorch_ops"] = stub
    try:
        yield stub
    finally:
        if saved is None:
            sys.modules.pop("xtorch_ops", None)
        else:
            sys.modules["xtorch_ops"] = saved


@contextlib.contextmanager
def _fake_fla_modules():
    """Install stand-in ``vllm.model_executor.layers.fla.ops[.*]`` modules.

    The installed vLLM build has no ``fla`` subpackage, so ``patch_fla_ops``
    is driven against fakes whose parents expose them.
    """
    import vllm.model_executor.layers as layers_real

    fla = ModuleType("vllm.model_executor.layers.fla")
    ops = ModuleType("vllm.model_executor.layers.fla.ops")
    chunk = ModuleType("vllm.model_executor.layers.fla.ops.chunk")
    rec = ModuleType("vllm.model_executor.layers.fla.ops.fused_recurrent")
    for m in (fla, ops, chunk, rec):
        m.__path__ = []
    names = [
        "vllm.model_executor.layers.fla",
        "vllm.model_executor.layers.fla.ops",
        "vllm.model_executor.layers.fla.ops.chunk",
        "vllm.model_executor.layers.fla.ops.fused_recurrent",
    ]
    saved = {name: sys.modules.get(name) for name in names}
    saved_attr = getattr(layers_real, "fla", None)
    has_attr = hasattr(layers_real, "fla")

    def old_chunk(*a, **k):
        return "OLD_CHUNK"

    def old_rec(*a, **k):
        return "OLD_REC"

    chunk.chunk_gated_delta_rule = old_chunk
    rec.fused_recurrent_gated_delta_rule = old_rec
    ops.chunk = chunk
    ops.fused_recurrent = rec
    ops.chunk_gated_delta_rule = old_chunk
    ops.fused_recurrent_gated_delta_rule = old_rec
    fla.ops = ops

    sys.modules.update(
        {
            names[0]: fla,
            names[1]: ops,
            names[2]: chunk,
            names[3]: rec,
        }
    )
    layers_real.fla = fla
    try:
        yield ops, chunk, rec
    finally:
        for name, entry in saved.items():
            if entry is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = entry
        if has_attr:
            layers_real.fla = saved_attr
        else:
            del layers_real.fla


@contextlib.contextmanager
def _fake_fused_recurrent_impl():
    """Install a recording fake for the Kunlunxin fused_recurrent impl.

    ``patch_fused_gdn_gating`` imports
    ``...impl.fla.fused_recurrent.fused_recurrent_gated_delta_rule`` while
    applying, so the fake must be in ``sys.modules`` under that path first.
    """
    calls = {}

    def fake_fwd(**kwargs):
        calls.clear()
        calls.update(kwargs)
        return "O", "HT"

    fake_mod = ModuleType(
        "vllm_fl.dispatch.backends.vendor.kunlunxin.impl.fla.fused_recurrent"
    )
    fake_mod.fused_recurrent_gated_delta_rule = fake_fwd
    name = (
        "vllm_fl.dispatch.backends.vendor.kunlunxin.impl.fla.fused_recurrent"
    )
    saved = sys.modules.get(name)
    sys.modules[name] = fake_mod
    try:
        yield calls
    finally:
        if saved is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = saved


@contextlib.contextmanager
def _restore_attr(owner, name):
    """Restore ``owner.name`` (deleting it if it was absent) on exit."""
    existed = hasattr(owner, name)
    saved = getattr(owner, name, None)
    try:
        yield
    finally:
        if existed:
            setattr(owner, name, saved)
        else:
            try:
                delattr(owner, name)
            except AttributeError:
                pass


def _apply_fused_gdn_gating():
    """Apply patch_fused_gdn_gating with xtorch_ops stubbed; returns the
    installed (prep, update) functions."""
    from vllm_fl.dispatch.backends.vendor.kunlunxin.patch import (
        patch_fused_gdn_gating,
    )

    def fake_kernel(*a, **k):
        return None

    import vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn as gdn

    real_prep = gdn.fused_post_conv_prep
    real_update = gdn.fused_sigmoid_gating_delta_rule_update
    real_gating = getattr(gdn, "fused_gdn_gating", None)
    with _xtorch_ops_stub(
        fused_gdn_gating=fake_kernel,
        fused_recurrent_gated_delta_rule_fwdv2=fake_kernel,
    ):
        patch_fused_gdn_gating()
    try:
        yield gdn.fused_post_conv_prep, gdn.fused_sigmoid_gating_delta_rule_update
    finally:
        gdn.fused_post_conv_prep = real_prep
        gdn.fused_sigmoid_gating_delta_rule_update = real_update
        if real_gating is None:
            if hasattr(gdn, "fused_gdn_gating"):
                del gdn.fused_gdn_gating
        else:
            gdn.fused_gdn_gating = real_gating


# ---------------------------------------------------------------------------
# patch_block_table_slot_mapping
# ---------------------------------------------------------------------------


class TestBlockTableSlotMapping:
    @pytest.mark.platforms("kunlunxin")
    def test_installed_on_real_block_table_class(self):
        from vllm.v1.worker.block_table import BlockTable

        from vllm_fl.dispatch.backends.vendor.kunlunxin.patch import (
            patch_block_table_slot_mapping,
        )

        real = BlockTable.compute_slot_mapping
        try:
            patch_block_table_slot_mapping()
            installed = BlockTable.compute_slot_mapping
            assert installed is not real
            assert installed.__name__ == "compute_slot_mapping_xpu"
        finally:
            BlockTable.compute_slot_mapping = real

    @pytest.mark.platforms("kunlunxin")
    def test_slot_values_match_hand_computed_reference(self):
        import numpy as np

        from vllm.v1.worker.block_table import BlockTable

        from vllm_fl.dispatch.backends.vendor.kunlunxin.patch import (
            patch_block_table_slot_mapping,
        )

        real = BlockTable.compute_slot_mapping
        try:
            patch_block_table_slot_mapping()
            block_size = 4
            table = BlockTable(
                block_size=block_size,
                max_num_reqs=4,
                max_num_blocks_per_req=8,
                max_num_batched_tokens=32,
                pin_memory=False,
                device=torch.device("cpu"),
                cp_kv_cache_interleave_size=1,
            )
            table.block_table.gpu.copy_(torch.arange(4 * 8).view(4, 8))

            # request 0: two tokens at seq positions 10, 11
            # request 1: three tokens at seq positions 20, 21, 22
            qsl = torch.tensor([0, 2, 5], dtype=torch.int32)
            positions = torch.tensor([10, 11, 20, 21, 22], dtype=torch.int64)
            table.compute_slot_mapping(2, qsl, positions)

            slot = table.slot_mapping.gpu
            # req0 row0: block numbers are table[0, 10//4=2]=2 and 2 (pos11)
            # slots: 2*4+10%4=10, 2*4+11%4=11
            assert slot[0].item() == 10
            assert slot[1].item() == 11
            # req1 row1: table[1, 20//4=5]=13 -> 13*4+0..2 = 52,53,54
            assert slot[2].item() == 52
            assert slot[3].item() == 53
            assert slot[4].item() == 54
            # the rest is padded with -1
            assert (slot[5:] == -1).all().item()
        finally:
            BlockTable.compute_slot_mapping = real

    @pytest.mark.platforms("kunlunxin")
    def test_none_slot_mapping_mode_returns_early(self):
        from vllm.v1.worker.block_table import BlockTable, SlotMappingMode

        from vllm_fl.dispatch.backends.vendor.kunlunxin.patch import (
            patch_block_table_slot_mapping,
        )

        real = BlockTable.compute_slot_mapping
        try:
            patch_block_table_slot_mapping()
            table = BlockTable(
                block_size=4,
                max_num_reqs=2,
                max_num_blocks_per_req=4,
                max_num_batched_tokens=8,
                pin_memory=False,
                device=torch.device("cpu"),
                cp_kv_cache_interleave_size=1,
                slot_mapping_mode=SlotMappingMode.NONE,
            )
            table.block_table.gpu.copy_(torch.zeros(2, 4))
            qsl = torch.tensor([0, 2], dtype=torch.int32)
            positions = torch.tensor([0, 1], dtype=torch.int64)
            table.compute_slot_mapping(1, qsl, positions)
            # NONE mode never writes the slot mapping
            assert (table.slot_mapping.gpu == 0).all().item()
        finally:
            BlockTable.compute_slot_mapping = real

    @pytest.mark.platforms("kunlunxin")
    def test_cp_world_size_masks_foreign_ranks(self):
        """DCP>1 masks slots not owned by this rank with PAD_SLOT_ID."""
        from vllm.v1.worker.block_table import BlockTable

        from vllm_fl.dispatch.backends.vendor.kunlunxin.patch import (
            patch_block_table_slot_mapping,
        )

        real = BlockTable.compute_slot_mapping
        try:
            patch_block_table_slot_mapping()
            block_size = 4
            table = BlockTable(
                block_size=block_size,
                max_num_reqs=2,
                max_num_blocks_per_req=4,
                max_num_batched_tokens=16,
                pin_memory=False,
                device=torch.device("cpu"),
                cp_kv_cache_interleave_size=1,
            )
            # simulate DCP world 2 / rank 0 (init left pcp/dcp at defaults 1)
            table.dcp_world_size = 2
            table.dcp_rank = 0
            table.block_table.gpu.copy_(torch.tensor([[3, 7, 0, 0]]))

            # virtual block = block_size * dcp_world_size = 8
            # positions 0..7 live in virtual block 0 -> physical block 3
            qsl = torch.tensor([0, 8], dtype=torch.int32)
            positions = torch.arange(8, dtype=torch.int64)
            table.compute_slot_mapping(1, qsl, positions)

            slot = table.slot_mapping.gpu
            # virtual block = block_size * dcp_world_size = 8; positions 0..7
            # all live in virtual block 0 -> physical block table[0, 0] = 3.
            # dcp ownership: (vbo // interleave) % 2 == 0 -> even vbo owned.
            # block offset inside the physical block is vbo // dcp_world_size.
            for pos in range(8):
                if pos % 2 == 1:
                    expected = -1
                else:
                    expected = 3 * block_size + pos // 2
                assert slot[pos].item() == expected, f"pos={pos}"
        finally:
            BlockTable.compute_slot_mapping = real


# ---------------------------------------------------------------------------
# patch_attention_backend_registry
# ---------------------------------------------------------------------------


class TestAttentionBackendRegistry:
    @pytest.mark.platforms("kunlunxin")
    def test_registers_kunlunxin_backend_as_custom(self):
        import vllm.v1.attention.backends.registry as registry

        from vllm_fl.dispatch.backends.vendor.kunlunxin.patch import (
            patch_attention_backend_registry,
        )

        member = registry.AttentionBackendEnum.CUSTOM
        was_overridden = member.is_overridden()
        try:
            patch_attention_backend_registry()
            assert member.is_overridden()
            assert (
                member.get_path()
                == "vllm_fl.dispatch.backends.vendor.kunlunxin.impl."
                "attention.KunlunxinAttentionBackend"
            )
            from vllm_fl.dispatch.backends.vendor.kunlunxin.impl.attention import (
                KunlunxinAttentionBackend,
            )

            assert member.get_class() is KunlunxinAttentionBackend
        finally:
            if not was_overridden:
                member.clear_override()


# ---------------------------------------------------------------------------
# patch_sampler_rng
# ---------------------------------------------------------------------------


class TestSamplerRng:
    @pytest.mark.platforms("kunlunxin")
    def test_wrapped_sample_broadcasts_across_tp(self, monkeypatch):
        import vllm.v1.sample.ops.topk_topp_sampler as sampler_mod

        from vllm_fl.dispatch.backends.vendor.kunlunxin.patch import (
            patch_sampler_rng,
        )

        real = sampler_mod.random_sample

        broadcasted = []
        monkeypatch.setattr(
            "vllm.distributed.get_tp_group",
            lambda: type(
                "Group",
                (),
                {
                    "world_size": 4,
                    "broadcast": staticmethod(
                        lambda tensor, src=0: broadcasted.append((tensor, src))
                    ),
                },
            )(),
        )

        try:
            patch_sampler_rng()
            wrapped = sampler_mod.random_sample
            assert wrapped is not real
            assert wrapped.__name__ == "_broadcast_random_sample"

            probs = torch.rand(2, 5)
            generators = {i: torch.Generator() for i in range(2)}
            out = wrapped(probs, generators)
            assert out.shape == (2,)
            # the sampled tokens were broadcast from rank 0
            assert len(broadcasted) == 1
            tensor, src = broadcasted[0]
            assert src == 0
            torch.testing.assert_close(tensor, out)
        finally:
            sampler_mod.random_sample = real

    @pytest.mark.platforms("kunlunxin")
    def test_uninitialized_tp_group_warns_but_returns(self, caplog):
        import vllm.v1.sample.ops.topk_topp_sampler as sampler_mod

        from vllm_fl.dispatch.backends.vendor.kunlunxin.patch import (
            patch_sampler_rng,
        )

        real = sampler_mod.random_sample
        try:
            patch_sampler_rng()
            wrapped = sampler_mod.random_sample
            probs = torch.rand(2, 5)
            generators = {i: torch.Generator() for i in range(2)}
            with caplog.at_level("WARNING", logger="vllm_fl.dispatch.backends."
                                   "vendor.kunlunxin.patch"):
                out = wrapped(probs, generators)
            assert out.shape == (2,)
            assert any(
                "Failed to broadcast tokens in TP group" in r.message
                for r in caplog.records
            )
        finally:
            sampler_mod.random_sample = real


# ---------------------------------------------------------------------------
# patch_causal_conv1d
# ---------------------------------------------------------------------------


class TestCausalConv1dAdapters:
    @pytest.mark.platforms("kunlunxin")
    def test_adapters_transpose_ncw_to_nwc_and_back(self):
        import vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn as gdn
        import vllm.model_executor.layers.mamba.ops.causal_conv1d as conv1d_lib

        import vllm_fl.dispatch
        import vllm_fl.dispatch.backends.vendor.kunlunxin.patch as patch_mod
        from vllm_fl.dispatch.backends.vendor.kunlunxin.patch import (
            patch_causal_conv1d,
        )

        real_gdn_fn = gdn.causal_conv1d_fn
        real_gdn_update = gdn.causal_conv1d_update
        real_lib_fn = conv1d_lib.causal_conv1d_fn
        real_lib_update = conv1d_lib.causal_conv1d_update
        real_resolve = vllm_fl.dispatch.resolve_op

        kernel_calls = []

        def fake_kernel(x, weight, bias, conv_states, query_start_loc, **kw):
            kernel_calls.append(
                {
                    "x": x.clone(),
                    "conv_states": conv_states.clone(),
                    "qsl": query_start_loc,
                    "kw": kw,
                }
            )
            # NCW-expected output is (dim, tokens); kernel got (tokens, dim)
            return torch.ones_like(x)

        def fake_update(x, conv_state, weight, bias, activation, **kw):
            kernel_calls.append(
                {
                    "x": x.clone(),
                    "conv_state": conv_state.clone(),
                    "bias": bias,
                    "activation": activation,
                    "kw": kw,
                }
            )
            return x

        vllm_fl.dispatch.resolve_op = lambda name: (
            fake_kernel if name == "causal_conv1d_fn" else fake_update
        )
        try:
            patch_causal_conv1d()
            # installed on both surfaces
            assert gdn.causal_conv1d_fn is not real_gdn_fn
            assert gdn.causal_conv1d_update is not real_gdn_update
            assert conv1d_lib.causal_conv1d_fn is gdn.causal_conv1d_fn
            assert conv1d_lib.causal_conv1d_update is gdn.causal_conv1d_update

            # NCW input (dim=6, tokens=3)
            x = torch.arange(18, dtype=torch.float32).view(6, 3)
            weight = torch.ones(6, 6)
            conv_states = torch.arange(12, dtype=torch.float32).view(1, 6, 2)
            qsl = torch.tensor([0, 3])
            out = gdn.causal_conv1d_fn(
                x, weight, None, conv_states, qsl, custom_kw=7
            )
            # adapter passed x transposed to NWC and transposed back
            call = kernel_calls[0]
            assert tuple(call["x"].shape) == (3, 6)
            torch.testing.assert_close(call["x"], x.transpose(0, 1))
            assert tuple(call["conv_states"].shape) == (1, 2, 6)
            assert call["kw"] == {"custom_kw": 7}
            assert tuple(out.shape) == (6, 3)

            # update adapter passes state transposed and x unchanged
            kernel_calls.clear()
            x2 = torch.arange(6, dtype=torch.float32).view(1, 6)
            state = torch.arange(12, dtype=torch.float32).view(1, 6, 2)
            out2 = gdn.causal_conv1d_update(
                x2, state, weight, bias="BIAS", activation="silu"
            )
            call = kernel_calls[0]
            assert call["x"] is not None
            torch.testing.assert_close(call["x"], x2)
            assert tuple(call["conv_state"].shape) == (1, 2, 6)
            assert call["bias"] == "BIAS"
            assert call["activation"] == "silu"
            torch.testing.assert_close(out2, x2)
        finally:
            vllm_fl.dispatch.resolve_op = real_resolve
            gdn.causal_conv1d_fn = real_gdn_fn
            gdn.causal_conv1d_update = real_gdn_update
            conv1d_lib.causal_conv1d_fn = real_lib_fn
            conv1d_lib.causal_conv1d_update = real_lib_update


# ---------------------------------------------------------------------------
# patch_fla_ops
# ---------------------------------------------------------------------------


class TestFlaOps:
    @pytest.mark.platforms("kunlunxin")
    def test_replaces_all_entry_points_with_kunlunxin_impls(self):
        import vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn as gdn

        real_gdn_chunk = gdn.fla_chunk_gated_delta_rule
        with _fake_fla_modules() as (ops, chunk, rec):
            from vllm_fl.dispatch.backends.vendor.kunlunxin.patch import (
                patch_fla_ops,
            )

            with _xtorch_ops_stub(
                chunk_gated_delta_rule_fwdv2=lambda *a, **k: None,
                fused_recurrent_gated_delta_rule_fwdv2=lambda *a, **k: None,
            ):
                # import the kunlunxin impls while the stub is active so the
                # module-scope `import xtorch_ops` resolves
                from vllm_fl.dispatch.backends.vendor.kunlunxin.impl.fla.chunk import (
                    chunk_gated_delta_rule as klx_chunk,
                )
                from vllm_fl.dispatch.backends.vendor.kunlunxin.impl.fla.fused_recurrent import (
                    fused_recurrent_gated_delta_rule as klx_recurrent,
                )
                patch_fla_ops()
            assert ops.chunk_gated_delta_rule is klx_chunk
            assert chunk.chunk_gated_delta_rule is klx_chunk
            assert gdn.fla_chunk_gated_delta_rule is klx_chunk
            assert ops.fused_recurrent_gated_delta_rule is klx_recurrent
            assert rec.fused_recurrent_gated_delta_rule is klx_recurrent
        gdn.fla_chunk_gated_delta_rule = real_gdn_chunk


# ---------------------------------------------------------------------------
# patch_fused_gdn_gating: fused_post_conv_prep_kunlunxin
# ---------------------------------------------------------------------------


class TestFusedPostConvPrep:
    @pytest.fixture
    def installed(self):
        yield from _apply_fused_gdn_gating()

    @pytest.mark.platforms("kunlunxin")
    @pytest.mark.parametrize(
        ("apply_l2norm", "output_g_exp"),
        [(True, False), (False, True)],
        ids=["l2norm", "g_exp"],
    )
    def test_numeric_oracle(self, installed, apply_l2norm, output_g_exp):
        prep, _ = installed
        torch.manual_seed(23)
        num_k_heads, head_k_dim, head_v_dim = 2, 3, 4
        num_v_heads = 3
        q_size = num_k_heads * head_k_dim
        k_size = num_k_heads * head_k_dim
        v_size = num_v_heads * head_v_dim
        conv_output = torch.randn(5, q_size + k_size + v_size)
        a = torch.randn(5, num_v_heads)
        b = torch.randn(5, num_v_heads)
        A_log = torch.randn(num_v_heads)
        dt_bias = torch.randn(num_v_heads)

        q, k, v, g, beta = prep(
            conv_output,
            a,
            b,
            A_log,
            dt_bias,
            num_k_heads,
            head_k_dim,
            head_v_dim,
            apply_l2norm=apply_l2norm,
            output_g_exp=output_g_exp,
        )

        # independent oracle
        q_ref, k_ref, v_ref = torch.split(
            conv_output, (q_size, k_size, v_size), dim=-1
        )
        q_ref = q_ref.reshape(-1, num_k_heads, head_k_dim)
        k_ref = k_ref.reshape(-1, num_k_heads, head_k_dim)
        v_ref = v_ref.reshape(-1, num_v_heads, head_v_dim)
        if apply_l2norm:
            q_ref = F.normalize(q_ref, p=2, dim=-1)
            k_ref = F.normalize(k_ref, p=2, dim=-1)
        g_ref = -torch.exp(A_log.float()) * F.softplus((a + dt_bias).float())
        if output_g_exp:
            g_ref = torch.exp(g_ref)
        beta_ref = torch.sigmoid(b.float())

        torch.testing.assert_close(q, q_ref)
        torch.testing.assert_close(k, k_ref)
        torch.testing.assert_close(v, v_ref)
        torch.testing.assert_close(g, g_ref)
        torch.testing.assert_close(beta, beta_ref)
        # layouts
        assert tuple(q.shape) == (5, num_k_heads, head_k_dim)
        assert tuple(k.shape) == (5, num_k_heads, head_k_dim)
        assert tuple(v.shape) == (5, num_v_heads, head_v_dim)

    @pytest.mark.platforms("kunlunxin")
    def test_defaults_l2norm_and_linear_g(self, installed):
        prep, _ = installed
        torch.manual_seed(29)
        conv_output = torch.randn(3, 2 * 2 + 2 * 2 + 2 * 3)
        a = torch.randn(3, 2)
        b = torch.randn(3, 2)
        A_log = torch.randn(2)
        dt_bias = torch.randn(2)
        q, k, v, g, beta = prep(conv_output, a, b, A_log, dt_bias, 2, 2, 3)
        # default apply_l2norm=True
        q_ref, _, _ = torch.split(conv_output, (4, 4, 6), dim=-1)
        q_ref = F.normalize(q_ref.reshape(-1, 2, 2), p=2, dim=-1)
        torch.testing.assert_close(q, q_ref)
        # default output_g_exp=False -> linear g
        g_ref = -torch.exp(A_log.float()) * F.softplus((a + dt_bias).float())
        torch.testing.assert_close(g, g_ref)


# ---------------------------------------------------------------------------
# patch_fused_gdn_gating: fused_sigmoid_gating_delta_rule_update_kunlunxin
# ---------------------------------------------------------------------------


class TestFusedSigmoidGatingDeltaRuleUpdate:
    @pytest.fixture
    def installed(self):
        from vllm_fl.dispatch.backends.vendor.kunlunxin.patch import (
            patch_fused_gdn_gating,
        )

        def fake_kernel(*a, **k):
            return None

        import vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn as gdn

        real_prep = gdn.fused_post_conv_prep
        real_update = gdn.fused_sigmoid_gating_delta_rule_update
        real_gating = getattr(gdn, "fused_gdn_gating", None)
        with _xtorch_ops_stub(
            fused_gdn_gating=fake_kernel,
            fused_recurrent_gated_delta_rule_fwdv2=fake_kernel,
        ), _fake_fused_recurrent_impl() as calls:
            patch_fused_gdn_gating()
            try:
                yield (
                    gdn.fused_sigmoid_gating_delta_rule_update,
                    calls,
                )
            finally:
                gdn.fused_post_conv_prep = real_prep
                gdn.fused_sigmoid_gating_delta_rule_update = real_update
                if real_gating is None:
                    if hasattr(gdn, "fused_gdn_gating"):
                        del gdn.fused_gdn_gating
                else:
                    gdn.fused_gdn_gating = real_gating

    def _inputs(self, seed=31):
        torch.manual_seed(seed)
        B, T, H, HV, K, V = 1, 4, 2, 3, 2, 2
        return {
            "A_log": torch.randn(HV),
            "a": torch.randn(T, HV),
            "b": torch.randn(T, HV),
            "dt_bias": torch.randn(HV),
            "q": torch.randn(B, T, H, K),
            "k": torch.randn(B, T, H, K),
            "v": torch.randn(B, T, HV, V),
        }

    @pytest.mark.platforms("kunlunxin")
    def test_gating_math_matches_oracle(self, installed):
        update, calls = installed
        x = self._inputs()
        result = update(
            x["A_log"], x["a"], x["b"], x["dt_bias"],
            x["q"], x["k"], x["v"],
            beta=2.0, threshold=15.0, scale=0.7,
            initial_state="STATE", inplace_final_state=False,
        )
        assert result == ("O", "HT")
        g_oracle = (
            -torch.exp(x["A_log"].float())
            * F.softplus((x["a"] + x["dt_bias"]).float(), beta=2.0, threshold=15.0)
        ).unsqueeze(0)
        torch.testing.assert_close(calls["g"], g_oracle)
        torch.testing.assert_close(
            calls["beta"], torch.sigmoid(x["b"].float()).to(x["q"].dtype).unsqueeze(0)
        )
        assert calls["scale"] == 0.7
        assert calls["inplace_final_state"] is False
        assert calls["initial_state"] == "STATE"
        assert calls["q"] is x["q"]

    @pytest.mark.platforms("kunlunxin")
    def test_kda_raises_not_implemented(self, installed):
        update, _ = installed
        x = self._inputs()
        with pytest.raises(NotImplementedError, match="KDA"):
            update(
                x["A_log"], x["a"], x["b"], x["dt_bias"],
                x["q"], x["k"], x["v"], is_kda=True,
            )

    @pytest.mark.platforms("kunlunxin")
    def test_use_qk_l2norm_pre_normalizes_and_disables_flag(self, installed):
        update, calls = installed
        x = self._inputs()
        update(
            x["A_log"], x["a"], x["b"], x["dt_bias"],
            x["q"], x["k"], x["v"],
            use_qk_l2norm_in_kernel=True,
        )
        torch.testing.assert_close(
            calls["q"], F.normalize(x["q"], p=2, dim=-1)
        )
        torch.testing.assert_close(
            calls["k"], F.normalize(x["k"], p=2, dim=-1)
        )
        assert calls["use_qk_l2norm_in_kernel"] is False

    @pytest.mark.platforms("kunlunxin")
    def test_no_l2norm_leaves_qk_untouched(self, installed):
        update, calls = installed
        x = self._inputs()
        update(
            x["A_log"], x["a"], x["b"], x["dt_bias"],
            x["q"], x["k"], x["v"],
            use_qk_l2norm_in_kernel=False,
        )
        assert calls["q"] is x["q"]
        assert calls["k"] is x["k"]
        assert calls["use_qk_l2norm_in_kernel"] is False

    @pytest.mark.platforms("kunlunxin")
    def test_ssm_state_indices_sliced_to_decode_count(self, installed):
        update, calls = installed
        x = self._inputs()
        indices = torch.arange(8)
        update(
            x["A_log"], x["a"], x["b"], x["dt_bias"],
            x["q"], x["k"], x["v"],
            cu_seqlens=torch.tensor([0, 2, 4]),
            ssm_state_indices=indices,
        )
        # cu_seqlens has 3 entries -> 2 decode sequences -> first 2 indices
        torch.testing.assert_close(
            calls["ssm_state_indices"], indices[: 3 - 1]
        )

    @pytest.mark.platforms("kunlunxin")
    def test_2d_indices_are_not_sliced(self, installed):
        update, calls = installed
        x = self._inputs()
        indices = torch.arange(6).view(2, 3)
        update(
            x["A_log"], x["a"], x["b"], x["dt_bias"],
            x["q"], x["k"], x["v"],
            cu_seqlens=torch.tensor([0, 2, 4]),
            ssm_state_indices=indices,
        )
        assert calls["ssm_state_indices"] is indices

    @pytest.mark.platforms("kunlunxin")
    def test_missing_indices_are_not_sliced(self, installed):
        update, calls = installed
        x = self._inputs()
        update(
            x["A_log"], x["a"], x["b"], x["dt_bias"],
            x["q"], x["k"], x["v"],
            cu_seqlens=torch.tensor([0, 4]),
        )
        assert calls["ssm_state_indices"] is None


# ---------------------------------------------------------------------------
# patch_ssm_cache_update
# ---------------------------------------------------------------------------


class TestSsmCacheUpdate:
    @pytest.mark.platforms("kunlunxin")
    def test_rewrites_forward_core_source_and_installs(self):
        import vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn as gdn

        from vllm_fl.dispatch.backends.vendor.kunlunxin.patch import (
            patch_ssm_cache_update,
        )

        cls = gdn.QwenGatedDeltaNetAttention
        real_core = cls._forward_core
        try:
            patch_ssm_cache_update()
            installed = cls._forward_core
            assert installed is not real_core
            # the rewritten function is exec'd from modified source, so it has
            # no retrievable source; verify via its code object instead.
            code = installed.__code__
            names = set(code.co_names) | set(code.co_varnames)
            assert "prefill_has_initial_state" in names
            assert "ssm_state" in names
            assert "prefill_state_indices" in names
            # the closure globals carry the injected helper
            helper = installed.__globals__.get("_kunlunxin_write_ssm_cache")
            assert helper is not None
            assert callable(helper)
        finally:
            cls._forward_core = real_core

    @pytest.mark.platforms("kunlunxin")
    def test_write_ssm_cache_reshapes_and_calls_flash_op(self):
        """The helper flattens states and calls reshape_and_cache_flash."""
        import vllm_fl.dispatch.backends.vendor.kunlunxin.impl.attention as att_mod
        from vllm_fl.dispatch.backends.vendor.kunlunxin.patches.patch_forward_core import (
            _kunlunxin_write_ssm_cache,
        )

        calls = []
        real_flash = att_mod.KunlunxinPagedAttention.reshape_and_cache_flash

        def recording_flash(last, none1, cast_state, none2, indices):
            calls.append(
                {
                    "written": last.clone(),
                    "cast_state": cast_state.clone(),
                    "indices": indices,
                }
            )
            return 0

        att_mod.KunlunxinPagedAttention.reshape_and_cache_flash = (
            staticmethod(recording_flash)
        )
        try:
            # ssm_state (N, HV, K, V) with HV=2, K=3, V=5
            ssm_state = torch.zeros(4, 2, 3, 5)
            # last (N, HV, K*2, V): an already-4d state shaped like (4, 2, 6, 5)
            last = torch.zeros(4, 2, 6, 5)
            indices = torch.tensor([0, 2])
            _kunlunxin_write_ssm_cache(ssm_state, last, indices)
        finally:
            att_mod.KunlunxinPagedAttention.reshape_and_cache_flash = real_flash

        assert len(calls) == 1
        call = calls[0]
        # last_recurrent_state.view(N, -1, V) merges HV and K: 2*6 = 12
        assert tuple(call["written"].shape) == (4, 12, 5)
        # ssm_state.view(N, 1, -1, V) flattens (HV, K) into one dim
        assert tuple(call["cast_state"].shape) == (4, 1, 6, 5)
        assert call["indices"] is indices
