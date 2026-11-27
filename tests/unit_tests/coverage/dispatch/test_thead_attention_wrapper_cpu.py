# Copyright 2025 BAAI. All rights reserved.

"""CPU contract for the T-Head PPU FA3 varlen bridge.

``impl/attention.py`` has heavy import-time side effects: it imports
``flash_attn_3._C`` to register ``torch.ops.flash_attn_3`` and then injects
four functions into the real ``vllm.v1.attention.backends.flash_attn``
module namespace. This module tests the *interface contract* of
``_thead_flash_attn_varlen_func`` (and the injected scheduler/version
helpers) entirely on CPU by:

1. stubbing ``flash_attn_3`` / ``flash_attn_3._C`` in sys.modules BEFORE
   importing impl/attention.py (save/install/restore in finally),
2. recording calls on a fake ``torch.ops.flash_attn_3.fwd``,
3. snapshotting and restoring every attribute the module injects into the
   vLLM flash_attn module.

The 35-positional-argument layout of the wheel call is pinned exactly from
attention.py L107-143.
"""

from __future__ import annotations

import importlib
import sys
import types

import pytest

pytestmark = [pytest.mark.supplementary]

_MODULE = "vllm_fl.dispatch.backends.vendor.thead.impl.attention"
_FLASH_ATTN_MOD = "vllm.v1.attention.backends.flash_attn"
_INJECTED_NAMES = (
    "flash_attn_varlen_func",
    "get_scheduler_metadata",
    "get_flash_attn_version",
    "reshape_and_cache_flash",
)


class _Sentinel:
    """Unique object so recorder positions are unambiguous."""

    def __init__(self, tag):
        self.tag = tag

    def __repr__(self):
        return f"<{self.tag}>"


class _QSentinel(_Sentinel):
    """Q sentinel carrying head_dim for the default softmax scale."""

    shape = (3, 5, 16)  # head_dim = 16 -> default scale 16 ** -0.5 = 0.25


@pytest.fixture()
def attention_bridge():
    """Import impl/attention.py with stubbed flash_attn_3; restore all.

    Returns (module, recorder) where recorder.calls holds each positional
    argument tuple passed to the fake torch.ops.flash_attn_3.fwd.
    """
    # The real vLLM flash_attn module must import on CPU for the injection
    # target to exist (it degrades gracefully without the FA wheel). On hosts
    # whose vLLM wheel cannot resolve ``current_platform`` at all (e.g. an
    # ``empty`` wheel without the CUDA extension), pre-set the module-level
    # lazy attribute to an inert platform so the import chain proceeds.
    try:
        flash_attn_mod = importlib.import_module(_FLASH_ATTN_MOD)
    except Exception:
        flash_attn_mod = _import_flash_attn_with_inert_platform()
        if flash_attn_mod is None:
            pytest.skip(
                "vLLM flash_attn module unavailable on this host"
            )
    if flash_attn_mod is None:  # pragma: no cover - defensive
        pytest.skip("vLLM flash_attn module unavailable on this host")

    import torch

    saved_modules = {}
    stubs = {
        "flash_attn_3": types.ModuleType("flash_attn_3"),
        "flash_attn_3._C": types.ModuleType("flash_attn_3._C"),
    }
    stubs["flash_attn_3"].__path__ = []
    for name, module in stubs.items():
        saved_modules[name] = sys.modules.get(name)
        sys.modules[name] = module

    # Snapshot every attribute the module will inject; missing attrs map to
    # a private "absent" marker so teardown can delete them again.
    absent = object()
    saved_attrs = {
        name: getattr(flash_attn_mod, name, absent) for name in _INJECTED_NAMES
    }
    # Snapshot the ops-namespace overrides (the real wheel may already have
    # registered fwd / get_scheduler_metadata on real hardware).
    ops_ns = torch.ops.flash_attn_3
    saved_fwd = getattr(ops_ns, "fwd", absent)
    saved_get_sm = getattr(ops_ns, "get_scheduler_metadata", absent)

    recorder = types.SimpleNamespace(calls=[])

    def fake_fwd(*args):
        recorder.calls.append(args)
        return ("OUT", "LSE", "D1", "D2")

    ops_ns.fwd = fake_fwd

    try:
        module = importlib.import_module(_MODULE)
        yield module, recorder
    finally:
        # Restore the flash_attn module namespace.
        for name, value in saved_attrs.items():
            if value is absent:
                if hasattr(flash_attn_mod, name):
                    delattr(flash_attn_mod, name)
            else:
                setattr(flash_attn_mod, name, value)
        # Restore (or remove) the torch.ops namespace overrides.
        if saved_fwd is absent:
            if hasattr(ops_ns, "fwd"):
                delattr(ops_ns, "fwd")
        else:
            ops_ns.fwd = saved_fwd
        if saved_get_sm is absent:
            if hasattr(ops_ns, "get_scheduler_metadata"):
                delattr(ops_ns, "get_scheduler_metadata")
        else:
            ops_ns.get_scheduler_metadata = saved_get_sm
        # Remove the imported modules so order-independence holds.
        for name in (
            _MODULE,
            "vllm_fl.dispatch.backends.vendor.thead.impl.attention",
            "vllm_fl.dispatch.backends.vendor.thead.impl",
            "vllm_fl.dispatch.backends.vendor.thead.impl.cache",
        ):
            sys.modules.pop(name, None)
        for name, module in saved_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
        # Drop vllm modules that only imported because of the inert platform
        # so later tests re-resolve them against the real environment.
        _drop_inert_platform_modules()


def _import_flash_attn_with_inert_platform():
    """Import vLLM's flash_attn module against an inert current_platform.

    Some vLLM wheels (``VLLM_TARGET_DEVICE=empty`` builds) cannot resolve
    ``vllm.platforms.current_platform`` at all — the lazy ``__getattr__``
    imports the CUDA platform, which needs the missing native extension.
    Pre-setting the module attribute with a minimal UnspecifiedPlatform
    subclass lets the flash_attn import chain proceed.

    Returns the imported module or None when even that cannot work. The
    inert platform stays published until
    ``_drop_inert_platform_modules`` runs (the fixture teardown), which
    restores the module's original resolution state.
    """
    try:
        import vllm.platforms as platforms_mod
        from vllm.platforms.interface import UnspecifiedPlatform
    except Exception:
        return None

    if getattr(platforms_mod, "_inert_platform_installed", False):
        # Already installed by an earlier fixture call in this session:
        # re-import directly.
        try:
            return importlib.import_module(_FLASH_ATTN_MOD)
        except Exception:
            return None

    class _InertPlatform(UnspecifiedPlatform):
        device_type = "cuda"
        vendor_name = "thead"
        dist_backend = "nccl"

    inert = _InertPlatform()
    marker = types.SimpleNamespace(
        saved_current=platforms_mod.__dict__.get("current_platform"),
        saved_marker=platforms_mod.__dict__.get("_current_platform"),
    )
    platforms_mod.__dict__["current_platform"] = inert
    platforms_mod._current_platform = inert
    platforms_mod.__dict__["_inert_platform_installed"] = True
    platforms_mod.__dict__["_inert_platform_state"] = marker
    try:
        return importlib.import_module(_FLASH_ATTN_MOD)
    except Exception:
        # Leave nothing half-armed when the fallback also fails.
        _drop_inert_platform_modules()
        return None


def _drop_inert_platform_modules():
    """Undo ``_import_flash_attn_with_inert_platform`` (no-op if unused)."""
    import vllm.platforms as platforms_mod

    if not platforms_mod.__dict__.pop("_inert_platform_installed", None):
        return
    marker = platforms_mod.__dict__.pop("_inert_platform_state", None)
    platforms_mod.__dict__.pop("current_platform", None)
    platforms_mod._current_platform = None
    if marker is not None:
        if marker.saved_current is not None:
            platforms_mod.__dict__["current_platform"] = marker.saved_current
        if marker.saved_marker is not None:
            platforms_mod._current_platform = marker.saved_marker


def _varlen(module, **overrides):
    """Call the bridge with sentinels; every kwarg is overridable."""
    kwargs = dict(
        q=_QSentinel("Q"),
        k=_Sentinel("K"),
        v=_Sentinel("V"),
        max_seqlen_q=5,
        cu_seqlens_q=_Sentinel("CUQ"),
        max_seqlen_k=9,
        cu_seqlens_k=None,
        seqused_k=_Sentinel("SEQK"),
        q_v=_Sentinel("QV"),
        dropout_p=0.0,
        softmax_scale=None,
        causal="CAUSAL",
        window_size=None,
        softcap=1.25,
        alibi_slopes=None,
        deterministic=False,
        return_attn_probs=False,
        block_table=_Sentinel("BT"),
        return_softmax_lse=False,
        out=_Sentinel("OUTBUF"),
        scheduler_metadata=_Sentinel("SCHED"),
        q_descale=_Sentinel("QD"),
        k_descale=_Sentinel("KD"),
        v_descale=_Sentinel("VD"),
        num_splits=11,
        s_aux=_Sentinel("SAUX"),
    )
    kwargs.update(overrides)
    return module._thead_flash_attn_varlen_func(**kwargs)


class TestVarlenGuardClauses:
    """Unsupported features fail fast with a clear error."""

    @pytest.mark.platforms("thead")
    def test_context_parallel_args_raise(self, attention_bridge):
        module, _ = attention_bridge
        with pytest.raises(NotImplementedError, match="context parallel"):
            _varlen(module, cp_world_size=2)
        with pytest.raises(NotImplementedError, match="context parallel"):
            _varlen(module, cp_rank=1)
        with pytest.raises(NotImplementedError, match="context parallel"):
            _varlen(module, cp_tot_seqused_k=_Sentinel("CPTOT"))

    @pytest.mark.platforms("thead")
    def test_fa4_dynamic_mask_args_raise(self, attention_bridge):
        module, _ = attention_bridge
        with pytest.raises(NotImplementedError, match="FA4 dynamic masks"):
            _varlen(module, dynamic_causal=lambda: True)
        with pytest.raises(NotImplementedError, match="FA4 dynamic masks"):
            _varlen(module, mask_mod=_Sentinel("MASKMOD"))
        with pytest.raises(NotImplementedError, match="FA4 dynamic masks"):
            _varlen(module, aux_tensors=[_Sentinel("AUX")])

    @pytest.mark.platforms("thead")
    def test_alibi_slopes_are_rejected(self, attention_bridge):
        module, _ = attention_bridge
        with pytest.raises(AssertionError, match="[Aa]libi"):
            _varlen(module, alibi_slopes=[0.5])


class TestVarlenDefaultsAndLayout:
    """Default handling and the exact 35-argument wheel layout."""

    @pytest.mark.platforms("thead")
    def test_default_softmax_scale_is_inverse_sqrt_head_dim(
        self, attention_bridge
    ):
        module, recorder = attention_bridge
        _varlen(module)
        args = recorder.calls[-1]
        assert args[23] == pytest.approx(16 ** -0.5), (
            "softmax_scale must default to head_dim ** -0.5"
        )

    @pytest.mark.platforms("thead")
    def test_explicit_softmax_scale_passthrough(self, attention_bridge):
        module, recorder = attention_bridge
        _varlen(module, softmax_scale=0.75)
        assert recorder.calls[-1][23] == 0.75

    @pytest.mark.platforms("thead")
    def test_window_none_becomes_open_window_tuple(self, attention_bridge):
        module, recorder = attention_bridge
        _varlen(module)
        args = recorder.calls[-1]
        assert args[25] == -1 and args[26] == -1, (
            "window_size=None must map to (-1, -1)"
        )

    @pytest.mark.platforms("thead")
    def test_window_list_passthrough(self, attention_bridge):
        module, recorder = attention_bridge
        _varlen(module, window_size=[7, -3])
        args = recorder.calls[-1]
        assert args[25] == 7 and args[26] == -3

    @pytest.mark.platforms("thead")
    def test_paged_attention_forces_max_seqlen_k_to_one(self, attention_bridge):
        module, recorder = attention_bridge
        _varlen(module, cu_seqlens_k=None, max_seqlen_k=4096)
        args = recorder.calls[-1]
        assert args[13] == 1, (
            "cu_seqlens_k=None (paged attention) must force max_seqlen_k=1"
        )
        assert 4096 not in args, "the caller's max_seqlen_k must be discarded"

    @pytest.mark.platforms("thead")
    def test_varlen_keeps_caller_max_seqlen_k(self, attention_bridge):
        module, recorder = attention_bridge
        _varlen(module, cu_seqlens_k=_Sentinel("CUK"), max_seqlen_k=4096)
        assert recorder.calls[-1][13] == 4096

    @pytest.mark.platforms("thead")
    def test_exact_35_positional_layout(self, attention_bridge):
        """Pin the full 35-argument wheel schema position by position."""
        module, recorder = attention_bridge
        _varlen(module)
        args = recorder.calls[-1]
        assert len(args) == 35
        # The forwarded sentinels are taken from the call itself; the literal
        # entries pin the fixed schema slots (None placeholders, constants).
        expected = (
            args[0],   # q
            args[1],   # k
            args[2],   # v
            None,      # 3: seqlens_q placeholder
            None,      # 4: k_new, v_new
            args[5],   # q_v
            args[6],   # out
            args[7],   # cu_seqlens_q
            None,      # 8: cu_seqlens_k (None for paged)
            None,      # 9: cu_seqlens_k_new
            None,      # 10: seqused_q
            args[11],  # seqused_k
            5,         # 12: max_seqlen_q
            1,         # 13: max_seqlen_k forced to 1 (paged)
            args[14],  # block_table
            None,      # 15: kv_batch_idx
            None,      # 16: leftpad_k
            None,      # 17: rotary_cos
            None,      # 18: rotary_sin
            None,      # 19: seqlens_rotary
            args[20],  # q_descale
            args[21],  # k_descale
            args[22],  # v_descale
            args[23],  # softmax_scale
            "CAUSAL",  # 24: causal
            -1,        # 25: window left
            -1,        # 26: window right
            0,         # 27: attention_chunk
            1.25,      # 28: softcap
            True,      # 29: is_rotary_interleaved
            args[30],  # scheduler_metadata
            11,        # 31: num_splits
            None,      # 32: pack_gqa
            0,         # 33: sm_margin
            args[34],  # s_aux
        )
        assert args == expected
        # Verify the sentinel identity mapping for every forwarded arg.
        by_tag = {a.tag: a for a in args if isinstance(a, _Sentinel)}
        for position, tag in (
            (0, "Q"),
            (1, "K"),
            (2, "V"),
            (5, "QV"),
            (6, "OUTBUF"),
            (7, "CUQ"),
            (11, "SEQK"),
            (14, "BT"),
            (20, "QD"),
            (21, "KD"),
            (22, "VD"),
            (30, "SCHED"),
            (34, "SAUX"),
        ):
            assert args[position] is by_tag[tag], f"position {position} must be {tag}"

    @pytest.mark.platforms("thead")
    def test_out_passthrough_and_return_modes(self, attention_bridge):
        module, recorder = attention_bridge
        out = _varlen(module)
        assert out == "OUT", "default return is fwd's first result"
        assert recorder.calls[-1][6].tag == "OUTBUF"

        result = _varlen(module, return_softmax_lse=True)
        assert result == ("OUT", "LSE"), (
            "return_softmax_lse must yield (out, softmax_lse)"
        )


class TestModuleNamespaceInjection:
    """The bridge injects its helpers into the vLLM flash_attn module."""

    @pytest.mark.platforms("thead")
    def test_varlen_func_injected_into_flash_attn(self, attention_bridge):
        import vllm.v1.attention.backends.flash_attn as fam

        module, _ = attention_bridge
        assert fam.flash_attn_varlen_func is module._thead_flash_attn_varlen_func

    @pytest.mark.platforms("thead")
    def test_get_scheduler_metadata_injected(self, attention_bridge):
        import vllm.v1.attention.backends.flash_attn as fam

        module, _ = attention_bridge
        assert fam.get_scheduler_metadata is module._thead_get_scheduler_metadata

    @pytest.mark.platforms("thead")
    def test_version_forced_to_three(self, attention_bridge):
        import vllm.v1.attention.backends.flash_attn as fam

        module, _ = attention_bridge
        assert fam.get_flash_attn_version() == 3
        assert fam.get_flash_attn_version(head_size=256) == 3

    @pytest.mark.platforms("thead")
    def test_cache_writer_injected(self, attention_bridge):
        import vllm.v1.attention.backends.flash_attn as fam

        module, _ = attention_bridge
        assert fam.reshape_and_cache_flash is module.reshape_and_cache_flash_thead


class TestSchedulerMetadataBridge:
    """_thead_get_scheduler_metadata bridges vLLM 0.28 metadata to the wheel."""

    @pytest.mark.platforms("thead")
    def test_scheduler_metadata_arg_layout(self, attention_bridge):
        import torch

        module, _ = attention_bridge
        ops_ns = torch.ops.flash_attn_3
        recorded = []
        absent = object()
        saved = getattr(ops_ns, "get_scheduler_metadata", absent)
        ops_ns.get_scheduler_metadata = lambda *args: recorded.append(args) or "SM"
        try:
            result = module._thead_get_scheduler_metadata(
                batch_size=2,
                max_seqlen_q=17,
                max_seqlen_k=33,
                num_heads_q=8,
                num_heads_kv=2,
                headdim=128,
                cache_seqlens=_Sentinel("CSL"),
                qkv_dtype="BF16",
                headdim_v=None,
                cu_seqlens_q=_Sentinel("CUQ"),
                cu_seqlens_k_new=_Sentinel("CUKN"),
                cache_leftpad=_Sentinel("CLP"),
                page_size=16,
                max_seqlen_k_new=7,
                causal=True,
                window_size=(-5, 9),
                softcap=False,
                num_splits=3,
                pack_gqa=None,
                sm_margin=1,
            )
        finally:
            if saved is absent:
                delattr(ops_ns, "get_scheduler_metadata")
            else:
                ops_ns.get_scheduler_metadata = saved

        assert result == "SM"
        args = recorded[0]
        assert len(args) == 24
        assert args[0] == 2  # batch_size
        assert args[1] == 17  # max_seqlen_q
        assert args[2] == 33  # max_seqlen_k
        assert args[3] == 8  # num_heads_q
        assert args[4] == 2  # num_heads_kv
        assert args[5] == 128  # headdim
        assert args[6] == 128  # headdim_v defaults to headdim
        assert args[7] == "BF16"  # qkv_dtype
        assert args[8].tag == "CSL"
        assert args[9].tag == "CUQ"
        assert args[10] is None
        assert args[11].tag == "CUKN"
        assert args[12] is None
        assert args[13].tag == "CLP"
        assert args[14] == 16  # page_size
        assert args[15] == 7  # max_seqlen_k_new
        assert args[16] is True  # causal
        assert args[17] == -5  # window left
        assert args[18] == 9  # window right
        assert args[19] == 0  # attention_chunk
        assert args[20] is False  # softcap
        assert args[21] == 3  # num_splits
        assert args[22] is None  # pack_gqa
        assert args[23] == 1  # sm_margin
