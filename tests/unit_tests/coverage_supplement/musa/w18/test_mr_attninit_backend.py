"""Extra coverage for attention-backend / metadata / cudagraph helpers.

Complements w3d/test_mr_kv_attn_cache.py by targeting the branches it does not
exercise:

* ``initialize_attn_backend``: already-initialized guard, dedupe by
  (backend name, spec, num_heads_q), ``num_heads`` fallback for non-attention
  layers, and the ``UniformTypeKVCacheSpecs`` per-layer spec selection.
* ``initialize_metadata_builders``: multi-group loop where a later group has no
  kernel block size (``... else None``).
* ``_check_and_update_cudagraph_mode``: empty backend list (no candidate) and
  equal-support backends that never replace the initial ALWAYS candidate,
  plus the ``uses_draft_model`` drafter path.
* ``calculate_reorder_batch_threshold``: single-group reduce.

Runners are ``object.__new__(ModelRunnerFL)`` with hand-set attributes; all
attention backends / groups / builders are local doubles.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vllm_fl.worker import model_runner as mr

MR = mr.ModelRunnerFL


def new_runner(**attrs):
    runner = object.__new__(MR)
    for name, value in attrs.items():
        setattr(runner, name, value)
    return runner


# ---------------------------------------------------------------------------
# initialize_attn_backend: dedupe + num_heads fallback
# ---------------------------------------------------------------------------
class _Backend:
    def __init__(self, name):
        self.name = name

    def full_cls_name(self):
        return self.name


class _Group:
    def __init__(self, attn_backend, layer_names, kv_cache_spec, group_id):
        self.attn_backend = attn_backend
        self.layer_names = layer_names
        self.kv_cache_spec = kv_cache_spec
        self.group_id = group_id


class _CGBuilder:
    def __init__(self, support):
        self._support = support

    def get_cudagraph_support(self, cfg, spec):
        return self._support


class _CGBackend:
    """Hashable attention-backend double (SimpleNamespace defines __eq__ and
    is therefore unhashable, but the source puts backends into sets/dicts)."""

    def __init__(self, name, support):
        self.__name__ = name
        self._support = support

    def get_builder_cls(self):
        return _CGBuilder(self._support)


def _install_attn_doubles(monkeypatch, layers):
    monkeypatch.setattr(
        mr, "get_layers_from_vllm_config", lambda cfg, lt, names: {n: layers[n] for n in names}
    )
    monkeypatch.setattr(mr, "check_attention_cp_compatibility", lambda cfg: None)
    monkeypatch.setattr(mr, "UniformTypeKVCacheSpecs", type("UniformTypeKVCacheSpecs", (), {}))
    monkeypatch.setattr(mr, "AttentionGroup", _Group)
    monkeypatch.setattr(
        mr,
        "create_fast_prefill_custom_backend",
        lambda name, base: _Backend("fast"),
        raising=False,
    )


def test_initialize_attn_backend_dedupes_and_falls_back_num_heads(monkeypatch):
    shared = _Backend("plain")
    layers = {
        "l1": SimpleNamespace(num_heads=8, get_attn_backend=lambda: shared),
        "l2": SimpleNamespace(num_heads=8, get_attn_backend=lambda: shared),
        "l3": SimpleNamespace(num_heads=4, get_attn_backend=lambda: shared),
        # No ``num_heads`` attribute -> getattr fallback of 0.
        "l4": SimpleNamespace(get_attn_backend=lambda: shared),
    }
    _install_attn_doubles(monkeypatch, layers)
    spec = object()

    runner = new_runner(
        attn_groups=[],
        kv_sharing_fast_prefill_eligible_layers=set(),
        vllm_config=SimpleNamespace(),
    )
    runner._check_and_update_cudagraph_mode = MagicMock()
    cfg = SimpleNamespace(
        kv_cache_groups=[SimpleNamespace(layer_names=["l1", "l2", "l3", "l4"], kv_cache_spec=spec)]
    )

    MR.initialize_attn_backend(runner, cfg, is_profiling=True)

    # One KV cache group -> one list of attention groups.
    assert len(runner.attn_groups) == 1
    groups = runner.attn_groups[0]
    # (plain, spec, 8) merges l1+l2; 4 and 0 fallback split out.
    assert sorted(len(g.layer_names) for g in groups) == [1, 1, 2]
    merged = next(g for g in groups if len(g.layer_names) == 2)
    assert merged.layer_names == ["l1", "l2"]
    assert merged.group_id == 0
    # All groups share the same backend + spec object.
    assert all(g.attn_backend is shared and g.kv_cache_spec is spec for g in groups)
    # is_profiling propagated to the cudagraph resolver.
    args, kwargs = runner._check_and_update_cudagraph_mode.call_args
    assert kwargs["is_profiling"] is True
    assert args[0] == [{shared}]


def test_initialize_attn_backend_uniform_type_spec_per_layer(monkeypatch):
    class Uniform:
        def __init__(self, specs):
            self.kv_cache_specs = specs

    shared = _Backend("plain")
    layers = {
        "a": SimpleNamespace(num_heads=8, get_attn_backend=lambda: shared),
        "b": SimpleNamespace(num_heads=8, get_attn_backend=lambda: shared),
    }

    monkeypatch.setattr(
        mr, "get_layers_from_vllm_config", lambda cfg, lt, names: {n: layers[n] for n in names}
    )
    monkeypatch.setattr(mr, "check_attention_cp_compatibility", lambda cfg: None)
    monkeypatch.setattr(mr, "UniformTypeKVCacheSpecs", Uniform)
    monkeypatch.setattr(mr, "AttentionGroup", _Group)

    spec_a, spec_b = object(), object()
    uniform = Uniform({"a": spec_a, "b": spec_b})
    runner = new_runner(
        attn_groups=[],
        kv_sharing_fast_prefill_eligible_layers=set(),
        vllm_config=SimpleNamespace(),
    )
    runner._check_and_update_cudagraph_mode = MagicMock()
    cfg = SimpleNamespace(
        kv_cache_groups=[SimpleNamespace(layer_names=["a", "b"], kv_cache_spec=uniform)]
    )

    MR.initialize_attn_backend(runner, cfg)

    groups = {g.layer_names[0]: g for g in runner.attn_groups[0]}
    # Different per-layer specs -> two distinct groups, each with its own spec.
    assert groups["a"].kv_cache_spec is spec_a
    assert groups["b"].kv_cache_spec is spec_b
    assert len(runner.attn_groups[0]) == 2


def test_initialize_attn_backend_rejects_reinitialization():
    runner = new_runner(attn_groups=[[object()]], vllm_config=SimpleNamespace())
    with pytest.raises(AssertionError):
        MR.initialize_attn_backend(runner, SimpleNamespace(kv_cache_groups=[]))


# ---------------------------------------------------------------------------
# initialize_metadata_builders: multi group + missing kernel block size
# ---------------------------------------------------------------------------
def test_initialize_metadata_builders_second_group_without_kernel_size():
    g0 = SimpleNamespace(create_metadata_builders=MagicMock())
    g1 = SimpleNamespace(create_metadata_builders=MagicMock())
    runner = new_runner(
        attn_groups=[[g0], [g1]],
        parallel_config=SimpleNamespace(use_ubatching=False),
        speculative_config=None,
        vllm_config=SimpleNamespace(),
        device="dev",
    )
    runner.calculate_reorder_batch_threshold = MagicMock()
    cfg = SimpleNamespace(kv_cache_groups=[object(), object()])

    MR.initialize_metadata_builders(runner, cfg, [16])

    g0.create_metadata_builders.assert_called_once_with(
        runner.vllm_config, "dev", 16, num_metadata_builders=1
    )
    g1.create_metadata_builders.assert_called_once_with(
        runner.vllm_config, "dev", None, num_metadata_builders=1
    )
    runner.calculate_reorder_batch_threshold.assert_called_once()


# ---------------------------------------------------------------------------
# _check_and_update_cudagraph_mode: empty / equal support / draft model
# ---------------------------------------------------------------------------
def test_check_and_update_cudagraph_mode_empty_backends_uses_always():
    runner = new_runner(
        compilation_config=SimpleNamespace(
            resolve_cudagraph_mode_and_sizes=MagicMock(return_value="MODE")
        ),
        cudagraph_dispatcher=SimpleNamespace(initialize_cudagraph_keys=MagicMock()),
        uniform_decode_query_len=3,
        kv_cache_config="kvcfg",
        max_num_reqs=8,
        vllm_config=SimpleNamespace(),
        speculative_config=None,
    )

    MR._check_and_update_cudagraph_mode(runner, [], [])

    args, kwargs = runner.compilation_config.resolve_cudagraph_mode_and_sizes.call_args
    assert args[0] is mr.AttentionCGSupport.ALWAYS
    assert args[1] is None
    assert kwargs["is_profiling"] is False
    runner.cudagraph_dispatcher.initialize_cudagraph_keys.assert_called_once_with("MODE", 3)


def test_check_and_update_cudagraph_mode_equal_support_keeps_none_candidate():
    always = SimpleNamespace(value=mr.AttentionCGSupport.ALWAYS.value)

    def make_backend(name):
        return _CGBackend(name, always)

    b1, b2 = make_backend("B1"), make_backend("B2")
    runner = new_runner(
        compilation_config=SimpleNamespace(
            resolve_cudagraph_mode_and_sizes=MagicMock(return_value="M")
        ),
        cudagraph_dispatcher=SimpleNamespace(initialize_cudagraph_keys=MagicMock()),
        uniform_decode_query_len=1,
        kv_cache_config="kvcfg",
        max_num_reqs=2,
        vllm_config=SimpleNamespace(),
        speculative_config=None,
    )

    MR._check_and_update_cudagraph_mode(
        runner, [{b1, b2}], [SimpleNamespace(kv_cache_spec="s")]
    )

    args = runner.compilation_config.resolve_cudagraph_mode_and_sizes.call_args.args
    assert args[0] is mr.AttentionCGSupport.ALWAYS
    # Equal support never beats the initial ALWAYS candidate -> stays None.
    assert args[1] is None


def test_check_and_update_cudagraph_mode_draft_model_drafter(monkeypatch):
    class Proposer:
        pass

    for name in (
        "EagleProposer",
        "DFlashProposer",
        "DraftModelProposer",
        "ExtractHiddenStatesProposer",
        "Gemma4Proposer",
    ):
        monkeypatch.setattr(mr, name, Proposer)

    drafter = Proposer()
    drafter.initialize_cudagraph_keys = MagicMock()
    spec = SimpleNamespace(
        use_eagle=lambda: False,
        uses_draft_model=lambda: True,
        uses_extract_hidden_states=lambda: False,
    )
    always = SimpleNamespace(value=mr.AttentionCGSupport.ALWAYS.value)
    backend = _CGBackend("B", always)
    runner = new_runner(
        compilation_config=SimpleNamespace(
            resolve_cudagraph_mode_and_sizes=MagicMock(return_value="MODE")
        ),
        cudagraph_dispatcher=SimpleNamespace(initialize_cudagraph_keys=MagicMock()),
        uniform_decode_query_len=2,
        kv_cache_config="kvcfg",
        max_num_reqs=4,
        vllm_config=SimpleNamespace(),
        speculative_config=spec,
        drafter=drafter,
    )

    MR._check_and_update_cudagraph_mode(
        runner, [{backend}], [SimpleNamespace(kv_cache_spec="s")]
    )

    drafter.initialize_cudagraph_keys.assert_called_once_with("MODE")


# ---------------------------------------------------------------------------
# calculate_reorder_batch_threshold: single group
# ---------------------------------------------------------------------------
def test_calculate_reorder_threshold_single_group():
    group = SimpleNamespace(
        get_metadata_builder=lambda: SimpleNamespace(reorder_batch_threshold=7)
    )
    runner = new_runner(attn_groups=[[group]])
    MR.calculate_reorder_batch_threshold(runner)
    assert runner.reorder_batch_threshold == 7
