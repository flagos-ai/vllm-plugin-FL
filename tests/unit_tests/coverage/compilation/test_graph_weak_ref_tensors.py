# Copyright (c) 2026 BAAI. All rights reserved.

"""CPU tests for FL's platform-agnostic graph plumbing (stub-tree doubles).

``vllm_fl/compilation/graph.py`` extends upstream's CUDA-graph wrapper with
vendor weak-reference helpers (NPU, GCU, Iluvatar), a device-dispatching
``weak_ref_tensors``, and registry helpers on ``GraphWrapper``;
``graph_params.py`` tracks per-descriptor capture state;
``graph_runtime.py`` orchestrates capture/replay around that state.

None of those modules import on a machine without the external ``vllm``
package, so every test here installs a minimal fake ``vllm`` tree in
``sys.modules`` (the external boundary only — the plugin sources under test
are loaded verbatim via ``importlib``) and then drives the real functions:

* the NPU/GCU weak-ref families: vendor primitive use and the error paths
  the real-vllm suite cannot reach (missing ``_C`` namespace, storage
  constructor absent);
* ``weak_ref_tensors`` dispatch on ``current_platform`` for the iluvatar,
  NPU, GCU, upstream-success and upstream-failure branches;
* ``GraphWrapper.unwrap`` / ``cudagraph_wrapper`` / ``clear_all_graphs`` and
  the ``_capture_graph`` pool-handle fallback (``graph_pool is None``);
* ``GraphParams`` validation, capture lifecycle and the module-level
  registry helpers;
* ``GraphRuntimeController`` capture/replay scopes with their nested
  finally-restoration, attention updater fan-out, backend binding rules and
  decoder-graph teardown.

The CPU-passthrough, missing-primitive, recursion, weak-ref-workspace,
scope-restoration and backend-dedup contracts that ``test_graph.py`` (with a
real vllm install) and ``tests/unit_tests/test_ascend_graph_contract.py``
already pin are deliberately not duplicated here.
"""

import importlib.util
import logging
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[4]
_GRAPH_SOURCE = _ROOT / "vllm_fl/compilation/graph.py"
_PARAMS_SOURCE = _ROOT / "vllm_fl/compilation/graph_params.py"
_RUNTIME_SOURCE = _ROOT / "vllm_fl/compilation/graph_runtime.py"


class _FakeIntermediateTensors:
    """Stand-in for vLLM's IntermediateTensors carrying a tensors mapping."""

    def __init__(self, tensors):
        self.tensors = tensors


class _FakeVendorTensor:
    """Duck-typed tensor for the vendor weak-ref paths.

    ``weak_ref_*`` only inspects ``device.type`` plus the storage metadata
    accessors, so a plain object (not a ``torch.Tensor`` subclass) is enough
    to drive the vendor branches without any accelerator runtime.  The GCU
    metadata path reads the accessor values directly; the NPU path (and the
    recursion dispatchers) additionally require
    ``isinstance(value, torch.Tensor)``, which this fake satisfies through
    the ``__class__`` property trick used by the core unit tests.
    """

    def __init__(
        self,
        device_type,
        shape=(2, 3),
        stride=(3, 1),
        storage_offset=5,
        nbytes=24,
        data_ptr=777,
        conj=False,
        neg=False,
    ):
        self.device = SimpleNamespace(type=device_type)
        self.shape = shape
        self.dtype = torch.float32
        self._stride = stride
        self._offset = storage_offset
        self._storage = SimpleNamespace(
            nbytes=lambda: nbytes, data_ptr=lambda: data_ptr
        )
        self._conj = conj
        self._neg = neg

    @property
    def __class__(self):
        return torch.Tensor

    def stride(self):
        return self._stride

    def storage_offset(self):
        return self._offset

    def untyped_storage(self):
        return self._storage

    def is_conj(self):
        return self._conj

    def is_neg(self):
        return self._neg


class _StubVllmTree:
    """Build the fake ``vllm`` module graph the compilation modules import.

    Only names ``graph.py`` / ``graph_runtime.py`` import at module scope are
    provided.  ``current_platform`` is mutable per test so each dispatch
    branch can be selected without reloading the sources.
    """

    def __init__(self, device_type="cuda", vendor_name="nvidia", graph_pool=12345):
        self.modules = {}
        self.calls = []
        self._installed = []
        self._previous = {}

        class _CaptureCounter:
            def __init__(self):
                self.num_cudagraph_captured = 0

        class _CUDAGraphMode:
            NONE = "NONE"
            FULL = "FULL"
            PIECEWISE = "PIECEWISE"

        self.counter = _CaptureCounter()
        self.cuda_graph_mode = _CUDAGraphMode

        class _NullCtx:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        class _Platform:
            torch_device_fn = SimpleNamespace(
                current_stream=lambda: "stub-stream",
                synchronize=lambda: None,
                graph=lambda graph, pool=None, stream=None: _NullCtx(),
                Stream=lambda: "stub-new-stream",
            )

            def get_global_graph_pool(self):
                return graph_pool

            def graph_pool_handle(self):
                return "stub-pool-handle"

        self.platform = _Platform()
        self.platform.device_type = device_type
        self.platform.vendor_name = vendor_name
        self._put("vllm")
        self._put("vllm.envs", VLLM_LOGGING_LEVEL="INFO")
        self._put("vllm.compilation")
        self._put("vllm.compilation.counter", compilation_counter=self.counter)
        self._put(
            "vllm.compilation.monitor",
            validate_cudagraph_capturing_enabled=lambda: None,
        )
        self._put(
            "vllm.compilation.cuda_graph",
            CUDAGraphStat=type("CUDAGraphStat", (), {}),
        )
        self._put(
            "vllm.config",
            CUDAGraphMode=_CUDAGraphMode,
            VllmConfig=type("VllmConfig", (), {}),
        )
        self._put("vllm.distributed")
        self._put("vllm.distributed.device_communicators")
        self._put(
            "vllm.distributed.device_communicators.pynccl_allocator",
            set_graph_pool_id=self.calls.append,
        )
        self._put(
            "vllm.forward_context",
            BatchDescriptor=type("BatchDescriptor", (), {}),
            get_forward_context=lambda: None,
            is_forward_context_available=lambda: False,
        )
        self._put("vllm.logger", init_logger=lambda name: logging.getLogger(name))
        self._put("vllm.platforms", current_platform=self.platform)
        self._put("vllm.utils")
        self._put(
            "vllm.utils.torch_utils",
            current_stream=lambda: "stub-stream",
            weak_ref_tensors=lambda value: ("upstream", value),
        )
        self._put("vllm.sequence", IntermediateTensors=_FakeIntermediateTensors)
        self._put("vllm.model_executor")
        self._put("vllm.model_executor.offloader")
        self._put(
            "vllm.model_executor.offloader.base",
            get_offloader=lambda: SimpleNamespace(
                sync_prev_onload=lambda: None, join_after_forward=lambda: None
            ),
        )
        # graph_runtime.bind_attention_groups imports this leaf lazily.
        self._put("vllm.v1")
        self._put("vllm.v1.attention")

        class _AttentionBackend:
            @staticmethod
            def get_impl_cls():
                return type("DefaultImpl", (), {})

        self._put("vllm.v1.attention.backend", AttentionBackend=_AttentionBackend)

    def _put(self, name: str, **attrs: Any):
        module = ModuleType(name)
        module.__path__ = []
        for key, value in attrs.items():
            setattr(module, key, value)
        self.modules[name] = module
        if name not in self._previous:
            self._previous[name] = sys.modules.get(name)
        sys.modules[name] = module
        self._installed.append(name)
        return module

    def set_platform(self, device_type: str, vendor_name: str = "nvidia"):
        """Point ``current_platform`` at a fresh dispatch configuration."""
        self.platform.device_type = device_type
        self.platform.vendor_name = vendor_name
        return self.platform

    def set_upstream_weak_ref(self, func):
        """Replace the upstream weak_ref_tensors used by the fallback path."""
        self.modules["vllm.utils.torch_utils"].weak_ref_tensors = func

    def restore(self):
        """Undo every sys.modules change this tree made."""
        for name in self._installed:
            previous = self._previous[name]
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
        self._installed.clear()
        self._previous.clear()


def _load_module(tree: _StubVllmTree, source: Path, module_name: str):
    """Exec a plugin source file once against the installed stub tree."""
    spec = importlib.util.spec_from_file_location(module_name, source)
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolves types through sys.modules[cls.__module__], so the
    # module must be registered before exec.
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module


@pytest.fixture
def fl_tree(monkeypatch):
    """Install the stub tree and load all three modules under test.

    The graph module is loaded under the real package path
    ``vllm_fl.compilation.graph`` because ``GraphRuntimeController``
    imports that exact name lazily; sibling names get distinct
    ``supp_``-prefixed identities.  Everything is removed again on teardown
    through ``monkeypatch`` plus an explicit snapshot of ``vllm_fl`` roots.
    """
    fl_roots = {
        name: sys.modules.get(name, None)
        for name in list(sys.modules)
        if name == "vllm_fl" or name.startswith("vllm_fl.")
    }
    for name in fl_roots:
        sys.modules.pop(name, None)

    tree = _StubVllmTree()
    try:
        # graph.py must live at its real dotted name for capture_scope's
        # ``from vllm_fl.compilation.graph import weak_ref_tensors``.
        # The two package stubs are installed/removed manually instead of
        # via ``monkeypatch.setitem``: this fixture pops the real ``vllm_fl``
        # entries *before* monkeypatch could snapshot them, so a setitem here
        # would record "absent" as the old value and monkeypatch's undo (which
        # runs after this fixture's finally) would DELETE the real modules
        # restored below, leaving later files with a fresh, attribute-less
        # ``vllm_fl.dispatch``. Restoring in the same finally that owns the
        # pop keeps the two halves of the swap strictly nested.
        graph_pkg = ModuleType("vllm_fl")
        graph_pkg.__path__ = []
        compilation_pkg = ModuleType("vllm_fl.compilation")
        compilation_pkg.__path__ = []
        sys.modules["vllm_fl"] = graph_pkg
        sys.modules["vllm_fl.compilation"] = compilation_pkg
        graph_pkg.compilation = compilation_pkg

        graph = _load_module(tree, _GRAPH_SOURCE, "vllm_fl.compilation.graph")
        params = _load_module(tree, _PARAMS_SOURCE, "vllm_fl.compilation.graph_params")
        runtime = _load_module(
            tree, _RUNTIME_SOURCE, "vllm_fl.compilation.graph_runtime"
        )
        yield SimpleNamespace(tree=tree, graph=graph, params=params, runtime=runtime)
    finally:
        tree.restore()
        sys.modules.pop("vllm_fl", None)
        sys.modules.pop("vllm_fl.compilation", None)
        # Remove every vllm_fl* entry added since the snapshot (the three
        # _load_module installs land on real package paths) and put back the
        # originals. Popping new entries matters as much as restoring old
        # ones: cuda_vendor's _fl_sys_modules snapshot fixture records "absent"
        # for entries its file did not import yet and would otherwise delete
        # real modules later files import — and any later
        # monkeypatch.setattr("vllm_fl.dispatch...") re-imports a fresh,
        # attribute-less vllm_fl.dispatch when the real one was dropped.
        for name in [
            name
            for name in list(sys.modules)
            if name == "vllm_fl" or name.startswith("vllm_fl.")
        ]:
            if name not in fl_roots:
                sys.modules.pop(name, None)
        for name, module in fl_roots.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


class _Descriptor:
    """Hashable BatchDescriptor stand-in keyed by num_tokens."""

    def __init__(self, num_tokens):
        self.num_tokens = num_tokens

    def __eq__(self, other):
        return isinstance(other, _Descriptor) and other.num_tokens == self.num_tokens

    def __hash__(self):
        return hash(("_Descriptor", self.num_tokens))

    def __repr__(self):
        return f"_Descriptor({self.num_tokens})"


# ---------------------------------------------------------------------------
# NPU weak references
# ---------------------------------------------------------------------------


class TestWeakRefNpuTensor:
    """``_weak_ref_npu_tensor`` single-tensor semantics."""

    def test_npu_tensor_uses_torch_npu_primitive(self, fl_tree, monkeypatch):
        """An NPU-device value is delegated to torch_npu._C._weak_ref_tensor."""
        received = []

        fake_npu = ModuleType("torch_npu")

        class _C:
            def _weak_ref_tensor(self, value):
                received.append(value)
                return ("weak", value)

        fake_npu._C = _C()
        monkeypatch.setitem(sys.modules, "torch_npu", fake_npu)

        value = _FakeVendorTensor("npu")
        result = fl_tree.graph._weak_ref_npu_tensor(value)
        assert result == ("weak", value)
        assert received == [value]


class TestWeakRefNpuTensors:
    """``_weak_ref_npu_tensors`` recursion mirrors the input structure."""

    @pytest.fixture(autouse=True)
    def _torch_npu_stub(self, monkeypatch):
        """Install a working torch_npu primitive for every test below."""
        fake_npu = ModuleType("torch_npu")

        class _C:
            def _weak_ref_tensor(self, value):
                return ("weak", value)

        fake_npu._C = _C()
        monkeypatch.setitem(sys.modules, "torch_npu", fake_npu)

    def test_scalars_and_unknown_objects_pass_through(self, fl_tree):
        """Non-container leaves are returned unchanged."""
        sentinel = object()
        for value in (7, "x", 3.5, None, sentinel):
            assert fl_tree.graph._weak_ref_npu_tensors(value) is value


# ---------------------------------------------------------------------------
# GCU weak references
# ---------------------------------------------------------------------------


class TestWeakRefGcuTensor:
    """``_weak_ref_gcu_tensor`` vendor reconstruction semantics."""

    @pytest.fixture(autouse=True)
    def _torch_gcu_stub(self, monkeypatch):
        """Install a recording torch_gcu and patch the torch storage entry point.

        ``torch._C._construct_storage_from_data_pointer`` is an external
        boundary (a vendor-specific PyTorch extension point); it is replaced
        for the test and restored afterwards.  The fake vendor constructor
        returns a real complex CPU tensor so the conj/neg flag setters can
        run against a tensor type that actually supports them.
        """
        self.constructed = []
        fake_gcu = ModuleType("torch_gcu")
        constructed = self.constructed

        class _C:
            def _construct_GCU_Tensor_From_Storage_And_Metadata(
                self, metadata, storage
            ):
                constructed.append((dict(metadata), storage))
                return torch.empty(3, dtype=torch.complex64)

        fake_gcu._C = _C()
        monkeypatch.setitem(sys.modules, "torch_gcu", fake_gcu)

        original = torch._C._construct_storage_from_data_pointer
        self.storages = []

        def fake_construct_storage(data_ptr, device, nbytes):
            storage = SimpleNamespace(
                data_ptr=lambda: data_ptr, nbytes=lambda: nbytes, device=device
            )
            self.storages.append(storage)
            return storage

        torch._C._construct_storage_from_data_pointer = fake_construct_storage
        yield
        torch._C._construct_storage_from_data_pointer = original

    def test_metadata_and_borrowed_storage_construction(self, fl_tree):
        """The vendor metadata dict mirrors the tensor's storage facts."""
        value = _FakeVendorTensor(
            "gcu",
            shape=(4, 6),
            stride=(6, 1),
            storage_offset=3,
            nbytes=96,
            data_ptr=4242,
        )
        result = fl_tree.graph._weak_ref_gcu_tensor(value)

        assert result is not value
        metadata, storage = self.constructed[0]
        assert metadata == {
            "nbytes": 96,
            "data_ptr": 4242,
            "size": (4, 6),
            "stride": (6, 1),
            "dtype": torch.float32,
            "device": value.device,
            "storage_offset": 3,
        }
        # The borrowed storage is built from the same pointer/size pair.
        assert storage.data_ptr() == 4242
        assert storage.nbytes() == 96
        assert storage in self.storages

    def test_conj_and_neg_flags_are_propagated(self, fl_tree):
        """Conjugated/negated inputs flip the same flags on the rebuild."""
        value = _FakeVendorTensor("gcu", conj=True, neg=True)
        result = fl_tree.graph._weak_ref_gcu_tensor(value)
        assert result.is_conj() is True
        assert result.is_neg() is True

    def test_plain_tensor_leaves_flags_clear(self, fl_tree):
        """Without conj/neg the rebuilt tensor keeps both flags unset."""
        result = fl_tree.graph._weak_ref_gcu_tensor(_FakeVendorTensor("gcu"))
        assert result.is_conj() is False
        assert result.is_neg() is False

    def test_storage_constructor_absent_raises_runtime_error(
        self, fl_tree, monkeypatch
    ):
        """Missing torch._C storage constructor is also a hard error."""
        fake_gcu = ModuleType("torch_gcu")

        class _C:
            def _construct_GCU_Tensor_From_Storage_And_Metadata(
                self, metadata, storage
            ):
                return torch.empty(1, dtype=torch.complex64)

        fake_gcu._C = _C()
        monkeypatch.setitem(sys.modules, "torch_gcu", fake_gcu)
        original = torch._C._construct_storage_from_data_pointer
        del torch._C._construct_storage_from_data_pointer
        try:
            with pytest.raises(RuntimeError, match="TorchGCU storage and tensor"):
                fl_tree.graph._weak_ref_gcu_tensor(_FakeVendorTensor("gcu"))
        finally:
            torch._C._construct_storage_from_data_pointer = original


class TestWeakRefGcuTensors:
    """``_weak_ref_gcu_tensors`` recursion mirrors the input structure."""

    @pytest.fixture(autouse=True)
    def _torch_gcu_stub(self, monkeypatch):
        """Install a working torch_gcu primitive for the recursion tests."""
        fake_gcu = ModuleType("torch_gcu")

        class _C:
            def _construct_GCU_Tensor_From_Storage_And_Metadata(
                self, metadata, storage
            ):
                return torch.empty(2, dtype=torch.complex64)

        fake_gcu._C = _C()
        monkeypatch.setitem(sys.modules, "torch_gcu", fake_gcu)
        original = torch._C._construct_storage_from_data_pointer
        torch._C._construct_storage_from_data_pointer = (
            lambda data_ptr, device, nbytes: SimpleNamespace(
                data_ptr=lambda: data_ptr, nbytes=lambda: nbytes
            )
        )
        yield
        torch._C._construct_storage_from_data_pointer = original

    def test_recursion_over_containers_and_intermediate_tensors(self, fl_tree):
        """Container types survive; GCU members are rebuilt; CPU is kept."""
        cpu = torch.arange(1.0)
        gcu = _FakeVendorTensor("gcu")
        payload = [cpu, (gcu, {"keep": cpu, "swap": gcu})]

        result = fl_tree.graph._weak_ref_gcu_tensors(payload)

        assert isinstance(result, list) and result[0] is cpu
        weak_item, mapping = result[1]
        assert isinstance(weak_item, torch.Tensor)  # rebuilt by the fake
        assert mapping["keep"] is cpu
        assert isinstance(mapping["swap"], torch.Tensor)

        rebuilt = fl_tree.graph._weak_ref_gcu_tensors(
            _FakeIntermediateTensors({"a": gcu, "b": cpu})
        )
        assert isinstance(rebuilt, _FakeIntermediateTensors)
        assert rebuilt.tensors["b"] is cpu
        assert isinstance(rebuilt.tensors["a"], torch.Tensor)

    def test_scalars_pass_through(self, fl_tree):
        """Leaves that are not tensors/containers are unchanged."""
        assert fl_tree.graph._weak_ref_gcu_tensors("s") == "s"
        assert fl_tree.graph._weak_ref_gcu_tensors(None) is None


# ---------------------------------------------------------------------------
# weak_ref_tensors dispatch
# ---------------------------------------------------------------------------


class TestWeakRefTensorsDispatch:
    """``weak_ref_tensors`` picks the helper from current_platform."""

    def test_cuda_iluvatar_delegates_to_iluvatar_module(self, fl_tree):
        """cuda + iluvatar vendor re-exports the vendor helper.

        The five ``vllm_fl.dispatch*`` entries are installed and removed
        manually instead of via ``monkeypatch.setitem``: fl_tree pops those
        names during setup (so setitem would record "absent") and restores
        the originals in its own finally, which runs BEFORE monkeypatch's
        undo — the undo would then delete the just-restored real modules and
        every later file would import a fresh, attribute-less
        ``vllm_fl.dispatch``.
        """
        ilu_graph = ModuleType("vllm_fl.dispatch.backends.vendor.iluvatar.graph")
        ilu_graph._weak_ref_iluvatar_tensors = lambda value: ("ilu", value)
        names = (
            "vllm_fl.dispatch",
            "vllm_fl.dispatch.backends",
            "vllm_fl.dispatch.backends.vendor",
            "vllm_fl.dispatch.backends.vendor.iluvatar",
            "vllm_fl.dispatch.backends.vendor.iluvatar.graph",
        )
        saved = {name: sys.modules.get(name) for name in names}
        try:
            for name in names:
                sys.modules[name] = ilu_graph
            sys.modules["vllm_fl.dispatch.backends.vendor"] = ModuleType("x")

            fl_tree.tree.set_platform("cuda", "iluvatar")
            value = torch.arange(2.0)
            assert fl_tree.graph.weak_ref_tensors(value) == ("ilu", value)
        finally:
            for name, module in saved.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module

    def test_npu_platform_routes_to_npu_family(self, fl_tree, monkeypatch):
        """device_type npu uses _weak_ref_npu_tensors."""
        fake_npu = ModuleType("torch_npu")

        class _C:
            def _weak_ref_tensor(self, value):
                return ("weak", value)

        fake_npu._C = _C()
        monkeypatch.setitem(sys.modules, "torch_npu", fake_npu)
        fl_tree.tree.set_platform("npu", "ascend")

        npu_tensor = _FakeVendorTensor("npu")
        result = fl_tree.graph.weak_ref_tensors([npu_tensor])
        assert result == [("weak", npu_tensor)]

    def test_gcu_platform_routes_to_gcu_family(self, fl_tree, monkeypatch):
        """device_type gcu uses _weak_ref_gcu_tensors."""
        fake_gcu = ModuleType("torch_gcu")

        class _C:
            def _construct_GCU_Tensor_From_Storage_And_Metadata(
                self, metadata, storage
            ):
                return torch.empty(2, dtype=torch.complex64)

        fake_gcu._C = _C()
        monkeypatch.setitem(sys.modules, "torch_gcu", fake_gcu)
        original = torch._C._construct_storage_from_data_pointer
        torch._C._construct_storage_from_data_pointer = lambda p, d, n: SimpleNamespace(
            data_ptr=lambda: p, nbytes=lambda: n
        )
        try:
            fl_tree.tree.set_platform("gcu", "enflame")
            cpu = torch.arange(2.0)
            assert fl_tree.graph.weak_ref_tensors(cpu) is cpu
        finally:
            torch._C._construct_storage_from_data_pointer = original

    def test_other_platforms_use_upstream_helper(self, fl_tree):
        """Unknown device types fall back to vLLM's weak_ref_tensors."""
        fl_tree.tree.set_platform("cuda", "nvidia")
        value = torch.arange(3.0)
        assert fl_tree.graph.weak_ref_tensors(value) == ("upstream", value)

    def test_upstream_failure_returns_value_unchanged(self, fl_tree):
        """If the upstream import/call fails the value passes through."""

        def boom(value):
            raise RuntimeError("upstream unavailable")

        fl_tree.tree.set_platform("rocm", "amd")
        fl_tree.tree.set_upstream_weak_ref(boom)
        value = torch.arange(3.0)
        assert fl_tree.graph.weak_ref_tensors(value) is value


# ---------------------------------------------------------------------------
# GraphWrapper helpers and capture
# ---------------------------------------------------------------------------


class _FakeGraph:
    """Graph object stand-in recording replay calls."""

    def __init__(self):
        self.replays = 0

    def replay(self):
        self.replays += 1


class TestGraphWrapperHelpers:
    """Registry helpers on GraphWrapper that upstream tests never touch."""

    @pytest.fixture
    def wrapper(self, fl_tree):
        """Build a GraphWrapper with a fake graph type (CPU has no CUDAGraph)."""
        fl_tree.graph.GraphWrapper._graph_type = _FakeGraph

        class _Runnable:
            marker = "delegated"

            def __call__(self, *args, **kwargs):
                return torch.arange(3.0)

        class _Config:
            compilation_config = SimpleNamespace()

        mode = SimpleNamespace(name="FULL")
        instance = fl_tree.graph.GraphWrapper(
            _Runnable(), _Config(), mode, cudagraph_options=fl_tree.graph.GraphOptions()
        )
        return instance

    def test_unwrap_returns_original_runnable(self, wrapper):
        """unwrap() exposes the wrapped runnable directly."""
        runnable = wrapper.unwrap()
        assert callable(runnable)
        assert runnable.marker == "delegated"

    def test_cudagraph_wrapper_property_returns_self(self, wrapper):
        """The cudagraph_wrapper compatibility property is identity."""
        assert wrapper.cudagraph_wrapper is wrapper

    def test_clear_all_graphs_clears_every_instance(self, wrapper, fl_tree):
        """clear_all_graphs clears entries across all live wrappers."""
        wrapper.concrete_graph_entries[_Descriptor(4)] = "entry"
        wrapper.concrete_graph_entries[_Descriptor(8)] = "entry2"
        assert len(wrapper.concrete_graph_entries) == 2

        fl_tree.graph.GraphWrapper.clear_all_graphs()

        assert wrapper.concrete_graph_entries == {}

    def test_capture_uses_platform_pool_handle_when_pool_is_none(
        self, fl_tree, monkeypatch
    ):
        """With graph_pool None, set_graph_pool_id receives the platform handle."""
        fl_tree.tree.set_platform("cuda", "nvidia")
        fl_tree.tree.platform.get_global_graph_pool = lambda: None
        fl_tree.graph.GraphWrapper._graph_type = _FakeGraph

        class _Runnable:
            def __call__(self, *args, **kwargs):
                return torch.arange(3.0)

        class _Config:
            compilation_config = SimpleNamespace()

        wrapper = fl_tree.graph.GraphWrapper(
            _Runnable(),
            _Config(),
            SimpleNamespace(name="FULL"),
            cudagraph_options=fl_tree.graph.GraphOptions(),
        )
        assert wrapper.graph_pool is None
        fl_tree.tree.calls.clear()

        entry = fl_tree.graph.GraphEntry(batch_descriptor=_Descriptor(2))
        wrapper._capture_graph(entry, (torch.arange(2.0),), {})

        assert fl_tree.tree.calls == ["stub-pool-handle"]
        assert isinstance(entry.graph, _FakeGraph)
        assert fl_tree.tree.counter.num_cudagraph_captured == 1

    def test_capture_with_pool_uses_wrapper_pool(self, fl_tree):
        """A non-None graph pool is forwarded instead of the platform handle."""
        fl_tree.tree.set_platform("cuda", "nvidia")
        fl_tree.graph.GraphWrapper._graph_type = _FakeGraph

        class _Runnable:
            def __call__(self, *args, **kwargs):
                return torch.arange(3.0)

        class _Config:
            compilation_config = SimpleNamespace()

        wrapper = fl_tree.graph.GraphWrapper(
            _Runnable(),
            _Config(),
            SimpleNamespace(name="FULL"),
            cudagraph_options=fl_tree.graph.GraphOptions(),
        )
        fl_tree.tree.calls.clear()

        entry = fl_tree.graph.GraphEntry(batch_descriptor=_Descriptor(2))
        wrapper._capture_graph(entry, (), {})

        assert fl_tree.tree.calls == [wrapper.graph_pool]


# ---------------------------------------------------------------------------
# GraphParams
# ---------------------------------------------------------------------------


class TestGraphParams:
    """Validation and lifecycle of the per-descriptor capture registry."""

    def test_prepare_capture_rejects_non_positive_sizes(self, fl_tree):
        """num_tokens must be a positive int; bools are rejected too."""
        params = fl_tree.params.GraphParams()
        for bad in (0, -1, 2.5, "8", None, True, False):
            with pytest.raises(ValueError, match="must be positive"):
                params.prepare_capture(_Descriptor(bad))

    def test_prepare_capture_initializes_empty_state(self, fl_tree):
        """A valid descriptor gets empty lists and a None workspace."""
        params = fl_tree.params.GraphParams()
        descriptor = _Descriptor(16)
        params.prepare_capture(descriptor)
        assert params.events[descriptor] == []
        assert params.workspaces[descriptor] is None
        assert params.handles[descriptor] == []
        assert params.attn_params[descriptor] == []

    def test_require_capture_rejects_unknown_descriptor(self, fl_tree):
        """require_capture fails for a never-prepared descriptor."""
        params = fl_tree.params.GraphParams()
        with pytest.raises(RuntimeError, match="were not prepared"):
            params.require_capture(_Descriptor(4))

    def test_require_capture_accepts_prepared_descriptor(self, fl_tree):
        """require_capture passes silently once prepared."""
        params = fl_tree.params.GraphParams()
        descriptor = _Descriptor(4)
        params.prepare_capture(descriptor)
        params.require_capture(descriptor)

    def test_clear_empties_all_registries(self, fl_tree):
        """clear() drops events, workspaces, handles and attn_params."""
        params = fl_tree.params.GraphParams()
        descriptor = _Descriptor(4)
        params.prepare_capture(descriptor)
        params.workspaces[descriptor] = object()
        params.clear()
        assert params.events == {}
        assert params.workspaces == {}
        assert params.handles == {}
        assert params.attn_params == {}

    def test_module_level_registry_helpers(self, fl_tree):
        """get/prepare/clear operate on one shared singleton."""
        singleton = fl_tree.params.get_graph_params()
        assert fl_tree.params._GRAPH_PARAMS is singleton

        descriptor = _Descriptor(6)
        fl_tree.params.prepare_graph_params(descriptor)
        assert descriptor in fl_tree.params.get_graph_params().attn_params

        fl_tree.params.clear_graph_params()
        assert fl_tree.params.get_graph_params().attn_params == {}


# ---------------------------------------------------------------------------
# GraphRuntimeController
# ---------------------------------------------------------------------------


class TestGraphRuntimeController:
    """Capture/replay orchestration around the parameter registry."""

    @pytest.fixture(autouse=True)
    def _clean_registry(self):
        """Nothing to clean before the module-level registry exists per load."""
        yield

    def test_batch_descriptor_rejects_non_int_tokens(self, fl_tree):
        """Non-int (or bool) num_tokens raises before any state changes."""
        controller = fl_tree.runtime.GraphRuntimeController(vllm_config="cfg")
        for bad in (2.5, "8", None, True):
            context = SimpleNamespace(batch_descriptor=SimpleNamespace(num_tokens=bad))
            with pytest.raises(RuntimeError, match="integer BatchDescriptor"):
                controller._batch_descriptor(context)

    def test_batch_descriptor_returns_descriptor(self, fl_tree):
        """A valid context yields its descriptor unchanged."""
        controller = fl_tree.runtime.GraphRuntimeController(vllm_config="cfg")
        context = SimpleNamespace(batch_descriptor=SimpleNamespace(num_tokens=7))
        assert controller._batch_descriptor(context).num_tokens == 7

    def test_capture_scope_weak_ref_failure_still_restores(self, fl_tree):
        """A failing weak_ref_workspace still clears capturing/_active."""
        controller = fl_tree.runtime.GraphRuntimeController(vllm_config="cfg")
        context = SimpleNamespace(batch_descriptor=_Descriptor(9), capturing=False)

        original = fl_tree.params.weak_ref_workspace

        def boom(descriptor, weak_ref):
            raise RuntimeError("weak ref failed")

        fl_tree.params.weak_ref_workspace = boom
        try:
            with (
                pytest.raises(RuntimeError, match="weak ref failed"),
                controller.capture_scope(context),
            ):
                assert context.capturing is True
        finally:
            fl_tree.params.weak_ref_workspace = original

        assert context.capturing is False
        assert controller._active is False

    def test_capture_scope_rejects_reentrancy(self, fl_tree):
        """Entering capture_scope while active trips the assert."""
        controller = fl_tree.runtime.GraphRuntimeController(vllm_config="cfg")
        context = SimpleNamespace(batch_descriptor=_Descriptor(3), capturing=False)
        with controller.capture_scope(context):
            nested = SimpleNamespace(batch_descriptor=_Descriptor(4), capturing=False)
            with (
                pytest.raises(AssertionError),
                controller.capture_scope(nested),
            ):
                pass

    def test_update_attention_requires_bound_backends(self, fl_tree):
        """Calling _update_attention before binding raises."""
        controller = fl_tree.runtime.GraphRuntimeController(vllm_config="cfg")
        controller.attention_impls = None
        with pytest.raises(RuntimeError, match="backends must be bound"):
            controller._update_attention(
                SimpleNamespace(batch_descriptor=_Descriptor(5))
            )

    def test_update_attention_creates_stream_once(self, fl_tree):
        """The update Stream is created lazily and reused across calls."""
        streams = []

        class _Stream:
            pass

        def make_stream():
            stream = _Stream()
            streams.append(stream)
            return stream

        fl_tree.tree.platform.torch_device_fn = SimpleNamespace(
            current_stream=lambda: SimpleNamespace(synchronize=lambda: None),
            synchronize=lambda: None,
            Stream=make_stream,
        )

        calls = []

        class _Impl:
            def update_graph_params(self, stream, context, descriptor, cfg):
                calls.append(stream)

        controller = fl_tree.runtime.GraphRuntimeController(vllm_config="cfg")
        controller.attention_impls = (_Impl(), _Impl())
        context = SimpleNamespace(batch_descriptor=_Descriptor(5))
        controller._update_attention(context)
        controller._update_attention(context)
        assert len(streams) == 1
        assert calls == [streams[0]] * 4

    def test_bind_attention_groups_rejects_unupdatable_impl(self, fl_tree):
        """An impl lacking update_graph_params is a hard error."""

        class _BadBackend:
            @staticmethod
            def get_impl_cls():
                return object()  # the impl itself, without the updater

        class _Group:
            def __init__(self, backend):
                self.backend = backend

        controller = fl_tree.runtime.GraphRuntimeController(vllm_config="cfg")
        with pytest.raises(RuntimeError, match="lacks update_graph_params"):
            controller.bind_attention_groups([[_Group(_BadBackend)]])

    def test_bind_attention_groups_wraps_factory_errors(self, fl_tree):
        """Factory exceptions surface as RuntimeError naming the backend."""

        class _BoomBackend:
            @staticmethod
            def get_impl_cls():
                raise ImportError("vendor module missing")

        class _Group:
            def __init__(self, backend):
                self.backend = backend

        controller = fl_tree.runtime.GraphRuntimeController(vllm_config="cfg")
        with pytest.raises(RuntimeError, match="Failed to resolve"):
            controller.bind_attention_groups([[_Group(_BoomBackend)]])

    def test_clear_decoder_graphs_clears_registry_and_wrappers(self, fl_tree):
        """clear_decoder_graphs drops graph entries and capture params."""
        fl_tree.graph.GraphWrapper._graph_type = _FakeGraph

        class _Runnable:
            def __call__(self):
                pass

        class _Config:
            compilation_config = SimpleNamespace()

        wrapper = fl_tree.graph.GraphWrapper(
            _Runnable(),
            _Config(),
            SimpleNamespace(name="FULL"),
            cudagraph_options=fl_tree.graph.GraphOptions(),
        )
        wrapper.concrete_graph_entries[_Descriptor(2)] = "entry"

        fl_tree.params.clear_graph_params()
        descriptor = _Descriptor(5)
        fl_tree.params.prepare_graph_params(descriptor)

        controller = fl_tree.runtime.GraphRuntimeController(
            vllm_config="cfg", full_graph_wrapper_type=fl_tree.graph.GraphWrapper
        )
        controller.clear_decoder_graphs()

        assert wrapper.concrete_graph_entries == {}
        assert fl_tree.params.get_graph_params().attn_params == {}

    def test_clear_decoder_graphs_without_wrapper_type(self, fl_tree):
        """A None wrapper_type still clears the parameter registry."""
        fl_tree.params.clear_graph_params()
        fl_tree.params.prepare_graph_params(_Descriptor(1))
        controller = fl_tree.runtime.GraphRuntimeController(vllm_config="cfg")
        controller.clear_decoder_graphs()
        assert fl_tree.params.get_graph_params().attn_params == {}
