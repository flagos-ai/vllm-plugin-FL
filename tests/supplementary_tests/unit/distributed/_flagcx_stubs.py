# Copyright (c) 2026 BAAI. All rights reserved.
"""Shared CPU stubs for testing ``vllm_fl.distributed.kv_transfer`` modules.

The modules under test import a large surface of vLLM internals plus the
FlagCX wrapper (``plugin.interservice.flagcx_wrapper``) at import time.  None
of those are installed in the CPU test environment.  This helper installs the
*minimum* leaf modules in ``sys.modules`` via ``monkeypatch.setitem`` so that
``vllm_fl.distributed.kv_transfer.flagcx_connector`` / ``...flagcx_stats``
execute their real code, with only the external boundaries stubbed:

* every vLLM import used by the connector (``vllm.config``, parallel_state,
  kv_connector framework, attention/kv-cache spec classes, ...);
* ``plugin.interservice.flagcx_wrapper.FLAGCXLibrary`` — the native FlagCX
  library boundary (mocked with :class:`unittest.mock.Mock`);
* ``torch.distributed`` is never touched: the connector only calls
  ``get_tensor_model_parallel_rank`` / ``get_tensor_model_parallel_world_size``
  from ``vllm.distributed.parallel_state``, which we stub directly.

The spec classes are real (empty) classes rather than ``object`` aliases, so
``isinstance`` dispatch inside the module under test (FullAttentionSpec vs
SlidingWindowSpec vs MambaSpec ...) behaves exactly as with real vLLM.
"""

import logging
import sys
from dataclasses import dataclass, field
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

# ---------------------------------------------------------------------------
# Real (empty) spec classes: isinstance dispatch in the module under test must
# distinguish them, so they cannot be `object` aliases.
# ---------------------------------------------------------------------------


class FullAttentionSpec:
    """Stub of vllm.v1.kv_cache_interface.FullAttentionSpec."""


class SlidingWindowSpec:
    """Stub of vllm.v1.kv_cache_interface.SlidingWindowSpec."""


class MambaSpec:
    """Stub of vllm.v1.kv_cache_interface.MambaSpec."""


class MLAAttentionSpec:
    """Stub of vllm.v1.kv_cache_interface.MLAAttentionSpec."""


class SlidingWindowMLASpec:
    """Stub of vllm.v1.kv_cache_interface.SlidingWindowMLASpec."""


class KVCacheSpec:
    """Stub of vllm.v1.kv_cache_interface.KVCacheSpec."""


@dataclass
class KVConnectorStats:
    """Stub of the vLLM KVConnectorStats dataclass base.

    Mirrors the real base: a single ``data`` dict field so the FlagCX
    subclass's ``@dataclass`` decoration and ``__post_init__`` run unchanged.
    """

    data: dict = field(default_factory=dict)

    def is_empty(self) -> bool:
        return not any(self.data.values())


class _OnceLogger:
    """Logger wrapper providing the ``*_once`` helpers vLLM loggers have."""

    def __init__(self, name):
        self._log = logging.getLogger(name)

    def __getattr__(self, item):
        return getattr(self._log, item)

    def warning_once(self, *args, **kwargs):
        self._log.warning(*args, **kwargs)

    def info_once(self, *args, **kwargs):
        self._log.info(*args, **kwargs)


def _leaf(name, **attrs):
    """Install a single leaf module in sys.modules (no parent handling)."""
    module = ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def _chain(name, **attrs):
    """Install a leaf module and any missing parent packages in sys.modules."""
    parts = name.split(".")
    for i in range(1, len(parts)):
        parent = ".".join(parts[:i])
        if parent not in sys.modules:
            sys.modules[parent] = ModuleType(parent)
    return _leaf(name, **attrs)


class FlagcxStubEnv:
    """Install/restore the whole stub environment around one import."""

    def __init__(self, monkeypatch, flagcx_library=None):
        self._monkeypatch = monkeypatch
        self.flagcx_lib = (
            flagcx_library if flagcx_library is not None else Mock(name="FLAGCXLibrary")
        )
        self.flagcx_lib.flagcxP2pEngineCreate.return_value = "engine-handle"
        self.flagcx_lib.flagcxP2pGetRpcPort.return_value = 12345

    def install(self):
        """Put every stub leaf into sys.modules via monkeypatch.setitem.

        The modules under test are (re)imported per test so their import-time
        bindings (FLAGCXLibrary, get_current_attn_backends, ...) always refer
        to *this* environment's objects, never to a previous test's stubs.
        """
        mp = self._monkeypatch
        setitem = mp.setitem

        # Drop previously imported modules under test (and the whole
        # vllm_fl.distributed package chain) so each test re-imports them
        # against this env's stubs. monkeypatch restores the previous entries
        # on teardown, keeping the suite order-independent.
        for name in list(sys.modules):
            if name == "vllm_fl.distributed" or name.startswith("vllm_fl.distributed."):
                setitem(sys.modules, name, None)
                sys.modules.pop(name, None)

        class KVConnectorBase_V1:
            def __init__(self, vllm_config, role, kv_cache_config=None):
                self._connector_metadata = None

        # The native FlagCX boundary.
        setitem(
            sys_modules_target(),
            "plugin.interservice.flagcx_wrapper",
            _leaf(
                "plugin.interservice.flagcx_wrapper",
                FLAGCXLibrary=lambda path: self.flagcx_lib,
            ),
        )
        for parent in ("plugin", "plugin.interservice"):
            setitem(
                sys_modules_target(),
                parent,
                sys.modules.get(parent, ModuleType(parent)),
            )

        # vLLM leaves used at import time by flagcx_connector / flagcx_stats.
        setitem(
            sys_modules_target(),
            "vllm.config",
            _leaf("vllm.config", VllmConfig=object),
        )
        utils_mod = _chain(
            "vllm.distributed.kv_transfer.kv_connector.utils",
            TransferTopology=_StubTransferTopology,
            get_current_attn_backends=lambda cfg: [],
        )
        setitem(
            sys_modules_target(),
            "vllm.distributed.kv_transfer.kv_connector.utils",
            utils_mod,
        )
        setitem(
            sys_modules_target(),
            "vllm.distributed.kv_transfer.kv_connector.v1.base",
            _chain(
                "vllm.distributed.kv_transfer.kv_connector.v1.base",
                KVConnectorBase_V1=KVConnectorBase_V1,
                KVConnectorMetadata=object,
                KVConnectorRole=SimpleNamespace(SCHEDULER="scheduler", WORKER="worker"),
                SupportsHMA=object,
            ),
        )
        setitem(
            sys_modules_target(),
            "vllm.distributed.kv_transfer.kv_connector.v1.metrics",
            _chain(
                "vllm.distributed.kv_transfer.kv_connector.v1.metrics",
                KVConnectorPromMetrics=_StubPromMetricsBase,
                KVConnectorStats=KVConnectorStats,
                PromMetric=object,
                PromMetricT=object,
            ),
        )
        setitem(
            sys_modules_target(),
            "vllm.distributed.parallel_state",
            _chain(
                "vllm.distributed.parallel_state",
                get_pp_group=lambda: None,
                get_tensor_model_parallel_rank=lambda: 0,
                get_tensor_model_parallel_world_size=lambda: 1,
            ),
        )
        setitem(
            sys_modules_target(),
            "vllm.forward_context",
            _leaf("vllm.forward_context", ForwardContext=object),
        )
        setitem(
            sys_modules_target(),
            "vllm.logger",
            _leaf("vllm.logger", init_logger=_OnceLogger),
        )
        setitem(
            sys_modules_target(),
            "vllm.model_executor.models.utils",
            _leaf(
                "vllm.model_executor.models.utils",
                extract_layer_index=_extract_layer_index,
            ),
        )
        setitem(
            sys_modules_target(),
            "vllm.utils.math_utils",
            _leaf("vllm.utils.math_utils", cdiv=lambda a, b: -(-a // b)),
        )
        setitem(
            sys_modules_target(),
            "vllm.utils.network_utils",
            _leaf(
                "vllm.utils.network_utils",
                get_ip=lambda: "10.0.0.1",
                make_zmq_path=lambda t, h, p: f"{t}://{h}:{p}",
                make_zmq_socket=lambda *a, **k: None,
            ),
        )
        setitem(
            sys_modules_target(),
            "vllm.v1.attention.backend",
            _leaf("vllm.v1.attention.backend", AttentionMetadata=object),
        )
        setitem(
            sys_modules_target(),
            "vllm.v1.attention.backends.utils",
            _leaf(
                "vllm.v1.attention.backends.utils",
                NULL_BLOCK_ID=-1,
                get_kv_cache_layout=lambda: "NHD",
            ),
        )
        setitem(
            sys_modules_target(),
            "vllm.v1.core.sched.output",
            _leaf("vllm.v1.core.sched.output", SchedulerOutput=object),
        )
        setitem(
            sys_modules_target(),
            "vllm.v1.kv_cache_interface",
            _leaf(
                "vllm.v1.kv_cache_interface",
                FullAttentionSpec=FullAttentionSpec,
                KVCacheSpec=KVCacheSpec,
                MambaSpec=MambaSpec,
                MLAAttentionSpec=MLAAttentionSpec,
                SlidingWindowMLASpec=SlidingWindowMLASpec,
                SlidingWindowSpec=SlidingWindowSpec,
            ),
        )
        setitem(
            sys_modules_target(),
            "vllm.v1.request",
            _leaf(
                "vllm.v1.request",
                RequestStatus=SimpleNamespace(FINISHED_LENGTH_CAPPED=3),
            ),
        )
        setitem(
            sys_modules_target(),
            "vllm.v1.worker.block_table",
            _leaf("vllm.v1.worker.block_table", BlockTable=object),
        )
        setitem(
            sys_modules_target(),
            "vllm.v1.worker.utils",
            _leaf(
                "vllm.v1.worker.utils",
                select_common_block_size=lambda bs, backends: bs,
            ),
        )
        setitem(
            sys_modules_target(),
            "vllm.v1.metrics.utils",
            _leaf(
                "vllm.v1.metrics.utils",
                create_metric_per_engine=_fan_out_metric_per_engine,
            ),
        )
        return self


def sys_modules_target():
    """sys.modules itself (monkeypatch.setitem target)."""
    return sys.modules


def _extract_layer_index(layer_name: str) -> int:
    """Parse ``model.layers.<N>...`` exactly like vLLM's helper."""
    parts = layer_name.split(".")
    for i, part in enumerate(parts):
        if part == "layers" and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                return -1
    return -1


def _fan_out_metric_per_engine(metric, per_engine):
    """Fan a metric object out to every engine, like vLLM's helper."""
    return {engine: metric for engine in per_engine}


class _StubPromMetricsBase:
    """Stub of KVConnectorPromMetrics: records constructor args only."""

    def __init__(self, vllm_config, metric_types, labelnames, per_engine_labelvalues):
        self.vllm_config = vllm_config
        self.metric_types = metric_types
        self.labelnames = labelnames
        self.per_engine_labelvalues = dict(per_engine_labelvalues)
        self._counter_cls = _CountingCounter
        self._histogram_cls = _CountingHistogram


class _CountingCounter:
    """Minimal stand-in for a prometheus Counter; records inc() values."""

    instances = []

    def __init__(self, name, documentation, labelnames):
        self.name = name
        self.documentation = documentation
        self.labelnames = labelnames
        self.values = []
        _CountingCounter.instances.append(self)

    def inc(self, value):
        self.values.append(value)


class _CountingHistogram:
    """Minimal stand-in for a prometheus Histogram; records observe()."""

    instances = []

    def __init__(self, name, documentation, buckets, labelnames):
        self.name = name
        self.documentation = documentation
        self.buckets = buckets
        self.labelnames = labelnames
        self.values = []
        _CountingHistogram.instances.append(self)

    def observe(self, value):
        self.values.append(value)


class _StubTransferTopology:
    """CPU double of vllm's TransferTopology (attribute carrier)."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.local_replicates_kv_cache = False
        self.is_kv_layout_blocks_first = True

    def handshake_target_ranks(self, remote_tp_size):
        return [0]


# ---------------------------------------------------------------------------
# Config doubles for constructing the connector classes.
# ---------------------------------------------------------------------------


class _Group:
    def __init__(self, spec, layer_names):
        self.kv_cache_spec = spec
        self.layer_names = layer_names


class _SlidingSpec(SlidingWindowSpec):
    def __init__(self, sliding_window, block_size):
        self.sliding_window = sliding_window
        self.block_size = block_size


class _FullSpec(FullAttentionSpec):
    pass


class _ModelConfig:
    use_mla = False

    def get_total_num_kv_heads(self):
        return 8


class _CacheConfig:
    block_size = 16


class _ParallelConfig:
    def __init__(self, dp_rank=0, tp_size=1):
        self.data_parallel_rank = dp_rank
        self.tensor_parallel_size = tp_size


class _SchedulerConfig:
    def __init__(self, disable_hybrid_kv_cache_manager=True):
        self.disable_hybrid_kv_cache_manager = disable_hybrid_kv_cache_manager


class VllmConfigDouble:
    """Minimal VllmConfig double covering both connector roles."""

    def __init__(
        self,
        role="kv_both",
        dp_rank=0,
        tp_size=1,
        extra_config=None,
        num_kv_heads=None,
        disable_hybrid_kv_cache_manager=True,
    ):
        self.kv_transfer_config = SimpleNamespace(
            engine_id="e0",
            kv_role=role,
            kv_connector_extra_config=extra_config if extra_config is not None else {},
        )
        self.parallel_config = _ParallelConfig(dp_rank, tp_size)
        self.scheduler_config = _SchedulerConfig(disable_hybrid_kv_cache_manager)
        self.cache_config = _CacheConfig()
        self.model_config = _ModelConfig()
        if num_kv_heads is not None:
            self.model_config.get_total_num_kv_heads = lambda n=num_kv_heads: n


def make_kv_cache_config(
    groups=None,
    has_mamba=False,
    sliding_window=None,
    block_size=16,
):
    """Build a KVCacheConfig double; one full-attention group by default."""
    if groups is None:
        groups = [_Group(_FullSpec(), ["model.layers.0.self_attn.attn"])]
    return SimpleNamespace(
        kv_cache_groups=groups,
        has_mamba_layers=has_mamba,
    )
