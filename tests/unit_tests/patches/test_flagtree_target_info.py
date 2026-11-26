# Copyright (c) 2026 BAAI. All rights reserved.

"""CPU tests for the FlagTree ``target_info`` AMD predicate patches.

``vllm_fl/patches/flagtree_target_info.py`` adds AMD architecture predicates
(``is_hip_gfx1250``, ``is_hip_cdna*``, ``is_hip_rdna``, ``is_hip_gfx11/12``)
that vLLM 0.32's bundled Triton kernels import but FlagTree's
``triton.language.target_info`` does not define. Each predicate must:

* return False when the target is not HIP (regardless of arch);
* return True only when the live gfx arch starts with one of the predicate's
  prefixes;
* leave attributes FlagTree already defines untouched;
* be idempotent (``lru_cache``) and report unavailability when Triton cannot
  be imported.

The module is loaded per-test via ``importlib`` with a stubbed
``vllm.logger`` (its only vllm dependency), and the FlagTree-like
``target_info`` module is faked in ``sys.modules`` so the predicate truth
table can be checked against every arch family without a real AMD GPU.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

_SOURCE = (
    Path(__file__).resolve().parents[3] / "vllm_fl/patches/flagtree_target_info.py"
)
_MODULE_NAME = "supp_flagtree_target_info_under_test"


class _FakeLogger:
    """Minimal vllm.logger stand-in recording debug/info calls."""

    def __init__(self):
        self.records = []

    def debug(self, msg, *args):
        self.records.append(("debug", msg % args if args else msg))

    def info(self, msg, *args):
        self.records.append(("info", msg % args if args else msg))


@pytest.fixture
def flagtree(monkeypatch):
    """Load a fresh copy of the patch module with vllm.logger stubbed."""
    logger = _FakeLogger()
    vllm_pkg = ModuleType("vllm")
    vllm_pkg.__path__ = []
    vllm_logger = ModuleType("vllm.logger")
    vllm_logger.init_logger = lambda name: logger
    monkeypatch.setitem(sys.modules, "vllm", vllm_pkg)
    monkeypatch.setitem(sys.modules, "vllm.logger", vllm_logger)

    spec = importlib.util.spec_from_file_location(_MODULE_NAME, _SOURCE)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, _MODULE_NAME, module)
    spec.loader.exec_module(module)
    module.logger = logger
    return module


class FakeTargetInfo:
    """FlagTree-like ``triton.language.target_info`` with a mutable arch."""

    def __init__(self, arch="", hip=True):
        self.arch = arch
        self.is_hip_returns = hip
        self.constexpr_decorated = []

    def is_hip(self):
        return self.is_hip_returns

    def current_target(self):
        class Target:
            pass

        target = Target()
        target.arch = self.arch
        return target

    def constexpr_function(self, function):
        self.constexpr_decorated.append(function.__name__)
        return function


@pytest.fixture
def fake_target_info(monkeypatch):
    """Install a fake triton.language.target_info in sys.modules.

    The patch module imports ``triton.language.target_info`` inside
    ``apply_flagtree_target_info``, so pre-seeding ``sys.modules`` (and the
    parent package attribute) controls what it sees; teardown restores whatever
    was there before.  On hosts without real Triton the parent packages are
    faked too — a statement like ``import triton.language.target_info`` still
    imports every missing parent from disk, so seeding only the leaf would
    fail on the missing ``triton`` package and the patch would report
    Triton unavailable.
    """
    fake = FakeTargetInfo()

    try:
        import triton.language as real_parent  # noqa: F401
    except ImportError:
        real_parent = None

    if real_parent is not None:
        saved_module = sys.modules.get("triton.language.target_info")
        saved_attr = getattr(real_parent, "target_info", None)
        monkeypatch.setitem(sys.modules, "triton.language.target_info", fake)
        real_parent.target_info = fake
    else:  # pragma: no cover - CPU-only Windows hosts without triton
        triton_pkg = ModuleType("triton")
        triton_pkg.__path__ = []
        language_pkg = ModuleType("triton.language")
        language_pkg.__path__ = []
        monkeypatch.setitem(sys.modules, "triton", triton_pkg)
        monkeypatch.setitem(sys.modules, "triton.language", language_pkg)
        monkeypatch.setitem(sys.modules, "triton.language.target_info", fake)
        saved_module = None
        saved_attr = None

    yield fake

    if real_parent is not None:
        if saved_module is not None:
            monkeypatch.setitem(
                sys.modules, "triton.language.target_info", saved_module
            )
        else:
            sys.modules.pop("triton.language.target_info", None)
        if saved_attr is not None:
            real_parent.target_info = saved_attr


@pytest.mark.parametrize(
    "arch,expected_true",
    [
        # Prefixes overlap by design: a gfx1250 arch also satisfies the
        # broader gfx12/rdna families, mirroring Triton's own semantics.
        ("gfx1250", {"is_hip_gfx1250", "is_hip_gfx12", "is_hip_rdna"}),
        ("gfx1251", {"is_hip_gfx1250", "is_hip_gfx12", "is_hip_rdna"}),
        ("gfx908", {"is_hip_cdna", "is_hip_cdna1"}),
        ("gfx90a", {"is_hip_cdna", "is_hip_cdna2"}),
        ("gfx942", {"is_hip_cdna"}),
        ("gfx950", {"is_hip_cdna"}),
        ("gfx1100", {"is_hip_rdna", "is_hip_gfx11"}),
        ("gfx1150", {"is_hip_rdna", "is_hip_gfx11"}),
        ("gfx1201", {"is_hip_rdna", "is_hip_gfx12"}),
        ("gfx1030", set()),  # RDNA3 has no predicate in the patch set
    ],
)
def test_predicate_truth_table_per_arch(
    flagtree, fake_target_info, arch, expected_true
):
    """Each arch makes exactly its family's predicates true, others false."""
    fake_target_info.arch = arch

    assert flagtree.apply_flagtree_target_info() is True

    all_predicates = {
        "is_hip_gfx1250",
        "is_hip_cdna",
        "is_hip_cdna1",
        "is_hip_cdna2",
        "is_hip_rdna",
        "is_hip_gfx11",
        "is_hip_gfx12",
    }
    for name in sorted(all_predicates):
        predicate = getattr(fake_target_info, name, None)
        assert callable(predicate), f"{name} was not added to target_info"
        expected = name in expected_true
        assert predicate() is expected, (
            f"{name}() for arch={arch}: expected {expected}, got {predicate()}"
        )


def test_non_hip_target_disables_every_predicate(flagtree, fake_target_info):
    """A CUDA/CPU target makes every added predicate False even on gfx names."""
    fake_target_info.arch = "gfx942"
    fake_target_info.is_hip_returns = False

    assert flagtree.apply_flagtree_target_info() is True
    for name in (
        "is_hip_gfx1250",
        "is_hip_cdna",
        "is_hip_cdna1",
        "is_hip_cdna2",
        "is_hip_rdna",
        "is_hip_gfx11",
        "is_hip_gfx12",
    ):
        assert getattr(fake_target_info, name)() is False, name


def test_existing_attributes_are_left_alone(flagtree, fake_target_info):
    """Predicates FlagTree already exports are never overwritten."""

    def real_flagtree_gfx1250():
        return "flagtree-original"

    fake_target_info.is_hip_gfx1250 = real_flagtree_gfx1250
    fake_target_info.arch = "gfx908"

    assert flagtree.apply_flagtree_target_info() is True
    assert fake_target_info.is_hip_gfx1250 is real_flagtree_gfx1250
    assert fake_target_info.is_hip_gfx1250() == "flagtree-original"
    # Missing siblings were still added.
    assert callable(fake_target_info.is_hip_cdna)
    assert fake_target_info.is_hip_cdna() is True


def test_added_predicates_are_decorated_as_constexpr(flagtree, fake_target_info):
    """When FlagTree offers constexpr_function, added predicates get wrapped."""
    fake_target_info.arch = "gfx908"

    flagtree.apply_flagtree_target_info()

    decorated = set(fake_target_info.constexpr_decorated)
    assert decorated == {
        "is_hip_gfx1250",
        "is_hip_cdna",
        "is_hip_cdna1",
        "is_hip_cdna2",
        "is_hip_rdna",
        "is_hip_gfx11",
        "is_hip_gfx12",
    }


def test_broken_constexpr_decorator_falls_back_to_plain(flagtree, fake_target_info):
    """A constexpr_function that raises must not break the patch."""

    def broken(function):
        raise RuntimeError("flagtree constexpr bug")

    fake_target_info.constexpr_function = broken
    fake_target_info.arch = "gfx908"

    assert flagtree.apply_flagtree_target_info() is True
    assert fake_target_info.is_hip_cdna() is True
    assert fake_target_info.is_hip_gfx1250() is False


def test_unimportable_triton_reports_false(flagtree, monkeypatch):
    """When triton.language.target_info cannot import, apply returns False."""
    monkeypatch.setitem(sys.modules, "triton.language.target_info", None)

    assert flagtree.apply_flagtree_target_info() is False
    assert any(level == "debug" for level, _ in flagtree.logger.records)


def test_current_target_failure_yields_empty_arch(flagtree, fake_target_info):
    """A raising current_target makes predicates False instead of crashing."""

    def broken_target():
        raise RuntimeError("no target on this stack")

    fake_target_info.current_target = broken_target

    assert flagtree.apply_flagtree_target_info() is True
    for name in ("is_hip_cdna", "is_hip_rdna"):
        assert getattr(fake_target_info, name)() is False, name


@pytest.mark.parametrize("target_value", [None, object()])
def test_non_string_arch_is_treated_as_empty(flagtree, fake_target_info, target_value):
    """A target without a string ``arch`` attribute must not crash predicates."""

    def current_target():
        return target_value

    fake_target_info.current_target = current_target

    assert flagtree.apply_flagtree_target_info() is True
    assert fake_target_info.is_hip_cdna() is False
    assert fake_target_info.is_hip_gfx11() is False


def test_apply_is_cached_and_logs_added_names(flagtree, fake_target_info):
    """Repeat calls return the cached result; the first call logs the adds."""
    fake_target_info.arch = "gfx908"

    first = flagtree.apply_flagtree_target_info()
    second = flagtree.apply_flagtree_target_info()

    assert first is True and second is True
    infos = [message for level, message in flagtree.logger.records if level == "info"]
    assert len(infos) == 1, "info log must be emitted once despite the lru_cache"
    assert "7 architecture predicate(s)" in infos[0]
    assert "is_hip_gfx1250" in infos[0]


def test_real_triton_import_path(monkeypatch):
    """Against the real installed triton, apply adds the missing names.

    FlagTree itself is not installed here, so the real
    ``triton.language.target_info`` lacks the predicates; the patch must add
    them and they must be False on a non-AMD host. Runs in a subprocess-free
    way by operating on the genuine module and restoring it afterwards.
    """
    pytest.importorskip("triton")

    import triton.language.target_info as real_target_info

    saved = {}
    for name in (
        "is_hip_gfx1250",
        "is_hip_cdna",
        "is_hip_cdna1",
        "is_hip_cdna2",
        "is_hip_rdna",
        "is_hip_gfx11",
        "is_hip_gfx12",
    ):
        saved[name] = getattr(real_target_info, name, None)
        if saved[name] is None:
            del saved[name]

    logger = _FakeLogger()
    vllm_pkg = ModuleType("vllm")
    vllm_pkg.__path__ = []
    vllm_logger = ModuleType("vllm.logger")
    vllm_logger.init_logger = lambda name: logger
    monkeypatch.setitem(sys.modules, "vllm", vllm_pkg)
    monkeypatch.setitem(sys.modules, "vllm.logger", vllm_logger)

    spec = importlib.util.spec_from_file_location(_MODULE_NAME + "_real", _SOURCE)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, _MODULE_NAME + "_real", module)
    spec.loader.exec_module(module)

    try:
        assert module.apply_flagtree_target_info() is True
        for name in saved:
            assert getattr(real_target_info, name, None) is not None
        # On a CUDA/CPU dev host is_hip() is False, so every added
        # predicate must be False rather than crash.
        assert real_target_info.is_hip_gfx1250() is False
        assert real_target_info.is_hip_cdna() is False
    finally:
        for name, value in saved.items():
            if value is None:
                if hasattr(real_target_info, name):
                    delattr(real_target_info, name)
            else:
                setattr(real_target_info, name, value)
