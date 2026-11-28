# Copyright 2025 BAAI. All rights reserved.
"""Multi-platform supplementary coverage test suite for vllm_fl.

This package holds tests that raise coverage of ``vllm_fl`` beyond what the
core ``tests/unit_tests`` / ``tests/functional_tests`` suites already cover,
focused on the four vendor platforms enabled on the 0.5.0-dev line:

* kunlunxin (P800)
* hygon (bw1000 / DCU)
* iluvatar (bi_v150 / CoreX)
* thead (zw810e / PPU)

Directory layout (organized by module-under-test domain, mirroring the
parent ``tests/unit_tests`` structure):

* ``dispatch/``  — vendor backend classes, patch surfaces, io, config
* ``worker/``     — scheduler, model runner, worker helpers
* ``ops/``       — custom ops registration and vendor impls
* ``patches/``    — cross-module monkey-patch surfaces
* ``quantization/`` — W4A8/W8A8 and MoE expert quantization
* ``models/``     — model-level supplementary coverage
* ``attention/``  — attention utils vendor branches
* ``distributed/`` — FlagCX connector/loader/stats

The functional counterpart lives in ``tests/functional_tests/coverage/``.

Integration with the standard tree:

* ``tests/run.py --scope unit`` points at ``tests/unit_tests/`` and
  therefore collects this directory automatically — no separate runner;
* every test carries the ``supplementary`` marker (registered in the root
  ``tests/conftest.py``) so the suite can still be selected or excluded
  with ``-m`` where a split run is wanted;
* vendor-specific tests carry ``pytest.mark.platforms("...")`` so a run for
  platform A skips (not fails) platform B's gated tests — the gate is
  implemented in this directory's ``conftest.py``;
* tests must not depend on vendor hardware by construction: CPU stub
  tests are platform-agnostic; real-device tests are marked ``gpu`` /
  ``multi_gpu`` and auto-skip on accelerator-less hosts.
"""
