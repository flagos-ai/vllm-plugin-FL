# Copyright 2025 BAAI. All rights reserved.
"""Multi-platform supplementary functional coverage tests for vllm_fl.

Real-device functional tests that raise coverage beyond what the core
``tests/functional_tests`` suites already cover (ops, compilation,
distributed). Counterpart of ``tests/unit_tests/coverage/``; the CPU-safe
unit-level version of each test here lives in that tree.

* collected automatically by ``tests/run.py --scope functional`` (this
  directory is inside ``tests/functional_tests/``);
* every test carries the ``supplementary`` marker (registered in the root
  ``tests/conftest.py``);
* vendor-specific tests carry ``pytest.mark.platforms("...")`` and are
  gated by the ``platforms`` hook in ``tests/unit_tests/coverage/conftest.py``
  when reached through the unit tree; tests here additionally skip on
  their own when the required device or library is absent.
"""
