# Copyright (c) 2025 BAAI. All rights reserved.

"""Supplementary coverage test suite for vllm_fl.

This package holds tests that raise coverage of ``vllm_fl`` beyond what the
core ``tests/unit_tests`` / ``tests/functional_tests`` suites already cover.
It is deliberately isolated from the core entry points:

* daily PR CI never collects this directory (``tests/run.py --scope unit``
  points at ``tests/unit_tests/`` only);
* the suite is executed through ``tests/supplementary_tests/run.py`` (a
  thin subclass of the core ``TestRunner``) on idle-time windows such as
  weekends, on an NVIDIA A100 runner, via the
  ``coverage_supplementary.yml`` workflow;
* every test carries the ``supplementary`` marker so the suite can be
  selected or excluded with ``-m`` even if it is ever pointed at by another
  runner.
"""
