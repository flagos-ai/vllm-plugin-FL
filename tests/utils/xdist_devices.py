# SPDX-License-Identifier: Apache-2.0
"""Assign a distinct visible device before each xdist worker imports Torch."""

import os

import pytest


def pytest_addoption(parser):
    parser.addoption("--test-device-env", default="")


@pytest.hookimpl(optionalhook=True)
def pytest_xdist_setupnodes(config, specs):
    variable = config.getoption("test_device_env")
    if not variable:
        return
    inherited = os.environ.get(variable, "").strip()
    devices = (
        [item.strip() for item in inherited.split(",") if item.strip()]
        if inherited and inherited.lower() != "all"
        else [str(index) for index in range(len(specs))]
    )
    if len(specs) > len(devices):
        raise pytest.UsageError(f"{len(specs)} workers exceed {variable}={inherited}")
    for spec, device in zip(specs, devices):
        # execnet applies this before importing pytest and the test conftests.
        spec.env[variable] = device
