# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# vllm-plugin-FL: vLLM Federated Learning Plugin
#
# This setup script builds the vllm_fl._C C++/CUDA extension via CMake.
# It supports CUDA and Ascend vendor builds controlled by
# the VLLM_VENDOR environment variable. The build pipeline:
#   1. Detects available tooling (cmake, ninja, sccache/ccache)
#   2. Configures and compiles the C++/CUDA sources under csrc/
#   3. Copies the resulting shared library (.so/.pyd) to the package directory

from __future__ import annotations

import glob
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from shutil import which

from setuptools import Distribution, Extension, setup
from setuptools.command.build_ext import build_ext

ROOT_DIR = Path(__file__).parent.resolve()
logger = logging.getLogger(__name__)

VLLM_VENDOR = os.environ.get("VLLM_VENDOR", "").lower()
MAX_JOBS = os.environ.get("MAX_JOBS")
NVCC_THREADS = os.environ.get("NVCC_THREADS")
CMAKE_BUILD_TYPE = os.environ.get("CMAKE_BUILD_TYPE")
VERBOSE = os.environ.get("VERBOSE", "0") == "1"

SUPPORTED_VENDORS = ("cuda", "ascend")
HAS_PREBUILT_BINARY = any(
    (ROOT_DIR / "vllm_fl" / "dispatch" / "backends" / "vendor" / "ascend" / "prebuilt").glob(
        "*/lib/*.so"
    )
)


class BinaryDistribution(Distribution):
    """Ensure wheels containing vendored shared libraries are platform tagged."""

    def has_ext_modules(self) -> bool:
        return True


def _is_cuda() -> bool:
    return VLLM_VENDOR == "cuda"


def _is_ascend() -> bool:
    return VLLM_VENDOR == "ascend"


def _ascend_prebuilt_dir() -> str:
    # A3 is the default delivery target. A2 builders select their common
    # family explicitly with SOC_VERSION=ascend910b (B1/B2/B3/B4 share it).
    soc = os.environ.get("SOC_VERSION", "ascend910_93").lower()
    if soc.startswith("ascend910b") or soc == "910b":
        return "ascend910b1"
    if soc.startswith("ascend910_93") or soc == "910c":
        return "ascend910_93"
    raise ValueError(f"Unsupported Ascend SOC_VERSION for Qwen3.6: {soc}")


def _which(name: str) -> bool:
    return which(name) is not None


class CMakeExtension(Extension):
    def __init__(self, name: str, cmake_lists_dir: str) -> None:
        super().__init__(name, sources=[])
        self.cmake_lists_dir = os.path.abspath(cmake_lists_dir)


class CMakeBuildExt(build_ext):
    did_config: dict[str, bool] = {}

    def run(self) -> None:
        self.build_extensions()

    def compute_num_jobs(self) -> tuple[int, int | None]:
        if MAX_JOBS is not None:
            num_jobs = int(MAX_JOBS)
            logger.info("Using MAX_JOBS=%d as the number of jobs.", num_jobs)
        else:
            try:
                num_jobs = len(os.sched_getaffinity(0))
            except AttributeError:
                num_jobs = os.cpu_count() or 1

        nvcc_threads = None
        if _is_cuda() and NVCC_THREADS is not None:
            nvcc_threads = int(NVCC_THREADS)
            logger.info("Using NVCC_THREADS=%d.", nvcc_threads)
            num_jobs = max(1, num_jobs // nvcc_threads)

        return num_jobs, nvcc_threads

    def configure(self, ext: CMakeExtension) -> None:
        if CMakeBuildExt.did_config.get(ext.cmake_lists_dir):
            return

        CMakeBuildExt.did_config[ext.cmake_lists_dir] = True
        cfg = CMAKE_BUILD_TYPE or ("Debug" if self.debug else "RelWithDebInfo")
        cmake_args = [
            f"-DCMAKE_BUILD_TYPE={cfg}",
            f"-DVLLM_VENDOR={VLLM_VENDOR}",
            f"-DVLLM_PYTHON_EXECUTABLE={sys.executable}",
        ]

        if _is_ascend():
            cmake_args.append(
                f"-DSOC_VERSION={os.environ.get('SOC_VERSION', 'ascend910_93')}"
            )
            if ascend_home := os.environ.get("ASCEND_HOME_PATH"):
                cmake_args.append(f"-DASCEND_HOME_PATH={ascend_home}")
            if torch_npu_path := os.environ.get("TORCH_NPU_PATH"):
                cmake_args.append(f"-DTORCH_NPU_PATH={torch_npu_path}")

        if VERBOSE:
            cmake_args.append("-DCMAKE_VERBOSE_MAKEFILE=ON")

        if _which("sccache"):
            cmake_args += [
                "-DCMAKE_C_COMPILER_LAUNCHER=sccache",
                "-DCMAKE_CXX_COMPILER_LAUNCHER=sccache",
                "-DCMAKE_CUDA_COMPILER_LAUNCHER=sccache",
            ]
        elif _which("ccache"):
            cmake_args += [
                "-DCMAKE_C_COMPILER_LAUNCHER=ccache",
                "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache",
                "-DCMAKE_CUDA_COMPILER_LAUNCHER=ccache",
            ]

        num_jobs, nvcc_threads = self.compute_num_jobs()
        if nvcc_threads:
            cmake_args.append(f"-DNVCC_THREADS={nvcc_threads}")

        build_tool = []
        if _which("ninja"):
            build_tool = ["-G", "Ninja"]
            cmake_args += [
                "-DCMAKE_JOB_POOL_COMPILE:STRING=compile",
                f"-DCMAKE_JOB_POOLS:STRING=compile={num_jobs}",
            ]

        extra_cmake_args = os.environ.get("CMAKE_ARGS")
        if extra_cmake_args:
            cmake_args += extra_cmake_args.split()

        subprocess.check_call(
            ["cmake", ext.cmake_lists_dir, *build_tool, *cmake_args],
            cwd=self.build_temp,
        )

    def build_extensions(self) -> None:
        try:
            subprocess.check_output(["cmake", "--version"], stderr=subprocess.STDOUT)
        except (OSError, subprocess.CalledProcessError) as exc:
            raise RuntimeError(
                "CMake is required to build vllm_fl._C. "
                "Install cmake and set VLLM_VENDOR to cuda or ascend."
            ) from exc

        ascend_source_root = None
        if _is_ascend():
            ascend_source_root = (
                ROOT_DIR
                / "vllm_fl"
                / "dispatch"
                / "backends"
                / "vendor"
                / "ascend"
                / "prebuilt"
                / _ascend_prebuilt_dir()
            )
            # This directory contains build outputs only. Recreate the exact
            # SoC/ABI payload so a wheel cannot retain stale diagnostic or
            # previous-Python artifacts.
            if ascend_source_root.exists():
                shutil.rmtree(ascend_source_root)
            subprocess.check_call(
                ["bash", str(ROOT_DIR / "csrc" / "ascend" / "build_opp.sh")],
                cwd=ROOT_DIR,
            )

        os.makedirs(self.build_temp, exist_ok=True)

        targets = []
        for ext in self.extensions:
            self.configure(ext)
            targets.append(ext.name.split(".")[-1])

        num_jobs, _ = self.compute_num_jobs()
        build_args = [
            "--build",
            ".",
            f"-j={num_jobs}",
            *[f"--target={name}" for name in targets],
        ]
        subprocess.check_call(["cmake", *build_args], cwd=self.build_temp)

        for ext in self.extensions:
            dest_path = Path(self.get_ext_fullpath(ext.name)).absolute()
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            target_name = ext.name.split(".")[-1]
            patterns = [
                f"{self.build_temp}/{VLLM_VENDOR}/{target_name}*.so",
                f"{self.build_temp}/{VLLM_VENDOR}/{target_name}*.pyd",
                f"{self.build_temp}/{target_name}*.so",
                f"{self.build_temp}/{target_name}*.pyd",
            ]
            built_ext = next(
                (match for pattern in patterns for match in glob.glob(pattern)),
                None,
            )
            if built_ext is None:
                raise RuntimeError(
                    f"Could not find built extension {target_name} in {self.build_temp}"
                )
            shutil.copy2(built_ext, dest_path)

        if _is_ascend() and ascend_source_root is not None:
            source_opp = ascend_source_root / "opp"
            dest_opp = (
                Path(self.build_lib)
                / "vllm_fl"
                / "dispatch"
                / "backends"
                / "vendor"
                / "ascend"
                / "prebuilt"
                / _ascend_prebuilt_dir()
                / "opp"
            )
            if not source_opp.is_dir():
                raise RuntimeError(
                    f"Ascend OPP build did not produce {source_opp}"
                )
            # build/lib is reused by setuptools across incremental wheel
            # builds. Replace the OPP payload atomically at directory
            # granularity so removed operators or an old vendor layout cannot
            # leak into a later wheel through copytree's merge semantics.
            if dest_opp.exists():
                shutil.rmtree(dest_opp)
            shutil.copytree(source_opp, dest_opp, dirs_exist_ok=True)


ext_modules = []
if VLLM_VENDOR:
    if VLLM_VENDOR not in SUPPORTED_VENDORS:
        raise ValueError(
            f"Unsupported vendor: {VLLM_VENDOR}. "
            f"Supported vendors: {', '.join(SUPPORTED_VENDORS)}"
        )
    if _is_ascend():
        ascend_dir = _ascend_prebuilt_dir()
        extension_name = (
            "vllm_fl.dispatch.backends.vendor.ascend.prebuilt."
            f"{ascend_dir}.lib._C_ascend"
        )
    else:
        extension_name = "vllm_fl._C"
    ext_modules.append(
        CMakeExtension(name=extension_name, cmake_lists_dir=str(ROOT_DIR / "csrc"))
    )


setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": CMakeBuildExt} if ext_modules else {},
    distclass=BinaryDistribution if HAS_PREBUILT_BINARY else Distribution,
)
