# Copyright (c) 2026 BAAI. All rights reserved.
#
# vllm-plugin-FL: vLLM Federated Learning Plugin
#
# This setup script builds the vllm_fl._C / vllm_fl._C_ascend vendor-specific
# C++ extension via CMake. It supports CUDA and Ascend backends controlled by
# the VLLM_VENDOR environment variable. The build pipeline:
#   1. Detects available tooling (cmake, ninja, sccache/ccache)
#   2. Configures and compiles the C++/CUDA/Ascend sources under csrc/
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

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

ROOT_DIR = Path(__file__).parent.resolve()
logger = logging.getLogger(__name__)

VLLM_VENDOR = os.environ.get("VLLM_VENDOR", "").lower()
MAX_JOBS = os.environ.get("MAX_JOBS")
NVCC_THREADS = os.environ.get("NVCC_THREADS")
CMAKE_BUILD_TYPE = os.environ.get("CMAKE_BUILD_TYPE")
VERBOSE = os.environ.get("VERBOSE", "0") == "1"

SUPPORTED_VENDORS = ("cuda", "ascend")


def _is_cuda() -> bool:
    return VLLM_VENDOR == "cuda"


def _is_ascend() -> bool:
    return VLLM_VENDOR == "ascend"


def _which(name: str) -> bool:
    return which(name) is not None


def _get_torch_npu_path() -> str:
    """Return the directory containing the installed torch_npu package."""
    try:
        import torch_npu
        return str(Path(torch_npu.__file__).parent)
    except Exception as exc:
        raise RuntimeError(
            "torch_npu is required for Ascend builds but could not be imported."
        ) from exc


def _get_ascend_home_path() -> str:
    """Return the ASCEND_HOME_PATH (CANN toolkit installation root)."""
    ascend_home = os.environ.get("ASCEND_HOME_PATH")
    if ascend_home:
        return ascend_home
    # Common default location.
    default = "/usr/local/Ascend/ascend-toolkit/latest"
    if os.path.isdir(default):
        return default
    raise RuntimeError(
        "ASCEND_HOME_PATH is not set and the default CANN path does not exist. "
        "Please set ASCEND_HOME_PATH to your CANN toolkit installation directory."
    )


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

        if _is_ascend():
            cmake_args += [
                f"-DTORCH_NPU_PATH={_get_torch_npu_path()}",
                f"-DASCEND_HOME_PATH={_get_ascend_home_path()}",
            ]
            soc_version = os.environ.get("SOC_VERSION")
            if soc_version:
                cmake_args.append(f"-DSOC_VERSION={soc_version}")
            if os.environ.get("BUILD_PTO_CHUNK_GDN"):
                cmake_args.append("-DBUILD_PTO_CHUNK_GDN=ON")

        build_tool = []
        # AscendC kernel auto-codegen currently assumes a Makefile generator
        # (link.txt, etc.). Do not use Ninja for Ascend builds even if available.
        if _is_cuda() and _which("ninja"):
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

    def _apply_cann_extract_host_stub_patch(self) -> Path | None:
        """Apply a temporary workaround patch for CANN 8.5.x extract_host_stub.py."""
        if not _is_ascend():
            return None
        script = ROOT_DIR / "csrc/ascend/patch_cann_extract_host_stub.py"
        if not script.is_file():
            raise RuntimeError(f"CANN patch script not found: {script}")
        backup_path = Path(self.build_temp) / "extract_host_stub.py.vllm_fl_bak"
        subprocess.check_call(
            [sys.executable, str(script), "apply", "--backup", str(backup_path)]
        )
        return backup_path

    def _restore_cann_extract_host_stub_patch(self, backup_path: Path | None) -> None:
        """Restore the original CANN extract_host_stub.py from backup."""
        if backup_path is None:
            return
        script = ROOT_DIR / "csrc/ascend/patch_cann_extract_host_stub.py"
        if not script.is_file():
            return
        subprocess.run(
            [sys.executable, str(script), "restore", "--backup", str(backup_path)],
            check=False,
        )

    def build_extensions(self) -> None:
        try:
            subprocess.check_output(["cmake", "--version"], stderr=subprocess.STDOUT)
        except (OSError, subprocess.CalledProcessError) as exc:
            raise RuntimeError(
                "CMake is required to build vllm_fl extensions. "
                "Install cmake and run with VLLM_VENDOR=cuda or VLLM_VENDOR=ascend."
            ) from exc

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

        cann_patch_backup: Path | None = None
        try:
            cann_patch_backup = self._apply_cann_extract_host_stub_patch()
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

                # Ascend builds produce an additional shared AscendC kernel library
                # that the extension links against. Copy it next to the extension so
                # the $ORIGIN rpath can resolve it at load time.
                if _is_ascend():
                    kernel_lib_patterns = [
                        f"{self.build_temp}/lib/libvllm_fl_kernels*.so",
                        f"{self.build_temp}/**/libvllm_fl_kernels*.so",
                    ]
                    for pattern in kernel_lib_patterns:
                        for kernel_lib in glob.glob(pattern, recursive=True):
                            shutil.copy2(kernel_lib, dest_path.parent / Path(kernel_lib).name)
                            break
                        else:
                            continue
                        break
        finally:
            self._restore_cann_extract_host_stub_patch(cann_patch_backup)


ext_modules = []
if VLLM_VENDOR:
    if VLLM_VENDOR not in SUPPORTED_VENDORS:
        raise ValueError(
            f"Unsupported vendor: {VLLM_VENDOR}. "
            f"Supported vendors: {', '.join(SUPPORTED_VENDORS)}"
        )
    ext_name = "vllm_fl._C_ascend" if _is_ascend() else "vllm_fl._C"
    ext_modules.append(
        CMakeExtension(name=ext_name, cmake_lists_dir=str(ROOT_DIR / "csrc"))
    )


setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": CMakeBuildExt} if ext_modules else {},
)
