# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-FL project

import glob
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from shutil import which
from typing import List

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext

ROOT_DIR = Path(__file__).parent.resolve()
logger = logging.getLogger(__name__)

# =============================================================================
# Environment Variables
# =============================================================================

VLLM_VENDOR = os.environ.get("VLLM_VENDOR", "").lower()
MAX_JOBS = os.environ.get("MAX_JOBS")
NVCC_THREADS = os.environ.get("NVCC_THREADS")
CMAKE_BUILD_TYPE = os.environ.get("CMAKE_BUILD_TYPE")
VERBOSE = os.environ.get("VERBOSE", "0") == "1"

SUPPORTED_VENDORS = ["cuda", "ascend"]


# =============================================================================
# Utility Functions
# =============================================================================

def is_sccache_available() -> bool:
    return which("sccache") is not None


def is_ccache_available() -> bool:
    return which("ccache") is not None


def is_ninja_available() -> bool:
    return which("ninja") is not None


def _is_cuda() -> bool:
    return VLLM_VENDOR == "cuda"


def _is_ascend() -> bool:
    return VLLM_VENDOR == "ascend"


# =============================================================================
# Version
# =============================================================================

def get_cuda_version() -> str:
    """Detect CUDA version from nvcc."""
    try:
        output = subprocess.check_output(["nvcc", "--version"], stderr=subprocess.STDOUT)
        output = output.decode("utf-8")
        # Parse "release X.Y" from nvcc output
        import re
        match = re.search(r"release (\d+)\.(\d+)", output)
        if match:
            major, minor = match.groups()
            return f"cu{major}{minor}"
    except Exception:
        pass
    return "cu"


def get_git_commit() -> str:
    """Get the first 8 characters of git commit hash."""
    try:
        output = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.STDOUT)
        return output.decode("utf-8").strip()[:8]
    except Exception:
        return "unknown"


def get_build_date() -> str:
    """Get current date in YYYYMMDD format."""
    from datetime import datetime
    return datetime.now().strftime("%Y%m%d")


def get_vllm_fl_version() -> str:
    version = "0.0.1.dev0"
    commit_id = get_git_commit()
    build_date = get_build_date()

    if VLLM_VENDOR == "cuda":
        cuda_ver = get_cuda_version()
        version += f"+g{commit_id}.{build_date}.{cuda_ver}"
    elif VLLM_VENDOR:
        version += f"+g{commit_id}.{build_date}.{VLLM_VENDOR}"
    else:
        version += f"+g{commit_id}.{build_date}"

    return version


# =============================================================================
# CMake Extension
# =============================================================================

class CMakeExtension(Extension):
    def __init__(self, name: str, cmake_lists_dir: str = ".", **kwa) -> None:
        super().__init__(name, sources=[], **kwa)
        self.cmake_lists_dir = os.path.abspath(cmake_lists_dir)


class cmake_build_ext(build_ext):
    """CMake build extension for vLLM-FL operators."""

    did_config: dict = {}

    def run(self):
        """Override run to skip the default extension copying."""
        self.build_extensions()

    def compute_num_jobs(self):
        """Compute number of parallel compilation jobs."""
        num_jobs = MAX_JOBS
        if num_jobs is not None:
            num_jobs = int(num_jobs)
            logger.info("Using MAX_JOBS=%d as the number of jobs.", num_jobs)
        else:
            try:
                num_jobs = len(os.sched_getaffinity(0))
            except AttributeError:
                num_jobs = os.cpu_count() or 1

        nvcc_threads = None
        if _is_cuda() and NVCC_THREADS is not None:
            nvcc_threads = int(NVCC_THREADS)
            logger.info(
                "Using NVCC_THREADS=%d as the number of nvcc threads.",
                nvcc_threads,
            )
            num_jobs = max(1, num_jobs // nvcc_threads)

        return num_jobs, nvcc_threads

    def configure(self, ext: CMakeExtension) -> None:
        """Configure cmake for the extension."""
        if ext.cmake_lists_dir in cmake_build_ext.did_config:
            return

        cmake_build_ext.did_config[ext.cmake_lists_dir] = True

        # Build type
        default_cfg = "Debug" if self.debug else "RelWithDebInfo"
        cfg = CMAKE_BUILD_TYPE or default_cfg

        cmake_args = [
            f"-DCMAKE_BUILD_TYPE={cfg}",
            f"-DVLLM_VENDOR={VLLM_VENDOR}",
            f"-DVLLM_PYTHON_EXECUTABLE={sys.executable}",
        ]

        if VERBOSE:
            cmake_args.append("-DCMAKE_VERBOSE_MAKEFILE=ON")

        # Compiler cache
        if is_sccache_available():
            cmake_args += [
                "-DCMAKE_C_COMPILER_LAUNCHER=sccache",
                "-DCMAKE_CXX_COMPILER_LAUNCHER=sccache",
                "-DCMAKE_CUDA_COMPILER_LAUNCHER=sccache",
            ]
        elif is_ccache_available():
            cmake_args += [
                "-DCMAKE_C_COMPILER_LAUNCHER=ccache",
                "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache",
                "-DCMAKE_CUDA_COMPILER_LAUNCHER=ccache",
            ]

        # Parallelism and build tool
        num_jobs, nvcc_threads = self.compute_num_jobs()

        if nvcc_threads:
            cmake_args.append(f"-DNVCC_THREADS={nvcc_threads}")

        if is_ninja_available():
            build_tool = ["-G", "Ninja"]
            cmake_args += [
                "-DCMAKE_JOB_POOL_COMPILE:STRING=compile",
                f"-DCMAKE_JOB_POOLS:STRING=compile={num_jobs}",
            ]
        else:
            build_tool = []

        # Additional cmake args from environment
        extra_cmake_args = os.environ.get("CMAKE_ARGS")
        if extra_cmake_args:
            cmake_args += extra_cmake_args.split()

        print(f"Configuring CMake for vendor: {VLLM_VENDOR}")
        print(f"  Source dir: {ext.cmake_lists_dir}")
        print(f"  Build dir: {self.build_temp}")

        subprocess.check_call(
            ["cmake", ext.cmake_lists_dir, *build_tool, *cmake_args],
            cwd=self.build_temp,
        )

    def build_extensions(self) -> None:
        """Build all extensions."""
        # Check CMake
        try:
            subprocess.check_output(["cmake", "--version"], stderr=subprocess.STDOUT)
        except (OSError, subprocess.CalledProcessError) as e:
            raise RuntimeError(
                f"CMake not available or not working: {e}\n"
                "Please install with:\n"
                f"  VLLM_VENDOR={VLLM_VENDOR} pip install --no-build-isolation -e ."
            ) from e

        # Create build directory
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        # Configure and collect targets
        targets = []
        for ext in self.extensions:
            self.configure(ext)
            target_name = ext.name.split(".")[-1]
            targets.append(target_name)

        # Build
        num_jobs, _ = self.compute_num_jobs()
        build_args = [
            "--build", ".",
            f"-j={num_jobs}",
            *[f"--target={name}" for name in targets],
        ]

        print(f"Building targets: {targets}")
        subprocess.check_call(["cmake", *build_args], cwd=self.build_temp)

        # Copy built extensions to where setuptools expects them
        for ext in self.extensions:
            # Get the full path where setuptools expects the extension
            dest_path = Path(self.get_ext_fullpath(ext.name)).absolute()
            dest_dir = dest_path.parent

            # Create destination directory if it doesn't exist
            dest_dir.mkdir(parents=True, exist_ok=True)

            # Find the built .so file in the build directory
            target_name = ext.name.split(".")[-1]
            # Look for the .so file in various possible locations
            so_patterns = [
                f"{self.build_temp}/{VLLM_VENDOR}/{target_name}*.so",
                f"{self.build_temp}/{target_name}*.so",
            ]

            built_so = None
            for pattern in so_patterns:
                matches = glob.glob(pattern)
                if matches:
                    built_so = matches[0]
                    break

            if built_so is None:
                raise RuntimeError(
                    f"Could not find built extension {target_name}.so in {self.build_temp}"
                )

            # Copy to destination with the correct name
            print(f"Copying {built_so} to {dest_path}")
            shutil.copy2(built_so, dest_path)


# =============================================================================
# Package Configuration
# =============================================================================

def read_readme() -> str:
    """Read the README file if present."""
    readme_path = ROOT_DIR / "README.md"
    if readme_path.is_file():
        return readme_path.read_text(encoding="utf-8")
    return ""


def get_requirements() -> List[str]:
    """Get Python package dependencies from requirements.txt."""
    requirements_path = ROOT_DIR / "requirements.txt"
    if not requirements_path.is_file():
        logger.warning("requirements.txt not found")
        return []

    def _read_requirements(filepath: Path) -> List[str]:
        resolved = []
        for line in filepath.read_text().strip().split("\n"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("-r "):
                inc_file = filepath.parent / line.split()[1]
                resolved += _read_requirements(inc_file)
            elif line.startswith("--"):
                continue
            else:
                resolved.append(line)
        return resolved

    try:
        return _read_requirements(requirements_path)
    except Exception as e:
        logger.warning(f"Failed to read requirements.txt: {e}")
        return []


# =============================================================================
# Extension Modules
# =============================================================================

ext_modules = []

if VLLM_VENDOR:
    if VLLM_VENDOR not in SUPPORTED_VENDORS:
        raise ValueError(
            f"Unsupported vendor: {VLLM_VENDOR}\n"
            f"Supported vendors: {SUPPORTED_VENDORS}"
        )
    csrc_dir = str(ROOT_DIR / "csrc")
    # Extension name is vllm_fl._C - will be importable as `import vllm_fl._C`
    ext_modules.append(CMakeExtension(name="vllm_fl._C", cmake_lists_dir=csrc_dir))


# =============================================================================
# Command Classes
# =============================================================================

if ext_modules:
    cmdclass = {"build_ext": cmake_build_ext}
else:
    cmdclass = {}


# =============================================================================
# Setup
# =============================================================================

setup(
    name="vllm_fl",
    version=get_vllm_fl_version(),
    author="vLLM-FL team",
    license="Apache 2.0",
    description="vLLM FL backend plugin with multi-vendor C++ operators",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/flagos-ai/vllm-plugin-FL",
    project_urls={
        "Homepage": "https://github.com/flagos-ai/vllm-plugin-FL",
    },
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: Apache Software License",
        "Intended Audience :: Developers",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    packages=find_packages(exclude=("docs", "examples", "tests*", "csrc")),
    python_requires=">=3.10",
    install_requires=get_requirements(),
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-asyncio",
            "black",
            "isort",
            "mypy",
        ],
    },
    entry_points={
        "vllm.platform_plugins": ["fl = vllm_fl:register"],
        "vllm.general_plugins": ["fl = vllm_fl:register_model"],
    },
    package_data={
        "vllm_fl": [
            "*.so",
            "dispatch/config/*.yaml",
        ],
    },
    include_package_data=True,
)
