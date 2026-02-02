#!/bin/bash
# Copyright (c) 2025 BAAI. All rights reserved.
#
# Kimi-K2.5 Benchmark Runner
#
# Usage:
#   ./run_benchmark.sh                    # Full benchmark
#   ./run_benchmark.sh --quick            # Quick test
#   ./run_benchmark.sh --profile          # Run profiler only
#   ./run_benchmark.sh --modes fp8        # FP8 only

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLUGIN_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Add paths
export PYTHONPATH="${PLUGIN_DIR}:${PYTHONPATH}"
export PYTHONPATH="/root/kimi2.5/triton_ops:${PYTHONPATH}"

# vLLM-FL environment
export VLLM_PLATFORM_PLUGIN=fl
export USE_FLAGGEMS=True
export GEMS_MODE=SELECTIVE
export VLLM_FL_PREFER=vendor
export VLLM_FL_ALLOW_VENDORS=triton_optimized,cuda
export VLLM_ENABLE_V1_MULTIPROCESSING=0

# Parse arguments
PROFILE_ONLY=false
QUICK=false
MODES=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --profile)
            PROFILE_ONLY=true
            shift
            ;;
        --quick)
            QUICK=true
            shift
            ;;
        --modes)
            MODES="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--profile] [--quick] [--modes 'baseline selective fp8']"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "Kimi-K2.5 Benchmark Suite"
echo "=========================================="
echo "PYTHONPATH: ${PYTHONPATH}"
echo ""

if [ "$PROFILE_ONLY" = true ]; then
    echo "Running profiler..."
    python "${SCRIPT_DIR}/profile_ops.py"
else
    echo "Running end-to-end benchmark..."
    ARGS=""
    [ "$QUICK" = true ] && ARGS="$ARGS --quick"
    [ -n "$MODES" ] && ARGS="$ARGS --modes $MODES"

    python "${SCRIPT_DIR}/benchmark_e2e.py" $ARGS
fi

echo ""
echo "Benchmark complete!"
