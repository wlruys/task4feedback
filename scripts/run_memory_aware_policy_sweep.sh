#!/usr/bin/env bash

# Memory-aware policy comprehensive benchmarking runner
# This script tests all 18 combinations of memory_aware_policy options

echo "==================================================================="
echo "Memory Aware Policy Comprehensive Benchmarking"
echo "==================================================================="
echo ""
echo "This will test all 18 combinations:"
echo "- Location State: launched, reserved, mapped (3 options)"
echo "- Overflow State: reserved, mapped, launched (3 options)"
echo "- Overflow Mode: full_spill, incoming_only (2 options)"
echo ""
echo "Total: 3 × 3 × 2 = 18 combinations"
echo ""

# Use the correct Python environment
PYTHON_CMD="/Users/wlruys/.local/share/mamba/envs/py313/bin/python"

# Default parameters for a reasonable benchmark
GRID_N=${1:-8}              # Grid size (8x8 = 64 cells)
STEPS=${2:-128}             # Number of simulation steps
LEVEL_MEMORY_GB=${3:-60}    # Per-level memory in GB
REPS=${4:-2}                # Repetitions per configuration
N_SEEDS=${5:-1}             # Number of seeds

echo "Parameters:"
echo "  Grid: ${GRID_N}×${GRID_N}"
echo "  Steps: ${STEPS}"
echo "  Level Memory: ${LEVEL_MEMORY_GB} GB"
echo "  Repetitions: ${REPS}"
echo "  Seeds: ${N_SEEDS}"
echo ""

# Create output directory with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="artifacts/memory_aware_policy_sweep_${TIMESTAMP}"

echo "Output directory: ${OUTPUT_DIR}"
echo ""
echo "Starting benchmark (this will take a while)..."
echo ""

# Run the comprehensive benchmark
$PYTHON_CMD scripts/bench_memory_aware_policy_sweep.py \
    --grid-n $GRID_N \
    --steps $STEPS \
    --level-memory-gb $LEVEL_MEMORY_GB \
    --reps $REPS \
    --n-seeds $N_SEEDS \
    --output-dir $OUTPUT_DIR \
    --seed 42

echo ""
echo "==================================================================="
echo "Benchmark completed!"
echo "==================================================================="
echo ""
echo "Results saved to: ${OUTPUT_DIR}"
echo ""
echo "Generated files:"
echo "- memory_aware_policy_sweep.csv           (raw data)"
echo "- memory_aware_policy_comparison_*.png    (comparison plots)"
echo ""

# Show the best configuration summary
if [ -f "${OUTPUT_DIR}/memory_aware_policy_sweep.csv" ]; then
    echo "Quick summary of results:"
    echo "========================"
    # Show CSV header and first few rows
    head -6 "${OUTPUT_DIR}/memory_aware_policy_sweep.csv" | column -t -s ','
    echo ""
    echo "See full analysis output above for best configurations!"
fi