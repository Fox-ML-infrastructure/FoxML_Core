#!/bin/bash
#
# Deterministic run launcher - sets environment variables BEFORE Python starts.
#
# Usage:
#   bin/run_deterministic.sh your_script.py [args...]
#   bin/run_deterministic.sh -m your.module [args...]
#
# This launcher sets:
#   - PYTHONHASHSEED (must be set before interpreter starts)
#   - Thread environment variables (OMP, MKL, etc.)
#   - REPRO_MODE=strict (opt-in to strict mode)
#
# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

set -e

# ============================================================================
# CONFIGURATION
# ============================================================================

# Seed for PYTHONHASHSEED (can be overridden)
export PYTHONHASHSEED="${PYTHONHASHSEED:-42}"

# Thread control (single-threaded for strict determinism)
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

# MKL settings for reproducibility
export MKL_THREADING_LAYER="${MKL_THREADING_LAYER:-GNU}"
export MKL_CBWR="${MKL_CBWR:-COMPATIBLE}"

# CUDA determinism
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

# STRICT MODE IS OPT-IN: This launcher enables it
# Normal runs without the launcher use best_effort by default
export REPRO_MODE="${REPRO_MODE:-strict}"

# ============================================================================
# LOGGING
# ============================================================================

echo "🔒 Determinism env vars set:"
echo "   PYTHONHASHSEED=$PYTHONHASHSEED"
echo "   REPRO_MODE=$REPRO_MODE"
echo "   OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "   MKL_NUM_THREADS=$MKL_NUM_THREADS"

# ============================================================================
# EXECUTION
# ============================================================================

# Run the provided command (can be python script or any executable)
# Usage:
#   bin/run_deterministic.sh python script.py     # explicit python
#   bin/run_deterministic.sh ./script.py          # if script is executable
#   bin/run_deterministic.sh -m module            # module mode
if [[ "$1" == "-m" ]]; then
    # Module mode: bin/run_deterministic.sh -m your.module
    exec python "$@"
elif [[ "$1" == *.py ]]; then
    # Script mode: bin/run_deterministic.sh script.py
    exec python "$@"
else
    # Direct executable mode: bin/run_deterministic.sh /path/to/python script.py
    exec "$@"
fi
