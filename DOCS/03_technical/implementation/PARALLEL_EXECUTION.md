# Parallel Execution for Target Ranking and Feature Selection

## Overview

This document describes the parallel execution infrastructure added to speed up target ranking and feature selection operations.

**NEW (2025-12-12)**: GPU acceleration is now enabled for target ranking and feature selection in addition to parallel execution. LightGBM, XGBoost, and CatBoost automatically use GPU when available, providing 10-50x speedup on large datasets.

**NEW (2025-12-20)**: Unified threading utilities from `TRAINING/common/threads.py` are now used across all models in feature selection and target ranking. This provides GPU-aware thread management (automatically limits CPU threads when GPU is enabled) and optimal OMP/MKL thread allocation based on model family type. The threading utilities read from `CONFIG/pipeline/threading.yaml`, which is shared with the training pipeline, ensuring consistent thread management across all phases.

## Architecture

### Components

1. **`TRAINING/common/parallel_exec.py`**: Core parallel execution utilities
   - `execute_parallel()`: Generic parallel execution using ProcessPoolExecutor or ThreadPoolExecutor
   - `execute_parallel_with_context()`: Parallel execution with shared context
   - `get_max_workers()`: Config-aware worker count calculation

2. **Integration Points**:
   - `TRAINING/ranking/target_ranker.py`: Parallel target evaluation
   - `TRAINING/ranking/feature_selector.py`: Parallel symbol processing

### Configuration

Threading configuration is shared across feature selection, target ranking, and model training. Settings are in `CONFIG/pipeline/threading.yaml`:

```yaml
threading:
  # Default Thread Counts
  defaults:
    default_threads: null  # null = calculated as max(1, cpu_count() - 1)
    mkl_threads: 1  # Default MKL threads
    openblas_threads: 1  # Default OpenBLAS threads
  
  # Thread Planning
  planning:
    reserve_threads: 1  # Reserve threads for system
    min_threads: 1  # Minimum threads to allocate
    max_threads: null  # null = no limit
  
  # Parallel Execution (Task-Level Parallelization)
  parallel:
    max_workers_process: null  # Auto-detect for CPU-bound tasks
    max_workers_thread: null   # Auto-detect for I/O-bound tasks
    enabled: true               # Master switch
```

**Note**: The threading utilities (`TRAINING/common/threads.py`) used by feature selection and target ranking read from this same config file. All models automatically use `plan_for_family()` to determine optimal OMP/MKL thread allocation and `thread_guard()` for GPU-aware thread limiting.

Task-specific flags:
- `CONFIG/target_configs.yaml`: `multi_target.parallel_targets: false` (default: sequential)
- `CONFIG/feature_selection/multi_model.yaml`: `parallel_symbols: false` (default: sequential)

## Usage

### Target Ranking

When `parallel_targets: true` is set in config:
- Targets are evaluated in parallel using ProcessPoolExecutor
- Each target evaluation is CPU-bound and independent
- Results are collected in completion order

### Feature Selection

When `parallel_symbols: true` is set in config:
- Symbols are processed in parallel using ProcessPoolExecutor
- Each symbol's feature selection is CPU-bound and independent
- Results are aggregated after all symbols complete

### Symbol-Specific Evaluation

For I/O-bound symbol-specific evaluations:
- Uses ThreadPoolExecutor (lighter weight, shared memory)
- Suitable for parallel symbol evaluation within a target

## Safety

- Respects existing config flags (`parallel_targets`, `parallel_symbols`)
- Falls back to sequential execution if:
  - Config flag is false
  - Only 1 item to process
  - max_workers=1
  - Parallel execution disabled globally
- Error handling: Failed items are logged but don't stop other tasks

## Performance Notes

- **CPU-bound tasks** (target evaluation, symbol processing): Use ProcessPoolExecutor
- **I/O-bound tasks** (symbol-specific evaluation): Use ThreadPoolExecutor
- Worker count is auto-calculated from available CPUs and config
- Threading infrastructure ensures no oversubscription with model training threads
