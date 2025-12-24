# Training Routing & Plan System - Summary

**Single, coherent description of the routing + training plan system for README/INTEGRATION_SUMMARY.**

## Overview

FoxML's training routing system makes reproducible, config-driven decisions about where to train models for each `(target, symbol)` pair. The system flows from metrics aggregation through routing decisions to actionable training plans that are automatically consumed by the training phase.

## End-to-End Flow

```text
Feature Selection + Stability Metrics
         │
         ▼
  METRICS/routing_candidates.(parquet|csv)
         │
         ▼
  Routing Plan Generator
         │
         ▼
    METRICS/routing_plan/
       routing_plan.{json,yaml,md}
         │
         ▼
  Training Plan Generator
         │
         ▼
    METRICS/training_plan/
       master_training_plan.json  ← Single source of truth
       training_plan.{json,yaml,md}  ← Convenience mirrors
         │
         ▼
  Training Plan Consumer (intelligent_trainer)
         │
         ▼
  Filtered Training Phase
    - CS targets filtered by plan ✅
    - Symbol-specific: planned, execution pending ⚠️
```

## Key Components

### 1. Metrics Aggregator
- Collects metrics from feature selection, stability analysis, and leakage detection
- Writes `METRICS/routing_candidates.parquet` (or `.csv` if `pyarrow` unavailable)
- **Parquet → CSV fallback** prevents runtime crashes when `pyarrow` isn't installed

### 2. Routing Plan Generator
- Consumes routing candidates + routing config
- Produces `METRICS/routing_plan/` with routing decisions per `(target, symbol)`
- Encodes decisions: CS jobs, symbol eligibility, route labels (`ROUTE_CROSS_SECTIONAL`, `ROUTE_SYMBOL_SPECIFIC`, `ROUTE_BOTH`, etc.)

### 3. Training Plan Generator
- Converts routing decisions into **explicit training jobs**
- Output: `METRICS/training_plan/master_training_plan.json` (single source of truth)
- Each job includes: `job_id`, `target`, `symbol`, `route`, `training_type`, `model_families`, `priority`, `reason`, `metadata`
- Also emits summary stats: jobs by route, type, and priority bucket

### 4. Training Plan Consumer
- Loads `master_training_plan.json` from disk
- Provides filtering helpers:
  - Filter allowed **targets** (for CS) ✅ **Implemented**
  - Filter allowed `(target, symbol)` combos (for symbol-specific) ⚠️ **Planned**
- Integrated into `intelligent_trainer.py`:
  - After routing plan → training plan are generated
  - Before training: reads plan and filters targets
  - Logs: `"📋 Training plan filter applied: 10 → 7 targets"`
  - **Backward compatible**: if plan missing, trains all targets

## Artifacts

### Routing Candidates
- **Location:** `METRICS/routing_candidates.parquet` (or `.csv`)
- **Format:** Parquet preferred, CSV fallback if `pyarrow` unavailable
- **Content:** Aggregated metrics from feature selection and stability

### Routing Plan
- **Location:** `METRICS/routing_plan/`
- **Files:** `routing_plan.json`, `.yaml`, `.md`
- **Content:** Routing decisions per `(target, symbol)` pair

### Training Plan
- **Location:** `METRICS/training_plan/`
- **Master File:** `master_training_plan.json` (single source of truth)
- **Convenience Mirrors:** `training_plan.json`, `.yaml`, `.md`
- **Content:** Actionable training jobs with full specifications

## Runtime Integration

### Automatic Flow

1. **Feature Selection** completes for all targets
2. **Routing Plan Generation** (automatic)
   - Aggregates metrics → `METRICS/routing_candidates.parquet`
   - Generates routing decisions → `METRICS/routing_plan/`
3. **Training Plan Generation** (automatic, if `generate_training_plan=True`)
   - Converts routing → job specs → `METRICS/training_plan/`
4. **Training Plan Consumption** (automatic)
   - Loads `master_training_plan.json`
   - Filters targets for CS training ✅
   - Filters symbols per target ⚠️ (planned)
5. **Training Execution**
   - Only approved jobs are executed
   - Logs show filtering results

### Example Log Output

```
[ROUTER] ✅ Training routing plan generated - see METRICS/routing_plan/ for details
[ROUTER] ✅ Training plan generated: METRICS/training_plan
[ROUTER]    Total jobs: 42
[ROUTER]    CS jobs: 5
[ROUTER]    Symbol jobs: 37

[FILTER] 📋 Training plan filter applied: 10 → 7 targets
```

## Current Status

### ✅ Implemented

- **Parquet → CSV fallback** for metrics aggregation
- **Routing plan generation** with full decision logic
- **Training plan generation** (job specs + summaries)
- **Training plan consumption** for **cross-sectional target filtering**
- **Automatic integration** with `intelligent_trainer.py`
- **Backward compatibility** (if plan missing, trains all targets)
- **Documentation** moved and updated under `DOCS/02_reference/training_routing/`

### ⚠️ Planned / Future Enhancements

- **Symbol-specific execution filtering**: Symbol-specific jobs exist in plan but aren't fully enforced at execution time
- **Model-family-level filtering**: Training plan includes `model_families`, but training loop still trains all families
- **Master plan structure**: Formalize `master_training_plan.json` as canonical source with derived views
- **Advanced routing logic**: Some stability-state rules and experimental-lane behaviors may need clearer encoding

## Configuration

**Routing Config:** `CONFIG/training_config/routing_config.yaml`

Key settings:
- Score thresholds (`min_score`, `strong_score`) for CS and symbol-specific
- Stability requirements (`stability_allowlist`)
- Sample size minimums (`min_sample_size`)
- Experimental lane settings (`enable_experimental_lane`, `max_fraction_symbols_per_target`)
- Both-strong behavior (`both_strong_behavior`: `ROUTE_BOTH`, `PREFER_CS`, `PREFER_SYMBOL`)

## Design Principles

1. **One master training plan = single source of truth** - Training phase only reads `master_training_plan.json`
2. **Many derived artifacts = views** - Other files are for humans, dashboards, and external schedulers
3. **Backward compatible** - If training plan missing, behavior falls back to old (train all targets)
4. **Non-blocking** - Routing/training plan generation fails gracefully if metrics unavailable
5. **Config-driven** - All routing decisions driven by `routing_config.yaml`, not hardcoded

## Files

**Core Implementation:**
- `TRAINING/orchestration/metrics_aggregator.py` - Metrics collection
- `TRAINING/orchestration/training_router.py` - Routing decisions
- `TRAINING/orchestration/training_plan_generator.py` - Job spec generation
- `TRAINING/orchestration/training_plan_consumer.py` - Plan consumption
- `TRAINING/orchestration/routing_integration.py` - Integration hooks
- `TRAINING/orchestration/intelligent_trainer.py` - Training orchestrator (integration)

**Configuration:**
- `CONFIG/training_config/routing_config.yaml` - Routing policy

**Documentation:**
- `DOCS/02_reference/training_routing/ARCHITECTURE.md` - Complete architecture
- `DOCS/02_reference/training_routing/IMPLEMENTATION_STATUS.md` - What's done vs. TODO
- `DOCS/02_reference/training_routing/MASTER_TRAINING_PLAN.md` - Master plan structure
- `DOCS/02_reference/training_routing/README.md` - User-facing guide

## Summary

**What you've built matches the architecture:** metrics → routing → training plan → filtered training. The remaining work is mostly deepening the routing logic and wiring symbol-specific/model-family filtering into the executor, not re-architecting anything.

**Net:** The system is **fully functional for cross-sectional training filtering** and **ready for symbol-specific execution filtering** when that pipeline integration is completed.
