# Training Routing & Plan System Architecture

**Complete end-to-end architecture of the metrics → routing → training plan → training pipeline.**

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
       by_target/<target>.json  ← Derived views
       by_symbol/<symbol>.json
       by_type/<type>.json
         │
         ▼
  Training Plan Consumer (intelligent_trainer)
         │
         ▼
  Filtered Training Phase
    - CS targets filtered by plan ✅
    - Symbol-specific: planned, execution pending ⚠️
```

## Component Architecture

### 1. Metrics Aggregator (`metrics_aggregator.py`)

**Purpose:** Collects and aggregates metrics from feature selection, stability analysis, and leakage detection into a unified routing candidates DataFrame.

**Inputs:**
- `feature_selections/{target}/model_metadata.json` - Per-symbol scores
- `feature_selections/{target}/target_confidence.json` - Cross-sectional confidence
- Stability snapshots from `TRAINING/stability/feature_importance/`
- Leakage detection outputs (if available)

**Outputs:**
- `METRICS/routing_candidates.parquet` (preferred) or `.csv` (fallback)
- `METRICS/routing_candidates.json` (human-readable)

**Key Features:**
- **Parquet → CSV fallback**: If `pyarrow` is not installed, automatically falls back to CSV with the same schema
- Aggregates both cross-sectional (pooled) and symbol-specific metrics
- Includes stability classification, sample sizes, leakage status

**Schema:**
```python
{
    "target": str,
    "symbol": Optional[str],  # None for CS
    "mode": "CROSS_SECTIONAL" | "SYMBOL",
    "score": float,
    "score_ci_low": Optional[float],
    "score_ci_high": Optional[float],
    "stability": "STABLE" | "DRIFTING" | "DIVERGED" | "UNKNOWN",
    "sample_size": int,
    "leakage_status": "SAFE" | "SUSPECT" | "BLOCKED" | "UNKNOWN",
    "failed_model_families": List[str],
    "stability_metrics": Dict[str, float],  # mean_overlap, std_overlap, mean_tau, std_tau
    "timestamp": str,
    "git_commit": str
}
```

### 2. Training Router (`training_router.py`)

**Purpose:** Makes routing decisions for each `(target, symbol)` pair based on metrics and config-driven rules.

**Inputs:**
- Routing candidates DataFrame
- Routing configuration (`CONFIG/training_config/routing_config.yaml`)

**Outputs:**
- `METRICS/routing_plan/routing_plan.json` - Machine-readable plan
- `METRICS/routing_plan/routing_plan.yaml` - YAML format
- `METRICS/routing_plan/routing_plan.md` - Human-readable report

**Decision Logic (Priority-Ordered):**

1. **Hard blocks**: Leakage detected, insufficient data → `ROUTE_BLOCKED`
2. **CS strong, local weak**: → `ROUTE_CROSS_SECTIONAL`
3. **Local strong, CS weak**: → `ROUTE_SYMBOL_SPECIFIC`
4. **Both strong**: → `ROUTE_BOTH` (or prefer one based on config)
5. **Experimental lane**: Unstable but promising → `ROUTE_EXPERIMENTAL_ONLY`
6. **Fallback**: → `ROUTE_BLOCKED`

**Route States:**
- `ROUTE_CROSS_SECTIONAL` - Train CS models only
- `ROUTE_SYMBOL_SPECIFIC` - Train symbol-specific models only
- `ROUTE_BOTH` - Train both (ensemble approach)
- `ROUTE_EXPERIMENTAL_ONLY` - Experimental lane (unstable but promising)
- `ROUTE_BLOCKED` - No training (leakage, insufficient data, etc.)

**Signal States:**
- `STRONG` - Meets strong score threshold + stability requirements
- `WEAK_BUT_OK` - Meets minimum score threshold + stability requirements
- `EXPERIMENTAL` - Meets experimental lane thresholds
- `DISALLOWED` - Fails minimum requirements

### 3. Training Plan Generator (`training_plan_generator.py`)

**Purpose:** Converts routing decisions into actionable training job specifications.

**Inputs:**
- Routing plan (from `training_router.py`)
- Model families list (optional, defaults to `["lightgbm", "xgboost"]`)

**Outputs:**
- `METRICS/training_plan/master_training_plan.json` - **Canonical plan (single source of truth)**
- `METRICS/training_plan/training_plan.json` - Convenience mirror
- `METRICS/training_plan/training_plan.yaml` - YAML format
- `METRICS/training_plan/training_plan.md` - Human-readable report
- `METRICS/training_plan/by_target/<target>.json` - Per-target views (future)
- `METRICS/training_plan/by_symbol/<symbol>.json` - Per-symbol views (future)
- `METRICS/training_plan/by_type/<type>.json` - Per-type views (future)

**Job Specification Schema:**
```jsonc
{
  "metadata": {
    "generated_at": "2025-12-11T18:45:00Z",
    "run_id": "20251211_184500",
    "git_commit": "abc1234",
    "config_hash": "routing_cfg_v3",
    "routing_plan_path": "METRICS/routing_plan/routing_plan.json",
    "metrics_snapshot": "METRICS/routing_candidates.parquet",
    "total_jobs": 42,
    "model_families": ["lightgbm", "xgboost"]
  },
  "jobs": [
    {
      "job_id": "cs_y_will_swing_low_10m_0.20",
      "target": "y_will_swing_low_10m_0.20",
      "symbol": null,
      "route": "ROUTE_CROSS_SECTIONAL",
      "training_type": "cross_sectional",
      "model_families": ["lightgbm", "xgboost"],
      "priority": 90,
      "reason": "CS strong and stable",
      "metadata": {
        "cs_state": "STRONG",
        "sample_size": 123456,
        "score": 0.62
      }
    },
    {
      "job_id": "sym_AAPL_y_will_swing_low_10m_0.20",
      "target": "y_will_swing_low_10m_0.20",
      "symbol": "AAPL",
      "route": "ROUTE_SYMBOL_SPECIFIC",
      "training_type": "symbol_specific",
      "model_families": ["lightgbm"],
      "priority": 80,
      "reason": "Local strong, CS weak for this symbol",
      "metadata": {
        "local_state": "STRONG",
        "sample_size": 8000,
        "score": 0.66
      }
    }
  ],
  "summary": {
    "by_route": {
      "ROUTE_CROSS_SECTIONAL": 5,
      "ROUTE_SYMBOL_SPECIFIC": 30,
      "ROUTE_BOTH": 7
    },
    "by_type": {
      "cross_sectional": 5,
      "symbol_specific": 37
    },
    "by_priority": {
      90: 5,
      80: 20,
      70: 17
    },
    "total_cs_jobs": 5,
    "total_symbol_jobs": 37,
    "total_blocked": 0
  }
}
```

**Key Design Principle:**
- **`master_training_plan.json` is the single source of truth** - the training phase only reads this file
- All other files are **derived views** for humans, dashboards, and external schedulers
- The training phase should never trust any file other than the master plan

### 4. Training Plan Consumer (`training_plan_consumer.py`)

**Purpose:** Loads and applies the training plan to filter which targets/symbols should be trained.

**API:**
```python
def load_training_plan(training_plan_dir: Path) -> Optional[Dict[str, Any]]
def filter_targets_by_training_plan(targets: List[str], training_plan: Dict, training_type: str) -> List[str]
def filter_symbols_by_training_plan(target: str, symbols: List[str], training_plan: Dict) -> List[str]
def should_train_target_symbol(training_plan: Optional[Dict], target: str, symbol: Optional[str], training_type: Optional[str]) -> bool
def apply_training_plan_filter(targets: List[str], symbols: List[str], training_plan_dir: Optional[Path], use_cs_plan: bool, use_symbol_plan: bool) -> Tuple[List[str], Dict[str, List[str]]]
```

**Integration:**
- Automatically called by `intelligent_trainer.py` before training starts
- Filters targets for cross-sectional training ✅ **Implemented**
- Filters symbols per target for symbol-specific training ⚠️ **Planned**
- Backward compatible: if training plan missing, trains all targets (old behavior)

### 5. Routing Integration (`routing_integration.py`)

**Purpose:** Main integration hooks that connect the routing system to the training pipeline.

**Key Function:**
```python
def generate_routing_plan_after_feature_selection(
    output_dir: Path,
    targets: List[str],
    symbols: List[str],
    routing_config_path: Optional[Path] = None,
    generate_training_plan: bool = True,  # Default: True
    model_families: Optional[List[str]] = None
) -> Optional[Dict[str, Any]]
```

**Flow:**
1. Aggregates metrics from feature selection outputs
2. Saves routing candidates (Parquet with CSV fallback)
3. Generates routing plan
4. Generates training plan (if `generate_training_plan=True`)
5. Returns routing plan dict

**Called by:** `intelligent_trainer.py` after feature selection completes

## Directory Structure

```
METRICS/
├── routing_candidates.parquet  (or .csv if pyarrow unavailable)
├── routing_candidates.json
│
├── routing_plan/
│   ├── routing_plan.json       # Machine-readable routing decisions
│   ├── routing_plan.yaml       # YAML format
│   └── routing_plan.md          # Human-readable report
│
└── training_plan/
    ├── master_training_plan.json  # ⭐ Single source of truth
    ├── training_plan.json         # Convenience mirror
    ├── training_plan.yaml         # YAML format
    ├── training_plan.md           # Human-readable report
    │
    ├── by_target/                 # Derived views (future)
    │   └── <target>.json
    ├── by_symbol/                 # Derived views (future)
    │   └── <symbol>.json
    └── by_type/                   # Derived views (future)
        ├── cross_sectional.json
        └── symbol_specific.json
```

## Configuration

**Routing Config:** `CONFIG/training_config/routing_config.yaml`

Key settings:
- Score thresholds (`min_score`, `strong_score`) for CS and symbol-specific
- Stability requirements (`stability_allowlist`)
- Sample size minimums (`min_sample_size`)
- Experimental lane settings (`enable_experimental_lane`, `max_fraction_symbols_per_target`)
- Both-strong behavior (`both_strong_behavior`: `ROUTE_BOTH`, `PREFER_CS`, `PREFER_SYMBOL`)
- Stability classification thresholds
- Feature safety requirements

## Integration with Training Phase

### Automatic Flow (Current Implementation)

1. **Feature Selection** completes for all targets
2. **Routing Plan Generation** (`generate_routing_plan_after_feature_selection`)
   - Aggregates metrics
   - Generates routing decisions
   - Saves to `METRICS/routing_plan/`
3. **Training Plan Generation** (automatic if `generate_training_plan=True`)
   - Converts routing decisions to job specs
   - Saves to `METRICS/training_plan/`
4. **Training Plan Consumption** (in `intelligent_trainer.py`)
   - Loads `master_training_plan.json`
   - Filters targets for CS training ✅
   - Filters symbols per target ⚠️ (planned)
5. **Training Execution**
   - Only approved jobs are executed
   - Logs show filtering results: `"Training plan filter applied: 10 → 7 targets"`

### Backward Compatibility

- If training plan is missing (e.g., older runs, debugging), behavior is **backward compatible**: all targets are trained as before
- No breaking changes to existing workflows

## Implementation Status

See `IMPLEMENTATION_STATUS.md` for detailed breakdown of what's implemented vs. what's planned.

## See Also

- `README.md` - User-facing guide
- `IMPLEMENTATION_STATUS.md` - What's done vs. TODO
- `MASTER_TRAINING_PLAN.md` - Master plan structure details
- `ROUTING_SYSTEM_SUMMARY.md` - Implementation details
- `ERRORS_FIXED.md` - Known issues and fixes
- `INTEGRATION_SUMMARY.md` - Integration with training phase
