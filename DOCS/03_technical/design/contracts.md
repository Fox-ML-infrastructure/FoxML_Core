# Contracts Reference

This document describes the **data contracts** (dataclasses, protocols) that define stage boundaries in the modeling pipeline.

## Purpose

Contracts make stage I/O explicit:
- Each stage has typed inputs and outputs
- Invariants are enforced at construction time
- Callers know exactly what to provide and expect

---

## Existing Contracts

### WriteScope

**Location:** [`TRAINING/orchestration/utils/scope_resolution.py:91`](../../../TRAINING/orchestration/utils/scope_resolution.py)

**Purpose:** Validated bundle for artifact writes. Prevents scope contamination bugs.

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `view` | `View` (enum) | CROSS_SECTIONAL or SYMBOL_SPECIFIC |
| `universe_sig` | `str` | Hash of symbol universe (required, never None) |
| `symbol` | `Optional[str]` | Symbol ticker (None for CS, required for SS) |
| `purpose` | `ScopePurpose` | FINAL or ROUTING_EVAL |
| `stage` | `Stage` (enum) | TARGET_RANKING, FEATURE_SELECTION, or TRAINING |

**Invariants (enforced in `__post_init__`):**
1. `universe_sig` is required (never None or empty)
2. CS view must have `symbol=None`
3. SS view must have non-empty `symbol`
4. `view` must be `View` enum (not raw string)
5. `stage` must be `Stage` enum (not raw string)

**Factory Methods:**
- `WriteScope.for_cross_sectional(universe_sig, stage)` - Create CS scope
- `WriteScope.for_symbol_specific(universe_sig, symbol, stage)` - Create SS scope
- `WriteScope.for_routing_eval(view, universe_sig, stage, symbol)` - Create routing eval scope
- `WriteScope.from_resolved_data_config(resolved_data_config, stage, symbol)` - Create from SST config

**Usage:**
```python
scope = WriteScope.for_symbol_specific(
    universe_sig="abc123",
    symbol="AAPL",
    stage=Stage.TRAINING
)
tracker.log_run(scope, metrics)  # Pass scope, not loose args
```

---

### RunContext

**Location:** [`TRAINING/orchestration/utils/run_context.py:59`](../../../TRAINING/orchestration/utils/run_context.py)

**Purpose:** Complete context for a training run. Eliminates manual parameter passing.

**Fields (key ones):**

| Field | Type | Description |
|-------|------|-------------|
| `X` | `np.ndarray / pd.DataFrame` | Feature matrix |
| `y` | `np.ndarray / pd.Series` | Target vector |
| `feature_names` | `List[str]` | Feature column names |
| `symbols` | `List[str]` | Symbol universe |
| `time_vals` | `np.ndarray` | Timestamp values |
| `target_column` | `str` | Target column name |
| `horizon_minutes` | `float` | Prediction horizon |
| `cv_splitter` | `Any` | CV splitter object |
| `stage` | `str` | Current pipeline stage |
| `view` | `Optional[str]` | **(ISSUE: should be View enum)** |
| `resolved_mode` | `Optional[str]` | **(ISSUE: duplicates view, should be removed)** |
| `data_scope` | `Optional[str]` | PANEL or SINGLE_SYMBOL |

**Invariants:**
- Validates required fields based on `reproducibility_mode`
- Required for COHORT_AWARE mode: X, y, feature_names, symbols, time_vals, target_column, horizon_minutes

**Known Issues:**
- Has both `.view` and `.resolved_mode` fields (pick one)
- Uses string types where enums should be used

---

### TaskSpec

**Location:** [`TRAINING/orchestration/routing/target_router.py:20`](../../../TRAINING/orchestration/routing/target_router.py)

**Purpose:** Specification for how to train a specific target.

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `task` | `str` | 'regression', 'binary', 'multiclass', 'ranking', 'survival' |
| `objective` | `str` | Learner-specific objective key |
| `metrics` | `List[str]` | Evaluation metrics |
| `needs_group` | `bool` | Ranking tasks need group sizes |
| `n_classes` | `Optional[int]` | For multiclass |
| `class_weighting` | `Optional[str]` | 'balanced' or None |
| `label_type` | `str` | numpy dtype for labels |

**Usage:**
```python
from TRAINING.orchestration.routing.target_router import spec_from_target

spec = spec_from_target("fwd_ret_60m")
# Returns: TaskSpec(task='regression', objective='regression', ...)

spec = spec_from_target("y_will_peak_60m_0.8")
# Returns: TaskSpec(task='binary', objective='binary', ...)
```

**Invariants:**
- Pattern matching determines task type (order matters - more specific patterns first)
- Unknown targets default to regression

---

### RoutingDecisions

**Location:** [`TRAINING/orchestration/utils/view_decisions.py:24`](../../../TRAINING/orchestration/utils/view_decisions.py)

**Purpose:** Wrapper for routing_decisions.json with single authoritative accessor.

**Key Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `get_final_route(target)` | `Optional[str]` | Authoritative final route for target |
| `get_all_routes()` | `Dict[str, str]` | All final routes |
| `get_target_details(target)` | `Optional[Dict]` | Detailed routing info (backward compat) |
| `load(path, strict)` | `RoutingDecisions` | Load from file |
| `from_output_dir(output_dir, strict)` | `Optional[RoutingDecisions]` | Load from output directory |

**Invariants:**
- `get_final_route()` ONLY reads from `final_routes`, never `routing_decisions[target].route`
- Strict mode checks for dual-source drift between `final_routes` and `routing_decisions`

**Schema v2 format:**
```json
{
    "schema_version": 2,
    "metadata": {
        "generated_at": "...",
        "universe_sig": "..."
    },
    "final_routes": {
        "fwd_ret_1d": "CROSS_SECTIONAL",
        "fwd_ret_60m": "SYMBOL_SPECIFIC"
    },
    "routing_decisions": { ... }
}
```

---

## Missing Contracts

These areas lack explicit contracts and should be addressed:

### Budget Computation

**Problem:** Lookback/leakage budgets computed in multiple places.

**Current locations:**
- `run_context.py:125` - `derive_purge_embargo()` method
- `ranking/utils/resolved_config.py` - `derive_purge_embargo()` function
- Various call sites compute budgets inline

**Proposed solution:** Single `BudgetSpec` dataclass computed once during config resolution.

### Artifact Path Builder

**Problem:** Multiple modules build artifact paths.

**Current locations:**
- `orchestration/utils/output_layout.py`
- `orchestration/utils/artifact_paths.py`
- `orchestration/utils/target_first_paths.py`

**Proposed solution:** Consolidate to single `ArtifactPathBuilder` with one entrypoint.

---

## Stage Contract Summary

| Stage | Input Contract | Output Contract | Invariants Enforced |
|-------|---------------|-----------------|---------------------|
| TARGET_RANKING | `RunContext` + data | `RoutingDecisions` | Lookback budgets, leakage filter |
| FEATURE_SELECTION | `RunContext` + `RoutingDecisions` | Feature list + provenance | Stability thresholds |
| TRAINING | `RunContext` + `TaskSpec` + features | Model artifacts | Deterministic seeds, fingerprint |

