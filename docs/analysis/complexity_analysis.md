# Training Pipeline Complexity Analysis

**Date**: 2025-01-16  
**Branch**: `analysis/complexity-review`  
**Goal**: Analyze the training pipeline architecture to identify accidental vs essential complexity, and propose targeted improvements.

---

## Executive Summary

The training pipeline is **correctly complex** for its domain (ML infra with strict leakage controls), but has **4 areas of accidental complexity** that can be reduced without losing safety:

1. **Unknown lookback warnings** - Logged as errors even when expected pre-enforcement
2. **Repeated budget/config recomputation** - Multiple redundant calls throughout pipeline
3. **Sample cap vs feature generation mismatch** - Computing long-window features that will always be dropped
4. **Degenerate folds handling** - Treated as surprise rather than first-class expected case

---

## Architecture Overview

### Current Pipeline Flow

```
Feature Generation
  ↓
Feature Filtering (target/label patterns, always-excluded, registry allowlist)
  ↓
SAFE_CANDIDATES (fingerprint: SAFE_CANDIDATES)
  ↓
Lookback Gatekeeper (apply_lookback_cap, cap=240m)
  ↓
POST_GATEKEEPER (fingerprint: POST_GATEKEEPER)
  ↓
Feature Pruning (correlation, variance, etc.)
  ↓
POST_PRUNE (fingerprint: POST_PRUNE)
  ↓
Budget Recompute (create_resolved_config, compute_budget)
  ↓
MODEL_TRAIN_INPUT (fingerprint: MODEL_TRAIN_INPUT)
  ↓
CV Splitter (PurgedTimeSeriesSplit)
  ↓
Model Training (CatBoost, XGBoost, etc.)
```

### Key Components

#### 1. Lookback Cap Enforcement (`TRAINING/utils/lookback_cap_enforcement.py`)

**Function**: `apply_lookback_cap()`
- **Input**: Feature list, cap_minutes (e.g., 240m), policy ("strict"|"drop"|"warn")
- **Output**: `LookbackCapResult` with safe_features, quarantined_features, budget, canonical_map
- **Stages**: FS_PRE, FS_POST, GATEKEEPER, POST_PRUNE

**Current behavior**:
- Unknown lookback (inf) → quarantined in strict mode
- Logs warning: "Unknown lookback (inf) should have been dropped/quarantined" even when expected pre-enforcement

#### 2. Budget Computation (`TRAINING/utils/leakage_budget.py`)

**Function**: `compute_budget()`
- **Input**: Feature names, interval_minutes, horizon_minutes, canonical_lookback_map (optional)
- **Output**: `LeakageBudget`, set_fingerprint, order_fingerprint
- **Cache**: Has budget cache (line 686-699) but still called multiple times

**Called at**:
- `apply_lookback_cap()` (line 281)
- `_enforce_final_safety_gate()` (gatekeeper)
- `train_and_evaluate_models()` POST_PRUNE (line 1300)
- `create_resolved_config()` (if recompute_lookback=True)

#### 3. Resolved Config (`TRAINING/utils/resolved_config.py`)

**Function**: `create_resolved_config()`
- **Input**: Feature names, lookback_max, interval, horizon, etc.
- **Output**: `ResolvedConfig` with purge_minutes, embargo_minutes, feature_lookback_max_minutes
- **Recomputed**: Multiple times throughout pipeline (pre-gatekeeper, post-gatekeeper, post-prune)

**Current flow**:
1. Pre-gatekeeper: `create_resolved_config()` with pre-enforcement lookback (may be inflated)
2. Post-gatekeeper: Not explicitly recomputed (uses pre-gatekeeper config)
3. Post-prune: `create_resolved_config()` again (line 1277) with final lookback

#### 4. CV Splitter (`TRAINING/utils/purged_time_series_split.py`)

**Function**: `PurgedTimeSeriesSplit`
- **Input**: n_splits, purge_time, embargo_time
- **Output**: Train/test splits with temporal gap
- **Used in**: `train_and_evaluate_models()` (line 1946)

**Current behavior**:
- CatBoost CV returns NaN → caught in exception handler (line 3306-3319)
- Logged as debug: "Target degenerate in some CV folds"
- No pre-CV validation for degenerate folds

#### 5. Feature Generation vs Sample Cap

**Current mismatch**:
- Sample cap: `max_cs_samples=1000` (from config)
- Data: 1000 rows @ 5m ≈ 3.5 days
- Features generated: 20d/60d rolling windows (e.g., `relative_performance_20d`, `volatility_60d`)
- Result: Long-window features always dropped by cap, but computed anyway

**Location**: `DATA_PROCESSING/features/simple_features.py` (line 663-683)

---

## Issues Identified

### Issue 1: Unknown Lookback Warning (Misleading)

**Location**: `TRAINING/utils/lookback_cap_enforcement.py` (line 233-239)

**Current behavior**:
```python
elif lookback == float("inf"):
    # Unknown lookback - treat as violation (same as over-cap)
    if log_mode == "debug":
        logger.debug(f"   {stage}: {feat_name} → unknown lookback (quarantined)")
    unknown_features.append(feat_name)
    quarantined_features.append(feat_name)
    quarantined_dict[feat_name] = float("inf")
```

**Problem**: 
- Unknown lookback is **expected** pre-enforcement (before gatekeeper)
- But warning suggests it's a bug: "should have been dropped/quarantined"
- Creates noise in logs

**Proposed fix**:
- Pre-enforcement: Log as INFO + "will be quarantined at gatekeeper"
- Post-enforcement: Log as WARNING/ERROR (actual violation)

### Issue 2: Repeated Budget/Config Recomputation

**Current call sites**:

1. **Pre-gatekeeper** (`evaluate_target_predictability`, line ~5000):
   ```python
   resolved_config = create_resolved_config(...)  # With pre-enforcement lookback
   ```

2. **Gatekeeper** (`_enforce_final_safety_gate`, line 403):
   ```python
   apply_lookback_cap(...)  # Calls compute_budget() internally
   ```

3. **Post-prune** (`train_and_evaluate_models`, line 1277):
   ```python
   resolved_config = create_resolved_config(...)  # Recompute with final lookback
   ```

4. **Post-prune policy check** (`train_and_evaluate_models`, line 1300):
   ```python
   budget, _, _ = compute_budget(...)  # Recompute budget again
   ```

**Problem**:
- Budget computed 3-4 times for same feature set
- Config recomputed 2-3 times
- Each recomputation logs config trace, budget summary, etc.
- Creates log noise and minor performance overhead

**Proposed fix**:
- Create **FeatureSet artifact** (features + fingerprint + canonical_map + budget + removal_reasons)
- Pass artifact through stages instead of recomputing
- Single canonical source of truth per stage

### Issue 3: Sample Cap vs Feature Windows Mismatch

**Current behavior**:
- Feature generation: Always generates 20d/60d features (line 663-683 in `simple_features.py`)
- Sample cap: `max_cs_samples=1000` (≈3.5 days @ 5m)
- Gatekeeper: Drops 20d/60d features (exceed 240m cap)
- Result: Wasted compute on features that will always be dropped

**Proposed fix**:
- Option A: Don't generate long-window families when `lookback_budget_minutes=240`
- Option B: Generate conditionally based on sample cap (if cap < window_size, skip)
- Option C: Keep generation but add early filter (skip families that exceed cap)

### Issue 4: Degenerate Folds (Not First-Class)

**Current behavior** (`model_evaluation.py`, line 3302-3319):
```python
try:
    scores = cross_val_score(model, X, y, cv=tscv, scoring=scoring, ...)
    valid_scores = scores[~np.isnan(scores)]
    primary_score = valid_scores.mean() if len(valid_scores) > 0 else np.nan
except (ValueError, TypeError) as e:
    if "Invalid classes" in error_str or "Expected" in error_str:
        logger.debug(f"    CatBoost: Target degenerate in some CV folds")
        primary_score = np.nan
```

**Problem**:
- Degenerate folds are **expected** for rare targets (swing/breakout labels)
- But handled as exception/surprise
- No pre-CV validation
- No clear fallback policy (fewer folds? different splitter? skip CV?)

**Proposed fix**:
- Add `check_cv_compatibility()` before CV (already exists in `target_validation.py`, line 114)
- Detect degenerate folds early (check class distribution per fold)
- Fallback policy: fewer folds, different splitter, or skip CV + train full
- Log as INFO with reason code (not debug/error)

---

## Call Graph Analysis

### Budget Computation Flow

```
evaluate_target_predictability()
  ├─ create_resolved_config() [1]  # Pre-gatekeeper
  │   └─ compute_feature_lookback_max()  # If recompute_lookback=True
  │
  ├─ _enforce_final_safety_gate() [2]  # Gatekeeper
  │   └─ apply_lookback_cap()
  │       └─ compute_budget()  # Budget from safe features
  │
  └─ train_and_evaluate_models() [3]
      ├─ create_resolved_config() [3a]  # Post-prune
      │   └─ compute_feature_lookback_max()  # If recompute_lookback=True
      │
      └─ compute_budget() [3b]  # Post-prune policy check
```

**Redundancy**: Budget computed 3-4 times, config recomputed 2-3 times.

### Feature Filtering Flow

```
Feature Generation
  ↓
filter_features_for_target()  # Target/label pattern filtering
  ↓
SAFE_CANDIDATES (fingerprint)
  ↓
apply_lookback_cap(FS_PRE)  # Pre-selection cap
  ↓
Feature Selection (multi-model)
  ↓
apply_lookback_cap(FS_POST)  # Post-selection cap
  ↓
POST_GATEKEEPER (fingerprint)
  ↓
apply_lookback_cap(GATEKEEPER)  # Final gatekeeper
  ↓
Feature Pruning
  ↓
POST_PRUNE (fingerprint)
```

**Redundancy**: Lookback cap applied 3 times (FS_PRE, FS_POST, GATEKEEPER), but each stage may use different canonical maps.

---

## Proposed Solutions

### Solution 1: FeatureSet Artifact (Unified Budget/Config)

**Goal**: Single canonical artifact per stage, no recomputation.

**Structure**:
```python
@dataclass
class FeatureSetArtifact:
    features: List[str]
    fingerprint_set: str
    fingerprint_ordered: str
    canonical_lookback_map: Dict[str, float]
    budget: LeakageBudget
    resolved_config: ResolvedConfig
    removal_reasons: Dict[str, str]  # Feature → reason (quarantined, pruned, etc.)
    stage: str
```

**Flow**:
```
SAFE_CANDIDATES → FeatureSetArtifact(SAFE_CANDIDATES)
  ↓
Gatekeeper → FeatureSetArtifact(POST_GATEKEEPER)  # Reuses canonical_map from SAFE_CANDIDATES
  ↓
Pruning → FeatureSetArtifact(POST_PRUNE)  # Reuses canonical_map, updates budget
  ↓
Training → Uses FeatureSetArtifact(POST_PRUNE)  # No recomputation
```

**Benefits**:
- Budget computed once per stage
- Config computed once per stage
- Canonical map reused (no recomputation)
- Clear provenance (removal_reasons)

### Solution 2: Unknown Lookback Logging Fix

**Change**: Stage-aware logging for unknown lookback.

**Pre-enforcement** (SAFE_CANDIDATES, FS_PRE):
```python
if lookback == float("inf"):
    logger.info(f"   {stage}: {feat_name} → unknown lookback (will be quarantined at gatekeeper)")
```

**Post-enforcement** (GATEKEEPER, POST_PRUNE):
```python
if lookback == float("inf"):
    logger.warning(f"   {stage}: {feat_name} → unknown lookback (quarantined - violation)")
```

### Solution 3: Feature Generation Alignment

**Option A** (Recommended): Early filter in feature generation.

**Location**: `DATA_PROCESSING/features/simple_features.py`

**Change**:
```python
def _compute_cross_sectional_features(self, features: pl.LazyFrame, 
                                      lookback_budget_cap: Optional[float] = None) -> pl.LazyFrame:
    # Skip 20d/60d families if cap < window_size
    if lookback_budget_cap is not None:
        if lookback_budget_cap < 20 * 1440:  # 20 days in minutes
            # Skip 20d/60d features
            return features  # Only return base features
```

**Option B**: Conditional generation based on sample cap.

**Location**: Feature generation config

**Change**:
```yaml
feature_generation:
  skip_long_windows_if_cap: true  # Skip 20d/60d if lookback_budget_minutes < window_size
```

### Solution 4: Degenerate Folds Policy

**Goal**: First-class handling of degenerate folds.

**Implementation**:

1. **Pre-CV validation** (`model_evaluation.py`, before line 1946):
   ```python
   from TRAINING.utils.target_validation import check_cv_compatibility
   is_compatible, reason = check_cv_compatibility(y, task_type, cv_folds)
   if not is_compatible:
       logger.info(f"  ℹ️  CV compatibility check: {reason}")
       # Fallback: fewer folds, different splitter, or skip CV
       cv_folds = max(2, cv_folds - 1)  # Reduce folds
       # Or: use different splitter (e.g., StratifiedKFold for classification)
   ```

2. **Fallback policy** (config-driven):
   ```yaml
   training:
     cv_degenerate_fallback: "reduce_folds"  # "reduce_folds" | "skip_cv" | "different_splitter"
     cv_min_folds: 2  # Minimum folds before skipping CV
   ```

3. **Logging** (INFO, not debug):
   ```python
   logger.info(f"  ℹ️  Degenerate folds detected: {reason}. Using fallback: {fallback_policy}")
   ```

---

## Implementation Priority

### Phase 1: Do Now (Correctness/Debuggability)

1. **Degenerate folds policy** (Issue 4)
   - Prevents silent failures
   - Makes results trustworthy
   - **Files**: `TRAINING/ranking/predictability/model_evaluation.py`, `TRAINING/utils/target_validation.py`

2. **Unknown lookback logging fix** (Issue 1)
   - Reduces log noise
   - Makes expected behavior clear
   - **Files**: `TRAINING/utils/lookback_cap_enforcement.py`

3. **FeatureSet artifact** (Issue 2 - partial)
   - Persist canonical artifact per run (minimal version)
   - **Files**: New `TRAINING/utils/feature_set_artifact.py`, update call sites

### Phase 2: Save for Later (Once E2E Stable)

1. **Full FeatureSet artifact integration** (Issue 2 - complete)
   - Replace all recomputation with artifact passing
   - **Files**: Multiple (see call graph)

2. **Feature generation alignment** (Issue 3)
   - Skip long-window families when cap < window_size
   - **Files**: `DATA_PROCESSING/features/simple_features.py`, feature generation config

---

## Metrics

### Current State
- Budget computed: **3-4 times** per run
- Config recomputed: **2-3 times** per run
- Unknown lookback warnings: **~10-50 per run** (depending on feature count)
- Degenerate fold handling: **Exception-based** (surprise)

### Target State
- Budget computed: **1 time per stage** (3 stages = 3 times, but no redundant calls)
- Config recomputed: **1 time per stage** (same)
- Unknown lookback warnings: **0 pre-enforcement** (INFO only), **actual violations only** post-enforcement
- Degenerate fold handling: **First-class policy** (expected case, clear fallback)

---

## Risk Assessment

### Low Risk
- Unknown lookback logging fix (INFO vs WARNING)
- Degenerate folds policy (adds validation, doesn't change core logic)

### Medium Risk
- FeatureSet artifact (changes data flow, but preserves invariants)
- Feature generation alignment (may affect feature availability)

### High Risk
- None (all changes are additive or logging-only)

---

## Next Steps

1. **Review this analysis** with team/stakeholders
2. **Prioritize Phase 1** items (correctness/debuggability)
3. **Create implementation plan** for Phase 1 (detailed file changes)
4. **Test Phase 1** changes on sample runs
5. **Defer Phase 2** until E2E is stable

---

## References

- Lookback cap enforcement: `TRAINING/utils/lookback_cap_enforcement.py`
- Budget computation: `TRAINING/utils/leakage_budget.py`
- Resolved config: `TRAINING/utils/resolved_config.py`
- Model evaluation: `TRAINING/ranking/predictability/model_evaluation.py`
- Feature generation: `DATA_PROCESSING/features/simple_features.py`
- Target validation: `TRAINING/utils/target_validation.py`

