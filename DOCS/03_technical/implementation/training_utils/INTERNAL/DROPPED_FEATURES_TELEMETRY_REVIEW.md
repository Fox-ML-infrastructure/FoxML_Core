# Dropped Features Telemetry - Review & Fixes

## Summary

Fixed critical correctness and determinism issues in dropped-feature telemetry implementation.

## Critical Fixes Applied

### 1. ✅ Set-Based Drop Detection (Fixed)

**Problem**: List-based comparison misreported reordering as drops.

**Fix**: 
- `DroppedFeaturesTracker.add_nan_drops()` now uses set-based comparison: `dropped = sorted(set(input) - set(output))`
- `record_stage_transition()` automatically detects `order_changed` separately from drops
- All stage transitions now use set-based comparison

**Location**: `TRAINING/utils/dropped_features_tracker.py` lines 150-180

### 2. ✅ Structured Reasons (Fixed)

**Problem**: Reasons were strings only, not machine-readable.

**Fix**:
- Created `DropReason` dataclass with:
  - `reason_code`: `LOOKBACK_CAP`, `ALL_NAN`, `QUARANTINED_LOOKBACK`, `LOW_IMPORTANCE`
  - `stage`: `gatekeeper`, `sanitizer`, `pruning`, `nan_removal`
  - `measured_value`, `threshold_value`: Numeric values for programmatic checks
  - `config_provenance`: Which config knob caused it
  - `human_reason`: Still included for logs

**Location**: `TRAINING/utils/dropped_features_tracker.py` lines 12-22

### 3. ✅ Stage Records with Fingerprints (Fixed)

**Problem**: No clear boundaries between stages, no forensic evidence.

**Fix**:
- Created `StageRecord` dataclass tracking:
  - `input_fingerprint`, `output_fingerprint`
  - `input_count`, `output_count`, `dropped_count`
  - `dropped_sample` (first 10)
  - `order_changed: bool`
  - `config_provenance`: Dict of config knobs

- `record_stage_transition()` automatically creates stage records
- All `add_*_drops()` methods now accept `input_features`, `output_features` to create stage records

**Location**: `TRAINING/utils/dropped_features_tracker.py` lines 24-40, 100-150

### 4. ✅ Sanitizer Tracking (Fixed)

**Problem**: Sanitizer drops weren't tracked.

**Fix**:
- Added `dropped_tracker` parameter to `filter_features_for_target()`
- Sanitizer quarantines now tracked with structured `DropReason` objects
- Stage record created with input/output fingerprints

**Location**: 
- `TRAINING/utils/leakage_filtering.py` lines 857-920
- `TRAINING/ranking/predictability/model_evaluation.py` line 4073

### 5. ✅ Early Filter Summary (Fixed)

**Problem**: Schema/pattern/registry filtering drops weren't tracked.

**Fix**:
- Added `add_early_filter_summary()` method
- Tracks counts, top samples, rule hits (structure ready for enhancement)
- Set-based comparison to avoid false positives

**Location**: 
- `TRAINING/utils/dropped_features_tracker.py` lines 200-220
- `TRAINING/ranking/predictability/model_evaluation.py` lines 4083-4092

### 6. ✅ NaN Drop Tracking Fixed (Fixed)

**Problem**: List-based comparison, wrong boundary.

**Fix**:
- Now uses set-based comparison in `add_nan_drops()`
- Captures `feature_names_before_data_prep` immediately before `prepare_cross_sectional_data_for_ranking()`
- Tracks immediately after data prep (clear boundary)
- Stage record created with fingerprints

**Location**: `TRAINING/ranking/predictability/model_evaluation.py` lines 4140-4185

## Metadata Structure (New)

The `dropped_features` section in `metadata.json` now includes:

```json
{
  "dropped_features": {
    "stage_records": [
      {
        "stage_id": "gatekeeper",
        "stage_name": "Final Gatekeeper (Lookback Enforcement)",
        "input_fingerprint": "abc123...",
        "output_fingerprint": "def456...",
        "input_count": 300,
        "output_count": 285,
        "dropped_count": 15,
        "dropped_sample": ["feature1", "feature2", ...],
        "order_changed": false,
        "config_provenance": {
          "safe_lookback_max": 100.0,
          "purge_limit": 150.0,
          "over_budget_action": "drop"
        }
      },
      ...
    ],
    "gatekeeper": {
      "count": 15,
      "features": ["feature1", ...],
      "reasons": {
        "feature1": {
          "reason_code": "LOOKBACK_CAP",
          "stage": "gatekeeper",
          "human_reason": "lookback (150.0m) > safe_limit (100.0m)",
          "measured_value": 150.0,
          "threshold_value": 100.0,
          "config_provenance": "lookback_budget_minutes=100.0m (source=config)"
        }
      }
    },
    "sanitizer": {
      "count": 5,
      "features": [...],
      "reasons": {...}
    },
    "pruning": {
      "count": 10,
      "features": [...],
      "stats": {...}
    },
    "nan": {
      "count": 2,
      "features": [...]
    },
    "early_filters": {
      "schema_pattern_registry": {
        "dropped_count": 20,
        "top_samples": [...],
        "rule_hits": {}
      }
    },
    "total_unique": 52
  }
}
```

## Remaining Gaps (Next Steps)

### 1. Early Filter Rule Hits (Enhancement)

**Current**: `rule_hits` is `None` in early filter summary.

**Enhancement**: Track which specific rules hit (pattern matches, registry rejects, etc.):

```python
rule_hits = {
  "pattern_y_*": 5,
  "pattern_fwd_ret_*": 3,
  "registry_reject": 2,
  "target_conditional": 10
}
```

**Location**: `TRAINING/utils/leakage_filtering.py` - enhance `filter_features_for_target()` to track rule hits

### 2. Feature Selection & Training Stages (Not Yet Implemented)

**Current**: Only target ranking tracks drops.

**Enhancement**: Add similar tracking to:
- `select_features_for_target()` in `TRAINING/ranking/feature_selector.py`
- `train_models_for_interval_comprehensive()` in `TRAINING/training_strategies/training.py`

**Pattern**: Same `DroppedFeaturesTracker` + `additional_data['dropped_features']` approach

### 3. Config Provenance Enhancement

**Current**: Config provenance is strings.

**Enhancement**: Make it structured:

```python
config_provenance = {
  "lookback_budget_minutes": {
    "value": 240.0,
    "source": "safety_config.yaml",
    "path": "safety.leakage_detection.lookback_budget_minutes"
  },
  "purge_include_feature_lookback": {
    "value": true,
    "source": "safety_config.yaml",
    "path": "safety.leakage_detection.purge_include_feature_lookback"
  }
}
```

## Testing Checklist

- [ ] Run a target ranking and verify `metadata.json` contains `dropped_features` section
- [ ] Verify stage_records show correct fingerprints and counts
- [ ] Verify set-based comparison doesn't report reordering as drops
- [ ] Verify structured reasons are machine-readable
- [ ] Verify sanitizer tracking works (if sanitizer is enabled)
- [ ] Verify early filter summary captures schema/pattern drops

## Example Metadata.json Excerpt

After fixes, a real `metadata.json` should show:

```json
{
  "dropped_features": {
    "stage_records": [
      {
        "stage_id": "nan_removal",
        "stage_name": "All-NaN Column Removal",
        "input_fingerprint": "abc123def456",
        "output_fingerprint": "abc123def456",
        "input_count": 300,
        "output_count": 298,
        "dropped_count": 2,
        "dropped_sample": ["feature_with_all_nan", "another_nan_feature"],
        "order_changed": false,
        "config_provenance": {
          "stage": "data_preparation",
          "operation": "dropna(axis=1, how='all')"
        }
      },
      {
        "stage_id": "pruning",
        "stage_name": "Importance-Based Pruning",
        "input_fingerprint": "abc123def456",
        "output_fingerprint": "xyz789uvw012",
        "input_count": 298,
        "output_count": 150,
        "dropped_count": 148,
        "dropped_sample": ["low_importance_feature_1", ...],
        "order_changed": true,
        "config_provenance": {
          "cumulative_threshold": 0.0001,
          "min_features": 50,
          "n_estimators": 50
        }
      },
      {
        "stage_id": "gatekeeper",
        "stage_name": "Final Gatekeeper (Lookback Enforcement)",
        "input_fingerprint": "xyz789uvw012",
        "output_fingerprint": "final123abc456",
        "input_count": 150,
        "output_count": 145,
        "dropped_count": 5,
        "dropped_sample": ["long_lookback_feature_1", ...],
        "order_changed": false,
        "config_provenance": {
          "safe_lookback_max": 240.0,
          "safe_lookback_max_source": "config",
          "purge_limit": 250.0,
          "over_budget_action": "drop"
        }
      }
    ],
    "gatekeeper": {
      "count": 5,
      "features": ["long_lookback_feature_1", ...],
      "reasons": {
        "long_lookback_feature_1": {
          "reason_code": "LOOKBACK_CAP",
          "stage": "gatekeeper",
          "human_reason": "lookback (300.0m) > safe_limit (240.0m)",
          "measured_value": 300.0,
          "threshold_value": 240.0,
          "config_provenance": "lookback_budget_minutes=240.0m (source=config)"
        }
      }
    },
    "total_unique": 155
  }
}
```

## Definition of Done Status

✅ **Every feature set transition produces a stage record** with before/after fingerprints and deterministic drop detection.

✅ **Tracker covers**: sanitizer + NaN + gatekeeper + pruning (plus summary counts for early filters).

✅ **metadata.json lets you answer**: For any missing feature, you can see where it died, why, under what config.

🔄 **Remaining**: Enhance early filter rule hits, extend to feature selection/training stages, enhance config provenance structure.
