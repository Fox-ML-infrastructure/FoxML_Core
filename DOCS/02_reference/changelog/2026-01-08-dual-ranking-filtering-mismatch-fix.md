# Dual Ranking and Filtering Mismatch Fix

**Date**: 2026-01-08  
**Status**: Implemented  
**Impact**: High - Prevents false positives in target ranking

## Overview

This change fixes a critical filtering mismatch between TARGET_RANKING and FEATURE_SELECTION stages that could cause false positives: targets ranking high using features that won't be available in training.

## Problem Statement

### The Mismatch

**Before this fix:**
- **TARGET_RANKING**: Used `for_ranking=True` which allowed "unknown but safe" features (features not in registry but passing basic leak checks)
- **FEATURE_SELECTION**: Used `for_ranking=False` (strict registry-only)
- **Result**: Targets could rank high using features that training would never see, leading to wasted compute on unviable targets

### Example Failure Mode

1. Target `fwd_ret_60m` ranks #3 using 200 features (including 50 "unknown" features)
2. Feature selection runs with strict registry-only → only 150 features available
3. Target's actual predictability drops significantly
4. Training proceeds anyway, wasting compute on a target that won't perform

## Solution: Dual Ranking

### Architecture

```
TARGET_RANKING Evaluation:
├── Screen Evaluation (for_ranking=True)
│   ├── Features: safe_family + registry_features
│   ├── Purpose: Discovery (find promising targets)
│   └── Output: score_screen, n_feats_screen
│
└── Strict Evaluation (for_ranking=False)
    ├── Features: registry_features only (exact training universe)
    ├── Purpose: Validation (ensure viability)
    └── Output: score_strict, n_feats_strict, mismatch_telemetry
```

### Key Changes

1. **Removed "unknown_safe" block** from `filter_features_for_target()` (ranking mode)
   - Ranking now uses "safe_family + registry" only (no unknown features)
   - Prevents false positives at the source

2. **Added dual ranking fields** to `TargetPredictabilityScore`:
   - `score_screen`: Composite score using screen features (safe+registry)
   - `score_strict`: Composite score using strict features (registry-only)
   - `strict_viability_flag`: True if target clears threshold in strict mode
   - `rank_delta`: Difference between screen and strict ranks
   - `mismatch_telemetry`: Feature counts, overlap, unknown feature rate

3. **Updated promotion logic** in `intelligent_trainer.py`:
   - Only promotes targets where `strict_viability_flag=True`
   - Logs warnings for targets with high `rank_delta` or unknown features

4. **Added mismatch telemetry**:
   - `n_feats_screen`, `n_feats_strict`: Feature counts
   - `topk_overlap`: Jaccard similarity of top-K features
   - `unknown_feature_count`: Count of features in screen but not registry
   - `registry_coverage_rate`: n_feats_strict / n_feats_screen

## Integration Points

### Pipeline Flow

```
1. TARGET_RANKING Stage
   ├── evaluate_target_predictability()
   │   ├── Screen evaluation (for_ranking=True)
   │   │   └── Computes: score_screen, n_feats_screen
   │   └── Strict evaluation (for_ranking=False)
   │       └── Computes: score_strict, n_feats_strict, mismatch_telemetry
   │
   └── rank_targets()
       ├── Sorts by composite_score (screen score)
       ├── Computes rank_delta (screen_rank - strict_rank)
       └── Sets strict_viability_flag (True if strict_rank < top_n)

2. Target Promotion (intelligent_trainer.py)
   ├── Filters rankings by strict_viability_flag
   ├── Logs warnings for high rank_delta or unknown features
   └── Returns top N viable targets

3. FEATURE_SELECTION Stage
   └── Uses strict registry-only (for_ranking=False)
       └── Ensures consistency with TARGET_RANKING strict evaluation
```

### Data Flow

```
TargetPredictabilityScore
├── to_dict() → JSON snapshots, cache files
├── build_clean_metrics_dict() → metrics.json (structured)
└── save_rankings() → CSV, YAML outputs
```

## Output Artifacts Updated

### Metrics Output

**`build_clean_metrics_dict()`** (metrics.json):
```json
{
  "score": {
    "composite": 0.75,
    "screen": 0.75,
    "strict": 0.72,
    "strict_viability": true,
    "rank_delta": 2
  },
  "mismatch_telemetry": {
    "n_feats_screen": 200,
    "n_feats_strict": 150,
    "topk_overlap": 0.85,
    "unknown_feature_count": 50,
    "registry_coverage_rate": 0.75
  }
}
```

### CSV Output

**`target_predictability_rankings.csv`**:
- Added columns: `score_screen`, `score_strict`, `strict_viability_flag`, `rank_delta`, `unknown_feature_count`, `registry_coverage_rate`

### YAML Output

**`target_prioritization.yaml`**:
- Added fields: `score_screen`, `score_strict`, `strict_viability_flag`, `rank_delta`, `mismatch_telemetry`

### Snapshots

**`snapshot.json`** (via `to_dict()`):
- All dual ranking fields automatically included
- `metrics_sha256` includes new fields (via `build_clean_metrics_dict()`)

## Determinism Impact

### Hash Changes (Expected)

- **`metrics_sha256`**: Will change (new fields in metrics dict)
- **Snapshot hashes**: Will change (new fields in result objects)
- **Backward compatibility**: Old snapshots load with `None` for new fields

### Determinism Guarantees

- **Future runs**: Deterministic (same inputs → same outputs)
- **Cross-run comparison**: New fields enable better drift detection
- **Registry incompleteness**: Now measurable via `unknown_feature_count`

## Benefits

1. **Prevents False Positives**: Targets that won't work in training are filtered early
2. **Improves Trust**: Mismatch telemetry makes drift visible
3. **Reduces Waste**: Less compute spent on unviable targets
4. **Enables Debugging**: `rank_delta` and `unknown_feature_count` identify issues
5. **Registry Health**: `registry_coverage_rate` measures registry completeness

## Migration Notes

### Breaking Changes

- **None**: All new fields are optional (`None` if not computed)
- Old snapshots load correctly with default values

### Behavior Changes

- **Target promotion**: Now filters by `strict_viability_flag`
- **Feature filtering**: Ranking mode no longer allows unknown features
- **Metrics hashes**: Will differ from previous runs (expected)

### Configuration

- No config changes required
- Existing configs continue to work
- New telemetry fields appear automatically

## Future Enhancements

1. **Full Strict Score Computation**: Currently telemetry-only; full re-evaluation with strict features would provide accurate `score_strict`
2. **Registry Population Workflow**: Automated workflow to populate registry from unknown features
3. **Whitelist Mechanism**: Small whitelist for new feature families before registry update
4. **Alerting**: Automatic alerts when `unknown_feature_count > threshold`

## Testing Recommendations

1. **Verify filtering**: Check that ranking mode no longer includes unknown features
2. **Check telemetry**: Verify `mismatch_telemetry` is populated correctly
3. **Validate promotion**: Ensure only `strict_viability_flag=True` targets are promoted
4. **Monitor warnings**: Check logs for high `rank_delta` or unknown feature warnings
5. **Compare hashes**: Verify `metrics_sha256` includes new fields

## Related Files

- `TRAINING/ranking/utils/leakage_filtering.py`: Removed unknown_safe block
- `TRAINING/ranking/predictability/scoring.py`: Added dual ranking fields
- `TRAINING/ranking/predictability/model_evaluation.py`: Added dual evaluation logic
- `TRAINING/ranking/target_ranker.py`: Added rank_delta computation
- `TRAINING/orchestration/intelligent_trainer.py`: Updated promotion logic
- `TRAINING/ranking/predictability/metrics_schema.py`: Added fields to metrics dict
- `TRAINING/ranking/predictability/reporting.py`: Added fields to CSV/YAML outputs
