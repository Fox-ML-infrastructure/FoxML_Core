# Stage Parity & Sample Limits

**Date**: 2026-01-07  
**Category**: Reproducibility, Determinism, Data Consistency  
**Impact**: High (fixes data sampling bug, adds TRAINING stage tracking)

## Summary

Three major improvements:
1. Fixed Feature Selection loading entire data history instead of respecting sample limits
2. Added full parity tracking for TRAINING stage (Stage 3)
3. Completed FS snapshot parity with TARGET_RANKING snapshots

## Problem

1. **Sample Limit Bug**: `compute_cross_sectional_importance()` was loading ALL data (~188k samples per symbol) instead of respecting `max_samples_per_symbol` config (e.g., 2k)
2. **No TRAINING Tracking**: Stage 3 had no snapshot mechanism - couldn't verify model determinism
3. **FS Snapshot Gaps**: Missing fields like `split_signature`, `metrics_sha256`, `n_effective`

## Changes

### Sample Limit Consistency

**Files Modified:**
- `TRAINING/ranking/cross_sectional_feature_ranker.py`
- `TRAINING/ranking/feature_selector.py`

**Fix:**
```python
# Before (loading ALL data):
mtf_data = load_mtf_data_for_ranking(data_dir, symbols)

# After (respecting limit):
mtf_data = load_mtf_data_for_ranking(data_dir, symbols, max_rows_per_symbol=max_rows_per_symbol)
```

**Impact:**
- All 3 stages (TR/FS/TRAINING) now use consistent `.tail(N)` sampling
- Expected: `n: 2000` per symbol instead of `n: 188779`

### TRAINING Stage Full Parity Tracking

**New Files:**
- `TRAINING/training_strategies/reproducibility/__init__.py`
- `TRAINING/training_strategies/reproducibility/schema.py`
- `TRAINING/training_strategies/reproducibility/io.py`

**New Schema (`TrainingSnapshot`):**
```python
@dataclass
class TrainingSnapshot:
    run_id: str
    timestamp: str
    stage: str = "TRAINING"
    view: str  # CROSS_SECTIONAL or SYMBOL_SPECIFIC
    target: str
    symbol: Optional[str]
    model_family: str
    
    # Fingerprints
    model_artifact_sha256: Optional[str]  # Hash of saved model file
    predictions_sha256: Optional[str]
    feature_fingerprint_input: Optional[str]
    feature_fingerprint_output: Optional[str]
    hyperparameters_signature: Optional[str]
    
    # Comparison group
    comparison_group: Dict[str, Any]  # Full parity with TR/FS
```

**Integration:**
- `TRAINING/training_strategies/execution/training.py` calls `create_and_save_training_snapshot()` after model save
- Global index: `globals/training_snapshot_index.json`

### FS Snapshot Full Parity

**Files Modified:**
- `TRAINING/stability/feature_importance/schema.py`
- `TRAINING/stability/feature_importance/io.py`
- `TRAINING/stability/feature_importance/hooks.py`

**New Fields in `FeatureSelectionSnapshot`:**
- `snapshot_seq`: Sequence number
- `metrics_sha256`: Hash of outputs.metrics
- `artifacts_manifest_sha256`: Hash of output artifacts
- `fingerprint_sources`: Documentation of fingerprint meanings
- `comparison_group.n_effective`: Row count
- `comparison_group.hyperparameters_signature`: Model config hash
- `comparison_group.feature_registry_hash`: Feature registry version
- `comparison_group.comparable_key`: Full reproducibility key

**Seed Fix:**
```python
# Before (derived seed - broke consistency):
train_seed = hash(base_seed + universe_sig)  # e.g., 198258262

# After (direct seed - consistent across stages):
train_seed = base_seed  # 42
```

## Verification

After next E2E run, check:

1. **Sample limits respected:**
   ```json
   // metadata.json
   "per_symbol_stats": {
     "AAPL": { "n": 2000 },  // was 188779
     ...
   }
   ```

2. **TRAINING snapshots created:**
   ```
   globals/training_snapshot_index.json
   ```

3. **FS snapshots have full parity:**
   ```json
   // fs_snapshot.json
   "comparison_group": {
     "train_seed": 42,  // was 198258262
     "n_effective": 20000,
     "comparable_key": "..."
   }
   ```

## Determinism Impact

**None.** Changes are:
- Bug fix (sample limits - was loading wrong data)
- Observational (new tracking)
- Metadata enrichment

Model computation unchanged when inputs are correct.

## Files Modified

| File | Change |
|------|--------|
| `TRAINING/ranking/cross_sectional_feature_ranker.py` | Added `max_rows_per_symbol` parameter |
| `TRAINING/ranking/feature_selector.py` | Pass sample limit to CS ranker |
| `TRAINING/common/utils/fingerprinting.py` | Use base_seed directly |
| `TRAINING/stability/feature_importance/schema.py` | FS snapshot full parity fields |
| `TRAINING/stability/feature_importance/io.py` | Pass parity fields through |
| `TRAINING/stability/feature_importance/hooks.py` | Accept parity fields |
| `TRAINING/training_strategies/reproducibility/*` | New TRAINING snapshot module |
| `TRAINING/training_strategies/execution/training.py` | Create training snapshots |
