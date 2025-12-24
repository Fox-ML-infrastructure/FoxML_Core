# Feature Registry - Phase 1 Complete ✅

**Date**: 2025-12-07  
**Status**: Phase 1 Infrastructure Complete

---

## ✅ What Was Built

### 1. FeatureRegistry Class (`TRAINING/common/feature_registry.py`)

- **Core functionality**:
  - Loads feature metadata from YAML config
  - Validates features against hard rules (lag_bars >= 0, etc.)
  - Auto-infers metadata for unknown features (backward compatible)
  - Checks if features are allowed for specific target horizons
  - Filters feature lists based on target horizon

- **Key methods**:
  - `is_allowed(feature_name, target_horizon)` - Check if feature can be used
  - `get_allowed_features(all_features, target_horizon)` - Filter feature list
  - `auto_infer_metadata(feature_name)` - Infer metadata for unknown features
  - `get_feature_metadata(feature_name)` - Get metadata (from registry or inferred)

### 2. Feature Registry Config (`CONFIG/feature_registry.yaml`)

- **26 explicit features** defined with metadata:
  - Lagged returns (ret_1, ret_5, ret_10, etc.)
  - Technical indicators (rsi_10, sma_20, ema_12, etc.)
  - Volume features
  - Volatility features
  - VWAP features
  - Momentum features
  - Rejected features (documented as leaky)

- **11 feature families** with pattern matching:
  - Safe families: lagged_returns, technical_indicators, volume_features, volatility_features
  - Rejected families: rejected_future_returns, rejected_time_to_hit, rejected_mfe_mdd, rejected_barriers, rejected_targets, rejected_predictions, rejected_metadata

- **Validation rules**:
  - Hard rules: lag_bars >= 0, allowed_horizons must be non-empty
  - Warnings: lag_bars == 0, source == 'unknown'

### 3. Integration with Existing Leakage Filtering

- **Updated `SCRIPTS/utils/leakage_filtering.py`**:
  - Added `use_registry` parameter (default: True)
  - Added `data_interval_minutes` parameter (default: 5) for horizon conversion
  - Converts target horizon from minutes to bars
  - Applies registry filtering before pattern-based filtering
  - Backward compatible (falls back to pattern-based if registry unavailable)

### 4. Auto-Inference for Unknown Features

- **Pattern matching** for common feature types:
  - `ret_N` → lagged return (lag = N)
  - `rsi_N`, `sma_N`, `ema_N` → technical indicators (lag = N)
  - `fwd_ret_*`, `ret_future_*` → rejected (forward returns)
  - `tth_*` → rejected (time-to-hit)
  - `mfe_*`, `mdd_*` → rejected (MFE/MDD)
  - `barrier_*` → rejected (barrier features)
  - `y_*`, `target_*` → rejected (target columns)
  - `p_*` → rejected (predictions)
  - Unknown → rejected by default (safe)

---

## 🧪 Testing Results

```python
✅ FeatureRegistry loads successfully
   Features: 26
   Families: 11

📊 Auto-inference test:
   ret_5: lag=5, horizons=[5, 15, 25, 60], rejected=False
   rsi_10: lag=10, horizons=[1, 3, 5, 12, 24, 60], rejected=False
   tth_5m: lag=0, horizons=[], rejected=True
   fwd_ret_5m: lag=-1, horizons=[], rejected=True
   unknown_feature: lag=0, horizons=[], rejected=True

🔍 is_allowed test (horizon=12 bars):
   ret_5: ✅ allowed
   rsi_10: ✅ allowed
   tth_5m: ❌ rejected
   fwd_ret_5m: ❌ rejected
   unknown_feature: ❌ rejected

📋 get_allowed_features test (horizon=12 bars):
   Allowed: ['ret_1', 'ret_5', 'rsi_10', 'sma_200']
```

---

## 🔄 How It Works

### Flow Diagram

```
1. filter_features_for_target() called
   ↓
2. Extract target horizon (minutes) from target column name
   ↓
3. Convert horizon to bars (minutes / data_interval_minutes)
   ↓
4. Load FeatureRegistry (singleton, cached)
   ↓
5. For each feature:
   a. Check explicit registry entry
   b. Check feature family patterns
   c. Auto-infer if unknown
   ↓
6. Filter: keep only features where is_allowed(feature, horizon_bars) == True
   ↓
7. Apply pattern-based filtering (additional safety layer)
   ↓
8. Return safe features
```

### Example

```python
# Target: fwd_ret_60m (60-minute forward return)
# Data interval: 5 minutes
# Horizon: 60m / 5m = 12 bars

features = ['ret_1', 'ret_5', 'rsi_10', 'tth_5m', 'fwd_ret_5m']

# Registry filtering:
# - ret_1: lag=1, horizons=[1,3,5,12,24,60] → ✅ allowed (12 in list)
# - ret_5: lag=5, horizons=[5,12,24,60] → ✅ allowed (12 in list)
# - rsi_10: lag=10, horizons=[1,3,5,12,24,60] → ✅ allowed (12 in list)
# - tth_5m: auto-inferred → rejected=True → ❌ rejected
# - fwd_ret_5m: auto-inferred → rejected=True → ❌ rejected

allowed = ['ret_1', 'ret_5', 'rsi_10']
```

---

## 📋 Next Steps (Phase 2)

1. **Validation & Enforcement**:
   - [ ] Add registry validation to feature selection pipeline
   - [ ] Add registry validation to training pipeline
   - [ ] Add registry validation to target ranking pipeline
   - [ ] Test with real data

2. **Automated Sentinels** (Phase 3):
   - [ ] Implement shifted-target test
   - [ ] Implement symbol-holdout test
   - [ ] Implement randomized-time test
   - [ ] Add to intelligent trainer as optional diagnostic mode

3. **Feature Importance Diff** (Phase 4):
   - [ ] Implement importance diff detector
   - [ ] Integration with feature selection
   - [ ] Reporting and flagging

---

## 🔗 Related Files

- `TRAINING/common/feature_registry.py` - Core registry class
- `CONFIG/feature_registry.yaml` - Feature metadata config
- `SCRIPTS/utils/leakage_filtering.py` - Integration point
- `docs/internal/planning/FEATURE_REGISTRY_DESIGN.md` - Full design doc

---

## ✅ Benefits Achieved

1. **Structural Safety**: Features with lag_bars < 0 are rejected at validation time
2. **Automated Detection**: Auto-inference catches common leaky patterns
3. **Backward Compatible**: Works with existing features via auto-inference
4. **Documentation**: Feature metadata serves as documentation
5. **Integration**: Seamlessly integrated with existing leakage filtering

---

## 🚨 Known Limitations

1. **Horizon Conversion**: Currently assumes 5-minute bars (configurable via parameter)
2. **Auto-Inference**: May be too conservative (rejects unknown features by default)
3. **Pattern Matching**: Limited to common patterns (can be extended)
4. **No Sentinels Yet**: Automated leakage sentinels not yet implemented (Phase 3)

---

## 📝 Usage Example

```python
from TRAINING.common.feature_registry import get_registry
from scripts.utils.leakage_filtering import filter_features_for_target

# Get registry
registry = get_registry()

# Check if feature is allowed
allowed = registry.is_allowed('ret_5', target_horizon_bars=12)  # True

# Filter feature list
all_features = ['ret_1', 'ret_5', 'rsi_10', 'tth_5m']
allowed_features = registry.get_allowed_features(all_features, target_horizon_bars=12)
# Returns: ['ret_1', 'ret_5', 'rsi_10']

# Use with existing leakage filtering
safe_features = filter_features_for_target(
    all_columns=['ret_1', 'ret_5', 'rsi_10', 'tth_5m'],
    target_column='fwd_ret_60m',
    verbose=True,
    use_registry=True,  # Enable registry filtering
    data_interval_minutes=5  # 5-minute bars
)
```

---

**Phase 1 Complete ✅**  
Ready for Phase 2: Validation & Enforcement

