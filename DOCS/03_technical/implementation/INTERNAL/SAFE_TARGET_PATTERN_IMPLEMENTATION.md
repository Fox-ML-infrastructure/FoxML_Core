# Safe Target Pattern Implementation

## Overview

Implemented the canonical per-target training contract to prevent duplicate column issues and ensure clean feature/target separation.

## Key Changes

### 1. Enhanced `strip_targets()` Function

```python
def strip_targets(cols, all_targets=None):
    """
    Remove ALL target-like columns from feature list.

    Args:
        cols: List of column names
        all_targets: Set of all discovered target columns (if None, uses heuristics)

    Returns:
        List of feature columns only (no targets, no symbol/timestamp)
    """
```

Features:
- Uses explicit `all_targets` set for precise filtering when available
- Falls back to heuristics for backward compatibility
- Excludes `symbol` and `timestamp` from features
- Comprehensive target pattern matching

### 2. New `collapse_identical_duplicate_columns()` Function

```python
def collapse_identical_duplicate_columns(df):
    """
    Collapse identical duplicate columns, raise error if conflicting.

    Args:
        df: DataFrame with potentially duplicate columns

    Returns:
        DataFrame with unique columns, duplicates removed
    """
```

Features:
- Detects and handles duplicate column names
- Collapses identical duplicates safely
- Raises clear error for conflicting duplicates (root cause of tolist() crashes)
- Provides detailed error messages for debugging

### 3. Updated `prepare_training_data_cross_sectional()` Function

New Parameters:
- Added `all_targets: set = None` parameter for precise target filtering

Enhanced Logic:
- Uses `strip_targets(common_features, all_targets)` for precise filtering
- Applies `collapse_identical_duplicate_columns()` to handle duplicates safely
- Validates training contract: `X = features only`, `y = exactly one target`
- Prevents target leakage into feature matrix

### 4. Updated `targets_for_interval()` Function

New Return Format:
```python
def targets_for_interval(...) -> tuple[List[str], set]:
    """
    Returns:
        (target_list, all_targets_set)
    """
```

Features:
- Returns both target list and complete target set
- Enables precise filtering in downstream functions
- Maintains backward compatibility

### 5. Updated Function Calls

Enhanced Parameter Passing:
- `train_models_for_interval()` now accepts `all_targets` parameter
- `prepare_training_data_cross_sectional()` receives `all_targets` for precise filtering
- Main training loop discovers and passes `all_targets` to all functions

## Training Contract Enforcement

### Canonical Per-Target Contract

For each target T you train:

1. X (features): Only engineered features (no labels), plus metadata (`symbol`/`timestamp` not in `X`)
2. y (label): Exactly the one target column T
3. Exclude: All other targets from `X`

### Safe Pattern Implementation

```python
# 1) Strip ALL targets from features
features = [c for c in features_all if c not in ALL_TARGETS and c not in ("symbol","timestamp")]

# 2) For each target T:
X = df[features]  # Only features, no targets
y = df[T]        # Only this target

# 3) Validate contract
assert T not in X.columns, f"Target {T} leaked into features!"
assert all(t not in X.columns for t in ALL_TARGETS), "Other targets leaked!"
```

## Benefits

### 1. Prevents Target Leakage
- Explicit target set prevents accidental inclusion
- Heuristic fallback maintains backward compatibility
- Clear error messages when leakage detected

### 2. Handles Duplicate Columns
- Identical duplicates collapsed safely
- Conflicting duplicates raise clear errors
- Prevents tolist() crashes from duplicate column names

### 3. Clean Separation
- Features and targets clearly separated
- Metadata columns excluded from features
- Training contract enforced at runtime

## Usage

### Automatic (Recommended)
The training pipeline automatically applies safe patterns:

```python
# Main training loop handles everything
train_models_for_interval(
    interval="5m",
    data_dir="data/data_labeled/interval=5m",
    # all_targets discovered automatically
)
```

### Manual (Advanced)
```python
# Discover all targets
target_list, all_targets = targets_for_interval("5m", data_dir)

# Prepare data with explicit target set
X, y = prepare_training_data_cross_sectional(
    df,
    target="y_will_peak_60m_0.8",
    all_targets=all_targets  # Explicit filtering
)
```

## Validation

### Runtime Checks
The system validates the training contract:

```python
# Check 1: Target not in features
assert target not in X.columns, f"Target {target} leaked into features!"

# Check 2: No other targets in features
leaked = [t for t in all_targets if t in X.columns]
assert not leaked, f"Other targets leaked: {leaked}"

# Check 3: No metadata in features
assert "symbol" not in X.columns, "Metadata 'symbol' in features!"
assert "timestamp" not in X.columns, "Metadata 'timestamp' in features!"
```

## Error Messages

### Duplicate Column Error
```
ValueError: Conflicting duplicate columns detected:
  - Column 'feature_A' appears 2 times with different values
  - Column 'feature_B' appears 3 times with different values
  
Action: Check data processing pipeline for duplicate column creation.
```

### Target Leakage Error
```
ValueError: Target 'y_will_peak_60m_0.8' found in feature matrix!
This indicates target leakage. Check feature selection logic.
```

## Files Modified

- `TRAINING/utils/data_preprocessor.py` - Enhanced `strip_targets()` and new `collapse_identical_duplicate_columns()`
- `TRAINING/utils/target_resolver.py` - Updated `targets_for_interval()` return format
- `TRAINING/training_strategies/training.py` - Main loop passes `all_targets` to all functions (split from original `train_with_strategies.py`)
- `TRAINING/strategies/single_task.py` - Uses `all_targets` for precise filtering

## Testing

### Unit Tests
```bash
cd TRAINING
python -m pytest tests/test_safe_target_pattern.py
```

### Integration Tests
```bash
# Run full training with validation
python -m TRAINING.training_strategies.main \
  --data-dir data/data_labeled/interval=5m \
  --validate-target-separation
```

## Status

Implementation complete. All training functions now enforce the safe target pattern contract.
