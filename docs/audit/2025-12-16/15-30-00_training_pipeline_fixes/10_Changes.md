# Changes

## Actions

### 1. Family Name Canonicalization (`TRAINING/training_strategies/utils.py`)
- **Enhanced `normalize_family_name()`**: Added more special cases, better normalization logic
- **Applied at all boundaries**:
  - `_run_family_isolated()` - before MODMAP lookup
  - `_run_family_inproc()` - before MODMAP lookup  
  - `train_model_comprehensive()` - before runtime policy lookup
  - `get_policy()` in `runtime_policy.py` - before POLICY lookup

### 2. Preflight Validation (`TRAINING/training_strategies/training.py`)
- **Added startup validation**: Checks families exist in registry before training
- **Separates selectors from trainers**: `MutualInformation`, `UnivariateSelection` marked as non-trainers
- **Fails fast**: Clear error messages with available families list

### 3. Banner Suppression (`TRAINING/common/license_banner.py`, `TRAINING/common/threads.py`, `TRAINING/orchestration/intelligent_trainer.py`)
- **Enhanced banner checks**: Suppresses in child processes, non-TTY contexts
- **Child environment**: Sets `FOXML_SUPPRESS_BANNER=1`, `TRAINER_ISOLATION_CHILD=1` in `child_env_for_family()`
- **Module-level guard**: `intelligent_trainer.py` checks env before printing

### 4. Reproducibility Tracking Fixes (`TRAINING/utils/reproducibility_tracker.py`)
- **Defensive Enum/string handling**: Added `nameish()`-like logic in `_compute_drift()` and `_save_to_cohort()`
- **Handles both string and Enum**: Checks `isinstance()` and `hasattr('value')` before calling `.upper()`

### 5. Model Saving Fix (`TRAINING/training_strategies/training.py`)
- **Moved `_pkg_ver` definition**: Outside conditional blocks to avoid "referenced before assignment"
- **Removed redundant import**: `joblib` already imported at top, removed local import

### 6. Run Summary Improvements (`TRAINING/training_strategies/training.py`)
- **Added family result tracking**: `trained_ok`, `failed`, `skipped` per target
- **Enhanced final summary**: Shows trained_ok count, failed targets with reasons
- **Better error handling**: Training failures continue with next family instead of re-raising

### 7. Feature Count Validation (`TRAINING/training_strategies/training.py`)
- **Added feature pipeline logging**: Logs requested → allowed → used counts
- **Collapse detection**: Warns if <50% of requested features are used

## Commands run

None (code changes only)

