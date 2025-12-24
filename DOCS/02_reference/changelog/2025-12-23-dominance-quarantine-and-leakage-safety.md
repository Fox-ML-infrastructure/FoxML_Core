# Dominance Quarantine and Leakage Safety Enhancements

**Date**: 2025-12-23  
**Type**: Feature Addition + Safety Enhancement

## Summary

Implemented a comprehensive dominance quarantine system for feature-level leakage detection and recovery, plus hard-exclusion of forward-looking features and small-panel leniency to prevent premature blocking.

## Changes

### 1. Dominance Quarantine System

**New Feature**: Auto-suspect → confirm → quarantine workflow for dominant-importance features.

**Implementation**:
- **Config**: Added `dominance_quarantine` section to `CONFIG/pipeline/training/safety.yaml`
  - Suspect triggers: `top1_share` (30%+), `top1_over_top2` (3×), `hard_top1_share` (40%+)
  - Confirm logic: Rerun once with suspects removed, evaluate score drop (absolute 0.15 or relative 25%)
  - Escalation policy: Quarantine feature-level, only block target/view if leakage persists after quarantine

- **Core Module**: Created `TRAINING/ranking/utils/dominance_quarantine.py`
  - `DominanceConfig`, `Suspect`, `ConfirmResult` dataclasses
  - `detect_suspects()` - Analyzes per-model importance percentages
  - `confirm_quarantine()` - Evaluates confirm result based on score drops
  - `write_suspects_artifact_with_data()` - Writes suspects to disk
  - `persist_confirmed_quarantine()` - Persists confirmed quarantine
  - `load_confirmed_quarantine()` - Loads confirmed quarantine for runtime filtering

- **Integration**:
  - **Suspect Detection**: Integrated in `model_evaluation.py` after importance computation
  - **Confirm Pass**: Reruns `train_and_evaluate_models()` with suspects removed, compares pre/post scores
  - **Runtime Quarantine**: Applied in `shared_ranking_harness.py`, `multi_model_feature_selection.py`, and `training.py`
  - **Escalation Policy**: Modified `metrics_aggregator.py` to downgrade BLOCKED to SUSPECT if confirmed quarantine exists

**Artifacts**:
- Suspects: `targets/<target>/reproducibility/feature_quarantine/suspects_<view>_<symbol>.json`
- Confirmed: `targets/<target>/reproducibility/feature_quarantine/confirmed_quarantine.json`

**Impact**: Prevents permanent feature drops on first trigger, allows recovery via rerun, only escalates to blocking target/view if leakage persists after quarantine.

### 2. Hard-Exclude Forward-Looking Features

**Safety Enhancement**: Hard-exclude `time_in_profit_*` and similar forward-looking profit/PnL features for forward-return targets.

**Implementation**:
- Added semantic rule in `target_conditional_exclusions.py` for forward-return targets
- Patterns excluded:
  - `time_in_profit_*` (e.g., `time_in_profit_15m`)
  - `profit_.*forward`
  - `pnl_.*forward`
  - `.*_profit_.*` (features with "profit" in name extending beyond horizon)

**Rationale**: Features like `time_in_profit_15m` that measure "profit over the next X minutes" are structurally future-looking for `fwd_ret_10m` targets. This prevents the dominance quarantine cycle from even starting.

**Impact**: Prevents label-proxy leakage before it can trigger dominance detection.

### 3. Small-Panel Leniency

**Safety Enhancement**: Config-driven small-panel leniency to downgrade blocking behavior when `n_symbols < threshold`.

**Implementation**:
- **Config**: Added `small_panel` section to `CONFIG/pipeline/training/safety.yaml`
  - `enabled: true`
  - `min_symbols_threshold: 10` (apply leniency when n_symbols < this)
  - `downgrade_block_to_suspect: true`
  - `log_warning: true`

- **Leakage Detection**: Modified `detect_leakage()` in `leakage_detection.py`
  - Checks `n_symbols` from symbols array
  - Downgrades `HIGH_SCORE`/`SUSPICIOUS` to `SUSPECT` when `n_symbols < threshold`
  - Logs warning when leniency is applied

- **Metrics Aggregator**: Modified `_load_leakage_status()` in `metrics_aggregator.py`
  - Loads `n_symbols` from run context
  - Applies small-panel leniency before checking confirmed quarantine
  - Downgrades `BLOCKED` to `SUSPECT` for small panels

**Impact**: Allows dominance quarantine to attempt recovery before blocking target/view in small-panel scenarios (common in E2E tests).

### 4. Fix detect_leakage Import Conflict

**Bug Fix**: Fixed `TypeError: unexpected keyword argument 'X'` crash in target ranking.

**Root Cause**: Two `detect_leakage` functions existed:
1. `TRAINING/ranking/predictability/leakage_detection.py` - Updated signature with `X`, `y`, `time_vals`, `symbols` parameters
2. `TRAINING/ranking/predictability/model_evaluation/leakage_helpers.py` - Old signature without those parameters

A later import at line 4980 in `model_evaluation.py` was overriding the correct version.

**Fix**:
- Removed conflicting import from `leakage_helpers.py` (commented out with explanation)
- Updated `model_evaluation/__init__.py` to import from `leakage_detection.py` instead
- Added error handling around both `detect_leakage()` calls (main and post-quarantine)
  - On error, logs exception and sets `leakage_flag = "UNKNOWN"` (safe default)
  - Allows pipeline to continue even if leakage detection fails internally

**Impact**: Prevents `TypeError` crashes, ensures correct function is used, makes pipeline resilient to leakage detection errors.

## Files Changed

1. `CONFIG/pipeline/training/safety.yaml` - Added `dominance_quarantine` and `small_panel` config sections
2. `TRAINING/ranking/utils/dominance_quarantine.py` - NEW: Core dominance quarantine module
3. `TRAINING/ranking/predictability/model_evaluation.py` - Integrated suspect detection, confirm pass, error handling
4. `TRAINING/ranking/shared_ranking_harness.py` - Applied runtime quarantine filtering
5. `TRAINING/ranking/multi_model_feature_selection.py` - Applied runtime quarantine filtering
6. `TRAINING/training_strategies/execution/training.py` - Applied runtime quarantine filtering
7. `TRAINING/orchestration/metrics_aggregator.py` - Small-panel leniency and escalation policy
8. `TRAINING/orchestration/training_router.py` - Escalation policy verification
9. `TRAINING/ranking/utils/target_conditional_exclusions.py` - Added forward-return exclusion rule
10. `TRAINING/ranking/predictability/leakage_detection.py` - Added small-panel leniency logic
11. `TRAINING/ranking/predictability/model_evaluation/__init__.py` - Fixed import to use correct `detect_leakage`
12. `tests/test_dominance_quarantine.py` - NEW: Unit tests for dominance quarantine
13. `tests/test_dominance_quarantine_integration.py` - NEW: Integration tests

## Testing

- Unit tests for `detect_suspects()`, `confirm_quarantine()`, and artifact writing
- Integration test for full flow: suspect → confirm → quarantine → escalation
- Forward-exclusion test: Verify `time_in_profit_*` features excluded for forward-return targets
- Small-panel test: Verify leakage warnings downgraded, not blocked when `n_symbols < 10`
- Recovery test: Verify dominance quarantine can recover before blocking target/view

## Impact

- **Leakage Detection**: More robust with feature-level quarantine and recovery mechanism
- **Pipeline Stability**: Leakage detection failures no longer crash target evaluation
- **Small-Panel Support**: Better handling of E2E test scenarios with few symbols
- **Forward-Looking Prevention**: Hard-exclusion prevents label-proxy features from entering pipeline
- **Config-Driven**: All thresholds and behaviors controlled via config (SST compliance)

