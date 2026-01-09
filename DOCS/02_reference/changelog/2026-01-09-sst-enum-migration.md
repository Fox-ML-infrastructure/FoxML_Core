# 2026-01-09: SST Enum Migration and WriteScope Adoption

## Summary

Complete migration to SST (Single Source of Truth) architecture with View/Stage enum adoption, WriteScope object migration, and unified helper functions. This ensures consistent scope handling, path construction, and data serialization across the entire codebase.

## Phase 1: View and Stage Enum Migration

### View Enum Migration (Phase 1.1)

**Problem**: Hardcoded view strings ("CROSS_SECTIONAL", "SYMBOL_SPECIFIC") scattered across 17 files, causing potential inconsistencies and making refactoring difficult.

**Fix**:
- Migrated all 67 instances across 17 files to use `View` enum from `scope_resolution.py`
- All function signatures now accept `Union[str, View]` for backward compatibility
- All comparisons use enum instances: `view_enum == View.CROSS_SECTIONAL` instead of `view == "CROSS_SECTIONAL"`
- All JSON serialization uses `view_enum.value` to ensure string output
- All path construction uses `str(view_enum)` which returns `.value` via `__str__`

**Files Updated**:
- `feature_selection_reporting.py`, `multi_model_feature_selection.py`, `metrics_aggregator.py`
- `training.py`, `artifact_paths.py`, `cohort_metadata.py`, `cross_sectional_data.py`
- `shared_ranking_harness.py`, `output_layout.py`, `diff_telemetry.py`, `target_first_paths.py`
- `intelligent_trainer.py`, `target_ranker.py`, `feature_selector.py`, `cross_sectional_feature_ranker.py`
- `model_evaluation.py`, `reproducibility_tracker.py`

**Backward Compatibility**:
- `_sanitize_for_json()` explicitly converts Enum types to `.value` for JSON serialization
- `View.from_string()` normalizes strings from JSON/metadata to enum instances
- Existing JSON files (metadata.json, snapshot.json, metrics.json) continue to work unchanged
- File paths remain identical (enum `__str__` returns `.value`)

### Stage Enum Migration (Phase 1.2)

**Problem**: Hardcoded stage strings ("TARGET_RANKING", "FEATURE_SELECTION", "TRAINING") scattered across 12 files.

**Fix**:
- Migrated all instances to use `Stage` enum from `scope_resolution.py`
- All function signatures now accept `Union[str, Stage]` for backward compatibility
- All comparisons use enum instances: `stage_enum == Stage.TARGET_RANKING`
- All JSON serialization uses `str(stage_enum)` which returns `.value`
- All path construction uses `str(stage_enum)` for consistent string conversion

**Files Updated**:
- `reproducibility_tracker.py` - Complete Stage enum migration with proper normalization
- `predictability/main.py`, `model_evaluation/reporting.py`, `leakage_detection.py` - TARGET_RANKING stage enum
- `multi_model_feature_selection.py`, `feature_selector.py` - FEATURE_SELECTION stage enum
- `target_ranker.py`, `dominance_quarantine.py` - TARGET_RANKING stage enum
- `training.py`, `reproducibility/io.py`, `reproducibility/schema.py` - TRAINING stage enum
- `intelligent_trainer.py` - TARGET_RANKING and FEATURE_SELECTION stage enum

**Backward Compatibility**:
- `Stage.from_string()` normalizes strings from JSON/metadata to enum instances (handles "MODEL_TRAINING" → "TRAINING" alias)
- Existing JSON files continue to work unchanged
- File paths remain identical

## Phase 2: WriteScope Adoption

### WriteScope Function Migration (Phase 2.1-2.2)

**Problem**: Functions accepting loose `(view, symbol, universe_sig, stage)` tuples are error-prone and don't enforce scope invariants.

**Fix**:
- Migrated 4 key functions to accept `WriteScope` objects:
  - `get_scoped_artifact_dir()` and `ensure_scoped_artifact_dir()` in `target_first_paths.py`
  - `model_output_dir()` in `target_first_paths.py`
  - `build_cohort_metadata()` in `cohort_metadata.py`
- All functions maintain backward compatibility with loose parameters
- When `scope` is provided, extracts view, symbol, universe_sig, and stage from WriteScope
- All 16 call sites remain compatible (using deprecated parameters for now)

**Impact**: Type-safe scope handling with invariant validation at construction time.

## Phase 3: Helper Function Unification

### Scope Resolution Migration (Phase 3.1)

**Problem**: Manual `resolved_data_config.get('view')` and `resolved_data_config.get('universe_sig')` patterns scattered across codebase.

**Fix**:
- Replaced manual scope extraction with `resolve_write_scope()` helper
- `feature_selector.py`: Consolidated manual universe_sig and view extraction into single `resolve_write_scope()` call
- `model_evaluation.py`: Replaced manual view/universe_sig extraction with `resolve_write_scope()` for canonical scope resolution
- All scope resolution now uses SST helper, ensuring consistent scope handling

### RunIdentity Factory Audit (Phase 3.2)

**Result**: Verified all RunIdentity constructions follow SST patterns
- Factory `create_stage_identity()` is used for creating new identities from scratch (9 instances)
- Manual constructions are for legitimate use cases: updating existing identities, finalizing partial identities, or copying with modifications
- All new identity creation uses factory pattern (SST-compliant)

### Cohort ID Unification (Phase 3.3)

**Problem**: Duplicate cohort ID generation logic in `ReproducibilityTracker._compute_cohort_id()` and `compute_cohort_id_from_metadata()`.

**Fix**:
- Created unified `compute_cohort_id()` helper in `TRAINING/orchestration/utils/cohort_id.py`
- Both implementations now delegate to unified helper (SST-compliant)
- Uses View enum for consistent view handling
- Uses `extract_universe_sig()` helper for SST-compliant universe signature access
- All cohort ID generation now uses single source of truth

### Config Hash Standardization (Phase 3.4)

**Problem**: Manual config hash computations using `json.dumps()` + `hashlib.sha256()` instead of shared helpers.

**Fix**:
- `reproducibility_tracker.py`: Replaced manual `json.dumps()` + `hashlib.sha256()` with `canonical_json()` + `sha256_short()`
- `diff_telemetry.py`: Replaced manual `hashlib.sha256()` calls with `sha256_short()` helper
- All config hashing now uses consistent logic: `canonical_json()` for normalization, `sha256_full()` or `sha256_short()` for hashing
- Hash lengths standardized: 8 chars for short hashes, 16 chars for medium, 64 chars for full identity keys

### Universe Signature Audit (Phase 3.5)

**Result**: Verified all universe signature computations use `compute_universe_signature()` helper
- All 22 instances across 10 files verified to use `compute_universe_signature()` from `run_context.py`
- Fallback manual computation in `fingerprinting.py` is defensive and acceptable (only used if helper unavailable)
- All universe signature computations now consistent and SST-compliant

## Syntax and Indentation Fixes

**Issues Fixed**:
- Fixed indentation error in `model_evaluation.py` (line 6011-6026) - try block had incorrect indentation for imports
- Fixed indentation error in `cross_sectional_feature_ranker.py` (line 703) - missing indentation after try statement
- Fixed orphaned else block in `multi_model_feature_selection.py` (line 4325) - removed duplicate else that didn't match any if
- Fixed try block indentation in `multi_model_feature_selection.py` (line 5453) - corrected indentation for code inside try block
- Fixed `UnboundLocalError` in `intelligent_trainer.py` (line 3970) - removed redundant local `Path` import that shadowed global import, causing `Path` to be referenced before assignment in `main()` function
- All Python files now compile without syntax errors (verified with `python -m py_compile` and AST parsing)
- All critical modules import successfully (verified import tests)

## SST Import Shadowing Fixes

**Issues Fixed**:
- **`model_evaluation.py:8129`**: Removed `Stage` from local import (already imported globally at line 41)
  - **Impact**: Fixes `UnboundLocalError: local variable 'Stage' referenced before assignment` at line 5390
  - **Root Cause**: Local import `from TRAINING.orchestration.utils.scope_resolution import WriteScope, ScopePurpose, Stage` shadowed global import, causing Python to treat `Stage` as a local variable throughout the function
  - **Fix**: Removed `Stage` from local import, kept only `WriteScope` and `ScopePurpose` which aren't imported globally

- **`shared_ranking_harness.py:286`**: Added global `Stage` import and removed redundant local import
  - **Impact**: Prevents `UnboundLocalError` in `create_run_context` method (line 756) where `Stage` is used
  - **Root Cause**: `Stage` was only imported locally in `build_panel` method, but also used in `create_run_context` method
  - **Fix**: Added `Stage` to global imports at line 24, removed local import at line 286

- **Redundant `Path` imports**: Removed redundant local `Path` imports where global import exists
  - `diff_telemetry.py`: Removed 3 local `Path` imports (lines 5333, 5467, 5498) - `Path` already imported globally at line 23
  - `target_routing.py`: Removed local `Path` import (line 446) - `Path` already imported globally at line 13
  - `training.py`: Removed 5 local `Path` imports (lines 667, 1357, 1410, 1621, 2356) - `Path` already imported globally at line 6
  - **Note**: These were safe (used after import) but redundant and could cause issues if code is refactored

**Verification**:
- All critical modules import without `UnboundLocalError` issues
- Path construction works correctly with enum values (verified with test paths)
- JSON serialization handles enum values correctly (enums inherit from `str`, and `_sanitize_for_json` explicitly converts to `.value`)

## Impact

- **Consistency**: All view/stage handling, scope resolution, and helper functions now use SST patterns
- **Type Safety**: WriteScope objects enforce scope invariants at construction time
- **Maintainability**: Single source of truth for all scope-related operations
- **Backward Compatibility**: All changes maintain full backward compatibility with existing JSON files, snapshots, and metrics
- **No Breaking Changes**: All file paths, JSON serialization, and existing data formats remain unchanged

## Files Changed

**Enum Migration** (29 files):
- View enum: 17 files
- Stage enum: 12 files

**WriteScope Migration** (4 functions):
- `target_first_paths.py`: 3 functions
- `cohort_metadata.py`: 1 function

**Helper Unification**:
- New file: `TRAINING/orchestration/utils/cohort_id.py`
- Updated: `reproducibility_tracker.py`, `training_strategies/reproducibility/io.py`
- Updated: `feature_selector.py`, `model_evaluation.py` (scope resolution)
- Updated: `reproducibility_tracker.py`, `diff_telemetry.py` (config hashing)

**Syntax Fixes**:
- `model_evaluation.py`: Fixed indentation in try block
- `cross_sectional_feature_ranker.py`: Fixed indentation after try statement
