# 2026-01-09: Scoring Eligibility Gates and View Cache Conflict Fix

## Scoring Calibration Fixes: Eligibility Gates and Quality Formula

### Fixed Double-Counting of Standard Error in Quality Calculation
- **Removed Stability from Quality**: Fixed issue where standard error (SE) was double-penalized in both t-stat (skill) and stability (quality) components
  - **Root Cause**: SE was used in both `skill = sigmoid(tstat/k)` where `tstat = mean / se`, and `stability = 1 / (1 + se/se_ref)`, causing double-penalty for high uncertainty
  - **Fix**: Removed stability from quality calculation. Quality now = `coverage × registry_coverage × sample_size` (multiplicative, not weighted sum)
  - **Impact**: SE uncertainty is now penalized only once (in t-stat), making scoring more coherent. Quality reflects things not already in t-stat (coverage, registry coverage, sample size)
  - **Files Modified**:
    - `TRAINING/ranking/predictability/composite_score.py`: Removed stability from quality (lines 391-407)
    - `CONFIG/ranking/metrics_schema.yaml`: Updated weights section, removed w_stab (lines 141-146)

### Added Eligibility Gates for Registry Coverage and Sample Size
- **Hard Gates for Ranking Eligibility**: Added explicit eligibility gates to prevent low-quality targets from ranking high
  - **Registry Coverage Gate**: Targets with `registry_coverage < 0.95` are marked `valid_for_ranking = False` with reason `LOW_REGISTRY_COVERAGE`
    - In eval mode: `registry_coverage_rate = None` fails gate (MISSING_REGISTRY_COVERAGE)
    - Prevents ranking targets with high unknown feature count (screen vs strict modes disjoint)
  - **Sample Size Gate**: Targets with `n_slices_valid < 20` are marked `valid_for_ranking = False` with reason `LOW_N_CS`
    - Prevents ranking targets with too few cross-sections for reliable statistics
  - **Sample Size Penalty**: Targets with `n_slices_valid` between 20-30 get quality penalty (0.7-1.0 linear interpolation), but still valid for ranking
  - **Run Intent Gate**: Smoke test runs are marked invalid for ranking (SMOKE_INTENT)
  - **Impact**: Only targets with `registry_coverage >= 0.95` and `n_slices_valid >= 20` are eligible for ranking. Clear eligibility via `valid_for_ranking` and `invalid_reasons` fields, not hidden in score
  - **Files Modified**:
    - `TRAINING/ranking/predictability/composite_score.py`: Added eligibility gates (lines 364-389), updated quality formula (lines 396-407)
    - `TRAINING/ranking/predictability/scoring.py`: Added eligibility fields to TargetPredictabilityScore (lines 155-158, 343-348)
    - `TRAINING/ranking/predictability/model_evaluation.py`: Pass eligibility to result (lines 7352-7393, 7790-7792)
    - `TRAINING/ranking/target_ranker.py`: Filter invalid targets from ranking (lines 1098-1106, 1135-1136)
    - `CONFIG/ranking/metrics_schema.yaml`: Added eligibility section (lines 163-169), bumped version to 1.2 (line 121)

### Scoring Signature and Versioning Updates
- **Deterministic Scoring Signature**: Updated scoring signature calculation to include eligibility params for determinism
  - Eligibility params (min_registry_coverage, require_registry_coverage_in_eval, min_n_for_ranking, min_n_for_full_quality) included in signature
  - Uses canonical JSON with sorted keys (SST pattern)
  - Version bumped to 1.2 in config
  - **Files Modified**:
    - `TRAINING/ranking/predictability/composite_score.py`: Updated scoring signature (lines 319-333), updated definition string (lines 428-435)

## View Cache Conflict Fix: SS→CS Promotion Root Cause

### Fixed View Cache Overriding Explicitly Requested Views
- **Root Cause**: When a cached view entry existed for a universe, it was unconditionally reused without checking if `requested_view` conflicted
  - **Scenario**: First evaluation (CROSS_SECTIONAL with 10 symbols) caches view for universe. Second evaluation (SYMBOL_SPECIFIC with 1 symbol, same universe) finds cached entry and reuses CROSS_SECTIONAL → conflict!
  - **Symptom**: `resolve_write_scope()` detects mismatch: `caller_view=SYMBOL_SPECIFIC` but `SST view=CROSS_SECTIONAL` → warns and forces `view_for_writes=SYMBOL_SPECIFIC`
- **Fix**: Added conflict checking before reusing cached view
  - Cache is only reused when:
    1. `requested_view` is None (auto mode), OR
    2. `requested_view` matches cached view
  - On conflict: Resolve fresh based on `requested_view` and panel size, log warning
  - Uses View enum for normalization (SST pattern)
  - **Impact**: Explicit SYMBOL_SPECIFIC requests no longer overridden by cached CROSS_SECTIONAL. Cache still reused when appropriate (no conflicts). Clear diagnostics via warning message
  - **Files Modified**:
    - `TRAINING/ranking/utils/cross_sectional_data.py`: Added conflict check (lines 389-454), fixed validation and save logic (lines 456-478)

### Fixed Multi-Symbol SYMBOL_SPECIFIC View Validation
- **Root Cause**: Auto-resolution logic didn't validate that `requested_view=SYMBOL_SPECIFIC` is compatible with `n_symbols_available > 1`. Multi-symbol runs (e.g., universe `f517a23ce02cdcad4887b95107f165cc69f15796ccfd07c3b8e1466fbd2102f5` with 10 symbols) incorrectly resolved to SYMBOL_SPECIFIC view, causing files to be written to `SYMBOL_SPECIFIC/` folders instead of `CROSS_SECTIONAL/` folders
- **Routing Impact**: The resolved view flows through entire pipeline:
  1. `resolved_data_config['view']` → used in `resolve_write_scope()` → `view_for_writes`
  2. `view_for_writes` → used in all path construction (`get_scoped_artifact_dir()`, `target_repro_dir()`, `ArtifactPaths.model_dir()`)
  3. **Path structure**:
     - CROSS_SECTIONAL: `targets/{target}/reproducibility/CROSS_SECTIONAL/universe={universe_sig}/artifact_type/`
     - SYMBOL_SPECIFIC: `targets/{target}/reproducibility/SYMBOL_SPECIFIC/symbol={symbol}/universe={universe_sig}/artifact_type/`
- **Fix**: Added validation to prevent SYMBOL_SPECIFIC view when `n_symbols_available > 1`
  - **Auto-resolution validation**: Before resolving view, validate `requested_view=SYMBOL_SPECIFIC` is compatible with `n_symbols`. If `n_symbols > 1`, clear invalid request and resolve to CROSS_SECTIONAL
  - **Cache validation**: Cached SYMBOL_SPECIFIC view is not reused for multi-symbol runs (requires `n_symbols=1`)
  - **FEATURE_SELECTION validation**: Added single-symbol check (`len(symbols) == 1`) before auto-detecting SYMBOL_SPECIFIC in `feature_selector.py` and `multi_model_feature_selection.py`
  - **Impact**: Multi-symbol runs now correctly route to `CROSS_SECTIONAL/universe={universe_sig}/` directories. Files no longer incorrectly written to `SYMBOL_SPECIFIC/symbol=.../` folders for multi-symbol runs. Clear warnings when invalid SYMBOL_SPECIFIC request is overridden
  - **Files Modified**:
    - `TRAINING/ranking/utils/cross_sectional_data.py`: Added validation before auto-resolution (lines 432-454), added cache compatibility check (lines 417-421)
    - `TRAINING/ranking/feature_selector.py`: Added single-symbol validation in auto-detection (lines 266-270)
    - `TRAINING/ranking/multi_model_feature_selection.py`: Added single-symbol validation using metadata symbols list (lines 5116-5132)

### SST Principles Maintained
- Uses View enum from `scope_resolution` for consistent view handling
- Normalizes views using `View.from_string()` (SST pattern)
- All parameters from config, no hardcoded defaults
- Clear separation: cache reuse vs fresh resolution
- Deterministic logic (canonical JSON with sorted keys for scoring signature)
