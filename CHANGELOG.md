# Changelog

All notable changes to FoxML Core will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

**For detailed technical changes:**
- [Changelog Index](DOCS/02_reference/changelog/README.md) – Per-day detailed changelogs with file paths, config keys, and implementation notes.

---

## [Unreleased]

### Recent Highlights

#### Single Source of Truth for Lookback Computation (2025-12-13) – **NEW**
- **Eliminated Split-Brain**: Fixed critical issue where different code paths computed different lookback values for the same feature set
- **Canonical Lookback Map**: All stages (sanitizer, gatekeeper, POST_PRUNE, POST_GATEKEEPER) now use the same canonical map from `compute_feature_lookback_max()`
- **POST_PRUNE Invariant Check**: Hard-fail check ensures `max(canonical_map[features]) == computed_lookback` (prevents regression)
- **Diagnostic Logging**: Gatekeeper/sanitizer now log ALL features exceeding cap (makes split-brain visible)
- **_Xd Pattern Inference**: Day-suffix features (`_60d`, `_20d`, `_3d`) now correctly inferred to `days * 1440` minutes
- **Unknown Lookback = Unsafe**: Unknown lookback now treated as `inf` (unsafe), not `0.0` (safe)
- **Readline Library Conflict Fix**: Fixed subprocess readline conflicts caused by Cursor AppImage's `LD_LIBRARY_PATH`
- **Results**: Sanitizer quarantines 1440m features upfront, `actual_max=150m <= cap=240m` after pruning, no late-stage CAP VIOLATION
- See [Single Source of Truth Fix](DOCS/02_reference/changelog/2025-12-13-single-source-of-truth.md) for details

#### Leakage Controls Structural Fixes (2025-12-13) – **NEW**
- **Unified Leakage Budget Calculator**: Single source of truth for feature lookback calculation (`TRAINING/utils/leakage_budget.py`)
- **Calendar Feature Classification**: Fixed calendar features (`day_of_week`, `holiday_dummy`, etc.) to resolve to 0m lookback (not 1440m)
- **Separate Purge/Embargo Validation**: Fixed validation to enforce two separate constraints:
  - `purge` covers feature lookback
  - `embargo` covers target horizon
  - Previously incorrectly required `purge >= lookback + horizon`
- **Hard-Stop on Violations**: Configurable policy (`strict`/`drop_features`/`warn`) with proper enforcement
- **Config-Driven Leakage Control**: New explicit config knobs for auditable behavior:
  - `over_budget_action`: `drop` | `hard_stop` | `warn` (what to do when features exceed budget)
  - `lookback_budget_minutes`: `auto` | `<number>` (how to compute required budget)
  - `lookback_buffer_minutes`: `<number>` (safety buffer for lookback budget)
  - `cv.embargo_extra_bars`: `<number>` (extra bars for embargo safety margin)
  - All settings are explicit and auditable (no vague toggles)
- **LeakageAssessment Dataclass**: Prevents contradictory reason strings like "overfit_likely; cv_not_suspicious"
- **CV Splitter Logging**: Run summary now shows splitter identity, purge, embargo, and max_feature_lookback_minutes
- **Policy Explicit Logging**: Active policy and feature drop lists are logged for auditability
- **Fingerprint Tracking System**: Set-invariant fingerprints with order-change detection ensure lookback computed on exact final feature set
- **LookbackResult Dataclass**: Type-safe return type prevents silent mis-wires
- **Explicit Stage Logging**: PRE_GATEKEEPER, POST_GATEKEEPER, POST_PRUNE, MODEL_TRAIN_INPUT stages logged with fingerprints
- **Leakage Canary Test**: Dedicated test config for pipeline integrity validation using known-leaky targets
- See [2025-12-13 leakage validation fix](DOCS/03_technical/fixes/2025-12-13-leakage-validation-fix.md), [fingerprint tracking](DOCS/03_technical/fixes/2025-12-13-lookback-fingerprint-tracking.md), and [implementation status](DOCS/03_technical/architecture/LEAKAGE_CONTROLS_IMPLEMENTATION_STATUS.md) for details

#### Feature Selection Unification & Critical Fixes (2025-12-13) – **NEW**
- **Shared Ranking Harness**: Feature selection now uses same harness as target ranking, ensuring identical evaluation contracts
- **Comprehensive Hardening**: Feature selection now has complete parity with target ranking (ghost busters, leak scan, stability tracking, linear models)
- **Same Output Structure**: Feature selection saves results in same format as target ranking (CSV, YAML, REPRODUCIBILITY structure)
- **Config-Driven**: Feature selection uses same config hierarchy and loading methods as target ranking
- **Critical Fixes**: Fixed shared harness unpack crashes, CatBoost dtype mis-typing, RFE/linear model failures, stability cross-model mixing, telemetry scoping issues
- **Last-Mile Improvements**: Failed model skip reasons in consensus summary, feature universe fingerprint for stability tracking
- See [2025-12-13 feature selection unification changelog](DOCS/02_reference/changelog/2025-12-13-feature-selection-unification.md) and [implementation verification](DOCS/03_technical/fixes/2025-12-13-implementation-verification.md) for details

#### Generalized Duration Parsing System & Audit Fixes (2025-12-13) – **NEW**
- **Duration Parsing System**: New duration parsing system handles time period formats (minutes, hours, days, bars, compound durations)
- **Interval-Aware Strictness**: Primary mechanism uses data resolution for purge/lookback enforcement
- **Fail-Closed Policy**: No silent fallbacks on parsing errors; non-auditable status impossible to miss
- **Lookback Detection Fix**: Fixed false positives where features with explicit short lookbacks (`_15m`, `_30m`) were incorrectly tagged as 1440m
- **Documentation Review**: Initial review completed to ensure realistic statements about capabilities
- See [2025-12-13 duration system changelog](DOCS/02_reference/changelog/2025-12-13-duration-system.md) for details

#### Documentation Organization (2025-12-13) – **NEW**
- **Index Files**: Created README.md index files for all DOCS subdirectories
- **Config Migration Docs**: Moved scattered CONFIG documentation to `DOCS/02_reference/configuration/migration/`
- **Audit Documentation**: Organized documentation audits in `DOCS/02_reference/audits/`
- **Cross-Linking**: Fixed all broken cross-links after file moves
- See [2025-12-13 changelog](DOCS/02_reference/changelog/2025-12-13.md) for details

#### Config Path Consolidation & Config Trace System (2025-12-13) – **NEW**
- **Config Reorganization**: Major restructure of `CONFIG/` directory into modular structure
- **Config Trace**: Comprehensive logging shows where each config value comes from
- **Max Samples Fix**: Fixed experiment config `max_samples_per_symbol` not being read from YAML
- **Output Directory Binning**: Now uses configured `max_rows_per_symbol` instead of full dataset size
- See [2025-12-13 changelog](DOCS/02_reference/changelog/2025-12-13.md) for details

#### License & Commercial Use Banner (2025-12-12) – **NEW**
- **Terminal Billboard**: Professional startup banner prints licensing information on every run
- **30-Day Evaluation Period**: Banner includes 30-day evaluation period for commercial organizations
- **Compliance**: Ensures users see licensing requirements even in automated systems
- See [2025-12-12 changelog](DOCS/02_reference/changelog/2025-12-12.md) for details

#### GPU Acceleration (2025-12-12) – **NEW**
- **GPU support**: XGBoost, CatBoost, and LightGBM now use GPU acceleration when available
- **Config-driven**: All GPU settings from `gpu_config.yaml` (SST)
- **Performance**: Significantly faster on large datasets (>100k samples)
- See [GPU Setup Guide](DOCS/01_tutorials/setup/GPU_SETUP.md) and [2025-12-12 changelog](DOCS/02_reference/changelog/2025-12-12.md) for details

#### Experiment Configuration System (2025-12-12) – **NEW**
- **Experiment configs**: Create reusable experiment configurations in `CONFIG/experiments/*.yaml`
- **Auto target discovery**: Automatically discover and rank all targets from your dataset
- See [Experiment Config Guide](DOCS/01_tutorials/configuration/EXPERIMENT_CONFIG_GUIDE.md) for details

#### Active Sanitization (Ghost Buster) (2025-12-12) – **NEW**
- **Proactive feature quarantine**: Automatically removes problematic features before training starts
- **Config-driven**: Fully configurable via `safety_config.yaml`
- See [Active Sanitization Guide](DOCS/03_technical/implementation/ACTIVE_SANITIZATION.md) for details

#### Critical Bug Fixes (2025-12-12) – **FIXED**
- **Mutual Information**: Fixed `random_state` SST compliance
- **XGBoost 3.1+ GPU**: Fixed compatibility with XGBoost 3.1+ (removed `gpu_id` parameter)
- **CatBoost GPU**: Added explicit verification that `task_type='GPU'` is set
- **Process Deadlock**: Fixed readline library conflict causing process hangs
- See [2025-12-12 changelog](DOCS/02_reference/changelog/2025-12-12.md) for complete list

---

### Stability Guarantees

- **Training results reproducible** across runs and hardware (centralized seeds + config-only hyperparameters)
- **Complete config centralization** – Pipeline behavior is controlled by YAML (Single Source of Truth)
- **SST enforcement** – Automated test prevents accidental reintroduction of hardcoded hyperparameters
- **Config schema backward compatible** – Existing configs continue to work with deprecation warnings where applicable
- **Modular architecture** – TRAINING module is self-contained with zero external script dependencies

### Known Issues & Limitations

- **Trading / execution modules** are out of scope for the core repo; FoxML Core focuses on ML research & infra
- **Feature engineering** still requires human review and domain validation
- **Full end-to-end test suite** is being expanded following SST + determinism changes
- **LOSO CV splitter**: LOSO view currently uses combined data; dedicated CV splitter is a future enhancement
- **Placebo test per symbol**: Symbol-specific strong targets should be validated with placebo tests - future enhancement

**For complete list:** See [Known Issues & Limitations](DOCS/02_reference/KNOWN_ISSUES.md)

---

### Added

**Recent additions (2025-12-12 - 2025-12-13):**
- **Shared Ranking Harness** - Unified evaluation contract for target ranking and feature selection
- **Feature Selection Unification** - Feature selection now has complete parity with target ranking
- Experiment Configuration System - Reusable experiment configs with auto target discovery
- Active Sanitization System - Proactive feature quarantine before training
- Dual-View Target Ranking System - Multiple evaluation views with automatic routing
- Config-Based Pipeline Interface - Minimal command-line usage
- Sample Size Binning System - Automatic binning of runs by sample size
- Trend Analysis System - Automated trend tracking across runs
- License & Commercial Use Banner - Professional startup banner with licensing information
- Training Routing & Planning System - Config-driven routing decisions
- Reproducibility Tracking System - End-to-end tracking with STABLE/DRIFTING/DIVERGED classification

**For complete details:** See [Changelog Index](DOCS/02_reference/changelog/README.md)

---

### Fixed

**Recent fixes (2025-12-12 - 2025-12-13):**
- **Leakage Controls Structural Fixes** (2025-12-13):
  - Fixed audit and gatekeeper using different lookback calculations (now unified)
  - Fixed calendar features incorrectly classified as 1440m lookback (now 0m)
  - Fixed validation incorrectly requiring `purge >= lookback + horizon` (now separate constraints)
  - Fixed contradictory reason strings ("overfit_likely; cv_not_suspicious")
  - Fixed missing import (`Any` from typing) in `leakage_budget.py`
  - Fixed missing import (`create_resolved_config`) in pruning section
  - Fixed top offenders list showing uncapped values when cap is applied
  - Fixed lookback mismatch warnings ("reported max=100.0m but actual max=86400.0m") with fingerprint tracking
  - Fixed MODEL_TRAIN_INPUT fingerprint computed before pruning (now POST_PRUNE)
  - Fixed gatekeeper missing features by using unified lookback calculator
  - Fixed tuple vs dataclass return type mismatch (`AttributeError: 'tuple' object has no attribute 'max_minutes'`)
  - Fixed NameError issues during fingerprint tracking implementation
  - Fixed TypeError: unexpected keyword argument in wrapper functions
- **Feature Selection Critical Fixes** (2025-12-13):
  - Fixed shared harness unpack crashes (tolerant unpack with length checking)
  - Fixed CatBoost treating numeric columns as text/categorical (hard dtype enforcement guardrail)
  - Fixed RFE `KeyError: 'n_features_to_select'` (safe defaults + clamping to [1, n_features])
  - Fixed Ridge/ElasticNet "Unknown model family" errors (full implementations with StandardScaler)
  - Fixed stability cross-model mixing (per-model-family snapshots, feature universe fingerprint)
  - Fixed telemetry scoping issues (view→route_type mapping, symbol=None for CROSS_SECTIONAL, cohort_id filtering)
  - Fixed uniform importance fallback polluting consensus (raises ValueError, marks model invalid)
  - Added failed model skip reasons in consensus summary (e.g., `ridge:zero_coefs`)
  - Added feature universe fingerprint for stability tracking (prevents comparing different candidate sets)
- Documentation Link Fixes - Fixed 404 errors when navigating between documentation
- Resolved Config System - Fixed purge calculation bug
- Reproducibility Tracker Fixes - Fixed NameError and save path issues
- Critical Horizon Unit Bug - Fixed barrier target generation using incorrect horizon units
- XGBoost 3.1+ GPU Compatibility - Fixed `gpu_id has been removed since 3.1` error
- CatBoost GPU Verification - Added explicit verification that `task_type='GPU'` is set
- Process Deadlock Fix - Fixed readline library conflict causing process hangs
- Config Path Consolidation - Fixed experiment config `max_samples_per_symbol` not being read from YAML

**For complete details:** See [Changelog Index](DOCS/02_reference/changelog/README.md)

---

### Changed

- **Determinism & Defaults** (2025-12-10): Removed hardcoded `random_state` and similar defaults
- **Logging** (2025-12-10): Replaced scattered logging flags with structured YAML-driven configuration
- **Documentation** (2025-12-09+): Restructured into 4-tier hierarchy with improved cross-linking

---

### Security

- Enhanced compliance and production-use documentation
- Documented license enforcement procedures and copyright notice requirements

---

## Versioning

Releases follow [Semantic Versioning](https://semver.org/):

- **MAJOR** – Incompatible API changes
- **MINOR** – Backwards-compatible functionality
- **PATCH** – Backwards-compatible bug fixes

## Categories

- **Added** – New features  
- **Changed** – Changes in existing functionality  
- **Fixed** – Bug fixes  
- **Security** – Security / compliance improvements  
- **Documentation** – Documentation changes
