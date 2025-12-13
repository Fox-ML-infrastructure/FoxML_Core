# Changelog

All notable changes to FoxML Core will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

**For detailed technical changes:**
- [Changelog Index](DOCS/02_reference/changelog/README.md) – Per-day detailed changelogs with file paths, config keys, and implementation notes.

---

## [Unreleased]

### Recent Highlights

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
