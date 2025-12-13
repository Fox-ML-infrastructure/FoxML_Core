# Changelog

All notable changes to FoxML Core will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

**For detailed technical changes:**
- [Changelog Index](DOCS/02_reference/changelog/README.md) – Per-day detailed changelogs with file paths, config keys, and implementation notes.

---

## [Unreleased]

### Highlights

#### GPU Acceleration for Target Ranking & Feature Selection (2025-12-12) – **NEW**
- **GPU support**: XGBoost, CatBoost, and LightGBM now use GPU acceleration when available
- **XGBoost 3.1+ compatible**: Automatically handles both new API (`device='cuda'`) and legacy API (`gpu_hist`)
- **CatBoost verification**: Explicit verification that `task_type='GPU'` is set (required for GPU usage)
- **Automatic detection**: Gracefully falls back to CPU if GPU unavailable
- **Config-driven**: All GPU settings from `gpu_config.yaml` (SST)
- **Performance**: Significantly faster on large datasets (>100k samples)
- **Enhanced logging**: Always logs GPU status with verification messages
- See [GPU Setup Guide](DOCS/01_tutorials/setup/GPU_SETUP.md) for configuration

#### Critical Bug Fixes (2025-12-12) – **FIXED**
- **Mutual Information**: Fixed `random_state` SST compliance (no more KeyError)
- **Audit violations**: Fixed false violations when Final Gatekeeper drops features
- **CatBoost**: Fixed feature importance calculation (requires training dataset)
- **XGBoost 3.1+ GPU**: Fixed compatibility with XGBoost 3.1+ (removed `gpu_id` parameter, now uses `device='cuda'`)
- **CatBoost GPU**: Added explicit verification that `task_type='GPU'` is set (CatBoost requires this to use GPU)
- **GPU Detection**: Made fully config-driven - all GPU settings from `gpu_config.yaml` (SST)
- All fixes maintain SST compliance (no hardcoded values)

#### Experiment Configuration System (2025-12-12) – **NEW**
- **Experiment configs**: Create reusable experiment configurations in `CONFIG/experiments/*.yaml`
- **Auto target discovery**: Automatically discover and rank all targets from your dataset
- **Flexible configuration**: Support for auto-discovery or manual target/feature selection
- **Validation**: Smart validation that adapts based on `auto_targets` setting
- **Documentation**: Complete guide in [Experiment Config Guide](DOCS/01_tutorials/configuration/EXPERIMENT_CONFIG_GUIDE.md)

#### Active Sanitization (Ghost Buster) (2025-12-12) – **NEW**
- **Proactive feature quarantine**: Automatically removes problematic features before training starts
- **Prevents discrepancies**: Eliminates conflicts where audit and auto-fix see different lookback values
- **Config-driven**: Fully configurable via `safety_config.yaml`
- See [Active Sanitization Guide](DOCS/03_technical/implementation/ACTIVE_SANITIZATION.md)

#### Complete Config-Driven Decision System (2025-12-12) – **NEW**
- **All thresholds configurable**: Decision-making and stability analysis thresholds now load from config
- **SST compliance**: No hardcoded thresholds remain in decision-making code
- **Config organization**: Documented which config files are used vs unused
- See [Config Audit](DOCS/02_reference/configuration/CONFIG_AUDIT.md) for details

#### Critical Horizon Unit Fix (2025-12-12) – **FIXED**
- **Bug fixed**: Barrier target generation incorrectly used `horizon_minutes` as `horizon_bars`
- **Impact**: All existing labeled datasets generated before this fix should be regenerated
- **Solution**: Versioned dataset generation with metadata tracking
- See detailed changelog for technical details

#### Dual-View Target Ranking (2025-12-12) – **NEW**
- **Multiple evaluation views**: Cross-sectional, symbol-specific, and LOSO (Leave-One-Symbol-Out)
- **Automatic routing**: Determines which view(s) to use per target
- **View consistency**: Maintains same view across ranking → feature selection → training
- See [Dual-View Target Ranking Guide](DOCS/03_technical/implementation/DUAL_VIEW_TARGET_RANKING.md)

#### Simplified Config-Based Pipeline (2025-12-12) – **NEW**
- **Minimal command-line usage**: All settings configurable in YAML files
- **Simple commands**: Run full pipeline with just `--output-dir` argument
- **Quick test mode**: `--quick` flag for fast iteration
- See [Simple Pipeline Usage Guide](DOCS/01_tutorials/SIMPLE_PIPELINE_USAGE.md)

#### Resolved Config System (2025-12-12) – **NEW**
- **Centralized resolution**: Single source of truth for all resolved configuration values
- **Consistent logging**: Single authoritative log line per category
- **Fixed purge calculation**: Corrected bug causing excessive purge windows
- See [Resolved Config Fix Guide](DOCS/03_technical/implementation/RESOLVED_CONFIG_FIX.md)

#### Trend Analysis System (2025-12-12) – **NEW**
- **Automated trend tracking**: Performance trends across runs with exponential decay weighting
- **Multi-stage support**: Integrated into target ranking, feature selection, and cross-sectional ranking
- **Audit-grade metadata**: Trend metadata stored in `metadata.json` for verification
- See [Trend Analyzer Verification Guide](DOCS/03_technical/implementation/TREND_ANALYZER_VERIFICATION.md)

#### Cohort-Aware Reproducibility (2025-12-12) – **NEW**
- **Cohort-based organization**: Runs organized by data cohort (sample size, symbols, date range, config)
- **Sample size binning**: RESULTS directory organized by sample size bins for easy comparison
- **Enhanced metadata**: Full symbols list and bin metadata in `metadata.json`
- See [Cohort-Aware Reproducibility Guide](DOCS/03_technical/implementation/COHORT_AWARE_REPRODUCIBILITY.md)

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
- **Full end-to-end test suite** is being expanded following SST + determinism changes (as of 2025-12-10)
- **LOSO CV splitter**: LOSO view currently uses combined data; dedicated CV splitter for train-on-all-but-one, validate-on-one is a future enhancement
- **Placebo test per symbol**: Symbol-specific strong targets should be validated with placebo tests (shuffle labels, assert AUC ~ 0.5) - future enhancement

---

### Added

- **Experiment Configuration System** (2025-12-12)
  - Experiment config files (`CONFIG/experiments/*.yaml`) for reusable experiment definitions
  - Auto target discovery and ranking
  - Flexible manual/auto target and feature selection
  - Comprehensive validation and error messages
  - See [Experiment Config Guide](DOCS/01_tutorials/configuration/EXPERIMENT_CONFIG_GUIDE.md)

- **Active Sanitization System** (2025-12-12)
  - Proactive feature quarantine before training
  - Configurable via `safety_config.yaml`
  - See [Active Sanitization Guide](DOCS/03_technical/implementation/ACTIVE_SANITIZATION.md)

- **Versioned Labeled Dataset Generation** (2025-12-12)
  - Scripts for generating versioned labeled datasets with corrected barrier targets
  - Metadata tracking (barrier_version, commit_hash, generation_date)
  - Validation tools for comparing old vs new labels

- **Dual-View Target Ranking System** (2025-12-12)
  - Multiple evaluation views (CROSS_SECTIONAL, SYMBOL_SPECIFIC, LOSO)
  - Automatic routing decisions
  - View consistency across pipeline stages

- **Config-Based Pipeline Interface** (2025-12-12)
  - New config file: `CONFIG/training_config/intelligent_training_config.yaml`
  - Minimal command-line usage
  - Test mode auto-detection

- **Sample Size Binning System** (2025-12-12)
  - Automatic binning of runs by sample size
  - RESULTS directory organized by bins for easy comparison
  - Audit-grade binning with versioning

- **Trend Analysis System** (2025-12-12)
  - Automated trend tracking across runs
  - Exponential decay weighting
  - Multi-stage integration (target ranking, feature selection, cross-sectional ranking)

- **Training Routing & Planning System** (2025-12-11)
  - Config-driven routing decisions
  - Automatic routing plan generation
  - Training plan generator
  - See [Training Routing Guide](DOCS/02_reference/training_routing/README.md)

- **Reproducibility Tracking System** (2025-12-11)
  - Reusable `ReproducibilityTracker` with tolerance-based comparisons
  - STABLE/DRIFTING/DIVERGED classification
  - Cross-sectional stability tracking
  - See [Reproducibility Tracking Guide](DOCS/03_technical/implementation/REPRODUCIBILITY_TRACKING.md)

---

### Fixed

- **Resolved Config System** (2025-12-12)
  - Centralized configuration resolution
  - Fixed purge calculation bug (was causing 1465m purge instead of 85m)
  - Consistent logging with single authoritative messages
  - See [Resolved Config Fix Guide](DOCS/03_technical/implementation/RESOLVED_CONFIG_FIX.md)

- **Reproducibility Tracker Fixes** (2025-12-12)
  - Fixed `ctx` NameError preventing metadata writes
  - Fixed feature importances save path
  - Fixed perfect CV detection false positives
  - Added missing metadata diagnostics

- **Critical Horizon Unit Bug** (2025-12-12)
  - Fixed barrier target generation using incorrect horizon units
  - All target functions now correctly convert `horizon_minutes` to `horizon_bars`
  - Time contract enforcement added

- **Time & Interval Handling** (2025-12-11)
  - Median-based gap filtering
  - Fixed-interval mode
  - Eliminated false "unclear interval" warnings

- **Parameter Passing & Validation** (2025-12-11)
  - Systematic fix for duplicated/incompatible params across all model families
  - Resolved CatBoost iteration synonyms and RNG param conflicts

- **Leakage Detection & Auto-Fixer** (2025-12-11)
  - Fixed confidence calculation
  - On-the-fly importance computation
  - Improved logging and visibility

---

### Changed

- **Determinism & Defaults** (2025-12-10)
  - Removed hardcoded `random_state` and similar defaults
  - All defaults now provided via central determinism system and config

- **Logging** (2025-12-10)
  - Replaced scattered logging flags with structured YAML-driven configuration
  - Reduced unnecessary WARNING noise

- **Documentation** (2025-12-09+)
  - Restructured into 4-tier hierarchy
  - 50+ existing docs rewritten; 50+ new docs added
  - Improved cross-linking and indices

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
