# Changelog

All notable changes to FoxML Core will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

**For detailed technical changes:**
- [Changelog Index](DOCS/02_reference/changelog/README.md) – Per-day detailed changelogs with file paths, config keys, and implementation notes.

---

## [Unreleased]

### Recent Highlights

#### Seed Tracking in Metadata (2025-12-15) – **NEW**
Fixed missing seed field in metadata.json for target ranking and feature selection runs. Seed is now properly extracted from config (`pipeline.determinism.base_seed`) and included in all reproducibility metadata.
- Seed now included in `metadata.json` for both CROSS_SECTIONAL and SYMBOL_SPECIFIC views
- Seed extracted from config and set on RunContext before logging
- Ensures full reproducibility tracking across all stages (target ranking, feature selection)
- Fixes issue where metadata.json showed `seed: null` for cross-sectional runs

#### Feature Selection Output Structure Refactor (2025-12-15) – **NEW**
Refactored feature selection output to write directly to all-caps folder structure (FEATURE_SELECTION/) instead of legacy `feature_selections/` folder.
- Removed intermediate `feature_selections/` folder structure
- Output now writes directly to `REPRODUCIBILITY/FEATURE_SELECTION/` structure
- Cleaner directory organization aligned with target ranking structure
- Improved consistency across all reproducibility outputs

#### Model Family Name Normalization (2025-12-15) – **NEW**
Fixed model family name normalization for capabilities map lookup to ensure consistent family name handling across the system.
- Normalized model family names for reliable capabilities map lookup
- Prevents lookup failures due to case/format inconsistencies

#### Experiment Config Documentation (2025-12-15) – **NEW**
Made experiment configs self-contained with comprehensive documentation and clear structure.
- Experiment configs now include detailed inline documentation
- Self-contained configs reduce need to cross-reference multiple files
- Improved clarity for experiment configuration

#### Symbol-Specific Evaluation Fixes (2025-12-15) – **NEW**
Fixed indentation and evaluation loop issues for symbol-specific target ranking, enabling proper SYMBOL_SPECIFIC evaluation.
- Fixed indentation bug in symbol-specific evaluation loop
- Enabled SYMBOL_SPECIFIC evaluation for classification targets
- Fixed CatBoost importance extraction for symbol-specific runs
- Improved CatBoost verbosity and feature importance snapshot generation

#### CatBoost GPU Fixes (2025-12-15) – **NEW**
Critical fixes for CatBoost GPU mode compatibility and feature importance output.
- Fixed CatBoost GPU requiring Pool objects instead of numpy arrays (automatic conversion via wrapper)
- Fixed sklearn clone compatibility for CatBoost wrapper (implements get_params/set_params)
- Fixed missing CatBoost feature importances in results directory (now saves to catboost_importances.csv)
- CatBoost GPU training now works correctly with cross-validation
→ [Detailed Changelog](DOCS/02_reference/changelog/2025-12-15-catboost-gpu-fixes.md)

#### IP Assignment Agreement Signed (2025-12-14) – **NEW**
IP Assignment Agreement has been signed, legally assigning all intellectual property from Jennifer Lewis (Individual) to Fox ML Infrastructure LLC.
- ✅ **Legally effective** - All IP now owned by Fox ML Infrastructure LLC
- Clean IP ownership structure for enterprise clients and monetization
- Perpetual and irrevocable assignment
- Supporting documentation organized in `LEGAL/ip_assignment_docs/`
→ [Detailed Changelog](DOCS/02_reference/changelog/2025-12-14-ip-assignment-signed.md) | [Signed Agreement](LEGAL/IP_ASSIGNMENT_AGREEMENT_SIGNED.pdf)

#### Execution Modules Added (2025-12-14) – **NEW**
Execution modules (`ALPACA_trading` and `IBKR_trading`) have been added back to the repository with comprehensive compliance framework and documentation organization.
- **ALPACA_trading**: Paper trading and backtesting framework (⚠️ has minor issues, needs testing)
- **IBKR_trading**: Production live trading system for Interactive Brokers (⚠️ untested, requires testing before production use)
- Comprehensive broker integration compliance framework with legal documentation
- 20 documentation files moved to centralized `DOCS/` structure
- 56 Python files updated with consistent copyright headers
- Complete trading modules documentation and cross-linking
→ [Detailed Changelog](DOCS/02_reference/changelog/2025-12-14-execution-modules.md) | [Trading Modules Overview](DOCS/02_reference/trading/TRADING_MODULES.md) | [Broker Compliance](LEGAL/BROKER_INTEGRATION_COMPLIANCE.md)

#### Enhanced Drift Tracking (2025-12-14) – **NEW**
Bulletproof drift tracking with fingerprints, severity tiers, critical metrics, and sanity checks. Can now definitively answer "What changed between baseline and current, and was it data, config, code, or stochasticity?"
- Fingerprints: git commit, config hash, data fingerprint (baseline + current) prove baseline is different
- Drift tiers: OK/WARN/ALERT with configurable thresholds (stricter for critical metrics)
- Critical metrics: Automatically tracks label_window, horizon, leakage flags, cv_scheme_id, etc.
- Sanity checks: Detects self-comparison and suspiciously identical runs
- Parquet files: Queryable long-format data alongside JSON for efficient cross-run analysis
- Zero handling: Explicit `rel_delta_status` for zero baselines (no ambiguous nulls)
→ [Detailed Changelog](DOCS/02_reference/changelog/2025-12-14-drift-tracking-enhancements.md)

#### Telemetry System (2025-12-14)
Sidecar-based telemetry system with view isolation. Telemetry files live alongside existing artifacts in cohort directories, enabling per-target, per-symbol, and per-cross-sectional drift tracking.
- Sidecar files: `telemetry_metrics.json` + `.parquet`, `telemetry_drift.json` + `.parquet`, `telemetry_trend.json` in each cohort folder
- View-level rollups: `CROSS_SECTIONAL/telemetry_rollup.json` + `.parquet`, `SYMBOL_SPECIFIC/telemetry_rollup.json` + `.parquet`
- Stage-level container: `TARGET_RANKING/telemetry_rollup.json` + `.parquet`
- View isolation: CS drift only compares to CS baselines, SS only to SS (baseline key: `stage:view:target[:symbol]`)
- Config-driven: All behavior controlled by `safety.telemetry` section
→ [Detailed Changelog](DOCS/02_reference/changelog/2025-12-14-telemetry-system.md)

#### Feature Selection and Config Fixes (2025-12-14)
Critical bug fixes for feature selection pipeline, experiment config loading, and target exclusion. Resolves cascading failures preventing feature selection from running.
- Fixed UnboundLocalError for `np` (11 model families now working)
- Fixed missing import and unpacking errors in shared harness
- Added honest routing diagnostics with per-symbol skip reasons
- Fixed experiment config loading (`max_targets_to_evaluate`, `top_n_targets`)
- Added target pattern exclusion (`exclude_target_patterns`)
- Fixed `hour_of_day` unknown lookback violation
→ [Detailed Changelog](DOCS/02_reference/changelog/2025-12-14-feature-selection-and-config-fixes.md)

#### Look-Ahead Bias Fixes (2025-12-14) – **NEW**
Comprehensive fixes for data leakage in feature engineering and model training. All fixes behind feature flags (default: OFF) for safe gradual rollout.
- Fix #1: Rolling windows exclude current bar
- Fix #2: CV-based normalization support
- Fix #3: pct_change() verification
- Fix #4: Feature renaming (beta_20d → volatility_20d_returns)
- Additional: Enhanced symbol-specific logging, fixed feature selection bug
→ [Detailed Changelog](DOCS/02_reference/changelog/2025-12-14-lookahead-bias-fixes.md) | [Fix Plan](DOCS/03_technical/fixes/LOOKAHEAD_BIAS_FIX_PLAN.md)

#### SST Enforcement Design Implementation (2025-12-13) – **NEW**
Provably split-brain free system with EnforcedFeatureSet contract, type boundary wiring, and boundary assertions across all training paths.
→ [Detailed Changelog](DOCS/02_reference/changelog/2025-12-13-sst-enforcement-design.md)

#### Single Source of Truth for Lookback Computation (2025-12-13)
Eliminated split-brain in lookback computation with canonical map, POST_PRUNE invariant check, and diagnostic logging.
→ [Detailed Changelog](DOCS/02_reference/changelog/2025-12-13-single-source-of-truth.md)

#### Leakage Controls Structural Fixes (2025-12-13) – **NEW**
Unified leakage budget calculator, calendar feature classification, separate purge/embargo validation, fingerprint tracking system.
→ [Detailed Changelog](DOCS/02_reference/changelog/2025-12-13.md) | [Leakage Validation Fix](DOCS/03_technical/fixes/2025-12-13-leakage-validation-fix.md)

#### Feature Selection Unification & Critical Fixes (2025-12-13) – **NEW**
Shared ranking harness, comprehensive hardening, critical fixes for CatBoost dtype, RFE, linear models, stability tracking.
→ [Detailed Changelog](DOCS/02_reference/changelog/2025-12-13-feature-selection-unification.md)

#### Generalized Duration Parsing System (2025-12-13) – **NEW**
New duration parsing system with interval-aware strictness and fail-closed policy.
→ [Detailed Changelog](DOCS/02_reference/changelog/2025-12-13-duration-system.md)

#### Documentation Organization (2025-12-13) – **NEW**
Index files, config migration docs, audit documentation organization, cross-linking fixes.
→ [Detailed Changelog](DOCS/02_reference/changelog/2025-12-13.md)

#### Config Path Consolidation & Config Trace System (2025-12-13) – **NEW**
Major CONFIG directory restructure, comprehensive config trace logging, max samples fix.
→ [Detailed Changelog](DOCS/02_reference/changelog/2025-12-13.md)

#### License & Commercial Use Banner (2025-12-12) – **NEW**
Professional startup banner with licensing information and 30-day evaluation period notice.
→ [Detailed Changelog](DOCS/02_reference/changelog/2025-12-12.md)

#### GPU Acceleration (2025-12-12) – **NEW**
GPU support for XGBoost, CatBoost, and LightGBM with config-driven settings.
→ [GPU Setup Guide](DOCS/01_tutorials/setup/GPU_SETUP.md) | [Detailed Changelog](DOCS/02_reference/changelog/2025-12-12.md)

#### Experiment Configuration System (2025-12-12) – **NEW**
Reusable experiment configurations with auto target discovery.
→ [Experiment Config Guide](DOCS/01_tutorials/configuration/EXPERIMENT_CONFIG_GUIDE.md)

#### Active Sanitization (Ghost Buster) (2025-12-12) – **NEW**
Proactive feature quarantine system that automatically removes problematic features before training.
→ [Active Sanitization Guide](DOCS/03_technical/implementation/ACTIVE_SANITIZATION.md)

#### Critical Bug Fixes (2025-12-12) – **FIXED**
Mutual Information SST compliance, XGBoost 3.1+ GPU compatibility, CatBoost GPU verification, process deadlock fix.
→ [Detailed Changelog](DOCS/02_reference/changelog/2025-12-12.md)

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

**Recent additions (2025-12-12 - 2025-12-14):**
- Target Pattern Exclusion - Per-experiment control to exclude specific target types (2025-12-14)
- Look-Ahead Bias Fixes - Feature flag-based fixes for data leakage (2025-12-14)
- Shared Ranking Harness - Unified evaluation contract for target ranking and feature selection
- Feature Selection Unification - Complete parity with target ranking
- Experiment Configuration System - Reusable experiment configs with auto target discovery
- Active Sanitization System - Proactive feature quarantine before training
- Dual-View Target Ranking System - Multiple evaluation views with automatic routing
- Config-Based Pipeline Interface - Minimal command-line usage
- Sample Size Binning System - Automatic binning of runs by sample size
- Trend Analysis System - Automated trend tracking across runs
- License & Commercial Use Banner - Professional startup banner
- Training Routing & Planning System - Config-driven routing decisions
- Reproducibility Tracking System - End-to-end tracking with STABLE/DRIFTING/DIVERGED classification

**For complete details:** See [Changelog Index](DOCS/02_reference/changelog/README.md)

---

### Fixed

**Recent fixes (2025-12-12 - 2025-12-14):**
- **Feature Selection and Config Fixes** (2025-12-14): Fixed UnboundLocalError for np (11 model families), missing import, unpacking error, routing diagnostics, experiment config loading, target exclusion, lookback enforcement
- **Look-Ahead Bias Fixes** (2025-12-14): Rolling windows exclude current bar, CV-based normalization, feature renaming, symbol-specific logging, feature selection bug (task_type collision)
- **Leakage Controls Structural Fixes** (2025-12-13): Unified lookback calculator, calendar feature classification, separate purge/embargo validation, fingerprint tracking, leakage canary test
- **Feature Selection Critical Fixes** (2025-12-13): Shared harness unpack crashes, CatBoost dtype mis-typing, RFE/linear model failures, stability cross-model mixing, telemetry scoping
- **Single Source of Truth** (2025-12-13): Eliminated split-brain in lookback computation, POST_PRUNE invariant check, _Xd pattern inference, readline library conflict
- Documentation Link Fixes, Resolved Config System, Reproducibility Tracker Fixes, Critical Horizon Unit Bug, XGBoost 3.1+ GPU Compatibility, CatBoost GPU Verification, Process Deadlock Fix, Config Path Consolidation

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
