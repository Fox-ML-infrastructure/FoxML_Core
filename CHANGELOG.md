# Changelog

All notable changes to FoxML Core will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

**For detailed technical changes:**
- [Changelog Index](DOCS/02_reference/changelog/README.md) – Per-day detailed changelogs with file paths, config keys, and implementation notes.

---

## [Unreleased]

### Recent Highlights

#### 2025-12-16 Updates
- **Diff Telemetry Integration**: Integrated diff telemetry into metadata.json and metrics.json outputs. Full audit trail in metadata (fingerprints, sources, excluded factors), lightweight queryable fields in metrics (flags, counts, summaries). All stages (TARGET_RANKING, FEATURE_SELECTION, TRAINING) now share the same telemetry contract. Backwards compatible with stable shapes for all edge cases.
- **Diff Telemetry Digest Hardening**: Implemented fail-fast assertion for JSON-primitive-only types (removed `default=str` fallback) and full SHA256 hash (64 hex chars, 256 bits) for maximum integrity. Normalization bugs now fail immediately rather than being silently hidden.
- **Feature Selection Structure**: Organized feature selection outputs to match target ranking layout (feature_importances/, metadata/, artifacts/). Eliminated scattered files and nested REPRODUCIBILITY directories.
- **Canonical Family ID System**: Migrated all model family registries to snake_case canonical IDs. All registries (`MODMAP`, `TRAINER_MODULE_MAP`, `POLICY`, `FAMILY_CAPS`) now use consistent snake_case keys (e.g., `"lightgbm"`, `"xgboost"`, `"meta_learning"`). Added startup validation to prevent key drift.
- **Feature Audit System**: Added comprehensive feature drop tracking with per-feature drop reasons. Generates CSV reports showing why features were dropped at each stage (registry filter, Polars select, pandas coercion, NaN drop, non-numeric drop).
- **Training Pipeline Fixes**: Fixed family name canonicalization, banner suppression in child processes, reproducibility tracking string/Enum handling, and model saving packaging bugs.
→ [Detailed Changelogs](DOCS/02_reference/changelog/2025-12-16-diff-telemetry-integration.md) | [Training Pipeline Fixes](DOCS/02_reference/changelog/2025-12-16-training-pipeline-fixes.md) | [Feature Selection Structure](DOCS/02_reference/changelog/2025-12-16-feature-selection-structure.md)

#### 2025-12-15 Updates
- **Metrics System Rename**: Renamed telemetry to metrics throughout codebase for better branding. All metrics stored locally - no user data collection, no external transmission.
- **Seed Tracking Fix**: Fixed missing seed field in metadata.json for full reproducibility tracking.
- **Feature Selection Improvements**: Output structure refactor, model family normalization, experiment config documentation.
- **CatBoost GPU Fixes**: Critical fixes for GPU mode compatibility and feature importance output.
→ [Detailed Changelog](DOCS/02_reference/changelog/2025-12-15-consolidated.md)

#### 2025-12-14 Updates
- **IP Assignment Signed**: All IP legally assigned to Fox ML Infrastructure LLC. ✅ Legally effective.
- **Execution Modules**: ALPACA_trading and IBKR_trading modules added with compliance framework. ⚠️ ALPACA has minor issues, IBKR untested.
- **Enhanced Drift Tracking**: Fingerprints, drift tiers (OK/WARN/ALERT), critical metrics, sanity checks, Parquet files.
→ [Detailed Changelogs](DOCS/02_reference/changelog/README.md#december)

#### 2025-12-14 Updates (continued)
- **Metrics System**: Sidecar-based metrics with view isolation, hierarchical rollups, local-only storage. [Renamed from Telemetry on 2025-12-15]
- **Feature Selection Fixes**: Critical bug fixes for feature selection pipeline, experiment config loading, target exclusion.
- **Look-Ahead Bias Fixes**: Comprehensive data leakage fixes behind feature flags (default: OFF).
→ [Detailed Changelogs](DOCS/02_reference/changelog/README.md#december)

#### 2025-12-13 Updates
- **SST Enforcement Design**: Provably split-brain free system with EnforcedFeatureSet contract.
- **Single Source of Truth**: Eliminated split-brain in lookback computation.
- **Leakage Controls**: Unified leakage budget calculator, fingerprint tracking system.
- **Feature Selection Unification**: Shared ranking harness, comprehensive hardening.
- **Duration System**: Generalized duration parsing with interval-aware strictness.
- **Config Consolidation**: Major CONFIG directory restructure, config trace logging.
→ [Detailed Changelogs](DOCS/02_reference/changelog/README.md#december)

#### 2025-12-12 Updates
- **GPU Acceleration**: GPU support for XGBoost, CatBoost, and LightGBM.
- **Experiment Configuration**: Reusable experiment configs with auto target discovery.
- **Active Sanitization**: Proactive feature quarantine system.
- **License Banner**: Professional startup banner with licensing information.
- **Critical Bug Fixes**: Mutual Information SST compliance, XGBoost 3.1+ GPU compatibility.
→ [Detailed Changelogs](DOCS/02_reference/changelog/README.md#december)

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

**Recent additions (2025-12-12 - 2025-12-16):**
- Feature Audit System - Per-feature drop tracking with CSV reports showing drop reasons (2025-12-16)
- Canonical Family ID System - Unified snake_case family IDs across all registries with startup validation (2025-12-16)
- Registry Validation - Startup assertions to prevent non-canonical keys and collisions (2025-12-16)
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

**Recent fixes (2025-12-12 - 2025-12-16):**
- **Training Pipeline Plumbing Fixes** (2025-12-16): Fixed family name canonicalization mismatches, banner suppression in child processes, reproducibility tracking string/Enum handling, model saving packaging bugs (`_pkg_ver`, `joblib` imports), feature selector vs trainer confusion
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
