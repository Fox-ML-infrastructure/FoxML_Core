# Changelog

All notable changes to FoxML Core will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

**For detailed technical changes:**
- [Changelog Index](DOCS/02_reference/changelog/README.md) – Per-day detailed changelogs with file paths, config keys, and implementation notes.

---

## [Unreleased]

### Highlights

- **Dual-View Target Ranking** (2025-12-12) – **NEW**: Target ranking now supports both cross-sectional and symbol-specific evaluation views:
  - **CROSS_SECTIONAL view**: Pooled cross-sectional samples (existing behavior)
  - **SYMBOL_SPECIFIC view**: Evaluate each target separately on each symbol's own time series
  - **LOSO view** (optional): Leave-One-Symbol-Out evaluation for generalization testing
  - **Routing decisions**: Automatic routing logic determines which view(s) to use per target (CROSS_SECTIONAL, SYMBOL_SPECIFIC, BOTH, or BLOCKED)
  - **View consistency**: Target ranking → feature ranking → training maintains the same view for consistency
  - **Reproducibility integration**: Full integration with existing reproducibility suite, storing view/symbol metadata in cohort directories
  - **Backward compatible**: Default behavior unchanged (CROSS_SECTIONAL only), existing code works without changes
  - See [Dual-View Target Ranking Guide](DOCS/03_technical/implementation/DUAL_VIEW_TARGET_RANKING.md)

- **Simplified Config-Based Pipeline** (2025-12-12) – **NEW**: Intelligent training pipeline now supports minimal command-line usage via configuration files:
  - **Config-driven defaults**: All settings (data, targets, features, model families) can be configured in `CONFIG/training_config/intelligent_training_config.yaml`
  - **Simple commands**: Run full pipeline with just `--output-dir` argument
  - **Quick test mode**: `--quick` flag for fast iteration (3 targets, 50 features)
  - **CLI overrides**: Command-line arguments still override config when needed
  - **Backward compatible**: Old command-line syntax still works
  - See [Simple Pipeline Usage Guide](DOCS/01_tutorials/SIMPLE_PIPELINE_USAGE.md)

- **Resolved Config System & Consistent Logging** (2025-12-12) – **NEW**: Centralized configuration resolution eliminates contradictory log messages:
  - **ResolvedConfig object**: Single source of truth for all resolved values (requested vs effective min_cs, purge/embargo, feature counts)
  - **Centralized purge/embargo derivation**: Fixed bug where purge was incorrectly calculated as `max(horizon, feature_lookback) + buffer` (causing 1465m purge for 60m horizon). Now correctly uses `horizon + buffer` only
  - **Consistent logging**: Single authoritative log line per category showing requested → effective values with explicit reasons
  - **Feature count chain**: Logs now show complete chain: `safe=307 → drop_all_nan=3 → final=304` instead of scattered messages
  - **Reproducibility integration**: ResolvedConfig values stored in metadata.json ensuring CV splitter and reproducibility tracker use same purge/embargo values
  - See [Resolved Config Fix Guide](DOCS/03_technical/implementation/RESOLVED_CONFIG_FIX.md)
- **Reproducibility Tracking Bug Fixes** (2025-12-12) – **FIXED**: Critical bugs in reproducibility tracking system:
  - **Fixed `ctx` NameError**: Resolved `NameError: name 'ctx' is not defined` that prevented `metadata.json` and `metrics.json` from being written (86 failures). Metadata files now write correctly for all runs.
  - **Fixed feature importances path**: Feature importances now saved under correct view directory (`target_rankings/feature_importances/{target}/{view}/{symbol?}/`) instead of always using `CROSS_SECTIONAL`.
  - **Fixed perfect CV detection**: Changed to use actual validation CV scores instead of training scores, eliminating false positives (was triggering on 0.9999 training score when actual CV was 0.687).
  - **Added metadata diagnostic**: Warns when metadata files are missing but `audit_report.json` exists, detecting partial writes from previous bugs.

- **Trend Analysis System** (2025-12-12) – **NEW**: Automated trend analysis across all pipeline stages with exponential decay weighting and regression detection:
  - **Trend tracking**: Automatically computes performance trends (slope, EWMA) across runs within comparable series using exponential decay weighting
  - **Multi-stage support**: Trend analysis now integrated into target ranking, feature selection (single symbol + aggregated), and cross-sectional feature ranking
  - **Audit-grade metadata**: Trend metadata (slope, current estimate, alerts) stored in `metadata.json` for post-hoc verification
  - **Explicit skip logging**: All skip conditions log explicit reasons (insufficient_runs, missing_metric, no_variance) - no silent no-ops
  - **Series key verification**: Series grouping uses stable identity fields only (cohort_id, stage, target, data_fingerprint) - no poisoned fields like run_id or timestamps
  - **Verification tools**: `verify_trend_analyzer.py` script provides comprehensive verification checklist
  - See [Trend Analyzer Verification Guide](DOCS/03_technical/implementation/TREND_ANALYZER_VERIFICATION.md)
- **Cohort-Aware Reproducibility & RESULTS Organization** (2025-12-12) – **NEW**: Complete overhaul of reproducibility tracking and output organization:
  - **Cohort-aware reproducibility**: Runs organized by data cohort (sample size, symbols, date range, config) with sample-adjusted drift detection. Only compares runs within the same cohort for statistically meaningful comparisons.
  - **RESULTS directory structure**: All runs (test and production) organized in `RESULTS/` directory by sample size bins for easy comparison: `RESULTS/sample_25k-50k/{run_name}/`
  - **Sample size binning**: Runs grouped into bins (0-5k, 5k-10k, 10k-25k, 25k-50k, 50k-100k, 100k-250k, 250k-500k, 500k-1M, 1M+) to enable easy comparison of runs with similar cross-sectional sample sizes
  - **Audit-grade binning**: Bin boundaries are unambiguous (EXCLUSIVE upper bounds), versioned (`sample_bin_v1`), and stored in `metadata.json` (bin_name, bin_min, bin_max, binning_scheme_version)
  - **Early N_effective estimation**: Automatically estimates sample size from data files or existing metadata during initialization to avoid `_pending/` directories
  - **Integrated backups**: Config backups now stored in run directory (`RESULTS/{bin_name}/{run_name}/backups/`) instead of `CONFIG/backups/`, keeping everything together
  - **Enhanced metadata**: `metadata.json` now includes full `symbols` list (sorted, deduplicated) and bin metadata for debugging and cohort identification
  - **Unified metadata extractor**: Centralized `cohort_metadata_extractor.py` utility used across all modules (target ranking, feature selection, training) for consistent cohort identification
  - **Immediate writes**: All reproducibility files flushed to disk immediately with `fsync()` for real-time visibility
  - See [Cohort-Aware Reproducibility Guide](DOCS/03_technical/implementation/COHORT_AWARE_REPRODUCIBILITY.md)
- **Training Routing System** (2025-12-11) – Config-driven routing system that makes reproducible decisions about where to train models (cross-sectional vs symbol-specific vs both vs experimental vs blocked) based on feature selection metrics, stability analysis, and leakage detection. Generates routing plans and training plans with automatic filtering. **NEW**: 2-stage training pipeline (CPU models first, then GPU models) and one-command end-to-end flow (target ranking → feature selection → training plan → training execution). See [Training Routing Guide](DOCS/02_reference/training_routing/README.md). **Status: Currently being tested.**
- **Reproducibility & Drift Tracking** (2025-12-11) – End-to-end reproducibility tracking across ranking, feature selection, and training with per-model metrics and three-tier classification (**STABLE / DRIFTING / DIVERGED**). Module-specific logs and cross-run comparison. See [Reproducibility Tracking Guide](DOCS/03_technical/implementation/REPRODUCIBILITY_TRACKING.md).
- **Model Parameter Sanitization** (2025-12-11) – Shared `config_cleaner.py` utility using `inspect.signature()` to strip unknown/duplicate params and normalize tricky cases (MLPRegressor `verbose`, CatBoost iteration synonyms and RNG params, sklearn/univariate quirks). Eliminates an entire class of "got multiple values" / "unexpected keyword" failures.
- **Interval Detection & Time Handling** (2025-12-11) – Median-based gap filtering for overnight/weekend gaps, `interval_detection.mode=fixed` for known bar intervals, and noise reduction (INFO instead of WARNING). Removes spurious "unclear interval" warnings on clean 5m data.
- **Leakage Detection & Auto-Fix Reliability** (2025-12-11) – Critical fix to detection confidence (no longer tied to raw importance), on-the-fly importance computation when missing, richer diagnostics, and hardened auto-fixer logging + backup behavior.
- **Single Source of Truth & Determinism** (2025-12-10) – Full config centralization for TRAINING; SST enforcement test; all hyperparameters and seeds now load from YAML, with centralized determinism system. 30+ hardcoded `random_state` and other defaults removed.
- **Architecture & Docs** (2025-12-09+) – Large monolithic training scripts refactored into modular components, intelligent training framework Phase 1 completed, Leakage Safety Suite wired end-to-end, and docs reorganized into a 4-tier hierarchy with expanded legal/commercial material.

---

### Stability Guarantees

- **Training results reproducible** across runs and hardware (centralized seeds + config-only hyperparameters).
- **Complete config centralization** – Pipeline behavior is controlled by YAML (Single Source of Truth).
- **SST enforcement** – Automated test prevents accidental reintroduction of hardcoded hyperparameters.
- **Config schema backward compatible** – Existing configs continue to work with deprecation warnings where applicable.
- **Modular architecture** – TRAINING module is self-contained with zero external script dependencies.

### Known Issues & Limitations

- **Trading / execution modules** are out of scope for the core repo; FoxML Core focuses on ML research & infra.
- **Feature engineering** still requires human review and domain validation.
- **Full end-to-end test suite** is being expanded following SST + determinism changes (as of 2025-12-10).
- **LOSO CV splitter**: LOSO view currently uses combined data; dedicated CV splitter for train-on-all-but-one, validate-on-one is a future enhancement.
- **Placebo test per symbol**: Symbol-specific strong targets should be validated with placebo tests (shuffle labels, assert AUC ~ 0.5) - future enhancement.

---

### Added

- **Dual-View Target Ranking System** (2025-12-12)
  - `TargetRankingView` enum (CROSS_SECTIONAL, SYMBOL_SPECIFIC, LOSO) for view specification
  - Extended `evaluate_target_predictability()` to accept `view` and `symbol` parameters
  - Symbol-specific evaluation loop in `rank_targets()` for per-symbol target ranking
  - Routing decision logic (`_compute_target_routing_decisions()`) with deterministic rules
  - Routing decisions saved to `routing_decisions.json` with summary statistics
  - `load_routing_decisions()` utility for loading routing decisions
  - Feature selection respects view/symbol from target ranking for consistency
  - Reproducibility tracker updated to handle view/symbol metadata
  - Directory structure: `TARGET_RANKING/{view}/{target}/symbol={symbol}/cohort={cohort_id}/`
  - `RunContext` extended with `view` field for target ranking
  - Configuration support in `target_ranking_config.yaml` for enabling/disabling views and routing thresholds

- **Config-Based Pipeline Interface** (2025-12-12)
  - New config file: `CONFIG/training_config/intelligent_training_config.yaml`
  - Config sections: `data`, `targets`, `features`, `model_families`, `output`, `cache`, `advanced`, `test`
  - Automatic config loading with CLI override support
  - `--quick` flag for fast test mode (overrides config with test-friendly defaults)
  - `--full` flag for explicit production mode
  - Priority order: CLI > config > defaults
  - Test mode auto-detection (when 'test' in output_dir name)
  - Documentation: `DOCS/01_tutorials/SIMPLE_PIPELINE_USAGE.md`

- **Sample Size Binning System** (2025-12-12)
  - Automatic binning of runs by sample size into readable ranges (e.g., `sample_25k-50k`) for easy comparison
  - Bins: 0-5k, 5k-10k, 10k-25k, 25k-50k, 50k-100k, 100k-250k, 250k-500k, 500k-1M, 1M+
  - Bin metadata stored in `metadata.json`: `bin_name`, `bin_min`, `bin_max`, `binning_scheme_version`
  - Unambiguous boundary rules (EXCLUSIVE upper bounds: `bin_min <= N_effective < bin_max`)
  - Versioned binning scheme (`sample_bin_v1`) for backward compatibility
  - Early N_effective estimation from data files or existing metadata to avoid `_pending/` directories
  - Bin is for directory organization only; trend series keys use stable identity (cohort_id, stage, target) - no bin_name

- **Trend Analysis System**
  - `TrendAnalyzer` class for analyzing performance trends across runs within comparable series
  - Exponential decay weighting (configurable half-life, default: 7 days) for time-weighted regression
  - Two series views: `STRICT` (all comparability keys must match) and `PROGRESS` (allows feature hash changes, marks breakpoints)
  - Automatic trend computation integrated into:
    - Target ranking (`TRAINING/ranking/predictability/model_evaluation.py`)
    - Feature selection - single symbol + aggregated (`TRAINING/ranking/feature_selector.py`)
    - Cross-sectional feature ranking (`TRAINING/ranking/cross_sectional_feature_ranker.py`)
  - Trend metadata stored in `metadata.json` with slope, EWMA, alerts, and status
  - `TREND_REPORT.json` artifact with comprehensive trend analysis across all series
  - `verify_trend_analyzer.py` verification script for validation and debugging
  - Explicit skip logging with reasons (insufficient_runs, missing_metric, no_variance)
  - Artifact index corruption handling (automatic rebuild on read error)

- **Training Routing & Planning System**
  - Config-driven routing decisions for each `(target, symbol)` pair based on metrics (scores, stability, leakage, sample sizes).
  - Routing states: `ROUTE_CROSS_SECTIONAL`, `ROUTE_SYMBOL_SPECIFIC`, `ROUTE_BOTH`, `ROUTE_EXPERIMENTAL_ONLY`, `ROUTE_BLOCKED`.
  - Automatic routing plan generation after feature selection with JSON/YAML/Markdown outputs.
  - Training plan generator that converts routing decisions into actionable training jobs with priorities and model families.
  - **Automatic integration**: Training phase automatically filters targets based on training plan (CS training filtering implemented).
  - Metrics aggregator collects metrics from feature selection, stability snapshots, and leakage detection.
  - Full documentation in `DOCS/02_reference/training_routing/`.
  - **Note**: Currently filters cross-sectional training targets; symbol-specific training execution is a future enhancement.
  - **Dual-view integration**: Training routing now respects dual-view target ranking decisions (CROSS_SECTIONAL vs SYMBOL_SPECIFIC).

- **Reproducibility Tracking System**
  - Reusable `ReproducibilityTracker` with tolerance-based comparisons and STABLE/DRIFTING/DIVERGED classification.
  - Per-model reproducibility metrics in feature selection (delta_score, Jaccard@K, importance_corr) stored in `model_metadata.json`.
  - **Cross-sectional stability tracking**: Factor robustness analysis for cross-sectional feature selection. Tracks top-K overlap and Kendall tau across runs with STABLE/DRIFTING/DIVERGED classification. Stricter thresholds (overlap ≥0.75, tau ≥0.65) than per-symbol since global factors should be more persistent. Stores snapshots and metadata for institutional-grade factor analysis.

- **Config & Safety Utilities**
  - `TRAINING/utils/config_cleaner.py` – shared parameter sanitization / validation for all model constructors.
  - Config sections for intelligent training, safety, and logging (e.g. `intelligent_training`, `safety.reproducibility`, logging profiles).
  - Target confidence & routing system (core / candidate / experimental) with configurable thresholds.
  - Production-grade Leakage Safety Suite: backup system, auto-fixer, sentinels, feature/target schema, safety configs.

- **Configuration & Observability**
  - Modular configuration system with typed schemas, experiment configs, and validation.
  - Structured logging config (`logging_config.yaml`) with profiles (default, debug_run, quiet) and backend verbosity control.

---

### Fixed

- **Resolved Config System & Consistent Logging** (2025-12-12) – **NEW**: Centralized configuration resolution and consistent logging:
  - **ResolvedConfig object**: Single source of truth for resolved values (requested vs effective min_cs, purge/embargo, feature counts)
  - **Centralized purge/embargo derivation**: Single `derive_purge_embargo()` function used everywhere (was causing 1465m purge instead of 85m for 60m horizon targets)
  - **Consistent logging**: Single authoritative log line per category showing requested → effective values with reasons
  - **Feature count chain**: Logs now show `safe=307 → drop_all_nan=3 → final=304` instead of scattered messages
  - **Purge calculation fix**: Removed incorrect `max(horizon, feature_lookback)` formula - purge now correctly uses `horizon + buffer` only
  - **Reproducibility integration**: ResolvedConfig values stored in metadata.json for audit trail
  - See [Resolved Config Fix Guide](DOCS/03_technical/implementation/RESOLVED_CONFIG_FIX.md)

- **Reproducibility Tracker Fixes** (2025-12-12):
  - Fixed `log_comparison()` signature to accept `route_type`, `symbol`, and `model_family` parameters
  - Fixed parameter extraction to prioritize explicit parameters over `additional_data`
  - Ensured view/symbol metadata correctly passed through to reproducibility tracking
  - Fixed cohort directory creation to use view as route_type for TARGET_RANKING stage

- **Backup Creation Fix** (2025-12-12):
  - Fixed backup creation when `backup_configs=False` - backups now only created when explicitly enabled
  - Added safety check in `_backup_configs()` to return early if backups disabled
  - Fixed direct call to `_backup_configs()` in `model_evaluation.py` to check `backup_configs` flag first

### Fixed

- **Time & Interval Handling**
  - Median-based gap filtering and fixed-interval mode to eliminate false "unclear interval" warnings and negative delta issues.

- **Parameter Passing & Validation**
  - Systematic fix for duplicated / incompatible params across all model families (MLPRegressor, CatBoost, RandomForest, Lasso, etc.).
  - Resolved CatBoost iteration synonyms (`iterations` vs `n_estimators`/`num_trees`) and RNG param conflicts (`random_seed` vs `random_state`).

- **Leakage Detection & Auto-Fixer**
  - Critical fix to confidence calculation (no longer using raw importance as confidence).
  - On-the-fly importance computation when upstream importances are missing.
  - Fixed auto-fixer logging format errors and improved visibility into detection inputs and decisions.

- **Reproducibility & Sampling**
  - Corrected reproducibility log discovery across timestamped output directories and added module-specific log locations.
  - Fixed `max_cs_samples` filtering so cross-sectional sampling respects config and avoids unnecessary memory blowups.

- **Config & Code Quality**
  - Hardened config loading (`inject_defaults`, YAML `None` handling) and eliminated silent fallback paths with new logging.
  - Resolved missing imports / F821 errors and minor syntax issues uncovered by linting.

---

### Changed

- **Determinism & Defaults**
  - Removed hardcoded `random_state` and similar defaults; all now provided via the central determinism system and config.
  - Cleaned duplicated default values; shared defaults now injected from a single location.

- **Logging**
  - Replaced scattered logging flags with structured YAML-driven configuration.
  - Reduced unnecessary WARNING noise in interval detection and reproducibility tracking; meaningful issues remain surfaced.

- **Docs & Legal**
  - Documentation restructured into a 4-tier hierarchy with better navigation and cross-linking.
  - Legal/commercial docs expanded and updated (compliance, license enforcement, pricing, corporate details).

---

### Security

- Enhanced compliance and production-use documentation.
- Documented license enforcement procedures and copyright notice requirements.

### Documentation

- 4-tier documentation hierarchy established.
- 50+ existing docs rewritten; 50+ new docs added.
- Cross-linking and indices improved for discoverability.

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
