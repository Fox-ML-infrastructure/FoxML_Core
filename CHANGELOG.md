# Changelog

All notable changes to FoxML Core will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

**For detailed technical changes:**
- [Changelog Index](DOCS/02_reference/changelog/README.md) – Per-day detailed changelogs with file paths, config keys, and implementation notes.

---

## [Unreleased]

### Highlights

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

---

### Added

- **Training Routing & Planning System**
  - Config-driven routing decisions for each `(target, symbol)` pair based on metrics (scores, stability, leakage, sample sizes).
  - Routing states: `ROUTE_CROSS_SECTIONAL`, `ROUTE_SYMBOL_SPECIFIC`, `ROUTE_BOTH`, `ROUTE_EXPERIMENTAL_ONLY`, `ROUTE_BLOCKED`.
  - Automatic routing plan generation after feature selection with JSON/YAML/Markdown outputs.
  - Training plan generator that converts routing decisions into actionable training jobs with priorities and model families.
  - **Automatic integration**: Training phase automatically filters targets based on training plan (CS training filtering implemented).
  - Metrics aggregator collects metrics from feature selection, stability snapshots, and leakage detection.
  - Full documentation in `DOCS/02_reference/training_routing/`.
  - **Note**: Currently filters cross-sectional training targets; symbol-specific training execution is a future enhancement.

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
