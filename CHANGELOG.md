# Changelog

All notable changes to FoxML Core will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

**For more detailed information:**
- [Detailed Changelog](DOCS/02_reference/CHANGELOG_DETAILED.md) - Comprehensive change details with file paths and config references

---

## [Unreleased]

### Highlights

- Phase 1 intelligent training framework completed and functioning properly
- Ranking & selection pipelines unified (interval handling, preprocessing, CatBoost)
- Boruta refactored into a statistical gatekeeper with base vs final consensus tracking
- **Target confidence & routing system** — Automatic quality assessment with configurable thresholds and operational buckets (core/candidate/experimental)
- New modular configuration + structured logging system
- Leakage Safety Suite hardened (auto-fixer + backup system + schema/registry)
- Documentation and legal reorganized into a 4-tier, enterprise-ready docs hierarchy
- Commercial license & pricing updated for enterprise quant infrastructure

---

### Stability Guarantees

- **Training results reproducible** across hardware (deterministic seeds, config-driven hyperparameters)
- **Config schema backward compatible** (existing configs continue to work)
- **Auto-fixer non-destructive by design** (atomic backups, manifest tracking, restore capabilities)
- **Leakage detection thresholds configurable** (no hardcoded magic numbers)
- **Modular architecture** (self-contained TRAINING module, zero external script dependencies)

### Known Issues & Limitations

- **Trading execution modules** (IBKR/Alpaca live trading) are not currently operational and require additional development
- **Feature engineering** still requires human review and validation (initial feature set was for testing)
- **Adaptive intelligence layer** in early phase (leakage detection and auto-fixer are production-ready, but adaptive learning over time is planned)
- **Ranking pipeline** may occasionally log false-positive leakage warnings for tree models (RF overfitting detection is conservative by design)
- **Later phases of the experiments workflow** (core models and sequential models) require implementation beyond Phase 1

---

### Added

#### Intelligent Training & Ranking

- **Unified ranking and selection pipelines** — Consistent interval handling, sklearn preprocessing, and CatBoost configuration across both pipelines. Shared target utilities and preprocessing helpers ensure identical behavior. See [`DOCS/01_tutorials/training/RANKING_SELECTION_CONSISTENCY.md`](DOCS/01_tutorials/training/RANKING_SELECTION_CONSISTENCY.md).
- **Boruta statistical gatekeeper** — Refactored from importance scorer to gatekeeper that modifies consensus scores via bonuses/penalties. Tracks base vs final consensus with explicit gate effect. ExtraTrees-based implementation with configurable thresholds. See [`CONFIG/feature_selection/multi_model.yaml`](CONFIG/feature_selection/multi_model.yaml).
- **Target confidence & routing system** — Automatic quality assessment for each target with configurable thresholds. Computes Boruta coverage, model coverage, score strength, and agreement ratio. Routes targets into operational buckets (core/candidate/experimental) based on confidence + score_tier. All thresholds and routing rules configurable via YAML. See [`CONFIG/feature_selection/multi_model.yaml`](CONFIG/feature_selection/multi_model.yaml) `confidence` section and [`DOCS/01_tutorials/training/INTELLIGENT_TRAINING_TUTORIAL.md`](DOCS/01_tutorials/training/INTELLIGENT_TRAINING_TUTORIAL.md#target-confidence-and-routing).
- **LightGBM GPU support** in target ranking with automatic detection and CPU fallback
- **TRAINING module self-contained** — All utilities migrated from `SCRIPTS/` to `TRAINING/utils/`, zero external dependencies

#### Configuration & Logging

- **Modular configuration system** with typed schemas, experiment configs, and validation. Single YAML file defines data, targets, and module overrides. Backward compatible with legacy configs. See [`DOCS/02_reference/configuration/MODULAR_CONFIG_SYSTEM.md`](DOCS/02_reference/configuration/MODULAR_CONFIG_SYSTEM.md).
- **Structured logging configuration** — Per-module and backend verbosity controls via YAML. Profile support (default, debug_run, quiet). No hardcoded logging flags. See [`CONFIG/logging_config.yaml`](CONFIG/logging_config.yaml).
- **Centralized training configs** — 9 YAML files for pipeline, GPU, memory, preprocessing, threading, safety, callbacks, optimizer, and system settings

#### Leakage Safety Suite

- **Production-grade backup system for auto-fixer** — Per-target, timestamped backups with manifests and git commit provenance. Atomic restore operations with retention policy and detailed error handling. Config-driven settings documented in safety config. Full behavior documented in [`DOCS/03_technical/research/LEAKAGE_ANALYSIS.md`](DOCS/03_technical/research/LEAKAGE_ANALYSIS.md).
- **Automated leakage detection and auto-fix** — Automatic detection and remediation with configurable thresholds. Pre-training leak scan, auto-rerun after fixes, and integration with leakage sentinels. See [`DOCS/02_reference/configuration/SAFETY_LEAKAGE_CONFIGS.md`](DOCS/02_reference/configuration/SAFETY_LEAKAGE_CONFIGS.md).
- **Feature/target schema system** — Explicit schema for classifying columns with mode-specific rules (ranking vs training). Feature registry with temporal metadata filtering. See [`CONFIG/feature_target_schema.yaml`](CONFIG/feature_target_schema.yaml) and [`CONFIG/feature_registry.yaml`](CONFIG/feature_registry.yaml).

#### GPU & Training Infra

- Base trainer scaffolding for 2D and 3D models
- Sequential models 3D preprocessing fixes
- XGBoost source-build stability improvements
- TensorFlow GPU initialization fixes

#### Docs & Legal

- **Documentation restructured** — 4-tier hierarchy with centralized docs in `DOCS/`. Code directories contain only code and minimal README pointers. See [`DOCS/INDEX.md`](DOCS/INDEX.md).
- **Roadmap restructured** — Added "What Works Today" section, reorganized priorities, refined wording for external consumption. See [`ROADMAP.md`](ROADMAP.md).
- **Legal documentation suite** — Compliance docs, license enforcement procedures, commercial use guides. See [`DOCS/LEGAL_INDEX.md`](DOCS/LEGAL_INDEX.md).
- **Legal documentation updates** — Enhanced decision matrix, FAQ, and subscription documentation for clarity and completeness. See [`LEGAL/DECISION_MATRIX.md`](LEGAL/DECISION_MATRIX.md), [`LEGAL/FAQ.md`](LEGAL/FAQ.md), [`LEGAL/SUBSCRIPTIONS.md`](LEGAL/SUBSCRIPTIONS.md).
- 55+ new documentation files created, 50+ existing files rewritten and standardized

#### Commercial

- **Commercial license pricing** updated to enterprise quant infrastructure standards (see [`COMMERCIAL_LICENSE.md`](COMMERCIAL_LICENSE.md) for full pricing tiers)
- Optional enterprise add-ons (dedicated support, integration, onboarding, private access)
- Enhanced copyright headers across codebase

---

### Changed

- **Logging system refactored** — Replaced hardcoded flags with structured configuration. All verbosity controlled via YAML without code changes. See [`CONFIG/logging_config.yaml`](CONFIG/logging_config.yaml).
- **Leakage filtering supports ranking mode** — Permissive rules for ranking, strict rules for training. Prevents false positives from overfitting detection.
- All model trainers updated to use centralized configs (preprocessing, callbacks, optimizers, safety guards)
- Pipeline, threading, memory, GPU, and system settings integrated into centralized config system

---

### Fixed

- **Feature selection pipeline** — Boruta `X_clean` error, double-counting, feature count mismatches. Interval detection warnings, CatBoost loss function for classification, sklearn NaN/dtype handling. See [`DOCS/02_reference/CHANGELOG_DETAILED.md`](DOCS/02_reference/CHANGELOG_DETAILED.md) for detailed notes.
- **Interval detection** — Fixed negative delta warnings from unsorted timestamps or wraparound. Now uses `abs()` on time deltas before unit detection and conversion. Prevents spurious warnings like "Timestamp delta -789300000000000.0 doesn't map to reasonable interval". See [`TRAINING/utils/data_interval.py`](TRAINING/utils/data_interval.py) and [`TRAINING/ranking/rank_target_predictability.py`](TRAINING/ranking/rank_target_predictability.py).
- **Path resolution** — Fixed inconsistent repo root calculations across moved files
- **Auto-fixer** — Import paths, training accuracy detection, pre-excluded feature checks
- **GPU and model issues** — VAE serialization, sequential models 3D preprocessing, XGBoost stability, TensorFlow GPU initialization, LSTM timeouts, Transformer OOM errors
- **Progress logging** — Fixed denominator when using `--max-targets-to-evaluate`

---

### Security

- Enhanced compliance documentation for production use
- License enforcement procedures documented
- Copyright notice requirements standardized

---

### Documentation

- Modular configuration system documentation (see [`DOCS/02_reference/configuration/MODULAR_CONFIG_SYSTEM.md`](DOCS/02_reference/configuration/MODULAR_CONFIG_SYSTEM.md))
- Documentation cleanup and consolidation — Integrated old folders, fixed cross-references, reorganized structure
- Comprehensive cross-linking and navigation improvements

---

## Future Work

### Adaptive Intelligence Layer (Planned)

The current intelligence layer provides automated target ranking, feature selection, leakage detection, and auto-fixing. Future enhancements will include:

- **Adaptive learning over time**: System learns from historical leakage patterns and feature performance
- **Dynamic threshold adjustment**: Automatically tunes detection thresholds based on observed patterns
- **Predictive leakage prevention**: Proactively flags potential leakage before training begins
- **Multi-target optimization**: Optimizes feature selection across multiple targets simultaneously

Adaptive intelligence layer design is documented in planning materials.

---

## Versioning

Releases follow [Semantic Versioning](https://semver.org/):
- **MAJOR** version for incompatible API changes
- **MINOR** version for functionality added in a backwards-compatible manner
- **PATCH** version for backwards-compatible bug fixes

---

## Categories

- **Added** – New features
- **Changed** – Changes in existing functionality
- **Commercial** – Business/pricing/licensing changes
- **Deprecated** – Soon-to-be removed features
- **Removed** – Removed features
- **Fixed** – Bug fixes
- **Security** – Security improvements
- **Documentation** – Documentation changes
