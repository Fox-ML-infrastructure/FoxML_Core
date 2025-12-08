# Changelog

All notable changes to FoxML Core will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

**Status**: Testing in progress - Reproducibility validation underway

**Note**: Backward functionality remains fully operational. The ranking and intelligent training pipeline is currently being tested for reproducibility. All existing training workflows continue to function as before.

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
- **Phase 2-3 of experiments workflow** (core models and sequential models) require implementation beyond Phase 1

**TL;DR**:
- **New**: Automated leakage detection + auto-fixer with production-grade backup system
- **New**: Centralized safety configs and feature/target schema system
- **New**: LightGBM GPU support in ranking + TRAINING module now self-contained
- **New**: Full compliance documentation suite + commercial pricing update

### Added

#### **Leakage Safety Suite**
- **Production-grade backup system for auto-fixer**:
  - Per-target timestamped backup structure: `CONFIG/backups/{target}/{timestamp}/files + manifest.json`
  - Automatic retention policy: Keeps last N backups per target (configurable, default: 20)
  - High-resolution timestamps: Uses microseconds to avoid collisions in concurrent scenarios
  - Manifest files with full provenance: Includes backup_version, source, target_name, timestamp, git_commit, file paths
  - Atomic restore operations: Writes to temp file first, then atomic rename (prevents partial writes)
  - Enhanced error handling: Lists available timestamps on unknown timestamp, validates manifest structure
  - Comprehensive observability: Logs backup creation, pruning, and restore operations with full context
  - Config-driven settings: `max_backups_per_target` configurable via `system_config.yaml` (default: 20, 0 = no limit)
  - Restoration helpers: `list_backups()` and `restore_backup()` static methods for backup management
  - Backward compatible: Legacy flat structure still supported (with warning) when no target_name provided
  - Git commit tracking: Captures git commit hash in manifest for debugging and provenance
- **Automated leakage detection and auto-fix system**:
  - `LeakageAutoFixer` class for automatic detection and remediation of data leakage
  - Integration with leakage sentinels (shifted-target, symbol-holdout, randomized-time tests)
  - Automatic config file updates (`excluded_features.yaml`, `feature_registry.yaml`)
  - Auto-fixer triggers automatically when perfect scores (≥0.99) are detected during target ranking
  - **Checks against pre-excluded features**: Filters out already-excluded features before detection to avoid redundant work
  - **Configurable auto-fixer thresholds** in `safety_config.yaml`:
    - CV score threshold (default: 0.99)
    - Training accuracy threshold (default: 0.999)
    - Training R² threshold (default: 0.999)
    - Perfect correlation threshold (default: 0.999)
    - Minimum confidence for auto-fix (default: 0.8)
    - Maximum features to fix per run (default: 20) - prevents overly aggressive fixes
    - Enable/disable auto-fixer flag
  - **Auto-rerun after leakage fixes**:
    - Automatic rerun of target evaluation after auto-fixer modifies configs
    - Configurable via `safety_config.yaml` (`auto_rerun` section):
      - `enabled`: Enable/disable auto-rerun (default: `true`)
      - `max_reruns`: Maximum reruns per target (default: `3`)
      - `rerun_on_perfect_train_acc`: Rerun on perfect training accuracy (default: `true`)
      - `rerun_on_high_auc_only`: Rerun on high AUC alone (default: `false`)
    - Stops automatically when no leakage detected or no config changes
    - Tracks attempt count and final status (`OK`, `SUSPICIOUS_STRONG`, `LEAKAGE_UNRESOLVED`, etc.)
  - **Pre-training leak scan**:
    - Detects near-copy features before model training (catches obvious leaks early)
    - Binary classification: detects features matching target with ≥99.9% accuracy
    - Regression: detects features with ≥99.9% correlation with target
    - Automatically removes leaky features before model training
    - Configurable thresholds in `safety_config.yaml` (min_match, min_corr)
  - **Feature/Target Schema** (`CONFIG/feature_target_schema.yaml`):
    - Explicit schema for classifying columns (metadata, targets, features)
    - Feature families with mode-specific rules (ranking vs. training)
    - Ranking mode: more permissive (allows basic OHLCV/TA features)
    - Training mode: strict rules (enforces all leakage filters)
  - **Configurable leakage detection thresholds**:
    - All hardcoded thresholds moved to `CONFIG/training_config/safety_config.yaml`
    - Pre-scan thresholds (min_match, min_corr, min_valid_pairs)
    - Ranking feature requirements (min_features_required, min_features_for_model)
    - Warning thresholds (classification, regression with forward_return/barrier variants)
    - Model alert thresholds (suspicious_score)
  - **Feature registry system** (`CONFIG/feature_registry.yaml`):
    - Structural rules based on temporal metadata (`lag_bars`, `allowed_horizons`, `source`)
    - Automatic filtering based on target horizon to prevent leakage
    - Support for short-horizon targets (added horizon=2 for 10-minute targets)
  - **Leakage sentinels** (`TRAINING/common/leakage_sentinels.py`):
    - Shifted target test – detects features encoding future information
    - Symbol holdout test – detects symbol-specific leakage
    - Randomized time test – detects temporal information leakage
  - **Feature importance diff detector** (`TRAINING/common/importance_diff_detector.py`):
    - Compares feature importances between full vs. safe feature sets
    - Identifies suspicious features with high importance in full model but low in safe model
- **LightGBM GPU support** in target ranking:
  - Automatic GPU detection and usage (CUDA/OpenCL)
  - GPU verification diagnostics
  - Fallback to CPU if GPU unavailable
- **TRAINING module self-contained**:
  - Moved all utility dependencies from `scripts/` to `TRAINING/utils/`
  - Moved `rank_target_predictability.py` to `TRAINING/ranking/`
  - Moved `multi_model_feature_selection.py` to `TRAINING/ranking/`
  - TRAINING module now has zero dependencies on `scripts/` folder
- Centralized configuration system with 9 training config YAML files (pipeline, GPU, memory, preprocessing, threading, safety, callbacks, optimizer, system)
- Config loader with nested access and family-specific overrides
- Compliance documentation suite:
  - `LICENSE_ENFORCEMENT.md` – License enforcement procedures and compliance requirements
  - `COMMERCIAL_USE.md` – Quick reference guide for commercial use
  - `COMPLIANCE_FAQ.md` – Frequently asked compliance questions
  - `PRODUCTION_USE_NOTIFICATION.md` – Production use notification form
  - `COPYRIGHT_NOTICE.md` – Copyright notice requirements
- Base trainer scaffolding for 2D and 3D models (`base_2d_trainer.py`, `base_3d_trainer.py`)
- Production use notification requirements
- Fork notification requirements
- Enhanced copyright headers across codebase (2025-2026 Fox ML Infrastructure LLC)

### Changed
- **Leakage Safety Suite improvements**:
  - **Leakage filtering now supports ranking mode**:
    - `filter_features_for_target()` accepts `for_ranking` parameter
    - Ranking mode: permissive rules, allows basic OHLCV/TA features even if in always_exclude
    - Training mode: strict rules (default, backward compatible)
    - Ensures ranking has sufficient features to evaluate target predictability
  - **Random Forest training accuracy no longer triggers critical leakage**:
    - High training accuracy (≥99.9%) now logged as warning, not error
    - Tree models can overfit to 100% training accuracy without leakage
    - Real leakage defense: schema filters + pre-training scan + time-purged CV
    - Prevents false positives from overfitting detection
- All model trainers updated to use centralized configs (preprocessing, callbacks, optimizers, safety guards)
- Pipeline, threading, memory, GPU, and system settings integrated into centralized config system
- Updated company address in Terms of Service (STE B 212 W. Troy St., Dothan, AL 36303)

### Fixed
- Fixed `_perfect_correlation_models` NameError in target ranking
- Fixed insufficient features handling (now properly filters targets with <2 features)
- Fixed early exit logic when leakage detected (removed false positive triggers)
- Improved error messages when no targets selected after ranking
- **Auto-fixer import path** — fixed `parents[3]` to `parents[2]` in `leakage_auto_fixer.py` for correct repo root detection
- **Auto-fixer training accuracy detection** — now passes actual training accuracy (from `model_metrics`) instead of CV scores to auto-fixer
- **Auto-fixer pattern-based fallback** — added fallback detection when `model_importance` is missing
- **LightGBM GPU verbose parameter** — moved `verbose` from `fit()` to model constructor (LightGBM API requirement)
- **Leakage filtering path resolution** — fixed config path lookup in `leakage_filtering.py` when moved to `TRAINING/utils/`
- **Hardcoded safety net in leakage filtering** — added fallback patterns to exclude known leaky features even when config fails to load
- **Path resolution in moved files** — corrected `parents[2]` vs `parents[3]` for repo root detection
- **Import paths after module migration** — all `scripts.utils.*` imports updated to `TRAINING.utils.*`
- **Auto-fixer pre-excluded feature check** — now filters out already-excluded features before detection to prevent redundant exclusions
- VAE serialization issues — custom Keras layers now properly imported before deserialization
- Sequential models 3D preprocessing issues — input shape handling corrected
- XGBoost source-build stability — persistent build directory and non-editable install
- Readline symbol lookup errors — environment variable fixes
- TensorFlow GPU initialization — CUDA library path resolution
- Type conversion issues in callback configs (min_lr, factor, patience, etc.)
- LSTM timeout issues — dynamic batch and epoch scaling implemented
- Transformer OOM errors — reduced batch size and attention heads, dynamic scaling
- CNN1D, LSTM, Transformer input shape mismatches — 3D to 2D reshape fixes

### Security
- Enhanced compliance documentation for production use
- License enforcement procedures documented
- Copyright notice requirements standardized

### Commercial
- Commercial license pricing updated to market-aligned enterprise infrastructure tiers:
  - 1–10 employees: $30,000/year
  - 11–50 employees: $75,000/year
  - 51–250 employees: $200,000/year
  - 251–1000 employees: $350,000/year
  - 1000+ employees: Starts at $750,000/year (custom quote)
- Added optional add-ons for enterprise deployments:
  - Dedicated Support Retainer: $3,500–$12,000/month (SLA-based)
  - Custom Integration Projects: $25,000–$150,000 one-time
  - Onboarding & Deployment: $15,000–$50,000 one-time
  - Private Slack / Direct Founder Access: $18,000–$60,000/year
  - Additional User Seats: $500–$2,000 per seat/year
- Pricing aligned with market comps for enterprise ML infrastructure and reflects value of replacing 4–8 engineers + MLOps team + compliance overhead

### Documentation
- **Documentation structure reorganization**:
  - Moved all CONFIG documentation to `docs/02_reference/configuration/`:
    - Configuration system overview, feature/target configs, training pipeline configs, safety/leakage configs, model configuration, usage examples
    - Created minimal `CONFIG/README.md` that points to docs folder
  - Moved all TRAINING documentation to `docs/` folder:
    - Implementation guides → `docs/03_technical/implementation/` (feature selection, training optimization, safe target pattern, first batch specs, strategy updates, experiments implementation)
    - Tutorial/workflow docs → `docs/01_tutorials/training/` (experiments workflow, quick start, operations, phase 1 feature engineering)
    - Created minimal `TRAINING/README.md` and `TRAINING/EXPERIMENTS/README.md` that point to docs
  - Created comprehensive legal documentation index (`docs/LEGAL_INDEX.md`):
    - Complete index of all legal, licensing, compliance, and enterprise documentation
    - Organized by category: Licensing, Terms & Policies, Enterprise & Compliance, Security, Legal Agreements, Consulting Services
  - Cleaned up main documentation index (`docs/INDEX.md`):
    - Removed duplicate sections (Implementation Guides was duplicating Tier D → Implementation)
    - Added "Project Status & Licensing" block after Quick Navigation (surfaces Roadmap, Changelog, Subscriptions, Legal Index)
    - Added "Who Should Read What" routing guide for different audiences
    - Clarified Model Training Guide differentiation (tutorial: "how to run it" vs specification: "what the system is")
    - Renamed "Additional Documentation" to "System Specifications" for clarity
  - Fixed all cross-references throughout documentation:
    - Updated all broken links to point to correct locations in docs/ folder
    - Added proper cross-links between related documentation
    - All relative paths corrected
  - Code directories now contain only code and minimal README pointers:
    - `CONFIG/` contains only YAML config files and minimal README
    - `TRAINING/` contains only code and minimal README
    - All documentation centralized in `docs/` for professional organization
- Updated `LEAKAGE_ANALYSIS.md` with pre-training leak scan and new config options
- Updated `INTELLIGENT_TRAINING_TUTORIAL.md` with configuration details
- Marked target ranking integration as completed in planning docs
- Updated `README.md` with direct commercial licensing focus and recent feature improvements
- Added NVLink-ready architecture planning documentation
- Updated `ROADMAP.md` with NVLink compatibility exploration and feature engineering revamp plans
- Hardened `COMMERCIAL_LICENSE.md` with enterprise-grade improvements (AGPL clarity, termination, audit, SaaS restrictions)
- Added comprehensive configuration documentation for all leakage detection thresholds
- 55+ new documentation files created
- 50+ existing files rewritten and standardized
- Enterprise-grade legal and commercial materials established
- 4-tier documentation hierarchy implemented
- Cross-linking and navigation improved
- Module reference documentation added
- Configuration schema documentation added

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

