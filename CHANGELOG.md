# Changelog

All notable changes to FoxML Core will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- **Automated leakage detection and auto-fix system**:
  - `LeakageAutoFixer` class for automatic detection and remediation of data leakage
  - Integration with leakage sentinels (shifted-target, symbol-holdout, randomized-time tests)
  - Automatic config file updates (`excluded_features.yaml`, `feature_registry.yaml`)
  - Auto-fixer triggers automatically when perfect scores (≥0.99) are detected during target ranking
  - **Configurable auto-fixer thresholds** in `safety_config.yaml`:
    - CV score threshold (default: 0.99)
    - Training accuracy threshold (default: 0.999)
    - Training R² threshold (default: 0.999)
    - Perfect correlation threshold (default: 0.999)
    - Minimum confidence for auto-fix (default: 0.8)
    - Enable/disable auto-fixer flag
- **Feature registry system** (`CONFIG/feature_registry.yaml`):
  - Structural rules based on temporal metadata (`lag_bars`, `allowed_horizons`, `source`)
  - Automatic filtering based on target horizon to prevent leakage
  - Support for short-horizon targets (added horizon=2 for 10-minute targets)
- **LightGBM GPU support** in target ranking:
  - Automatic GPU detection and usage (CUDA/OpenCL)
  - GPU verification diagnostics
  - Fallback to CPU if GPU unavailable
- **Leakage sentinels** (`TRAINING/common/leakage_sentinels.py`):
  - Shifted target test – detects features encoding future information
  - Symbol holdout test – detects symbol-specific leakage
  - Randomized time test – detects temporal information leakage
- **Feature importance diff detector** (`TRAINING/common/importance_diff_detector.py`):
  - Compares feature importances between full vs. safe feature sets
  - Identifies suspicious features with high importance in full model but low in safe model
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
- All model trainers updated to use centralized configs (preprocessing, callbacks, optimizers, safety guards)
- Pipeline, threading, memory, GPU, and system settings integrated into centralized config system
- Updated company address in Terms of Service (STE B 212 W. Troy St., Dothan, AL 36303)

### Fixed
- **Auto-fixer import path** — fixed `parents[3]` to `parents[2]` in `leakage_auto_fixer.py` for correct repo root detection
- **Auto-fixer training accuracy detection** — now passes actual training accuracy (from `model_metrics`) instead of CV scores to auto-fixer
- **Auto-fixer pattern-based fallback** — added fallback detection when `model_importance` is missing
- **LightGBM GPU verbose parameter** — moved `verbose` from `fit()` to model constructor (LightGBM API requirement)
- **Leakage filtering path resolution** — fixed config path lookup in `leakage_filtering.py` when moved to `TRAINING/utils/`
- **Hardcoded safety net in leakage filtering** — added fallback patterns to exclude known leaky features even when config fails to load
- **Path resolution in moved files** — corrected `parents[2]` vs `parents[3]` for repo root detection
- **Import paths after module migration** — all `scripts.utils.*` imports updated to `TRAINING.utils.*`
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
- Pricing aligned with market comps (Databricks, Hopsworks, QuantConnect Institution) and reflects value of replacing 4–8 engineers + MLOps team + compliance overhead

### Documentation
- 55+ new documentation files created
- 50+ existing files rewritten and standardized
- Enterprise-grade legal and commercial materials established
- 4-tier documentation hierarchy implemented
- Cross-linking and navigation improved
- Module reference documentation added
- Configuration schema documentation added

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

