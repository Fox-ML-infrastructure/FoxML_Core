# Changelog

All notable changes to FoxML Core will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0] - 2025-12-06

### Added
- Centralized configuration system with 9 training config YAML files (pipeline, GPU, memory, preprocessing, threading, safety, callbacks, optimizer, system)
- Config loader with nested access and family-specific overrides
- Compliance documentation suite:
  - `LICENSE_ENFORCEMENT.md` - License enforcement procedures and compliance requirements
  - `COMMERCIAL_USE.md` - Quick reference guide for commercial use
  - `COMPLIANCE_FAQ.md` - Frequently asked compliance questions
  - `PRODUCTION_USE_NOTIFICATION.md` - Production use notification form
  - `COPYRIGHT_NOTICE.md` - Copyright notice requirements
- Base trainer scaffolding for 2D and 3D models (`base_2d_trainer.py`, `base_3d_trainer.py`)
- Production use notification requirements
- Fork notification requirements
- Enhanced copyright headers across codebase (2025-2026 Fox ML Infrastructure LLC)

### Changed
- All model trainers updated to use centralized configs (preprocessing, callbacks, optimizers, safety guards)
- Pipeline, threading, memory, GPU, and system settings integrated into centralized config system
- Updated company address in Terms of Service (STE B 212 W. Troy St., Dothan, AL 36303)

### Fixed
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
- Commercial license pricing recalibrated to align with enterprise market norms:
  - 1-10 employees: $18,000 → $25,200/year
  - 11-50 employees: $36,000 → $60,000/year
  - 51-250 employees: $90,000 → $165,000/year
  - 251-1000 employees: $180,000 → $252,000/year
  - 1000+ employees: $300,000 → $500,000+/year (custom quote)

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

- **Added** - New features
- **Changed** - Changes in existing functionality
- **Deprecated** - Soon-to-be removed features
- **Removed** - Removed features
- **Fixed** - Bug fixes
- **Security** - Security improvements
- **Commercial** - Business and licensing changes
- **Documentation** - Documentation changes

