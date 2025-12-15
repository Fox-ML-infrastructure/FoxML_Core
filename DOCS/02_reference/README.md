# Reference Documentation

Complete technical reference for daily use, API documentation, and system specifications.

## Purpose

This directory contains comprehensive reference documentation for developers, ML engineers, and system administrators. These documents serve as authoritative sources for APIs, configurations, data formats, and system behavior.

## Contents

### API Reference
- **[Module Reference](api/MODULE_REFERENCE.md)** - Python API for all modules
- **[Intelligent Trainer API](api/INTELLIGENT_TRAINER_API.md)** - Intelligent training pipeline API
- **[CLI Reference](api/CLI_REFERENCE.md)** - Command-line interface reference
- **[Data Processing API](api/DATA_PROCESSING_API.md)** - Data processing module API
- **[Config Schema](api/CONFIG_SCHEMA.md)** - Configuration schema reference

### Configuration
- **[Configuration System Overview](configuration/README.md)** - Centralized configuration system
- **[Config README](configuration/CONFIG_README.md)** - CONFIG directory structure and organization
- **[Config README Defaults](configuration/CONFIG_README_DEFAULTS.md)** - Default configuration values
- **[Modular Config System](configuration/MODULAR_CONFIG_SYSTEM.md)** - Modular configs and experiment configs
- **[Feature & Target Configs](configuration/FEATURE_TARGET_CONFIGS.md)** - Feature/target configuration
- **[Model Configuration](configuration/MODEL_CONFIGURATION.md)** - Model hyperparameters
- **[Safety & Leakage Configs](configuration/SAFETY_LEAKAGE_CONFIGS.md)** - Leakage detection configs
- **[Config Loader API](configuration/CONFIG_LOADER_API.md)** - Programmatic config loading
- **[Config Audit](configuration/CONFIG_AUDIT.md)** - Config folder audit and organization

### Data Reference
- **[Data Format Spec](data/DATA_FORMAT_SPEC.md)** - Data format specifications
- **[Column Reference](data/COLUMN_REFERENCE.md)** - Column documentation
- **[Data Sanity Rules](data/DATA_SANITY_RULES.md)** - Data validation rules

### Models Reference
- **[Model Catalog](models/MODEL_CATALOG.md)** - All available models
- **[Model Config Reference](models/MODEL_CONFIG_REFERENCE.md)** - Model configurations
- **[Training Parameters](models/TRAINING_PARAMETERS.md)** - Training settings

### Systems Reference
- **[Pipeline Reference](systems/PIPELINE_REFERENCE.md)** - Data pipeline specifications
- **[Training Routing System](training_routing/README.md)** - Routing decisions and training plans
- **[Target Ranking](target_ranking/README.md)** - Target ranking system

### Changelog
- **[Changelog Index](changelog/README.md)** - Per-day detailed changelogs

**Note**: Documentation audits have been moved to [Executive Documentation](../00_executive/audits/README.md) for public transparency.

### Roadmap
- **[Roadmap](roadmap/ROADMAP.md)** - Executive summary and high-level development priorities
- **[Roadmap Index](roadmap/README.md)** - Detailed per-date roadmap with component status

### General
- **[Known Issues](KNOWN_ISSUES.md)** - Current limitations and known issues
- **[Detailed Changelog](CHANGELOG_DETAILED.md)** - Comprehensive changelog

## Who Should Read This

- **Developers** - API Reference, Configuration Reference
- **ML Engineers** - Models Reference, Systems Reference
- **System Administrators** - Configuration Reference, Systems Reference
- **QA/Testing** - Known Issues (audits moved to executive docs for transparency)

## Related Documentation

- [Tutorials](../01_tutorials/) - Step-by-step guides
- [Technical Documentation](../03_technical/) - Deep technical details
- [Executive Documentation](../00_executive/) - High-level overviews

