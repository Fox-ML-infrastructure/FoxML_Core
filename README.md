# FoxML Core — ML Cross-Sectional Infrastructure

**FoxML is research-grade ML infrastructure. It assumes prior experience with Python, Linux, and quantitative workflows.**

> ⚠️ **Disclaimer:** This software is provided for research and educational purposes only. It does not constitute financial advice, and no guarantees of returns or performance are made. Use at your own risk.

> 💻 **Interface:** This is a command-line and config-driven system. There is no graphical UI, web dashboard, or visual interface. All interaction is via YAML configuration files and Python scripts.

> **License:**  
> - **AGPL-3.0** (open source)  
> - **Commercial License** (for proprietary or for-profit use)  
>
> See [LICENSE](LICENSE) and [COMMERCIAL_LICENSE.md](COMMERCIAL_LICENSE.md) for full terms.

> **🔍 Reproducibility & Auditability:** This system supports **bitwise deterministic runs** via strict mode (`bin/run_deterministic.sh`) for financial audit compliance, plus comprehensive auditability (full tracking of inputs, configs, and outputs). See [Deterministic Runs](DOCS/02_reference/configuration/DETERMINISTIC_RUNS.md).

---

> **⚠️ ACTIVE DEVELOPMENT — EXPECT BREAKING CHANGES**  
> This project is under **heavy active development**. Breaking changes may occur without notice. APIs, configuration schemas, directory structures, and file formats may change between commits. Use at your own risk in production environments. See [ROADMAP.md](DOCS/02_reference/roadmap/ROADMAP.md) for current status and known issues.

> **🎯 Version 1.0 Definition:** See [FOXML_1.0_MANIFEST.md](DOCS/00_executive/product/FOXML_1.0_MANIFEST.md) for the capability boundary that defines what constitutes FoxML 1.0.

> **📝 See [CHANGELOG.md](CHANGELOG.md) for recent technical and compliance changes.**

---

FoxML Core is an ML infrastructure stack for cross-sectional and panel data, supporting both pooled cross-sectional training and symbol-specific (per-symbol) training modes. Designed for any machine learning applications requiring temporal validation and reproducibility. It provides a config-driven ML pipeline architecture designed for ML infra teams, data scientists, and researchers.

> **📊 Testing & Development:** All testing, validation, and development work is performed using **5-minute interval data**. The software supports various data intervals, but all tests, benchmarks, and development workflows use 5-minute bars as the standard reference.

Developed and maintained by **Jennifer Lewis**  
Independent Contractor • ML Engineering • Cross-Sectional ML Systems • Systems Architecture  
🔗 [LinkedIn](https://www.linkedin.com/in/jennifer-l-3434b3315)

---

## Quick Overview

FoxML Core provides:

- **3-stage intelligent pipeline**: Automated target ranking → feature selection → model training with config-driven routing decisions
- **Dual-view evaluation**: Supports both cross-sectional (pooled across symbols) and symbol-specific (per-symbol) training modes for comprehensive target analysis
- **Automatic routing**: Intelligently routes targets to cross-sectional, symbol-specific, or both training modes based on performance metrics and data characteristics
- **Task-aware evaluation**: Unified handling of regression (IC-based) and classification (AUC-based) targets with normalized skill scores
- **GPU acceleration** for target ranking, feature selection, and model training (LightGBM, XGBoost, CatBoost)
- **Bitwise deterministic runs** via strict mode for financial audit compliance and regulatory requirements
  - *Yes, outputs remain deterministic even while you're grinding OSRS, watching YouTube, or questioning your life choices at 3 AM. We tested it.*
- **Config-based usage** with minimal command-line arguments
- **Leakage detection system** with pre-training leak detection and auto-fix
- **Single Source of Truth (SST)** config system - all 20 model families use config-driven hyperparameters
- **Multi-model training systems** with 20+ model families (GPU-accelerated)
- **Local metrics tracking** - Model performance metrics (ROC-AUC, R², feature importance) stored locally for reproducibility. No external data transmission, no user data collection.

### Fingerprinting & Reproducibility

- **3-stage pipeline architecture**: TARGET_RANKING (ranks targets by predictability) → FEATURE_SELECTION (selects optimal features) → TRAINING (trains models with routing decisions)
- **Dual-view support**: Each stage evaluates targets in both CROSS_SECTIONAL (pooled) and SYMBOL_SPECIFIC (per-symbol) views for comprehensive analysis
- **SHA256 fingerprinting** for all pipeline components — data, config, features, targets, splits, hyperparameters, and routing decisions
- **RunIdentity system** with two-phase construction (partial → finalized) and strict/replicate key separation for audit-grade traceability
- **Diff telemetry** — automatic comparison with previous runs, distinguishing true regressions from acceptable nondeterminism
- **Feature importance snapshots** with cross-run stability analysis and drift detection
- **Stage-scoped output layout** — target-first directory organization with `stage=TARGET_RANKING/`, `stage=FEATURE_SELECTION/`, `stage=TRAINING/` separation for human-readable auditability
- **Cohort-based reproducibility** — each (target, view, universe, cohort) combination gets its own snapshot with full fingerprint lineage

**For detailed capabilities:** See [Architecture Overview](DOCS/00_executive/ARCHITECTURE_OVERVIEW.md)

---

## System Requirements

FoxML Core is designed to scale from laptop experiments to production-scale deployments. Hardware requirements scale with your data size and pipeline configuration.

**Small Experiments (Laptop-Friendly):**
- **RAM**: 16-32GB minimum
- **Use Cases**: Small sample sizes, limited universe sizes, reduced feature counts
- Suitable for development, testing, and small-scale research

**Production / Ideal Configuration:**
- **RAM**: 128GB minimum, 512GB-1TB recommended for best performance
- **Use Cases**: Large sample counts, extensive universe sizes, full feature sets
- Enables full pipeline execution without memory constraints

**Scaling Factors:**
- **Sample count**: More samples require more memory for data loading and model training
- **Universe size**: Larger symbol universes increase memory usage proportionally
- **Feature count**: Feature count directly affects hardware usage (more features = more memory and compute)

**Universe Batching:**
The pipeline supports batching large universes across multiple runs. While batching works, **running with as few batches as possible is ideal for best results** - it enables better cross-sectional analysis and more comprehensive feature selection across the full universe.

---

## Domain Focus

FoxML Core is **general-purpose ML cross-sectional infrastructure** for panel data and time-series workflows. The architecture provides domain-agnostic primitives with built-in safeguards (leakage detection, temporal validation, feature registry systems).

**Domain Applications:** Financial time series, IoT sensor data, healthcare, clickstream analytics, and any panel data with temporal structure.

**For detailed domain information:** See [Architecture Overview](DOCS/00_executive/ARCHITECTURE_OVERVIEW.md)

---

## Intended Use

### Appropriate Use Cases
- Research and experimentation
- ML workflow and architecture study
- Open-source projects
- Internal engineering reference
- Production deployments

### Not Appropriate For
- Unmodified production deployment without proper testing and validation

**FoxML Core provides ML infrastructure and architecture, not domain-specific applications or pre-built solutions.**

---

## Getting Started

**New users start here:**
- **[Quick Start](DOCS/00_executive/QUICKSTART.md)** - Get running in 5 minutes
- **[Getting Started](DOCS/00_executive/GETTING_STARTED.md)** - Complete onboarding guide
- **[Architecture Overview](DOCS/00_executive/ARCHITECTURE_OVERVIEW.md)** - System at a glance
- **[Deterministic Runs](DOCS/02_reference/configuration/DETERMINISTIC_RUNS.md)** - Bitwise reproducible runs for audit compliance

**Complete documentation:**
- **[Documentation Index](DOCS/INDEX.md)** - Full documentation navigation
- **[Tutorials](DOCS/01_tutorials/)** - Step-by-step guides
- **[Reference Docs](DOCS/02_reference/)** - Technical reference
- **[Technical Appendices](DOCS/03_technical/)** - Deep technical topics

---

## Repository Structure

```
FoxML_Core/
├── DATA_PROCESSING/       (Pipelines & feature engineering)
├── TRAINING/              (Model training & research workflows)
├── CONFIG/                (Configuration management system)
├── DOCS/                  (Technical documentation)
└── SCRIPTS/               (Utilities & tools)
```

**For detailed structure:** See [Architecture Overview](DOCS/00_executive/ARCHITECTURE_OVERVIEW.md)

---

## Reporting Issues

For bug reports, feature requests, or technical issues:

- **GitHub Issues**: [Open an issue](https://github.com/Fox-ML-infrastructure/FoxML_Core/issues) (preferred for bug reports and feature requests)
- **Email**: jenn.lewis5789@gmail.com (for security issues, sensitive bugs, or private inquiries)

For questions or organizational engagements:  
**jenn.lewis5789@gmail.com**

---
