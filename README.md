# FoxML Core — ML Cross-Sectional Infrastructure

**FoxML is research-grade ML infrastructure. It assumes prior experience with Python, Linux, and quantitative workflows. Continued development will most likely be private going forward, I may continue pushing public updates, im not sure yet. 
Keeping every update OSS adds pressure that isn’t sustainable right now.This repository reflects a stable snapshot of the project, including key determinism work completed up to this point. This does not mean development has stopped — only that I won’t be as transparent about ongoing work.
Im pretty sure that this will run with deterministic settings, if not someone can reach out and let me know im prioritizing continued development. 
| bin/run_deterministic.sh python TRAINING/orchestration/intelligent_trainer.py --experiment-config determinism_test |
This is the command Ive been running tests with.**

> ⚠️ **Disclaimer:** This software is provided for research and educational purposes only. It does not constitute financial advice, and no guarantees of returns or performance are made. Use at your own risk.
>  

> 💻 **Interface:** This is a command-line and config-driven system. There is no graphical UI, web dashboard, or visual interface. All interaction is via YAML configuration files and Python scripts.

> **License:**  
> - **AGPL-3.0** (open source)  
> - **Commercial License** (for proprietary or for-profit use)  
>
> See [LICENSE](LICENSE) and [COMMERCIAL_LICENSE.md](COMMERCIAL_LICENSE.md) for full terms. 

> **📝 See [CHANGELOG.md](CHANGELOG.md) for recent technical and compliance changes.**

---

FoxML Core is an ML infrastructure stack for cross-sectional and panel data for any machine learning applications. It provides a config-driven ML pipeline architecture designed for ML infra teams, data scientists, and researchers.

> **📊 Testing & Development:** All testing, validation, and development work is performed using **5-minute interval data**. The software supports various data intervals, but all tests, benchmarks, and development workflows use 5-minute bars as the standard reference.

Developed and maintained by **Jennifer Lewis**  
Independent Contractor • ML Engineering • Cross-Sectional ML Systems • Systems Architecture  
🔗 [LinkedIn](https://www.linkedin.com/in/jennifer-l-3434b3315)

---

## Quick Overview

FoxML Core provides:

- **Intelligent training pipeline** with automated target ranking and feature selection
- **GPU acceleration** for target ranking, feature selection, and model training (LightGBM, XGBoost, CatBoost)
- **Bitwise deterministic runs** via strict mode for financial audit compliance and regulatory requirements
- **Config-based usage** with minimal command-line arguments
- **Leakage detection system** with pre-training leak detection and auto-fix
- **Single Source of Truth (SST)** config system - all 20 model families use config-driven hyperparameters
- **Multi-model training systems** with 20+ model families (GPU-accelerated)
- **Reproducibility tracking** with end-to-end reproducibility verification and auditability
- **Local metrics tracking** - Model performance metrics (ROC-AUC, R², feature importance) stored locally for reproducibility. No external data transmission, no user data collection.

**For detailed capabilities:** See [Architecture Overview](DOCS/00_executive/ARCHITECTURE_OVERVIEW.md)

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
