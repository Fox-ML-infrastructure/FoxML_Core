# FoxML Core — ML Cross-Sectional Infrastructure

**FoxML is research-grade ML infrastructure with deterministic strict mode + full fingerprint lineage. It assumes prior experience with Python, Linux, and quantitative workflows.**

> ⚠️ **Disclaimer:** This software is provided for research and educational purposes only. It does not constitute financial advice, and no guarantees of returns or performance are made. Use at your own risk.

> 💻 **Interface:** This is a command-line and config-driven system. There is no graphical UI, web dashboard, or visual interface. All interaction is via YAML configuration files and Python scripts.

> **License:**  
> - **AGPL-3.0** (open source)  
> - **Commercial License** (for proprietary or for-profit use)  
>
> **Licensing TL;DR:**
> - **AGPL-3.0**: If you deploy as a service, AGPL obligations apply unless you have a commercial license
> - **Commercial License**: Available for proprietary / closed deployments
> - See [LICENSE](LICENSE) and [COMMERCIAL_LICENSE.md](COMMERCIAL_LICENSE.md) for full terms

> **🔍 Reproducibility & Auditability:** This system supports **bitwise deterministic runs** via strict mode (`bin/run_deterministic.sh`) for financial audit compliance. Bitwise determinism requires CPU-only execution, pinned dependencies, fixed thread env vars, and deterministic data ordering. Note: Not guaranteed across different CPUs/BLAS versions/kernels/drivers/filesystem ordering. See [Deterministic Runs](DOCS/02_reference/configuration/DETERMINISTIC_RUNS.md).

---

> **⚠️ ACTIVE DEVELOPMENT — EXPECT BREAKING CHANGES**  
> This project is under **heavy active development**. Breaking changes may occur without notice. APIs, configuration schemas, directory structures, and file formats may change between commits. Use at your own risk in production environments. See [ROADMAP.md](DOCS/02_reference/roadmap/ROADMAP.md) for current status and known issues.

> **🎯 Version 1.0 Definition:** See [FOXML_1.0_MANIFEST.md](DOCS/00_executive/product/FOXML_1.0_MANIFEST.md) for the capability boundary that defines what constitutes FoxML 1.0.

> **📝 See [CHANGELOG.md](CHANGELOG.md) for recent technical and compliance changes.**

---

FoxML Core is an ML infrastructure stack for cross-sectional and panel data, supporting both pooled cross-sectional training and symbol-specific (per-symbol) training modes. Designed for any machine learning applications requiring temporal validation and reproducibility. It provides a config-driven ML pipeline architecture designed for ML infra teams, data scientists, and researchers.

**Why This Exists:** Most ML repos are notebooks + scripts; FoxML is pipeline + audit artifacts. Designed to make research reproducible, comparable, and reviewable. This is infrastructure-first ML tooling, not a collection of example notebooks.

> **📊 Testing & Development:** All testing, validation, and development work is performed using **5-minute interval data**. The software supports various data intervals, but all tests, benchmarks, and development workflows use 5-minute bars as the standard reference.

Developed and maintained by **Jennifer Lewis**  
Independent Contractor • ML Engineering • Cross-Sectional ML Systems • Systems Architecture  
🔗 [LinkedIn](https://www.linkedin.com/in/jennifer-l-3434b3315)

---

## What You Get

- **3-stage intelligent pipeline**: Automated target ranking → feature selection → model training with config-driven routing decisions
- **Dual-view evaluation**: Supports both cross-sectional (pooled across symbols) and symbol-specific (per-symbol) training modes for comprehensive target analysis
- **Automatic routing**: Intelligently routes targets to cross-sectional, symbol-specific, or both training modes based on performance metrics and data characteristics
- **Task-aware evaluation**: Unified handling of regression (IC-based) and classification (AUC-based) targets with normalized skill scores
- **GPU acceleration** for target ranking, feature selection, and model training (LightGBM, XGBoost, CatBoost)
- **Bitwise deterministic runs** via strict mode (CPU-only, pinned dependencies, fixed thread env vars, deterministic data ordering) for financial audit compliance. Note: Not guaranteed across different CPUs/BLAS versions/kernels/drivers/filesystem ordering.
  - *Yes, outputs remain deterministic even while you're grinding OSRS, watching YouTube, or questioning your life choices at 3 AM. We tested it.*
- **Config-based usage** with minimal command-line arguments
- **Leakage detection system** with pre-training leak detection and auto-fix
- **Single Source of Truth (SST)** config system - all 20 model families use config-driven hyperparameters
- **Multi-model training systems** with 20 model families (LightGBM, XGBoost, CatBoost, MLP, CNN1D, LSTM, Transformer, TabCNN, TabLSTM, TabTransformer, RewardBased, QuantileLightGBM, NGBoost, GMMRegime, ChangePoint, FTRLProximal, VAE, GAN, Ensemble, MetaLearning, MultiTask) - GPU-accelerated where supported
- **Local metrics tracking** - By default, runs are local-only; no data is sent externally. All metrics stored locally for reproducibility.

---

## Quick Start (30 Seconds)

```bash
# One-line install
bash bin/install.sh

# Activate environment
conda activate trader

# Test installation
bash bin/test_install.sh

# Run a quick test
python -m TRAINING.orchestration.intelligent_trainer \
    --output-dir "test_install"

# Check results
ls RESULTS/runs/*/globals/config.resolved.json
```

**Manual install:** `conda env create -f environment.yml && conda activate trader`

See [Quick Start Guide](DOCS/00_executive/QUICKSTART.md) for full setup.

---

## Key Concepts

**Cross-sectional vs Symbol-specific:**
- **Cross-sectional**: Pooled training across all symbols (learns patterns common across the universe)
- **Symbol-specific**: Per-symbol training (learns patterns unique to each symbol)

**Pipeline Stages:**
- **Target Ranking**: Ranks targets by predictability using multiple model families
- **Feature Selection**: Selects optimal features per target using importance analysis
- **Training**: Trains final models with routing decisions (cross-sectional, symbol-specific, or both)

**Determinism Modes:**
- **Strict mode**: Bitwise deterministic (CPU-only, single-threaded, pinned deps) - use `bin/run_deterministic.sh`
- **Best-effort mode**: Seeded but may vary (allows GPU, multi-threading) - default behavior

**Fingerprints vs RunIdentity:**
- **Fingerprints**: SHA256 hashes of individual components (data, config, features, targets)
- **RunIdentity**: Complete run signature combining all fingerprints for full traceability

---

## Fingerprinting & Reproducibility

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

**CPU Recommendations:**
- **Stable clocks**: Disable turbo boost/overclocking features for stability and consistency
- **Undervolting**: Slight undervolting is recommended for stability (reduces thermal throttling and power fluctuations)
- **Newer CPUs**: Generally perform better due to improved instruction sets and efficiency
- **Core count**: More cores are beneficial, but some operations are single-threaded, so core count only helps with parallel aspects of the pipeline
- **Base clock speed**: Faster base clocks improve performance across all operations
- **Best practice**: Disable turbo boost features and use stable, consistent clock speeds for reproducible results

**GPU Considerations:**
- **VRAM dependent**: GPU performance is primarily limited by available VRAM rather than compute cores
- **Non-determinism**: GPU operations introduce slight non-determinism (generally within acceptable tolerances) due to parallel floating-point arithmetic where operation ordering is not guaranteed
- **Strict mode**: For bitwise deterministic runs, GPU is automatically disabled for tree models (LightGBM, XGBoost, CatBoost) in strict mode
- **Best practice**: More VRAM and newer GPU architectures generally provide better performance when GPU acceleration is enabled

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
- Production deployments (when pinned to tagged releases / frozen configs)

### Not Appropriate For
- Unmodified production deployment without proper testing and validation
- Production use of `main` branch (not stable for production)

**Production Note:** Production is supported when pinned to tagged releases / frozen configs; `main` branch is not stable for production use.

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
