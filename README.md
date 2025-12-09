# FoxML Core — ML & Quantitative Systems Infrastructure

> **📋 See [ROADMAP.md](ROADMAP.md) for most recent updates, current development focus, and status of modules (including known broken/untested components).**  
> **📝 See [CHANGELOG.md](CHANGELOG.md) for recent technical and compliance changes.**

FoxML Core is:

- A leakage-safe ML research and training infrastructure stack for cross-sectional trading
- A reference architecture for high-throughput, config-driven ML pipelines
- Designed for quant desks, infra teams, and researchers (architecture, not alpha)

FoxML Core is a high-performance research and machine learning infrastructure stack.  
It provides a reference-grade architecture for ML pipelines, quantitative workflows, reproducible experiments, and HPC-optimized engineering without exposing proprietary signals or production-ready trading systems.

> **Note:** Personal / academic use is AGPL-3.0. **Company / production use requires a commercial license.**

Developed and maintained by **Jennifer Lewis**  
Independent Contractor • ML Engineering • Quantitative Research • Systems Architecture

---------------------------------------------------------------------

## Overview

FoxML Core demonstrates:

- **Intelligent training pipeline** with automated target ranking and feature selection
- **Leakage-safe research architecture** with pre-training leak detection and auto-fix
- **Scalable ML workflow design** with config-driven orchestration
- **High-throughput data processing** with Polars-optimized pipelines
- **Multi-model training systems** with 20+ model families (GPU-accelerated)
- **HPC-compatible orchestration patterns** (NVLink-ready architecture planned)

This is a **research infrastructure system**, not a trading bot or financial product.

---------------------------------------------------------------------

## Domain Focus & Extensibility

FoxML Core is **optimized and tested for financial time series** (cross-sectional trading data, market microstructure, price/volume features). The architecture is designed for financial ML workflows with domain-specific safeguards (leakage detection, temporal validation, feature registry systems).

**Architecture Property:** While built for finance, the core primitives are domain-agnostic by design:
- Config-driven orchestration
- Automated target ranking and feature selection
- Leakage-safe validation frameworks
- Multi-model training systems
- Time-series-aware cross-validation

**Non-Financial Workloads:** Other time-series or panel data domains (IoT, healthcare, clickstreams, etc.) are architecturally possible but require:
- Custom data loaders and feature engineering blocks
- Domain-specific target definitions and metrics
- Appropriate leakage rules and validation patterns
- Domain expertise for proper configuration

**Official Support:** Official support, reference implementations, and tested workflows currently focus on financial data. Non-financial use cases are not officially supported, though the architecture enables such extensions with appropriate domain adaptation.

This positions FoxML Core as **serious infrastructure** (not a one-off HFT tool) while maintaining clear boundaries about current support scope.

---------------------------------------------------------------------

## Intended Use

### Appropriate Use Cases
- Research and experimentation  
- ML workflow and architecture study  
- Institutional or academic analysis  
- Internal engineering reference  
- Benchmarking and systems review  

### Not Appropriate For
- Turnkey trading or retail strategy deployment  
- Financial advice or automated trading  
- Commercial use without a license  
- Unmodified production HFT or revenue-generating systems  

FoxML Core provides **architecture**, not alpha.

---------------------------------------------------------------------

## Licensing & Commercial Use

FoxML Core is released under the **AGPL-3.0** license. Personal / academic research uses AGPL. Organizations using FoxML Core in production or revenue-generating environments should obtain a commercial license.

That means:

- Individual developers, students, and researchers can use FoxML Core under AGPL-3.0 for **personal and academic research, evaluation, and experimentation**.
- Organizations that want to use FoxML Core in **production, revenue-generating, or internal trading / research environments** will almost always want a **commercial license** to avoid AGPL copyleft obligations and to get clear commercial terms.

> ⚠️ **If you're running this inside a company (trading desk, fund, fintech, SaaS, etc.), you should assume you need a commercial license unless your legal team is explicitly comfortable with AGPL-3.0 obligations.**

### Commercial License (Recommended for Organizations)

Commercial licensing gives your organization the right to:

- Use FoxML Core for **commercial and internal production use**
- Integrate FoxML into proprietary systems without AGPL disclosure obligations
- Get access to **enterprise-focused terms** and optional support / integration work

**Annual License Tiers**

- 1–10 employees — $150,000 / year  
- 11–50 employees — $350,000 / year  
- 51–250 employees — $750,000 / year  
- 251–1000 employees — $1,500,000–$2,500,000 / year  
- 1000+ employees — $5,000,000–$12,000,000+ / year (custom enterprise quote)

To begin the licensing process, email:

> 📧 **jenn.lewis5789@gmail.com**  
> Subject: `FoxML Core Commercial Licensing`

Please include:

- Your name and role
- Organization name + website or LinkedIn
- Organization size (from the tiers above)
- Primary use case (1–3 sentences)
- Desired start timeline

### Optional Add-Ons

For trading desks and infra teams that need more than a license, optional services are available, including:

- **Dedicated support retainers** (priority fixes, direct maintainer access)
- **Custom integration projects** (fit FoxML into your existing infra and data)
- **Onboarding & deployment assistance**
- **Private Slack / direct founder access**

See:

- [`LEGAL/COMMERCIAL_USE.md`](LEGAL/COMMERCIAL_USE.md) for commercial license terms
- [`LEGAL/SUBSCRIPTIONS.md`](LEGAL/SUBSCRIPTIONS.md) for detailed pricing and add-ons
- [`LEGAL/COMMERCIAL_LICENSE.md`](LEGAL/COMMERCIAL_LICENSE.md) for the full FoxML Core Commercial License text

---------------------------------------------------------------------

## Consulting Services

I provide advanced engineering and research infrastructure consulting for organizations requiring specialized ML, quantitative, or HPC systems.

For full policies and contracting information, see the [`LEGAL/`](LEGAL/) directory in this repository. 

### Core Expertise

#### Machine Learning Infrastructure
- **Intelligent training pipeline** with automated target ranking and selection
- **Automated leakage detection** with pre-training scans and auto-fix system
- **Config-driven orchestration** with 9 centralized config files
- End-to-end pipeline architecture with leakage-safe validation
- Configurable model zoos (20+ families: LightGBM, XGBoost, Random Forest, Neural Networks, etc.)
- GPU-optimized training with CUDA support

#### Quantitative Research Engineering
- **Automated target predictability assessment** with multi-model evaluation
- **Feature/target schema system** with ranking vs. training mode rules
- Walk-forward analysis frameworks with time-purged cross-validation
- Strict leakage auditing with sentinel tests (shifted-target, symbol-holdout, randomized-time)
- Research-oriented data pipelines with Polars optimization
- Feature engineering at scale with registry-based temporal rules  

#### High-Performance Computing
- Multi-node and GPU workflows  
- **NVLink-ready architecture** (planned: device group abstraction, multi-GPU scheduling)
- System-level throughput optimization  
- Parallel experiment execution
- GPU topology detection and logging  

#### Systems Architecture
- Reproducible, config-driven workflows  
- Enterprise deployment patterns  
- Code correctness and architecture review  

### Engagement Model
- Remote contract work  
- Hourly, project-based, or retainer  
- SOW-defined milestones  
- Professional communication  
- Full documentation with deliverables  

---------------------------------------------------------------------

## Documentation

**New users start here:**
- **[Quick Start](DOCS/00_executive/QUICKSTART.md)** - Get running in 5 minutes
- **[Getting Started](DOCS/00_executive/GETTING_STARTED.md)** - Complete onboarding guide
- **[Architecture Overview](DOCS/00_executive/ARCHITECTURE_OVERVIEW.md)** - System at a glance

**Complete documentation:**
- **[Documentation Index](DOCS/INDEX.md)** - Full documentation navigation
- **[Tutorials](DOCS/01_tutorials/)** - Step-by-step guides
- **[Reference Docs](DOCS/02_reference/)** - Technical reference
- **[Technical Appendices](DOCS/03_technical/)** - Deep technical topics

**Documentation structure:**
- **[Architecture](DOCS/ARCHITECTURE.md)** - Documentation organization
- **[Style Guide](DOCS/STYLE_GUIDE.md)** - Writing guidelines
- **[Migration Plan](DOCS/MIGRATION_PLAN.md)** - Migration status  

---------------------------------------------------------------------

## Repository Structure

FoxML Core repository structure:

```
FoxML_Core/
├── trading/               (Paper trading integrations, if applicable)
├── IBKR_trading/          (IBKR trading module - contains C++ code, untested)
├── data_processing/       (Pipelines & feature engineering)
├── TRAINING/              (Model training & research workflows)
├── CONFIG/                (Configuration management system)
├── DOCS/                  (Technical documentation)
└── SCRIPTS/               (Utilities & tools)
```

**Note:** C++ code exists only within the `IBKR_trading/cpp_engine/` module and is currently untested. The rest of the codebase is Python-based.

---------------------------------------------------------------------

## Professional Standards

- Defined scopes and clear communication  
- NDA-compliant handling of client datasets  
- High-quality, reproducible engineering  
- Minimal onboarding overhead for teams  
- SOW-structured engagements  

---------------------------------------------------------------------

## Contact
linkedin: https://www.linkedin.com/in/jennifer-lewis-3434b3315/

For consulting, enterprise licensing, or organizational engagements:  
**jenn.lewis5789@gmail.com**
