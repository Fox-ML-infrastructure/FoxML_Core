# FoxML Core — ML Cross-Sectional Infrastructure

> **📋 See [ROADMAP.md](ROADMAP.md) for most recent updates, current development focus, and status of modules (including known broken/untested components).**  
> **📝 See [CHANGELOG.md](CHANGELOG.md) for recent technical and compliance changes.**

FoxML Core is:

- A leakage-safe ML infrastructure stack for cross-sectional and panel data for any machine learning applications
- A reference architecture for high-throughput, config-driven ML pipelines
- Designed for ML infra teams, data scientists, and researchers (architecture, not alpha)

FoxML Core is a high-performance research and machine learning infrastructure stack.  
It provides a reference-grade architecture for ML pipelines, quantitative workflows, reproducible experiments, and performance-optimized engineering (single-node HPC optimized; distributed HPC planned) without exposing proprietary signals or production-ready trading systems.

> **Note:** Personal / academic use is AGPL-3.0. **Company / production use requires a commercial license.**

Developed and maintained by **Jennifer Lewis**  
Independent Contractor • ML Engineering • Cross-Sectional ML Systems • Systems Architecture

---------------------------------------------------------------------

## Overview

FoxML Core demonstrates:

- **Intelligent training pipeline** with automated target ranking and feature selection
- **Leakage-safe research architecture** with pre-training leak detection and auto-fix
- **Scalable ML workflow design** with complete Single Source of Truth (SST) config system - all 52+ model trainers use config-driven hyperparameters for full reproducibility
- **High-throughput data processing** with Polars-optimized pipelines
- **Multi-model training systems** with 20+ model families (GPU-accelerated)
- **HPC-compatible orchestration patterns** (single-node optimized; distributed HPC planned/WIP)

This is a **cross-sectional ML infrastructure system** designed for panel data and time-series workflows for any machine learning applications.

---------------------------------------------------------------------

## Domain Focus & Extensibility

FoxML Core is **general-purpose ML cross-sectional infrastructure** optimized for panel data and time-series workflows. The architecture provides domain-agnostic primitives with built-in safeguards (leakage detection, temporal validation, feature registry systems).

**Core Capabilities:**
- Config-driven orchestration
- Automated target ranking and feature selection
- Leakage-safe validation frameworks
- Multi-model training systems
- Time-series-aware cross-validation
- Cross-sectional data processing

**Development Approach:** The system is currently developed and tested using financial time series data, but the architecture is **domain-agnostic by design**. The vision is to provide general-purpose cross-sectional ML infrastructure that works across any domain.

**Domain Applications:** The system is designed for cross-sectional and panel data across multiple domains:
- **Financial time series** (market data, price/volume features) — *currently used for development*
- **IoT sensor data** (device metrics, time-series sensors)
- **Healthcare** (patient records, longitudinal studies)
- **Clickstream analytics** (user behavior, event sequences)
- **Any panel data** with temporal structure

**Extensibility:** Custom data loaders, feature engineering blocks, domain-specific target definitions, and appropriate leakage rules can be configured for any domain. The architecture is domain-agnostic by design.

---------------------------------------------------------------------

## Intended Use

### Appropriate Use Cases
- Research and experimentation  
- ML workflow and architecture study  
- Institutional or academic analysis  
- Internal engineering reference  
- Benchmarking and systems review  

### Not Appropriate For
- Commercial use without a license  
- Unmodified production deployment without proper testing and validation

**FoxML Core provides ML infrastructure and architecture, not domain-specific applications or pre-built solutions.**

---------------------------------------------------------------------

## Licensing & Commercial Use

> ⚠️ **BUSINESS USE REQUIRES A COMMERCIAL LICENSE**
>
> - If you're running this inside a company or fund, you almost certainly need a commercial license.
> - Non-commercial academic / personal research is allowed under AGPL-3.0 (see `LICENSE`).
> - Full details and pricing: `LEGAL/SUBSCRIPTIONS.md`.

FoxML Core is released under the **AGPL-3.0** license. Personal / academic research uses AGPL. Organizations using FoxML Core in production or revenue-generating environments should obtain a commercial license.

That means:

- Individual developers, students, and researchers can use FoxML Core under AGPL-3.0 for **personal and academic research, evaluation, and experimentation** (see `LEGAL/SUBSCRIPTIONS.md` for full definition).
- Organizations that want to use FoxML Core in **production, revenue-generating, or internal research environments** will almost always want a **commercial license** to avoid AGPL copyleft obligations and to get clear commercial terms.

**If you are using FoxML Core inside any business or organization, assume you need a commercial license.** See `LEGAL/SUBSCRIPTIONS.md` for details.

### Commercial License (Recommended for Organizations)

Commercial licensing gives your organization the right to:

- Use FoxML Core for **commercial and internal production use**
- Integrate FoxML into proprietary systems without AGPL disclosure obligations
- Get access to **enterprise-focused terms** and optional support / integration work

**Indicative annual pricing** starts in the low six figures for small teams and scales to multi-million-dollar enterprise agreements for large institutions.

See [`LEGAL/SUBSCRIPTIONS.md`](LEGAL/SUBSCRIPTIONS.md) for detailed tiers and examples.

To begin the licensing process, email:

> 📧 **jenn.lewis5789@gmail.com**  
> Subject: `FoxML Core Commercial Licensing`

Please include:

- Your name and role
- Organization name + website or LinkedIn
- Organization size
- Primary use case (1–3 sentences)
- Desired start timeline

### Optional Add-Ons

For ML teams and infra teams that need more than a license, optional services are available, including:

- **Dedicated support retainers** (priority fixes, direct maintainer access)
- **Custom integration projects** (fit FoxML into your existing infra and data)
- **Onboarding & deployment assistance**
- **Private Slack / direct founder access**

See:

- [`LEGAL/COMMERCIAL_USE.md`](LEGAL/COMMERCIAL_USE.md) for commercial license terms
- [`LEGAL/SUBSCRIPTIONS.md`](LEGAL/SUBSCRIPTIONS.md) for detailed pricing and add-ons
- [`COMMERCIAL_LICENSE.md`](COMMERCIAL_LICENSE.md) for the full FoxML Core Commercial License text

---------------------------------------------------------------------

## Enterprise Partnerships

FoxML Core is available for enterprise licensing and custom development partnerships for organizations requiring specialized ML, quantitative, or HPC systems.

For full policies and contracting information, see the [`LEGAL/`](LEGAL/) directory in this repository. 

### Core Expertise

#### Machine Learning Infrastructure
- **Intelligent training pipeline** with automated target ranking and selection
- **Automated leakage detection** with pre-training scans and auto-fix system
- **Config-driven orchestration** with 9 centralized config files
- End-to-end pipeline architecture with leakage-safe validation
- Configurable model zoos (20+ families: LightGBM, XGBoost, Random Forest, Neural Networks, etc.)
- GPU-optimized training with CUDA support

#### Cross-Sectional ML Engineering
- **Automated target predictability assessment** with multi-model evaluation
- **Feature/target schema system** with ranking vs. training mode rules
- Walk-forward analysis frameworks with time-purged cross-validation
- Strict leakage auditing with sentinel tests (shifted-target, entity-holdout, randomized-time)
- Research-oriented data pipelines with Polars optimization
- Feature engineering at scale with registry-based temporal rules  

#### High-Performance Computing
- Multi-node and GPU workflows  
- **NVLink-ready architecture** (planned: device group abstraction, multi-GPU scheduling)
- System-level throughput optimization  
- Parallel experiment execution
- GPU topology detection and logging  

#### Systems Architecture
- Complete Single Source of Truth (SST) - reproducible, config-driven workflows with zero hardcoded values  
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
├── DATA_PROCESSING/       (Pipelines & feature engineering)
├── TRAINING/              (Model training & research workflows)
├── CONFIG/                (Configuration management system)
├── DOCS/                  (Technical documentation)
└── SCRIPTS/               (Utilities & tools)
```

**Note:** The codebase is Python-based with optimized data processing pipelines.

---------------------------------------------------------------------

## Professional Standards

- Defined scopes and clear communication  
- NDA-compliant handling of client datasets  
- High-quality, reproducible engineering with complete config centralization (SST) ensuring same config → same results  
- Minimal onboarding overhead for teams  
- SOW-structured engagements  

---------------------------------------------------------------------

## Contact
linkedin: https://www.linkedin.com/in/jennifer-lewis-3434b3315/

For enterprise licensing or organizational engagements:  
**jenn.lewis5789@gmail.com**
