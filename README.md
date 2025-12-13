# FoxML Core — ML Cross-Sectional Infrastructure

> **⚠️ ACTIVE DEVELOPMENT — EXPECT BREAKING CHANGES**  
> This project is under **heavy active development**. Breaking changes may occur without notice. APIs, configuration schemas, directory structures, and file formats may change between commits. Use at your own risk in production environments. See [ROADMAP.md](ROADMAP.md) for current status and known issues.

> **📝 See [CHANGELOG.md](CHANGELOG.md) for recent technical and compliance changes.**

FoxML Core is an ML infrastructure stack for cross-sectional and panel data for any machine learning applications. It provides a config-driven ML pipeline architecture designed for ML infra teams, data scientists, and researchers.

**⚠️ Development Status:** This project is under **heavy active development**. Breaking changes may occur without notice. See [ROADMAP.md](ROADMAP.md) for current status, known issues, and planned fixes.

> **Note:** Personal / academic use is AGPL-3.0. **Company / production use requires a commercial license.**

> **📊 Testing & Development:** All testing, validation, and development work is performed using **5-minute interval data**. The software supports various data intervals, but all tests, benchmarks, and development workflows use 5-minute bars as the standard reference.

Developed and maintained by **Jennifer Lewis**  
Independent Contractor • ML Engineering • Cross-Sectional ML Systems • Systems Architecture

---

## Quick Overview

FoxML Core provides:

- **Intelligent training pipeline** with automated target ranking and feature selection
- **GPU acceleration** for target ranking, feature selection, and model training (LightGBM, XGBoost, CatBoost)
- **Config-based usage** with minimal command-line arguments
- **Leakage detection system** with pre-training leak detection and auto-fix
- **Single Source of Truth (SST)** config system - all 20 model families use config-driven hyperparameters
- **Multi-model training systems** with 20+ model families (GPU-accelerated)
- **Reproducibility tracking** with end-to-end reproducibility verification

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
- Institutional or academic analysis  
- Internal engineering reference  
- Benchmarking and systems review  

### Not Appropriate For
- Commercial use without a license  
- Unmodified production deployment without proper testing and validation

**FoxML Core provides ML infrastructure and architecture, not domain-specific applications or pre-built solutions.**

---

## Licensing & Commercial Use

> ⚠️ **BUSINESS USE REQUIRES A COMMERCIAL LICENSE**
>
> - If you're running this inside a company or fund, you almost certainly need a commercial license.
> - Non-commercial academic / personal research is allowed under AGPL-3.0 (see `LICENSE`).
> - Full details and pricing: [`LEGAL/SUBSCRIPTIONS.md`](LEGAL/SUBSCRIPTIONS.md).

FoxML Core is released under the **AGPL-3.0** license. Personal / academic research uses AGPL. Organizations using FoxML Core in production or revenue-generating environments should obtain a commercial license.

**If you are using FoxML Core inside any business or organization, assume you need a commercial license.**

### Commercial License

Commercial licensing gives your organization the right to:
- Use FoxML Core for **commercial and internal production use**
- Integrate FoxML into proprietary systems without AGPL disclosure obligations
- Get access to **commercial license terms** and optional support / integration work

**Pricing Structure:**

**Paid Pilots (Required for Commercial Use):**
- **Pilot (30 days):** $15,000–$30,000 — 1 environment, limited scope, async support
- **Pilot+ (60–90 days):** $35,000–$90,000 — 2 environments, onboarding calls, defined milestones
- **Credit:** 50–100% of pilot fee applies toward first-year license if converted

**Annual License Tiers** (based on using team/desk size, not total company headcount):
- **Team** (1–5 users, 1 environment): $75,000/year
- **Desk** (6–20 users, up to 2 environments): $150,000–$300,000/year
- **Division** (21–75 users, up to 3 environments): $350,000–$750,000/year
- **Enterprise** (76–250 users, multi-environment): $800,000–$2,000,000/year
- **>250 users / multi-region / regulated bank:** Custom pricing

**Support Options** (annual add-ons):
- **Standard Support** (included): Email/issues, best-effort response, no SLA
- **Business Support:** +$25,000/year — Response targets, business hours coverage
- **Enterprise Support:** +$75,000–$150,000/year — Priority response, scheduled calls
- **Premium Support:** +$200,000–$400,000/year — Extended hours, direct engineering access (availability limited)

**Professional Services** (one-time):
- **Onboarding/Integration:** $25,000–$150,000 — Training, architecture review, initial setup
- **On-Prem/High-Security Deployment:** $150,000–$500,000+ — Custom deployment and security configuration

**Important:** Licensing is scoped to the business unit/desk using the software, not the parent company's total headcount. This makes pricing accessible for teams and desks within larger organizations.

**To begin the licensing process:**
📧 **jenn.lewis5789@gmail.com**  
Subject: `FoxML Core Commercial Licensing`

**For complete licensing information:**
- [`LEGAL/SUBSCRIPTIONS.md`](LEGAL/SUBSCRIPTIONS.md) - Detailed tiers and pricing
- [`LEGAL/COMMERCIAL_USE.md`](LEGAL/COMMERCIAL_USE.md) - Commercial license terms
- [`COMMERCIAL_LICENSE.md`](COMMERCIAL_LICENSE.md) - Full commercial license text

---

## Getting Started

**New users start here:**
- **[Quick Start](DOCS/00_executive/QUICKSTART.md)** - Get running in 5 minutes
- **[Getting Started](DOCS/00_executive/GETTING_STARTED.md)** - Complete onboarding guide
- **[Architecture Overview](DOCS/00_executive/ARCHITECTURE_OVERVIEW.md)** - System at a glance

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

## Enterprise Partnerships

FoxML Core is available for commercial licensing and custom development partnerships for organizations requiring ML, quantitative, or HPC systems.

**Core Expertise:**
- Machine Learning Infrastructure
- Cross-Sectional ML Engineering
- High-Performance Computing
- Systems Architecture

**For full policies and contracting information:** See the [`LEGAL/`](LEGAL/) directory in this repository.

---

## Reporting Issues

For bug reports, feature requests, or technical issues:

- **GitHub Issues**: [Open an issue](https://github.com/Fox-ML-infrastructure/FoxML_Core/issues) (preferred for bug reports and feature requests)
- **Email**: jenn.lewis5789@gmail.com (for security issues, sensitive bugs, or private inquiries)

For commercial licensing or organizational engagements:  
**jenn.lewis5789@gmail.com**
