# FoxML Core — ML Cross-Sectional Infrastructure

> **⚠️ ACTIVE DEVELOPMENT — EXPECT BREAKING CHANGES**  
> This project is under **heavy active development**. Breaking changes may occur without notice. APIs, configuration schemas, directory structures, and file formats may change between commits. Use at your own risk in production environments. See [ROADMAP.md](ROADMAP.md) for current status and known issues.

> **📝 See [CHANGELOG.md](CHANGELOG.md) for recent technical and compliance changes.**

FoxML Core is an ML infrastructure stack for cross-sectional and panel data for any machine learning applications. It provides a config-driven ML pipeline architecture designed for ML infra teams, data scientists, and researchers.

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
- A toaster ding will be added for the completion noise. 

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

**Pricing:**

**For pricing information, please contact the maintainer:**

📧 **jenn.lewis5789@gmail.com**

All commercial licensing begins with a paid pilot. Pilot options and annual license tiers are available based on your team size, deployment requirements, and usage needs.

**Support Options:**
- **Standard Support** (included): Email/issues, best-effort response, no SLA
- **Business Support:** Response targets, business hours coverage (contact for pricing)
- **Enterprise Support:** Priority response, scheduled calls (contact for pricing)
- **Premium Support:** Extended hours, direct engineering access (contact for pricing, availability limited)

**Professional Services:**
- **Onboarding/Integration:** Training, architecture review, initial setup (contact for pricing)
- **On-Prem/High-Security Deployment:** Custom deployment and security configuration (contact for pricing)

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

---

## Support Development

If you appreciate this work and want to support ongoing development, you can contribute here: [Support Development](https://www.paypal.com/ncp/payment/75FQ39PTAUNFQ)

![Coffee Fund QR Code](DOCS/images/coffee_fund_qrcode.png)

Contributions are voluntary and provided as-is. They do not confer ownership, governance rights, priority support, or any implied agreement beyond appreciation for the work.
