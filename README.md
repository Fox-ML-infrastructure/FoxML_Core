# FoxML Core — ML Cross-Sectional Infrastructure

## Licensing

**FoxML Core is dual-licensed:**
- **AGPL v3** for open research, experimentation, and transparency
- **Commercial license** required for proprietary, internal, or production deployments

### AGPL License (Free Use)

The AGPL v3 license is intended for:
- Individual researchers and developers
- Open-source contributors
- Academic use and research
- Evaluation and experimentation
- Projects that can comply with AGPL copyleft requirements

**AGPL Requirements:** If you modify FoxML Core and deploy it as a service (including internal services), you must publish your changes under AGPL v3 unless you have a commercial license.

### Commercial License (Required for Proprietary Use)

Commercial licenses are required for:
- Proprietary internal use
- Production trading systems
- Closed-source integrations
- Hosted services where you cannot comply with AGPL
- Organizations that need to modify and deploy without source disclosure obligations

**Evaluation:** Organizations can use a **30-day $0 evaluation** (strict limits: non-production, no client deliverables). Continued evaluation requires a **paid Pilot ($35k, credited to Year 1)**.

**Pricing:** Starts at **$120,000/year** per desk/team. See [`LEGAL/QUICK_REFERENCE.md`](LEGAL/QUICK_REFERENCE.md) for complete pricing.

📧 **jenn.lewis5789@gmail.com** | Subject: `FoxML Core Commercial Licensing`

**For complete licensing information:** [`LEGAL/QUICK_REFERENCE.md`](LEGAL/QUICK_REFERENCE.md) (one-page summary) | [`COMMERCIAL_LICENSE.md`](COMMERCIAL_LICENSE.md) (full terms)

**📋 Strategic Licensing Roadmap:** See [`LEGAL/ROADMAP_BOUNDARIES.md`](LEGAL/ROADMAP_BOUNDARIES.md) for licensing evolution strategy. **v1.x is the last fully open window** — core remains open, production layer will require commercial license in v2.0+.

---

> **⚠️ ACTIVE DEVELOPMENT — EXPECT BREAKING CHANGES**  
> This project is under **heavy active development**. Breaking changes may occur without notice. APIs, configuration schemas, directory structures, and file formats may change between commits. Use at your own risk in production environments. See [ROADMAP.md](DOCS/02_reference/roadmap/ROADMAP.md) for current status and known issues.

> **🎯 Version 1.0 Definition:** See [FOXML_1.0_MANIFEST.md](DOCS/00_executive/product/FOXML_1.0_MANIFEST.md) for the capability boundary that defines what constitutes FoxML 1.0.

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
- **Config-based usage** with minimal command-line arguments
- **Leakage detection system** with pre-training leak detection and auto-fix
- **Single Source of Truth (SST)** config system - all 20 model families use config-driven hyperparameters
- **Multi-model training systems** with 20+ model families (GPU-accelerated)
- **Reproducibility tracking** with end-to-end reproducibility verification
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
- Research and experimentation (AGPL or commercial license)
- ML workflow and architecture study
- Open-source projects (AGPL)
- Internal engineering reference (commercial license)
- Production deployments (commercial license)

### Not Appropriate For
- Commercial use without a license
- Unmodified production deployment without proper testing and validation

**FoxML Core provides ML infrastructure and architecture, not domain-specific applications or pre-built solutions.**

---

## Commercial Licensing

**Pricing:** Commercial licenses start at **$120,000/year** per desk/team, depending on team size and deployment scope.

**Evaluation Process:**
1. **30-day $0 evaluation** (strict limits: non-production, no client deliverables, no redistribution)
2. **Paid Pilot** ($35,000, 4–6 weeks, credited 100% to Year 1 if converted within 60 days)
3. **Annual License** (starts at $120k/year)

**What's Included:**
- Right to use internally + commercially
- Access to releases during term
- License to modify + integrate
- No copyleft obligations

**Support & Services:**
- Standard Support (included): Best-effort email/issues
- Business Support: $30,000–$60,000/year (response targets, scheduled cadence)
- Enterprise Support: $100,000+/year (priority + calls + escalation)
- Professional Services: $250–$400/hr or packaged blocks

**Important:** Licensing is scoped to the business unit/desk using the software, not the parent company's total headcount.

**To begin the licensing process:**
📧 **jenn.lewis5789@gmail.com**  
Subject: `FoxML Core Commercial Licensing`

**For complete licensing information:**
- [`LEGAL/QUICK_REFERENCE.md`](LEGAL/QUICK_REFERENCE.md) - ⭐ **Quick one-page summary** (start here)
- [`LEGAL/SUBSCRIPTIONS.md`](LEGAL/SUBSCRIPTIONS.md) - Detailed licensing process
- [`COMMERCIAL_LICENSE.md`](COMMERCIAL_LICENSE.md) - Full commercial license text
- [`LICENSE`](LICENSE) - Free use terms
- [`LEGAL/ROADMAP_BOUNDARIES.md`](LEGAL/ROADMAP_BOUNDARIES.md) - ⚠️ **Strategic licensing roadmap** (v1.x last fully open window)
- [`LEGAL/GRADUATED_CLOSE_MODEL.md`](LEGAL/GRADUATED_CLOSE_MODEL.md) - Graduated close model details

**⚠️ Contact Policy: Never send direct mail to any registered business address. All points of contact must be conducted via email (jenn.lewis5789@gmail.com) until stated otherwise.**

---

## Quick FAQ

**Q: Can I use it for free?**  
A: Yes, under AGPL v3 for open research, experimentation, and projects that can comply with AGPL copyleft requirements. If you deploy it as a service, you must publish your modifications unless you have a commercial license.

**Q: When do I need a commercial license?**  
A: Commercial licenses are required for proprietary internal use, production systems, closed-source integrations, or any deployment where you cannot comply with AGPL source disclosure requirements.

**Q: What if I just want to evaluate it?**  
A: Organizations can use a **30-day $0 evaluation** (strict limits: non-production, no client deliverables). Continued evaluation requires a **paid Pilot ($35k)**.

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
