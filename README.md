# FoxML Core — ML Cross-Sectional Infrastructure

## License (TL;DR)

**FoxML Core is source-available. Free for personal use and non-commercial academic research. Organizational use requires a paid commercial license.**

### Decision Rule

**If you're an organization (company, university, lab, or any legal entity), you need a license — except:**
- Pure non-commercial academic research at qualifying tax-exempt institutions (see definitions below)

**Evaluation:** Organizations can use a **30-day $0 evaluation** (strict limits: non-production, no client deliverables). Continued evaluation requires a **paid Pilot ($35k, credited to Year 1)**.

**Pricing:** Starts at **$120,000/year** per desk/team. See [`LEGAL/QUICK_REFERENCE.md`](LEGAL/QUICK_REFERENCE.md) for complete pricing.

📧 **jenn.lewis5789@gmail.com** | Subject: `FoxML Core Commercial Licensing`

**For complete licensing information:** [`LEGAL/QUICK_REFERENCE.md`](LEGAL/QUICK_REFERENCE.md) (one-page summary) | [`COMMERCIAL_LICENSE.md`](COMMERCIAL_LICENSE.md) (full terms)

---

### Definitions

**Personal use:** Individual, non-commercial use by a natural person, not for any business purpose. Excludes sole proprietors, freelancers, contractors, or any use in connection with business activities or client work.

**Non-commercial academic research:** Research conducted at a qualifying tax-exempt educational institution that:
- Is not funded by, sponsored by, or integrated into commercial operations
- Does not generate revenue or support revenue-generating activities
- Is conducted solely for academic or research purposes
- Does not involve internal tooling, operational systems, or client deliverables

**Organizational use:** Any use by or for a legal entity (company, university, lab, etc.), including:
- Internal evaluation, pilots, PoCs, dev/staging, or production use
- Use supporting revenue, operations, services, or internal tooling
- Use by employees, contractors, interns, or affiliates in scope of their work
- Institutional deployment (even at non-profit universities)

**Note:** Unauthorized organizational use may constitute copyright infringement. This is not legal advice; see `LICENSE` and `COMMERCIAL_LICENSE.md` for complete terms.

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

**For detailed capabilities:** See [Architecture Overview](DOCS/00_executive/ARCHITECTURE_OVERVIEW.md)

---

## Domain Focus

FoxML Core is **general-purpose ML cross-sectional infrastructure** for panel data and time-series workflows. The architecture provides domain-agnostic primitives with built-in safeguards (leakage detection, temporal validation, feature registry systems).

**Domain Applications:** Financial time series, IoT sensor data, healthcare, clickstream analytics, and any panel data with temporal structure.

**For detailed domain information:** See [Architecture Overview](DOCS/00_executive/ARCHITECTURE_OVERVIEW.md)

---

## Intended Use

### Appropriate Use Cases
- Research and experimentation (personal or non-commercial academic)
- ML workflow and architecture study
- Non-commercial academic analysis at qualifying institutions
- Internal engineering reference (with commercial license)
- Benchmarking and systems review (with commercial license)

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

---

## Quick FAQ

**Q: We're a university lab—do we need a license?**  
A: It depends. **Pure non-commercial academic research** at a qualifying tax-exempt institution: no license needed. **Institutional use, sponsored/industry work, or internal tooling**: yes, you need a commercial license. See definitions above.

**Q: Can I use it for free?**  
A: Only for personal use (individual, non-commercial) or non-commercial academic research at qualifying institutions. Any organizational use requires a commercial license.

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

## Support Development

If you appreciate this work and want to support ongoing development, you can contribute here: [Support Development](https://www.paypal.com/ncp/payment/4BHHB4YCVVTH8)

![Mountain Dew Fund QR Code](DOCS/images/mountain_dew_fund_qrcode.png)

Contributions are voluntary and provided as-is. They do not confer ownership, governance rights, priority support, or any implied agreement beyond appreciation for the work.
