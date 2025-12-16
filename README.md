# FoxML Core — ML Cross-Sectional Infrastructure

## License

**FoxML Core is source-available. Free for personal use and non-commercial academic research. Organizational use requires a paid commercial license.**

- ✅ **Free use:** Personal use (individual, non-commercial) and non-commercial academic research at qualifying tax-exempt institutions
- 💰 **Commercial license required for:** Any organizational use, including companies, universities, labs, or any legal entity

**Examples requiring a commercial license:**
- Any use by or for a company, university, lab, or organization
- Freelancers/contractors using it for client work
- Sole proprietors using it for business operations
- Employees using it in the scope of their work
- Any evaluation, pilot, or PoC done by an organization (after 30-day evaluation period)

📧 **jenn.lewis5789@gmail.com** | Subject: `FoxML Core Commercial Licensing`

**Organizational use requires a paid license.** Commercial licensing is per team/desk and starts at $120,000/year. Paid pilot required for organizational evaluation.

**FoxML Core is source-available: free for personal use and non-commercial academic research. Organizational use requires a paid commercial license. This keeps the project sustainable and enables continued development.**

**→ [Commercial Licensing](DOCS/02_reference/commercial/COMMERCIAL.md)** | **→ [License Terms](LICENSE)** | **→ [Commercial License](COMMERCIAL_LICENSE.md)**

---

> **⚠️ ACTIVE DEVELOPMENT — EXPECT BREAKING CHANGES**  
> This project is under **heavy active development**. Breaking changes may occur without notice. APIs, configuration schemas, directory structures, and file formats may change between commits. Use at your own risk in production environments. See [ROADMAP.md](DOCS/02_reference/roadmap/ROADMAP.md) for current status and known issues.

> **🎯 Version 1.0 Definition:** See [FOXML_1.0_MANIFEST.md](DOCS/00_executive/product/FOXML_1.0_MANIFEST.md) for the capability boundary that defines what constitutes FoxML 1.0.

> **📝 See [CHANGELOG.md](CHANGELOG.md) for recent technical and compliance changes.**

---

## ⚠️ **ORGANIZATIONAL USE REQUIRES A PAID LICENSE**

**If an organization uses this, they need a license.**

**Organizational Use includes:**
- Any use by or for a company, university, lab, or other organization
- Any internal evaluation, pilots, PoCs, dev/staging, or production use
- Any use supporting revenue, operations, services, or internal tooling
- Any evaluation/pilot done by an organization
- Any use by employees, contractors, interns, or affiliates in the scope of their work

**Free use is limited to:**
- Personal use (individual, non-commercial, not for any business purpose)
- Non-commercial academic research at qualifying tax-exempt institutions (see `LICENSE` for full definition)

**Unauthorized organizational use is copyright infringement.**

📧 **jenn.lewis5789@gmail.com** | Subject: `FoxML Core Commercial Licensing`

**Business use requires a paid license.** Commercial licensing is offered via paid Pilot + Annual License. Typical deployments are enterprise engagements with onboarding, integration support, and maintained release access. Fees are specified in Ordering Documents. See [`LEGAL/SUBSCRIPTIONS.md`](LEGAL/SUBSCRIPTIONS.md) for details.

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

FoxML Core is **source-available** with a commercial license required for organizational use.

- **Free use**: Personal use and non-commercial academic research (see `LICENSE`)
- **Commercial License**: Required for any organizational use (see `COMMERCIAL_LICENSE.md`)

**If you are using FoxML Core inside any business or organization, you need a commercial license.**

Source code is publicly available for inspection and study, but organizational use requires a paid license.

### Commercial License

Commercial licensing gives your organization the right to:
- Use FoxML Core for **commercial and internal production use**
- Integrate FoxML into proprietary systems without AGPL disclosure obligations
- Get access to **commercial license terms** and optional support / integration work

**Pricing:**

**Commercial licenses start at $120,000/year per desk/team, depending on team size and deployment scope.**

For detailed pricing information, please contact the maintainer:

📧 **jenn.lewis5789@gmail.com**  
🔗 [LinkedIn](https://www.linkedin.com/in/jennifer-l-3434b3315)

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
- [`LEGAL/QUICK_REFERENCE.md`](LEGAL/QUICK_REFERENCE.md) - ⭐ **Quick one-page summary** (start here)
- [`DOCS/02_reference/commercial/COMMERCIAL.md`](DOCS/02_reference/commercial/COMMERCIAL.md) - Pricing & packages
- [`COMMERCIAL_LICENSE.md`](COMMERCIAL_LICENSE.md) - Full commercial license text
- [`LEGAL/SUBSCRIPTIONS.md`](LEGAL/SUBSCRIPTIONS.md) - Detailed licensing process

### Quick FAQ

**Q: We're a university lab—do we need a license?**  
A: If it's institutional use, sponsored/industry work, or internal tooling: yes, you need a commercial license. Pure non-commercial academic research at a qualifying tax-exempt institution: no license needed (see `LICENSE` for full definition).

**Q: Can I use it for free?**  
A: Only for personal use (individual, non-commercial) or non-commercial academic research at qualifying institutions. Any organizational use requires a commercial license.

**Q: What if I just want to evaluate it?**  
A: Organizations can evaluate for 30 days without a paid license (see `COMMERCIAL_LICENSE.md`). Continued use after evaluation requires a commercial license.

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
