# FoxML Core — Licensing Roadmap & Strategic Boundaries

**Last Updated:** 2025-12-16  
**Status:** Active Strategic Plan

---

## Executive Summary

FoxML Core is **source-available** with a clear licensing model. This document outlines the strategic roadmap for license evolution and defines what remains open vs. what will be held back in future versions.

**Key Principle:** We protect through **authority signaling and clear boundaries**, not through disappearance.

---

## Current State (v1.x)

### ✅ Fully Open (Source-Available)

**What's visible:**
- Core training infrastructure
- Feature engineering pipelines
- Model evaluation frameworks
- Reproducibility systems
- Leakage detection and safety mechanisms
- Cross-sectional ranking systems
- Configuration management

**What this enables:**
- Academic researchers can study, learn, and build on the architecture
- Organizations can evaluate the approach before committing
- The community can validate the technical approach
- Contributors can understand the system deeply

**What this does NOT enable:**
- Commercial use without a license
- Hosted service resale
- Derivative monetization
- Competing infrastructure services

---

## Strategic Boundaries (v1.x → v2.0+)

### 🎯 Architecture Asymmetry (Graduated Close Model)

**Core remains visible. Production glue will be held back.**

#### What Stays Open (Core Platform)
- Training infrastructure
- Feature engineering
- Model evaluation
- Safety mechanisms
- Core algorithms and methodologies

#### What Will Be Held Back (v2.0+)
- **Orchestration glue** — Production deployment recipes, scaling patterns
- **Production presets** — Optimized configurations for specific environments
- **Validation harnesses** — Enterprise-grade testing and compliance tooling
- **Integration patterns** — Turnkey connectors and adapters
- **Performance optimizations** — Production-grade performance tuning

**Rationale:** People can *see* it works without being able to turnkey it. The core remains open for validation; the production layer requires engagement.

---

## Version Roadmap

### v1.x (Current) — Last Fully Open Window

**Status:** ✅ **Fully source-available**

- All core components visible
- Academic research free
- Commercial use requires license
- Enforcement active

**Timeline:** v1.x will remain fully open through the end of 2025 and into early 2026, subject to milestone completion.

### v2.0+ (Future) — Graduated Close

**Status:** 🔒 **Core open, production layer closed**

- Core training infrastructure: **Open**
- Production orchestration: **Closed** (commercial license required)
- Enterprise tooling: **Closed** (commercial license required)
- Scaling recipes: **Closed** (commercial license required)

**Timeline:** v2.0 will be announced with 90 days' notice before implementation.

**Transition Policy:**
- v1.x will remain available under current license terms
- v2.0+ will introduce the graduated close model
- Existing commercial licensees will receive v2.0+ access as part of their license
- Academic researchers will retain access to core components

---

## Enforcement & Authority Signaling

### Active Monitoring

**We monitor for:**
- Unauthorized commercial use
- Hosted service resale
- Derivative monetization
- Competing infrastructure services
- License violations

**Detection methods:**
- Public code audits
- GitHub fork analysis
- Compliance audits
- Third-party reports
- Self-reporting

### Enforcement Actions

**Violations will be pursued:**
- Copyright infringement claims (up to $150,000 per work under U.S. Copyright Act)
- Injunctive relief
- Monetary damages
- Recovery of attorneys' fees and costs

**Self-reporting:** Contact jenn.lewis5789@gmail.com within 30 days of discovery for reduced penalties.

---

## Communication Strategy

### Calm, Deliberate Transitions

**We communicate changes as:**
- ✅ Evolution and maturity
- ✅ Strategic business decisions
- ✅ Protection of legitimate users

**We do NOT communicate as:**
- ❌ Panic or fear
- ❌ Reactive responses
- ❌ Sudden changes

### Timeline for v2.0 Announcement

**When v2.0 is ready:**
1. **90 days before implementation:** Announce roadmap and timeline
2. **60 days before:** Provide detailed migration guide
3. **30 days before:** Final notice and transition support
4. **Implementation:** Graduated close takes effect

**This frames it as evolution, not panic.**

---

## What This Means for Users

### Academic Researchers

**v1.x:** Full access, no changes  
**v2.0+:** Core components remain open, production layer requires commercial license

### Commercial Users

**v1.x:** Commercial license required  
**v2.0+:** Commercial license required (includes production layer access)

### Contributors

**v1.x:** Contributions welcome, subject to CLA  
**v2.0+:** Contributions welcome, subject to CLA (core components)

---

## Key Principles

1. **We protect through authority, not disappearance**
   - Clear boundaries
   - Active enforcement
   - Deliberate communication

2. **We maintain core openness for validation**
   - Academic research remains free
   - Core components stay visible
   - Community can validate approach

3. **We hold back production glue**
   - Orchestration recipes
   - Scaling patterns
   - Enterprise tooling

4. **We communicate deliberately**
   - 90-day notice for major changes
   - Clear rationale
   - Migration support

---

## Contact

**Questions about this roadmap:**
📧 jenn.lewis5789@gmail.com  
Subject: `Roadmap Inquiry`

**License compliance:**
📧 jenn.lewis5789@gmail.com  
Subject: `License Compliance Inquiry`

---

**This is a living document. Updates will be communicated with 90 days' notice for major changes.**

