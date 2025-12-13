# Fox ML Infrastructure — Strategic Roadmap

FoxML Core is the core research and training engine of Fox ML Infrastructure.

Direction and priorities for ongoing development. Timelines are aspirational and subject to change.

**For detailed status and technical roadmap:**
- [Detailed Roadmap](DOCS/02_reference/roadmap/README.md) – Per-date detailed roadmap with component status, limitations, and priorities
- [Known Issues & Limitations](DOCS/02_reference/KNOWN_ISSUES.md) – Features that don't work yet or have limitations

---

## Development Philosophy

FoxML Core is maintained with an **enterprise reliability mindset**:
* Critical issues are fixed immediately.
* Documentation and legal compliance are treated as first-class deliverables.
* Stability always precedes new features.
* GPU acceleration, orchestration, and model tooling will remain backwards-compatible.
* **UI integration approach** — UI is intentionally decoupled from the core. FoxML Core provides Python programmatic interfaces and integration points so teams can plug into their existing dashboards, monitoring tools, or UX layers. *As always, priorities are subject to change based upon client demand.*

---

## Current Status — Winter 2025

### What Works Today ✅

**You can use these capabilities right now:**

* ✅ **Full TRAINING Pipeline** — Complete one-command workflow: target ranking → feature selection → training plan generation → model training
* ✅ **GPU Acceleration** — Target ranking and feature selection now use GPU for LightGBM, XGBoost, and CatBoost (NEW 2025-12-12)
* ✅ **Training Routing & Planning System** — Config-driven routing decisions, automatic training plan generation, 2-stage training pipeline (CPU → GPU)
* ✅ **GPU-Accelerated Model Training** — Train all 20 model families with GPU acceleration where available
* ✅ **Centralized YAML Configuration** — Complete Single Source of Truth (SST) system with structured configs for experiments, model families, system parameters, decision policies, stability analysis, and GPU settings (NEW 2025-12-12)
* ✅ **Experiment Configuration System** — Reusable experiment configs with auto target discovery (NEW 2025-12-12)
* ✅ **Decision-Making System** — Automated decision policies with configurable thresholds (EXPERIMENTAL)
* ✅ **Bayesian Patch Policy** — Thompson sampling for adaptive config tuning (EXPERIMENTAL)
* ✅ **Reproducibility Tracking** — End-to-end reproducibility tracking with STABLE/DRIFTING/DIVERGED classification
* ✅ **Leakage Detection & Auto-Fix** — Pre-training leak detection with automatic feature exclusion and Final Gatekeeper
* ✅ **Complete Documentation & Legal** — Full 4-tier docs hierarchy + enterprise legal package

**This is production-grade ML infrastructure, not a prototype.**

---

## Core System Status

### ✅ Fully Operational

* **TRAINING Pipeline** — Fully operational. Intelligent training framework integrated and working. End-to-end testing underway.
* **GPU Acceleration** — Enabled for target ranking and feature selection (LightGBM, XGBoost, CatBoost). All settings config-driven from `gpu_config.yaml`.
* **Single Source of Truth (SST)** — Complete config centralization. All hyperparameters, seeds, thresholds, and GPU settings load from YAML.
* **Reproducibility Tracking** — End-to-end tracking across ranking, feature selection, and training with trend analysis.
* **Model Parameter Sanitization** — Shared utility prevents parameter passing errors.
* **Target Ranking & Selection** — Integrated and operational with auto-discovery support.
* **Documentation** — 4-tier hierarchy complete with cross-linking.

### ⚠️ Experimental (Use with Caution)

* **Decision-Making System** — Under active testing. Use `dry_run` mode until validated.
* **Stability Analysis** — Under active testing. Thresholds may need adjustment.

**For complete list of limitations, experimental features, and troubleshooting:**
- [Known Issues & Limitations](DOCS/02_reference/KNOWN_ISSUES.md) — Features that don't work yet, experimental features, and known limitations

---

## Development Priorities

### Immediate (Next 1-2 Weeks)
1. Complete end-to-end testing of full pipeline
2. Validate and tune decision system
3. Verify GPU acceleration is working correctly
4. Keep documentation in sync

### Short-Term (Next Month)
1. Performance optimization (parallel execution)
2. Multi-GPU support
3. Advanced routing strategies
4. Dataset migration tools

### Long-Term (Next Quarter)
1. Enhanced UI integration points
2. Advanced analytics and reporting
3. Scalability improvements
4. Additional model families

---

## Phase Status

### Phase 0 — Stability & Documentation ✅
**Status**: Complete

### Phase 1 — Intelligent Training Framework ✅
**Status**: Fully operational. End-to-end testing underway.

### Phase 2 — Advanced Features 🔄
**Status**: In progress. Decision-making and stability analysis (experimental).

---

**For detailed status, limitations, and technical roadmap:**
- [Detailed Roadmap](DOCS/02_reference/roadmap/README.md)
- [Known Issues & Limitations](DOCS/02_reference/KNOWN_ISSUES.md)
- [Changelog](CHANGELOG.md) – Recent changes and fixes
