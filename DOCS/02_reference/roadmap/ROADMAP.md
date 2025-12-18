# Fox ML Infrastructure — Strategic Roadmap

FoxML Core is the core research and training engine of Fox ML Infrastructure.

Direction and priorities for ongoing development. Timelines are aspirational and subject to change.

**For detailed status and technical roadmap:**
- [Detailed Roadmap](README.md) – Per-date detailed roadmap with component status, limitations, and priorities
- [Known Issues & Limitations](../KNOWN_ISSUES.md) – Features that don't work yet or have limitations

---

## Development Philosophy

FoxML Core is maintained with an **enterprise reliability mindset**:
* Critical issues are fixed immediately.
* Documentation and legal compliance are treated as first-class deliverables.
* Stability always precedes new features.
* GPU acceleration, orchestration, and model tooling will remain backwards-compatible.
* **UI integration approach** — UI is intentionally decoupled from the core. FoxML Core provides Python programmatic interfaces and integration points so teams can plug into their existing dashboards, monitoring tools, or UX layers.

---

## Current Status — Winter 2025

### What Works Today ✅

**You can use these capabilities right now:**

* ✅ **Full TRAINING Pipeline** — Complete one-command workflow: target ranking → feature selection → training plan generation → model training
* ✅ **GPU Acceleration** — Target ranking, feature selection, and model training use GPU (LightGBM, XGBoost, CatBoost)
* ✅ **Training Routing & Planning System** — Config-driven routing decisions, automatic training plan generation, 2-stage training pipeline (CPU → GPU)
* ✅ **Centralized YAML Configuration** — Single Source of Truth (SST) system with structured configs
* ✅ **Experiment Configuration System** — Reusable experiment configs with auto target discovery
* ✅ **Reproducibility Tracking** — End-to-end reproducibility tracking with STABLE/DRIFTING/DIVERGED classification
* ✅ **Leakage Detection & Auto-Fix** — Pre-training leak detection with automatic feature exclusion
* ✅ **Documentation & Legal** — 4-tier docs hierarchy + commercial legal package

**This is research-grade ML infrastructure under active development.**

---

## Core System Status

### ✅ Fully Operational

* **TRAINING Pipeline** — Operational. Intelligent training framework integrated and working.
* **GPU Acceleration** — Enabled for target ranking, feature selection, and model training. All settings config-driven.
* **Single Source of Truth (SST)** — Config centralization. All hyperparameters, seeds, thresholds, and GPU settings load from YAML.
* **Reproducibility Tracking** — End-to-end tracking across ranking, feature selection, and training with trend analysis.
* **Target Ranking & Selection** — Integrated and operational with auto-discovery support.
* **Documentation** — 4-tier hierarchy complete with cross-linking.

### ⚠️ Experimental (Use with Caution)

* **Decision-Making System** — Under active testing. Use `dry_run` mode until validated.
* **Stability Analysis** — Under active testing. Thresholds may need adjustment.

**For complete list of limitations, experimental features, and troubleshooting:**
- [Known Issues & Limitations](../KNOWN_ISSUES.md) — Features that don't work yet, experimental features, and known limitations

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
- [Detailed Roadmap](README.md)
- [Known Issues & Limitations](../KNOWN_ISSUES.md)
- [Changelog](../../../CHANGELOG.md) – Recent changes and fixes
