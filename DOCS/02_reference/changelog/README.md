# Changelog Index

This directory contains detailed per-day changelogs for FoxML Core. For the lightweight root changelog, see [CHANGELOG.md](../../../CHANGELOG.md).

## 2025

### December

- **2025-12-18 (Code Modularization)** — Major code refactoring: Split 7 large files (2,000-6,800 lines) into modular components, created 23 new utility/module files, reorganized utils folder into domain-specific subdirectories, centralized common utilities (file_utils, cache_manager, config_hashing, etc.), fixed all import errors, maintained full backward compatibility. Total: 103 files changed, ~2,000+ lines extracted.
  → [View](2025-12-18-code-modularization.md)

- **2025-12-17 (Metric Deltas in Diff Artifacts)** — Fixed empty `metric_deltas` issue. Implemented 3-tier reporting (summary, structured deltas, full metrics), z-score noise detection, impact classification, and proper separation of nondeterminism from regression. All numeric metrics now captured and deltas always computed.
  → [View](2025-12-17-metric-deltas-in-diff-artifacts.md)

- **2025-12-17 (Training Pipeline Audit Fixes)** — Fixed 10 critical contract breaks across family IDs, routing, plan consumption, feature schema, and counting/tracking. Key fixes: family normalization, reproducibility tracking, preflight filtering, routing plan respect, symbol-specific route, feature pipeline threshold and diagnostics.
  → [View](2025-12-17-training-pipeline-audit-fixes.md)

- **2025-12-16 (Feature Selection Structure)** — Organized feature selection outputs to match target ranking layout. Eliminated scattered files and nested REPRODUCIBILITY directories.
  → [View](2025-12-16-feature-selection-structure.md)

- **2025-12-15 (Consolidated)** — Metrics system rename, seed tracking fixes, feature selection improvements, CatBoost GPU fixes, privacy documentation updates.  
  → [View](2025-12-15-consolidated.md)
  
- **2025-12-15 (CatBoost GPU Fixes)** — Fixed CatBoost GPU mode requiring Pool objects, sklearn clone compatibility, and missing feature importance output. CatBoost GPU training now works correctly and feature importances are saved to results directory.  
  → [View](2025-12-15-catboost-gpu-fixes.md)
  
- **2025-12-15 (Metrics Rename)** — Renamed telemetry to metrics throughout codebase. All metrics stored locally - no user data collection.  
  → [View](2025-12-15-metrics-rename.md)

- **2025-12-14 (IP Assignment Agreement Signed)** — IP Assignment Agreement signed, legally assigning all IP from individual to Fox ML Infrastructure LLC. ✅ Legally effective.  
  → [View](2025-12-14-ip-assignment-signed.md)

- **2025-12-14 (Execution Modules Added)** — ALPACA_trading and IBKR_trading modules added with compliance framework, documentation organization, copyright headers. ⚠️ ALPACA has minor issues, IBKR is untested  
  → [View](2025-12-14-execution-modules.md)

- **2025-12-14 (Enhanced Drift Tracking)** — Fingerprints (git commit, config hash, data fingerprint), drift tiers (OK/WARN/ALERT), critical metrics tracking, sanity checks, Parquet files for queryable data  
  → [View](2025-12-14-drift-tracking-enhancements.md)

- **2025-12-14 (Telemetry System)** — Sidecar-based telemetry with view isolation, hierarchical rollups (cohort → view → stage), baseline key format for drift comparison, config-driven behavior, Parquet files  
  → [View](2025-12-14-telemetry-system.md)

- **2025-12-14 (Feature Selection and Config Fixes)** — Fixed UnboundLocalError for np (11 model families), missing import, unpacking error, routing diagnostics, experiment config loading, target exclusion, lookback enforcement  
  → [View](2025-12-14-feature-selection-and-config-fixes.md)

- **2025-12-14 (Look-Ahead Bias Fixes)** — Rolling windows exclude current bar, CV-based normalization, pct_change verification, feature renaming, symbol-specific logging, feature selection bug fix  
  → [View](2025-12-14-lookahead-bias-fixes.md)

- **2025-12-13 (SST Enforcement Design)** — EnforcedFeatureSet contract, type boundary wiring, boundary assertions, no rediscovery rule, full coverage across all training paths  
  → [View](2025-12-13-sst-enforcement-design.md)

- **2025-12-13 (Single Source of Truth)** — Eliminated split-brain in lookback computation, POST_PRUNE invariant check, _Xd pattern inference, readline library conflict fix  
  → [View](2025-12-13-single-source-of-truth.md)

- **2025-12-13 (Fingerprint Tracking)** — Fingerprint Tracking System, LookbackResult Dataclass, Explicit Stage Logging, Leakage Canary Test  
  → [View](2025-12-13-fingerprint-tracking.md)

- **2025-12-13 (Feature Selection Unification)** — Shared Ranking Harness, Comprehensive Hardening, Same Output Structure, Config-Driven Setup  
  → [View](2025-12-13-feature-selection-unification.md)

- **2025-12-13 (Duration System)** — Generalized Duration Parsing System, Lookback Detection Precedence Fix, Documentation Review, Non-Auditable Status Markers  
  → [View](2025-12-13-duration-system.md)

- **2025-12-13** — Config Path Consolidation, Config Trace System, Max Samples Fix, Output Directory Binning Fix  
  → [View](2025-12-13.md)

- **2025-12-12** — Trend Analysis System Extension (Feature Selection), Cohort-Aware Reproducibility System, RESULTS Directory Organization, Integrated Backups, Enhanced Metadata  
  → [View](2025-12-12.md)

- **2025-12-11** — Training Routing System, Reproducibility Tracking, Leakage Fixes, Interval Detection, Param Sanitization, Cross-Sectional Stability  
  → [View](2025-12-11.md)

- **2025-12-10** — SST Enforcement, Determinism System, Config Centralization  
  → [View](2025-12-10.md)

- **General** — Intelligent Training Framework, Leakage Safety Suite, Configuration System, Documentation  
  → [View](general.md)

---

## Documentation Audits

Quality assurance audits and accuracy checks (publicly available for transparency):

- **[Documentation Accuracy Check](../../00_executive/audits/DOCS_ACCURACY_CHECK.md)** - Accuracy audit results and fixes (2025-12-13)
- **[Unverified Claims Analysis](../../00_executive/audits/DOCS_UNVERIFIED_CLAIMS.md)** - Claims without verified test coverage (2025-12-13)
- **[Marketing Language Removal](../../00_executive/audits/MARKETING_LANGUAGE_REMOVED.md)** - Marketing terms removed for accuracy (2025-12-13)
- **[Dishonest Statements Fixed](../../00_executive/audits/DISHONEST_STATEMENTS_FIXED.md)** - Final pass fixing contradictions and overselling (2025-12-13)

---

## Navigation

- [Root Changelog](../../../CHANGELOG.md) - Executive summary
- [Documentation Index](../../INDEX.md) - Complete documentation navigation
