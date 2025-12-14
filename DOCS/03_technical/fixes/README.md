# Fixes Documentation

Known issues, bug fixes, and migration notes.

## Contents

- **[Known Issues](KNOWN_ISSUES.md)** - Current issues and limitations
- **[Bug Fixes](BUG_FIXES.md)** - Fix history
- **[Migration Notes](MIGRATION_NOTES.md)** - Migration guide
- **[TensorFlow Executable Stack Fix](TENSORFLOW_EXECUTABLE_STACK_FIX.md)** - TensorFlow fix

## Recent Critical Fixes (2025-12-13)

### SST Enforcement Design Implementation

- **[SST Enforcement Design](2025-12-13-sst-enforcement-design.md)** - Complete SST enforcement design implementation with EnforcedFeatureSet contract, type boundary wiring, and boundary assertions

### Leakage Controls & Fingerprint Tracking

- **[Single Source of Truth Fix](2025-12-13-single-source-of-truth-fix.md)** - Main fix summary for single source of truth lookback computation
- **[Single Source of Truth Complete](2025-12-13-single-source-of-truth-complete.md)** - Complete fix summary with all details
- **[Single Source of Truth Detailed](2025-12-13-single-source-of-truth-fix-detailed.md)** - Detailed technical notes
- **[POST_PRUNE Invariant Check](2025-12-13-post-prune-invariant-check.md)** - Invariant check implementation
- **[XD Pattern Fix](2025-12-13-xd-pattern-fix.md)** - Day-suffix pattern inference fix
- **[Gatekeeper Lookback Fix](2025-12-13-gatekeeper-lookback-fix.md)** - Gatekeeper lookback computation fix
- **[Lookback Detection Fix](2025-12-13-lookback-detection-fix.md)** - Lookback detection precedence fix
- **[Lookback Inference Consistency Fix](2025-12-13-lookback-inference-consistency-fix.md)** - Consistency fixes for lookback inference
- **[Canonical Map XD Fix](2025-12-13-canonical-map-xd-fix.md)** - Canonical map day-suffix fix
- **[Lookback Fingerprint Tracking](2025-12-13-lookback-fingerprint-tracking.md)** - Initial fingerprint tracking implementation to ensure lookback computed on exact final feature set
- **[Fingerprint Improvements](2025-12-13-fingerprint-improvements.md)** - Set-invariant fingerprints, LookbackResult dataclass, explicit stage logging
- **[Lookback Result Migration](2025-12-13-lookback-result-dataclass-migration.md)** - Migration from tuple to dataclass return types
- **[Leakage Validation Fix](2025-12-13-leakage-validation-fix.md)** - Unified leakage budget calculator, calendar feature classification, separate purge/embargo validation

### Feature Selection Critical Fixes

- **[Implementation Verification](2025-12-13-implementation-verification.md)** - Complete verification of all 6 critical checks + 2 last-mile improvements
- **[Critical Fixes](2025-12-13-critical-fixes.md)** - Detailed root-cause analysis and fixes for shared harness, CatBoost dtype, RFE, linear models
- **[Telemetry Scoping Fix](2025-12-13-telemetry-scoping-fix.md)** - Telemetry scoping implementation (view→route_type mapping, cohort filtering)
- **[Sharp Edges Verification](2025-12-13-sharp-edges-verification.md)** - Verification against user checklist (view consistency, symbol policy, cohort filtering)
- **[Stability and Dtype Fixes](2025-12-13-stability-and-dtype-fixes.md)** - Stability per-model-family and dtype enforcement fixes
- **[Feature Selection Fixes](2025-12-13-feature-selection-fixes.md)** - Additional feature selection fixes and improvements
- **[Telemetry Scoping Audit](2025-12-13-telemetry-scoping-audit.md)** - Audit of telemetry scoping against user's checklist

### Look-Ahead Bias Fixes (2025-12-14)

- **[Look-Ahead Bias Fix Plan](LOOKAHEAD_BIAS_FIX_PLAN.md)** - Complete analysis of 4 critical look-ahead bias issues and required fixes
- **[Safe Implementation Plan](LOOKAHEAD_BIAS_SAFE_IMPLEMENTATION.md)** - Feature flag-based implementation strategy with gradual rollout plan
- **Status**: ✅ All fixes implemented (behind feature flags, default: OFF)
- **Branch**: `fix/lookahead-bias-fixes`
- **Fixes**:
  - Fix #1: Rolling windows exclude current bar
  - Fix #2: CV-based normalization support
  - Fix #3: pct_change() verification (handled by Fix #1)
  - Fix #4: Feature renaming (beta_20d → volatility_20d_returns)
- **Additional**: Symbol-specific evaluation logging, feature selection bug fix (task_type collision)

## Related Documentation

- [Known Issues](../../02_reference/KNOWN_ISSUES.md) - Reference documentation
- [Changelog](../../02_reference/changelog/README.md) - Change history
- [Look-Ahead Bias Fixes Changelog](../../02_reference/changelog/2025-12-14-lookahead-bias-fixes.md) - Complete changelog for look-ahead bias fixes
- [Feature Selection Unification Changelog](../../02_reference/changelog/2025-12-13-feature-selection-unification.md) - Complete changelog for feature selection unification

