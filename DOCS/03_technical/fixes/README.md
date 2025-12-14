# Fixes Documentation

Known issues, bug fixes, and migration notes.

## Contents

- **[Known Issues](KNOWN_ISSUES.md)** - Current issues and limitations
- **[Bug Fixes](BUG_FIXES.md)** - Fix history
- **[Migration Notes](MIGRATION_NOTES.md)** - Migration guide
- **[TensorFlow Executable Stack Fix](TENSORFLOW_EXECUTABLE_STACK_FIX.md)** - TensorFlow fix

## Recent Critical Fixes (2025-12-13)

### Leakage Controls & Fingerprint Tracking

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

## Related Documentation

- [Known Issues](../../02_reference/KNOWN_ISSUES.md) - Reference documentation
- [Changelog](../../02_reference/changelog/README.md) - Change history
- [Feature Selection Unification Changelog](../../02_reference/changelog/2025-12-13-feature-selection-unification.md) - Complete changelog for feature selection unification

