# Scope Violation Firewall PR1 - Roadmap (2024-12-23 21:15)

**Prompt:** "Fix scope violation where symbol-specific cohorts (cohort=sy_...) incorrectly appear under CROSS_SECTIONAL/ view"

## Context

- Screenshot showed `cohort=sy_*` artifacts under `CROSS_SECTIONAL/` paths
- Root cause: mode resolution leaking across universes
- Symptoms: feature importance and artifacts saved at wrong scope

## Plan (PR1: Correctness Firewall)

1. Create OutputLayout dataclass (SST for all output paths)
2. Enforce view+universe scoping with hard invariants
3. Add validate_cohort_id() to catch scope violations at write time
4. Update _save_to_cohort() with backward-compatible firewall
5. Add strict mode config flag for gradual rollout
6. Create killer integration test

## Success Criteria

- [x] OutputLayout enforces view in {CROSS_SECTIONAL, SYMBOL_SPECIFIC}
- [x] SYMBOL_SPECIFIC requires symbol, CROSS_SECTIONAL cannot have symbol
- [x] universe_sig required in all paths
- [x] validate_cohort_id() catches cs_ vs sy_ mismatches
- [x] _save_to_cohort() validates on every write (with fallback for legacy)
- [x] All firewall tests pass

