# Diff Summary

**Files touched**

- `TRAINING/orchestration/utils/output_layout.py` (+255/-0) - NEW: OutputLayout dataclass
- `TRAINING/orchestration/utils/reproducibility_tracker.py` (+95/-5) - Firewall in _save_to_cohort, view param in _compute_cohort_id
- `CONFIG/pipeline/training/safety.yaml` (+8/-0) - output_layout.strict_scope_partitioning flag
- `tests/test_scope_violation_firewall.py` (+340/-0) - NEW: Killer integration tests

**Total:** 4 files, ~700 insertions, ~5 deletions

## Notes

- Backward compatible: _save_to_cohort() falls back to legacy paths if metadata incomplete
- Telemetry: SCOPE VIOLATION RISK logs even when view is missing (finds bad callers)
- Strict mode disabled by default; enable after PR2 migration

