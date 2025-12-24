# Risks & Assumptions

## Assumptions

- All callers of _save_to_cohort() eventually pass view in cohort_metadata
- universe_sig (or universe_id) is populated in cohort_metadata by cross_sectional_data.py
- Existing callers without proper metadata get warnings (not errors in default mode)

## Risks

- **Legacy callers may hit warning spam**: Mitigated by fallback behavior
- **Strict mode could break pipeline if enabled prematurely**: Default is false
- **_compute_cohort_id() now requires view**: Callers must update (PR2 scope)

## Rollback

```bash
git checkout HEAD~1 -- TRAINING/orchestration/utils/output_layout.py
git checkout HEAD~1 -- TRAINING/orchestration/utils/reproducibility_tracker.py
git checkout HEAD~1 -- CONFIG/pipeline/training/safety.yaml
git checkout HEAD~1 -- tests/test_scope_violation_firewall.py
```

