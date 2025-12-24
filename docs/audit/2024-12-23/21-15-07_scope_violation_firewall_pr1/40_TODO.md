# TODO / Follow-ups

## PR2 Migration (owner: me)

- [ ] Update all callers of _compute_cohort_id() to pass view parameter
- [ ] Update save_multi_model_results() to use OutputLayout
- [ ] Update save_feature_importances_for_reproducibility() to use OutputLayout
- [ ] Convert artifact_paths.py to use OutputLayout internally
- [ ] Convert target_first_paths.py to use OutputLayout (or wrapper)
- [ ] Enable strict_scope_partitioning=true after migration complete

## Monitoring (owner: me)

- [ ] Watch logs for "SCOPE VIOLATION RISK" entries (telemetry for bad callers)
- [ ] Watch logs for "Missing ... in metadata" warnings (legacy callers needing update)

---

Links: See `docs/audit/2024-12-23/21-15-07_scope_violation_firewall_pr1/`

