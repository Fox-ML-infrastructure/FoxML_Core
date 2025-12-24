# Changes

## Actions

- `TRAINING/orchestration/utils/output_layout.py`: **ADD** - New OutputLayout dataclass (SST for output paths)
- `TRAINING/orchestration/utils/reproducibility_tracker.py`: **EDIT** - Import OutputLayout, add firewall to _save_to_cohort(), update _compute_cohort_id() to require view
- `CONFIG/pipeline/training/safety.yaml`: **EDIT** - Add output_layout.strict_scope_partitioning config flag
- `tests/test_scope_violation_firewall.py`: **ADD** - Killer integration test for scope violations

## Commands Run

```bash
# Test firewall validation
python << 'EOF'
from TRAINING.orchestration.utils.output_layout import OutputLayout, validate_cohort_metadata
# ... test cases ...
EOF
# Result: ALL FIREWALL TESTS PASSED
```

