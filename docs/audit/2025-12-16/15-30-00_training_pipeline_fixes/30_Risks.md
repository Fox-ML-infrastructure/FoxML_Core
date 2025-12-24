# Risks & Assumptions

## Assumptions
- All trainer families use TitleCase in MODMAP/TRAINER_MODULE_MAP (verified: yes)
- Banner should never print in child processes (correct: library code shouldn't print)
- Feature count collapse is a data quality issue, not a bug (needs investigation)

## Risks
- **Low**: Normalization might not catch all edge cases (mitigated by special_cases dict)
- **Low**: Banner suppression might hide legitimate issues (mitigated by TTY check + force flag)
- **Medium**: Preflight validation might reject valid families if normalization fails (mitigated by comprehensive special_cases)

## Rollback
```bash
# Selective revert
git checkout HEAD~1 -- TRAINING/training_strategies/utils.py
git checkout HEAD~1 -- TRAINING/training_strategies/family_runners.py
git checkout HEAD~1 -- TRAINING/training_strategies/training.py
git checkout HEAD~1 -- TRAINING/common/runtime_policy.py
git checkout HEAD~1 -- TRAINING/common/license_banner.py
git checkout HEAD~1 -- TRAINING/common/threads.py
git checkout HEAD~1 -- TRAINING/orchestration/intelligent_trainer.py
git checkout HEAD~1 -- TRAINING/utils/reproducibility_tracker.py
```

