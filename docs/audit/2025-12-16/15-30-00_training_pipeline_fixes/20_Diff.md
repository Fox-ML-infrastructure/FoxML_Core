# Diff Summary

**Files touched**
- `TRAINING/training_strategies/utils.py` (+25/−10) - Enhanced normalization
- `TRAINING/training_strategies/family_runners.py` (+8/−0) - Normalization at boundaries
- `TRAINING/training_strategies/training.py` (+85/−15) - Preflight, summary, feature validation
- `TRAINING/common/runtime_policy.py` (+10/−0) - Normalization in get_policy
- `TRAINING/common/license_banner.py` (+12/−2) - Child process suppression
- `TRAINING/common/threads.py` (+4/−0) - Banner suppression in child env
- `TRAINING/orchestration/intelligent_trainer.py` (+3/−2) - Module-level banner guard
- `TRAINING/utils/reproducibility_tracker.py` (+30/−5) - Defensive Enum/string handling

**Notes**
- All changes are backward compatible
- No breaking API changes
- Banner suppression is opt-out (can force with `FOXML_FORCE_BANNER=1`)

