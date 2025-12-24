"""
Documented Hook Points

List of all documented hook points for allowlist validation.

Import this in CI to enforce that only known hook points are used.
"""

# All documented hook points (for allowlist validation)
DOCUMENTED_HOOK_POINTS = [
    # Target Ranking
    'before_target_ranking',
    'after_target_ranking',
    'on_target_ranking_error',
    
    # Feature Selection
    'before_feature_selection',
    'after_feature_selection',
    'on_feature_selection_error',
    
    # Training
    'before_training',
    'after_training',
    'on_training_error',
    
    # Data Processing
    'before_data_load',
    'after_data_load',
    'before_data_preprocessing',
    'after_data_preprocessing',
]

# Convenience function for CI
def get_allowed_hook_points():
    """Return list of documented hook points for allowlist."""
    return DOCUMENTED_HOOK_POINTS.copy()
