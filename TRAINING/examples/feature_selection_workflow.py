"""
Copyright (c) 2025 Fox ML Infrastructure

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

"""
Feature Selection Workflow Example

Demonstrates the recommended workflow:
1. Train with all features to get feature importance
2. Select top N features
3. Retrain with selected features for better performance

This implements Step 3 & 4 from the optimization guide.
"""


import numpy as np
import pandas as pd
import logging
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from strategies.single_task import SingleTaskStrategy
from utils.feature_selection import (
    select_top_features,
    get_feature_importance_from_strategy,
    create_feature_report,
    auto_select_features
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def feature_selection_workflow(
    X: np.ndarray,
    y_dict: dict,
    feature_names: list,
    config: dict,
    n_features: int = 50,
    primary_target: str = 'fwd_ret_5m'
):
    """
    Complete feature selection workflow.
    
    Args:
        X: Full feature matrix (N, F) where F is large (e.g., 421)
        y_dict: Dictionary of targets
        feature_names: List of all feature names
        config: Configuration dict
        n_features: Number of features to select (default: 50)
        primary_target: Target to use for feature selection
        
    Returns:
        X_selected: Reduced feature matrix (N, n_features)
        selected_features: List of selected feature names
        strategy_final: Final trained strategy on selected features
    """
    
    logger.info("=" * 80)
    logger.info("FEATURE SELECTION WORKFLOW")
    logger.info("=" * 80)
    logger.info(f"Initial features: {X.shape[1]}")
    logger.info(f"Target for selection: {primary_target}")
    logger.info(f"Target features: {n_features}")
    
    # ========================================
    # STEP 1: Train initial model on all features
    # ========================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 1: Training initial model to get feature importance")
    logger.info("=" * 80)
    
    # Use only the primary target for initial training
    y_primary = {primary_target: y_dict[primary_target]}
    
    strategy_initial = SingleTaskStrategy(config)
    strategy_initial.train(X, y_primary, feature_names)
    
    # ========================================
    # STEP 2: Get feature importance and select top features
    # ========================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: Extracting feature importance")
    logger.info("=" * 80)
    
    sorted_features, sorted_importances = get_feature_importance_from_strategy(
        strategy_initial,
        primary_target,
        feature_names
    )
    
    if len(sorted_features) == 0:
        logger.error("Failed to get feature importance!")
        return X, feature_names, strategy_initial
    
    # Create feature importance report
    report = create_feature_report(
        sorted_features,
        sorted_importances,
        top_n=n_features,
        output_file='feature_importance_report.csv'
    )
    
    logger.info(f"\nTop 10 features:")
    for i, (feat, imp) in enumerate(zip(sorted_features[:10], sorted_importances[:10]), 1):
        logger.info(f"  {i}. {feat}: {imp:.6f}")
    
    # ========================================
    # STEP 3: Select top features
    # ========================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: Selecting top features")
    logger.info("=" * 80)
    
    # Get indices of top features
    top_feature_indices = [feature_names.index(f) for f in sorted_features[:n_features]]
    X_selected = X[:, top_feature_indices]
    selected_features = sorted_features[:n_features]
    
    logger.info(f"Selected {len(selected_features)} features")
    logger.info(f"New shape: {X_selected.shape}")
    
    # ========================================
    # STEP 4: Retrain on selected features with all targets
    # ========================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 4: Retraining on selected features (all targets)")
    logger.info("=" * 80)
    
    strategy_final = SingleTaskStrategy(config)
    results = strategy_final.train(X_selected, y_dict, selected_features)
    
    logger.info("\n" + "=" * 80)
    logger.info("WORKFLOW COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Reduced features from {X.shape[1]} to {X_selected.shape[1]}")
    logger.info(f"Trained {len(results)} models")
    
    return X_selected, selected_features, strategy_final


def add_regime_feature(
    X_selected: np.ndarray,
    selected_features: list,
    regime_config: dict = None
) -> tuple:
    """
    Add GMM regime as a feature (optional Step 4 enhancement).
    
    Args:
        X_selected: Selected feature matrix
        selected_features: List of selected feature names
        regime_config: Configuration for GMM (default: n_components=3)
        
    Returns:
        X_with_regime: Feature matrix with regime added
        features_with_regime: Feature names with regime added
    """
    from sklearn.mixture import GaussianMixture
    
    if regime_config is None:
        regime_config = {'n_components': 3, 'random_state': 42}
    
    logger.info(f"\nAdding GMM regime feature (n_components={regime_config['n_components']})")
    
    # Train GMM on key features (e.g., volatility and momentum)
    # For simplicity, use first 5 features
    gmm = GaussianMixture(**regime_config)
    gmm.fit(X_selected[:, :5])
    
    # Get regime predictions
    regime_labels = gmm.predict(X_selected).reshape(-1, 1)
    
    # Add regime as new feature
    X_with_regime = np.column_stack([X_selected, regime_labels])
    features_with_regime = selected_features + ['gmm_regime']
    
    logger.info(f"Regime distribution: {np.bincount(regime_labels.ravel())}")
    logger.info(f"New shape: {X_with_regime.shape}")
    
    return X_with_regime, features_with_regime


# Example usage
if __name__ == "__main__":
    # Load your data
    # X = ... (N, 421)
    # y_dict = {'fwd_ret_5m': ..., 'mfe_5m': ..., 'mdd_5m': ..., ...}
    # feature_names = [...]
    
    # Configuration with robust hyperparameters
    config = {
        'models': {
            'lightgbm': {
                'max_depth': 8,
                'num_leaves': 96,
                'learning_rate': 0.03,
                'n_estimators': 1000,
                'subsample': 0.75,
                'colsample_bytree': 0.75,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1,
            }
        },
        'early_stopping_rounds': 50,
    }
    
    # Run workflow
    # X_selected, selected_features, strategy = feature_selection_workflow(
    #     X, y_dict, feature_names, config,
    #     n_features=50,
    #     primary_target='fwd_ret_5m'
    # )
    
    # Optional: Add regime feature
    # X_final, features_final = add_regime_feature(X_selected, selected_features)
    
    print("\nWorkflow complete! Use X_selected and selected_features for predictions.")

