#!/usr/bin/env python3

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
Phase 1: Feature Engineering & Selection

This script:
1. Loads all features (~421)
2. Trains LightGBM to get feature importance
3. Selects top N features
4. Trains VAE for latent features
5. Trains GMM for regime labels
6. Saves all artifacts to metadata/
"""


import argparse
import json
import logging
import sys
from pathlib import Path
import numpy as np
import joblib
import yaml

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from strategies.single_task import SingleTaskStrategy
from utils.feature_selection import (
    select_top_features,
    get_feature_importance_from_strategy,
    create_feature_report
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_data(data_dir):
    """
    Load training data.
    
    MODIFY THIS FUNCTION to match your data loading logic.
    """
    logger.info(f"Loading data from {data_dir}")
    
    # TODO: Replace this with your actual data loading code
    # Example:
    # import pandas as pd
    # df = pd.read_parquet(f"{data_dir}/training_data.parquet")
    # X = df[feature_columns].values
    # y_dict = {target: df[target].values for target in targets}
    # feature_names = feature_columns
    
    # Placeholder for now
    raise NotImplementedError(
        "Please implement load_data() to load your specific data format.\n"
        "Expected outputs: X (N, F), y_dict (dict of targets), feature_names (list)"
    )
    
    return X, y_dict, feature_names


def select_features(X, y_dict, feature_names, config):
    """
    Select top N features based on LightGBM importance.
    
    Args:
        X: Feature matrix (N, F)
        y_dict: Dictionary of targets
        feature_names: List of feature names
        config: Configuration dict
        
    Returns:
        X_selected: Selected features (N, n_features)
        selected_features: Names of selected features
        feature_report: DataFrame with importance scores
    """
    logger.info("=" * 80)
    logger.info("STEP 1: Feature Selection")
    logger.info("=" * 80)
    
    n_features = config['feature_selection']['n_features']
    primary_target = config['feature_selection']['primary_target']
    
    logger.info(f"Training LightGBM on {primary_target} to get feature importance")
    logger.info(f"Will select top {n_features} features from {X.shape[1]} total")
    
    # Train LightGBM with optimized config
    lgbm_config = {
        'models': {
            'lightgbm': config.get('lightgbm', {})
        },
        'early_stopping_rounds': config.get('early_stopping_rounds', 50)
    }
    
    strategy = SingleTaskStrategy(lgbm_config)
    strategy.train(X, {primary_target: y_dict[primary_target]}, feature_names)
    
    # Get feature importance
    sorted_features, sorted_importances = get_feature_importance_from_strategy(
        strategy, primary_target, feature_names
    )
    
    if len(sorted_features) == 0:
        raise ValueError("Failed to get feature importance from LightGBM")
    
    # Create feature report
    feature_report = create_feature_report(
        sorted_features,
        sorted_importances,
        top_n=n_features
    )
    
    # Select top features
    top_features = sorted_features[:n_features]
    top_indices = [feature_names.index(f) for f in top_features]
    X_selected = X[:, top_indices]
    
    logger.info(f"✅ Selected {len(top_features)} features")
    logger.info(f"Top 10 features: {top_features[:10]}")
    
    return X_selected, top_features, feature_report


def train_vae(X_selected, config):
    """
    Train VAE for latent feature extraction.
    
    Args:
        X_selected: Selected features (N, F)
        config: Configuration dict
        
    Returns:
        vae_features: Latent features (N, latent_dim)
        vae_model: Trained VAE model
    """
    logger.info("=" * 80)
    logger.info("STEP 2: VAE Feature Engineering")
    logger.info("=" * 80)
    
    vae_config = config['feature_engineering']['vae']
    
    if not vae_config.get('enabled', True):
        logger.info("VAE disabled in config, skipping")
        return None, None
    
    latent_dim = vae_config['latent_dim']
    logger.info(f"Training VAE with latent_dim={latent_dim}")
    
    try:
        from model_fun.vae_trainer import VAETrainer
        
        vae = VAETrainer({'latent_dim': latent_dim})
        vae.train(X_selected, X_selected)  # VAE uses X as both input and target
        
        # Get latent features
        vae_features = vae.encode(X_selected)
        
        logger.info(f"✅ VAE trained, extracted {vae_features.shape[1]} latent features")
        
        return vae_features, vae
        
    except ImportError:
        logger.warning("VAE trainer not available, skipping")
        return None, None
    except Exception as e:
        logger.error(f"VAE training failed: {e}")
        logger.warning("Continuing without VAE features")
        return None, None


def train_gmm(X_selected, config):
    """
    Train GMM for regime detection.
    
    Args:
        X_selected: Selected features (N, F)
        config: Configuration dict
        
    Returns:
        regime_labels: Regime labels (N, 1)
        gmm_model: Trained GMM model
    """
    logger.info("=" * 80)
    logger.info("STEP 3: GMM Regime Detection")
    logger.info("=" * 80)
    
    gmm_config = config['feature_engineering']['gmm']
    
    if not gmm_config.get('enabled', True):
        logger.info("GMM disabled in config, skipping")
        return None, None
    
    n_components = gmm_config['n_components']
    logger.info(f"Training GMM with n_components={n_components}")
    
    try:
        from sklearn.mixture import GaussianMixture
        
        # Train on first few features (typically volatility, momentum, etc.)
        n_features_for_gmm = min(5, X_selected.shape[1])
        gmm = GaussianMixture(
            n_components=n_components,
            random_state=42,
            covariance_type='full'
        )
        gmm.fit(X_selected[:, :n_features_for_gmm])
        
        # Get regime labels
        regime_labels = gmm.predict(X_selected[:, :n_features_for_gmm]).reshape(-1, 1)
        
        # Log regime distribution
        regime_counts = np.bincount(regime_labels.ravel())
        logger.info(f"Regime distribution: {regime_counts}")
        logger.info(f"✅ GMM trained with {n_components} regimes")
        
        return regime_labels, gmm
        
    except ImportError:
        logger.warning("sklearn not available for GMM, skipping")
        return None, None
    except Exception as e:
        logger.error(f"GMM training failed: {e}")
        logger.warning("Continuing without GMM features")
        return None, None


def save_artifacts(output_dir, selected_features, feature_report, vae_model, gmm_model):
    """Save all Phase 1 artifacts"""
    logger.info("=" * 80)
    logger.info("STEP 4: Saving Artifacts")
    logger.info("=" * 80)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save feature list
    feature_list_path = output_dir / 'top_50_features.json'
    with open(feature_list_path, 'w') as f:
        json.dump(selected_features, f, indent=2)
    logger.info(f"✅ Saved feature list to {feature_list_path}")
    
    # Save feature importance report
    report_path = output_dir / 'feature_importance_report.csv'
    feature_report.to_csv(report_path, index=False)
    logger.info(f"✅ Saved feature report to {report_path}")
    
    # Save VAE model
    if vae_model is not None:
        vae_path = output_dir / 'vae_encoder.joblib'
        joblib.dump(vae_model, vae_path)
        logger.info(f"✅ Saved VAE model to {vae_path}")
    
    # Save GMM model
    if gmm_model is not None:
        gmm_path = output_dir / 'gmm_model.joblib'
        joblib.dump(gmm_model, gmm_path)
        logger.info(f"✅ Saved GMM model to {gmm_path}")
    
    # Save summary
    summary = {
        'n_features_original': None,  # Will be set by caller
        'n_features_selected': len(selected_features),
        'n_features_vae': vae_model.config['latent_dim'] if vae_model else 0,
        'n_features_gmm': 1 if gmm_model else 0,
        'n_features_total': len(selected_features) + (vae_model.config['latent_dim'] if vae_model else 0) + (1 if gmm_model else 0),
        'selected_features': selected_features[:10],  # Top 10 for summary
    }
    
    summary_path = output_dir / 'phase1_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"✅ Saved summary to {summary_path}")
    
    return summary


def main():
    parser = argparse.ArgumentParser(description='Phase 1: Feature Engineering & Selection')
    parser.add_argument('--data-dir', required=True, help='Directory containing training data')
    parser.add_argument('--config', required=True, help='Path to feature selection config YAML')
    parser.add_argument('--output-dir', required=True, help='Directory to save artifacts (metadata/)')
    parser.add_argument('--log-dir', default='logs', help='Directory for log files')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Load data
    logger.info("Loading training data...")
    X, y_dict, feature_names = load_data(args.data_dir)
    logger.info(f"Loaded {X.shape[0]} samples with {X.shape[1]} features")
    logger.info(f"Targets: {list(y_dict.keys())}")
    
    # Step 1: Feature Selection
    X_selected, selected_features, feature_report = select_features(
        X, y_dict, feature_names, config
    )
    
    # Step 2: VAE Feature Engineering
    vae_features, vae_model = train_vae(X_selected, config)
    
    # Step 3: GMM Regime Detection
    regime_labels, gmm_model = train_gmm(X_selected, config)
    
    # Step 4: Save Artifacts
    summary = save_artifacts(
        args.output_dir,
        selected_features,
        feature_report,
        vae_model,
        gmm_model
    )
    summary['n_features_original'] = X.shape[1]
    
    # Final summary
    logger.info("=" * 80)
    logger.info("PHASE 1 COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Original features: {X.shape[1]}")
    logger.info(f"Selected features: {len(selected_features)}")
    logger.info(f"VAE features: {vae_features.shape[1] if vae_features is not None else 0}")
    logger.info(f"GMM features: {1 if regime_labels is not None else 0}")
    logger.info(f"Total features: {summary['n_features_total']}")
    logger.info(f"Reduction: {X.shape[1]} → {summary['n_features_total']} ({summary['n_features_total']/X.shape[1]*100:.1f}%)")
    logger.info("=" * 80)
    logger.info(f"Artifacts saved to: {args.output_dir}")
    logger.info("Next step: Run Phase 2 (core model training)")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.error(f"Phase 1 failed: {e}", exc_info=True)
        sys.exit(1)

