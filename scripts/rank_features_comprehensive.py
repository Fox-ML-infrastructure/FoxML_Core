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
Comprehensive Feature Ranking System

Combines target-dependent (model importance) and target-independent 
(data quality, variance, multicollinearity) metrics to identify features
with the best "edge" across multiple dimensions.

Ranking Dimensions:
1. Target-Dependent (Predictive Edge):
   - Model-based feature importance (LightGBM, XGBoost, RF, NN)
   - Cross-model consensus
   - Target-specific predictive power

2. Target-Independent (Data Quality Edge):
   - Missing value rate (lower = better)
   - Variance/standard deviation (higher = more informative)
   - Distribution quality (skewness, kurtosis)
   - Inter-feature correlation (lower = less redundant)

3. Composite Edge Score:
   - Weighted combination of all metrics
   - Normalized to 0-1 scale
   - Ranks features by overall "edge"
"""


import argparse
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np
from collections import defaultdict
from dataclasses import dataclass, asdict
import json
import warnings
from scipy import stats

# Add project root FIRST (before any scripts.* imports)
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from CONFIG.config_loader import load_model_config
import yaml

# Import checkpoint utility (after path is set)
from scripts.utils.checkpoint import CheckpointManager

# Setup logging with journald support
from scripts.utils.logging_setup import setup_logging
logger = setup_logging(
    script_name="rank_features_comprehensive",
    level=logging.INFO,
    use_journald=True
)

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


@dataclass
class FeatureEdgeMetrics:
    """Comprehensive edge metrics for a single feature"""
    feature_name: str
    
    # Target-dependent (predictive edge)
    model_importance_mean: float = 0.0
    model_importance_std: float = 0.0
    model_consensus: float = 0.0  # Fraction of models that agree
    n_models: int = 0
    
    # Target-independent (data quality edge)
    missing_rate: float = 0.0  # Fraction of missing values
    variance: float = 0.0
    std_dev: float = 0.0
    skewness: float = 0.0
    kurtosis_val: float = 0.0
    max_correlation: float = 0.0  # Highest correlation with another feature
    n_high_corr: int = 0  # Number of features with |r| > 0.9
    
    # Composite scores
    predictive_edge: float = 0.0  # Normalized model importance
    quality_edge: float = 0.0  # Normalized data quality
    redundancy_penalty: float = 0.0  # Penalty for high correlation
    composite_edge: float = 0.0  # Final weighted score
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'FeatureEdgeMetrics':
        """Create from dictionary"""
        return cls(**d)


def load_sample_data(
    symbol: str,
    data_dir: Path,
    max_samples: int = 50000
) -> pd.DataFrame:
    """Load sample data for a symbol"""
    # Try both path structures
    data_paths = [
        data_dir / f"interval=5m/symbol={symbol}/{symbol}.parquet",
        data_dir / f"symbol={symbol}/{symbol}.parquet"
    ]
    
    data_path = None
    for path in data_paths:
        if path.exists():
            data_path = path
            break
    
    if data_path is None:
        raise FileNotFoundError(f"Data not found for {symbol}. Tried: {data_paths}")
    
    df = pd.read_parquet(data_path)
    
    # Sample if too large
    if len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42)
        logger.info(f"  Sampled {max_samples} rows from {symbol}")
    
    return df


def filter_safe_features(df: pd.DataFrame, target_column: str = None) -> List[str]:
    """Filter out leaking features using the exclusion config"""
    sys.path.insert(0, str(_REPO_ROOT / "scripts"))
    from filter_leaking_features import filter_features
    
    all_columns = df.columns.tolist()
    safe_columns = filter_features(all_columns, verbose=False)
    
    # Remove target if specified
    if target_column and target_column in safe_columns:
        safe_columns = [c for c in safe_columns if c != target_column]
    
    # Remove non-numeric columns
    numeric_cols = df[safe_columns].select_dtypes(include=[np.number]).columns.tolist()
    
    return numeric_cols


def compute_target_dependent_metrics(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    model_families: List[str] = None
) -> Dict[str, Dict[str, float]]:
    """
    Compute model-based feature importance (target-dependent)
    
    Returns:
        Dict mapping feature_name -> {importance_mean, importance_std, consensus, n_models}
    """
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    import lightgbm as lgb
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.neural_network import MLPRegressor
    
    if model_families is None:
        model_families = ['lightgbm', 'random_forest', 'neural_network']
    
    # Validate target before training
    sys.path.insert(0, str(_REPO_ROOT / "scripts" / "utils"))
    try:
        from target_validation import validate_target
        is_valid, error_msg = validate_target(y, min_samples=10, min_class_samples=2)
        if not is_valid:
            logger.debug(f"    Skipping models: {error_msg}")
            return {feat: {'importance_mean': 0.0, 'importance_std': 0.0, 'consensus': 0.0, 'n_models': 0}
                    for feat in feature_names}
    except ImportError:
        # Fallback
        unique_vals = np.unique(y[~np.isnan(y)])
        if len(unique_vals) < 2:
            return {feat: {'importance_mean': 0.0, 'importance_std': 0.0, 'consensus': 0.0, 'n_models': 0}
                    for feat in feature_names}
    
    # Determine task type
    unique_vals = np.unique(y[~np.isnan(y)])
    is_binary = len(unique_vals) == 2 and set(unique_vals).issubset({0, 1, 0.0, 1.0})
    is_multiclass = len(unique_vals) <= 10 and all(
        isinstance(v, (int, np.integer)) or (isinstance(v, float) and v.is_integer())
        for v in unique_vals
    )
    is_classification = is_binary or is_multiclass
    
    # Impute for neural networks
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    X_imputed = imputer.fit_transform(X)
    X_scaled = scaler.fit_transform(X_imputed)
    
    # GPU detection for LightGBM
    gpu_params = {}
    try:
        test_model = lgb.LGBMRegressor(device='cuda', n_estimators=1, verbose=-1)
        test_model.fit(np.random.rand(10, 5), np.random.rand(10))
        gpu_params = {'device': 'cuda', 'gpu_device_id': 0}
    except:
        pass
    
    # Store importance from each model
    all_importances = defaultdict(list)
    
    # LightGBM
    if 'lightgbm' in model_families:
        try:
            if is_classification:
                if is_multiclass:
                    model = lgb.LGBMClassifier(
                        n_estimators=100,
                        max_depth=6,
                        num_class=len(unique_vals),
                        verbose=-1,
                        random_state=42,
                        **gpu_params
                    )
                else:
                    model = lgb.LGBMClassifier(
                        n_estimators=100,
                        max_depth=6,
                        verbose=-1,
                        random_state=42,
                        **gpu_params
                    )
            else:
                model = lgb.LGBMRegressor(
                    n_estimators=100,
                    max_depth=6,
                    verbose=-1,
                    random_state=42,
                    **gpu_params
                )
            
            try:
                model.fit(X_imputed, y)
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    for i, feat in enumerate(feature_names):
                        all_importances[feat].append(importances[i])
            except (ValueError, TypeError) as e:
                error_str = str(e).lower()
                if any(kw in error_str for kw in ['invalid classes', 'number of classes', 'too few']):
                    logger.debug(f"    LightGBM: Target degenerate")
                else:
                    logger.warning(f"LightGBM failed: {e}")
        except Exception as e:
            logger.warning(f"LightGBM failed: {e}")
    
    # Random Forest
    if 'random_forest' in model_families:
        try:
            if is_classification:
                from sklearn.ensemble import RandomForestClassifier
                model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                    n_jobs=1
                )
            else:
                model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                    n_jobs=1
                )
            
            try:
                model.fit(X_imputed, y)
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    for i, feat in enumerate(feature_names):
                        all_importances[feat].append(importances[i])
            except (ValueError, TypeError) as e:
                error_str = str(e).lower()
                if any(kw in error_str for kw in ['invalid classes', 'too few']):
                    logger.debug(f"    Random Forest: Target degenerate")
                else:
                    logger.warning(f"Random Forest failed: {e}")
        except Exception as e:
            logger.warning(f"Random Forest failed: {e}")
    
    # Neural Network (permutation importance approximation)
    if 'neural_network' in model_families:
        try:
            if is_classification:
                from sklearn.neural_network import MLPClassifier
                model = MLPClassifier(
                    hidden_layer_sizes=(64, 32),
                    max_iter=200,
                    random_state=42,
                    early_stopping=True,
                    validation_fraction=0.1
                )
            else:
                model = MLPRegressor(
                    hidden_layer_sizes=(64, 32),
                    max_iter=200,
                    random_state=42,
                    early_stopping=True,
                    validation_fraction=0.1
                )
            
            try:
                model.fit(X_scaled, y)
                
                # Simple permutation importance (fast approximation)
                baseline_score = model.score(X_scaled, y)
                for i, feat in enumerate(feature_names):
                    X_permuted = X_scaled.copy()
                    np.random.shuffle(X_permuted[:, i])
                    permuted_score = model.score(X_permuted, y)
                    importance = baseline_score - permuted_score
                    all_importances[feat].append(max(0, importance))
            except (ValueError, TypeError) as e:
                error_str = str(e).lower()
                if any(kw in error_str for kw in ['least populated class', 'too few', 'invalid classes']):
                    logger.debug(f"    Neural Network: Target too imbalanced")
                else:
                    logger.warning(f"Neural Network failed: {e}")
        except Exception as e:
            logger.warning(f"Neural Network failed: {e}")
    
    # Aggregate per feature
    feature_metrics = {}
    for feat in feature_names:
        importances = all_importances.get(feat, [])
        if importances:
            feature_metrics[feat] = {
                'importance_mean': np.mean(importances),
                'importance_std': np.std(importances),
                'n_models': len(importances),
                'consensus': len(importances) / len(model_families) if model_families else 0.0
            }
        else:
            feature_metrics[feat] = {
                'importance_mean': 0.0,
                'importance_std': 0.0,
                'n_models': 0,
                'consensus': 0.0
            }
    
    return feature_metrics


def compute_target_independent_metrics(
    df: pd.DataFrame,
    feature_names: List[str]
) -> Dict[str, Dict[str, float]]:
    """
    Compute data quality metrics (target-independent)
    
    Returns:
        Dict mapping feature_name -> {missing_rate, variance, std_dev, skewness, kurtosis, max_correlation, n_high_corr}
    """
    feature_metrics = {}
    
    # Compute correlation matrix once
    feature_df = df[feature_names].select_dtypes(include=[np.number])
    corr_matrix = feature_df.corr().abs()
    
    for feat in feature_names:
        if feat not in df.columns:
            continue
        
        series = df[feat]
        
        # Skip non-numeric
        if not pd.api.types.is_numeric_dtype(series):
            continue
        
        # Remove NaN for calculations
        clean_series = series.dropna()
        
        if len(clean_series) == 0:
            feature_metrics[feat] = {
                'missing_rate': 1.0,
                'variance': 0.0,
                'std_dev': 0.0,
                'skewness': 0.0,
                'kurtosis_val': 0.0,
                'max_correlation': 0.0,
                'n_high_corr': 0
            }
            continue
        
        # Missing rate
        missing_rate = series.isna().sum() / len(series)
        
        # Variance and std
        variance = clean_series.var()
        std_dev = clean_series.std()
        
        # Distribution metrics
        try:
            skewness = float(stats.skew(clean_series))
            kurtosis_val = float(stats.kurtosis(clean_series))
        except:
            skewness = 0.0
            kurtosis_val = 0.0
        
        # Correlation metrics
        if feat in corr_matrix.columns:
            feat_corr = corr_matrix[feat].drop(feat)  # Remove self-correlation
            max_correlation = feat_corr.max() if len(feat_corr) > 0 else 0.0
            n_high_corr = (feat_corr > 0.9).sum()
        else:
            max_correlation = 0.0
            n_high_corr = 0
        
        feature_metrics[feat] = {
            'missing_rate': missing_rate,
            'variance': variance,
            'std_dev': std_dev,
            'skewness': abs(skewness),  # Use absolute value
            'kurtosis_val': abs(kurtosis_val),  # Use absolute value
            'max_correlation': max_correlation,
            'n_high_corr': n_high_corr
        }
    
    return feature_metrics


def compute_composite_edge_scores(
    all_metrics: List[FeatureEdgeMetrics],
    weights: Dict[str, float] = None
) -> List[FeatureEdgeMetrics]:
    """
    Compute composite edge scores from all metrics
    
    Weights:
    - predictive_edge: 0.50 (model importance)
    - quality_edge: 0.30 (data quality)
    - redundancy_penalty: -0.20 (multicollinearity penalty)
    """
    if weights is None:
        weights = {
            'predictive': 0.50,
            'quality': 0.30,
            'redundancy': -0.20
        }
    
    # Normalize each metric to 0-1 scale
    importance_vals = [m.model_importance_mean for m in all_metrics]
    missing_vals = [m.missing_rate for m in all_metrics]
    variance_vals = [m.variance for m in all_metrics]
    corr_vals = [m.max_correlation for m in all_metrics]
    
    # Normalize (handle edge cases)
    max_importance = max(importance_vals) if importance_vals else 1.0
    max_variance = max(variance_vals) if variance_vals else 1.0
    
    for metric in all_metrics:
        # Predictive edge (higher importance = better, normalized)
        metric.predictive_edge = (
            metric.model_importance_mean / max_importance
            if max_importance > 0 else 0.0
        ) * metric.model_consensus  # Weight by consensus
        
        # Quality edge (lower missing = better, higher variance = better)
        completeness_score = 1.0 - metric.missing_rate
        variance_score = (
            metric.variance / max_variance
            if max_variance > 0 else 0.0
        )
        metric.quality_edge = 0.6 * completeness_score + 0.4 * variance_score
        
        # Redundancy penalty (higher correlation = worse)
        metric.redundancy_penalty = metric.max_correlation
        
        # Composite edge (weighted combination)
        metric.composite_edge = (
            weights['predictive'] * metric.predictive_edge +
            weights['quality'] * metric.quality_edge +
            weights['redundancy'] * metric.redundancy_penalty
        )
        
        # Ensure non-negative
        metric.composite_edge = max(0.0, metric.composite_edge)
    
    return all_metrics


def rank_features_comprehensive(
    symbols: List[str],
    data_dir: Path,
    target_column: str = None,
    model_families: List[str] = None,
    max_samples: int = 50000
) -> List[FeatureEdgeMetrics]:
    """
    Comprehensive feature ranking across multiple dimensions
    
    Args:
        symbols: List of symbols to analyze
        data_dir: Path to data directory
        target_column: Optional target column for predictive metrics
        model_families: List of model families to use
        max_samples: Max samples per symbol
    
    Returns:
        List of FeatureEdgeMetrics, sorted by composite_edge
    """
    logger.info("="*70)
    logger.info("COMPREHENSIVE FEATURE RANKING")
    logger.info("="*70)
    logger.info(f"Symbols: {', '.join(symbols)}")
    logger.info(f"Target: {target_column or 'N/A (target-independent only)'}")
    logger.info(f"Model families: {', '.join(model_families or ['lightgbm', 'random_forest', 'neural_network'])}")
    logger.info("")
    
    # Aggregate metrics across symbols
    all_target_dependent = defaultdict(lambda: {'importances': [], 'n_models': []})
    all_target_independent = defaultdict(lambda: {'metrics': []})
    
    for symbol in symbols:
        logger.info(f"Processing {symbol}...")
        
        try:
            # Load data
            df = load_sample_data(symbol, data_dir, max_samples)
            
            # Filter safe features
            feature_names = filter_safe_features(df, target_column)
            
            if not feature_names:
                logger.warning(f"  No safe features found for {symbol}")
                continue
            
            logger.info(f"  Found {len(feature_names)} safe features")
            
            # Target-independent metrics (always computed)
            target_indep = compute_target_independent_metrics(df, feature_names)
            for feat, metrics in target_indep.items():
                all_target_independent[feat]['metrics'].append(metrics)
            
            # Target-dependent metrics (only if target provided)
            if target_column and target_column in df.columns:
                # Prepare data
                X = df[feature_names].values
                y = df[target_column].values
                
                # Remove rows with NaN in target
                valid_mask = ~np.isnan(y)
                X = X[valid_mask]
                y = y[valid_mask]
                
                if len(y) > 0:
                    target_dep = compute_target_dependent_metrics(
                        X, y, feature_names, model_families
                    )
                    
                    for feat, metrics in target_dep.items():
                        all_target_dependent[feat]['importances'].append(metrics['importance_mean'])
                        all_target_dependent[feat]['n_models'].append(metrics['n_models'])
        
        except Exception as e:
            logger.warning(f"  Failed for {symbol}: {e}")
            continue
    
    # Aggregate across symbols
    logger.info("\nAggregating metrics across symbols...")
    
    all_metrics = []
    all_feature_names = set(all_target_independent.keys())
    if target_column:
        all_feature_names.update(all_target_dependent.keys())
    
    for feat in all_feature_names:
        # Aggregate target-independent
        target_indep_list = all_target_independent[feat]['metrics']
        if not target_indep_list:
            continue
        
        # Average across symbols
        avg_missing = np.mean([m['missing_rate'] for m in target_indep_list])
        avg_variance = np.mean([m['variance'] for m in target_indep_list])
        avg_std = np.mean([m['std_dev'] for m in target_indep_list])
        avg_skew = np.mean([m['skewness'] for m in target_indep_list])
        avg_kurt = np.mean([m['kurtosis_val'] for m in target_indep_list])
        avg_max_corr = np.mean([m['max_correlation'] for m in target_indep_list])
        avg_n_high_corr = np.mean([m['n_high_corr'] for m in target_indep_list])
        
        # Aggregate target-dependent
        if target_column and feat in all_target_dependent:
            importances = all_target_dependent[feat]['importances']
            n_models_list = all_target_dependent[feat]['n_models']
            
            avg_importance = np.mean(importances) if importances else 0.0
            std_importance = np.std(importances) if len(importances) > 1 else 0.0
            avg_n_models = np.mean(n_models_list) if n_models_list else 0.0
            consensus = avg_n_models / len(model_families) if model_families else 0.0
        else:
            avg_importance = 0.0
            std_importance = 0.0
            avg_n_models = 0.0
            consensus = 0.0
        
        metric = FeatureEdgeMetrics(
            feature_name=feat,
            model_importance_mean=avg_importance,
            model_importance_std=std_importance,
            model_consensus=consensus,
            n_models=int(avg_n_models),
            missing_rate=avg_missing,
            variance=avg_variance,
            std_dev=avg_std,
            skewness=avg_skew,
            kurtosis_val=avg_kurt,
            max_correlation=avg_max_corr,
            n_high_corr=int(avg_n_high_corr)
        )
        
        all_metrics.append(metric)
    
    # Compute composite scores
    logger.info("Computing composite edge scores...")
    all_metrics = compute_composite_edge_scores(all_metrics)
    
    # Sort by composite edge
    all_metrics.sort(key=lambda x: x.composite_edge, reverse=True)
    
    logger.info(f"\nRanked {len(all_metrics)} features")
    
    return all_metrics


def save_rankings(
    rankings: List[FeatureEdgeMetrics],
    output_dir: Path,
    target_column: str = None
):
    """Save comprehensive feature rankings to files"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert to DataFrame
    records = [asdict(m) for m in rankings]
    df = pd.DataFrame(records)
    
    # Save CSV
    csv_path = output_dir / "feature_rankings_comprehensive.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved rankings to {csv_path}")
    
    # Save JSON (for programmatic access)
    json_path = output_dir / "feature_rankings_comprehensive.json"
    with open(json_path, 'w') as f:
        json.dump(records, f, indent=2, default=str)
    logger.info(f"Saved JSON to {json_path}")
    
    # Save summary report
    report_path = output_dir / "feature_ranking_report.md"
    with open(report_path, 'w') as f:
        f.write("# Comprehensive Feature Ranking Report\n\n")
        f.write(f"**Target**: {target_column or 'N/A (target-independent)'}\n\n")
        f.write(f"**Total Features Ranked**: {len(rankings)}\n\n")
        f.write("## Top 50 Features by Composite Edge\n\n")
        f.write("| Rank | Feature | Composite Edge | Predictive | Quality | Redundancy |\n")
        f.write("|------|---------|----------------|------------|---------|------------|\n")
        
        for i, metric in enumerate(rankings[:50], 1):
            f.write(f"| {i} | `{metric.feature_name}` | "
                   f"{metric.composite_edge:.4f} | "
                   f"{metric.predictive_edge:.4f} | "
                   f"{metric.quality_edge:.4f} | "
                   f"{metric.max_correlation:.4f} |\n")
        
        f.write("\n## Metrics Explanation\n\n")
        f.write("- **Composite Edge**: Weighted combination of all metrics (0-1 scale)\n")
        f.write("- **Predictive**: Model-based importance (target-dependent)\n")
        f.write("- **Quality**: Data completeness + variance (target-independent)\n")
        f.write("- **Redundancy**: Max correlation with other features (lower = better)\n")
    
    logger.info(f"Saved report to {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive feature ranking (target-dependent + target-independent)"
    )
    parser.add_argument(
        '--symbols',
        type=str,
        default='AAPL,MSFT,GOOGL,TSLA,JPM',
        help='Comma-separated list of symbols'
    )
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=Path('data/data_labeled'),
        help='Path to data directory'
    )
    parser.add_argument(
        '--target',
        type=str,
        default=None,
        help='Target column for predictive metrics (optional)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('results/feature_rankings'),
        help='Output directory for rankings'
    )
    parser.add_argument(
        '--model-families',
        type=str,
        nargs='+',
        default=['lightgbm', 'random_forest', 'neural_network'],
        help='Model families to use for predictive metrics'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=50000,
        help='Max samples per symbol'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from checkpoint if available'
    )
    parser.add_argument(
        '--clear-checkpoint',
        action='store_true',
        help='Clear existing checkpoint and start fresh'
    )
    
    args = parser.parse_args()
    
    symbols = [s.strip() for s in args.symbols.split(',')]
    
    # Initialize checkpoint manager
    checkpoint_file = args.output_dir / "checkpoint.json"
    checkpoint = CheckpointManager(
        checkpoint_file=checkpoint_file,
        item_key_fn=lambda item: item if isinstance(item, str) else item[0]  # symbol name
    )
    
    # Clear checkpoint if requested
    if args.clear_checkpoint:
        checkpoint.clear()
        logger.info("Cleared checkpoint - starting fresh")
    
    # Load completed symbols
    completed = checkpoint.load_completed()
    logger.info(f"Found {len(completed)} completed symbol evaluations in checkpoint")
    
    # Filter symbols based on checkpoint
    symbols_to_process = symbols
    if args.resume and completed:
        symbols_to_process = [s for s in symbols if s not in completed]
        logger.info(f"Resuming: {len(symbols_to_process)} symbols remaining, {len(completed)} already completed")
    
    # Rank features
    rankings = rank_features_comprehensive(
        symbols=symbols_to_process if not args.resume else symbols,  # Process all if resuming to merge
        data_dir=args.data_dir,
        target_column=args.target,
        model_families=args.model_families,
        max_samples=args.max_samples
    )
    
    # Save checkpoint after processing
    if symbols_to_process:
        checkpoint.save_item("_all_symbols", [r.to_dict() for r in rankings])
    
    # Merge with checkpoint results if resuming
    if args.resume and completed:
        checkpoint_rankings = []
        for symbol, data in completed.items():
            if symbol != "_all_symbols" and isinstance(data, list):
                checkpoint_rankings.extend([FeatureEdgeMetrics.from_dict(d) for d in data])
        
        # Merge and deduplicate by feature_name
        existing_features = {r.feature_name for r in rankings}
        for r in checkpoint_rankings:
            if r.feature_name not in existing_features:
                rankings.append(r)
    
    # Save results
    save_rankings(rankings, args.output_dir, args.target)
    
    # Print top 20
    logger.info("\n" + "="*70)
    logger.info("TOP 20 FEATURES BY COMPOSITE EDGE")
    logger.info("="*70)
    for i, metric in enumerate(rankings[:20], 1):
        logger.info(
            f"{i:2d}. {metric.feature_name:40s} | "
            f"Edge: {metric.composite_edge:.4f} | "
            f"Predictive: {metric.predictive_edge:.4f} | "
            f"Quality: {metric.quality_edge:.4f} | "
            f"Max Corr: {metric.max_correlation:.4f}"
        )


if __name__ == '__main__':
    main()

