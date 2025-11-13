"""
Target Predictability Ranking

Uses multiple model families to evaluate which of your 63 targets are most predictable.
This helps prioritize compute: train models on high-predictability targets first.

Methodology:
1. For each target, train multiple model families on sample data
2. Calculate predictability scores:
   - Model R² scores (cross-validated)
   - Feature importance magnitude (mean absolute SHAP/importance)
   - Consistency across models (low std = high confidence)
3. Rank targets by composite predictability score
4. Output ranked list with recommendations

Usage:
  # Rank all enabled targets
  python scripts/rank_target_predictability.py
  
  # Test on specific symbols first
  python scripts/rank_target_predictability.py --symbols AAPL,MSFT,GOOGL
  
  # Rank specific targets
  python scripts/rank_target_predictability.py --targets peak_60m,valley_60m,swing_high_15m
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass
import yaml
import json
from collections import defaultdict
import warnings

# Suppress sklearn feature name warnings (harmless)
warnings.filterwarnings('ignore', message='X does not have valid feature names')
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# Add project root
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TargetPredictabilityScore:
    """Predictability assessment for a single target"""
    target_name: str
    target_column: str
    mean_r2: float
    std_r2: float
    mean_importance: float  # Mean absolute importance
    consistency: float  # 1 - CV(R²) - lower is better
    n_models: int
    model_scores: Dict[str, float]
    composite_score: float = 0.0


def load_target_configs() -> Dict[str, Dict]:
    """Load target configurations"""
    config_path = _REPO_ROOT / "CONFIG" / "target_configs.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config['targets']


def discover_all_targets(symbol: str, data_dir: Path) -> Dict[str, Dict]:
    """
    Auto-discover all valid targets from data (non-degenerate).
    
    Returns dict of {target_name: config} for all valid targets found.
    """
    import pandas as pd
    
    # Load sample data to discover targets
    symbol_dir = data_dir / f"symbol={symbol}"
    parquet_file = symbol_dir / f"{symbol}.parquet"
    
    if not parquet_file.exists():
        raise FileNotFoundError(f"Cannot discover targets: {parquet_file} not found")
    
    df = pd.read_parquet(parquet_file)
    
    # Find all y_ columns
    all_targets = [c for c in df.columns if c.startswith('y_')]
    
    # Filter out degenerate targets (single class)
    valid_targets = {}
    degenerate_count = 0
    
    for target_col in all_targets:
        y = df[target_col].dropna()
        if len(y) == 0:
            continue
        
        unique_vals = y.unique()
        n_unique = len(unique_vals)
        
        # Skip degenerate targets (single class)
        if n_unique == 1:
            degenerate_count += 1
            continue
        
        # Skip first_touch targets (they're leaked - correlated with hit_direction features)
        if 'first_touch' in target_col:
            degenerate_count += 1
            continue
        
        # Create a simple config for this target
        target_name = target_col.replace('y_will_', '').replace('y_', '')
        valid_targets[target_name] = {
            'target_column': target_col,
            'description': f"Auto-discovered target: {target_col}",
            'use_case': f"{'Classification' if n_unique <= 10 else 'Regression'} target",
            'top_n': 60,
            'method': 'mean',
            'enabled': True
        }
    
    logger.info(f"  Discovered {len(valid_targets)} valid targets")
    logger.info(f"  Skipped {degenerate_count} degenerate targets (single class)")
    
    return valid_targets


def load_sample_data(
    symbol: str,
    data_dir: Path,
    max_samples: int = 10000
) -> pd.DataFrame:
    """Load sample data for a symbol"""
    symbol_dir = data_dir / f"symbol={symbol}"
    parquet_file = symbol_dir / f"{symbol}.parquet"
    
    if not parquet_file.exists():
        logger.warning(f"  Symbol {symbol} not found in dataset, skipping")
        raise FileNotFoundError(f"Data not found: {parquet_file}")
    
    df = pd.read_parquet(parquet_file)
    
    # Sample if too large
    if len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42)
    
    return df


def prepare_features_and_target(
    df: pd.DataFrame,
    target_column: str
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Prepare features and target for modeling"""
    
    # Check target exists
    if target_column not in df.columns:
        raise ValueError(f"Target '{target_column}' not in data")
    
    # Drop NaN in target
    df = df.dropna(subset=[target_column])
    
    if df.empty:
        raise ValueError("No valid data after dropping NaN in target")
    
    # LEAKAGE PREVENTION: Filter out leaking features
    import sys
    sys.path.insert(0, str(_REPO_ROOT / "scripts"))
    from filter_leaking_features import filter_features
    
    all_columns = df.columns.tolist()
    safe_columns = filter_features(all_columns, verbose=False)
    
    # Keep only safe features + target
    safe_columns_with_target = [c for c in safe_columns if c != target_column] + [target_column]
    df = df[safe_columns_with_target]
    
    # Prepare features (exclude target explicitly)
    X = df.drop(columns=[target_column], errors='ignore')
    
    # Drop object dtypes
    object_cols = X.select_dtypes(include=['object']).columns.tolist()
    if object_cols:
        X = X.drop(columns=object_cols)
    
    y = df[target_column]
    feature_names = X.columns.tolist()
    
    return X.to_numpy(), y.to_numpy(), feature_names


def train_and_evaluate_models(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    model_families: List[str] = None
) -> Tuple[Dict[str, float], float]:
    """
    Train multiple models and return scores + importance magnitude
    
    Returns:
        model_scores: Dict of R² scores per model
        mean_importance: Mean absolute feature importance
    """
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler
    import lightgbm as lgb
    
    if model_families is None:
        model_families = ['lightgbm', 'random_forest', 'neural_network']
    
    model_scores = {}
    importance_magnitudes = []
    
    # Determine task type (fixed detection)
    unique_vals = np.unique(y[~np.isnan(y)])
    is_binary = len(unique_vals) == 2 and set(unique_vals).issubset({0, 1, 0.0, 1.0})
    is_multiclass = len(unique_vals) <= 10 and all(isinstance(v, (int, np.integer)) or v.is_integer() for v in unique_vals)
    is_classification = is_binary or is_multiclass
    
    # Use R² for both (works for classification too, measures explained variance)
    scoring = 'r2'
    
    # LightGBM
    if 'lightgbm' in model_families:
        try:
            # GPU settings (will fallback to CPU if GPU not available)
            gpu_params = {}
            try:
                # Try CUDA first (fastest)
                test_model = lgb.LGBMRegressor(device='cuda', n_estimators=1, verbose=-1)
                test_model.fit(np.random.rand(10, 5), np.random.rand(10))
                gpu_params = {'device': 'cuda', 'gpu_device_id': 0}
                logger.info("  Using GPU (CUDA) for LightGBM")
            except:
                try:
                    # Try OpenCL
                    test_model = lgb.LGBMRegressor(device='gpu', n_estimators=1, verbose=-1)
                    test_model.fit(np.random.rand(10, 5), np.random.rand(10))
                    gpu_params = {'device': 'gpu', 'gpu_platform_id': 0, 'gpu_device_id': 0}
                    logger.info("  Using GPU (OpenCL) for LightGBM")
                except:
                    logger.info("  Using CPU for LightGBM")
            
            if is_binary:
                model = lgb.LGBMClassifier(
                    objective='binary',
                    n_estimators=200,
                    learning_rate=0.05,
                    num_leaves=31,
                    verbose=-1,
                    random_state=42,
                    **gpu_params
                )
            elif is_multiclass:
                n_classes = len(unique_vals)
                model = lgb.LGBMClassifier(
                    objective='multiclass',
                    num_class=n_classes,
                    n_estimators=200,
                    learning_rate=0.05,
                    num_leaves=31,
                    verbose=-1,
                    random_state=42,
                    **gpu_params
                )
            else:
                model = lgb.LGBMRegressor(
                    objective='regression',
                    n_estimators=200,
                    learning_rate=0.05,
                    num_leaves=31,
                    verbose=-1,
                    random_state=42,
                    **gpu_params
                )
            
            scores = cross_val_score(model, X, y, cv=3, scoring=scoring, n_jobs=1)
            model_scores['lightgbm'] = scores.mean()
            
            # Train once to get importance
            model.fit(X, y)
            importance_magnitudes.append(np.mean(np.abs(model.feature_importances_)))
            
        except Exception as e:
            logger.warning(f"LightGBM failed: {e}")
    
    # Random Forest
    if 'random_forest' in model_families:
        try:
            from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
            
            if is_binary or is_multiclass:
                model = RandomForestClassifier(n_estimators=100, max_depth=10, 
                                              random_state=42, n_jobs=2)
            else:
                model = RandomForestRegressor(n_estimators=100, max_depth=10, 
                                             random_state=42, n_jobs=2)
            
            scores = cross_val_score(model, X, y, cv=3, scoring=scoring, n_jobs=1)
            model_scores['random_forest'] = scores.mean()
            
            model.fit(X, y)
            importance_magnitudes.append(np.mean(np.abs(model.feature_importances_)))
            
        except Exception as e:
            logger.warning(f"RandomForest failed: {e}")
    
    # Neural Network
    if 'neural_network' in model_families:
        try:
            from sklearn.neural_network import MLPRegressor, MLPClassifier
            from sklearn.impute import SimpleImputer
            
            # Handle NaN values (neural networks can't handle them)
            imputer = SimpleImputer(strategy='median')
            X_imputed = imputer.fit_transform(X)
            
            # Scale for NN
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_imputed)
            
            if is_binary or is_multiclass:
                model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200,
                                     early_stopping=True, random_state=42)
            else:
                model = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=200,
                                    early_stopping=True, random_state=42)
            
            scores = cross_val_score(model, X_scaled, y, cv=3, scoring=scoring, n_jobs=1)
            model_scores['neural_network'] = scores.mean()
            
            # Permutation importance magnitude (simplified)
            model.fit(X_scaled, y)
            baseline_score = model.score(X_scaled, y)
            perm_scores = []
            for i in range(min(10, X.shape[1])):  # Sample 10 features
                X_perm = X_scaled.copy()
                np.random.shuffle(X_perm[:, i])
                perm_score = model.score(X_perm, y)
                perm_scores.append(abs(baseline_score - perm_score))
            
            importance_magnitudes.append(np.mean(perm_scores))
            
        except Exception as e:
            logger.warning(f"NeuralNetwork failed: {e}")
    
    mean_importance = np.mean(importance_magnitudes) if importance_magnitudes else 0.0
    
    return model_scores, mean_importance


def calculate_composite_score(
    mean_r2: float,
    std_r2: float,
    mean_importance: float,
    n_models: int
) -> float:
    """
    Calculate composite predictability score
    
    Components:
    - Mean R²: Higher is better (0-1)
    - Consistency: Lower std is better
    - Importance magnitude: Higher is better
    - Model agreement: More models = more confidence
    """
    
    # Normalize components
    r2_component = max(0, mean_r2)  # 0-1
    consistency_component = 1.0 / (1.0 + std_r2)  # Higher when std is low
    importance_component = min(1.0, mean_importance / 100.0)  # Normalize to 0-1
    
    # Weighted average
    composite = (
        0.50 * r2_component +        # 50% weight on R²
        0.25 * consistency_component + # 25% on consistency
        0.25 * importance_component    # 25% on importance magnitude
    )
    
    # Bonus for more models (up to 10% boost)
    model_bonus = min(0.1, n_models * 0.02)
    composite = composite * (1.0 + model_bonus)
    
    return composite


def evaluate_target_predictability(
    target_name: str,
    target_config: Dict[str, Any],
    symbols: List[str],
    data_dir: Path,
    model_families: List[str]
) -> TargetPredictabilityScore:
    """Evaluate predictability of a single target across symbols"""
    
    target_column = target_config['target_column']
    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluating: {target_name} ({target_column})")
    logger.info(f"{'='*60}")
    
    all_model_scores = []
    all_importances = []
    
    for symbol in symbols:
        try:
            logger.info(f"  {symbol}...")
            
            # Load data
            df = load_sample_data(symbol, data_dir, max_samples=10000)
            
            # Prepare features
            X, y, feature_names = prepare_features_and_target(df, target_column)
            
            # Check if target is degenerate in this sample (single class)
            unique_vals = np.unique(y)
            if len(unique_vals) == 1:
                logger.warning(f"    ⚠️  Skipping: Target has only 1 unique value in sample")
                continue
            
            # Train and evaluate
            model_scores, importance = train_and_evaluate_models(
                X, y, feature_names, model_families
            )
            
            if model_scores:
                all_model_scores.append(model_scores)
                all_importances.append(importance)
                
                scores_str = ", ".join([f"{k}={v:.3f}" for k, v in model_scores.items()])
                logger.info(f"    ✓ Scores: {scores_str}, importance={importance:.2f}")
            
        except Exception as e:
            logger.warning(f"    ✗ Failed: {e}")
            continue
    
    if not all_model_scores:
        logger.warning(f"  ⚠️  No successful evaluations for {target_name} (skipping)")
        return TargetPredictabilityScore(
            target_name=target_name,
            target_column=target_column,
            mean_r2=-999.0,  # Flag for degenerate/failed targets
            std_r2=1.0,
            mean_importance=0.0,
            consistency=0.0,
            n_models=0,
            model_scores={}
        )
    
    # Aggregate across symbols and models
    all_scores_by_model = defaultdict(list)
    for scores_dict in all_model_scores:
        for model_name, score in scores_dict.items():
            all_scores_by_model[model_name].append(score)
    
    # Calculate statistics
    model_means = {model: np.mean(scores) for model, scores in all_scores_by_model.items()}
    mean_r2 = np.mean(list(model_means.values()))
    std_r2 = np.std(list(model_means.values()))
    mean_importance = np.mean(all_importances)
    consistency = 1.0 - (std_r2 / (abs(mean_r2) + 1e-6))
    
    # Composite score
    composite = calculate_composite_score(
        mean_r2, std_r2, mean_importance, len(all_scores_by_model)
    )
    
    result = TargetPredictabilityScore(
        target_name=target_name,
        target_column=target_column,
        mean_r2=mean_r2,
        std_r2=std_r2,
        mean_importance=mean_importance,
        consistency=consistency,
        n_models=len(all_scores_by_model),
        model_scores=model_means,
        composite_score=composite
    )
    
    logger.info(f"  📊 Summary: R²={mean_r2:.3f}±{std_r2:.3f}, "
               f"importance={mean_importance:.2f}, composite={composite:.3f}")
    
    return result


def save_rankings(
    results: List[TargetPredictabilityScore],
    output_dir: Path
):
    """Save target predictability rankings"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Sort by composite score
    results = sorted(results, key=lambda x: x.composite_score, reverse=True)
    
    # Create DataFrame
    df = pd.DataFrame([{
        'rank': i + 1,
        'target_name': r.target_name,
        'target_column': r.target_column,
        'composite_score': r.composite_score,
        'mean_r2': r.mean_r2,
        'std_r2': r.std_r2,
        'mean_importance': r.mean_importance,
        'consistency': r.consistency,
        'n_models': r.n_models,
        **{f'{model}_r2': score for model, score in r.model_scores.items()},
        'recommendation': _get_recommendation(r)
    } for i, r in enumerate(results)])
    
    # Save CSV
    df.to_csv(output_dir / "target_predictability_rankings.csv", index=False)
    logger.info(f"\n✅ Saved rankings to target_predictability_rankings.csv")
    
    # Save YAML with recommendations
    yaml_data = {
        'target_rankings': [
            {
                'rank': i + 1,
                'target': r.target_name,
                'composite_score': float(r.composite_score),
                'mean_r2': float(r.mean_r2),
                'recommendation': _get_recommendation(r)
            }
            for i, r in enumerate(results)
        ]
    }
    
    with open(output_dir / "target_predictability_rankings.yaml", 'w') as f:
        yaml.dump(yaml_data, f, default_flow_style=False)
    
    logger.info(f"✅ Saved YAML to target_predictability_rankings.yaml")


def _get_recommendation(score: TargetPredictabilityScore) -> str:
    """Get recommendation based on predictability score"""
    if score.composite_score >= 0.7:
        return "PRIORITIZE - Strong predictive signal"
    elif score.composite_score >= 0.5:
        return "ENABLE - Good predictive signal"
    elif score.composite_score >= 0.3:
        return "TEST - Moderate signal, worth exploring"
    else:
        return "DEPRIORITIZE - Weak signal, low ROI"


def main():
    parser = argparse.ArgumentParser(
        description="Rank target predictability across model families"
    )
    parser.add_argument("--symbols", type=str, default="AAPL,MSFT,GOOGL,TSLA,JPM",
                       help="Symbols to test on (default: 5 representative stocks)")
    parser.add_argument("--data-dir", type=Path,
                       default=_REPO_ROOT / "data/data_labeled/interval=5m")
    parser.add_argument("--output-dir", type=Path,
                       default=_REPO_ROOT / "results/target_rankings")
    parser.add_argument("--targets", type=str,
                       help="Specific targets to evaluate (comma-separated), default: all enabled")
    parser.add_argument("--discover-all", action="store_true",
                       help="Auto-discover and rank ALL targets from data (ignores config)")
    parser.add_argument("--model-families", type=str,
                       default="lightgbm,random_forest,neural_network",
                       help="Model families to use")
    
    args = parser.parse_args()
    
    # Parse inputs
    symbols = [s.strip() for s in args.symbols.split(',')]
    model_families = [m.strip() for m in args.model_families.split(',')]
    
    logger.info("="*80)
    logger.info("🎯 Target Predictability Ranking")
    logger.info("="*80)
    logger.info(f"Test symbols: {', '.join(symbols)}")
    logger.info(f"Model families: {', '.join(model_families)}")
    
    # Discover or load targets
    if args.discover_all:
        logger.info("🔍 Auto-discovering ALL targets from data...")
        targets_to_eval = discover_all_targets(symbols[0], args.data_dir)
        logger.info(f"Found {len(targets_to_eval)} valid targets\n")
    else:
        # Load target configs
        target_configs = load_target_configs()
        
        # Filter targets
        if args.targets:
            requested = [t.strip() for t in args.targets.split(',')]
            targets_to_eval = {k: v for k, v in target_configs.items() if k in requested}
        else:
            # Only evaluate enabled targets
            targets_to_eval = {k: v for k, v in target_configs.items() if v.get('enabled', False)}
        
        logger.info(f"Evaluating {len(targets_to_eval)} targets\n")
    
    # Evaluate each target
    results = []
    for target_name, target_config in targets_to_eval.items():
        result = evaluate_target_predictability(
            target_name, target_config, symbols, args.data_dir, model_families
        )
        # Skip degenerate targets (marked with mean_r2 = -999)
        if result.mean_r2 != -999.0:
            results.append(result)
    
    # Save rankings
    save_rankings(results, args.output_dir)
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("📊 TARGET PREDICTABILITY RANKINGS")
    logger.info("="*80)
    
    results = sorted(results, key=lambda x: x.composite_score, reverse=True)
    
    for i, result in enumerate(results, 1):
        logger.info(f"\n{i:2d}. {result.target_name:25s} | Score: {result.composite_score:.3f}")
        logger.info(f"    R²: {result.mean_r2:.3f} ± {result.std_r2:.3f}")
        logger.info(f"    Importance: {result.mean_importance:.2f}")
        logger.info(f"    Recommendation: {_get_recommendation(result)}")
    
    logger.info("\n" + "="*80)
    logger.info("✅ Target ranking complete!")
    logger.info(f"Results saved to: {args.output_dir}")
    logger.info("="*80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

