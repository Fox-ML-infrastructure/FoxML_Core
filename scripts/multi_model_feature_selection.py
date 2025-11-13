"""
Multi-Model Feature Selection Pipeline

Combines feature importance from multiple model families to find robust features
that have predictive power across diverse architectures.

Strategy:
1. Train multiple model families (tree-based, neural, specialized)
2. Extract importance using best method per family:
   - Tree models: Native feature_importances_ (gain/split)
   - Neural networks: SHAP TreeExplainer approximation or permutation
   - Linear models: Absolute coefficients
3. Aggregate across models AND symbols
4. Rank by consensus: features important across multiple model types

This avoids model-specific biases and finds truly predictive features.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np
import lightgbm as lgb
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import json
from collections import defaultdict
from dataclasses import dataclass
import warnings

# Add project root to path
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from CONFIG.config_loader import load_model_config
import yaml

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Suppress warnings from SHAP/sklearn
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='X does not have valid feature names')


@dataclass
class ModelFamilyConfig:
    """Configuration for a model family"""
    name: str
    importance_method: str  # 'native', 'shap', 'permutation'
    enabled: bool
    config: Dict[str, Any]
    weight: float = 1.0  # Weight in final aggregation


@dataclass
class ImportanceResult:
    """Result from a single model's feature importance calculation"""
    model_family: str
    symbol: str
    importance_scores: pd.Series
    method: str
    train_score: float


def load_multi_model_config(config_path: Path = None) -> Dict[str, Any]:
    """Load multi-model feature selection configuration"""
    if config_path is None:
        config_path = _REPO_ROOT / "CONFIG" / "multi_model_feature_selection.yaml"
    
    if not config_path.exists():
        logger.warning(f"Config not found: {config_path}, using defaults")
        return get_default_config()
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Loaded multi-model config from {config_path}")
    return config


def get_default_config() -> Dict[str, Any]:
    """Default configuration if file doesn't exist"""
    return {
        'model_families': {
            'lightgbm': {
                'enabled': True,
                'importance_method': 'native',
                'weight': 1.0,
                'config': {
                    'objective': 'regression_l1',
                    'metric': 'mae',
                    'n_estimators': 300,
                    'learning_rate': 0.05,
                    'num_leaves': 31,
                    'verbose': -1
                }
            },
            'xgboost': {
                'enabled': True,
                'importance_method': 'native',
                'weight': 1.0,
                'config': {
                    'objective': 'reg:squarederror',
                    'n_estimators': 300,
                    'learning_rate': 0.05,
                    'max_depth': 6,
                    'verbosity': 0
                }
            },
            'random_forest': {
                'enabled': True,
                'importance_method': 'native',
                'weight': 0.8,
                'config': {
                    'n_estimators': 200,
                    'max_depth': 15,
                    'max_features': 'sqrt',
                    'n_jobs': 4
                }
            },
            'neural_network': {
                'enabled': True,
                'importance_method': 'permutation',
                'weight': 1.2,
                'config': {
                    'hidden_layer_sizes': (128, 64),
                    'max_iter': 300,
                    'early_stopping': True,
                    'validation_fraction': 0.1
                }
            }
        },
        'aggregation': {
            'per_symbol_method': 'mean',
            'cross_model_method': 'weighted_mean',
            'require_min_models': 2,
            'consensus_threshold': 0.5
        },
        'sampling': {
            'max_samples_per_symbol': 50000,
            'validation_split': 0.2
        }
    }


def safe_load_dataframe(file_path: Path) -> pd.DataFrame:
    """Safely load a parquet file"""
    try:
        return pd.read_parquet(file_path)
    except Exception as e:
        logger.error(f"Failed to load {file_path}: {e}")
        raise


def extract_native_importance(model, feature_names: List[str]) -> pd.Series:
    """Extract native feature importance from tree-based models"""
    if hasattr(model, 'feature_importance'):
        # LightGBM
        importance = model.feature_importance(importance_type='gain')
    elif hasattr(model, 'feature_importances_'):
        # sklearn models (RF, XGBoost sklearn API, etc.)
        importance = model.feature_importances_
    elif hasattr(model, 'get_score'):
        # XGBoost native API
        score_dict = model.get_score(importance_type='gain')
        importance = np.array([score_dict.get(f, 0.0) for f in feature_names])
    else:
        raise ValueError("Model does not have native feature importance")
    
    return pd.Series(importance, index=feature_names)


def extract_shap_importance(model, X: np.ndarray, feature_names: List[str],
                           max_samples: int = 1000) -> pd.Series:
    """Extract SHAP-based feature importance"""
    try:
        import shap
    except ImportError:
        logger.warning("SHAP not available, falling back to permutation importance")
        return extract_permutation_importance(model, X, None, feature_names)
    
    # Sample for computational efficiency
    if len(X) > max_samples:
        indices = np.random.choice(len(X), max_samples, replace=False)
        X_sample = X[indices]
    else:
        X_sample = X
    
    try:
        # TreeExplainer for tree models
        if hasattr(model, 'tree_'):
            explainer = shap.TreeExplainer(model)
        else:
            # KernelExplainer for other models (slower)
            explainer = shap.KernelExplainer(model.predict, X_sample[:100])
        
        shap_values = explainer.shap_values(X_sample)
        
        # Handle multi-output or single output
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        
        # Mean absolute SHAP value per feature
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        
        return pd.Series(mean_abs_shap, index=feature_names)
    
    except Exception as e:
        logger.warning(f"SHAP extraction failed: {e}, falling back to permutation")
        return extract_permutation_importance(model, X, None, feature_names)


def extract_permutation_importance(model, X: np.ndarray, y: np.ndarray,
                                   feature_names: List[str],
                                   n_repeats: int = 5) -> pd.Series:
    """Extract permutation importance"""
    try:
        from sklearn.inspection import permutation_importance
        
        # Need y for permutation importance
        if y is None:
            logger.warning("No y provided for permutation importance, returning zeros")
            return pd.Series(0.0, index=feature_names)
        
        result = permutation_importance(
            model, X, y,
            n_repeats=n_repeats,
            random_state=42,
            n_jobs=1
        )
        
        return pd.Series(result.importances_mean, index=feature_names)
    
    except Exception as e:
        logger.error(f"Permutation importance failed: {e}")
        return pd.Series(0.0, index=feature_names)


def train_model_and_get_importance(
    model_family: str,
    family_config: Dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str]
) -> Tuple[Any, pd.Series, str]:
    """Train a single model family and extract importance"""
    
    importance_method = family_config['importance_method']
    model_config = family_config['config']
    
    # Train model based on family
    if model_family == 'lightgbm':
        lgb_data = lgb.Dataset(X, label=y, feature_name=feature_names)
        model = lgb.train(
            params=model_config,
            train_set=lgb_data,
            num_boost_round=model_config.get('n_estimators', 300),
            callbacks=[lgb.log_evaluation(period=0)]
        )
        train_score = model.best_score.get('training', {}).get(model_config.get('metric', 'l1'), 0.0)
    
    elif model_family == 'xgboost':
        try:
            import xgboost as xgb
            model = xgb.XGBRegressor(**model_config)
            model.fit(X, y)
            train_score = model.score(X, y)
        except ImportError:
            logger.error("XGBoost not available")
            return None, pd.Series(0.0, index=feature_names), importance_method, 0.0
    
    elif model_family == 'random_forest':
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(**model_config, random_state=42)
        model.fit(X, y)
        train_score = model.score(X, y)
    
    elif model_family == 'neural_network':
        from sklearn.neural_network import MLPRegressor
        from sklearn.preprocessing import StandardScaler
        from sklearn.impute import SimpleImputer
        
        # Handle NaN values (neural networks can't handle them)
        imputer = SimpleImputer(strategy='median')
        X_imputed = imputer.fit_transform(X)
        
        # Scale for neural networks
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_imputed)
        
        model = MLPRegressor(**model_config, random_state=42)
        model.fit(X_scaled, y)
        train_score = model.score(X_scaled, y)
        
        # Use scaled data for importance
        X = X_scaled
    
    else:
        logger.error(f"Unknown model family: {model_family}")
        return None, pd.Series(0.0, index=feature_names), importance_method, 0.0
    
    # Extract importance based on method
    if importance_method == 'native':
        importance = extract_native_importance(model, feature_names)
    elif importance_method == 'shap':
        importance = extract_shap_importance(model, X, feature_names)
    elif importance_method == 'permutation':
        importance = extract_permutation_importance(model, X, y, feature_names)
    else:
        logger.error(f"Unknown importance method: {importance_method}")
        importance = pd.Series(0.0, index=feature_names)
    
    return model, importance, importance_method, train_score


def process_single_symbol(
    symbol: str,
    data_path: Path,
    target_column: str,
    model_families_config: Dict[str, Dict[str, Any]],
    max_samples: int = 50000
) -> List[ImportanceResult]:
    """Process a single symbol with multiple model families"""
    
    results = []
    
    try:
        # Load data
        df = safe_load_dataframe(data_path)
        
        # Validate target
        if target_column not in df.columns:
            logger.warning(f"Skipping {symbol}: Target '{target_column}' not found")
            return results
        
        # Drop NaN in target
        df = df.dropna(subset=[target_column])
        if df.empty:
            logger.warning(f"Skipping {symbol}: No valid data after dropping NaN")
            return results
        
        # Sample if too large
        if len(df) > max_samples:
            df = df.sample(n=max_samples, random_state=42)
        
        # LEAKAGE PREVENTION: Filter out leaking features
        sys.path.insert(0, str(_REPO_ROOT / "scripts"))
        from filter_leaking_features import filter_features
        
        all_columns = df.columns.tolist()
        safe_columns = filter_features(all_columns, verbose=False)
        
        # Keep only safe features + target
        safe_columns_with_target = [c for c in safe_columns if c != target_column] + [target_column]
        df = df[safe_columns_with_target]
        
        # Prepare features (target already in safe list, so exclude it explicitly)
        X = df.drop(columns=[target_column], errors='ignore')
        
        # Drop object dtypes
        object_cols = X.select_dtypes(include=['object']).columns.tolist()
        if object_cols:
            X = X.drop(columns=object_cols)
        
        y = df[target_column]
        feature_names = X.columns.tolist()
        
        if not feature_names:
            logger.warning(f"Skipping {symbol}: No features after filtering")
            return results
        
        # Convert to numpy
        X_arr = X.to_numpy()
        y_arr = y.to_numpy()
        
        # Train each enabled model family
        for family_name, family_config in model_families_config.items():
            if not family_config.get('enabled', False):
                continue
            
            try:
                logger.info(f"  {symbol}: Training {family_name}...")
                model, importance, method, train_score = train_model_and_get_importance(
                    family_name, family_config, X_arr, y_arr, feature_names
                )
                
                if importance is not None and importance.sum() > 0:
                    result = ImportanceResult(
                        model_family=family_name,
                        symbol=symbol,
                        importance_scores=importance,
                        method=method,
                        train_score=train_score
                    )
                    results.append(result)
                    logger.info(f"    {family_name}: score={train_score:.4f}, "
                              f"top feature={importance.idxmax()} ({importance.max():.2f})")
                
            except Exception as e:
                logger.error(f"  {symbol}: {family_name} failed: {e}")
                continue
        
        logger.info(f"✅ {symbol}: Completed {len(results)}/{len(model_families_config)} models")
        
    except Exception as e:
        logger.error(f"❌ {symbol}: Processing failed: {e}", exc_info=True)
    
    return results


def aggregate_multi_model_importance(
    all_results: List[ImportanceResult],
    model_families_config: Dict[str, Dict[str, Any]],
    aggregation_config: Dict[str, Any],
    top_n: Optional[int] = None
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Aggregate feature importance across models AND symbols
    
    Strategy:
    1. Group by model family
    2. Aggregate within each family across symbols
    3. Weight by family weight
    4. Combine across families
    5. Rank by consensus
    """
    
    if not all_results:
        return pd.DataFrame(), []
    
    # Group results by model family
    family_results = defaultdict(list)
    for result in all_results:
        family_results[result.model_family].append(result)
    
    # Aggregate within each family
    family_scores = {}
    for family_name, results in family_results.items():
        # Combine importances across symbols for this family
        importances_df = pd.concat(
            [r.importance_scores for r in results],
            axis=1,
            sort=False
        ).fillna(0)
        
        # Aggregate across symbols (mean by default)
        method = aggregation_config.get('per_symbol_method', 'mean')
        if method == 'mean':
            family_score = importances_df.mean(axis=1)
        elif method == 'median':
            family_score = importances_df.median(axis=1)
        else:
            family_score = importances_df.mean(axis=1)
        
        # Apply family weight
        weight = model_families_config[family_name].get('weight', 1.0)
        family_scores[family_name] = family_score * weight
        
        logger.info(f"📊 {family_name}: Aggregated {len(results)} symbols, "
                   f"weight={weight}, top={family_score.idxmax()}")
    
    # Combine across families
    combined_df = pd.DataFrame(family_scores)
    
    # Calculate consensus score
    cross_model_method = aggregation_config.get('cross_model_method', 'weighted_mean')
    if cross_model_method == 'weighted_mean':
        consensus_score = combined_df.mean(axis=1)
    elif cross_model_method == 'median':
        consensus_score = combined_df.median(axis=1)
    elif cross_model_method == 'geometric_mean':
        # Geometric mean (good for multiplicative effects)
        consensus_score = np.exp(np.log(combined_df + 1e-10).mean(axis=1))
    else:
        consensus_score = combined_df.mean(axis=1)
    
    # Calculate consensus metrics
    n_models = combined_df.shape[1]
    frequency = (combined_df > 0).sum(axis=1)
    frequency_pct = (frequency / n_models) * 100
    
    # Standard deviation across models (lower = more consensus)
    consensus_std = combined_df.std(axis=1)
    
    # Create summary DataFrame
    summary_df = pd.DataFrame({
        'feature': consensus_score.index,
        'consensus_score': consensus_score.values,
        'n_models_agree': frequency,
        'consensus_pct': frequency_pct,
        'std_across_models': consensus_std,
    })
    
    # Add per-family scores
    for family_name in family_scores.keys():
        summary_df[f'{family_name}_score'] = combined_df[family_name].values
    
    # Sort by consensus score
    summary_df = summary_df.sort_values('consensus_score', ascending=False).reset_index(drop=True)
    
    # Filter by minimum consensus if specified
    min_models = aggregation_config.get('require_min_models', 1)
    summary_df = summary_df[summary_df['n_models_agree'] >= min_models]
    
    # Select top N
    if top_n:
        summary_df = summary_df.head(top_n)
    
    selected_features = summary_df['feature'].tolist()
    
    return summary_df, selected_features


def save_multi_model_results(
    summary_df: pd.DataFrame,
    selected_features: List[str],
    all_results: List[ImportanceResult],
    output_dir: Path,
    metadata: Dict[str, Any]
):
    """Save multi-model feature selection results"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Selected features list
    with open(output_dir / "selected_features.txt", "w") as f:
        for feature in selected_features:
            f.write(f"{feature}\n")
    logger.info(f"✅ Saved {len(selected_features)} features to selected_features.txt")
    
    # 2. Detailed summary CSV
    summary_df.to_csv(output_dir / "feature_importance_multi_model.csv", index=False)
    logger.info(f"✅ Saved detailed multi-model summary to feature_importance_multi_model.csv")
    
    # 3. Per-model-family breakdowns
    for family_name in summary_df.columns:
        if family_name.endswith('_score') and family_name not in ['consensus_score']:
            family_df = summary_df[['feature', family_name]].copy()
            family_df = family_df.sort_values(family_name, ascending=False)
            family_csv = output_dir / f"importance_{family_name.replace('_score', '')}.csv"
            family_df.to_csv(family_csv, index=False)
    
    # 4. Model agreement matrix
    model_families = list(set(r.model_family for r in all_results))
    agreement_matrix = pd.DataFrame(
        index=selected_features[:20],  # Top 20 for readability
        columns=model_families
    )
    
    for result in all_results:
        for feature in selected_features[:20]:
            if feature in result.importance_scores.index:
                current = agreement_matrix.loc[feature, result.model_family]
                score = result.importance_scores[feature]
                if pd.isna(current):
                    agreement_matrix.loc[feature, result.model_family] = score
                else:
                    agreement_matrix.loc[feature, result.model_family] = max(current, score)
    
    agreement_matrix.to_csv(output_dir / "model_agreement_matrix.csv")
    logger.info(f"✅ Saved model agreement matrix")
    
    # 5. Metadata JSON
    metadata['n_selected_features'] = len(selected_features)
    metadata['n_total_results'] = len(all_results)
    metadata['model_families_used'] = list(set(r.model_family for r in all_results))
    
    with open(output_dir / "multi_model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"✅ Saved metadata")


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Model Feature Selection: Find robust features across model families"
    )
    parser.add_argument("--symbols", type=str, 
                       help="Comma-separated symbols (default: all in data_dir)")
    parser.add_argument("--data-dir", type=Path,
                       default=_REPO_ROOT / "data/data_labeled/interval=5m",
                       help="Directory with labeled data")
    parser.add_argument("--output-dir", type=Path,
                       default=_REPO_ROOT / "DATA_PROCESSING/data/features/multi_model",
                       help="Output directory")
    parser.add_argument("--target-column", type=str,
                       default="y_will_peak_60m_0.8",
                       help="Target column for training")
    parser.add_argument("--top-n", type=int, default=60,
                       help="Number of features to select")
    parser.add_argument("--config", type=Path,
                       help="Path to multi-model config YAML")
    parser.add_argument("--num-workers", type=int, default=4,
                       help="Parallel workers (sequential per symbol)")
    parser.add_argument("--enable-families", type=str,
                       help="Comma-separated families to enable (e.g., lightgbm,xgboost,neural_network)")
    
    args = parser.parse_args()
    
    # Load config
    config = load_multi_model_config(args.config)
    
    # Override enabled families if specified
    if args.enable_families:
        enabled = [f.strip() for f in args.enable_families.split(',')]
        for family in config['model_families']:
            config['model_families'][family]['enabled'] = family in enabled
    
    # Count enabled families
    enabled_families = [f for f, cfg in config['model_families'].items() if cfg.get('enabled')]
    
    logger.info("="*80)
    logger.info("🚀 Multi-Model Feature Selection Pipeline")
    logger.info("="*80)
    logger.info(f"Target: {args.target_column}")
    logger.info(f"Top N: {args.top_n}")
    logger.info(f"Enabled model families ({len(enabled_families)}): {', '.join(enabled_families)}")
    logger.info(f"Output: {args.output_dir}")
    logger.info("-"*80)
    
    # Find symbols
    if not args.data_dir.exists():
        logger.error(f"❌ Data directory not found: {args.data_dir}")
        return 1
    
    symbol_dirs = [d for d in args.data_dir.glob("symbol=*") if d.is_dir()]
    labeled_files = []
    for symbol_dir in symbol_dirs:
        symbol_name = symbol_dir.name.replace("symbol=", "")
        parquet_file = symbol_dir / f"{symbol_name}.parquet"
        if parquet_file.exists():
            labeled_files.append((symbol_name, parquet_file))
    
    if args.symbols:
        requested = [s.upper().strip() for s in args.symbols.split(',')]
        labeled_files = [(sym, path) for sym, path in labeled_files if sym.upper() in requested]
    
    if not labeled_files:
        logger.error("❌ No labeled files found")
        return 1
    
    logger.info(f"📊 Processing {len(labeled_files)} symbols")
    
    # Process symbols (sequential to avoid GPU/memory conflicts)
    all_results = []
    for i, (symbol, path) in enumerate(labeled_files, 1):
        logger.info(f"\n[{i}/{len(labeled_files)}] Processing {symbol}...")
        results = process_single_symbol(
            symbol, path, args.target_column,
            config['model_families'],
            config['sampling']['max_samples_per_symbol']
        )
        all_results.extend(results)
    
    if not all_results:
        logger.error("❌ No results collected")
        return 1
    
    logger.info(f"\n{'='*80}")
    logger.info(f"📈 Aggregating {len(all_results)} model results...")
    logger.info(f"{'='*80}")
    
    # Aggregate across models and symbols
    summary_df, selected_features = aggregate_multi_model_importance(
        all_results,
        config['model_families'],
        config['aggregation'],
        args.top_n
    )
    
    if summary_df.empty:
        logger.error("❌ No features selected")
        return 1
    
    # Save results
    metadata = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'target_column': args.target_column,
        'top_n': args.top_n,
        'n_symbols': len(labeled_files),
        'enabled_families': enabled_families,
        'config': config
    }
    
    save_multi_model_results(
        summary_df, selected_features, all_results,
        args.output_dir, metadata
    )
    
    # Print summary
    logger.info(f"\n{'='*80}")
    logger.info("✅ Multi-Model Feature Selection Complete!")
    logger.info(f"{'='*80}")
    logger.info(f"\n📊 Top 10 Features by Consensus:")
    for i, row in summary_df.head(10).iterrows():
        logger.info(f"  {i+1:2d}. {row['feature']:30s} | "
                   f"score={row['consensus_score']:8.2f} | "
                   f"agree={row['n_models_agree']}/{len(enabled_families)} | "
                   f"std={row['std_across_models']:6.2f}")
    
    logger.info(f"\n📁 Output files:")
    logger.info(f"  • {args.output_dir}/selected_features.txt")
    logger.info(f"  • {args.output_dir}/feature_importance_multi_model.csv")
    logger.info(f"  • {args.output_dir}/model_agreement_matrix.csv")
    logger.info(f"  • {args.output_dir}/importance_<family>.csv (per-family)")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

