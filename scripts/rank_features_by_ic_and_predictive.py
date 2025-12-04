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
Feature Ranking by IC (Information Coefficient) and Predictive Power

This script provides a clear, actionable workflow:
1. Ranks features by IC (correlation with target) - simple, interpretable
2. Ranks features by predictive power (model importance) - what models find useful
3. Combines with your target rankings - focuses on your best targets
4. Provides clear recommendations - tells you what to do next

IC (Information Coefficient) = Correlation between feature and target
- High IC = feature moves with target (good predictor)
- Low IC = feature doesn't correlate (may still be useful in combination)

Predictive Power = Model-based feature importance
- What models actually use to make predictions
- More sophisticated than IC (captures non-linear relationships)
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
import yaml
import warnings

# Add project root FIRST (before any scripts.* imports)
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Import checkpoint utility (after path is set)
from scripts.utils.checkpoint import CheckpointManager

# Setup logging with journald support
from scripts.utils.logging_setup import setup_logging
logger = setup_logging(
    script_name="rank_features_by_ic_and_predictive",
    level=logging.INFO,
    use_journald=True
)

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


@dataclass
class FeatureICScore:
    """IC and predictive power metrics for a feature"""
    feature_name: str
    target_name: str
    
    # IC metrics (correlation-based)
    ic_mean: float = 0.0  # Mean IC across symbols
    ic_std: float = 0.0  # Std of IC (consistency)
    ic_abs_mean: float = 0.0  # Mean absolute IC (magnitude)
    ic_rank: int = 0  # Rank by IC
    
    # Predictive power (model-based)
    predictive_power_mean: float = 0.0  # Mean importance across models
    predictive_power_std: float = 0.0
    predictive_rank: int = 0  # Rank by predictive power
    
    # Per-model importance (NEW)
    model_importances: Dict[str, float] = None  # Dict of {model_name: importance}
    
    # Combined score
    combined_score: float = 0.0  # Weighted combination
    combined_rank: int = 0  # Final rank
    
    # Target quality (from your target rankings)
    target_r2: float = 0.0  # Target's R² score
    target_rank: int = 0  # Target's rank in your rankings
    
    def __post_init__(self):
        if self.model_importances is None:
            self.model_importances = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'FeatureICScore':
        """Create from dictionary"""
        return cls(**d)


def load_target_rankings(rankings_path: Path) -> Dict[str, Dict[str, Any]]:
    """Load target predictability rankings"""
    if not rankings_path.exists():
        logger.warning(f"Target rankings not found: {rankings_path}")
        return {}
    
    with open(rankings_path) as f:
        if rankings_path.suffix == '.yaml':
            data = yaml.safe_load(f)
        else:
            data = json.load(f)
    
    # Extract target info - handle different YAML structures
    targets = {}
    
    if 'target_rankings' in data:
        # New format: target_rankings list
        for item in data['target_rankings']:
            target_name = item.get('target', '')
            # Try to find target column - may need to construct from name
            # Common patterns: 
            #   valle60m_0.8 -> y_will_valley_60m_0.8
            #   peak_60m_0.8 -> y_will_peak_60m_0.8
            #   vallemdd_15m_0.005 -> y_will_valley_mdd_15m_0.005
            target_col = item.get('target_column', '')
            if not target_col:
                # Try to construct from target name
                target_lower = target_name.lower()
                if 'valle' in target_lower:
                    # Handle valle60m_0.8 or vallemdd_15m_0.005
                    if 'valle60m' in target_lower or 'valle_60m' in target_lower:
                        target_col = f"y_will_valley_60m_{target_name.split('_')[-1]}"
                    elif 'vallemdd' in target_lower:
                        # vallemdd_15m_0.005 -> y_will_valley_mdd_15m_0.005
                        parts = target_name.split('_')
                        if len(parts) >= 3:
                            target_col = f"y_will_valley_mdd_{parts[1]}_{parts[2]}"
                        else:
                            target_col = f"y_will_valley_mdd_{target_name.split('_', 1)[-1]}"
                    else:
                        # Generic valley
                        target_col = f"y_will_valley_{target_name.split('_', 1)[-1]}"
                elif 'peak' in target_lower:
                    # Handle peak_60m_0.8 or peak_mfe_15m_0.005
                    if 'peak_mfe' in target_lower:
                        # peak_mfe_15m_0.005 -> y_will_peak_mfe_15m_0.005
                        parts = target_name.split('_')
                        if len(parts) >= 3:
                            target_col = f"y_will_peak_mfe_{parts[2]}_{parts[3]}"
                        else:
                            target_col = f"y_will_peak_mfe_{target_name.split('_', 2)[-1]}"
                    else:
                        # peak_60m_0.8 -> y_will_peak_60m_0.8
                        target_col = f"y_will_peak_{target_name.split('_', 1)[-1]}"
                elif 'swing' in target_lower:
                    # swing_high_60m_0.05 -> y_will_swing_high_60m_0.05
                    target_col = f"y_will_{target_name}"
                else:
                    # Generic: just add y_ prefix
                    target_col = f"y_{target_name}"
            
            r2 = item.get('mean_r2', 0.0)
            if isinstance(r2, str) and r2 == '.nan':
                r2 = 0.0
            elif r2 is None or (isinstance(r2, float) and np.isnan(r2)):
                r2 = 0.0
            
            targets[target_col] = {
                'r2': float(r2),
                'rank': item.get('rank', 999),
                'name': target_name
            }
    elif 'rankings' in data:
        # Alternative format
        for i, item in enumerate(data['rankings'], 1):
            target_col = item.get('target_column', '')
            if not target_col:
                target_col = item.get('target_name', '')
            
            r2 = item.get('mean_r2', 0.0)
            if isinstance(r2, str) and r2 == '.nan':
                r2 = 0.0
            elif r2 is None or (isinstance(r2, float) and np.isnan(r2)):
                r2 = 0.0
            
            targets[target_col] = {
                'r2': float(r2),
                'rank': i,
                'name': item.get('target_name', target_col)
            }
    
    # Filter out targets with R² <= 0 (not predictable)
    targets = {k: v for k, v in targets.items() if v['r2'] > 0}
    
    logger.info(f"Loaded {len(targets)} valid targets from rankings")
    return targets


def load_sample_data(
    symbol: str,
    data_dir: Path,
    max_samples: int = 50000
) -> pd.DataFrame:
    """Load sample data for a symbol"""
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
        raise FileNotFoundError(f"Data not found for {symbol}")
    
    df = pd.read_parquet(data_path)
    
    if len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42)
    
    return df


def filter_safe_features(df: pd.DataFrame, target_column: str = None) -> List[str]:
    """Filter out leaking features"""
    sys.path.insert(0, str(_REPO_ROOT / "scripts"))
    from filter_leaking_features import filter_features
    
    all_columns = df.columns.tolist()
    safe_columns = filter_features(all_columns, verbose=False)
    
    if target_column and target_column in safe_columns:
        safe_columns = [c for c in safe_columns if c != target_column]
    
    numeric_cols = df[safe_columns].select_dtypes(include=[np.number]).columns.tolist()
    return numeric_cols


def compute_ic(
    feature: pd.Series,
    target: pd.Series
) -> float:
    """Compute Information Coefficient (correlation)"""
    # Align and remove NaN
    aligned = pd.DataFrame({'feature': feature, 'target': target}).dropna()
    
    if len(aligned) < 10:  # Need minimum samples
        return 0.0
    
    try:
        # Check for zero variance (constant values)
        if aligned['feature'].std() == 0 or aligned['target'].std() == 0:
            return 0.0
        
        # Check for all NaN after dropna
        if aligned.empty or len(aligned) < 2:
            return 0.0
        
        # Compute correlation with error handling
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            ic = aligned['feature'].corr(aligned['target'])
        
        return ic if not (np.isnan(ic) or np.isinf(ic)) else 0.0
    except (ValueError, TypeError, ZeroDivisionError):
        return 0.0


def compute_predictive_power(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    model_families: List[str] = None,
    multi_model_config: Dict[str, Any] = None,
    target_column: str = None  # For horizon extraction and purge calculation
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    """
    Compute model-based feature importance using multiple model families
    
    Returns:
        aggregated_importance: Dict[feature_name -> mean_importance]
        per_model_importance: Dict[model_name -> Dict[feature_name -> importance]]
    """
    import lightgbm as lgb
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    
    if model_families is None:
        # Load from multi-model config if available
        if multi_model_config:
            model_families = [
                name for name, config in multi_model_config.get('model_families', {}).items()
                if config.get('enabled', False)
            ]
        else:
            model_families = ['lightgbm', 'random_forest', 'xgboost', 'neural_network']
    
    logger.debug(f"    Using model families: {model_families}")
    
    # Validate target before training
    sys.path.insert(0, str(_REPO_ROOT / "scripts" / "utils"))
    try:
        from target_validation import validate_target
        is_valid, error_msg = validate_target(y, min_samples=10, min_class_samples=2)
        if not is_valid:
            logger.debug(f"    Skipping models: {error_msg}")
            return {}, {}
    except ImportError:
        # Fallback to basic check
        y_clean = y[~np.isnan(y)]
        if len(y_clean) < 10:
            return {}, {}
        unique_vals = np.unique(y_clean)
        if len(unique_vals) < 2:
            return {}, {}
    
    # Determine task type - be more conservative (default to regression)
    y_clean = y[~np.isnan(y)]
    unique_vals = np.unique(y_clean)
    n_unique = len(unique_vals)
    
    # Check if binary (exactly 2 values, both 0/1)
    unique_set = set(unique_vals)
    is_binary = (
        n_unique == 2 and 
        unique_set.issubset({0, 1, 0.0, 1.0})
    )
    
    # Check if multiclass (3-10 integer values, not binary)
    is_multiclass = (
        not is_binary and
        n_unique >= 3 and
        n_unique <= 10 and
        all(isinstance(v, (int, np.integer)) or 
            (isinstance(v, float) and not np.isnan(v) and float(v).is_integer())
            for v in unique_vals)
    )
    
    # Default to regression for continuous targets or if uncertain
    use_classification = is_binary or is_multiclass
    
    # Impute
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    
    # GPU detection
    gpu_params = {}
    try:
        test_model = lgb.LGBMRegressor(device='cuda', n_estimators=1, verbose=-1)
        test_model.fit(np.random.rand(10, 5), np.random.rand(10))
        gpu_params = {'device': 'cuda', 'gpu_device_id': 0}
    except:
        pass
    
    all_importances = defaultdict(list)
    per_model_importances = defaultdict(lambda: defaultdict(list))  # model -> feature -> [importances]
    
    # Load model configs if available
    model_configs = {}
    if multi_model_config:
        model_configs = multi_model_config.get('model_families', {})
    
    # LightGBM
    if 'lightgbm' in model_families:
        try:
            # Use regression by default (safer for financial targets)
            # Only use classification if we're very confident
            if is_binary:
                # Binary classification - don't specify num_class (auto-detects as binary)
                model = lgb.LGBMClassifier(
                    n_estimators=100,
                    max_depth=6,
                    objective='binary',  # Explicitly set binary
                    verbose=-1,
                    random_state=42,
                    **gpu_params
                )
            elif is_multiclass:
                # Multiclass - need to specify num_class
                model = lgb.LGBMClassifier(
                    n_estimators=100,
                    max_depth=6,
                    num_class=n_unique,  # Number of classes
                    objective='multiclass',
                    verbose=-1,
                    random_state=42,
                    **gpu_params
                )
            else:
                # Regression (default for continuous targets)
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
                    for i, feat in enumerate(feature_names):
                        imp = model.feature_importances_[i]
                        all_importances[feat].append(imp)
                        per_model_importances['lightgbm'][feat].append(imp)
            except (ValueError, TypeError) as e:
                error_str = str(e).lower()
                if any(kw in error_str for kw in ['invalid classes', 'number of classes', 'too few']):
                    logger.debug(f"    LightGBM: Target degenerate - {e}")
                else:
                    logger.debug(f"LightGBM failed: {e}")
        except Exception as e:
            logger.debug(f"LightGBM failed: {e}")
    
    # XGBoost
    if 'xgboost' in model_families:
        try:
            import xgboost as xgb
            
            if is_binary:
                model = xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=6,
                    objective='binary:logistic',
                    verbosity=0,
                    random_state=42
                )
            elif is_multiclass:
                model = xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=6,
                    objective='multi:softprob',
                    num_class=n_unique,
                    verbosity=0,
                    random_state=42
                )
            else:
                model = xgb.XGBRegressor(
                    n_estimators=100,
                    max_depth=6,
                    objective='reg:squarederror',
                    verbosity=0,
                    random_state=42
                )
            
            try:
                model.fit(X_imputed, y)
                if hasattr(model, 'feature_importances_'):
                    for i, feat in enumerate(feature_names):
                        imp = model.feature_importances_[i]
                        all_importances[feat].append(imp)
                        per_model_importances['xgboost'][feat].append(imp)
            except (ValueError, TypeError) as e:
                error_str = str(e).lower()
                if any(kw in error_str for kw in ['invalid classes', 'expected', 'too few']):
                    logger.debug(f"    XGBoost: Target degenerate - {e}")
                else:
                    logger.debug(f"XGBoost failed: {e}")
        except Exception as e:
            logger.debug(f"XGBoost failed: {e}")
    
    # Random Forest
    if 'random_forest' in model_families:
        try:
            # Use regression by default
            if is_binary:
                from sklearn.ensemble import RandomForestClassifier
                model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                    n_jobs=1
                )
            elif is_multiclass:
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
            
            model.fit(X_imputed, y)
            if hasattr(model, 'feature_importances_'):
                for i, feat in enumerate(feature_names):
                    imp = model.feature_importances_[i]
                    all_importances[feat].append(imp)
                    per_model_importances['random_forest'][feat].append(imp)
        except Exception as e:
            logger.debug(f"Random Forest failed: {e}")
    
    # Neural Network (permutation importance)
    if 'neural_network' in model_families:
        try:
            from sklearn.neural_network import MLPRegressor, MLPClassifier
            from sklearn.preprocessing import StandardScaler
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_imputed)
            
            if is_binary:
                model = MLPClassifier(
                    hidden_layer_sizes=(64, 32),
                    max_iter=200,
                    random_state=42,
                    early_stopping=True,
                    validation_fraction=0.1
                )
            elif is_multiclass:
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
                    importance = max(0, baseline_score - permuted_score)
                    all_importances[feat].append(importance)
                    per_model_importances['neural_network'][feat].append(importance)
            except (ValueError, TypeError) as e:
                error_str = str(e).lower()
                if any(kw in error_str for kw in ['least populated class', 'too few', 'invalid classes']):
                    logger.debug(f"    Neural Network: Target too imbalanced - {e}")
                else:
                    logger.debug(f"Neural Network failed: {e}")
        except Exception as e:
            logger.debug(f"Neural Network failed: {e}")
    
    # Histogram Gradient Boosting
    if 'histogram_gradient_boosting' in model_families:
        try:
            from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
            
            if is_binary or is_multiclass:
                model = HistGradientBoostingClassifier(
                    max_iter=100,
                    max_depth=6,
                    random_state=42
                )
            else:
                model = HistGradientBoostingRegressor(
                    max_iter=100,
                    max_depth=6,
                    random_state=42
                )
            
            model.fit(X_imputed, y)
            if hasattr(model, 'feature_importances_'):
                for i, feat in enumerate(feature_names):
                    imp = model.feature_importances_[i]
                    all_importances[feat].append(imp)
                    per_model_importances['histogram_gradient_boosting'][feat].append(imp)
        except Exception as e:
            logger.debug(f"Histogram Gradient Boosting failed: {e}")
    
    # CatBoost
    if 'catboost' in model_families:
        try:
            import catboost as cb
            
            if is_binary:
                model = cb.CatBoostClassifier(iterations=100, depth=6, verbose=False, random_seed=42)
            elif is_multiclass:
                model = cb.CatBoostClassifier(iterations=100, depth=6, verbose=False, random_seed=42)
            else:
                model = cb.CatBoostRegressor(iterations=100, depth=6, verbose=False, random_seed=42)
            
            try:
                model.fit(X_imputed, y)
                importance = model.get_feature_importance()
                for i, feat in enumerate(feature_names):
                    imp = importance[i] if i < len(importance) else 0.0
                    all_importances[feat].append(imp)
                    per_model_importances['catboost'][feat].append(imp)
            except (ValueError, TypeError) as e:
                error_str = str(e).lower()
                if any(kw in error_str for kw in ['invalid classes', 'expected', 'too few']):
                    logger.debug(f"    CatBoost: Target degenerate - {e}")
                else:
                    logger.debug(f"CatBoost failed: {e}")
        except ImportError:
            logger.debug("CatBoost not available (pip install catboost)")
        except Exception as e:
            logger.debug(f"CatBoost failed: {e}")
    
    # Lasso (L1 Regularization)
    if 'lasso' in model_families:
        try:
            from sklearn.linear_model import Lasso
            
            model = Lasso(alpha=0.1, max_iter=1000, random_state=42)
            model.fit(X_imputed, y)
            importance = np.abs(model.coef_)
            for i, feat in enumerate(feature_names):
                imp = importance[i] if i < len(importance) else 0.0
                all_importances[feat].append(imp)
                per_model_importances['lasso'][feat].append(imp)
        except Exception as e:
            logger.debug(f"Lasso failed: {e}")
    
    # Mutual Information (Information-theoretic, no model training)
    if 'mutual_information' in model_families:
        try:
            from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
            
            if is_binary or is_multiclass:
                importance = mutual_info_classif(X_imputed, y, random_state=42, discrete_features='auto')
            else:
                importance = mutual_info_regression(X_imputed, y, random_state=42, discrete_features='auto')
            
            for i, feat in enumerate(feature_names):
                imp = importance[i] if i < len(importance) else 0.0
                all_importances[feat].append(imp)
                per_model_importances['mutual_information'][feat].append(imp)
        except Exception as e:
            logger.debug(f"Mutual Information failed: {e}")
    
    # Univariate Feature Selection (F-test)
    if 'univariate_selection' in model_families:
        try:
            from sklearn.feature_selection import f_regression, f_classif
            
            if is_binary or is_multiclass:
                scores, pvalues = f_classif(X_imputed, y)
            else:
                scores, pvalues = f_regression(X_imputed, y)
            
            # Normalize scores (F-statistics can be very large)
            if np.max(scores) > 0:
                importance = scores / np.max(scores)
            else:
                importance = scores
            
            for i, feat in enumerate(feature_names):
                imp = importance[i] if i < len(importance) else 0.0
                all_importances[feat].append(imp)
                per_model_importances['univariate_selection'][feat].append(imp)
        except Exception as e:
            logger.debug(f"Univariate Selection failed: {e}")
    
    # Recursive Feature Elimination (RFE)
    if 'rfe' in model_families:
        try:
            from sklearn.feature_selection import RFE
            from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
            
            n_features_to_select = min(50, len(feature_names))
            
            if is_binary or is_multiclass:
                estimator = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=1)
            else:
                estimator = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=1)
            
            selector = RFE(estimator, n_features_to_select=n_features_to_select, step=5)
            selector.fit(X_imputed, y)
            
            # Convert ranking to importance (lower rank = more important)
            ranking = selector.ranking_
            importance = 1.0 / (ranking + 1e-6)  # Avoid division by zero
            
            for i, feat in enumerate(feature_names):
                imp = importance[i] if i < len(importance) else 0.0
                all_importances[feat].append(imp)
                per_model_importances['rfe'][feat].append(imp)
        except Exception as e:
            logger.debug(f"RFE failed: {e}")
    
    # Boruta
    if 'boruta' in model_families:
        try:
            from boruta import BorutaPy
            from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
            
            if is_binary or is_multiclass:
                rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=1)
            else:
                rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=1)
            
            boruta = BorutaPy(rf, n_estimators='auto', verbose=0, random_state=42, max_iter=100)
            boruta.fit(X_imputed, y)
            
            # Convert to importance: selected=1.0, tentative=0.5, rejected=0.1
            ranking = boruta.ranking_
            selected = boruta.support_
            importance = np.where(selected, 1.0, np.where(ranking == 2, 0.5, 0.1))
            
            for i, feat in enumerate(feature_names):
                imp = importance[i] if i < len(importance) else 0.0
                all_importances[feat].append(imp)
                per_model_importances['boruta'][feat].append(imp)
        except ImportError:
            logger.debug("Boruta not available (pip install Boruta)")
        except Exception as e:
            logger.debug(f"Boruta failed: {e}")
    
    # Stability Selection
    if 'stability_selection' in model_families:
        try:
            from sklearn.linear_model import LassoCV, LogisticRegressionCV
            from scripts.utils.purged_time_series_split import PurgedTimeSeriesSplit
            from scripts.utils.leakage_filtering import _extract_horizon, _load_leakage_config
            
            # Calculate purge_overlap from target horizon
            # CRITICAL: Auto-detect interval to prevent data leakage
            from scripts.utils.data_interval import detect_interval_from_dataframe
            try:
                # Try to detect from dataframe if available in calling context
                # If df is not available here, will use default with warning
                if 'df' in locals() or 'df' in globals():
                    df_var = locals().get('df') or globals().get('df')
                    if df_var is not None and hasattr(df_var, 'columns') and 'ts' in df_var.columns:
                        data_interval_minutes = detect_interval_from_dataframe(df_var, timestamp_column='ts', default=5)
                    else:
                        data_interval_minutes = 5
                        logger.warning("⚠️  Using hardcoded 5-minute interval assumption (dataframe not available)")
                else:
                    data_interval_minutes = 5
                    logger.warning(
                        "⚠️  Using hardcoded 5-minute interval assumption. "
                        "If your data uses a different interval, this may cause data leakage."
                    )
            except Exception as e:
                data_interval_minutes = 5
                logger.warning(f"⚠️  Failed to auto-detect interval: {e}, using default 5m")
            purge_buffer_bars = 5  # Safety buffer
            
            leakage_config = _load_leakage_config()
            target_horizon_minutes = _extract_horizon(target_column, leakage_config) if target_column else None
            
            if target_horizon_minutes is not None:
                target_horizon_bars = target_horizon_minutes // data_interval_minutes
                purge_overlap = target_horizon_bars + purge_buffer_bars
            else:
                # Fallback: conservative default (60m = 12 bars + 5 buffer)
                purge_overlap = 17
            
            # Create purged CV splitter
            # NOTE: Using row-count based purging (legacy). For better accuracy, use time-based purging:
            # purged_cv = PurgedTimeSeriesSplit(n_splits=3, purge_overlap_time=pd.Timedelta(minutes=target_horizon_minutes), time_column_values=timestamps)
            purged_cv = PurgedTimeSeriesSplit(n_splits=3, purge_overlap=purge_overlap)
            
            n_bootstrap = 50  # Reduced for speed
            stability_scores = np.zeros(X_imputed.shape[1])
            
            for _ in range(n_bootstrap):
                indices = np.random.choice(len(X_imputed), size=len(X_imputed), replace=True)
                X_boot, y_boot = X_imputed[indices], y[indices]
                
                try:
                    if is_binary or is_multiclass:
                        model = LogisticRegressionCV(Cs=10, cv=purged_cv, random_state=42, max_iter=1000, n_jobs=1)
                    else:
                        model = LassoCV(cv=purged_cv, random_state=42, max_iter=1000, n_jobs=1)
                    
                    model.fit(X_boot, y_boot)
                    coef = model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_
                    stability_scores += (np.abs(coef) > 1e-6).astype(int)
                except:
                    continue  # Skip failed bootstrap iterations
            
            # Normalize to 0-1 (fraction of times selected)
            importance = stability_scores / n_bootstrap
            
            for i, feat in enumerate(feature_names):
                imp = importance[i] if i < len(importance) else 0.0
                all_importances[feat].append(imp)
                per_model_importances['stability_selection'][feat].append(imp)
        except Exception as e:
            logger.debug(f"Stability Selection failed: {e}")
    
    # Aggregate across all models (for backward compatibility)
    feature_importance = {}
    for feat in feature_names:
        importances = all_importances.get(feat, [])
        feature_importance[feat] = np.mean(importances) if importances else 0.0
    
    # Build per-model breakdown (average across any repeats)
    per_model_importance = {}
    for model_name in model_families:
        per_model_importance[model_name] = {}
        for feat in feature_names:
            model_imps = per_model_importances.get(model_name, {}).get(feat, [])
            per_model_importance[model_name][feat] = np.mean(model_imps) if model_imps else 0.0
    
    return feature_importance, per_model_importance


def load_multi_model_config(config_path: Path = None) -> Dict[str, Any]:
    """Load multi-model feature selection configuration"""
    if config_path is None:
        config_path = _REPO_ROOT / "CONFIG" / "multi_model_feature_selection.yaml"
    
    if not config_path.exists():
        logger.debug(f"Multi-model config not found: {config_path}, using defaults")
        return None
    
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded multi-model config from {config_path}")
        return config
    except Exception as e:
        logger.warning(f"Failed to load multi-model config: {e}")
        return None


def rank_features_by_ic_and_predictive(
    symbols: List[str],
    data_dir: Path,
    targets: List[str],
    target_rankings: Dict[str, Dict[str, Any]] = None,
    model_families: List[str] = None,
    max_samples: int = 50000,
    ic_weight: float = 0.4,
    predictive_weight: float = 0.6,
    multi_model_config: Dict[str, Any] = None,
    min_cs: int = 10,
    max_cs_samples: Optional[int] = None
) -> List[FeatureICScore]:
    """
    Rank features by IC and predictive power across multiple targets
    
    Args:
        symbols: List of symbols
        data_dir: Data directory
        targets: List of target columns to evaluate
        target_rankings: Optional target rankings (for weighting)
        model_families: Model families for predictive power
        max_samples: Max samples per symbol
        ic_weight: Weight for IC score (default 0.4)
        predictive_weight: Weight for predictive power (default 0.6)
    """
    logger.info("="*70)
    logger.info("FEATURE RANKING BY IC AND PREDICTIVE POWER")
    logger.info("="*70)
    logger.info(f"Symbols: {', '.join(symbols)}")
    logger.info(f"Targets: {len(targets)} targets")
    logger.info(f"Model families: {', '.join(model_families or ['default'])}")
    logger.info(f"IC weight: {ic_weight}, Predictive weight: {predictive_weight}")
    logger.info(f"Cross-sectional: min_cs={min_cs}, max_cs_samples={max_cs_samples or 1000}")
    logger.info("")
    
    # Load all symbols at once (cross-sectional data loading)
    from scripts.utils.cross_sectional_data import load_mtf_data_for_ranking, prepare_cross_sectional_data_for_ranking
    from scripts.utils.leakage_filtering import filter_features_for_target
    
    logger.info(f"Loading data for {len(symbols)} symbols...")
    mtf_data = load_mtf_data_for_ranking(data_dir, symbols, max_rows_per_symbol=max_samples)
    
    if not mtf_data:
        logger.error("No data loaded for any symbols")
        return []
    
    # Aggregate metrics across targets (using cross-sectional data)
    all_ic_scores = defaultdict(lambda: defaultdict(list))  # feature -> target -> [ic values]
    all_predictive_scores = defaultdict(lambda: defaultdict(list))  # feature -> target -> [importance values]
    all_model_scores = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))  # feature -> model -> target -> [importances]
    
    # Process each target with cross-sectional data
    for target_col in targets:
        logger.info(f"Processing {target_col}...")
        
        # Check if target exists in any symbol
        target_found = False
        for symbol, df in mtf_data.items():
            if target_col in df.columns:
                target_found = True
                break
        
        if not target_found:
            logger.debug(f"  Target '{target_col}' not found in any symbol, skipping")
            continue
        
        # Apply leakage filtering to feature list BEFORE preparing data
        sample_df = next(iter(mtf_data.values()))
        all_columns = sample_df.columns.tolist()
        safe_columns = filter_features_for_target(all_columns, target_col, verbose=False)
        
        logger.info(f"  Filtered to {len(safe_columns)} safe features for {target_col}")
        
        # Prepare cross-sectional data (matches training pipeline)
        X, y, feature_names, symbols_array, time_vals = prepare_cross_sectional_data_for_ranking(
            mtf_data, target_col, min_cs=min_cs, max_cs_samples=max_cs_samples, feature_names=safe_columns
        )
        
        if X is None or y is None or len(feature_names) == 0:
            logger.warning(f"  Failed to prepare cross-sectional data for {target_col}")
            continue
        
        logger.info(f"  Cross-sectional data: {len(X)} samples, {len(feature_names)} features")
        
        # Compute IC for each feature (on cross-sectional data)
        # Convert to DataFrame for easier IC computation
        feature_df = pd.DataFrame(X, columns=feature_names)
        target_series = pd.Series(y)
        
        for feat in feature_names:
            if feat not in feature_df.columns:
                continue
            
            feature_series = feature_df[feat]
            ic = compute_ic(feature_series, target_series)
            if not np.isnan(ic):
                all_ic_scores[feat][target_col].append(ic)
        
        # Compute predictive power (model-based) on cross-sectional data
        try:
            if len(y) > 100:  # Need enough samples
                # Validate target before computing predictive power
                sys.path.insert(0, str(_REPO_ROOT / "scripts" / "utils"))
                try:
                    from target_validation import validate_target
                    is_valid, error_msg = validate_target(y, min_samples=100, min_class_samples=2)
                    if not is_valid:
                        logger.debug(f"    Skipping {target_col}: {error_msg}")
                        continue
                except ImportError:
                    # Fallback
                    unique_y = np.unique(y[~np.isnan(y)])
                    if len(unique_y) < 2:
                        logger.debug(f"    Skipping {target_col}: degenerate target")
                        continue
                
                predictive_scores, per_model_scores = compute_predictive_power(
                    X, y, feature_names, model_families, multi_model_config, target_column=target_col
                )
                
                for feat, importance in predictive_scores.items():
                    if importance > 0:
                        all_predictive_scores[feat][target_col].append(importance)
                
                # Store per-model scores
                for model_name, model_feat_scores in per_model_scores.items():
                    for feat, imp in model_feat_scores.items():
                        if imp > 0:
                            all_model_scores[feat][model_name][target_col].append(imp)
        except Exception as e:
            logger.debug(f"    Predictive power failed for {target_col}: {e}")
    
    # Aggregate across symbols and targets
    logger.info("\nAggregating scores...")
    
    all_scores = []
    
    # Get all unique features
    all_features = set(all_ic_scores.keys()) | set(all_predictive_scores.keys())
    
    for feat in all_features:
        # Aggregate IC across targets (weighted by target quality if available)
        ic_values = []
        ic_abs_values = []
        
        for target_col in targets:
            target_ics = all_ic_scores[feat].get(target_col, [])
            if target_ics:
                # Weight by target quality if rankings available
                weight = 1.0
                if target_rankings and target_col in target_rankings:
                    target_info = target_rankings[target_col]
                    weight = max(0.1, target_info.get('r2', 0.0))  # Weight by R²
                
                for ic in target_ics:
                    ic_values.append(ic * weight)
                    ic_abs_values.append(abs(ic) * weight)
        
        # Aggregate predictive power
        predictive_values = []
        for target_col in targets:
            target_importances = all_predictive_scores[feat].get(target_col, [])
            if target_importances:
                weight = 1.0
                if target_rankings and target_col in target_rankings:
                    target_info = target_rankings[target_col]
                    weight = max(0.1, target_info.get('r2', 0.0))
                
                for imp in target_importances:
                    predictive_values.append(imp * weight)
        
        # Compute statistics
        ic_mean = np.mean(ic_values) if ic_values else 0.0
        ic_std = np.std(ic_values) if len(ic_values) > 1 else 0.0
        ic_abs_mean = np.mean(ic_abs_values) if ic_abs_values else 0.0
        
        predictive_mean = np.mean(predictive_values) if predictive_values else 0.0
        predictive_std = np.std(predictive_values) if len(predictive_values) > 1 else 0.0
        
        # Get best target info (for display)
        best_target = None
        best_target_r2 = 0.0
        best_target_rank = 999
        
        for target_col in targets:
            if target_rankings and target_col in target_rankings:
                target_info = target_rankings[target_col]
                r2 = target_info.get('r2', 0.0)
                if r2 > best_target_r2:
                    best_target = target_col
                    best_target_r2 = r2
                    best_target_rank = target_info.get('rank', 999)
        
        # Aggregate per-model importances
        model_importances_dict = {}
        if model_families:
            for model_name in model_families:
                model_imps = []
                for target_col in targets:
                    target_imps = all_model_scores[feat][model_name].get(target_col, [])
                    if target_imps:
                        weight = 1.0
                        if target_rankings and target_col in target_rankings:
                            weight = max(0.1, target_rankings[target_col].get('r2', 0.0))
                        model_imps.extend([imp * weight for imp in target_imps])
                
                if model_imps:
                    model_importances_dict[model_name] = np.mean(model_imps)
                else:
                    model_importances_dict[model_name] = 0.0
        
        # Combined score
        # Normalize to 0-1 scale
        ic_normalized = min(1.0, max(0.0, (ic_abs_mean + 1.0) / 2.0))  # IC is -1 to 1, map to 0-1
        predictive_normalized = predictive_mean / (predictive_mean + 1e-6) if predictive_mean > 0 else 0.0
        # Normalize predictive by max (will do after all computed)
        
        score = FeatureICScore(
            feature_name=feat,
            target_name=best_target or targets[0] if targets else '',
            ic_mean=ic_mean,
            ic_std=ic_std,
            ic_abs_mean=ic_abs_mean,
            predictive_power_mean=predictive_mean,
            predictive_power_std=predictive_std,
            model_importances=model_importances_dict,
            target_r2=best_target_r2,
            target_rank=best_target_rank,
            combined_score=ic_weight * ic_normalized + predictive_weight * predictive_normalized
        )
        
        all_scores.append(score)
    
    # Normalize predictive scores (relative to max)
    max_predictive = max([s.predictive_power_mean for s in all_scores]) if all_scores else 1.0
    if max_predictive > 0:
        for score in all_scores:
            score.predictive_power_mean = score.predictive_power_mean / max_predictive
    
    # Recompute combined scores with normalized predictive
    for score in all_scores:
        ic_normalized = min(1.0, max(0.0, (score.ic_abs_mean + 1.0) / 2.0))
        predictive_normalized = score.predictive_power_mean
        score.combined_score = ic_weight * ic_normalized + predictive_weight * predictive_normalized
    
    # Rank
    all_scores.sort(key=lambda x: x.ic_abs_mean, reverse=True)
    for i, score in enumerate(all_scores, 1):
        score.ic_rank = i
    
    all_scores.sort(key=lambda x: x.predictive_power_mean, reverse=True)
    for i, score in enumerate(all_scores, 1):
        score.predictive_rank = i
    
    all_scores.sort(key=lambda x: x.combined_score, reverse=True)
    for i, score in enumerate(all_scores, 1):
        score.combined_rank = i
    
    logger.info(f"Ranked {len(all_scores)} features")
    
    return all_scores


def save_rankings(
    rankings: List[FeatureICScore],
    output_dir: Path,
    top_n: int = 100
):
    """Save feature rankings"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert to DataFrame (main rankings)
    records = [asdict(r) for r in rankings]
    df = pd.DataFrame(records)
    
    # Expand model_importances dict into columns
    if 'model_importances' in df.columns and len(df) > 0:
        model_cols = {}
        for idx, row in df.iterrows():
            model_imps = row['model_importances'] if isinstance(row['model_importances'], dict) else {}
            for model_name, imp in model_imps.items():
                col_name = f"importance_{model_name}"
                if col_name not in model_cols:
                    model_cols[col_name] = [0.0] * len(df)
                model_cols[col_name][idx] = imp
        
        # Add model importance columns
        for col_name, values in model_cols.items():
            df[col_name] = values
        
        # Drop the dict column
        df = df.drop(columns=['model_importances'])
    
    # Save main CSV
    csv_path = output_dir / "feature_rankings_ic_predictive.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved rankings to {csv_path}")
    
    # Create per-model breakdown CSV (feature x model matrix)
    if rankings and rankings[0].model_importances:
        model_names = list(rankings[0].model_importances.keys())
        model_breakdown = pd.DataFrame({
            'feature_name': [r.feature_name for r in rankings]
        })
        
        for model_name in model_names:
            model_breakdown[f"{model_name}_importance"] = [
                r.model_importances.get(model_name, 0.0) for r in rankings
            ]
        
        # Add combined metrics
        model_breakdown['combined_score'] = [r.combined_score for r in rankings]
        model_breakdown['ic_abs_mean'] = [r.ic_abs_mean for r in rankings]
        model_breakdown['predictive_power_mean'] = [r.predictive_power_mean for r in rankings]
        
        # Sort by combined score
        model_breakdown = model_breakdown.sort_values('combined_score', ascending=False)
        
        model_csv_path = output_dir / "feature_rankings_by_model.csv"
        model_breakdown.to_csv(model_csv_path, index=False)
        logger.info(f"Saved per-model breakdown to {model_csv_path}")
    
    # Save top N report
    report_path = output_dir / "feature_ranking_report.md"
    with open(report_path, 'w') as f:
        f.write("# Feature Ranking by IC and Predictive Power\n\n")
        f.write(f"**Total Features Ranked**: {len(rankings)}\n\n")
        f.write("## Top Features by Combined Score\n\n")
        f.write("| Rank | Feature | Combined | IC (abs) | Predictive | Best Target | Target R² |\n")
        f.write("|------|---------|----------|----------|------------|-------------|----------|\n")
        
        for i, score in enumerate(rankings[:top_n], 1):
            f.write(f"| {i} | `{score.feature_name}` | "
                   f"{score.combined_score:.4f} | "
                   f"{score.ic_abs_mean:.4f} | "
                   f"{score.predictive_power_mean:.4f} | "
                   f"`{score.target_name}` | "
                   f"{score.target_r2:.4f} |\n")
        
        f.write("\n## Metrics Explanation\n\n")
        f.write("- **Combined Score**: Weighted combination of IC and predictive power\n")
        f.write("- **IC (abs)**: Mean absolute correlation with targets (higher = better)\n")
        f.write("- **Predictive**: Model-based feature importance (higher = better)\n")
        f.write("- **Best Target**: Target where this feature performs best\n")
        f.write("- **Target R²**: R² score of the best target (from your rankings)\n")
    
    logger.info(f"Saved report to {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Rank features by IC (correlation) and predictive power"
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
        help='Data directory'
    )
    parser.add_argument(
        '--targets',
        type=str,
        nargs='+',
        default=None,
        help='Target columns to evaluate (default: auto-discover from rankings)'
    )
    parser.add_argument(
        '--target-rankings',
        type=Path,
        default=Path('results/final_clean/target_predictability_rankings.yaml'),
        help='Path to target rankings file'
    )
    parser.add_argument(
        '--top-n-targets',
        type=int,
        default=10,
        help='Use top N targets from rankings (if targets not specified)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('results/feature_ic_predictive_ranking'),
        help='Output directory'
    )
    parser.add_argument(
        '--ic-weight',
        type=float,
        default=0.4,
        help='Weight for IC score (default: 0.4)'
    )
    parser.add_argument(
        '--predictive-weight',
        type=float,
        default=0.6,
        help='Weight for predictive power (default: 0.6)'
    )
    parser.add_argument(
        '--model-families',
        type=str,
        nargs='+',
        default=None,
        help='Model families for predictive power (default: use all enabled from multi_model_feature_selection.yaml)'
    )
    parser.add_argument(
        '--multi-model-config',
        type=Path,
        default=None,
        help='Path to multi-model config (default: CONFIG/multi_model_feature_selection.yaml)'
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
    parser.add_argument(
        '--min-cs',
        type=int,
        default=10,
        help='Minimum cross-sectional size per timestamp (default: 10)'
    )
    parser.add_argument(
        '--max-cs-samples',
        type=int,
        default=None,
        help='Maximum samples per timestamp for cross-sectional sampling (default: 1000)'
    )
    
    args = parser.parse_args()
    
    symbols = [s.strip() for s in args.symbols.split(',')]
    
    # Load multi-model config
    multi_model_config = None
    if args.multi_model_config:
        multi_model_config = load_multi_model_config(args.multi_model_config)
    else:
        multi_model_config = load_multi_model_config()  # Try default path
    
    # Determine model families
    if args.model_families:
        model_families = args.model_families
    elif multi_model_config:
        # Use enabled models from config
        model_families = [
            name for name, config in multi_model_config.get('model_families', {}).items()
            if config.get('enabled', False)
        ]
        logger.info(f"Using {len(model_families)} model families from config: {', '.join(model_families)}")
    else:
        # Default fallback
        model_families = ['lightgbm', 'random_forest', 'xgboost', 'neural_network']
        logger.info(f"Using default model families: {', '.join(model_families)}")
    
    # Load target rankings
    target_rankings = load_target_rankings(args.target_rankings)
    
    # Determine targets
    if args.targets:
        targets = args.targets
    else:
        # Use top N targets from rankings
        if target_rankings:
            sorted_targets = sorted(
                target_rankings.items(),
                key=lambda x: x[1].get('r2', 0.0),
                reverse=True
            )
            targets = [t[0] for t in sorted_targets[:args.top_n_targets]]
            logger.info(f"Using top {len(targets)} targets from rankings")
        else:
            logger.error("No targets specified and no rankings found!")
            return
    
    # Initialize checkpoint manager
    checkpoint_file = args.output_dir / "checkpoint.json"
    checkpoint = CheckpointManager(
        checkpoint_file=checkpoint_file,
        item_key_fn=lambda item: item if isinstance(item, str) else item[0]  # target name
    )
    
    # Clear checkpoint if requested
    if args.clear_checkpoint:
        checkpoint.clear()
        logger.info("Cleared checkpoint - starting fresh")
    
    # Load completed targets
    completed = checkpoint.load_completed()
    logger.info(f"Found {len(completed)} completed target evaluations in checkpoint")
    
    # Filter targets based on checkpoint
    targets_to_process = targets
    if args.resume and completed:
        # Only process targets not in checkpoint
        targets_to_process = [t for t in targets if t not in completed]
        logger.info(f"Resuming: {len(targets_to_process)} targets remaining, {len(completed)} already completed")
    
    # Rank features (processes all targets, but we can checkpoint per target if needed)
    # For now, checkpoint the entire result
    rankings = rank_features_by_ic_and_predictive(
        symbols=symbols,
        data_dir=args.data_dir,
        targets=targets_to_process if not args.resume else targets,  # Process all if resuming to merge
        target_rankings=target_rankings,
        model_families=model_families,
        max_samples=args.max_samples,
        ic_weight=args.ic_weight,
        predictive_weight=args.predictive_weight,
        multi_model_config=multi_model_config,
        min_cs=args.min_cs,
        max_cs_samples=args.max_cs_samples
    )
    
    # Save checkpoint after processing
    if targets_to_process:
        checkpoint.save_item("_all_targets", [r.to_dict() for r in rankings])
    
    # Merge with checkpoint results if resuming
    if args.resume and completed:
        checkpoint_rankings = []
        for target, data in completed.items():
            if target != "_all_targets" and isinstance(data, list):
                checkpoint_rankings.extend([FeatureICScore.from_dict(d) for d in data])
        
        # Merge and deduplicate by feature_name + target_name
        existing_keys = {(r.feature_name, r.target_name) for r in rankings}
        for r in checkpoint_rankings:
            if (r.feature_name, r.target_name) not in existing_keys:
                rankings.append(r)
    
    # Save results
    save_rankings(rankings, args.output_dir)
    
    # Print top 20
    logger.info("\n" + "="*70)
    logger.info("TOP 20 FEATURES BY COMBINED SCORE")
    logger.info("="*70)
    for i, score in enumerate(rankings[:20], 1):
        logger.info(
            f"{i:2d}. {score.feature_name:40s} | "
            f"Combined: {score.combined_score:.4f} | "
            f"IC: {score.ic_abs_mean:.4f} | "
            f"Predictive: {score.predictive_power_mean:.4f} | "
            f"Target: {score.target_name}"
        )


if __name__ == '__main__':
    main()

