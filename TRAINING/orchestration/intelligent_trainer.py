"""
Copyright (c) 2025-2026 Fox ML Infrastructure LLC

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
Intelligent Training Orchestrator

Integrates target ranking and feature selection into the training pipeline
while preserving all existing functionality and leakage-free behavior.

This is a new entry point that wraps train_with_strategies.py, adding:
- Automatic target ranking and selection
- Automatic feature selection per target
- Caching of ranking/selection results
- Backward compatibility with existing workflows
"""

import sys
import os
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import json
import hashlib
import time
import datetime
import numpy as np

# Add project root to path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Add TRAINING to path
_TRAINING_ROOT = Path(__file__).resolve().parents[1]
if str(_TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(_TRAINING_ROOT))

# Import ranking/selection modules
from TRAINING.ranking import (
    rank_targets,
    discover_targets,
    load_target_configs,
    select_features_for_target,
    load_multi_model_config
)

# Import new config system (optional - for backward compatibility)
try:
    from CONFIG.config_builder import (
        build_feature_selection_config,
        build_target_ranking_config,
        build_training_config
    )
    from CONFIG.config_schemas import ExperimentConfig, FeatureSelectionConfig, TargetRankingConfig, TrainingConfig
    _NEW_CONFIG_AVAILABLE = True
except ImportError:
    _NEW_CONFIG_AVAILABLE = False
    logger.debug("New config system not available, using legacy configs")

# Import existing training pipeline functions
# We import the module but don't run main() - we call functions directly
import TRAINING.train_with_strategies as train_module
from TRAINING.train_with_strategies import (
    load_mtf_data,
    train_models_for_interval_comprehensive,
    ALL_FAMILIES
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import leakage sentinels
try:
    from TRAINING.common.leakage_sentinels import LeakageSentinel, SentinelResult
    _SENTINELS_AVAILABLE = True
except ImportError:
    _SENTINELS_AVAILABLE = False
    logger.debug("Leakage sentinels not available")

# Import pandas for sentinel diagnostics
try:
    import pandas as pd
    _PANDAS_AVAILABLE = True
except ImportError:
    _PANDAS_AVAILABLE = False


def _json_default(obj: Any) -> Any:
    """
    Fallback serializer for json.dump when saving ranking cache.
    Handles pandas / numpy / datetime objects.
    """
    # Datetime-like
    if isinstance(obj, (datetime.datetime, datetime.date)):
        # ISO-8601 string is human readable and round-trippable enough for our use
        return obj.isoformat()
    
    # Pandas Timestamp (must check after datetime since pd.Timestamp is a subclass)
    if _PANDAS_AVAILABLE:
        try:
            import pandas as pd
            if isinstance(obj, pd.Timestamp):
                return obj.isoformat()
        except ImportError:
            pass
    
    # Numpy scalars
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    
    # Numpy arrays
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    
    # Anything else falls back to string representation
    return str(obj)


class IntelligentTrainer:
    """
    Intelligent training orchestrator with integrated ranking and selection.
    """
    
    def __init__(
        self,
        data_dir: Path,
        symbols: List[str],
        output_dir: Path,
        cache_dir: Optional[Path] = None,
        add_timestamp: bool = True,
        experiment_config: Optional['ExperimentConfig'] = None  # New typed config (optional)
    ):
        """
        Initialize the intelligent trainer.
        
        Args:
            data_dir: Directory containing symbol data
            symbols: List of symbols to train on
            output_dir: Output directory for training results
            cache_dir: Optional cache directory for ranking/selection results
            add_timestamp: If True, append timestamp to output_dir to make runs distinguishable
            experiment_config: Optional ExperimentConfig object [NEW - preferred]
        """
        from datetime import datetime
        
        # NEW: Use experiment config if provided
        if experiment_config is not None and _NEW_CONFIG_AVAILABLE:
            self.data_dir = experiment_config.data_dir
            self.symbols = experiment_config.symbols
            self.experiment_config = experiment_config
        else:
            self.data_dir = Path(data_dir)
            self.symbols = symbols
            self.experiment_config = None
        
        # Add timestamp to output directory if requested
        if add_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path(output_dir)
            # Only add timestamp if output_dir doesn't already have one (avoid double-timestamping)
            if not any(c.isdigit() for c in output_dir.name[-15:]):
                self.output_dir = output_dir.parent / f"{output_dir.name}_{timestamp}"
            else:
                self.output_dir = output_dir
        else:
            self.output_dir = Path(output_dir)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"📁 Output directory: {self.output_dir}")
        
        self.cache_dir = Path(cache_dir) if cache_dir else self.output_dir / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "target_rankings").mkdir(exist_ok=True)
        (self.output_dir / "feature_selections").mkdir(exist_ok=True)
        (self.output_dir / "training_results").mkdir(exist_ok=True)
        (self.output_dir / "leakage_diagnostics").mkdir(exist_ok=True)
        
        # Cache paths
        self.target_ranking_cache = self.cache_dir / "target_rankings.json"
        self.feature_selection_cache = self.cache_dir / "feature_selections"
        self.feature_selection_cache.mkdir(parents=True, exist_ok=True)
        
        # Initialize leakage sentinel if available
        if _SENTINELS_AVAILABLE:
            self.sentinel = LeakageSentinel()
        else:
            self.sentinel = None
    
    def _get_cache_key(self, symbols: List[str], config_hash: str) -> str:
        """Generate cache key from symbols and config."""
        key_str = f"{sorted(symbols)}_{config_hash}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _load_cached_rankings(self, cache_key: str, use_cache: bool = True) -> Optional[List[Dict[str, Any]]]:
        """Load cached target rankings."""
        if not use_cache or not self.target_ranking_cache.exists():
            return None
        
        try:
            with open(self.target_ranking_cache, 'r') as f:
                cache_data = json.load(f)
                return cache_data.get(cache_key)
        except Exception as e:
            logger.warning(f"Failed to load ranking cache: {e}")
            return None
    
    def _save_cached_rankings(self, cache_key: str, rankings: List[Dict[str, Any]]):
        """Save target rankings to cache."""
        try:
            cache_data = {}
            if self.target_ranking_cache.exists():
                with open(self.target_ranking_cache, 'r') as f:
                    cache_data = json.load(f)
            
            cache_data[cache_key] = rankings
            with open(self.target_ranking_cache, 'w') as f:
                json.dump(cache_data, f, indent=2, default=_json_default)
        except Exception as e:
            logger.warning(f"Failed to save ranking cache: {e}")
    
    def _get_feature_cache_path(self, target: str) -> Path:
        """Get cache path for feature selection results."""
        return self.feature_selection_cache / f"{target}.json"
    
    def _load_cached_features(self, target: str) -> Optional[List[str]]:
        """Load cached feature selection for a target."""
        cache_path = self._get_feature_cache_path(target)
        if not cache_path.exists():
            return None
        
        try:
            with open(cache_path, 'r') as f:
                cache_data = json.load(f)
                return cache_data.get('selected_features')
        except Exception as e:
            logger.warning(f"Failed to load feature cache for {target}: {e}")
            return None
    
    def _save_cached_features(self, target: str, features: List[str]):
        """Save feature selection results to cache."""
        cache_path = self._get_feature_cache_path(target)
        try:
            cache_data = {
                'target': target,
                'selected_features': features,
                'timestamp': time.time()
            }
            with open(cache_path, 'w') as f:
                json.dump(cache_data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save feature cache for {target}: {e}")
    
    def rank_targets_auto(
        self,
        top_n: int = 5,
        model_families: Optional[List[str]] = None,
        multi_model_config: Optional[Dict[str, Any]] = None,
        force_refresh: bool = False,
        use_cache: bool = True,
        max_targets_to_evaluate: Optional[int] = None,  # Limit number of targets to evaluate (for faster testing)
        target_ranking_config: Optional['TargetRankingConfig'] = None  # New typed config (optional)
    ) -> List[str]:
        """
        Automatically rank targets and return top N.
        
        Args:
            top_n: Number of top targets to return
            model_families: Optional list of model families to use [LEGACY]
            multi_model_config: Optional multi-model config [LEGACY]
            force_refresh: If True, ignore cache and re-rank
            use_cache: If True, use cached rankings if available
            max_targets_to_evaluate: Optional limit on number of targets to evaluate (for faster testing)
            target_ranking_config: Optional TargetRankingConfig object [NEW - preferred]
        
        Returns:
            List of top N target names
        """
        logger.info(f"🎯 Ranking targets (top {top_n})...")
        
        # Generate cache key
        config_hash = hashlib.md5(
            json.dumps({
                'model_families': model_families or [],
                'symbols': sorted(self.symbols)
            }, sort_keys=True).encode()
        ).hexdigest()
        cache_key = self._get_cache_key(self.symbols, config_hash)
        
        # Check cache
        if not force_refresh and use_cache:
            cached = self._load_cached_rankings(cache_key, use_cache=True)
            if cached:
                logger.info(f"✅ Using cached target rankings ({len(cached)} targets)")
                # Return top N from cache
                top_targets = [r['target_name'] for r in cached[:top_n]]
                return top_targets
        
        # Discover or load targets
        try:
            # Try to discover targets from data
            sample_symbol = self.symbols[0]
            targets_dict = discover_targets(sample_symbol, self.data_dir)
            logger.info(f"Discovered {len(targets_dict)} targets from data")
        except Exception as e:
            logger.warning(f"Target discovery failed: {e}, loading from config")
            # Fallback to config
            targets_config = load_target_configs()
            targets_dict = {
                name: config for name, config in targets_config.items()
                if config.get('enabled', False)
            }
            logger.info(f"Loaded {len(targets_dict)} enabled targets from config")
        
        if not targets_dict:
            logger.error("No targets found")
            return []
        
        # NEW: Build target ranking config from experiment config if available
        if target_ranking_config is None and self.experiment_config and _NEW_CONFIG_AVAILABLE:
            target_ranking_config = build_target_ranking_config(self.experiment_config)
        
        # Default model families if not provided
        if model_families is None:
            if target_ranking_config and _NEW_CONFIG_AVAILABLE:
                # Extract from typed config
                model_families = [
                    name for name, config in target_ranking_config.model_families.items()
                    if config.get('enabled', False)
                ]
            elif multi_model_config:
                model_families_dict = multi_model_config.get('model_families', {})
                model_families = [
                    name for name, config in model_families_dict.items()
                    if config and config.get('enabled', False)
                ]
            else:
                model_families = ['lightgbm', 'random_forest', 'neural_network']
        
        # Get explicit interval from experiment config if available
        explicit_interval = None
        experiment_config = None
        try:
            if hasattr(self, 'experiment_config') and self.experiment_config:
                experiment_config = self.experiment_config
                explicit_interval = getattr(self.experiment_config.data, 'bar_interval', None)
        except Exception:
            pass
        
        # Rank targets
        logger.info(f"Evaluating {len(targets_dict)} targets with {len(model_families)} model families...")
        rankings = rank_targets(
            targets=targets_dict,
            symbols=self.symbols,
            data_dir=self.data_dir,
            model_families=model_families,
            multi_model_config=multi_model_config,
            output_dir=self.output_dir / "target_rankings",
            top_n=None,  # Get all rankings for caching
            max_targets_to_evaluate=max_targets_to_evaluate,  # Limit evaluation if specified
            target_ranking_config=target_ranking_config,  # Pass typed config if available
            explicit_interval=explicit_interval,  # Pass explicit interval
            experiment_config=experiment_config  # Pass experiment config
        )
        
        # Save to cache
        if use_cache:
            cache_data = [r.to_dict() for r in rankings]
            self._save_cached_rankings(cache_key, cache_data)
        
        # Return top N
        top_targets = [r.target_name for r in rankings[:top_n]]
        logger.info(f"✅ Top {len(top_targets)} targets: {', '.join(top_targets)}")
        
        return top_targets
    
    def select_features_auto(
        self,
        target: str,
        top_m: int = 100,
        model_families_config: Optional[Dict[str, Any]] = None,
        multi_model_config: Optional[Dict[str, Any]] = None,
        force_refresh: bool = False,
        use_cache: bool = True,
        feature_selection_config: Optional['FeatureSelectionConfig'] = None  # New typed config (optional)
    ) -> List[str]:
        """
        Automatically select top M features for a target.
        
        Args:
            target: Target column name
            top_m: Number of top features to return
            model_families_config: Optional model families config [LEGACY]
            multi_model_config: Optional multi-model config [LEGACY]
            force_refresh: If True, ignore cache and re-select
            use_cache: If True, use cached features if available
            feature_selection_config: Optional FeatureSelectionConfig object [NEW - preferred]
        
        Returns:
            List of top M feature names
        """
        logger.info(f"🔍 Selecting features for {target} (top {top_m})...")
        
        # Check cache
        if not force_refresh and use_cache:
            cached = self._load_cached_features(target)
            if cached:
                logger.info(f"✅ Using cached features for {target} ({len(cached)} features)")
                return cached[:top_m]
        
        # NEW: Build feature selection config from experiment config if available
        if feature_selection_config is None and self.experiment_config and _NEW_CONFIG_AVAILABLE:
            # Create a temporary experiment config with this target
            temp_exp = ExperimentConfig(
                name=self.experiment_config.name,
                data_dir=self.experiment_config.data_dir,
                symbols=self.experiment_config.symbols,
                target=target,
                interval=self.experiment_config.interval,
                max_samples_per_symbol=self.experiment_config.max_samples_per_symbol,
                feature_selection_overrides={'top_n': top_m}
            )
            feature_selection_config = build_feature_selection_config(temp_exp)
        
        # LEGACY: Load config if not provided
        if multi_model_config is None and feature_selection_config is None:
            multi_model_config = load_multi_model_config()
        
        # Select features
        feature_output_dir = self.output_dir / "feature_selections" / target
        
        # Extract explicit_interval from experiment_config for feature selection
        explicit_interval = None
        if self.experiment_config is not None:
            # Try to get bar_interval from config
            if hasattr(self.experiment_config, 'data') and hasattr(self.experiment_config.data, 'bar_interval'):
                explicit_interval = self.experiment_config.data.bar_interval
            # Also check direct bar_interval property (convenience)
            elif hasattr(self.experiment_config, 'bar_interval'):
                explicit_interval = self.experiment_config.bar_interval
            # Legacy: check interval field
            elif hasattr(self.experiment_config, 'interval'):
                explicit_interval = self.experiment_config.interval
        
        selected_features, _ = select_features_for_target(
            target_column=target,
            symbols=self.symbols,
            data_dir=self.data_dir,
            model_families_config=model_families_config,
            multi_model_config=multi_model_config,
            top_n=top_m,
            output_dir=feature_output_dir,
            feature_selection_config=feature_selection_config,  # Pass typed config if available
            explicit_interval=explicit_interval,  # Pass explicit interval to avoid auto-detection warnings
            experiment_config=self.experiment_config  # Pass experiment config for data.bar_interval
        )
        
        # Load confidence and apply routing
        try:
            from TRAINING.orchestration.target_routing import (
                load_target_confidence,
                classify_target_from_confidence,
                save_target_routing_metadata
            )
            
            # Get routing config from multi_model config
            routing_config = None
            if multi_model_config:
                routing_config = multi_model_config.get('confidence', {}).get('routing', {})
            elif feature_selection_config and hasattr(feature_selection_config, 'config'):
                # Try to extract from typed config
                routing_config = feature_selection_config.config.get('confidence', {}).get('routing', {})
            
            conf = load_target_confidence(feature_output_dir, target)
            if conf:
                routing = classify_target_from_confidence(conf, routing_config=routing_config)
                save_target_routing_metadata(feature_output_dir, target, conf, routing)
                
                # Log routing decision
                logger.info(
                    f"🎯 Target {target}: confidence={conf['confidence']} "
                    f"(score_tier={conf.get('score_tier', 'LOW')}, "
                    f"reason={conf.get('low_confidence_reason', 'N/A')}) "
                    f"→ bucket={routing['bucket']}, "
                    f"allowed_in_production={routing['allowed_in_production']}"
                )
        except Exception as e:
            logger.debug(f"Failed to load/route confidence for {target}: {e}")
        
        # Save to cache
        if selected_features:
            self._save_cached_features(target, selected_features)
        
        logger.info(f"✅ Selected {len(selected_features)} features for {target}")
        
        return selected_features
    
    def train_with_intelligence(
        self,
        auto_targets: bool = False,
        top_n_targets: int = 5,
        auto_features: bool = False,
        top_m_features: int = 100,
        targets: Optional[List[str]] = None,
        features: Optional[List[str]] = None,
        families: Optional[List[str]] = None,
        strategy: str = 'single_task',
        use_cache: bool = True,
        run_leakage_diagnostics: bool = False,
        max_targets_to_evaluate: Optional[int] = None,  # Limit number of targets to evaluate (for faster testing)
        **train_kwargs
    ) -> Dict[str, Any]:
        """
        Train models with intelligent target/feature selection.
        
        Args:
            auto_targets: If True, automatically rank and select targets
            top_n_targets: Number of top targets to select (if auto_targets=True)
            auto_features: If True, automatically select features per target
            top_m_features: Number of top features per target (if auto_features=True)
            targets: Manual target list (overrides auto_targets if provided)
            max_targets_to_evaluate: Optional limit on number of targets to evaluate (for faster testing)
            features: Manual feature list (overrides auto_features if provided)
            families: Model families to train
            strategy: Training strategy ('single_task', 'multi_task', 'cascade')
            run_leakage_diagnostics: If True, run leakage sentinel tests after training
            **train_kwargs: Additional arguments passed to train_with_strategies
        
        Returns:
            Training results dictionary
        """
        logger.info("🚀 Starting intelligent training pipeline")
        
        # Step 1: Target selection
        if auto_targets and targets is None:
            logger.info("="*80)
            logger.info("STEP 1: Automatic Target Ranking")
            logger.info("="*80)
            targets = self.rank_targets_auto(
                top_n=top_n_targets,
                use_cache=use_cache,
                max_targets_to_evaluate=max_targets_to_evaluate
            )
            if not targets:
                raise ValueError(
                    "No targets selected after ranking. This usually means:\n"
                    "  1. All targets have insufficient features (short-horizon targets need features with allowed_horizons)\n"
                    "  2. All targets were degenerate (single class, zero variance, extreme imbalance)\n"
                    "  3. All targets failed evaluation\n\n"
                    "Consider:\n"
                    "  - Using targets with longer horizons (more features available)\n"
                    "  - Adding more features to CONFIG/feature_registry.yaml with shorter allowed_horizons\n"
                    "  - Checking CONFIG/excluded_features.yaml (may be too restrictive)\n"
                    "  - Using --no-auto-targets and providing manual --targets list"
                )
        elif targets is None:
            # Fallback: discover all targets
            logger.info("Discovering all targets from data...")
            sample_symbol = self.symbols[0]
            targets_dict = discover_targets(sample_symbol, self.data_dir)
            targets = list(targets_dict.keys())
            logger.info(f"Using all {len(targets)} discovered targets")
        
        logger.info(f"📋 Selected {len(targets)} targets: {', '.join(targets[:5])}{'...' if len(targets) > 5 else ''}")
        
        # Step 2: Feature selection (per target if auto_features)
        target_features = {}
        if auto_features and features is None:
            logger.info("="*80)
            logger.info("STEP 2: Automatic Feature Selection")
            logger.info("="*80)
            for target in targets:
                target_features[target] = self.select_features_auto(
                    target=target,
                    top_m=top_m_features,
                    use_cache=use_cache
                )
        elif features:
            # Use same features for all targets
            for target in targets:
                target_features[target] = features
        
        # Step 3: Training
        logger.info("="*80)
        logger.info("STEP 3: Model Training")
        logger.info("="*80)
        
        # Load MTF data for all symbols
        logger.info(f"Loading data for {len(self.symbols)} symbols...")
        mtf_data = load_mtf_data(
            data_dir=str(self.data_dir),
            symbols=self.symbols,
            max_rows_per_symbol=train_kwargs.get('max_rows_per_symbol')
        )
        
        if not mtf_data:
            raise ValueError(f"Failed to load data for any symbols: {self.symbols}")
        
        logger.info(f"✅ Loaded data for {len(mtf_data)} symbols")
        
        # Prepare training parameters
        interval = 'cross_sectional'  # Use cross-sectional training
        families_list = families or ALL_FAMILIES
        output_dir_str = str(self.output_dir / "training_results")
        
        # Get training parameters from kwargs or config
        min_cs = train_kwargs.get('min_cs', 10)
        max_cs_samples = train_kwargs.get('max_cs_samples')
        max_rows_train = train_kwargs.get('max_rows_train')
        
        logger.info(f"Training {len(targets)} targets with strategy '{strategy}'")
        logger.info(f"Model families: {len(families_list)} families")
        if target_features:
            logger.info(f"Using selected features per target (top {top_m_features} per target)")
            # Log feature counts per target
            for target, feat_list in list(target_features.items())[:3]:
                logger.info(f"  {target}: {len(feat_list)} features")
            if len(target_features) > 3:
                logger.info(f"  ... and {len(target_features) - 3} more targets")
        
        # Pass selected features to training pipeline
        # If target_features is empty, training will auto-discover features
        features_to_use = target_features if target_features else None
        
        # Call the training function
        logger.info("Starting model training...")
        training_results = train_models_for_interval_comprehensive(
            interval=interval,
            targets=targets,
            mtf_data=mtf_data,
            families=families_list,
            strategy=strategy,
            output_dir=output_dir_str,
            min_cs=min_cs,
            max_cs_samples=max_cs_samples,
            max_rows_train=max_rows_train,
            target_features=features_to_use
        )
        
        logger.info("="*80)
        
        # Count trained models
        total_models = sum(
            len(target_results) 
            for target_results in training_results.get('models', {}).values()
        )
        
        # CRITICAL: Fail loudly if 0 models were trained
        # training_results is the dict returned from train_models_for_interval_comprehensive
        # It should have 'models', 'failed_targets', 'failed_reasons' keys
        failed_targets = training_results.get('failed_targets', [])
        failed_reasons = training_results.get('failed_reasons', {})
        
        # If not found at top level, check if it's nested in a 'results' key
        if not failed_targets and 'results' in training_results:
            failed_targets = training_results['results'].get('failed_targets', [])
            failed_reasons = training_results['results'].get('failed_reasons', {})
        
        if total_models == 0:
            logger.error("="*80)
            logger.error("❌ TRAINING RUN FAILED: 0 models trained across %d targets", len(targets))
            logger.error("="*80)
            logger.error("Failed targets: %d / %d", len(failed_targets), len(targets))
            if failed_targets:
                logger.error("Failed target list: %s", failed_targets[:10])
                # Log most common failure reason
                if failed_reasons:
                    reason_counts = {}
                    for reason in failed_reasons.values():
                        reason_counts[reason] = reason_counts.get(reason, 0) + 1
                    most_common = max(reason_counts.items(), key=lambda x: x[1])
                    logger.error("Most common failure reason: %s (%d targets)", most_common[0], most_common[1])
            logger.error("="*80)
            logger.error("This indicates a critical data preparation issue.")
            logger.error("Check logs above for 'all-NaN feature columns' or 'No valid data after cleaning' messages.")
            logger.error("="*80)
            status = 'failed_no_models'
        else:
            logger.info("✅ Training completed successfully")
            logger.info("="*80)
            status = 'completed'
        
        logger.info(f"Trained {total_models} models across {len(targets)} targets")
        if failed_targets:
            logger.warning(f"⚠️ {len(failed_targets)} targets failed data preparation and were skipped")
        
        # Final status summary
        if status == 'failed_no_models':
            logger.error("="*80)
            logger.error("❌ TRAINING PIPELINE FAILED - NO MODELS TRAINED")
            logger.error("="*80)
            logger.error("Action required: Check diagnostic logs above to identify why all features became NaN")
            logger.error("="*80)
        
        # Create run-level confidence summary
        try:
            from TRAINING.orchestration.target_routing import collect_run_level_confidence_summary
            from TRAINING.ranking import load_multi_model_config
            
            # Get routing config
            multi_model_config = load_multi_model_config()
            routing_config = None
            if multi_model_config:
                routing_config = multi_model_config.get('confidence', {}).get('routing', {})
            
            feature_selections_dir = self.output_dir / "feature_selections"
            if feature_selections_dir.exists():
                all_confidence = collect_run_level_confidence_summary(
                    feature_selections_dir=feature_selections_dir,
                    output_dir=self.output_dir,
                    routing_config=routing_config
                )
                
                if all_confidence:
                    # Log summary stats
                    high_conf = sum(1 for c in all_confidence if c.get('confidence') == 'HIGH')
                    medium_conf = sum(1 for c in all_confidence if c.get('confidence') == 'MEDIUM')
                    low_conf = sum(1 for c in all_confidence if c.get('confidence') == 'LOW')
                    logger.info(f"📊 Confidence summary: {high_conf} HIGH, {medium_conf} MEDIUM, {low_conf} LOW")
        except Exception as e:
            logger.debug(f"Failed to create run-level confidence summary: {e}")
        
        # Run leakage diagnostics if enabled
        sentinel_results = {}
        if run_leakage_diagnostics:
            logger.info("="*80)
            logger.info("STEP 4: Leakage Diagnostics (Sentinels)")
            logger.info("="*80)
            try:
                sentinel_results = self._run_leakage_diagnostics(
                    training_results, targets, mtf_data, train_kwargs
                )
            except Exception as e:
                logger.warning(f"Leakage diagnostics failed: {e}")
                sentinel_results = {'error': str(e)}
        
        return {
            'targets': targets,
            'target_features': target_features,
            'strategy': strategy,
            'training_results': training_results,
            'total_models': total_models,
            'sentinel_results': sentinel_results,
            'status': status,  # Use status from above (either 'completed' or 'failed_no_models')
            'failed_targets': failed_targets,
            'failed_reasons': failed_reasons
        }


def main():
    """Main entry point for intelligent training orchestrator."""
    parser = argparse.ArgumentParser(
        description='Intelligent Training Orchestrator with Target Ranking and Feature Selection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-select top 5 targets and top 100 features per target
  python -m TRAINING.orchestration.intelligent_trainer \\
      --data-dir data/data_labeled/interval=5m \\
      --symbols AAPL MSFT GOOGL \\
      --auto-targets --top-n-targets 5 \\
      --auto-features --top-m-features 100

  # Manual targets, auto features
  python -m TRAINING.orchestration.intelligent_trainer \\
      --data-dir data/data_labeled/interval=5m \\
      --symbols AAPL MSFT \\
      --targets fwd_ret_5m fwd_ret_15m \\
      --auto-features --top-m-features 50

  # Use cached rankings (faster)
  python -m TRAINING.orchestration.intelligent_trainer \\
      --data-dir data/data_labeled/interval=5m \\
      --symbols AAPL MSFT \\
      --auto-targets --top-n-targets 5 \\
      --no-refresh-cache
        """
    )
    
    # Core arguments
    parser.add_argument('--data-dir', type=Path, required=False,
                       help='Data directory containing symbol data (required unless --experiment-config provided)')
    parser.add_argument('--symbols', nargs='+', required=False,
                       help='Symbols to train on (required unless --experiment-config provided)')
    parser.add_argument('--output-dir', type=Path, default=Path('intelligent_output'),
                       help='Output directory for results')
    parser.add_argument('--cache-dir', type=Path,
                       help='Cache directory for rankings/selections (default: output_dir/cache)')
    
    # Target selection
    parser.add_argument('--auto-targets', action='store_true', default=True,
                       help='Automatically rank and select targets (default: True)')
    parser.add_argument('--no-auto-targets', dest='auto_targets', action='store_false',
                       help='Disable automatic target ranking')
    parser.add_argument('--top-n-targets', type=int, default=5,
                       help='Number of top targets to select (default: 5)')
    parser.add_argument('--max-targets-to-evaluate', type=int, default=None,
                       help='Limit number of targets to evaluate for faster testing (default: evaluate all)')
    parser.add_argument('--targets', nargs='+',
                       help='Manual target list (overrides --auto-targets)')
    
    # Feature selection
    parser.add_argument('--auto-features', action='store_true', default=True,
                       help='Automatically select features per target (default: True)')
    parser.add_argument('--no-auto-features', dest='auto_features', action='store_false',
                       help='Disable automatic feature selection')
    parser.add_argument('--top-m-features', type=int, default=100,
                       help='Number of top features per target (default: 100)')
    parser.add_argument('--features', nargs='+',
                       help='Manual feature list (overrides --auto-features)')
    
    # Training arguments (passed through to train_with_strategies)
    parser.add_argument('--families', nargs='+',
                       help='Model families to train')
    parser.add_argument('--strategy', choices=['single_task', 'multi_task', 'cascade'],
                       default='single_task',
                       help='Training strategy (default: single_task)')
    parser.add_argument('--min-cs', type=int, default=10,
                       help='Minimum cross-sectional samples required (default: 10)')
    parser.add_argument('--max-rows-per-symbol', type=int,
                       help='Maximum rows to load per symbol (for testing)')
    parser.add_argument('--max-rows-train', type=int,
                       help='Maximum training rows (for testing)')
    parser.add_argument('--max-cs-samples', type=int,
                       help='Maximum cross-sectional samples per timestamp')
    parser.add_argument('--run-leakage-diagnostics', action='store_true',
                       help='Run leakage sentinel tests after training (optional diagnostic mode)')
    
    # Cache control
    parser.add_argument('--force-refresh', action='store_true',
                       help='Force refresh of cached rankings/selections')
    parser.add_argument('--no-refresh-cache', action='store_true',
                       help='Never refresh cache (use existing only)')
    parser.add_argument('--no-cache', action='store_true',
                       help='Disable caching entirely')
    
    # Config files
    parser.add_argument('--target-ranking-config', type=Path,
                       help='Path to target ranking config YAML (default: CONFIG/training_config/target_ranking_config.yaml)')
    parser.add_argument('--multi-model-config', type=Path,
                       help='Path to multi-model feature selection config YAML (default: CONFIG/multi_model_feature_selection.yaml)')
    parser.add_argument('--experiment-config', type=str,
                       help='Experiment config name (without .yaml) from CONFIG/experiments/ [NEW - preferred]')
    
    args = parser.parse_args()
    
    # NEW: Load experiment config if provided
    experiment_config = None
    if args.experiment_config and _NEW_CONFIG_AVAILABLE:
        try:
            from CONFIG.config_builder import load_experiment_config
            experiment_config = load_experiment_config(args.experiment_config)
            logger.info(f"✅ Loaded experiment config: {experiment_config.name}")
            # Use experiment config values if CLI args not provided
            if not args.data_dir:
                args.data_dir = experiment_config.data_dir
            if not args.symbols:
                args.symbols = experiment_config.symbols
        except Exception as e:
            logger.error(f"Failed to load experiment config '{args.experiment_config}': {e}")
            raise
    
    # Validate required args (either from CLI or experiment config)
    if not args.data_dir:
        parser.error("--data-dir is required (or provide --experiment-config)")
    if not args.symbols:
        parser.error("--symbols is required (or provide --experiment-config)")
    
    # Create orchestrator
    trainer = IntelligentTrainer(
        data_dir=args.data_dir,
        symbols=args.symbols,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        experiment_config=experiment_config  # Pass experiment config if loaded
    )
    
    # Load configs (legacy support)
    target_ranking_config = None
    if args.target_ranking_config:
        target_ranking_config = load_target_configs(args.target_ranking_config)
    
    multi_model_config = None
    if args.multi_model_config:
        multi_model_config = load_multi_model_config(args.multi_model_config)
    elif not experiment_config:
        # Only load default if no experiment config
        multi_model_config = load_multi_model_config()
    
    # Determine cache usage
    use_cache = not args.no_cache
    
    # Run training
    try:
        results = trainer.train_with_intelligence(
            auto_targets=args.auto_targets,
            top_n_targets=args.top_n_targets,
            max_targets_to_evaluate=args.max_targets_to_evaluate,
            auto_features=args.auto_features,
            top_m_features=args.top_m_features,
            targets=args.targets,
            features=args.features,
            families=args.families,
            strategy=args.strategy,
            force_refresh=args.force_refresh,
            use_cache=use_cache,
            run_leakage_diagnostics=args.run_leakage_diagnostics,
            min_cs=args.min_cs,
            max_rows_per_symbol=args.max_rows_per_symbol,
            max_rows_train=args.max_rows_train,
            max_cs_samples=args.max_cs_samples
        )
        
        logger.info("="*80)
        logger.info("✅ Intelligent training pipeline completed")
        logger.info("="*80)
        logger.info(f"Targets: {len(results['targets'])}")
        logger.info(f"Strategy: {results['strategy']}")
        logger.info(f"Status: {results['status']}")
        
        return 0
    
    except Exception as e:
        logger.error(f"❌ Training failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

