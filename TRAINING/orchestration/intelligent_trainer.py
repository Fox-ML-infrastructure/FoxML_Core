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
    # Logger not yet initialized, will be set up below
    pass

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
                output_dir_name = f"{output_dir.name}_{timestamp}"
            else:
                output_dir_name = output_dir.name
        else:
            output_dir = Path(output_dir)
            output_dir_name = output_dir.name
        
        # Put ALL runs in RESULTS directory, organized by sample size (N_effective)
        # Structure: RESULTS/{N_effective}/{run_name}/
        # Try to determine N_effective early from data or existing metadata
        repo_root = Path(__file__).parent.parent.parent  # Go up from TRAINING/orchestration/ to repo root
        results_dir = repo_root / "RESULTS"
        
        # Try to estimate N_effective early (before first target is processed)
        self._n_effective = self._estimate_n_effective_early()
        
        if self._n_effective is not None:
            # Create directory directly in RESULTS/{N_effective}/{run_name}/
            self.output_dir = results_dir / str(self._n_effective) / output_dir_name
            self._initial_output_dir = self.output_dir  # Same location, no move needed
            logger.info(f"📁 Output directory: {self.output_dir} (organized by sample size N={self._n_effective})")
        else:
            # Fallback: start in _pending/ - will be moved to N_effective directory after first target is processed
            self._initial_output_dir = results_dir / "_pending" / output_dir_name
            self.output_dir = self._initial_output_dir
            logger.info(f"📁 Output directory: {self.output_dir} (will be organized by sample size after first target)")
        
        self._run_name = output_dir_name  # Store for move operation
        
        # Create directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.cache_dir = Path(cache_dir) if cache_dir else self.output_dir / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize cache file paths
        self.target_ranking_cache = self.cache_dir / "target_rankings.json"
        self.feature_selection_cache = self.cache_dir / "feature_selections"
        self.feature_selection_cache.mkdir(parents=True, exist_ok=True)
    
    def _estimate_n_effective_early(self) -> Optional[int]:
        """
        Try to estimate N_effective early from data files or existing metadata.
        
        Returns:
            Estimated N_effective or None if cannot be determined
        """
        logger.debug("🔍 Attempting early N_effective estimation...")
        
        # Method 1: Check if there's existing metadata from a previous run with same symbols/data
        # (This handles the case where you're re-running with same data)
        try:
            repo_root = Path(__file__).parent.parent.parent
            results_dir = repo_root / "RESULTS"
            
            # Look for existing runs with same symbols (quick check)
            if results_dir.exists():
                logger.debug(f"Checking existing runs in {results_dir} for matching symbols: {self.symbols}")
                for n_dir in results_dir.iterdir():
                    if n_dir.is_dir() and n_dir.name.isdigit():
                        # Check if there's a recent run with similar structure
                        for run_dir in n_dir.iterdir():
                            if run_dir.is_dir():
                                # Check metadata.json in REPRODUCIBILITY
                                for metadata_file in run_dir.rglob("REPRODUCIBILITY/TARGET_RANKING/*/cohort=*/metadata.json"):
                                    try:
                                        import json
                                        with open(metadata_file, 'r') as f:
                                            metadata = json.load(f)
                                        # Check if symbols match (rough check)
                                        existing_symbols = metadata.get('symbols', [])
                                        if existing_symbols and set(existing_symbols) == set(self.symbols):
                                            n_effective = metadata.get('N_effective')
                                            if n_effective and n_effective > 0:
                                                logger.info(f"🔍 Found matching N_effective={n_effective} from previous run with same symbols")
                                                return int(n_effective)
                                    except Exception as e:
                                        logger.debug(f"Failed to read {metadata_file}: {e}")
                                        continue
        except Exception as e:
            logger.debug(f"Could not check existing metadata for N_effective: {e}")
        
        # Method 2: Quick sample from data files to estimate sample size
        try:
            import pandas as pd
            total_rows = 0
            
            # Sample first few symbols to estimate
            sample_symbols = self.symbols[:3] if len(self.symbols) > 3 else self.symbols
            logger.debug(f"Sampling {len(sample_symbols)} symbols from {self.data_dir} to estimate N_effective")
            
            for symbol in sample_symbols:
                # Try multiple possible paths
                possible_paths = [
                    self.data_dir / f"symbol={symbol}" / f"{symbol}.parquet",
                    self.data_dir / symbol / f"{symbol}.parquet",
                    self.data_dir / f"{symbol}.parquet"
                ]
                
                data_path = None
                for path in possible_paths:
                    if path.exists():
                        data_path = path
                        break
                
                if data_path is None:
                    logger.debug(f"Data file not found for {symbol} (tried: {[str(p) for p in possible_paths]})")
                    continue
                
                logger.debug(f"Found data file for {symbol}: {data_path}")
                
                try:
                    # Use parquet metadata if available (faster - no data load)
                    try:
                        import pyarrow.parquet as pq
                        parquet_file = pq.ParquetFile(data_path)
                        symbol_rows = parquet_file.metadata.num_rows
                        total_rows += symbol_rows
                        logger.debug(f"  {symbol}: {symbol_rows} rows (from parquet metadata)")
                    except ImportError:
                        # pyarrow not available, try pandas
                        logger.debug(f"  pyarrow not available, using pandas for {symbol}")
                        symbol_rows = len(pd.read_parquet(data_path))
                        total_rows += symbol_rows
                        logger.debug(f"  {symbol}: {symbol_rows} rows (from pandas)")
                    except Exception as e:
                        logger.debug(f"  Could not read parquet metadata for {symbol}: {e}, trying pandas")
                        # Fallback: actually count rows (slower but works)
                        symbol_rows = len(pd.read_parquet(data_path))
                        total_rows += symbol_rows
                        logger.debug(f"  {symbol}: {symbol_rows} rows (from pandas fallback)")
                except Exception as e:
                    logger.debug(f"Could not read {data_path} for sample size estimation: {e}")
                    continue
            
            # Extrapolate to all symbols
            if total_rows > 0 and len(sample_symbols) > 0:
                avg_per_symbol = total_rows / len(sample_symbols)
                estimated_total = int(avg_per_symbol * len(self.symbols))
                logger.info(f"🔍 Estimated N_effective={estimated_total} from data file sampling ({len(sample_symbols)} symbols sampled, {total_rows} total rows)")
                return estimated_total
            else:
                logger.debug(f"Could not estimate N_effective: total_rows={total_rows}, sample_symbols={len(sample_symbols)}")
        except Exception as e:
            logger.warning(f"Could not estimate N_effective from data files: {e}")
            import traceback
            logger.debug(f"Traceback: {traceback.format_exc()}")
        
        logger.debug("Could not determine N_effective early, will use _pending/ and organize after first target")
        return None
    
    def _organize_by_cohort(self):
        """
        Organize the run directory by sample size (N_effective) after first target is processed.
        Moves from RESULTS/_pending/{run_name}/ to RESULTS/{N_effective}/{run_name}/
        
        Example: RESULTS/25000/test_run_20251212_010000/
        
        Note: If N_effective was already determined in __init__, this is a no-op.
        """
        # If N_effective was already set and we're not in _pending/, we're already organized
        if self._n_effective is not None and "_pending" not in str(self.output_dir):
            return  # Already organized
        
        # If N_effective was set early but we're still in _pending/, move now
        if self._n_effective is not None and "_pending" in str(self.output_dir):
            repo_root = Path(__file__).parent.parent.parent
            results_dir = repo_root / "RESULTS"
            new_output_dir = results_dir / str(self._n_effective) / self._run_name
            
            if new_output_dir.exists():
                logger.warning(f"Sample size directory {new_output_dir} already exists, not moving")
                self.output_dir = new_output_dir
                self.cache_dir = self.output_dir / "cache"
                self.target_ranking_cache = self.cache_dir / "target_rankings.json"
                self.feature_selection_cache = self.cache_dir / "feature_selections"
                return
            
            import shutil
            new_output_dir.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"📁 Moving run from {self.output_dir} to {new_output_dir} (N={self._n_effective} determined early)")
            try:
                shutil.move(str(self.output_dir), str(new_output_dir))
                self.output_dir = new_output_dir
                self.cache_dir = self.output_dir / "cache"
                self.target_ranking_cache = self.cache_dir / "target_rankings.json"
                self.feature_selection_cache = self.cache_dir / "feature_selections"
                logger.info(f"✅ Organized run by sample size (N={self._n_effective}): {self.output_dir}")
                return
            except Exception as move_error:
                logger.error(f"Failed to move directory: {move_error}")
                # Stay in current location if move fails
                return
        
        try:
            # Try to find N_effective from REPRODUCIBILITY directory metadata.json
            # REPRODUCIBILITY structure: self.output_dir/target_rankings/REPRODUCIBILITY/TARGET_RANKING/...
            # OR: self.output_dir/REPRODUCIBILITY/TARGET_RANKING/... (if output_dir is already target_rankings)
            
            # Check both possible locations
            possible_repro_dirs = [
                self.output_dir / "target_rankings" / "REPRODUCIBILITY",
                self.output_dir / "REPRODUCIBILITY"
            ]
            
            target_ranking_dir = None
            for repro_dir in possible_repro_dirs:
                candidate = repro_dir / "TARGET_RANKING"
                if candidate.exists():
                    target_ranking_dir = candidate
                    break
            
            if target_ranking_dir is None:
                logger.info(f"TARGET_RANKING directory not found at expected paths (checked: {[str(d / 'TARGET_RANKING') for d in possible_repro_dirs]})")
                # Try recursive search as fallback
                logger.info(f"Trying recursive search in {self._initial_output_dir}")
                for metadata_file in self._initial_output_dir.rglob("REPRODUCIBILITY/TARGET_RANKING/*/cohort=*/metadata.json"):
                    if metadata_file.exists():
                        try:
                            import json
                            with open(metadata_file, 'r') as f:
                                metadata = json.load(f)
                            n_effective = metadata.get('N_effective')
                            if n_effective is not None and n_effective > 0:
                                self._n_effective = int(n_effective)
                                logger.info(f"🔍 Found N_effective via recursive search: {self._n_effective} at {metadata_file.parent}")
                                break
                        except Exception as e:
                            logger.debug(f"Failed to read metadata from {metadata_file}: {e}")
                            continue
                if self._n_effective is None:
                    logger.warning(f"Could not find N_effective via recursive search in {self._initial_output_dir}")
                    return
            else:
                # Find first target's metadata.json to extract N_effective
                for target_dir in target_ranking_dir.iterdir():
                    if not target_dir.is_dir():
                        continue
                    
                    # Look for cohort= directories
                    for cohort_dir in target_dir.iterdir():
                        if cohort_dir.is_dir() and cohort_dir.name.startswith("cohort="):
                            metadata_file = cohort_dir / "metadata.json"
                            if metadata_file.exists():
                                try:
                                    import json
                                    with open(metadata_file, 'r') as f:
                                        metadata = json.load(f)
                                    n_effective = metadata.get('N_effective')
                                    if n_effective is not None and n_effective > 0:
                                        self._n_effective = int(n_effective)
                                        logger.info(f"🔍 Found N_effective: {self._n_effective} from {metadata_file}")
                                        break
                                except Exception as e:
                                    logger.debug(f"Failed to read metadata from {metadata_file}: {e}")
                                    continue
                    if self._n_effective is not None:
                        break
            
            # If we found N_effective, move the directory
            if self._n_effective is not None:
                # Move the entire run directory to N_effective-organized location
                repo_root = Path(__file__).parent.parent.parent
                results_dir = repo_root / "RESULTS"
                new_output_dir = results_dir / str(self._n_effective) / self._run_name
                
                if new_output_dir.exists():
                    logger.warning(f"Sample size directory {new_output_dir} already exists, not moving")
                    # Still update paths to point to existing directory
                    self.output_dir = new_output_dir
                    self.cache_dir = self.output_dir / "cache"
                    self.target_ranking_cache = self.cache_dir / "target_rankings.json"
                    self.feature_selection_cache = self.cache_dir / "feature_selections"
                    logger.info(f"📁 Using existing sample size directory: {self.output_dir}")
                    return
                
                # Move the directory
                import shutil
                new_output_dir.parent.mkdir(parents=True, exist_ok=True)
                
                logger.info(f"📁 Moving run from {self._initial_output_dir} to {new_output_dir}")
                try:
                    shutil.move(str(self._initial_output_dir), str(new_output_dir))
                    self.output_dir = new_output_dir
                    
                    # Update cache_dir path and cache file paths
                    self.cache_dir = self.output_dir / "cache"
                    self.target_ranking_cache = self.cache_dir / "target_rankings.json"
                    self.feature_selection_cache = self.cache_dir / "feature_selections"
                    
                    logger.info(f"✅ Organized run by sample size (N={self._n_effective}): {self.output_dir}")
                    return
                except Exception as move_error:
                    logger.error(f"Failed to move directory: {move_error}")
                    logger.debug(f"Move error traceback: {traceback.format_exc()}")
                    # Stay in _pending/ if move fails
                    return
            
            logger.debug(f"No metadata.json found to extract N_effective, waiting for first target")
        except Exception as e:
            logger.warning(f"Could not organize by sample size (will stay in _pending/): {e}")
            import traceback
            logger.debug(f"Traceback: {traceback.format_exc()}")
            # Stay in _pending/ if we can't determine cohort
        
        # If we still haven't organized, try one more time with more aggressive search
        if self._n_effective is None:
            try:
                # Search more broadly - check if REPRODUCIBILITY exists anywhere in the run directory
                for metadata_file in self._initial_output_dir.rglob("REPRODUCIBILITY/TARGET_RANKING/*/cohort=*/metadata.json"):
                    if metadata_file.exists():
                        try:
                            import json
                            with open(metadata_file, 'r') as f:
                                metadata = json.load(f)
                            n_effective = metadata.get('N_effective')
                            if n_effective is not None and n_effective > 0:
                                self._n_effective = int(n_effective)
                                
                                repo_root = Path(__file__).parent.parent.parent
                                results_dir = repo_root / "RESULTS"
                                new_output_dir = results_dir / str(self._n_effective) / self._run_name
                                
                                if new_output_dir.exists():
                                    logger.warning(f"Sample size directory {new_output_dir} already exists, not moving")
                                    self.output_dir = new_output_dir
                                    self.cache_dir = self.output_dir / "cache"
                                    self.target_ranking_cache = self.cache_dir / "target_rankings.json"
                                    self.feature_selection_cache = self.cache_dir / "feature_selections"
                                    return
                                
                                import shutil
                                new_output_dir.parent.mkdir(parents=True, exist_ok=True)
                                logger.info(f"📁 Moving run from {self._initial_output_dir} to {new_output_dir} (found via recursive search, N={self._n_effective})")
                                shutil.move(str(self._initial_output_dir), str(new_output_dir))
                                self.output_dir = new_output_dir
                                self.cache_dir = self.output_dir / "cache"
                                self.target_ranking_cache = self.cache_dir / "target_rankings.json"
                                self.feature_selection_cache = self.cache_dir / "feature_selections"
                                logger.info(f"✅ Organized run by sample size (N={self._n_effective}): {self.output_dir}")
                                return
                        except Exception as e:
                            logger.debug(f"Failed to read metadata from {metadata_file}: {e}")
                            continue
            except Exception as e2:
                logger.debug(f"Recursive search also failed: {e2}")
        
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
        target_ranking_config: Optional['TargetRankingConfig'] = None,  # New typed config (optional)
        min_cs: Optional[int] = None,  # Load from config if None
        max_cs_samples: Optional[int] = None,  # Load from config if None
        max_rows_per_symbol: Optional[int] = None  # Load from config if None
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
            min_cs: Minimum cross-sectional size per timestamp (loads from config if None)
            max_cs_samples: Maximum samples per timestamp for cross-sectional sampling (loads from config if None)
            max_rows_per_symbol: Maximum rows to load per symbol (loads from config if None)
        
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
            experiment_config=experiment_config,  # Pass experiment config
            min_cs=min_cs,  # Pass min_cs from config
            max_cs_samples=max_cs_samples,  # Pass max_cs_samples from config
            max_rows_per_symbol=max_rows_per_symbol  # Pass max_rows_per_symbol from config
        )
        
        # After target ranking completes, organize by sample size (N_effective)
        # This moves the entire directory (including all REPRODUCIBILITY data) to RESULTS/{N_effective}/{run_name}/
        if self._n_effective is None and rankings:
            logger.info("🔍 Attempting to organize run by sample size (N_effective)...")
            logger.info(f"   Current output_dir: {self.output_dir}")
            logger.info(f"   Initial output_dir: {self._initial_output_dir}")
            self._organize_by_cohort()
            if self._n_effective is not None:
                logger.info(f"✅ Successfully organized run by sample size (N={self._n_effective}): {self.output_dir}")
                logger.info(f"   Moved from: {self._initial_output_dir}")
                logger.info(f"   Moved to: {self.output_dir}")
            else:
                logger.warning("⚠️  Could not determine N_effective, run will stay in _pending/")
                logger.warning(f"   Run directory: {self._initial_output_dir}")
                # Try to help debug - check if REPRODUCIBILITY exists
                repro_check = self._initial_output_dir / "target_rankings" / "REPRODUCIBILITY"
                if repro_check.exists():
                    logger.warning(f"   REPRODUCIBILITY found at: {repro_check}")
                else:
                    logger.warning(f"   REPRODUCIBILITY not found at: {repro_check}")
        
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
        
        # Extract data limits from train_kwargs (passed from main)
        min_cs = train_kwargs.get('min_cs', 10)
        max_cs_samples = train_kwargs.get('max_cs_samples')
        max_rows_per_symbol = train_kwargs.get('max_rows_per_symbol')
        
        # Step 1: Target selection
        if auto_targets and targets is None:
            logger.info("="*80)
            logger.info("STEP 1: Automatic Target Ranking")
            logger.info("="*80)
            targets = self.rank_targets_auto(
                top_n=top_n_targets,
                use_cache=use_cache,
                max_targets_to_evaluate=max_targets_to_evaluate,
                min_cs=min_cs,
                max_cs_samples=max_cs_samples,
                max_rows_per_symbol=max_rows_per_symbol
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
        
        # Step 2.5: Generate training routing plan (if feature selection completed)
        training_plan_dir = None
        if target_features:
            try:
                from TRAINING.orchestration.routing_integration import generate_routing_plan_after_feature_selection
                routing_plan = generate_routing_plan_after_feature_selection(
                    output_dir=self.output_dir,
                    targets=list(target_features.keys()),
                    symbols=self.symbols,
                    generate_training_plan=True,  # Generate training plan
                    model_families=families  # Use specified families
                )
                if routing_plan:
                    logger.info("✅ Training routing plan generated - see METRICS/routing_plan/ for details")
                    # Set training plan directory for filtering
                    training_plan_dir = self.output_dir / "METRICS" / "training_plan"
            except Exception as e:
                logger.debug(f"Failed to generate routing plan (non-critical): {e}")
        
        # Step 3: Training
        logger.info("="*80)
        logger.info("STEP 3: Model Training")
        logger.info("="*80)
        
        # Apply training plan filter if available
        filtered_targets = targets
        filtered_symbols_by_target = {t: self.symbols for t in targets}  # Default: all symbols per target
        training_plan = None
        
        if training_plan_dir:
            try:
                # Validate training_plan_dir
                training_plan_dir = Path(training_plan_dir)
                if not training_plan_dir.exists():
                    logger.debug(f"Training plan directory does not exist: {training_plan_dir}")
                elif training_plan_dir.exists():
                    from TRAINING.orchestration.training_plan_consumer import (
                        apply_training_plan_filter,
                        load_training_plan,
                        get_model_families_for_job
                    )
                    
                    # Load training plan with error handling
                    try:
                        training_plan = load_training_plan(training_plan_dir)
                    except Exception as e:
                        logger.warning(f"Failed to load training plan: {e}, proceeding without filtering")
                        training_plan = None
                    
                    if training_plan:
                        try:
                            filtered_targets, filtered_symbols_by_target = apply_training_plan_filter(
                                targets=targets,
                                symbols=self.symbols,
                                training_plan_dir=training_plan_dir,
                                use_cs_plan=True,
                                use_symbol_plan=True
                            )
                            
                            # Validate filtered results
                            if not isinstance(filtered_targets, list):
                                logger.warning(f"apply_training_plan_filter returned invalid filtered_targets: {type(filtered_targets)}, using original")
                                filtered_targets = targets
                            
                            if not isinstance(filtered_symbols_by_target, dict):
                                logger.warning(f"apply_training_plan_filter returned invalid filtered_symbols_by_target: {type(filtered_symbols_by_target)}, using default")
                                filtered_symbols_by_target = {t: self.symbols for t in filtered_targets}
                            
                            if len(filtered_targets) < len(targets):
                                logger.info(f"📋 Training plan filter applied: {len(targets)} → {len(filtered_targets)} targets")
                            
                            # Log symbol filtering per target - safely
                            for target in filtered_targets[:10]:  # Limit logging to first 10
                                try:
                                    filtered_symbols = filtered_symbols_by_target.get(target, self.symbols)
                                    if isinstance(filtered_symbols, list) and len(filtered_symbols) < len(self.symbols):
                                        logger.info(f"📋 Filtered symbols for {target}: {len(self.symbols)} → {len(filtered_symbols)} symbols")
                                except Exception as e:
                                    logger.debug(f"Error logging symbol filter for {target}: {e}")
                            
                            # Update target_features to only include filtered targets - safely
                            if isinstance(target_features, dict):
                                try:
                                    target_features = {t: f for t, f in target_features.items() if t in filtered_targets}
                                except Exception as e:
                                    logger.warning(f"Failed to filter target_features: {e}, keeping original")
                        except Exception as e:
                            logger.warning(f"Failed to apply training plan filter: {e}, using all targets")
                            filtered_targets = targets
                            filtered_symbols_by_target = {t: self.symbols for t in targets}
            except Exception as e:
                logger.warning(f"Failed to apply training plan filter (non-critical): {e}", exc_info=True)
        
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
        
        # Extract model families from training plan if available, otherwise use provided/default
        families_list = families or ALL_FAMILIES
        target_families_map = {}  # Per-target families from training plan
        
        if training_plan and filtered_targets:
            # Get model families per target from training plan - with error handling
            for target in filtered_targets:
                if not isinstance(target, str) or not target:
                    logger.warning(f"Skipping invalid target in family extraction: {target}")
                    continue
                
                try:
                    plan_families = get_model_families_for_job(
                        training_plan,
                        target=target,
                        symbol=None,
                        training_type="cross_sectional"
                    )
                    
                    if plan_families:
                        # Validate plan_families is a list
                        if not isinstance(plan_families, list):
                            logger.warning(f"plan_families for {target} is not a list: {type(plan_families)}")
                            continue
                        
                        # Filter to only include families that are in the provided/default list
                        try:
                            filtered_plan_families = [f for f in plan_families if f in families_list]
                            if filtered_plan_families:
                                target_families_map[target] = filtered_plan_families
                                logger.debug(f"📋 Target {target}: using {len(filtered_plan_families)} families from plan")
                        except Exception as e:
                            logger.warning(f"Failed to filter plan families for {target}: {e}")
                except Exception as e:
                    logger.warning(f"Failed to get model families for {target}: {e}")
                    continue
            
            if target_families_map:
                # If all targets have the same families, use that as the global list
                all_target_families = set()
                for target_fams in target_families_map.values():
                    all_target_families.update(target_fams)
                
                # Use intersection of all target families (most restrictive)
                # Only compute intersection if we have at least one target with families
                if filtered_targets and filtered_targets[0] in target_families_map:
                    common_families = set(target_families_map[filtered_targets[0]])
                    for target in filtered_targets[1:]:
                        if target in target_families_map:
                            common_families &= set(target_families_map[target])
                    
                    if common_families:
                        families_list = sorted(common_families)
                        logger.info(f"📋 Using common model families from training plan: {families_list}")
                    else:
                        # Targets have different families - will need per-target filtering
                        logger.info(f"📋 Targets have different model families in plan - using union: {sorted(all_target_families)}")
                        families_list = sorted(all_target_families)
                else:
                    # No targets in map, use union of all families found
                    if all_target_families:
                        families_list = sorted(all_target_families)
                        logger.info(f"📋 Using model families from training plan: {families_list}")
            else:
                logger.debug("No model families found in training plan, using provided/default families")
        
        # Validate families_list is not empty
        if not families_list:
            logger.warning("⚠️ No model families available after filtering! Using default families.")
            families_list = ALL_FAMILIES if not families else families
        
        # Validate filtered_targets is not empty
        if not filtered_targets:
            logger.warning("⚠️ All targets were filtered out by training plan! Training will be skipped.")
            logger.warning("   This may indicate an issue with the training plan or routing decisions.")
        
        output_dir_str = str(self.output_dir / "training_results")
        
        # Get training parameters from kwargs or config (min_cs and max_cs_samples already extracted above)
        max_rows_train = train_kwargs.get('max_rows_train')
        
        # Early return if no targets to train
        if not filtered_targets:
            logger.error("❌ No targets to train after filtering. Exiting.")
            return {
                "status": "skipped",
                "reason": "All targets filtered out by training plan",
                "targets_requested": len(targets),
                "targets_filtered": 0
            }
        
        logger.info(f"Training {len(filtered_targets)} targets with strategy '{strategy}'")
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
            targets=filtered_targets,  # Use filtered targets
            mtf_data=mtf_data,
            families=families_list,
            strategy=strategy,
            output_dir=output_dir_str,
            min_cs=min_cs,
            max_cs_samples=max_cs_samples,
            max_rows_train=max_rows_train,
            target_features=features_to_use,
            target_families=target_families_map if target_families_map else None  # Per-target families from plan
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
        
        # Generate trend summary (if reproducibility tracking is available)
        trend_summary = None
        try:
            from TRAINING.utils.reproducibility_tracker import ReproducibilityTracker
            from pathlib import Path
            
            # Find REPRODUCIBILITY directory
            repro_dir = self.output_dir / "REPRODUCIBILITY"
            if not repro_dir.exists():
                # Try alternative location (if organized by cohort)
                repro_dir = self.output_dir.parent / "REPRODUCIBILITY"
            
            if repro_dir.exists():
                # Create tracker to access trend summary method
                tracker = ReproducibilityTracker(output_dir=self.output_dir)
                trend_summary = tracker.generate_trend_summary(view="STRICT", min_runs_for_trend=3)
                
                if trend_summary.get("status") == "ok":
                    logger.info("="*80)
                    logger.info("TREND ANALYSIS SUMMARY")
                    logger.info("="*80)
                    logger.info(f"Series analyzed: {trend_summary.get('n_series', 0)}")
                    logger.info(f"Trends computed: {trend_summary.get('n_trends', 0)}")
                    
                    if trend_summary.get("declining_trends"):
                        logger.warning(f"⚠️  {len(trend_summary['declining_trends'])} declining trends detected")
                        for decl in trend_summary["declining_trends"][:5]:
                            logger.warning(f"  - {decl['metric']}: slope={decl['slope']:.6f}/day ({decl['series'][:50]}...)")
                    
                    if trend_summary.get("alerts"):
                        logger.info(f"ℹ️  {len(trend_summary['alerts'])} trend alerts")
                        for alert in trend_summary["alerts"][:3]:  # Show first 3
                            severity_icon = "⚠️" if alert.get('severity') == 'warning' else "ℹ️"
                            logger.info(f"  {severity_icon} {alert.get('message', '')[:100]}")
                    
                    logger.info("="*80)
        except Exception as e:
            logger.debug(f"Could not generate trend summary: {e}")
            # Don't fail if trend analysis fails
        
        return {
            'targets': targets,
            'target_features': target_features,
            'strategy': strategy,
            'training_results': training_results,
            'total_models': total_models,
            'sentinel_results': sentinel_results,
            'status': status,  # Use status from above (either 'completed' or 'failed_no_models')
            'failed_targets': failed_targets,
            'failed_reasons': failed_reasons,
            'trend_summary': trend_summary
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
    
    # Target/feature selection (moved to config - CLI only for manual overrides)
    parser.add_argument('--targets', nargs='+',
                       help='Manual target list (overrides config auto_targets)')
    parser.add_argument('--features', nargs='+',
                       help='Manual feature list (overrides config auto_features)')
    
    # Training arguments (moved to config - CLI only for manual overrides)
    parser.add_argument('--families', nargs='+',
                       help='Model families to train (overrides config)')
    
    # Testing/debugging overrides (use sparingly - prefer config)
    parser.add_argument('--override-max-samples', type=int,
                       help='OVERRIDE: Max samples per symbol (testing only, overrides config)')
    parser.add_argument('--override-max-rows', type=int,
                       help='OVERRIDE: Max rows per symbol (testing only, overrides config)')
    
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
    
    # NEW: Load experiment config if provided (PREFERRED)
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
    
    # Load intelligent training settings from config (SST)
    try:
        from CONFIG.config_loader import get_cfg
        _CONFIG_AVAILABLE = True
    except ImportError:
        _CONFIG_AVAILABLE = False
        logger.warning("Config loader not available, using hardcoded defaults")
    
    if _CONFIG_AVAILABLE:
        # Check for test mode override (for E2E testing)
        use_test_config = args.output_dir and 'test' in str(args.output_dir).lower()
        
        if use_test_config:
            # Load test config if available (for E2E testing)
            test_cfg = get_cfg("test.intelligent_training", default={}, config_name="pipeline_config")
            if test_cfg:
                logger.info("📋 Using test configuration (detected 'test' in output-dir)")
                intel_cfg = test_cfg
            else:
                intel_cfg = get_cfg("intelligent_training", default={}, config_name="pipeline_config")
        else:
            # Load from config (SST)
            intel_cfg = get_cfg("intelligent_training", default={}, config_name="pipeline_config")
        
        auto_targets = intel_cfg.get('auto_targets', True)
        top_n_targets = intel_cfg.get('top_n_targets', 5)
        max_targets_to_evaluate = intel_cfg.get('max_targets_to_evaluate', None)
        auto_features = intel_cfg.get('auto_features', True)
        top_m_features = intel_cfg.get('top_m_features', 100)
        strategy = intel_cfg.get('strategy', 'single_task')
        min_cs = intel_cfg.get('min_cs', 10)
        max_rows_per_symbol = intel_cfg.get('max_rows_per_symbol', None)
        max_rows_train = intel_cfg.get('max_rows_train', None)
        max_cs_samples = intel_cfg.get('max_cs_samples', None)
        run_leakage_diagnostics = intel_cfg.get('run_leakage_diagnostics', False)
        
        # If max_cs_samples not in intelligent_training, try pipeline.data_limits
        if max_cs_samples is None:
            max_cs_samples = get_cfg("pipeline.data_limits.max_cs_samples", default=None, config_name="pipeline_config")
    else:
        # Fallback defaults (FALLBACK_DEFAULT_OK)
        auto_targets = True
        top_n_targets = 5
        max_targets_to_evaluate = None
        auto_features = True
        top_m_features = 100
        strategy = 'single_task'
        min_cs = 10
        max_rows_per_symbol = None
        max_rows_train = None
        max_cs_samples = None
        run_leakage_diagnostics = False
    
    # CLI overrides (for testing/debugging only - warn user)
    if args.override_max_samples:
        logger.warning("⚠️  Using CLI override for max_samples (testing only - not SST compliant)")
        max_rows_per_symbol = args.override_max_samples
    if args.override_max_rows:
        logger.warning("⚠️  Using CLI override for max_rows (testing only - not SST compliant)")
        max_rows_per_symbol = args.override_max_rows
    
    # Manual overrides (targets/features/families) - these are allowed as they're explicit choices
    targets = args.targets  # Manual target list (overrides auto_targets)
    features = args.features  # Manual feature list (overrides auto_features)
    families = args.families  # Manual family list (overrides config)
    
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
    
    # Determine cache usage (operational flag - allowed in CLI)
    use_cache = not args.no_cache
    
    # Run training with config-driven settings
    try:
        results = trainer.train_with_intelligence(
            auto_targets=auto_targets,
            top_n_targets=top_n_targets,
            max_targets_to_evaluate=max_targets_to_evaluate,
            auto_features=auto_features,
            top_m_features=top_m_features,
            targets=targets,  # Manual override if provided
            features=features,  # Manual override if provided
            families=families,  # Manual override if provided
            strategy=strategy,
            force_refresh=args.force_refresh,
            use_cache=use_cache,
            run_leakage_diagnostics=run_leakage_diagnostics,
            min_cs=min_cs,
            max_rows_per_symbol=max_rows_per_symbol,
            max_rows_train=max_rows_train,
            max_cs_samples=max_cs_samples
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

