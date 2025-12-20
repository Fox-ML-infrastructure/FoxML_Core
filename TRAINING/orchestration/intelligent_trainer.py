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

# Import config loader for centralized path resolution
try:
    from CONFIG.config_loader import (
        get_experiment_config_path,
        load_experiment_config,
        load_training_config,
        CONFIG_DIR
    )
    _CONFIG_LOADER_AVAILABLE = True
except ImportError:
    _CONFIG_LOADER_AVAILABLE = False
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

# Helper functions to get/load experiment config (with fallback)
# These are defined after logger setup so we can use logger if needed
def _get_experiment_config_path(exp_name: str) -> Path:
    """Get experiment config path using config loader if available, otherwise fallback."""
    if _CONFIG_LOADER_AVAILABLE:
        return get_experiment_config_path(exp_name)
    else:
        return Path("CONFIG/experiments") / f"{exp_name}.yaml"

def _load_experiment_config_safe(exp_name: str) -> Dict[str, Any]:
    """Load experiment config using config loader if available, otherwise fallback."""
    if _CONFIG_LOADER_AVAILABLE:
        try:
            return load_experiment_config(exp_name)
        except FileNotFoundError:
            return {}
    else:
        import yaml
        exp_file = Path("CONFIG/experiments") / f"{exp_name}.yaml"
        if exp_file.exists():
            with open(exp_file, 'r') as f:
                return yaml.safe_load(f) or {}
        return {}

# Print license banner on startup (compliance and commercial use notice)
# CRITICAL: Only print in main process, not in child processes or when suppressed
try:
    import os
    if not os.getenv("FOXML_SUPPRESS_BANNER") and not os.getenv("TRAINER_ISOLATION_CHILD"):
        from TRAINING.common.license_banner import print_license_banner_once
        print_license_banner_once()
except Exception:
    # Don't fail if banner can't be printed
    pass

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


# Import from modular components
from TRAINING.orchestration.intelligent_trainer.utils import json_default as _json_default


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
        experiment_config: Optional['ExperimentConfig'] = None,  # New typed config (optional)
        max_rows_per_symbol: Optional[int] = None,  # For output directory binning
        max_cs_samples: Optional[int] = None  # For output directory binning
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
        
        # Store config limits for output directory binning
        self._max_rows_per_symbol = max_rows_per_symbol
        self._max_cs_samples = max_cs_samples
        
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
        
        # Put ALL runs in RESULTS directory, organized by comparison group metadata
        # Structure: RESULTS/runs/{comparison_group_dir}/{date_time_run_name}/
        # Comparison group dir is computed from configs at startup (data, symbols, n_effective, etc.)
        repo_root = Path(__file__).parent.parent.parent  # Go up from TRAINING/orchestration/ to repo root
        results_dir = repo_root / "RESULTS"
        runs_dir = results_dir / "runs"  # Organize all runs under runs/ subdirectory
        
        # Try to estimate N_effective early (before first target is processed)
        self._n_effective = self._estimate_n_effective_early()
        
        # Compute comparison group directory from configs available at startup
        # This organizes runs by metadata from the start, not moved later
        comparison_group_dir = self._compute_comparison_group_dir_at_startup()
        
        if comparison_group_dir:
            # Create directory structure: RESULTS/runs/{comparison_group_dir}/{date_time_run_name}/
            self.output_dir = runs_dir / comparison_group_dir / output_dir_name
            logger.info(f"📁 Output directory: {self.output_dir} (organized by comparison group: {comparison_group_dir})")
        elif self._n_effective is not None:
            # Fallback: Use sample size bin if comparison group can't be computed
            bin_info = self._get_sample_size_bin(self._n_effective)
            bin_name = bin_info["bin_name"]
            self.output_dir = runs_dir / bin_name / output_dir_name
            logger.info(f"📁 Output directory: {self.output_dir} (organized by sample size bin: {bin_name}, N={self._n_effective})")
            self._bin_info = bin_info
        else:
            # Final fallback: start in _pending/ - will be organized after first target
            self.output_dir = runs_dir / "_pending" / output_dir_name
            logger.info(f"📁 Output directory: {self.output_dir} (will be organized after first target)")
        
        self._run_name = output_dir_name  # Store for reference
        
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
        Try to estimate N_effective early from config limits, existing metadata, or data files.
        
        Priority:
        1. Use configured max_rows_per_symbol * num_symbols (if config limits are set)
        2. Check existing metadata from previous runs
        3. Estimate from data files
        
        Returns:
            Estimated N_effective or None if cannot be determined
        """
        logger.info("🔍 Attempting early N_effective estimation...")
        
        # Method 0: Use configured limits if available (PREFERRED - reflects actual data limits)
        if self._max_rows_per_symbol is not None and self._max_rows_per_symbol > 0:
            # Calculate expected N_effective based on config: max_rows_per_symbol * num_symbols
            # This reflects the actual data that will be loaded, not the full dataset size
            expected_n = self._max_rows_per_symbol * len(self.symbols)
            logger.info(f"🔍 Using configured max_rows_per_symbol={self._max_rows_per_symbol} × {len(self.symbols)} symbols = {expected_n} for output directory binning")
            return expected_n
        
        # Also check experiment config if available
        if self.experiment_config:
            try:
                exp_name = self.experiment_config.name
                if _CONFIG_LOADER_AVAILABLE:
                    exp_file = get_experiment_config_path(exp_name)
                    if exp_file.exists():
                        exp_yaml = load_experiment_config(exp_name)
                        exp_data = exp_yaml.get('data', {})
                        config_max_rows = exp_data.get('max_rows_per_symbol') or exp_data.get('max_samples_per_symbol')
                        if config_max_rows is not None and config_max_rows > 0:
                            expected_n = config_max_rows * len(self.symbols)
                            logger.info(f"🔍 Using experiment config max_rows_per_symbol={config_max_rows} × {len(self.symbols)} symbols = {expected_n} for output directory binning")
                            return expected_n
                else:
                    # Fallback for when config loader is not available
                    exp_yaml = _load_experiment_config_safe(exp_name)
                    exp_data = exp_yaml.get('data', {})
                    config_max_rows = exp_data.get('max_rows_per_symbol') or exp_data.get('max_samples_per_symbol')
                    if config_max_rows is not None and config_max_rows > 0:
                        expected_n = config_max_rows * len(self.symbols)
                        logger.info(f"🔍 Using experiment config max_rows_per_symbol={config_max_rows} × {len(self.symbols)} symbols = {expected_n} for output directory binning")
                        return expected_n
            except Exception as e:
                logger.debug(f"Could not read max_rows_per_symbol from experiment config: {e}")
        
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
            logger.info(f"🔍 Sampling {len(sample_symbols)} symbols from {self.data_dir} to estimate N_effective")
            
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
                    logger.warning(f"⚠️  Data file not found for {symbol} (tried: {[str(p) for p in possible_paths]})")
                    continue
                
                logger.info(f"  ✓ Found data file for {symbol}: {data_path}")
                
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
        
        logger.info("⚠️  Could not determine N_effective early, will use _pending/ and organize after first target")
        return None
    
    def _compute_comparison_group_dir_at_startup(self) -> Optional[str]:
        """
        Compute comparison group directory name from configs available at startup.
        
        Uses metadata available at config load time:
        - data_dir, symbols, min_cs, max_cs_samples → dataset_signature
        - n_effective (estimated) → n_effective
        - view/routing (if known) → routing_signature
        
        Note: task_signature, model_family, feature_signature come later and are not
        included in the directory name (they're in the comparison group key for diff telemetry).
        
        Returns:
            Comparison group directory name (e.g., "data-012e801c_route-fcabc6e9_n-988")
            or None if cannot be computed
        """
        import hashlib
        
        try:
            from TRAINING.orchestration.utils.diff_telemetry import ComparisonGroup
            
            # Compute dataset signature from available configs
            dataset_parts = []
            
            # Experiment config path (if available)
            if self.experiment_config:
                if _CONFIG_LOADER_AVAILABLE:
                    exp_file = get_experiment_config_path(self.experiment_config.name)
                else:
                    exp_file = _get_experiment_config_path(self.experiment_config.name)
                if exp_file.exists():
                    dataset_parts.append(f"exp:{self.experiment_config.name}")
            
            # Data directory path (normalized)
            if self.data_dir:
                data_dir_str = str(self.data_dir.resolve())
                dataset_parts.append(f"data_dir={data_dir_str}")
            
            # Symbols (sorted for consistency)
            if self.symbols:
                symbols_str = "|".join(sorted(self.symbols))
                dataset_parts.append(f"symbols={symbols_str}")
            
            # Config limits (if available)
            if self._max_cs_samples is not None:
                dataset_parts.append(f"max_cs_samples={self._max_cs_samples}")
            
            # Also check experiment config for min_cs, max_cs_samples
            min_cs = None
            max_cs_samples = self._max_cs_samples
            if self.experiment_config:
                try:
                    exp_name = self.experiment_config.name
                    exp_yaml = _load_experiment_config_safe(exp_name)
                    if exp_yaml:
                        exp_data = exp_yaml.get('data', {})
                        min_cs = exp_data.get('min_cs')
                        if max_cs_samples is None:
                            max_cs_samples = exp_data.get('max_cs_samples') or exp_data.get('max_rows_per_symbol')
                        if min_cs is not None:
                            dataset_parts.append(f"min_cs={min_cs}")
                except Exception:
                    pass
            
            # Compute dataset signature hash
            dataset_signature = None
            if dataset_parts:
                dataset_str = "|".join(sorted(dataset_parts))
                dataset_signature = hashlib.sha256(dataset_str.encode()).hexdigest()[:16]
            
            # Routing signature (default to CROSS_SECTIONAL if not known)
            # This can be refined later when view is determined
            routing_signature = None
            routing_str = "view=CROSS_SECTIONAL"  # Default
            routing_signature = hashlib.sha256(routing_str.encode()).hexdigest()[:16]
            
            # Build partial comparison group (what we know at startup)
            comparison_group = ComparisonGroup(
                dataset_signature=dataset_signature,
                routing_signature=routing_signature,
                n_effective=self._n_effective
            )
            
            # Generate directory name
            dir_name = comparison_group.to_dir_name()
            return dir_name if dir_name != "default" else None
            
        except Exception as e:
            logger.debug(f"Could not compute comparison group directory at startup: {e}")
            return None
    
    def _get_sample_size_bin(self, n_effective: int) -> Dict[str, Any]:
        """
        Bin N_effective into readable ranges for grouping similar sample sizes.
        
        **Boundary Rules (CRITICAL - DO NOT CHANGE WITHOUT VERSIONING):**
        - Boundaries are EXCLUSIVE upper bounds: `bin_min <= N_effective < bin_max`
        - Example: `sample_25k-50k` means `25000 <= N_effective < 50000`
        - This ensures unambiguous binning (50,000 always goes to `sample_50k-100k`, never `sample_25k-50k`)
        
        **Binning Scheme Version:** `sample_bin_v1`
        - If you change thresholds, increment version and update this docstring
        - Old runs retain their original bin metadata for backward compatibility
        
        **Bins (v1):**
        - sample_0-5k: 0 <= N < 5,000
        - sample_5k-10k: 5,000 <= N < 10,000
        - sample_10k-25k: 10,000 <= N < 25,000
        - sample_25k-50k: 25,000 <= N < 50,000
        - sample_50k-100k: 50,000 <= N < 100,000
        - sample_100k-250k: 100,000 <= N < 250,000
        - sample_250k-500k: 250,000 <= N < 500,000
        - sample_500k-1M: 500,000 <= N < 1,000,000
        - sample_1M+: N >= 1,000,000
        
        This groups runs with similar cross-sectional sample sizes together for easy comparison.
        **Note:** Bin is for directory organization only. Trend series keys use stable identity (cohort_id, stage, target)
        and do NOT include bin_name to prevent fragmentation when binning scheme changes.
        
        Args:
            n_effective: Effective sample size
            
        Returns:
            Dict with keys: bin_name, bin_min, bin_max, binning_scheme_version
        """
        BINNING_SCHEME_VERSION = "sample_bin_v1"
        
        # Define bins with EXCLUSIVE upper bounds (bin_min <= N < bin_max)
        bins = [
            (0, 5000, "sample_0-5k"),
            (5000, 10000, "sample_5k-10k"),
            (10000, 25000, "sample_10k-25k"),
            (25000, 50000, "sample_25k-50k"),
            (50000, 100000, "sample_50k-100k"),
            (100000, 250000, "sample_100k-250k"),
            (250000, 500000, "sample_250k-500k"),
            (500000, 1000000, "sample_500k-1M"),
            (1000000, float('inf'), "sample_1M+")
        ]
        
        for bin_min, bin_max, bin_name in bins:
            if bin_min <= n_effective < bin_max:
                return {
                    "bin_name": bin_name,
                    "bin_min": bin_min,
                    "bin_max": bin_max if bin_max != float('inf') else None,
                    "binning_scheme_version": BINNING_SCHEME_VERSION
                }
        
        # Fallback (should never reach here)
        return {
            "bin_name": "sample_unknown",
            "bin_min": None,
            "bin_max": None,
            "binning_scheme_version": BINNING_SCHEME_VERSION
        }
    
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
            bin_info = self._get_sample_size_bin(self._n_effective)
            bin_name = bin_info["bin_name"]
            if not hasattr(self, '_bin_info'):
                self._bin_info = bin_info
            new_output_dir = results_dir / bin_name / self._run_name
            
            if new_output_dir.exists():
                logger.warning(f"Sample size directory {new_output_dir} already exists, not moving")
                self.output_dir = new_output_dir
                self.cache_dir = self.output_dir / "cache"
                self.target_ranking_cache = self.cache_dir / "target_rankings.json"
                self.feature_selection_cache = self.cache_dir / "feature_selections"
                return
            
            import shutil
            bin_info = self._get_sample_size_bin(self._n_effective)
            bin_name = bin_info["bin_name"]
            if not hasattr(self, '_bin_info'):
                self._bin_info = bin_info
            new_output_dir.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"📁 Moving run from {self.output_dir} to {new_output_dir} (N={self._n_effective} determined early, bin={bin_name})")
            try:
                shutil.move(str(self.output_dir), str(new_output_dir))
                self.output_dir = new_output_dir
                self.cache_dir = self.output_dir / "cache"
                self.target_ranking_cache = self.cache_dir / "target_rankings.json"
                self.feature_selection_cache = self.cache_dir / "feature_selections"
                logger.info(f"✅ Organized run by sample size bin (N={self._n_effective}, bin={bin_name}): {self.output_dir}")
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
                # Move the entire run directory to sample-size-bin-organized location
                repo_root = Path(__file__).parent.parent.parent
                results_dir = repo_root / "RESULTS"
                bin_info = self._get_sample_size_bin(self._n_effective)
                bin_name = bin_info["bin_name"]
                if not hasattr(self, '_bin_info'):
                    self._bin_info = bin_info
                new_output_dir = results_dir / bin_name / self._run_name
                
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
                    
                    bin_name = self._get_sample_size_bin(self._n_effective)
                    logger.info(f"✅ Organized run by sample size bin (N={self._n_effective}, bin={bin_name}): {self.output_dir}")
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
                                bin_info = self._get_sample_size_bin(self._n_effective)
                                bin_name = bin_info["bin_name"]
                                if not hasattr(self, '_bin_info'):
                                    self._bin_info = bin_info
                                logger.info(f"📁 Moving run from {self._initial_output_dir} to {new_output_dir} (found via recursive search, N={self._n_effective}, bin={bin_name})")
                                shutil.move(str(self._initial_output_dir), str(new_output_dir))
                                self.output_dir = new_output_dir
                                self.cache_dir = self.output_dir / "cache"
                                self.target_ranking_cache = self.cache_dir / "target_rankings.json"
                                self.feature_selection_cache = self.cache_dir / "feature_selections"
                                logger.info(f"✅ Organized run by sample size bin (N={self._n_effective}, bin={bin_name}): {self.output_dir}")
                                return
                        except Exception as e:
                            logger.debug(f"Failed to read metadata from {metadata_file}: {e}")
                            continue
            except Exception as e2:
                logger.debug(f"Recursive search also failed: {e2}")
        
        # Create target-first structure
        from TRAINING.orchestration.utils.target_first_paths import initialize_run_structure
        initialize_run_structure(self.output_dir)
        
        # Create initial manifest
        from TRAINING.orchestration.utils.manifest import create_manifest
        try:
            create_manifest(self.output_dir, run_id=self.output_dir.name)
        except Exception as e:
            logger.warning(f"Failed to create initial manifest: {e}")
        
        # Keep legacy directories for backward compatibility during transition
        (self.output_dir / "training_results").mkdir(exist_ok=True)
        (self.output_dir / "leakage_diagnostics").mkdir(exist_ok=True)
        
        # Create new structure directories (keep for backward compatibility)
        (self.output_dir / "DECISION" / "TARGET_RANKING").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "DECISION" / "FEATURE_SELECTION").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "REPRODUCIBILITY" / "TARGET_RANKING").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "REPRODUCIBILITY" / "FEATURE_SELECTION").mkdir(parents=True, exist_ok=True)
        
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
        from TRAINING.common.utils.config_hashing import compute_config_hash_from_values
        # Use centralized config hashing for consistency
        return compute_config_hash_from_values(symbols=sorted(symbols), config_hash=config_hash)[:32]  # Truncate for backward compatibility
    
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
        from TRAINING.common.utils.cache_manager import load_cache
        cache_path = self._get_feature_cache_path(target)
        cache_data = load_cache(cache_path, verify_hash=False)
        if cache_data:
            return cache_data.get('selected_features')
        return None
    
    def _save_cached_features(self, target: str, features: List[str]):
        """Save feature selection results to cache."""
        from TRAINING.common.utils.cache_manager import save_cache
        cache_path = self._get_feature_cache_path(target)
        cache_data = {
            'target': target,
            'selected_features': features
        }
        save_cache(cache_path, cache_data, include_timestamp=True)
    
    def rank_targets_auto(
        self,
        top_n: int = 5,
        model_families: Optional[List[str]] = None,
        multi_model_config: Optional[Dict[str, Any]] = None,
        force_refresh: bool = False,
        use_cache: bool = True,
        max_targets_to_evaluate: Optional[int] = None,  # Limit number of targets to evaluate (for faster testing)
        targets_to_evaluate: Optional[List[str]] = None,  # NEW: Whitelist of specific targets to evaluate (works with auto_targets=true)
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
            targets_to_evaluate: Optional whitelist of specific targets to evaluate (works with auto_targets=true)
            target_ranking_config: Optional TargetRankingConfig object [NEW - preferred]
            min_cs: Minimum cross-sectional size per timestamp (loads from config if None)
            max_cs_samples: Maximum samples per timestamp for cross-sectional sampling (loads from config if None)
            max_rows_per_symbol: Maximum rows to load per symbol (loads from config if None)
        
        Returns:
            List of top N target names
        """
        logger.info(f"🎯 Ranking targets (top {top_n})...")
        if max_targets_to_evaluate is not None:
            logger.info(f"📊 max_targets_to_evaluate limit: {max_targets_to_evaluate}")
        
        # Generate cache key
        config_hash = hashlib.md5(
            json.dumps({
                'model_families': model_families or [],
                'symbols': sorted(self.symbols),
                'max_targets_to_evaluate': max_targets_to_evaluate,  # Include limit in cache key
                'targets_to_evaluate': sorted(targets_to_evaluate) if targets_to_evaluate else []  # NEW: Include whitelist in cache key
            }, sort_keys=True).encode()
        ).hexdigest()
        cache_key = self._get_cache_key(self.symbols, config_hash)
        
        # Check cache
        if not force_refresh and use_cache:
            cached = self._load_cached_rankings(cache_key, use_cache=True)
            if cached:
                # Respect max_targets_to_evaluate even when using cache
                if max_targets_to_evaluate is not None and max_targets_to_evaluate > 0:
                    if len(cached) > max_targets_to_evaluate:
                        logger.info(f"✅ Using cached rankings, truncating to {max_targets_to_evaluate} targets (cache had {len(cached)})")
                        cached = cached[:max_targets_to_evaluate]
                    else:
                        logger.info(f"✅ Using cached target rankings ({len(cached)} targets, limit={max_targets_to_evaluate})")
                else:
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
        
        # Filter out excluded target patterns (from experiment config)
        exclude_patterns = []
        if self.experiment_config:
            try:
                import yaml
                exp_name = self.experiment_config.name
                exp_file = _get_experiment_config_path(exp_name)
                if exp_file.exists():
                    exp_yaml = _load_experiment_config_safe(exp_name)
                    intel_training = exp_yaml.get('intelligent_training', {})
                    if intel_training:
                        exclude_patterns = intel_training.get('exclude_target_patterns', [])
                        if exclude_patterns:
                            logger.info(f"📋 Loaded exclude_target_patterns from experiment config: {exclude_patterns}")
                else:
                    logger.debug(f"Experiment config file not found: {exp_file}")
            except Exception as e:
                logger.warning(f"Could not load exclude_target_patterns from experiment config: {e}")
        else:
            logger.debug("No experiment_config available, skipping exclude_target_patterns")
        
        if exclude_patterns:
            original_count = len(targets_dict)
            filtered_targets = {}
            for target_name, target_config in targets_dict.items():
                # Check if target matches any exclusion pattern
                excluded = False
                for pattern in exclude_patterns:
                    if pattern in target_name:
                        excluded = True
                        break
                if not excluded:
                    filtered_targets[target_name] = target_config
            targets_dict = filtered_targets
            excluded_count = original_count - len(targets_dict)
            if excluded_count > 0:
                logger.info(f"📋 Excluded {excluded_count} targets matching patterns: {exclude_patterns}")
                logger.info(f"📋 Remaining {len(targets_dict)} targets after exclusion")
        
        # NEW: Apply targets_to_evaluate whitelist if specified (works with auto_targets=true)
        if targets_to_evaluate:
            original_count = len(targets_dict)
            whitelisted_targets = {}
            targets_to_evaluate_set = set(targets_to_evaluate)
            for target_name, target_config in targets_dict.items():
                if target_name in targets_to_evaluate_set:
                    whitelisted_targets[target_name] = target_config
            targets_dict = whitelisted_targets
            filtered_count = original_count - len(targets_dict)
            if filtered_count > 0:
                logger.info(f"📋 Applied targets_to_evaluate whitelist: {len(targets_dict)} targets remain (filtered out {filtered_count})")
            if len(targets_dict) == 0:
                logger.warning(f"⚠️  targets_to_evaluate whitelist resulted in 0 targets. Check that whitelist targets exist in discovered targets.")
        
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
            output_dir=self.output_dir,  # Pass base output_dir (not target_rankings subdir)
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
                bin_info = self._get_sample_size_bin(self._n_effective)
                bin_name = bin_info["bin_name"]
                if not hasattr(self, '_bin_info'):
                    self._bin_info = bin_info
                logger.info(f"✅ Successfully organized run by sample size bin (N={self._n_effective}, bin={bin_name}): {self.output_dir}")
                logger.info(f"   Moved from: {self._initial_output_dir}")
                logger.info(f"   Moved to: {self.output_dir}")
            else:
                logger.warning("⚠️  Could not determine N_effective, run will stay in _pending/")
                logger.warning(f"   Run directory: {self._initial_output_dir}")
                # Try to help debug - check if REPRODUCIBILITY exists (check both new and old structures)
                repro_check_new = self._initial_output_dir / "REPRODUCIBILITY"
                repro_check_old = self._initial_output_dir / "target_rankings" / "REPRODUCIBILITY"
                if repro_check_new.exists():
                    logger.warning(f"   REPRODUCIBILITY found at: {repro_check_new}")
                elif repro_check_old.exists():
                    logger.warning(f"   REPRODUCIBILITY found at (old structure): {repro_check_old}")
                else:
                    logger.warning(f"   REPRODUCIBILITY not found at: {repro_check_new} or {repro_check_old}")
        
        # Generate metrics rollups after target ranking completes
        try:
            from TRAINING.orchestration.utils.reproducibility_tracker import ReproducibilityTracker
            tracker = ReproducibilityTracker(output_dir=self.output_dir / "target_rankings")
            run_id = self._run_name.replace("_", "-")  # Use run name as run_id
            tracker.generate_metrics_rollups(stage="TARGET_RANKING", run_id=run_id)
            logger.debug("✅ Generated metrics rollups for TARGET_RANKING")
        except Exception as e:
            logger.debug(f"Failed to generate metrics rollups: {e}")
        
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
        feature_selection_config: Optional['FeatureSelectionConfig'] = None,  # New typed config (optional)
        view: str = "CROSS_SECTIONAL",  # Must match target ranking view
        symbol: Optional[str] = None  # Required for SYMBOL_SPECIFIC view
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
            # Copy data config from original (includes bar_interval)
            from copy import deepcopy
            temp_data = deepcopy(self.experiment_config.data) if hasattr(self.experiment_config, 'data') and self.experiment_config.data else None
            
            temp_exp = ExperimentConfig(
                name=self.experiment_config.name,
                data_dir=self.experiment_config.data_dir,
                symbols=self.experiment_config.symbols,
                target=target,
                data=temp_data,
                max_samples_per_symbol=self.experiment_config.max_samples_per_symbol,
                feature_selection_overrides={'top_n': top_m}
            )
            feature_selection_config = build_feature_selection_config(temp_exp)
        
        # LEGACY: Load config if not provided
        if multi_model_config is None and feature_selection_config is None:
            multi_model_config = load_multi_model_config()
        
        # Select features - write directly to REPRODUCIBILITY/FEATURE_SELECTION structure
        target_name_clean = target.replace('/', '_').replace('\\', '_')
        # FIX: Construct output_dir based on view (CROSS_SECTIONAL vs SYMBOL_SPECIFIC)
        if view == "SYMBOL_SPECIFIC" and symbol:
            # SYMBOL_SPECIFIC: REPRODUCIBILITY/FEATURE_SELECTION/SYMBOL_SPECIFIC/{target}/symbol={symbol}/
            feature_output_dir = self.output_dir / "REPRODUCIBILITY" / "FEATURE_SELECTION" / "SYMBOL_SPECIFIC" / target_name_clean / f"symbol={symbol}"
        else:
            # CROSS_SECTIONAL: REPRODUCIBILITY/FEATURE_SELECTION/CROSS_SECTIONAL/{target}/
            feature_output_dir = self.output_dir / "REPRODUCIBILITY" / "FEATURE_SELECTION" / "CROSS_SECTIONAL" / target_name_clean
        
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
        
        # Filter symbols based on view
        symbols_to_use = self.symbols
        if view == "SYMBOL_SPECIFIC" and symbol:
            symbols_to_use = [symbol]
        elif view == "LOSO" and symbol:
            # LOSO: train on all symbols except symbol
            symbols_to_use = [s for s in self.symbols if s != symbol]
        
        selected_features, _ = select_features_for_target(
            target_column=target,
            symbols=symbols_to_use,
            data_dir=self.data_dir,
            model_families_config=model_families_config,
            multi_model_config=multi_model_config,
            top_n=top_m,
            output_dir=feature_output_dir,
            feature_selection_config=feature_selection_config,  # Pass typed config if available
            explicit_interval=explicit_interval,  # Pass explicit interval to avoid auto-detection warnings
            experiment_config=self.experiment_config,  # Pass experiment config for data.bar_interval
            view=view,  # Pass view to ensure consistency
            symbol=symbol  # Pass symbol for SYMBOL_SPECIFIC view
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
        decision_apply_mode: bool = False,  # NEW: Enable apply mode for decisions
        decision_dry_run: bool = False,  # NEW: Dry-run mode (show patch without applying)
        decision_min_level: int = 2,  # NEW: Minimum decision level to apply
        targets: Optional[List[str]] = None,
        features: Optional[List[str]] = None,
        families: Optional[List[str]] = None,
        strategy: str = 'single_task',
        use_cache: bool = True,
        run_leakage_diagnostics: bool = False,
        max_targets_to_evaluate: Optional[int] = None,  # Limit number of targets to evaluate (for faster testing)
        targets_to_evaluate: Optional[List[str]] = None,  # NEW: Whitelist of specific targets to evaluate (works with auto_targets=true)
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
        
        # Pre-run decision hook: Load latest decision and optionally apply to config
        resolved_config_patch = {}
        decision_artifact_dir = None
        try:
            from TRAINING.decisioning.decision_engine import DecisionEngine
            from TRAINING.orchestration.utils.cohort_metadata_extractor import extract_cohort_metadata
            
            # Try to extract cohort metadata early (for decision loading)
            # This is approximate - actual cohort_id will be computed later
            try:
                cohort_metadata = extract_cohort_metadata(
                    symbols=self.symbols,
                    min_cs=train_kwargs.get('min_cs', 10),
                    max_cs_samples=train_kwargs.get('max_cs_samples')
                )
                cohort_id = None
                segment_id = None
                if cohort_metadata:
                    # Compute approximate cohort_id (will be refined later)
                    from TRAINING.orchestration.utils.reproducibility_tracker import ReproducibilityTracker
                    temp_tracker = ReproducibilityTracker(output_dir=self.output_dir)
                    cohort_id = temp_tracker._compute_cohort_id(cohort_metadata, route_type=None)
                    
                    # Try to get segment_id from index
                    repro_dir = self.output_dir / "REPRODUCIBILITY"
                    index_file = repro_dir / "index.parquet"
                    if index_file.exists():
                        try:
                            import pandas as pd
                            df = pd.read_parquet(index_file)
                            cohort_mask = df['cohort_id'] == cohort_id
                            if cohort_mask.any() and 'segment_id' in df.columns:
                                # Get latest segment_id for this cohort
                                segment_id = int(df[cohort_mask]['segment_id'].iloc[-1])
                        except Exception:
                            pass
                
                if cohort_id and (decision_apply_mode or decision_dry_run):
                    repro_dir = self.output_dir / "REPRODUCIBILITY"
                    index_file = repro_dir / "index.parquet"
                    if index_file.exists():
                        engine = DecisionEngine(index_file, apply_mode=decision_apply_mode)
                        latest_decision = engine.load_latest(cohort_id, base_dir=self.output_dir.parent)
                        
                        if latest_decision:
                            logger.info(f"📊 Decision selection: cohort_id={cohort_id}, segment_id={segment_id}, "
                                      f"decision_level={latest_decision.decision_level}, "
                                      f"actions={latest_decision.decision_action_mask}, "
                                      f"reasons={latest_decision.decision_reason_codes}")
                            
                            if latest_decision.decision_level >= decision_min_level:
                                # Create artifact directory for receipts (one location, one format)
                                decision_artifact_dir = self.output_dir / "REPRODUCIBILITY" / "patches"
                                decision_artifact_dir.mkdir(parents=True, exist_ok=True)
                                
                                # Always save receipts (dry-run or apply) - NON-NEGOTIABLE
                                patched_config, patch, warnings = engine.apply_patch(train_kwargs, latest_decision)
                                
                                # Receipt 1: Decision used (always)
                                decision_used_file = decision_artifact_dir / "decision_used.json"
                                with open(decision_used_file, 'w') as f:
                                    json.dump(latest_decision.to_dict(), f, indent=2, default=str)
                                
                                # Receipt 2: Resolved config baseline (always)
                                try:
                                    import yaml
                                    resolved_config_file = decision_artifact_dir / "resolved_config.yaml"
                                    with open(resolved_config_file, 'w') as f:
                                        yaml.dump(train_kwargs.copy(), f, default_flow_style=False, sort_keys=False)
                                except ImportError:
                                    resolved_config_file = decision_artifact_dir / "resolved_config.json"
                                    with open(resolved_config_file, 'w') as f:
                                        json.dump(train_kwargs.copy(), f, indent=2, default=str)
                                
                                # Receipt 3: Applied patch (or "none" if no patch)
                                patch_file = decision_artifact_dir / "applied_patch.json"
                                
                                if decision_dry_run:
                                    # Dry-run: show patch without applying
                                    with open(patch_file, 'w') as f:
                                        json.dump({
                                            "mode": "dry_run",
                                            "decision_run_id": latest_decision.run_id,
                                            "cohort_id": cohort_id,
                                            "segment_id": segment_id,
                                            "patch": patch if patch else "none",
                                            "warnings": warnings,
                                            "keys_changed": list(patch.keys()) if patch else [],
                                        }, f, indent=2, default=str)
                                    
                                    logger.info("="*80)
                                    logger.info("🔍 DRY RUN: Decision Application Preview")
                                    logger.info("="*80)
                                    logger.info(f"Decision selected: {latest_decision.run_id}")
                                    logger.info(f"  Cohort: {cohort_id}, Segment: {segment_id}")
                                    logger.info(f"  Level: {latest_decision.decision_level}, Actions: {latest_decision.decision_action_mask}")
                                    logger.info(f"  Reasons: {latest_decision.decision_reason_codes}")
                                    if patch:
                                        logger.info(f"Patch that WOULD be applied:")
                                        for key, value in patch.items():
                                            old_val = train_kwargs.get(key.split('.')[0]) if '.' in key else train_kwargs.get(key)
                                            logger.info(f"  {key}: {old_val} → {value}")
                                        logger.info(f"Keys that would change: {list(patch.keys())}")
                                    else:
                                        logger.info("No patch (no actions or all actions skipped)")
                                    if warnings:
                                        for w in warnings:
                                            logger.warning(f"  ⚠️  {w}")
                                    logger.info(f"📄 Receipts saved to: {decision_artifact_dir.relative_to(self.output_dir)}/")
                                    logger.info("  - decision_used.json")
                                    logger.info("  - resolved_config.yaml")
                                    logger.info(f"  - applied_patch.json (mode: dry_run)")
                                    logger.info("="*80)
                                elif decision_apply_mode:
                                    # Apply decision patch to config
                                    resolved_config_patch = patch
                                    train_kwargs.update(patched_config)
                                    
                                    # Save patched config (receipt - overwrites resolved_config.yaml)
                                    try:
                                        import yaml
                                        patched_config_file = decision_artifact_dir / "resolved_config.yaml"
                                        with open(patched_config_file, 'w') as f:
                                            yaml.dump(patched_config, f, default_flow_style=False, sort_keys=False)
                                    except ImportError:
                                        patched_config_file = decision_artifact_dir / "resolved_config.json"
                                        with open(patched_config_file, 'w') as f:
                                            json.dump(patched_config, f, indent=2, default=str)
                                    
                                    # Update config_hash to reflect patch
                                    import hashlib
                                    patch_hash = hashlib.sha256(json.dumps(patch, sort_keys=True).encode()).hexdigest()[:8]
                                    train_kwargs['decision_patch_hash'] = patch_hash
                                    
                                    logger.info("="*80)
                                    logger.info("🔧 APPLY MODE: Decision Patch Applied")
                                    logger.info("="*80)
                                    logger.info(f"Decision: {latest_decision.run_id} (cohort={cohort_id}, segment={segment_id})")
                                    if patch:
                                        logger.info(f"Patch applied:")
                                        for key, value in patch.items():
                                            logger.info(f"  {key}: → {value}")
                                        logger.info(f"Keys changed: {list(patch.keys())}")
                                    else:
                                        logger.info("No patch (patch='none')")
                                    if warnings:
                                        for w in warnings:
                                            logger.warning(f"  ⚠️  {w}")
                                    logger.info(f"🔑 Config hash updated with patch_hash: {patch_hash}")
                                    logger.info(f"📄 Receipts saved to: {decision_artifact_dir.relative_to(self.output_dir)}/")
                                    logger.info("  - decision_used.json")
                                    logger.info("  - resolved_config.yaml (patched)")
                                    logger.info(f"  - applied_patch.json (mode: apply)")
                                    logger.info("="*80)
                            else:
                                logger.info(f"⏭️  Decision level {latest_decision.decision_level} < {decision_min_level}, skipping application")
                        else:
                            logger.debug(f"No decision found for cohort_id={cohort_id}")
            except Exception as e:
                logger.debug(f"Pre-run decision loading failed (non-critical): {e}")
                import traceback
                logger.debug(traceback.format_exc())
        except ImportError:
            logger.debug("Decision engine not available, skipping pre-run hook")
        
        # Extract data limits from train_kwargs (passed from main, potentially patched by decisions)
        min_cs = train_kwargs.get('min_cs', 10)
        max_cs_samples = train_kwargs.get('max_cs_samples')
        max_rows_per_symbol = train_kwargs.get('max_rows_per_symbol')
        
        # Step 1: Target selection
        if auto_targets and targets is None:
            logger.info("="*80)
            logger.info("STEP 1: Automatic Target Ranking")
            logger.info("="*80)
            if max_targets_to_evaluate is not None:
                logger.info(f"🔢 max_targets_to_evaluate={max_targets_to_evaluate} (type: {type(max_targets_to_evaluate).__name__})")
            targets = self.rank_targets_auto(
                top_n=top_n_targets,
                use_cache=use_cache,
                max_targets_to_evaluate=max_targets_to_evaluate,
                targets_to_evaluate=targets_to_evaluate,  # NEW: Pass whitelist
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
        
        # Step 1.5: Apply training plan filter BEFORE feature selection (if available)
        # This avoids wasting time selecting features for targets that will be filtered out
        filtered_targets = targets
        filtered_symbols_by_target = {t: self.symbols for t in targets}
        training_plan = None
        training_plan_dir = None
        
        # Check if training plan exists (from previous run or generated earlier)
        potential_plan_dir = self.output_dir / "METRICS" / "training_plan"
        if potential_plan_dir.exists():
            training_plan_dir = potential_plan_dir
            try:
                from TRAINING.orchestration.training_plan_consumer import (
                    apply_training_plan_filter,
                    load_training_plan
                )
                training_plan = load_training_plan(training_plan_dir)
                if training_plan:
                    filtered_targets, filtered_symbols_by_target = apply_training_plan_filter(
                        targets=targets,
                        symbols=self.symbols,
                        training_plan_dir=training_plan_dir,
                        use_cs_plan=True,
                        use_symbol_plan=True
                    )
                    if len(filtered_targets) < len(targets):
                        logger.info(f"📋 Training plan filter applied BEFORE feature selection: {len(targets)} → {len(filtered_targets)} targets")
            except Exception as e:
                logger.debug(f"Could not apply training plan filter before feature selection: {e}, will filter after")
                filtered_targets = targets
        
        # Step 2: Feature selection (per target if auto_features)
        # CRITICAL: Use same view as target ranking for consistency
        # Only select features for filtered_targets to avoid waste
        target_features = {}
        if auto_features and features is None:
            logger.info("="*80)
            logger.info("STEP 2: Automatic Feature Selection")
            logger.info("="*80)
            
            # Load routing decisions to determine view per target
            routing_decisions = {}
            try:
                from TRAINING.ranking.target_routing import load_routing_decisions
                # load_routing_decisions now automatically checks new and legacy locations
                routing_decisions = load_routing_decisions(output_dir=self.output_dir)
                if routing_decisions:
                    # CRITICAL FIX: Log routing decision count and validate consistency
                    n_decisions = len(routing_decisions)
                    logger.info(f"Loaded routing decisions for {n_decisions} targets")
                    # Log summary of routes for debugging
                    route_counts = {}
                    for target, decision in routing_decisions.items():
                        route = decision.get('route', 'UNKNOWN')
                        route_counts[route] = route_counts.get(route, 0) + 1
                    logger.debug(f"Routing decision summary: {route_counts}")
            except Exception as e:
                logger.debug(f"Could not load routing decisions: {e}, using CROSS_SECTIONAL for all targets")
            
            # Only select features for filtered targets (avoids waste)
            for target in filtered_targets:
                # Determine view from routing decision
                route_info = routing_decisions.get(target, {})
                route = route_info.get('route', 'CROSS_SECTIONAL')
                
                # Default to CROSS_SECTIONAL if no routing decision exists
                if not routing_decisions or target not in routing_decisions:
                    route = 'CROSS_SECTIONAL'
                
                # CRITICAL FIX: Check if cross-sectional is explicitly DISABLED in routing plan
                cs_info = route_info.get('cross_sectional', {})
                cs_route_status = cs_info.get('route', 'ENABLED') if isinstance(cs_info, dict) else 'ENABLED'
                
                # Handle different route types
                if route == 'CROSS_SECTIONAL':
                    # CRITICAL FIX: Respect routing plan - skip CS feature selection if DISABLED
                    if cs_route_status == 'DISABLED':
                        logger.warning(
                            f"Skipping cross-sectional feature selection for {target}: "
                            f"CS route is DISABLED in routing plan (reason: {cs_info.get('reason', 'unknown')})"
                        )
                        # Don't select features for this target
                        continue
                    # Cross-sectional feature selection only
                    target_features[target] = self.select_features_auto(
                        target=target,
                        top_m=top_m_features,
                        use_cache=use_cache,
                        view="CROSS_SECTIONAL",
                        symbol=None
                    )
                elif route == 'SYMBOL_SPECIFIC':
                    # Symbol-specific feature selection only
                    # CRITICAL FIX: winner_symbols should come from routing plan, but validate it's not empty
                    winner_symbols = route_info.get('winner_symbols', [])
                    
                    # If winner_symbols is empty or None, check if we should use all symbols
                    # This can happen if routing plan didn't populate winner_symbols correctly
                    if not winner_symbols or len(winner_symbols) == 0:
                        logger.warning(
                            f"⚠️ SYMBOL_SPECIFIC route for {target} has no winner_symbols in routing plan. "
                            f"Falling back to all symbols: {self.symbols}"
                        )
                        winner_symbols = self.symbols
                    else:
                        # Validate winner_symbols are actually in our symbol list
                        valid_symbols = [s for s in winner_symbols if s in self.symbols]
                        if len(valid_symbols) < len(winner_symbols):
                            invalid = set(winner_symbols) - set(self.symbols)
                            logger.warning(
                                f"⚠️ SYMBOL_SPECIFIC route for {target} has invalid symbols in winner_symbols: {invalid}. "
                                f"Using only valid symbols: {valid_symbols}"
                            )
                        winner_symbols = valid_symbols if valid_symbols else self.symbols
                    
                    if not winner_symbols:
                        logger.error(f"❌ No valid symbols for SYMBOL_SPECIFIC route for {target}, skipping")
                        continue
                    
                    logger.info(f"📊 SYMBOL_SPECIFIC route for {target}: training {len(winner_symbols)} symbols: {winner_symbols}")
                    target_features[target] = {}
                    for symbol in winner_symbols:
                        target_features[target][symbol] = self.select_features_auto(
                            target=target,
                            top_m=top_m_features,
                            use_cache=use_cache,
                            view="SYMBOL_SPECIFIC",
                            symbol=symbol
                        )
                elif route == 'BOTH':
                    # Both cross-sectional and symbol-specific
                    # Store in a structured format: {'cross_sectional': [...], 'symbol_specific': {symbol: [...]}}
                    cs_features = self.select_features_auto(
                        target=target,
                        top_m=top_m_features,
                        use_cache=use_cache,
                        view="CROSS_SECTIONAL",
                        symbol=None
                    )
                    winner_symbols = route_info.get('winner_symbols', self.symbols)
                    if not winner_symbols:
                        winner_symbols = self.symbols
                    symbol_features = {}
                    for symbol in winner_symbols:
                        symbol_features[symbol] = self.select_features_auto(
                            target=target,
                            top_m=top_m_features,
                            use_cache=use_cache,
                            view="SYMBOL_SPECIFIC",
                            symbol=symbol
                        )
                    target_features[target] = {
                        'cross_sectional': cs_features,
                        'symbol_specific': symbol_features,
                        'route': 'BOTH'
                    }
                elif route == 'BLOCKED':
                    logger.warning(f"Skipping feature selection for {target} (BLOCKED: {route_info.get('reason', 'suspicious score')})")
                    # Don't select features for blocked targets - skip this target
                    target_features[target] = []  # Empty list to avoid KeyError downstream
        elif features:
            # Use same features for all filtered targets
            for target in filtered_targets:
                target_features[target] = features
        
        # Generate metrics rollups after feature selection completes (all targets processed)
        if auto_features and target_features:
            try:
                from TRAINING.orchestration.utils.reproducibility_tracker import ReproducibilityTracker
                tracker = ReproducibilityTracker(output_dir=self.output_dir)
                run_id = self._run_name.replace("_", "-")  # Use run name as run_id
                tracker.generate_metrics_rollups(stage="FEATURE_SELECTION", run_id=run_id)
                logger.debug("✅ Generated metrics rollups for FEATURE_SELECTION")
            except Exception as e:
                logger.debug(f"Failed to generate metrics rollups for FEATURE_SELECTION: {e}")
        
        # Step 2.5: Generate training routing plan (if feature selection completed)
        # Note: training_plan_dir may already be set from Step 1.5
        if target_features and not training_plan_dir:
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
        
        # Apply training plan filter if available (may have been applied earlier)
        # If not already filtered, try to filter now
        if training_plan_dir is None:
            # Try to find training plan directory
            potential_plan_dir = self.output_dir / "METRICS" / "training_plan"
            if potential_plan_dir.exists():
                training_plan_dir = potential_plan_dir
        
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
                            # Also exclude BLOCKED targets (they have empty feature lists)
                            if isinstance(target_features, dict):
                                try:
                                    filtered_target_features = {}
                                    for t, f in target_features.items():
                                        if t in filtered_targets:
                                            # Skip BLOCKED targets (empty list) and targets with no features
                                            if isinstance(f, list) and len(f) == 0:
                                                logger.debug(f"Skipping {t} from target_features (BLOCKED or no features)")
                                                continue
                                            filtered_target_features[t] = f
                                    target_features = filtered_target_features
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
            # Log feature counts per target (handle different structures)
            for target, feat_data in list(target_features.items())[:3]:
                if isinstance(feat_data, list):
                    logger.info(f"  {target}: {len(feat_data)} features (CROSS_SECTIONAL)")
                elif isinstance(feat_data, dict):
                    if 'cross_sectional' in feat_data and 'symbol_specific' in feat_data:
                        # BOTH route
                        cs_count = len(feat_data['cross_sectional']) if isinstance(feat_data['cross_sectional'], list) else 0
                        sym_count = len(feat_data['symbol_specific']) if isinstance(feat_data['symbol_specific'], dict) else 0
                        logger.info(f"  {target}: {cs_count} CS features + {sym_count} symbol-specific sets (BOTH)")
                    else:
                        # SYMBOL_SPECIFIC route
                        sym_count = len(feat_data) if isinstance(feat_data, dict) else 0
                        total_feat_count = sum(len(v) if isinstance(v, list) else 0 for v in feat_data.values()) if isinstance(feat_data, dict) else 0
                        logger.info(f"  {target}: {total_feat_count} features across {sym_count} symbols (SYMBOL_SPECIFIC)")
                else:
                    logger.info(f"  {target}: {type(feat_data).__name__} structure")
            if len(target_features) > 3:
                logger.info(f"  ... and {len(target_features) - 3} more targets")
        
        # Pass selected features and routing decisions to training pipeline
        # If target_features is empty, training will auto-discover features
        features_to_use = target_features if target_features else None
        
        # Pass routing decisions to training so it knows which view to use
        routing_decisions_for_training = {}
        try:
            from TRAINING.ranking.target_routing import load_routing_decisions
            # load_routing_decisions now automatically checks new and legacy locations
            routing_decisions_for_training = load_routing_decisions(output_dir=self.output_dir)
            if routing_decisions_for_training:
                # CRITICAL FIX: Log routing decision count and validate consistency
                n_decisions = len(routing_decisions_for_training)
                logger.info(f"Loaded routing decisions for training: {n_decisions} targets")
                # Validate count matches expected (if we have filtered_targets)
                if 'filtered_targets' in locals() and filtered_targets:
                    if n_decisions != len(filtered_targets):
                        logger.warning(
                            f"⚠️ Routing decision count mismatch: {n_decisions} decisions vs {len(filtered_targets)} filtered targets. "
                            f"This may indicate duplicate entries or stale routing decisions."
                        )
        except Exception as e:
            logger.debug(f"Could not load routing decisions for training: {e}")
        
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
            target_families=target_families_map if target_families_map else None,  # Per-target families from plan
            routing_decisions=routing_decisions_for_training  # Pass routing decisions
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
            
            # Look for feature selection results in REPRODUCIBILITY/FEATURE_SELECTION structure
            feature_selections_dir = self.output_dir / "REPRODUCIBILITY" / "FEATURE_SELECTION"
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
            from TRAINING.orchestration.utils.reproducibility_tracker import ReproducibilityTracker
            # Path is already imported at top of file
            # Find REPRODUCIBILITY directory
            repro_dir = self.output_dir / "REPRODUCIBILITY"
            if not repro_dir.exists():
                # Try alternative location (backward compatibility for old structure)
                # Only check parent if output_dir is a module subdirectory
                if self.output_dir.name in ["target_rankings", "feature_selections", "training_results"]:
                    repro_dir = self.output_dir.parent / "REPRODUCIBILITY"
            
            if repro_dir.exists():
                # Create tracker to access trend summary method
                tracker = ReproducibilityTracker(output_dir=self.output_dir)
                trend_summary = tracker.generate_trend_summary(view="STRICT", min_runs_for_trend=2)
                
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
    
    # Core arguments (now optional - can come from config)
    parser.add_argument('--data-dir', type=Path, required=False,
                       help='Data directory (overrides config, required if not in config)')
    parser.add_argument('--symbols', nargs='+', required=False,
                       help='Symbols to train on (overrides config, required if not in config)')
    parser.add_argument('--output-dir', type=Path, required=False,
                       help='Output directory (overrides config, default: intelligent_output)')
    parser.add_argument('--cache-dir', type=Path,
                       help='Cache directory (overrides config, default: output_dir/cache)')
    
    # Simple config-based mode
    parser.add_argument('--config', type=str,
                       help='Config profile name (loads from CONFIG/training_config/intelligent_training_config.yaml)')
    
    # Target/feature selection (moved to config - CLI only for manual overrides)
    parser.add_argument('--targets', nargs='+',
                       help='Manual target list (overrides config auto_targets)')
    parser.add_argument('--features', nargs='+',
                       help='Manual feature list (overrides config auto_features)')
    
    # Training arguments (moved to config - CLI only for manual overrides)
    parser.add_argument('--families', nargs='+',
                       help='Model families to train (overrides config)')
    
    # Quick presets
    parser.add_argument('--quick', action='store_true',
                       help='Quick test mode: 3 targets, 50 features, limited evaluation')
    parser.add_argument('--full', action='store_true',
                       help='Full production mode: all defaults from config')
    
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
    
    # Decision application
    parser.add_argument('--apply-decisions', type=str, choices=['off', 'dry_run', 'apply'], default='off',
                       help='Decision application mode: off (assist mode), dry_run (show patch without applying), apply (auto-apply patches)')
    
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
    
    # Load intelligent training config (NEW - allows simple command-line usage)
    # Use config loader if available, otherwise fallback to direct path
    intel_config_data = {}
    if _CONFIG_LOADER_AVAILABLE:
        try:
            intel_config_data = load_training_config("intelligent_training_config")
            logger.info("✅ Loaded intelligent training config using config loader")
        except Exception as e:
            logger.warning(f"Could not load intelligent training config via loader: {e}")
    else:
        # Fallback: Try new location first (pipeline/training/), then old (training_config/)
        intel_config_file = Path("CONFIG/pipeline/training/intelligent.yaml")
        if not intel_config_file.exists():
            intel_config_file = Path("CONFIG/training_config/intelligent_training_config.yaml")
        if intel_config_file.exists():
            try:
                import yaml
                with open(intel_config_file, 'r') as f:
                    intel_config_data = yaml.safe_load(f) or {}
                logger.info(f"✅ Loaded intelligent training config from {intel_config_file}")
            except Exception as e:
                logger.warning(f"Could not load intelligent training config: {e}")
    
    # Get config file path for logging (needed for trace output)
    if _CONFIG_LOADER_AVAILABLE:
        try:
            from CONFIG.config_loader import get_config_path
            intel_config_file = get_config_path("intelligent_training_config")
        except:
            intel_config_file = Path("CONFIG/pipeline/training/intelligent.yaml")
    else:
        intel_config_file = Path("CONFIG/pipeline/training/intelligent.yaml")
        if not intel_config_file.exists():
            intel_config_file = Path("CONFIG/training_config/intelligent_training_config.yaml")
    
    # Apply config values if CLI args not provided
    if not args.data_dir and intel_config_data.get('data', {}).get('data_dir'):
        args.data_dir = Path(intel_config_data['data']['data_dir'])
    if not args.symbols and intel_config_data.get('data', {}).get('symbols'):
        args.symbols = intel_config_data['data']['symbols']
    if not args.output_dir and intel_config_data.get('output', {}).get('output_dir'):
        args.output_dir = Path(intel_config_data['output']['output_dir'])
    if not args.cache_dir and intel_config_data.get('output', {}).get('cache_dir'):
        args.cache_dir = Path(intel_config_data['output']['cache_dir']) if intel_config_data['output']['cache_dir'] else None
    
    # Apply quick/full presets
    if args.quick:
        logger.info("🚀 Quick test mode enabled")
        intel_config_data.setdefault('targets', {})['max_targets_to_evaluate'] = 3
        intel_config_data.setdefault('targets', {})['top_n_targets'] = 3
        intel_config_data.setdefault('features', {})['top_m_features'] = 50
    elif args.full:
        logger.info("🏭 Full production mode enabled")
        # Use all config defaults
    
    # Validate required args (either from CLI, config, or experiment config)
    if not args.data_dir:
        parser.error("--data-dir is required (or set in CONFIG/pipeline/training/intelligent.yaml)")
    if not args.symbols:
        parser.error("--symbols is required (or set in CONFIG/pipeline/training/intelligent.yaml)")
    if not args.output_dir:
        args.output_dir = Path('intelligent_output')
    
    # Load intelligent training settings (prioritize new config file, fallback to old system)
    try:
        from CONFIG.config_loader import get_cfg
        _CONFIG_AVAILABLE = True
    except ImportError:
        _CONFIG_AVAILABLE = False
        logger.warning("Config loader not available, using hardcoded defaults")
    
    # Use new intelligent_training_config.yaml if available, otherwise fallback to pipeline_config.yaml
    if intel_config_data:
        # Use new config file
        targets_cfg = intel_config_data.get('targets', {})
        features_cfg = intel_config_data.get('features', {})
        data_cfg = intel_config_data.get('data', {})
        advanced_cfg = intel_config_data.get('advanced', {})
        cache_cfg = intel_config_data.get('cache', {})
        
        # NEW: Merge experiment config data section into data_cfg (experiment config takes priority)
        if experiment_config:
            # Merge experiment config data into data_cfg (experiment config overrides intelligent_training_config)
            if hasattr(experiment_config, 'data') and experiment_config.data:
                # Merge experiment config data values
                exp_data_dict = {
                    'data_dir': str(experiment_config.data_dir) if experiment_config.data_dir else None,
                    'symbols': experiment_config.symbols if experiment_config.symbols else None,
                    'interval': experiment_config.data.bar_interval if experiment_config.data.bar_interval else None,
                    'max_rows_per_symbol': experiment_config.max_samples_per_symbol if hasattr(experiment_config, 'max_samples_per_symbol') else None,
                    'max_samples_per_symbol': experiment_config.max_samples_per_symbol if hasattr(experiment_config, 'max_samples_per_symbol') else None,
                }
                # Update data_cfg with experiment config values (only non-None values)
                for key, value in exp_data_dict.items():
                    if value is not None:
                        data_cfg[key] = value
                logger.debug(f"📋 Merged experiment config data into data_cfg: {exp_data_dict}")
        
            # Also read data section directly from YAML file to get min_cs, max_cs_samples, max_rows_train
            # (these aren't in ExperimentConfig object, so read from YAML)
            try:
                import yaml
                exp_name = experiment_config.name
                exp_file = _get_experiment_config_path(exp_name)
                if exp_file.exists():
                    exp_yaml = _load_experiment_config_safe(exp_name)
                    exp_data_section = exp_yaml.get('data', {})
                    # Merge ALL data section keys from YAML (experiment config takes priority)
                    # This includes: min_cs, max_cs_samples, max_rows_train, max_samples_per_symbol, max_rows_per_symbol
                    for key in ['min_cs', 'max_cs_samples', 'max_rows_train', 'max_samples_per_symbol', 'max_rows_per_symbol']:
                        if key in exp_data_section:
                            data_cfg[key] = exp_data_section[key]
                            logger.debug(f"📋 Loaded {key}={exp_data_section[key]} from experiment config YAML")
            except Exception as e:
                logger.debug(f"Could not load data section from experiment config YAML: {e}")
        
        auto_targets = targets_cfg.get('auto_targets', True)
        top_n_targets = targets_cfg.get('top_n_targets', 10)
        max_targets_to_evaluate = targets_cfg.get('max_targets_to_evaluate', None)
        manual_targets = targets_cfg.get('manual_targets', [])
        targets_to_evaluate = targets_cfg.get('targets_to_evaluate', [])  # NEW: Whitelist support
        
        # Track config source for debug logging
        config_sources = {
            'max_targets_to_evaluate': 'base_config',
            'top_n_targets': 'base_config',
            'targets_to_evaluate': 'base_config'
        }
        
        # NEW: Extract manual_targets, max_targets_to_evaluate, etc. from experiment config if available (overrides config file)
        if experiment_config:
            try:
                import yaml
                # Path is already imported at top of file
                exp_name = experiment_config.name
                exp_file = _get_experiment_config_path(exp_name)
                if exp_file.exists():
                    exp_yaml = _load_experiment_config_safe(exp_name)
                    intel_training = exp_yaml.get('intelligent_training', {})
                    if intel_training:
                        exp_manual_targets = intel_training.get('manual_targets', [])
                        if exp_manual_targets:
                            manual_targets = exp_manual_targets
                            logger.info(f"📋 Using manual targets from experiment config: {manual_targets}")
                        exp_auto_targets = intel_training.get('auto_targets', True)
                        if not exp_auto_targets:
                            auto_targets = False
                            logger.info(f"📋 Disabled auto_targets from experiment config (using manual targets)")
                        # Extract max_targets_to_evaluate from experiment config (overrides base config)
                        exp_max_targets = intel_training.get('max_targets_to_evaluate')
                        logger.debug(f"🔍 DEBUG: Found max_targets_to_evaluate in experiment config: {exp_max_targets} (type: {type(exp_max_targets).__name__})")
                        if exp_max_targets is not None:
                            # Ensure it's an integer (YAML might load as int or string)
                            try:
                                max_targets_to_evaluate = int(exp_max_targets)
                                config_sources['max_targets_to_evaluate'] = 'experiment_config'
                                logger.info(f"📋 Using max_targets_to_evaluate={max_targets_to_evaluate} from experiment config")
                            except (ValueError, TypeError) as e:
                                logger.warning(f"⚠️  Invalid max_targets_to_evaluate value '{exp_max_targets}' in experiment config, ignoring: {e}")
                                # Keep existing value (from base config or default)
                        else:
                            logger.debug(f"🔍 DEBUG: max_targets_to_evaluate not found in experiment config intelligent_training section (current value: {max_targets_to_evaluate})")
                        # Extract top_n_targets from experiment config (overrides base config)
                        exp_top_n = intel_training.get('top_n_targets')
                        if exp_top_n is not None:
                            top_n_targets = exp_top_n
                            config_sources['top_n_targets'] = 'experiment_config'
                            logger.info(f"📋 Using top_n_targets={top_n_targets} from experiment config")
                        # Extract targets_to_evaluate whitelist from experiment config (NEW)
                        exp_targets_whitelist = intel_training.get('targets_to_evaluate', [])
                        if exp_targets_whitelist:
                            targets_to_evaluate = exp_targets_whitelist if isinstance(exp_targets_whitelist, list) else [exp_targets_whitelist]
                            config_sources['targets_to_evaluate'] = 'experiment_config'
                            logger.info(f"📋 Using targets_to_evaluate whitelist from experiment config: {targets_to_evaluate}")
            except Exception as e:
                logger.debug(f"Could not load intelligent_training from experiment config: {e}")
            
            # Fallback: Use targets.primary ONLY if auto_targets is False and no manual_targets specified
            # If auto_targets is True, we should auto-discover targets, not use primary as fallback
            if not manual_targets and not auto_targets and hasattr(experiment_config, 'target') and experiment_config.target:
                manual_targets = [experiment_config.target]
                logger.info(f"📋 Using primary target from experiment config (auto_targets=false): {manual_targets}")
        
        auto_features = features_cfg.get('auto_features', True)
        top_m_features = features_cfg.get('top_m_features', 100)
        manual_features = features_cfg.get('manual_features', [])
        
        # Model families from config (can be overridden by CLI)
        config_families = intel_config_data.get('model_families', [])
        
        strategy = intel_config_data.get('strategy', 'single_task')
        min_cs = data_cfg.get('min_cs', 10)
        # Support both max_rows_per_symbol and max_samples_per_symbol (backward compatibility)
        max_rows_per_symbol = data_cfg.get('max_rows_per_symbol') or data_cfg.get('max_samples_per_symbol', None)
        max_rows_train = data_cfg.get('max_rows_train', None)
        max_cs_samples = data_cfg.get('max_cs_samples', 1000)
        run_leakage_diagnostics = advanced_cfg.get('run_leakage_diagnostics', False)
        
        # Decision application mode
        decisions_cfg = intel_config_data.get('decisions', {})
        decision_apply_mode_config = decisions_cfg.get('apply_mode', 'off')
        decision_min_level = decisions_cfg.get('min_level_to_apply', 2)
        
        use_cache = cache_cfg.get('use_cache', True)
        force_refresh_config = cache_cfg.get('force_refresh', False)
        
        # Check for test mode override (for E2E testing)
        # NOTE: Experiment config takes priority over test config
        use_test_config = args.output_dir and 'test' in str(args.output_dir).lower()
        if use_test_config and intel_config_data.get('test'):
            test_cfg = intel_config_data['test']
            logger.info("📋 Using test configuration (detected 'test' in output-dir)")
            # Only apply test config if experiment config didn't override these values
            # Check if experiment config already set these (experiment config takes priority)
            exp_has_max_targets = config_sources.get('max_targets_to_evaluate') == 'experiment_config'
            exp_has_top_n = config_sources.get('top_n_targets') == 'experiment_config'
            exp_has_top_m = False  # Check separately for top_m_features
            if experiment_config:
                try:
                    exp_name = experiment_config.name
                    exp_yaml = _load_experiment_config_safe(exp_name)
                    if exp_yaml:
                        intel_training = exp_yaml.get('intelligent_training', {})
                        if intel_training:
                            # Double-check by looking at the YAML directly
                            if 'max_targets_to_evaluate' in intel_training:
                                exp_has_max_targets = True
                            if 'top_n_targets' in intel_training:
                                exp_has_top_n = True
                            if 'top_m_features' in intel_training:
                                exp_has_top_m = True
                except Exception:
                    pass
            
            # Only override if experiment config didn't set these values
            if 'max_targets_to_evaluate' in test_cfg and not exp_has_max_targets:
                max_targets_to_evaluate = test_cfg.get('max_targets_to_evaluate')
                config_sources['max_targets_to_evaluate'] = 'test_config'
                logger.info(f"📋 Using max_targets_to_evaluate={max_targets_to_evaluate} from test config (experiment config did not override)")
            elif exp_has_max_targets:
                logger.debug(f"📋 Skipping test config max_targets_to_evaluate (experiment config value={max_targets_to_evaluate} takes priority)")
            if 'top_n_targets' in test_cfg and not exp_has_top_n:
                top_n_targets = test_cfg.get('top_n_targets')
                config_sources['top_n_targets'] = 'test_config'
                logger.info(f"📋 Using top_n_targets={top_n_targets} from test config (experiment config did not override)")
            elif exp_has_top_n:
                logger.debug(f"📋 Skipping test config top_n_targets (experiment config value={top_n_targets} takes priority)")
            if 'top_m_features' in test_cfg and not exp_has_top_m:
                top_m_features = test_cfg.get('top_m_features')
                logger.info(f"📋 Using top_m_features={top_m_features} from test config")
        
        # Debug logging: Show final config values and their sources
        logger.debug(f"🔍 Config precedence summary:")
        logger.debug(f"   max_targets_to_evaluate={max_targets_to_evaluate} (source: {config_sources.get('max_targets_to_evaluate', 'unknown')})")
        logger.debug(f"   top_n_targets={top_n_targets} (source: {config_sources.get('top_n_targets', 'unknown')})")
        if targets_to_evaluate:
            logger.debug(f"   targets_to_evaluate={targets_to_evaluate} (source: {config_sources.get('targets_to_evaluate', 'unknown')})")
    elif _CONFIG_AVAILABLE:
        # Fallback to old config system
        use_test_config = args.output_dir and 'test' in str(args.output_dir).lower()

        if use_test_config:
            test_cfg = get_cfg("test.intelligent_training", default={}, config_name="pipeline_config")
            if test_cfg:
                logger.info("📋 Using test configuration (detected 'test' in output-dir)")
                intel_cfg = test_cfg
            else:
                intel_cfg = get_cfg("intelligent_training", default={}, config_name="pipeline_config")
        else:
            intel_cfg = get_cfg("intelligent_training", default={}, config_name="pipeline_config")

        auto_targets = intel_cfg.get('auto_targets', True)
        top_n_targets = intel_cfg.get('top_n_targets', 5)
        max_targets_to_evaluate = intel_cfg.get('max_targets_to_evaluate', None)
        targets_to_evaluate = intel_cfg.get('targets_to_evaluate', [])  # NEW: Whitelist support
        auto_features = intel_cfg.get('auto_features', True)
        top_m_features = intel_cfg.get('top_m_features', 100)
        strategy = intel_cfg.get('strategy', 'single_task')
        min_cs = intel_cfg.get('min_cs', 10)
        
        # Experiment config overrides test config (experiment config takes priority)
        if experiment_config:
            try:
                import yaml
                exp_name = experiment_config.name
                exp_file = _get_experiment_config_path(exp_name)
                if exp_file.exists():
                    exp_yaml = _load_experiment_config_safe(exp_name)
                    intel_training = exp_yaml.get('intelligent_training', {})
                    if intel_training:
                        # Override with experiment config values (experiment config takes priority)
                        if 'max_targets_to_evaluate' in intel_training:
                            max_targets_to_evaluate = intel_training.get('max_targets_to_evaluate')
                            logger.info(f"📋 Using max_targets_to_evaluate={max_targets_to_evaluate} from experiment config (overrides test config)")
                        if 'top_n_targets' in intel_training:
                            top_n_targets = intel_training.get('top_n_targets')
                            logger.info(f"📋 Using top_n_targets={top_n_targets} from experiment config (overrides test config)")
                        if 'top_m_features' in intel_training:
                            top_m_features = intel_training.get('top_m_features')
                            logger.info(f"📋 Using top_m_features={top_m_features} from experiment config (overrides test config)")
                        # Extract targets_to_evaluate whitelist from experiment config (NEW)
                        exp_targets_whitelist = intel_training.get('targets_to_evaluate', [])
                        if exp_targets_whitelist:
                            targets_to_evaluate = exp_targets_whitelist if isinstance(exp_targets_whitelist, list) else [exp_targets_whitelist]
                            logger.info(f"📋 Using targets_to_evaluate whitelist from experiment config: {targets_to_evaluate}")
            except Exception as e:
                logger.debug(f"Could not load intelligent_training from experiment config: {e}")
        
        # Priority: experiment_config object > intel_cfg dict (backward compatibility)
        # Support both max_rows_per_symbol and max_samples_per_symbol
        if experiment_config and hasattr(experiment_config, 'max_samples_per_symbol'):
            max_rows_per_symbol = experiment_config.max_samples_per_symbol
        else:
            max_rows_per_symbol = intel_cfg.get('max_rows_per_symbol') or intel_cfg.get('max_samples_per_symbol', None)
        
        max_rows_train = intel_cfg.get('max_rows_train', None)
        max_cs_samples = intel_cfg.get('max_cs_samples', None)
        run_leakage_diagnostics = intel_cfg.get('run_leakage_diagnostics', False)
        
        # Decision application mode (legacy config path)
        decisions_cfg = intel_cfg.get('decisions', {})
        decision_apply_mode_config = decisions_cfg.get('apply_mode', 'off')
        decision_min_level = decisions_cfg.get('min_level_to_apply', 2)
        
        use_cache = True
        force_refresh_config = False
        manual_targets = []
        manual_features = []
        config_families = []
        
        # NEW: Extract manual_targets from experiment config if available
        # Load the raw YAML to access intelligent_training section
        if experiment_config:
            try:
                import yaml
                # Path is already imported at top of file
                # Find the experiment config file
                exp_name = experiment_config.name
                exp_file = _get_experiment_config_path(exp_name)
                if exp_file.exists():
                    exp_yaml = _load_experiment_config_safe(exp_name)
                    # Extract intelligent_training section
                    intel_training = exp_yaml.get('intelligent_training', {})
                    if intel_training:
                        exp_manual_targets = intel_training.get('manual_targets', [])
                        if exp_manual_targets:
                            manual_targets = exp_manual_targets
                            logger.info(f"📋 Using manual targets from experiment config: {manual_targets}")
                        # Also check auto_targets setting
                        exp_auto_targets = intel_training.get('auto_targets', True)
                        if not exp_auto_targets:
                            auto_targets = False
                            logger.info(f"📋 Disabled auto_targets from experiment config (using manual targets)")
                        # Extract max_targets_to_evaluate from experiment config (overrides base config)
                        exp_max_targets = intel_training.get('max_targets_to_evaluate')
                        if exp_max_targets is not None:
                            # Ensure it's an integer (YAML might load as int or string)
                            try:
                                max_targets_to_evaluate = int(exp_max_targets)
                                logger.info(f"📋 Using max_targets_to_evaluate={max_targets_to_evaluate} from experiment config")
                            except (ValueError, TypeError) as e:
                                logger.warning(f"⚠️  Invalid max_targets_to_evaluate value '{exp_max_targets}' in experiment config, ignoring: {e}")
                                # Keep existing value (from base config or default)
                        # Extract top_n_targets from experiment config (overrides base config)
                        exp_top_n = intel_training.get('top_n_targets')
                        if exp_top_n is not None:
                            top_n_targets = exp_top_n
                            logger.info(f"📋 Using top_n_targets={top_n_targets} from experiment config")
            except Exception as e:
                logger.debug(f"Could not load intelligent_training from experiment config: {e}")
            
            # Fallback: Use targets.primary ONLY if auto_targets is False and no manual_targets specified
            # If auto_targets is True, we should auto-discover targets, not use primary as fallback
            if not manual_targets and not auto_targets and hasattr(experiment_config, 'target') and experiment_config.target:
                manual_targets = [experiment_config.target]
                logger.info(f"📋 Using primary target from experiment config (auto_targets=false): {manual_targets}")
        
        if max_cs_samples is None:
            max_cs_samples = get_cfg("pipeline.data_limits.max_cs_samples", default=None, config_name="pipeline_config")
    else:
        # Fallback defaults
        auto_targets = True
        top_n_targets = 5
        max_targets_to_evaluate = None
        targets_to_evaluate = []  # NEW: Initialize whitelist
        auto_features = True
        top_m_features = 100
        strategy = 'single_task'
        min_cs = 10
        max_rows_per_symbol = None
        max_rows_train = None
        max_cs_samples = None
        run_leakage_diagnostics = False
        
        # Decision application mode (fallback defaults)
        decision_apply_mode_config = 'off'
        decision_min_level = 2
        
        use_cache = True
        force_refresh_config = False
        manual_targets = []
        manual_features = []
        config_families = []
    
    # ============================================================================
    # CONFIG TRACE: Comprehensive logging of config loading and precedence
    # ============================================================================
    logger.info("=" * 80)
    logger.info("📋 CONFIG TRACE: Configuration Loading and Precedence")
    logger.info("=" * 80)
    
    # Track loaded files
    loaded_files = []
    if intel_config_file.exists():
        loaded_files.append(("intelligent_training_config.yaml", str(intel_config_file.resolve())))
    if experiment_config:
        exp_file = _get_experiment_config_path(experiment_config.name)
        if exp_file.exists():
            loaded_files.append((f"experiment: {experiment_config.name}.yaml", str(exp_file.resolve())))
    
    logger.info(f"📁 Loaded config files (in order):")
    for i, (name, path) in enumerate(loaded_files, 1):
        logger.info(f"   {i}. {name}")
        logger.info(f"      → {path}")
    
    # Track config value sources (before CLI overrides)
    config_trace = {}
    
    def trace_value(key: str, value: Any, source: str, section: str = ""):
        """Track where a config value came from"""
        full_key = f"{section}.{key}" if section else key
        if full_key not in config_trace:
            config_trace[full_key] = []
        config_trace[full_key].append({
            'value': value,
            'source': source
        })
    
    # Trace key config values from intelligent_training_config.yaml
    if intel_config_data:
        data_cfg = intel_config_data.get('data', {})
        targets_cfg = intel_config_data.get('targets', {})
        features_cfg = intel_config_data.get('features', {})
        
        trace_value("min_cs", min_cs, 
                    f"intelligent_training_config.yaml → data.min_cs = {data_cfg.get('min_cs', 'default=10')}",
                    "data")
        trace_value("max_cs_samples", max_cs_samples,
                    f"intelligent_training_config.yaml → data.max_cs_samples = {data_cfg.get('max_cs_samples', 'default=1000')}",
                    "data")
        # Check if experiment config overrode this value
        exp_max_rows = None
        if experiment_config:
            try:
                import yaml
                exp_name = experiment_config.name
                exp_file = _get_experiment_config_path(exp_name)
                if exp_file.exists():
                    exp_yaml = _load_experiment_config_safe(exp_name)
                    exp_data = exp_yaml.get('data', {})
                    exp_max_rows = exp_data.get('max_rows_per_symbol') or exp_data.get('max_samples_per_symbol')
            except Exception:
                pass
        
        if exp_max_rows is not None:
            trace_value("max_rows_per_symbol", max_rows_per_symbol,
                        f"intelligent_training_config.yaml → data.max_rows_per_symbol = {data_cfg.get('max_rows_per_symbol') or data_cfg.get('max_samples_per_symbol', 'default=None')}",
                        "data")
            trace_value("max_rows_per_symbol", exp_max_rows,
                        f"experiment: {experiment_config.name}.yaml → data.max_rows_per_symbol = {exp_max_rows} (OVERRIDE)",
                        "data")
        else:
            trace_value("max_rows_per_symbol", max_rows_per_symbol,
                        f"intelligent_training_config.yaml → data.max_rows_per_symbol = {data_cfg.get('max_rows_per_symbol') or data_cfg.get('max_samples_per_symbol', 'default=None')}",
                        "data")
        trace_value("max_rows_train", max_rows_train,
                    f"intelligent_training_config.yaml → data.max_rows_train = {data_cfg.get('max_rows_train', 'default=None')}",
                    "data")
        trace_value("auto_targets", auto_targets,
                    f"intelligent_training_config.yaml → targets.auto_targets = {targets_cfg.get('auto_targets', 'default=True')}",
                    "targets")
        trace_value("top_n_targets", top_n_targets,
                    f"intelligent_training_config.yaml → targets.top_n_targets = {targets_cfg.get('top_n_targets', 'default=10')}",
                    "targets")
        trace_value("max_targets_to_evaluate", max_targets_to_evaluate,
                    f"intelligent_training_config.yaml → targets.max_targets_to_evaluate = {targets_cfg.get('max_targets_to_evaluate', 'default=None')}",
                    "targets")
        trace_value("auto_features", auto_features,
                    f"intelligent_training_config.yaml → features.auto_features = {features_cfg.get('auto_features', 'default=True')}",
                    "features")
        trace_value("top_m_features", top_m_features,
                    f"intelligent_training_config.yaml → features.top_m_features = {features_cfg.get('top_m_features', 'default=100')}",
                    "features")
    
    # Check for experiment config overrides
    if experiment_config:
        exp_file = _get_experiment_config_path(experiment_config.name)
        if exp_file.exists():
            try:
                exp_yaml = _load_experiment_config_safe(experiment_config.name)
                exp_data = exp_yaml.get('data', {})
                # Trace all data section keys that exist in experiment config
                for key in ['min_cs', 'max_cs_samples', 'max_rows_train', 'max_samples_per_symbol', 'max_rows_per_symbol']:
                    if key in exp_data:
                        trace_value(key, exp_data[key],
                                    f"experiment: {experiment_config.name}.yaml → data.{key} = {exp_data[key]} (OVERRIDE)",
                                    "data")
                # Trace intelligent_training section overrides
                exp_intel = exp_yaml.get('intelligent_training', {})
                if exp_intel:
                    for key in ['max_targets_to_evaluate', 'top_n_targets', 'auto_targets']:
                        if key in exp_intel:
                            trace_value(key, exp_intel[key],
                                        f"experiment: {experiment_config.name}.yaml → intelligent_training.{key} = {exp_intel[key]} (OVERRIDE)",
                                        "targets" if key in ['max_targets_to_evaluate', 'top_n_targets', 'auto_targets'] else "")
            except Exception as e:
                logger.debug(f"Could not trace experiment config: {e}")
    
    # CLI overrides (for testing/debugging only - warn user)
    if args.override_max_samples:
        logger.warning("⚠️  Using CLI override for max_samples (testing only - not SST compliant)")
        max_rows_per_symbol = args.override_max_samples
        trace_value("max_rows_per_symbol", max_rows_per_symbol, 
                    f"CLI --override-max-samples = {args.override_max_samples} (OVERRIDE)",
                    "data")
    if args.override_max_rows:
        logger.warning("⚠️  Using CLI override for max_rows (testing only - not SST compliant)")
        max_rows_per_symbol = args.override_max_rows
        trace_value("max_rows_per_symbol", max_rows_per_symbol,
                    f"CLI --override-max-rows = {args.override_max_rows} (OVERRIDE)",
                    "data")
    if args.targets:
        trace_value("manual_targets", args.targets, "CLI --targets (OVERRIDE)", "targets")
    if args.features:
        trace_value("manual_features", args.features, "CLI --features (OVERRIDE)", "features")
    if args.families:
        trace_value("model_families", args.families, "CLI --families (OVERRIDE)", "training")
    
    # Log final resolved values with source chain
    logger.info("")
    logger.info("🔍 Key Config Values (with source chain):")
    key_configs = [
        ("data.min_cs", min_cs),
        ("data.max_cs_samples", max_cs_samples),
        ("data.max_rows_per_symbol", max_rows_per_symbol),
        ("data.max_rows_train", max_rows_train),
        ("targets.auto_targets", auto_targets),
        ("targets.top_n_targets", top_n_targets),
        ("targets.max_targets_to_evaluate", max_targets_to_evaluate),
        ("features.auto_features", auto_features),
        ("features.top_m_features", top_m_features),
    ]
    
    for key, final_value in key_configs:
        if key in config_trace:
            sources = config_trace[key]
            logger.info(f"   {key}: {final_value}")
            for i, source_info in enumerate(sources, 1):
                arrow = "→" if i < len(sources) else "✓"
                logger.info(f"      {i}. {arrow} {source_info['source']}")
        else:
            logger.info(f"   {key}: {final_value} (no trace - using default or hardcoded)")
    
    # Check for conflicts (same key from multiple sources with different values)
    logger.info("")
    logger.info("⚠️  Conflict Detection:")
    conflicts = []
    for key, sources in config_trace.items():
        if len(sources) > 1:
            values = [s['value'] for s in sources]
            # Check if values are actually different (handle None, int/str conversions)
            unique_values = set(str(v) if v is not None else 'None' for v in values)
            if len(unique_values) > 1:
                conflicts.append((key, sources))
    
    if conflicts:
        logger.warning(f"   Found {len(conflicts)} potential conflicts:")
        for key, sources in conflicts:
            logger.warning(f"      {key}:")
            for source_info in sources:
                logger.warning(f"         - {source_info['source']}")
    else:
        logger.info("   ✅ No conflicts detected (all sources agree or override cleanly)")
    
    # Log working directory and config paths
    logger.info("")
    logger.info("📂 Environment:")
    logger.info(f"   Working directory: {os.getcwd()}")
    logger.info(f"   Project root: {_PROJECT_ROOT}")
    if _CONFIG_LOADER_AVAILABLE:
        logger.info(f"   Config directory: {CONFIG_DIR.resolve()}")
    else:
        logger.info(f"   Config directory: {Path('CONFIG').resolve()}")
    
    logger.info("=" * 80)
    logger.info("")
    
    # ============================================================================
    # End of config trace
    # ============================================================================
    
    # Manual overrides (targets/features/families) - CLI > config > defaults
    targets = args.targets if args.targets else (manual_targets if manual_targets else None)
    features = args.features if args.features else (manual_features if manual_features else None)
    families = args.families if args.families else (config_families if config_families else None)
    
    # If families still not set, use defaults
    if not families:
        families = ['lightgbm', 'xgboost', 'random_forest']
    
    # Create orchestrator
    # Pass config limits for output directory binning (use configured values, not full dataset size)
    trainer = IntelligentTrainer(
        data_dir=args.data_dir,
        symbols=args.symbols,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        experiment_config=experiment_config,  # Pass experiment config if loaded
        max_rows_per_symbol=max_rows_per_symbol,  # For output directory binning
        max_cs_samples=max_cs_samples  # For output directory binning
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
    
    # Determine cache usage (CLI overrides config)
    if args.no_cache:
        use_cache = False
    elif args.force_refresh:
        use_cache = True  # Still use cache, but force refresh
    else:
        use_cache = use_cache  # From config
    
    # Force refresh from config or CLI
    force_refresh = args.force_refresh or force_refresh_config
    
    # Decision application mode (CLI overrides config)
    decision_mode = args.apply_decisions if hasattr(args, 'apply_decisions') and args.apply_decisions else decision_apply_mode_config
    decision_apply_mode = (decision_mode == 'apply')
    decision_dry_run = (decision_mode == 'dry_run')
    
    # Run training with config-driven settings
    try:
        results = trainer.train_with_intelligence(
            auto_targets=auto_targets,
            top_n_targets=top_n_targets,
            max_targets_to_evaluate=max_targets_to_evaluate,
            targets_to_evaluate=targets_to_evaluate,  # NEW: Pass whitelist
            auto_features=auto_features,
            top_m_features=top_m_features,
            targets=targets,  # Manual override if provided
            features=features,  # Manual override if provided
            families=families,  # Manual override if provided
            strategy=strategy,
            force_refresh=force_refresh,
            use_cache=use_cache,
            run_leakage_diagnostics=run_leakage_diagnostics,
            min_cs=min_cs,
            max_rows_per_symbol=max_rows_per_symbol,
            max_rows_train=max_rows_train,
            max_cs_samples=max_cs_samples,
            decision_apply_mode=decision_apply_mode,
            decision_dry_run=decision_dry_run,
            decision_min_level=decision_min_level if 'decision_min_level' in locals() else 2
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

