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
Target Ranking Module

Extracted from scripts/rank_target_predictability.py to enable integration
into the training pipeline. All leakage-free behavior is preserved by
reusing the original functions.
"""

import sys
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import warnings

# Add project root to path for imports
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Import original functions to preserve leakage-free behavior
from TRAINING.ranking.rank_target_predictability import (
    TargetPredictabilityScore,
    evaluate_target_predictability as _evaluate_target_predictability,
    discover_all_targets as _discover_all_targets,
    load_target_configs as _load_target_configs,
    save_rankings as _save_rankings
)
from TRAINING.utils.task_types import TargetConfig, TaskType

# Suppress expected warnings
warnings.filterwarnings('ignore', message='X does not have valid feature names')
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=RuntimeWarning, module='sklearn')

logger = logging.getLogger(__name__)


def evaluate_target_predictability(
    target_name: str,
    target_config: Dict[str, Any] | TargetConfig,
    symbols: List[str],
    data_dir: Path,
    model_families: List[str],
    multi_model_config: Dict[str, Any] = None,
    output_dir: Path = None,
    min_cs: int = 10,
    max_cs_samples: Optional[int] = None,
    max_rows_per_symbol: int = 50000
) -> TargetPredictabilityScore:
    """
    Evaluate predictability of a single target across symbols.
    
    This is a wrapper around the original function to preserve all
    leakage-free behavior (PurgedTimeSeriesSplit, leakage filtering, etc.).
    
    Args:
        target_name: Display name of target
        target_config: TargetConfig object or dict with target config
        symbols: List of symbols to evaluate on
        data_dir: Directory containing symbol data
        model_families: List of model family names to use
        multi_model_config: Multi-model config dict
        output_dir: Optional output directory for results
        min_cs: Minimum cross-sectional size per timestamp
        max_cs_samples: Maximum samples per timestamp for cross-sectional sampling
        max_rows_per_symbol: Maximum rows to load per symbol
    
    Returns:
        TargetPredictabilityScore object with predictability metrics
    """
    return _evaluate_target_predictability(
        target_name=target_name,
        target_config=target_config,
        symbols=symbols,
        data_dir=data_dir,
        model_families=model_families,
        multi_model_config=multi_model_config,
        output_dir=output_dir,
        min_cs=min_cs,
        max_cs_samples=max_cs_samples,
        max_rows_per_symbol=max_rows_per_symbol
    )


def discover_targets(
    symbol: str,
    data_dir: Path
) -> Dict[str, TargetConfig]:
    """
    Auto-discover all valid targets from data (non-degenerate).
    
    Preserves all leakage-free filtering (excludes first_touch targets, etc.).
    
    Args:
        symbol: Symbol to use for discovery
        data_dir: Directory containing symbol data
    
    Returns:
        Dict mapping target_name -> TargetConfig
    """
    return _discover_all_targets(symbol, data_dir)


def load_target_configs() -> Dict[str, Dict]:
    """
    Load target configurations from CONFIG/target_configs.yaml.
    
    Returns:
        Dict mapping target_name -> target config dict
    """
    return _load_target_configs()


def rank_targets(
    targets: Dict[str, TargetConfig | Dict[str, Any]],
    symbols: List[str],
    data_dir: Path,
    model_families: List[str],
    multi_model_config: Dict[str, Any] = None,
    output_dir: Path = None,
    min_cs: int = 10,
    max_cs_samples: Optional[int] = None,
    max_rows_per_symbol: int = 50000,
    top_n: Optional[int] = None
) -> List[TargetPredictabilityScore]:
    """
    Rank multiple targets by predictability.
    
    This function evaluates all targets and returns them sorted by
    composite predictability score. All leakage-free behavior is preserved.
    
    Args:
        targets: Dict mapping target_name -> TargetConfig or config dict
        symbols: List of symbols to evaluate on
        data_dir: Directory containing symbol data
        model_families: List of model family names to use
        multi_model_config: Multi-model config dict
        output_dir: Optional output directory for results
        min_cs: Minimum cross-sectional size per timestamp
        max_cs_samples: Maximum samples per timestamp for cross-sectional sampling
        max_rows_per_symbol: Maximum rows to load per symbol
        top_n: Optional limit on number of top targets to return
    
    Returns:
        List of TargetPredictabilityScore objects, sorted by composite_score (descending)
    """
    results = []
    
    logger.info(f"Ranking {len(targets)} targets across {len(symbols)} symbols")
    logger.info(f"Model families: {', '.join(model_families)}")
    
    for idx, (target_name, target_config) in enumerate(targets.items(), 1):
        logger.info(f"[{idx}/{len(targets)}] Evaluating {target_name}...")
        
        try:
            result = evaluate_target_predictability(
                target_name=target_name,
                target_config=target_config,
                symbols=symbols,
                data_dir=data_dir,
                model_families=model_families,
                multi_model_config=multi_model_config,
                output_dir=output_dir,
                min_cs=min_cs,
                max_cs_samples=max_cs_samples,
                max_rows_per_symbol=max_rows_per_symbol
            )
            
            # Skip degenerate targets (marked with mean_score = -999)
            if result.mean_score != -999.0:
                results.append(result)
            else:
                logger.info(f"  Skipped degenerate target: {target_name}")
        
        except Exception as e:
            logger.error(f"  Failed to evaluate {target_name}: {e}")
            # Continue with next target
    
    # Sort by composite score (descending)
    results.sort(key=lambda r: r.composite_score, reverse=True)
    
    # Apply top_n limit if specified
    if top_n is not None and top_n > 0:
        results = results[:top_n]
        logger.info(f"Returning top {len(results)} targets")
    
    # Save rankings if output_dir provided
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        _save_rankings(results, output_dir)
    
    return results

