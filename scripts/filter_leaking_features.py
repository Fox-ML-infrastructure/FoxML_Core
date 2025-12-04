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
Feature Filtering Utilities

Load exclusion config and filter features to prevent data leakage.
"""


import re
from pathlib import Path
from typing import List, Set
import yaml
import logging

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[1]


def load_exclusion_config(config_path: Path = None) -> dict:
    """Load feature exclusion configuration"""
    if config_path is None:
        config_path = _REPO_ROOT / "CONFIG" / "excluded_features.yaml"
    
    if not config_path.exists():
        logger.warning(f"Exclusion config not found: {config_path}")
        return {}
    
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_excluded_features(config: dict = None) -> Set[str]:
    """
    Get set of features to exclude based on configuration.
    
    Returns:
        Set of feature names to exclude
    """
    if config is None:
        config = load_exclusion_config()
    
    excluded = set()
    
    # Add definite leaks (always excluded)
    if config.get('exclude_definite_leaks', True):
        excluded.update(config.get('definite_leaks', []))
    
    # Add probable leaks (optional, default off)
    if config.get('exclude_probable_leaks', False):
        excluded.update(config.get('probable_leaks', []))
    
    # Add temporal overlap features (for 60m targets, exclude 30m+ windows)
    if config.get('exclude_temporal_overlap', True):
        excluded.update(config.get('temporal_overlap_30m_plus', []))
    
    # Add metadata columns
    if config.get('exclude_metadata', True):
        excluded.update(config.get('metadata_columns', []))
    
    return excluded


def matches_target_pattern(column_name: str, patterns: List[str]) -> bool:
    """Check if column matches any target pattern (regex)"""
    for pattern in patterns:
        if re.match(pattern, column_name):
            return True
    return False


def filter_features(
    all_columns: List[str],
    config: dict = None,
    verbose: bool = True
) -> List[str]:
    """
    Filter feature list to remove leaking/metadata/target columns.
    
    Args:
        all_columns: List of all column names in dataset
        config: Exclusion config (loads from file if None)
        verbose: Whether to log filtering stats
    
    Returns:
        List of safe feature columns
    """
    if config is None:
        config = load_exclusion_config()
    
    # Get excluded feature names
    excluded_set = get_excluded_features(config)
    
    # Get target patterns
    target_patterns = config.get('target_patterns', []) if config.get('exclude_targets', True) else []
    
    # Filter columns
    safe_features = []
    excluded_by_name = []
    excluded_by_pattern = []
    
    for col in all_columns:
        # Check if in exclusion set
        if col in excluded_set:
            excluded_by_name.append(col)
            continue
        
        # Check if matches target pattern
        if matches_target_pattern(col, target_patterns):
            excluded_by_pattern.append(col)
            continue
        
        # Safe feature
        safe_features.append(col)
    
    # Log stats
    if verbose:
        logger.info(f"Feature filtering:")
        logger.info(f"  Total columns: {len(all_columns)}")
        logger.info(f"  Safe features: {len(safe_features)}")
        logger.info(f"  Excluded by name: {len(excluded_by_name)}")
        logger.info(f"  Excluded by pattern: {len(excluded_by_pattern)}")
        
        if excluded_by_name:
            logger.info(f"  Top excluded: {excluded_by_name[:5]}")
    
    return safe_features


def filter_dataframe_features(df, config: dict = None, verbose: bool = True):
    """
    Filter dataframe to keep only safe features.
    
    Args:
        df: pandas or polars DataFrame
        config: Exclusion config
        verbose: Whether to log stats
    
    Returns:
        DataFrame with only safe features
    """
    import pandas as pd
    
    all_columns = df.columns.tolist()
    safe_features = filter_features(all_columns, config, verbose)
    
    # Keep safe features only
    return df[safe_features]


# Convenience function for scripts
def get_safe_feature_list(all_columns: List[str], exclude_probable: bool = False) -> List[str]:
    """
    Quick utility to get safe features from a list of columns.
    
    Args:
        all_columns: All column names
        exclude_probable: If True, also exclude probable leaks (more conservative)
    
    Returns:
        List of safe feature names
    """
    config = load_exclusion_config()
    config['exclude_probable_leaks'] = exclude_probable
    return filter_features(all_columns, config, verbose=False)


if __name__ == "__main__":
    import pandas as pd
    
    # Test on AAPL data
    print("Testing feature filtering...")
    df = pd.read_parquet(_REPO_ROOT / "data/data_labeled/interval=5m/symbol=AAPL/AAPL.parquet")
    
    print(f"\nBefore filtering: {len(df.columns)} columns")
    
    # Filter with default settings (exclude definite leaks only)
    df_safe = filter_dataframe_features(df, verbose=True)
    print(f"\nAfter filtering: {len(df_safe.columns)} columns")
    
    # Filter with conservative settings (exclude probable leaks too)
    config = load_exclusion_config()
    config['exclude_probable_leaks'] = True
    df_conservative = filter_dataframe_features(df, config=config, verbose=True)
    print(f"\nConservative filtering: {len(df_conservative.columns)} columns")
    
    # Show example safe features
    print(f"\nExample safe features (first 20):")
    for i, col in enumerate(df_safe.columns[:20], 1):
        print(f"  {i:2d}. {col}")

