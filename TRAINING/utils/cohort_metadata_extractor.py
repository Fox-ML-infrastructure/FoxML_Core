"""
Cohort Metadata Extractor

Utility module for extracting cohort metadata from various data sources
for use with cohort-aware reproducibility tracking.

This module provides a unified interface for extracting:
- N_effective_cs (sample size)
- n_symbols (number of symbols)
- date_range (start/end timestamps)
- cs_config (cross-sectional configuration: min_cs, max_cs_samples, etc.)

Usage:
    from TRAINING.utils.cohort_metadata_extractor import extract_cohort_metadata, format_for_reproducibility_tracker
    
    # Example 1: From numpy arrays and dataframes (target ranking)
    metadata = extract_cohort_metadata(
        X=X,  # numpy array or pandas DataFrame
        symbols=symbols,  # list of symbols or array
        time_vals=time_vals,  # optional timestamps
        min_cs=min_cs,
        max_cs_samples=max_cs_samples,
        mtf_data=mtf_data  # optional: dict of symbol -> DataFrame
    )
    
    # Format for reproducibility tracker (splits into metrics and additional_data)
    cohort_metrics, cohort_additional_data = format_for_reproducibility_tracker(metadata)
    
    # Then pass to reproducibility tracker
    tracker.log_comparison(
        stage="target_ranking",
        item_name=target_name,
        metrics={"mean_score": score, **cohort_metrics},
        additional_data={**cohort_additional_data}
    )
    
    # Example 2: From symbols list only (feature selection)
    metadata = extract_cohort_metadata(
        symbols=symbols,
        mtf_data=mtf_data  # optional: for date range extraction
    )
    cohort_metrics, cohort_additional_data = format_for_reproducibility_tracker(metadata)
    
    # Example 3: Direct overrides (when you have exact values)
    metadata = extract_cohort_metadata(
        n_samples=10000,  # Direct override
        n_symbols=5,  # Direct override
        date_start="2024-01-01",
        date_end="2024-03-31",
        min_cs=10,
        max_cs_samples=1000
    )
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Union, List, Tuple
import logging
import hashlib

logger = logging.getLogger(__name__)


def extract_cohort_metadata(
    X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
    symbols: Optional[Union[List[str], np.ndarray, pd.Series]] = None,
    time_vals: Optional[Union[np.ndarray, pd.Series, List]] = None,
    y: Optional[Union[np.ndarray, pd.Series]] = None,  # Label vector for data fingerprint
    mtf_data: Optional[Dict[str, pd.DataFrame]] = None,
    min_cs: Optional[int] = None,
    max_cs_samples: Optional[int] = None,
    leakage_filter_version: Optional[str] = None,
    universe_id: Optional[str] = None,
    n_samples: Optional[int] = None,  # Direct override for sample size
    n_symbols: Optional[int] = None,  # Direct override for symbol count
    date_start: Optional[Union[str, pd.Timestamp]] = None,  # Direct override for date range
    date_end: Optional[Union[str, pd.Timestamp]] = None,
    compute_data_fingerprint: bool = True,  # Whether to compute data fingerprint
    compute_per_symbol_stats: bool = True  # Whether to compute per-symbol statistics
) -> Dict[str, Any]:
    """
    Extract cohort metadata from various data sources.
    
    This function tries multiple strategies to extract the required metadata:
    1. Direct parameters (n_samples, n_symbols, date_start, date_end) - highest priority
    2. From X array/DataFrame (sample size)
    3. From symbols list/array (symbol count)
    4. From time_vals array (date range)
    5. From mtf_data dict (date range, symbol count)
    
    Args:
        X: Feature matrix (numpy array or pandas DataFrame) - used for sample size
        symbols: List/array of symbols - used for symbol count
        time_vals: Array of timestamps - used for date range extraction
        mtf_data: Dict of symbol -> DataFrame - used for date range and symbol count
        min_cs: Minimum cross-sectional size (for cs_config)
        max_cs_samples: Maximum cross-sectional samples (for cs_config)
        leakage_filter_version: Leakage filter version (for cs_config)
        universe_id: Universe identifier (for cs_config)
        n_samples: Direct override for sample size (highest priority)
        n_symbols: Direct override for symbol count (highest priority)
        date_start: Direct override for date range start (highest priority)
        date_end: Direct override for date range end (highest priority)
    
    Returns:
        Dict with keys:
            - N_effective_cs: int (sample size)
            - n_symbols: int (number of symbols)
            - date_range: dict with 'start_ts' and 'end_ts' (ISO format strings)
            - cs_config: dict with min_cs, max_cs_samples, leakage_filter_version, universe_id
    """
    metadata = {}
    
    # 1. Extract N_effective_cs (sample size)
    if n_samples is not None:
        metadata['N_effective_cs'] = int(n_samples)
    elif X is not None:
        try:
            if isinstance(X, pd.DataFrame):
                metadata['N_effective_cs'] = len(X)
            elif isinstance(X, np.ndarray):
                metadata['N_effective_cs'] = X.shape[0] if len(X.shape) > 0 else 0
            else:
                metadata['N_effective_cs'] = len(X)
        except (TypeError, AttributeError, IndexError) as e:
            logger.debug(f"Could not extract N_effective_cs from X: {e}")
            pass
    
    # 2. Extract n_symbols and symbols list
    symbols_list = None
    if symbols is not None:
        if isinstance(symbols, (list, tuple)):
            # Get unique symbols and store the list
            unique_symbols = set(symbols)
            metadata['n_symbols'] = len(unique_symbols)
            symbols_list = sorted(list(unique_symbols))  # Sorted for stable diffs
        elif isinstance(symbols, (np.ndarray, pd.Series)):
            unique_symbols = np.unique(symbols) if isinstance(symbols, np.ndarray) else symbols.unique()
            metadata['n_symbols'] = len(unique_symbols)
            symbols_list = sorted([str(s) for s in unique_symbols])  # Convert to strings and sort
        else:
            try:
                unique_symbols = set(symbols)
                metadata['n_symbols'] = len(unique_symbols)
                symbols_list = sorted([str(s) for s in unique_symbols])
            except (TypeError, AttributeError):
                pass
    elif mtf_data is not None:
        metadata['n_symbols'] = len(mtf_data)
        symbols_list = sorted(list(mtf_data.keys()))  # Use dict keys as symbols
    
    # Override with direct n_symbols if provided
    if n_symbols is not None:
        metadata['n_symbols'] = int(n_symbols)
    
    # Store symbols list if we extracted it
    if symbols_list:
        metadata['symbols'] = symbols_list
    
    # 3. Extract date_range
    date_range = {}
    
    # Try direct overrides first
    if date_start is not None:
        try:
            if isinstance(date_start, str):
                date_range['start_ts'] = pd.Timestamp(date_start).isoformat()
            else:
                date_range['start_ts'] = pd.Timestamp(date_start).isoformat()
        except Exception as e:
            logger.debug(f"Failed to parse date_start {date_start}: {e}")
    
    if date_end is not None:
        try:
            if isinstance(date_end, str):
                date_range['end_ts'] = pd.Timestamp(date_end).isoformat()
            else:
                date_range['end_ts'] = pd.Timestamp(date_end).isoformat()
        except Exception as e:
            logger.debug(f"Failed to parse date_end {date_end}: {e}")
    
    # If we don't have both, try to extract from time_vals
    if 'start_ts' not in date_range or 'end_ts' not in date_range:
        if time_vals is not None:
            try:
                time_vals_len = len(time_vals)
            except (TypeError, AttributeError):
                time_vals_len = 0
            if time_vals_len > 0:
                try:
                    # Convert to pandas Timestamp if needed
                    if isinstance(time_vals, np.ndarray):
                        timestamps = pd.to_datetime(time_vals)
                    elif isinstance(time_vals, pd.Series):
                        timestamps = time_vals
                    elif isinstance(time_vals, list):
                        timestamps = pd.to_datetime(time_vals)
                    else:
                        timestamps = pd.to_datetime(time_vals)
                    
                    if 'start_ts' not in date_range:
                        date_range['start_ts'] = pd.Timestamp(timestamps.min()).isoformat()
                    if 'end_ts' not in date_range:
                        date_range['end_ts'] = pd.Timestamp(timestamps.max()).isoformat()
                except Exception as e:
                    logger.debug(f"Failed to extract date range from time_vals: {e}")
    
    # If still missing, try to extract from mtf_data
    if ('start_ts' not in date_range or 'end_ts' not in date_range) and mtf_data is not None:
        try:
            all_timestamps = []
            for symbol_df in mtf_data.values():
                if isinstance(symbol_df, pd.DataFrame):
                    # Try common timestamp column names
                    for col in ['timestamp', 'time', 'date', 'datetime', 'ts']:
                        if col in symbol_df.columns:
                            all_timestamps.extend(symbol_df[col].dropna().tolist())
                            break
                    # Also try index if it's a DatetimeIndex
                    if isinstance(symbol_df.index, pd.DatetimeIndex):
                        all_timestamps.extend(symbol_df.index.dropna().tolist())
            
            if all_timestamps:
                # Convert to pandas Timestamp if needed
                if isinstance(all_timestamps[0], str):
                    all_timestamps = pd.to_datetime(all_timestamps)
                elif not isinstance(all_timestamps[0], pd.Timestamp):
                    all_timestamps = pd.to_datetime(all_timestamps)
                
                if 'start_ts' not in date_range:
                    date_range['start_ts'] = pd.Timestamp(min(all_timestamps)).isoformat()
                if 'end_ts' not in date_range:
                    date_range['end_ts'] = pd.Timestamp(max(all_timestamps)).isoformat()
        except Exception as e:
            logger.debug(f"Failed to extract date range from mtf_data: {e}")
    
    metadata['date_range'] = date_range if date_range else {}
    
    # 4. Build cs_config
    cs_config = {}
    if min_cs is not None:
        cs_config['min_cs'] = int(min_cs)
    if max_cs_samples is not None:
        cs_config['max_cs_samples'] = int(max_cs_samples)
    if leakage_filter_version is not None:
        cs_config['leakage_filter_version'] = str(leakage_filter_version)
    if universe_id is not None:
        cs_config['universe_id'] = str(universe_id)
    
    metadata['cs_config'] = cs_config
    
    # 5. Compute data fingerprint (hash of timestamps + symbols + label vector)
    if compute_data_fingerprint:
        try:
            fingerprint_components = []
            
            # Add timestamps (sorted, unique)
            if time_vals is not None:
                try:
                    if isinstance(time_vals, (np.ndarray, pd.Series, list)):
                        time_vals_clean = pd.to_datetime(time_vals).dropna().sort_values()
                        # Use unique timestamps only (panel data has duplicates)
                        unique_times = time_vals_clean.unique()
                        fingerprint_components.append(f"times:{','.join(str(t) for t in unique_times[:1000])}")  # Limit to first 1000 for performance
                except Exception as e:
                    logger.debug(f"Could not add timestamps to fingerprint: {e}")
            
            # Add symbols (sorted, unique)
            if symbols_list:
                fingerprint_components.append(f"symbols:{','.join(symbols_list)}")
            
            # Add label vector hash (if available)
            if y is not None:
                try:
                    if isinstance(y, (np.ndarray, pd.Series)):
                        # Hash the label vector (use first 10000 samples for performance)
                        y_sample = y[:10000] if len(y) > 10000 else y
                        y_hash = hashlib.sha256(y_sample.tobytes() if isinstance(y_sample, np.ndarray) else y_sample.values.tobytes()).hexdigest()[:16]
                        fingerprint_components.append(f"y_hash:{y_hash}")
                except Exception as e:
                    logger.debug(f"Could not add label hash to fingerprint: {e}")
            
            if fingerprint_components:
                fingerprint_str = "|".join(fingerprint_components)
                metadata['data_fingerprint'] = hashlib.sha256(fingerprint_str.encode()).hexdigest()[:16]
        except Exception as e:
            logger.debug(f"Failed to compute data fingerprint: {e}")
    
    # 6. Compute per-symbol statistics (n, start, end for each symbol)
    if compute_per_symbol_stats and symbols is not None and time_vals is not None:
        try:
            per_symbol_stats = {}
            
            # Convert to arrays for easier processing
            if isinstance(symbols, (list, tuple)):
                symbols_array = np.array(symbols)
            elif isinstance(symbols, pd.Series):
                symbols_array = symbols.values
            else:
                symbols_array = symbols
            
            if isinstance(time_vals, (list, tuple)):
                time_vals_array = pd.to_datetime(time_vals).values
            elif isinstance(time_vals, pd.Series):
                time_vals_array = time_vals.values
            else:
                time_vals_array = pd.to_datetime(time_vals).values
            
            # Ensure same length
            if len(symbols_array) == len(time_vals_array):
                # Group by symbol
                unique_symbols = np.unique(symbols_array)
                for symbol in unique_symbols:
                    symbol_mask = symbols_array == symbol
                    symbol_times = time_vals_array[symbol_mask]
                    
                    if len(symbol_times) > 0:
                        per_symbol_stats[str(symbol)] = {
                            'n': int(np.sum(symbol_mask)),
                            'start': pd.Timestamp(symbol_times.min()).isoformat(),
                            'end': pd.Timestamp(symbol_times.max()).isoformat()
                        }
            
            if per_symbol_stats:
                metadata['per_symbol_stats'] = per_symbol_stats
        except Exception as e:
            logger.debug(f"Failed to compute per-symbol stats: {e}")
    
    return metadata


def format_for_reproducibility_tracker(
    cohort_metadata: Dict[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Format cohort metadata for use with ReproducibilityTracker.log_comparison().
    
    Splits the metadata into:
    - metrics: Contains N_effective_cs (for metrics dict)
    - additional_data: Contains n_symbols, date_range, cs_config (for additional_data dict)
    
    Args:
        cohort_metadata: Output from extract_cohort_metadata()
    
    Returns:
        Tuple of (metrics_dict, additional_data_dict)
    """
    metrics = {}
    additional_data = {}
    
    # N_effective_cs goes in metrics
    if 'N_effective_cs' in cohort_metadata:
        metrics['N_effective_cs'] = cohort_metadata['N_effective_cs']
    
    # Everything else goes in additional_data
    if 'n_symbols' in cohort_metadata:
        additional_data['n_symbols'] = cohort_metadata['n_symbols']
    if 'symbols' in cohort_metadata and cohort_metadata['symbols']:
        additional_data['symbols'] = cohort_metadata['symbols']  # Pass symbols list
    if 'date_range' in cohort_metadata and cohort_metadata['date_range']:
        additional_data['date_range'] = cohort_metadata['date_range']
    if 'cs_config' in cohort_metadata and cohort_metadata['cs_config']:
        additional_data['cs_config'] = cohort_metadata['cs_config']
    
    return metrics, additional_data
