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
Performance Audit Utilities

Tracks call counts and timing for heavy functions to identify "accidental multiplicative work"
- Functions called multiple times with same input fingerprint
- Expensive operations in nested loops
- Missing cache opportunities
"""

import hashlib
import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from TRAINING.common.utils.config_hashing import compute_config_hash

logger = logging.getLogger(__name__)


@dataclass
class CallRecord:
    """Record of a single function call"""
    func_name: str
    duration: float
    rows: Optional[int] = None
    cols: Optional[int] = None
    stage: str = "unknown"
    cache_hit: bool = False
    input_fingerprint: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


class PerformanceAuditor:
    """
    Track call counts and timing for heavy functions.
    
    Identifies "accidental multiplicative work" by detecting:
    - Functions called multiple times with same input fingerprint
    - Expensive operations in nested loops
    - Missing cache opportunities
    """
    
    def __init__(self, enabled: bool = True):
        """
        Initialize performance auditor.
        
        Args:
            enabled: If False, tracking is disabled (no-op)
        """
        self.enabled = enabled
        self.calls: List[CallRecord] = []
        self._fingerprint_cache: Dict[str, str] = {}  # Cache for input fingerprints
        
    def track_call(
        self,
        func_name: str,
        duration: float,
        rows: Optional[int] = None,
        cols: Optional[int] = None,
        stage: str = "unknown",
        cache_hit: bool = False,
        input_fingerprint: Optional[str] = None,
        **kwargs
    ) -> None:
        """
        Track a function call.
        
        Args:
            func_name: Name of the function
            duration: Duration in seconds
            rows: Number of rows in input data (if applicable)
            cols: Number of columns/features (if applicable)
            stage: Stage name (e.g., "feature_selection", "target_ranking", "training")
            cache_hit: Whether this was a cache hit
            input_fingerprint: Optional pre-computed fingerprint
            **kwargs: Additional metadata to include in fingerprint
        """
        if not self.enabled:
            return
        
        # Compute fingerprint if not provided
        if input_fingerprint is None and kwargs:
            input_fingerprint = self._compute_fingerprint(func_name, **kwargs)
        
        record = CallRecord(
            func_name=func_name,
            duration=duration,
            rows=rows,
            cols=cols,
            stage=stage,
            cache_hit=cache_hit,
            input_fingerprint=input_fingerprint
        )
        self.calls.append(record)
    
    def _compute_fingerprint(self, func_name: str, **kwargs) -> str:
        """
        Compute fingerprint from function name and kwargs.
        
        Args:
            func_name: Function name
            **kwargs: Input parameters
        
        Returns:
            Hexadecimal hash string
        """
        # Create cache key for fingerprint computation
        cache_key = f"{func_name}:{json.dumps(kwargs, sort_keys=True)}"
        
        if cache_key in self._fingerprint_cache:
            return self._fingerprint_cache[cache_key]
        
        # Compute hash
        fingerprint_data = {
            'func': func_name,
            **kwargs
        }
        fingerprint = compute_config_hash(fingerprint_data)
        self._fingerprint_cache[cache_key] = fingerprint
        return fingerprint
    
    def report_multipliers(self, min_calls: int = 2) -> Dict[str, List[Dict[str, Any]]]:
        """
        Find functions called multiple times with same fingerprint.
        
        Args:
            min_calls: Minimum number of calls to report (default: 2)
        
        Returns:
            Dictionary mapping function names to lists of multiplier findings
        """
        if not self.enabled or not self.calls:
            return {}
        
        # Group calls by (func_name, fingerprint)
        grouped = defaultdict(list)
        for call in self.calls:
            if call.input_fingerprint:
                key = (call.func_name, call.input_fingerprint)
                grouped[key].append(call)
        
        # Find multipliers
        multipliers = {}
        for (func_name, fingerprint), calls in grouped.items():
            if len(calls) >= min_calls:
                total_duration = sum(c.duration for c in calls)
                avg_duration = total_duration / len(calls)
                
                multipliers[func_name] = multipliers.get(func_name, [])
                multipliers[func_name].append({
                    'fingerprint': fingerprint[:16],  # Short hash for readability
                    'call_count': len(calls),
                    'total_duration': total_duration,
                    'avg_duration': avg_duration,
                    'wasted_duration': total_duration - avg_duration,  # Time that could be saved with caching
                    'stage': calls[0].stage,
                    'rows': calls[0].rows,
                    'cols': calls[0].cols,
                    'cache_hits': sum(1 for c in calls if c.cache_hit),
                    'cache_misses': sum(1 for c in calls if not c.cache_hit)
                })
        
        return multipliers
    
    def report_summary(self) -> Dict[str, Any]:
        """
        Generate summary report of all tracked calls.
        
        Returns:
            Dictionary with summary statistics
        """
        if not self.enabled or not self.calls:
            return {}
        
        # Group by function name
        by_func = defaultdict(list)
        for call in self.calls:
            by_func[call.func_name].append(call)
        
        summary = {
            'total_calls': len(self.calls),
            'unique_functions': len(by_func),
            'functions': {}
        }
        
        for func_name, calls in by_func.items():
            total_duration = sum(c.duration for c in calls)
            avg_duration = total_duration / len(calls)
            max_duration = max(c.duration for c in calls)
            min_duration = min(c.duration for c in calls)
            
            # Count cache hits
            cache_hits = sum(1 for c in calls if c.cache_hit)
            cache_misses = len(calls) - cache_hits
            
            summary['functions'][func_name] = {
                'call_count': len(calls),
                'total_duration': total_duration,
                'avg_duration': avg_duration,
                'min_duration': min_duration,
                'max_duration': max_duration,
                'cache_hit_rate': cache_hits / len(calls) if calls else 0.0,
                'cache_hits': cache_hits,
                'cache_misses': cache_misses
            }
        
        return summary
    
    def report_nested_loops(self) -> List[Dict[str, Any]]:
        """
        Detect potential nested loop issues (same function called rapidly in sequence).
        
        Returns:
            List of potential nested loop findings
        """
        if not self.enabled or len(self.calls) < 3:
            return []
        
        findings = []
        
        # Group consecutive calls of same function
        i = 0
        while i < len(self.calls) - 2:
            func_name = self.calls[i].func_name
            consecutive = [self.calls[i]]
            
            # Collect consecutive calls to same function
            j = i + 1
            while j < len(self.calls) and self.calls[j].func_name == func_name:
                consecutive.append(self.calls[j])
                j += 1
            
            # If we have 3+ consecutive calls, flag as potential nested loop
            if len(consecutive) >= 3:
                time_span = consecutive[-1].timestamp - consecutive[0].timestamp
                total_duration = sum(c.duration for c in consecutive)
                
                findings.append({
                    'func_name': func_name,
                    'consecutive_calls': len(consecutive),
                    'time_span': time_span,
                    'total_duration': total_duration,
                    'stage': consecutive[0].stage,
                    'avg_duration': total_duration / len(consecutive),
                    'potential_multiplier': len(consecutive)
                })
            
            i = j
        
        return findings
    
    def save_report(self, output_path: Path) -> None:
        """
        Save audit report to JSON file.
        
        Args:
            output_path: Path to save report
        """
        if not self.enabled:
            return
        
        report = {
            'summary': self.report_summary(),
            'multipliers': self.report_multipliers(),
            'nested_loops': self.report_nested_loops(),
            'all_calls': [
                {
                    'func_name': c.func_name,
                    'duration': c.duration,
                    'rows': c.rows,
                    'cols': c.cols,
                    'stage': c.stage,
                    'cache_hit': c.cache_hit,
                    'fingerprint': c.input_fingerprint[:16] if c.input_fingerprint else None,
                    'timestamp': c.timestamp
                }
                for c in self.calls
            ]
        }
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"💾 Saved performance audit report to {output_path}")
    
    def reset(self) -> None:
        """Reset all tracking data."""
        self.calls.clear()
        self._fingerprint_cache.clear()


# Global auditor instance (singleton pattern)
_global_auditor: Optional[PerformanceAuditor] = None


def get_auditor(enabled: bool = True) -> PerformanceAuditor:
    """
    Get global performance auditor instance.
    
    Args:
        enabled: If True, enable tracking (default: True)
    
    Returns:
        PerformanceAuditor instance
    """
    global _global_auditor
    if _global_auditor is None:
        _global_auditor = PerformanceAuditor(enabled=enabled)
    return _global_auditor


def track_performance(
    func_name: Optional[str] = None,
    stage: str = "unknown",
    compute_fingerprint: bool = True
):
    """
    Decorator to track function performance.
    
    Args:
        func_name: Optional function name (defaults to decorated function name)
        stage: Stage name (e.g., "feature_selection", "target_ranking")
        compute_fingerprint: If True, compute input fingerprint from args/kwargs
    
    Usage:
        @track_performance(stage="feature_selection")
        def expensive_function(X, y, **kwargs):
            ...
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            auditor = get_auditor()
            if not auditor.enabled:
                return func(*args, **kwargs)
            
            # Compute fingerprint if requested
            fingerprint = None
            if compute_fingerprint:
                try:
                    # Try to extract data dimensions for fingerprint
                    fingerprint_kwargs = {}
                    if args and hasattr(args[0], 'shape'):  # First arg might be data array
                        fingerprint_kwargs['data_shape'] = args[0].shape
                    fingerprint_kwargs.update(kwargs)
                    fingerprint = auditor._compute_fingerprint(
                        func_name or func.__name__,
                        **fingerprint_kwargs
                    )
                except Exception:
                    pass  # Skip fingerprint if computation fails
            
            # Track call
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                # Try to extract dimensions from result or args
                rows, cols = None, None
                if args and hasattr(args[0], 'shape'):
                    rows, cols = args[0].shape[0], args[0].shape[1] if len(args[0].shape) > 1 else None
                
                auditor.track_call(
                    func_name=func_name or func.__name__,
                    duration=duration,
                    rows=rows,
                    cols=cols,
                    stage=stage,
                    cache_hit=False,  # Decorator can't know about cache
                    input_fingerprint=fingerprint
                )
                
                return result
            except Exception as e:
                duration = time.time() - start_time
                auditor.track_call(
                    func_name=func_name or func.__name__,
                    duration=duration,
                    stage=stage,
                    cache_hit=False,
                    input_fingerprint=fingerprint
                )
                raise
        
        return wrapper
    return decorator

