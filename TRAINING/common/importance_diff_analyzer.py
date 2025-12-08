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
Importance Diff Analyzer

High-level analyzer that trains models with full vs safe feature sets
and compares their importances to detect potential leakage.

This is a convenience wrapper around ImportanceDiffDetector that handles
the full workflow: data loading, model training, importance extraction, and reporting.
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import json

from .importance_diff_detector import ImportanceDiffDetector, SuspiciousFeature

logger = logging.getLogger(__name__)


class ImportanceDiffAnalyzer:
    """
    Analyzes feature importance differences between full and safe feature sets.
    
    Workflow:
    1. Load data for a target
    2. Train model with all features (full set)
    3. Train model with only safe features (registry-validated)
    4. Compare importances to detect suspicious features
    5. Generate report
    """
    
    def __init__(
        self,
        diff_threshold: float = 0.1,
        relative_diff_threshold: float = 0.5,
        min_importance_full: float = 0.01
    ):
        """
        Initialize analyzer.
        
        Args:
            diff_threshold: Absolute difference threshold
            relative_diff_threshold: Relative difference threshold
            min_importance_full: Minimum importance in full model to consider
        """
        self.detector = ImportanceDiffDetector(
            diff_threshold=diff_threshold,
            relative_diff_threshold=relative_diff_threshold,
            min_importance_full=min_importance_full
        )
    
    def analyze_target(
        self,
        target_column: str,
        symbols: List[str],
        data_dir: Path,
        model_family: str = 'lightgbm',
        model_config: Dict[str, Any] = None,
        max_samples_per_symbol: int = 50000,
        output_dir: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Analyze a target by comparing full vs safe feature sets.
        
        Args:
            target_column: Target column name
            symbols: List of symbols to analyze
            data_dir: Directory containing symbol data
            model_family: Model family to use for comparison
            model_config: Optional model config
            max_samples_per_symbol: Maximum samples per symbol
            output_dir: Optional output directory for results
        
        Returns:
            Dictionary with analysis results
        """
        logger.info(f"Analyzing importance diff for target: {target_column}")
        
        # This is a placeholder for the full implementation
        # Full implementation would:
        # 1. Load data for all symbols
        # 2. Prepare features (all features)
        # 3. Train model_full with all features
        # 4. Filter features using registry
        # 5. Train model_safe with safe features only
        # 6. Compare importances
        # 7. Generate report
        
        logger.info("Importance diff analysis requires training two model sets")
        logger.info("This is a placeholder - full implementation would train models here")
        
        return {
            'target_column': target_column,
            'status': 'placeholder',
            'message': 'Full implementation requires model training (see design doc)'
        }
    
    def analyze_from_results(
        self,
        results_full: Dict[str, Any],
        results_safe: Dict[str, Any],
        feature_names_full: List[str],
        feature_names_safe: List[str],
        model_full: Any = None,
        model_safe: Any = None,
        importance_method: str = 'auto'
    ) -> Dict[str, Any]:
        """
        Analyze importance diff from pre-trained models.
        
        Args:
            results_full: Results from full feature set training
            results_safe: Results from safe feature set training
            feature_names_full: Feature names for full model
            feature_names_safe: Feature names for safe model
            model_full: Optional pre-trained full model
            model_safe: Optional pre-trained safe model
            importance_method: Method to extract importance
        
        Returns:
            Dictionary with detection results
        """
        if model_full is None or model_safe is None:
            logger.warning("Models not provided, cannot analyze importance diff")
            return {
                'status': 'error',
                'message': 'Models required for importance diff analysis'
            }
        
        # Use detector to compare importances
        report = self.detector.detect_and_report(
            model_full=model_full,
            model_safe=model_safe,
            feature_names_full=feature_names_full,
            feature_names_safe=feature_names_safe,
            importance_method=importance_method
        )
        
        return report

