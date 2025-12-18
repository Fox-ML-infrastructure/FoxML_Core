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
Reporting Functions for Leakage Detection

Functions for saving feature importances and logging suspicious features.
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

# Add project root for _REPO_ROOT
import sys
_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def save_feature_importances(
    target_column: str,
    symbol: str,
    feature_importances: Dict[str, Dict[str, float]],
    output_dir: Path = None,
    view: str = "CROSS_SECTIONAL"
) -> None:
    """
    Save detailed per-model, per-feature importance scores to CSV files.
    
    Creates structure:
    {output_dir}/feature_importances/
      {target_name}/
        {symbol}/
          lightgbm_importances.csv
          xgboost_importances.csv
          random_forest_importances.csv
          ...
    
    Args:
        target_column: Name of the target being evaluated
        symbol: Symbol being evaluated
        feature_importances: Dict of {model_name: {feature: importance}}
        output_dir: Base output directory (defaults to results/)
    """
    if output_dir is None:
        output_dir = _REPO_ROOT / "results"
    
    # Create directory structure in REPRODUCIBILITY/TARGET_RANKING/{view}/{target}/{symbol}/feature_importances/
    target_name_clean = target_column.replace('/', '_').replace('\\', '_')
    if output_dir.name == "target_rankings":
        repro_base = output_dir.parent / "REPRODUCIBILITY" / "TARGET_RANKING"
    else:
        repro_base = output_dir / "REPRODUCIBILITY" / "TARGET_RANKING"
    
    if view == "SYMBOL_SPECIFIC" and symbol:
        importances_dir = repro_base / view / target_name_clean / f"symbol={symbol}" / "feature_importances"
    else:
        importances_dir = repro_base / view / target_name_clean / "feature_importances"
    importances_dir.mkdir(parents=True, exist_ok=True)
    
    # Save per-model CSV files
    for model_name in sorted(feature_importances.keys()):
        importances = feature_importances[model_name]
        if not importances:
            continue
        
        # Create DataFrame sorted by importance
        df = pd.DataFrame([
            {'feature': feat, 'importance': imp}
            for feat, imp in sorted(importances.items())
        ])
        df = df.sort_values('importance', ascending=False)
        
        # Normalize to percentages
        total = df['importance'].sum()
        if total > 0:
            df['importance_pct'] = (df['importance'] / total * 100).round(2)
            df['cumulative_pct'] = df['importance_pct'].cumsum().round(2)
        else:
            df['importance_pct'] = 0.0
            df['cumulative_pct'] = 0.0
        
        # Reorder columns
        df = df[['feature', 'importance', 'importance_pct', 'cumulative_pct']]
        
        # Save to CSV
        csv_file = importances_dir / f"{model_name}_importances.csv"
        df.to_csv(csv_file, index=False)
    
    logger.info(f"  💾 Saved feature importances to: {importances_dir}")


def log_suspicious_features(
    target_column: str,
    symbol: str,
    suspicious_features: Dict[str, List[Tuple[str, float]]]
) -> None:
    """
    Log suspicious features to a file for later analysis.
    
    Args:
        target_column: Name of the target being evaluated
        symbol: Symbol being evaluated
        suspicious_features: Dict of {model_name: [(feature, importance), ...]}
    """
    leak_report_file = _REPO_ROOT / "results" / "leak_detection_report.txt"
    leak_report_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(leak_report_file, 'a') as f:
        f.write(f"\n{'='*80}\n")
        f.write(f"Target: {target_column} | Symbol: {symbol}\n")
        f.write(f"{'='*80}\n")
        
        for model_name, features in suspicious_features.items():
            if features:
                f.write(f"\n{model_name.upper()} - Suspicious Features:\n")
                f.write(f"{'-'*80}\n")
                for feat, imp in sorted(features, key=lambda x: x[1], reverse=True):
                    f.write(f"  {feat:50s} | Importance: {imp:.1%}\n")
                f.write("\n")
    
    logger.info(f"  Leak detection report saved to: {leak_report_file}")

