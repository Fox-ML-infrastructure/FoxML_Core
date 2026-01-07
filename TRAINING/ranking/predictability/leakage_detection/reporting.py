# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial

"""
Reporting Functions for Leakage Detection

Functions for saving feature importances and logging suspicious features.
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional

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
    view: str = "CROSS_SECTIONAL",
    universe_sig: Optional[str] = None  # PATCH 4: Required for proper scoping
) -> None:
    """
    Save detailed per-model, per-feature importance scores to CSV files.
    
    Creates structure (with universe_sig):
    targets/{target}/reproducibility/{view}/universe={sig}/(symbol={sym})/feature_importances/
      lightgbm_importances.csv
      xgboost_importances.csv
      ...
    
    Args:
        target_column: Name of the target being evaluated
        symbol: Symbol being evaluated
        feature_importances: Dict of {model_name: {feature: importance}}
        output_dir: Base output directory (defaults to results/)
        view: CROSS_SECTIONAL or SYMBOL_SPECIFIC
        universe_sig: Universe signature from SST (required for proper scoping)
    """
    # PATCH 4: Require universe_sig for proper scoping
    if not universe_sig:
        logger.error(
            f"SCOPE BUG: universe_sig not provided for {target_column} feature importances. "
            f"Cannot create view-scoped paths. Feature importances will not be written."
        )
        return  # Don't write to unscoped location
    
    if output_dir is None:
        output_dir = _REPO_ROOT / "results"
    
    # Find base run directory for target-first structure
    base_output_dir = output_dir
    for _ in range(10):
        if base_output_dir.name == "RESULTS" or (base_output_dir / "targets").exists():
            break
        if not base_output_dir.parent.exists():
            break
        base_output_dir = base_output_dir.parent
    
    target_clean = target_column.replace('/', '_').replace('\\', '_')
    
    # PATCH 4: Use OutputLayout for properly scoped paths
    try:
        from TRAINING.orchestration.utils.output_layout import OutputLayout
        from TRAINING.orchestration.utils.target_first_paths import ensure_target_structure
        
        ensure_target_structure(base_output_dir, target_clean)
        
        # Only pass symbol if view is SYMBOL_SPECIFIC
        symbol_for_layout = symbol if view == "SYMBOL_SPECIFIC" else None
        
        layout = OutputLayout(
            output_root=base_output_dir,
            target=target_clean,
            view=view,
            universe_sig=universe_sig,
            symbol=symbol_for_layout,
            stage="TARGET_RANKING",  # Explicit stage for proper path scoping
        )
        target_importances_dir = layout.feature_importance_dir()
        target_importances_dir.mkdir(parents=True, exist_ok=True)
        
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
            
            # Save to properly scoped location
            target_csv_file = target_importances_dir / f"{model_name}_importances.csv"
            df.to_csv(target_csv_file, index=False)
        
        logger.info(f"  💾 Saved feature importances to: {target_importances_dir}")
    except Exception as e:
        logger.warning(f"Failed to save feature importances to target-first structure: {e}")


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

