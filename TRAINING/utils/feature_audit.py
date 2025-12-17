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
Feature Audit System - Track Feature Drop Reasons

This module provides instrumentation to track why features are dropped
at each stage of the data preparation pipeline, producing a detailed CSV
report for debugging feature collapse issues.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class FeatureDropRecord:
    """Record of why a feature was dropped at a specific stage."""
    feature_name: str
    stage: str  # e.g., "registry_filter", "polars_select", "pandas_coercion", "nan_drop", "non_numeric_drop"
    reason: str  # e.g., "missing_in_df", "excluded_by_registry", "all_null", "non_numeric", "failed_coercion"
    dtype_polars: Optional[str] = None
    dtype_pandas: Optional[str] = None
    null_fraction: Optional[float] = None
    n_unique: Optional[int] = None
    sample_value: Optional[Any] = None
    metadata: Optional[Dict[str, Any]] = None


class FeatureAuditor:
    """
    Tracks feature drops through the data preparation pipeline.
    
    Usage:
        auditor = FeatureAuditor(target="fwd_ret_5d")
        auditor.record_requested(requested_features)
        auditor.record_registry_allowed(allowed_features)
        auditor.record_present_in_polars(polars_df)
        auditor.record_kept_for_training(pandas_df)
        auditor.record_used_in_X(X, feature_names)
        auditor.write_report(output_dir)
    """
    
    def __init__(self, target: str):
        self.target = target
        self.requested_features: Set[str] = set()
        self.registry_allowed: Set[str] = set()
        self.present_in_polars: Set[str] = set()
        self.kept_for_training: Set[str] = set()
        self.used_in_X: Set[str] = set()
        self.drop_records: List[FeatureDropRecord] = []
        
    def record_requested(self, features: List[str]):
        """Record features requested from config/initial discovery."""
        self.requested_features = set(features)
        logger.debug(f"[FeatureAudit] {self.target}: {len(self.requested_features)} features requested")
    
    def record_registry_allowed(self, features: List[str], all_features: List[str] = None):
        """Record features allowed by registry filtering."""
        self.registry_allowed = set(features)
        if all_features:
            excluded = set(all_features) - self.registry_allowed
            for feat in excluded:
                self.drop_records.append(FeatureDropRecord(
                    feature_name=feat,
                    stage="registry_filter",
                    reason="excluded_by_registry"
                ))
        logger.debug(f"[FeatureAudit] {self.target}: {len(self.registry_allowed)} features allowed by registry")
    
    def record_present_in_polars(self, polars_df, feature_names: List[str] = None):
        """Record features actually present in Polars DataFrame."""
        if hasattr(polars_df, 'columns'):
            self.present_in_polars = set(polars_df.columns)
        else:
            self.present_in_polars = set()
        
        # Track missing features
        if feature_names:
            missing = set(feature_names) - self.present_in_polars
            for feat in missing:
                self.drop_records.append(FeatureDropRecord(
                    feature_name=feat,
                    stage="polars_select",
                    reason="missing_in_df"
                ))
        logger.debug(f"[FeatureAudit] {self.target}: {len(self.present_in_polars)} features present in Polars")
    
    def record_kept_for_training(self, pandas_df: pd.DataFrame, feature_names: List[str] = None):
        """Record features kept after pandas conversion and initial filtering."""
        if feature_names:
            self.kept_for_training = set(feature_names) & set(pandas_df.columns)
        else:
            self.kept_for_training = set(pandas_df.columns)
        
        # Track dtype and null stats for kept features
        for feat in self.kept_for_training:
            if feat in pandas_df.columns:
                col = pandas_df[feat]
                record = FeatureDropRecord(
                    feature_name=feat,
                    stage="pandas_coercion",
                    reason="kept",
                    dtype_pandas=str(col.dtype),
                    null_fraction=col.isna().mean(),
                    n_unique=col.nunique(),
                    sample_value=col.iloc[0] if len(col) > 0 else None
                )
                # Only add if not already recorded
                if not any(r.feature_name == feat and r.stage == "pandas_coercion" for r in self.drop_records):
                    self.drop_records.append(record)
        
        logger.debug(f"[FeatureAudit] {self.target}: {len(self.kept_for_training)} features kept for training")
    
    def record_dropped_all_nan(self, dropped_features: List[str], pandas_df: pd.DataFrame):
        """Record features dropped because they're all NaN."""
        for feat in dropped_features:
            if feat in pandas_df.columns:
                col = pandas_df[feat]
                self.drop_records.append(FeatureDropRecord(
                    feature_name=feat,
                    stage="nan_drop",
                    reason="all_null",
                    dtype_pandas=str(col.dtype),
                    null_fraction=1.0,
                    n_unique=0
                ))
    
    def record_dropped_non_numeric(self, dropped_features: List[str], pandas_df: pd.DataFrame):
        """Record features dropped because they're non-numeric."""
        for feat in dropped_features:
            if feat in pandas_df.columns:
                col = pandas_df[feat]
                self.drop_records.append(FeatureDropRecord(
                    feature_name=feat,
                    stage="non_numeric_drop",
                    reason="non_numeric",
                    dtype_pandas=str(col.dtype),
                    null_fraction=col.isna().mean() if hasattr(col, 'isna') else None,
                    n_unique=col.nunique() if hasattr(col, 'nunique') else None
                ))
    
    def record_used_in_X(self, feature_names: List[str], X: np.ndarray):
        """Record features actually used in final feature matrix X."""
        self.used_in_X = set(feature_names)
        logger.debug(f"[FeatureAudit] {self.target}: {len(self.used_in_X)} features used in X")
    
    def write_report(self, output_dir: Path) -> Path:
        """Write feature audit report to CSV."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create summary DataFrame
        summary_rows = []
        
        # Count drops by stage and reason
        stage_reasons: Dict[str, Dict[str, int]] = {}
        for record in self.drop_records:
            if record.stage not in stage_reasons:
                stage_reasons[record.stage] = {}
            if record.reason not in stage_reasons[record.stage]:
                stage_reasons[record.stage][record.reason] = 0
            stage_reasons[record.stage][record.reason] += 1
        
        # Build summary
        summary_rows.append({
            "metric": "requested",
            "count": len(self.requested_features),
            "stage": "initial",
            "reason": "config_or_discovery"
        })
        summary_rows.append({
            "metric": "registry_allowed",
            "count": len(self.registry_allowed),
            "stage": "registry_filter",
            "reason": "passed_registry_validation"
        })
        summary_rows.append({
            "metric": "present_in_polars",
            "count": len(self.present_in_polars),
            "stage": "polars_select",
            "reason": "exists_in_dataframe"
        })
        summary_rows.append({
            "metric": "kept_for_training",
            "count": len(self.kept_for_training),
            "stage": "pandas_coercion",
            "reason": "passed_initial_filtering"
        })
        summary_rows.append({
            "metric": "used_in_X",
            "count": len(self.used_in_X),
            "stage": "final",
            "reason": "in_feature_matrix"
        })
        
        # Add drop counts
        for stage, reasons in stage_reasons.items():
            for reason, count in reasons.items():
                summary_rows.append({
                    "metric": f"dropped_{stage}",
                    "count": count,
                    "stage": stage,
                    "reason": reason
                })
        
        summary_df = pd.DataFrame(summary_rows)
        
        # Create detailed drop records DataFrame
        if self.drop_records:
            drop_df = pd.DataFrame([asdict(r) for r in self.drop_records])
        else:
            drop_df = pd.DataFrame(columns=["feature_name", "stage", "reason", "dtype_polars", "dtype_pandas", 
                                           "null_fraction", "n_unique", "sample_value", "metadata"])
        
        # Write CSVs
        safe_target = self.target.replace("/", "_").replace("\\", "_")
        summary_path = output_dir / f"feature_audit_{safe_target}_summary.csv"
        drop_path = output_dir / f"feature_audit_{safe_target}_drops.csv"
        
        summary_df.to_csv(summary_path, index=False)
        drop_df.to_csv(drop_path, index=False)
        
        logger.info(f"[FeatureAudit] {self.target}: Wrote audit reports to {output_dir}")
        logger.info(f"[FeatureAudit] {self.target}: Summary: requested={len(self.requested_features)}, "
                   f"allowed={len(self.registry_allowed)}, present={len(self.present_in_polars)}, "
                   f"kept={len(self.kept_for_training)}, used={len(self.used_in_X)}")
        
        return summary_path

