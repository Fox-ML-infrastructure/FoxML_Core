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
Metrics Aggregator

Collects metrics from feature selection, stability analysis, and leakage detection
outputs and aggregates them into a routing_candidates DataFrame for the training router.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)


class MetricsAggregator:
    """
    Aggregates metrics from various pipeline stages into routing candidates.
    """
    
    def __init__(self, output_dir: Path):
        """
        Initialize aggregator.
        
        Args:
            output_dir: Base output directory (e.g., feature_selections/)
        """
        self.output_dir = Path(output_dir)
    
    def aggregate_routing_candidates(
        self,
        targets: List[str],
        symbols: List[str],
        git_commit: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Aggregate routing candidates from all available metrics.
        
        Args:
            targets: List of target names
            symbols: List of symbol names
            git_commit: Git commit hash
        
        Returns:
            DataFrame with routing candidates (one row per target or (target, symbol))
        """
        rows = []
        
        for target in targets:
            # Cross-sectional metrics
            cs_metrics = self._load_cross_sectional_metrics(target)
            if cs_metrics:
                rows.append(cs_metrics)
            
            # Symbol-specific metrics
            for symbol in symbols:
                sym_metrics = self._load_symbol_metrics(target, symbol)
                if sym_metrics:
                    rows.append(sym_metrics)
        
        if not rows:
            logger.warning("No routing candidates found")
            return pd.DataFrame()
        
        df = pd.DataFrame(rows)
        
        # Add metadata
        df["timestamp"] = datetime.utcnow().isoformat()
        df["git_commit"] = git_commit or "unknown"
        
        return df
    
    def _load_cross_sectional_metrics(self, target: str) -> Optional[Dict[str, Any]]:
        """
        Load cross-sectional metrics for a target.
        
        Args:
            target: Target name
        
        Returns:
            Dict with metrics or None if not found
        """
        target_dir = self.output_dir / target
        
        # Try to find CS metrics from various sources
        # 1. Feature selection outputs
        fs_dir = target_dir / "feature_selections"
        if not fs_dir.exists():
            fs_dir = target_dir
        
        # Look for model_metadata.json or target_confidence.json
        metadata_path = fs_dir / "model_metadata.json"
        confidence_path = fs_dir / "target_confidence.json"
        
        score = None
        sample_size = None
        failed_families = []
        leakage_status = "UNKNOWN"
        
        # Load from model_metadata.json (aggregated across symbols)
        if metadata_path.exists():
            try:
                with open(metadata_path) as f:
                    metadata = json.load(f)
                
                # Aggregate scores across all symbols
                scores = []
                for key, data in metadata.items():
                    if isinstance(data, dict) and "score" in data:
                        score_val = data["score"]
                        if score_val is not None and not np.isnan(score_val):
                            scores.append(score_val)
                
                if scores:
                    score = np.mean(scores)
            except Exception as e:
                logger.debug(f"Failed to load CS metrics from {metadata_path}: {e}")
        
        # Load from target_confidence.json
        if confidence_path.exists():
            try:
                with open(confidence_path) as f:
                    conf = json.load(f)
                
                # Extract score
                if score is None:
                    score = conf.get("mean_score", None)
                
                # Extract sample size (if available)
                sample_size = conf.get("sample_size", None)
            except Exception as e:
                logger.debug(f"Failed to load CS metrics from {confidence_path}: {e}")
        
        # Load stability metrics
        stability_metrics = self._load_stability_metrics(target, universe_id="ALL")
        
        # Classify stability
        stability = self._classify_stability_from_metrics(stability_metrics)
        
        # Load leakage status
        leakage_status = self._load_leakage_status(target, symbol=None)
        
        if score is None:
            return None
        
        return {
            "target": target,
            "symbol": None,  # CS has no symbol
            "mode": "CROSS_SECTIONAL",
            "score": float(score),
            "score_ci_low": None,  # Would need to compute from CV
            "score_ci_high": None,
            "stability": stability,
            "sample_size": int(sample_size) if sample_size else 0,
            "leakage_status": leakage_status,
            "feature_set_id": None,  # Would need to hash feature set
            "failed_model_families": failed_families,
            "stability_metrics": stability_metrics
        }
    
    def _load_symbol_metrics(self, target: str, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Load symbol-specific metrics for a (target, symbol) pair.
        
        Args:
            target: Target name
            symbol: Symbol name
        
        Returns:
            Dict with metrics or None if not found
        """
        target_dir = self.output_dir / target
        
        # Look for symbol-specific outputs
        # Could be in feature_selections/{target}/{symbol}/ or similar
        symbol_dir = target_dir / symbol
        fs_dir = target_dir / "feature_selections" / symbol
        
        # Try both locations
        metadata_path = None
        for path in [fs_dir / "model_metadata.json", symbol_dir / "model_metadata.json"]:
            if path.exists():
                metadata_path = path
                break
        
        score = None
        sample_size = None
        failed_families = []
        model_status = "UNKNOWN"
        leakage_status = "UNKNOWN"
        
        if metadata_path and metadata_path.exists():
            try:
                with open(metadata_path) as f:
                    metadata = json.load(f)
                
                # Extract scores per model family
                scores = []
                for key, data in metadata.items():
                    if isinstance(data, dict):
                        if "score" in data:
                            score_val = data["score"]
                            if score_val is not None and not np.isnan(score_val):
                                scores.append(score_val)
                        
                        # Check for failed families
                        if "reproducibility" in data:
                            repro = data["reproducibility"]
                            if isinstance(repro, dict):
                                # Check for failure indicators
                                if repro.get("status") == "FAILED":
                                    # Extract family from key
                                    parts = key.split(":")
                                    if len(parts) >= 3:
                                        family = parts[2]
                                        failed_families.append(family)
                
                if scores:
                    score = np.mean(scores)
                    model_status = "OK"
                else:
                    model_status = "FAILED"
            except Exception as e:
                logger.debug(f"Failed to load symbol metrics from {metadata_path}: {e}")
        
        # Load stability metrics
        stability_metrics = self._load_stability_metrics(target, universe_id=symbol)
        
        # Classify stability
        stability = self._classify_stability_from_metrics(stability_metrics)
        
        # Load leakage status
        leakage_status = self._load_leakage_status(target, symbol=symbol)
        
        if score is None:
            return None
        
        return {
            "target": target,
            "symbol": symbol,
            "mode": "SYMBOL",
            "score": float(score),
            "score_ci_low": None,
            "score_ci_high": None,
            "stability": stability,
            "sample_size": int(sample_size) if sample_size else 0,
            "leakage_status": leakage_status,
            "feature_set_id": None,
            "failed_model_families": failed_families,
            "model_status": model_status,
            "stability_metrics": stability_metrics
        }
    
    def _load_stability_metrics(
        self,
        target: str,
        universe_id: Optional[str] = None
    ) -> Optional[Dict[str, float]]:
        """
        Load stability metrics from feature importance snapshots.
        
        Args:
            target: Target name
            universe_id: Universe ID (symbol name or "ALL" for CS)
        
        Returns:
            Dict with stability metrics or None
        """
        try:
            from TRAINING.stability.feature_importance.io import load_snapshots
            from TRAINING.stability.feature_importance.analysis import compute_stability_metrics
            
            # Determine method based on universe
            if universe_id == "ALL" or universe_id is None:
                method = "multi_model_aggregated"
            else:
                method = "lightgbm"  # Default per-symbol method
            
            # Load snapshots
            snapshots = load_snapshots(
                target_name=target,
                method=method,
                universe_id=universe_id,
                base_dir=self.output_dir.parent if "feature_selections" in str(self.output_dir) else self.output_dir
            )
            
            if len(snapshots) < 2:
                return None
            
            # Compute stability metrics
            metrics = compute_stability_metrics(snapshots, top_k=20)
            return metrics
        except Exception as e:
            logger.debug(f"Failed to load stability metrics for {target}/{universe_id}: {e}")
            return None
    
    def _classify_stability_from_metrics(
        self,
        stability_metrics: Optional[Dict[str, float]]
    ) -> str:
        """
        Classify stability category from metrics.
        
        Args:
            stability_metrics: Dict with mean_overlap, std_overlap, mean_tau, std_tau
        
        Returns:
            Stability category string
        """
        if stability_metrics is None:
            return "UNKNOWN"
        
        mean_overlap = stability_metrics.get("mean_overlap", np.nan)
        std_overlap = stability_metrics.get("std_overlap", np.nan)
        mean_tau = stability_metrics.get("mean_tau", np.nan)
        std_tau = stability_metrics.get("std_tau", np.nan)
        
        # Check for divergence
        if not np.isnan(std_overlap) and std_overlap > 0.20:
            return "DIVERGED"
        if not np.isnan(std_tau) and std_tau > 0.25:
            return "DIVERGED"
        
        # Check for stability
        if (not np.isnan(mean_overlap) and mean_overlap >= 0.70 and
            not np.isnan(mean_tau) and mean_tau >= 0.60):
            return "STABLE"
        
        # Check for drifting
        if (not np.isnan(mean_overlap) and mean_overlap >= 0.50 and
            not np.isnan(mean_tau) and mean_tau >= 0.40):
            return "DRIFTING"
        
        return "UNKNOWN"
    
    def _load_leakage_status(
        self,
        target: str,
        symbol: Optional[str] = None
    ) -> str:
        """
        Load leakage status for target (and optionally symbol).
        
        Args:
            target: Target name
            symbol: Optional symbol name
        
        Returns:
            Leakage status string
        """
        # Look for leakage detection outputs
        # This would depend on where leakage detection stores its results
        # For now, default to UNKNOWN
        
        # Could check for:
        # - leakage_detection/{target}/results.json
        # - feature_selections/{target}/leakage_status.json
        # etc.
        
        return "UNKNOWN"  # Placeholder
    
    def save_routing_candidates(
        self,
        candidates_df: pd.DataFrame,
        output_path: Optional[Path] = None
    ) -> Path:
        """
        Save routing candidates to disk.
        
        Args:
            candidates_df: DataFrame with routing candidates
            output_path: Optional output path (defaults to METRICS/routing_candidates.parquet)
        
        Returns:
            Path where file was saved
        """
        if output_path is None:
            metrics_dir = self.output_dir.parent / "METRICS"
            metrics_dir.mkdir(parents=True, exist_ok=True)
            output_path = metrics_dir / "routing_candidates.parquet"
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as parquet (with fallback to CSV if parquet not available)
        try:
            candidates_df.to_parquet(output_path, index=False)
            logger.info(f"✅ Saved routing candidates: {output_path}")
        except Exception as e:
            logger.warning(f"Failed to save as parquet ({e}), falling back to CSV")
            csv_path = output_path.with_suffix(".csv")
            candidates_df.to_csv(csv_path, index=False)
            output_path = csv_path
            logger.info(f"✅ Saved routing candidates: {output_path}")
        
        # Also save as JSON for human inspection
        json_path = output_path.with_suffix(".json")
        candidates_df.to_json(json_path, orient="records", indent=2)
        logger.info(f"✅ Saved routing candidates JSON: {json_path}")
        
        return output_path
