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
        # Determine base output directory (walk up from REPRODUCIBILITY/FEATURE_SELECTION)
        base_output_dir = self.output_dir
        while base_output_dir.name in ["FEATURE_SELECTION", "TARGET_RANKING", "REPRODUCIBILITY", "CROSS_SECTIONAL", "feature_selections", "target_rankings"]:
            base_output_dir = base_output_dir.parent
            if not base_output_dir.parent.exists() or base_output_dir.name == "RESULTS":
                break
        
        target_name_clean = target.replace('/', '_').replace('\\', '_')
        
        # Try target-first structure first: targets/<target>/reproducibility/
        from TRAINING.orchestration.utils.target_first_paths import get_target_reproducibility_dir
        target_repro_dir = get_target_reproducibility_dir(base_output_dir, target_name_clean)
        target_fs_dir = target_repro_dir / "CROSS_SECTIONAL"
        
        # Look for multi_model_metadata.json or target_confidence.json in target-first structure
        metadata_path = target_fs_dir / "multi_model_metadata.json"
        confidence_path = target_fs_dir / "target_confidence.json"
        
        # Fallback to legacy structure if not found in target-first
        if not metadata_path.exists() and not confidence_path.exists():
            legacy_fs_dir = base_output_dir / "REPRODUCIBILITY" / "FEATURE_SELECTION" / "CROSS_SECTIONAL" / target_name_clean
            if not metadata_path.exists():
                metadata_path = legacy_fs_dir / "multi_model_metadata.json"
            if not confidence_path.exists():
                confidence_path = legacy_fs_dir / "target_confidence.json"
        
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
        # Determine base output directory (walk up from REPRODUCIBILITY/FEATURE_SELECTION)
        base_output_dir = self.output_dir
        while base_output_dir.name in ["FEATURE_SELECTION", "TARGET_RANKING", "REPRODUCIBILITY", "SYMBOL_SPECIFIC", "CROSS_SECTIONAL", "feature_selections", "target_rankings"]:
            base_output_dir = base_output_dir.parent
            if not base_output_dir.parent.exists() or base_output_dir.name == "RESULTS":
                break
        
        target_name_clean = target.replace('/', '_').replace('\\', '_')
        
        # Try target-first structure first: targets/<target>/reproducibility/SYMBOL_SPECIFIC/symbol=<symbol>/
        from TRAINING.orchestration.utils.target_first_paths import get_target_reproducibility_dir
        target_repro_dir = get_target_reproducibility_dir(base_output_dir, target_name_clean)
        target_fs_dir = target_repro_dir / "SYMBOL_SPECIFIC" / f"symbol={symbol}"
        
        # Look for multi_model_metadata.json in target-first structure
        metadata_path = target_fs_dir / "multi_model_metadata.json"
        
        # Fallback to legacy structure if not found
        if not metadata_path.exists():
            legacy_fs_dir = base_output_dir / "REPRODUCIBILITY" / "FEATURE_SELECTION" / "SYMBOL_SPECIFIC" / target_name_clean / f"symbol={symbol}"
            if legacy_fs_dir.exists():
                metadata_path = legacy_fs_dir / "multi_model_metadata.json"
            else:
                metadata_path = None
        
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
            from TRAINING.stability.feature_importance.io import load_snapshots, get_snapshot_base_dir
            from TRAINING.stability.feature_importance.analysis import compute_stability_metrics
            
            # Determine method based on universe
            if universe_id == "ALL" or universe_id is None:
                method = "multi_model_aggregated"
            else:
                method = "lightgbm"  # Default per-symbol method
            
            # Determine base output directory (RESULTS/{run}/)
            # output_dir might be: REPRODUCIBILITY/FEATURE_SELECTION or REPRODUCIBILITY/TARGET_RANKING
            # Walk up to find the run-level directory
            base_output_dir = self.output_dir
            while base_output_dir.name in ["FEATURE_SELECTION", "TARGET_RANKING", "REPRODUCIBILITY", "feature_selections", "target_rankings"]:
                base_output_dir = base_output_dir.parent
                if not base_output_dir.parent.exists() or base_output_dir.name == "RESULTS":
                    break
            
            # Determine view and symbol from context
            # For metrics aggregator, we need to search both CROSS_SECTIONAL and SYMBOL_SPECIFIC
            # Try CROSS_SECTIONAL first (for universe_id == "ALL" or None)
            view = "CROSS_SECTIONAL" if (universe_id == "ALL" or universe_id is None) else "SYMBOL_SPECIFIC"
            symbol = None if view == "CROSS_SECTIONAL" else universe_id
            
            # Build REPRODUCIBILITY path for snapshots
            target_name_clean = target.replace('/', '_').replace('\\', '_')
            if view == "SYMBOL_SPECIFIC" and symbol:
                # Try FEATURE_SELECTION first (for feature selection metrics)
                repro_base_fs = base_output_dir / "REPRODUCIBILITY" / "FEATURE_SELECTION" / view / target_name_clean / f"symbol={symbol}"
                snapshot_base_dir_fs = get_snapshot_base_dir(repro_base_fs)
                # Also try TARGET_RANKING
                repro_base_tr = base_output_dir / "REPRODUCIBILITY" / "TARGET_RANKING" / view / target_name_clean / f"symbol={symbol}"
                snapshot_base_dir_tr = get_snapshot_base_dir(repro_base_tr)
            else:
                # Try FEATURE_SELECTION first
                repro_base_fs = base_output_dir / "REPRODUCIBILITY" / "FEATURE_SELECTION" / view / target_name_clean
                snapshot_base_dir_fs = get_snapshot_base_dir(repro_base_fs)
                # Also try TARGET_RANKING
                repro_base_tr = base_output_dir / "REPRODUCIBILITY" / "TARGET_RANKING" / view / target_name_clean
                snapshot_base_dir_tr = get_snapshot_base_dir(repro_base_tr)
            
            # Try loading from target-first first, then legacy locations (feature selection and target ranking)
            snapshots = []
            for snapshot_base_dir in [snapshot_base_dir_target, snapshot_base_dir_fs, snapshot_base_dir_tr]:
                if snapshot_base_dir is None:
                    continue
                if snapshot_base_dir.exists():
                    try:
                        found = load_snapshots(snapshot_base_dir, target, method)
                        snapshots.extend(found)
                    except Exception as e:
                        logger.debug(f"Failed to load snapshots from {snapshot_base_dir}: {e}")
            
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
