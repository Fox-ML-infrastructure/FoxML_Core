# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial

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
        # Load resolved_mode from run context (SST)
        resolved_mode = None
        try:
            from TRAINING.orchestration.utils.run_context import get_resolved_mode
            resolved_mode = get_resolved_mode(self.output_dir)
            if resolved_mode:
                logger.info(f"📋 Using resolved_mode={resolved_mode} from run context (SST) for metrics aggregation")
        except Exception as e:
            logger.debug(f"Could not load resolved_mode from run context: {e}, will use inferred mode")
        
        rows = []
        
        for target in targets:
            # Cross-sectional metrics (use resolved_mode for mode field and path construction)
            cs_metrics = self._load_cross_sectional_metrics(target, resolved_mode=resolved_mode)
            if cs_metrics:
                rows.append(cs_metrics)
                logger.debug(f"✅ Loaded CS metrics for {target}")
            else:
                logger.warning(f"⚠️  No CS metrics found for {target}")
            
            # Symbol-specific metrics (use resolved_mode for mode field)
            symbols_found = 0
            symbols_missing = []
            for symbol in symbols:
                sym_metrics = self._load_symbol_metrics(target, symbol, resolved_mode=resolved_mode)
                if sym_metrics:
                    rows.append(sym_metrics)
                    symbols_found += 1
                else:
                    symbols_missing.append(symbol)
            
            if symbols_found > 0:
                logger.debug(f"✅ Loaded symbol metrics for {target}: {symbols_found}/{len(symbols)} symbols")
            if symbols_missing:
                logger.warning(f"⚠️  No symbol metrics found for {target}: {len(symbols_missing)}/{len(symbols)} symbols missing: {symbols_missing[:5]}{'...' if len(symbols_missing) > 5 else ''}")
        
        if not rows:
            logger.warning("No routing candidates found")
            return pd.DataFrame()
        
        df = pd.DataFrame(rows)
        
        # Add metadata
        df["timestamp"] = datetime.utcnow().isoformat()
        df["git_commit"] = git_commit or "unknown"
        
        return df
    
    def _load_cross_sectional_metrics(self, target: str, resolved_mode: Optional[str] = None) -> Optional[Dict[str, Any]]:
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
        
        # SST Architecture: Read from canonical location (reproducibility/cohort) first
        # Then check reference pointer, then legacy locations
        from TRAINING.orchestration.utils.target_first_paths import (
            get_target_reproducibility_dir, get_target_metrics_dir
        )
        target_repro_dir = get_target_reproducibility_dir(base_output_dir, target_name_clean)
        # Use resolved_mode for path construction (SST)
        view_for_path = resolved_mode if resolved_mode else "CROSS_SECTIONAL"
        target_fs_dir = target_repro_dir / view_for_path
        metadata_path = target_fs_dir / "multi_model_metadata.json"
        confidence_path = target_fs_dir / "target_confidence.json"
        
        score = None
        sample_size = None
        failed_families = []
        leakage_status = "UNKNOWN"
        metrics_data = None
        
        # 1. Try canonical location: find latest cohort in reproducibility/CROSS_SECTIONAL
        if target_fs_dir.exists():
            cohort_dirs = [d for d in target_fs_dir.iterdir() 
                          if d.is_dir() and d.name.startswith("cohort=")]
            if cohort_dirs:
                # Sort by cohort ID (assuming numeric or timestamp-based)
                latest_cohort = sorted(cohort_dirs, key=lambda x: x.name, reverse=True)[0]
                canonical_parquet = latest_cohort / "metrics.parquet"
                canonical_json = latest_cohort / "metrics.json"
                
                # Try parquet first (canonical)
                if canonical_parquet.exists():
                    try:
                        import pandas as pd
                        df = pd.read_parquet(canonical_parquet)
                        if len(df) > 0:
                            metrics_data = df.iloc[0].to_dict()
                            score = metrics_data.get('mean_score')
                            sample_size = metrics_data.get('N_effective_cs')
                            logger.debug(f"✅ Loaded metrics from canonical location: {canonical_parquet}")
                    except Exception as e:
                        logger.debug(f"Failed to load metrics from canonical parquet: {e}")
                
                # Fallback to JSON in canonical location
                if metrics_data is None and canonical_json.exists():
                    try:
                        with open(canonical_json, 'r') as f:
                            metrics_data = json.load(f)
                            score = metrics_data.get('mean_score')
                            sample_size = metrics_data.get('N_effective_cs')
                            logger.debug(f"✅ Loaded metrics from canonical JSON: {canonical_json}")
                    except Exception as e:
                        logger.debug(f"Failed to load metrics from canonical JSON: {e}")
        
        # 2. Fallback to reference pointer in metrics/ directory
        if metrics_data is None:
            target_metrics_dir = get_target_metrics_dir(base_output_dir, target_name_clean)
            view_metrics_dir = target_metrics_dir / "view=CROSS_SECTIONAL"
            ref_file = view_metrics_dir / "latest_ref.json"
            
            if ref_file.exists():
                try:
                    with open(ref_file, 'r') as f:
                        ref_data = json.load(f)
                    canonical_path = Path(ref_data.get("canonical_path", ""))
                    if canonical_path.exists():
                        from TRAINING.common.utils.metrics import MetricsWriter
                        metrics_data = MetricsWriter.export_metrics_json_from_parquet(canonical_path)
                        score = metrics_data.get('mean_score')
                        sample_size = metrics_data.get('N_effective_cs')
                        logger.debug(f"✅ Loaded metrics via reference pointer: {canonical_path}")
                except Exception as e:
                    logger.debug(f"Failed to follow reference pointer: {e}")
            
            # Also try direct read from metrics/ (legacy compatibility)
            if metrics_data is None:
                metrics_parquet = view_metrics_dir / "metrics.parquet"
                metrics_file = view_metrics_dir / "metrics.json"
                
                if metrics_parquet.exists():
                    try:
                        import pandas as pd
                        df = pd.read_parquet(metrics_parquet)
                        if len(df) > 0:
                            metrics_data = df.iloc[0].to_dict()
                            score = metrics_data.get('mean_score')
                            sample_size = metrics_data.get('N_effective_cs')
                    except Exception as e:
                        logger.debug(f"Failed to load metrics from parquet: {e}")
                elif metrics_file.exists():
                    try:
                        with open(metrics_file, 'r') as f:
                            metrics_data = json.load(f)
                            score = metrics_data.get('mean_score')
                            sample_size = metrics_data.get('N_effective_cs')
                    except Exception as e:
                        logger.debug(f"Failed to load metrics from JSON: {e}")
        
        # 3. Last resort: legacy structure
        if metrics_data is None:
            legacy_fs_dir = base_output_dir / "REPRODUCIBILITY" / "FEATURE_SELECTION" / "CROSS_SECTIONAL" / target_name_clean
            legacy_metrics_file = legacy_fs_dir / "metrics.json"
            if legacy_metrics_file.exists():
                try:
                    with open(legacy_metrics_file, 'r') as f:
                        metrics_data = json.load(f)
                        score = metrics_data.get('mean_score')
                        sample_size = metrics_data.get('N_effective_cs')
                except Exception as e:
                    logger.debug(f"Failed to load metrics from legacy location: {e}")
        
        if not metadata_path.exists() and not confidence_path.exists():
            legacy_fs_dir = base_output_dir / "REPRODUCIBILITY" / "FEATURE_SELECTION" / "CROSS_SECTIONAL" / target_name_clean
            if not metadata_path.exists():
                metadata_path = legacy_fs_dir / "multi_model_metadata.json"
            if not confidence_path.exists():
                confidence_path = legacy_fs_dir / "target_confidence.json"
        
        # Load from model_metadata.json (aggregated across symbols) - fallback if metrics not found
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
        
        # Use resolved_mode for mode field (SST)
        mode_for_row = resolved_mode if resolved_mode else "CROSS_SECTIONAL"
        return {
            "target": target,
            "symbol": None,  # CS has no symbol
            "mode": mode_for_row,
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
    
    def _load_symbol_metrics(self, target: str, symbol: str, resolved_mode: Optional[str] = None) -> Optional[Dict[str, Any]]:
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
        
        # Try target-first structure first: targets/<target>/reproducibility/{resolved_mode}/symbol=<symbol>/
        # Use resolved_mode for path construction (SST)
        from TRAINING.orchestration.utils.target_first_paths import get_target_reproducibility_dir
        target_repro_dir = get_target_reproducibility_dir(base_output_dir, target_name_clean)
        view_for_path = resolved_mode if resolved_mode else "SYMBOL_SPECIFIC"
        target_fs_dir = target_repro_dir / view_for_path / f"symbol={symbol}"
        
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
        
        # Fallback to CROSS_SECTIONAL cohort metadata for sample_size when symbol metrics missing
        if not metadata_path or not metadata_path.exists():
            # Try to get sample_size from CROSS_SECTIONAL cohort metadata
            cs_target_fs_dir = target_repro_dir / "CROSS_SECTIONAL"
            if cs_target_fs_dir.exists():
                cohort_dirs = [d for d in cs_target_fs_dir.iterdir() 
                               if d.is_dir() and d.name.startswith("cohort=")]
                if cohort_dirs:
                    latest_cohort = max(cohort_dirs, key=lambda d: d.stat().st_mtime)
                    metadata_file = latest_cohort / "metadata.json"
                    if metadata_file.exists():
                        try:
                            with open(metadata_file) as f:
                                cohort_meta = json.load(f)
                                # Extract sample_size from cohort metadata
                                sample_size = cohort_meta.get('n_samples', cohort_meta.get('N', None))
                                if sample_size:
                                    logger.debug(f"Using CROSS_SECTIONAL cohort metadata for sample_size: {sample_size}")
                        except Exception as e:
                            logger.debug(f"Failed to load CROSS_SECTIONAL cohort metadata: {e}")
        
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
        leakage_status = self._load_leakage_status(target, symbol=symbol, resolved_mode=resolved_mode)
        
        if score is None:
            return None
        
        return {
            "target": target,
            "symbol": symbol,
            "mode": resolved_mode if resolved_mode else "SYMBOL",  # Use resolved_mode (SST)
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
            
            # Filter snapshots by universe_id (symbol) if in SYMBOL_SPECIFIC mode
            # This prevents comparing stability across different symbols (which is expected to have low overlap)
            if view == "SYMBOL_SPECIFIC" and symbol:
                # Filter to snapshots with matching symbol in universe_id
                symbol_prefix = f"{symbol}:"
                filtered_snapshots = [
                    s for s in snapshots
                    if s.universe_id and (s.universe_id.startswith(symbol_prefix) or s.universe_id == symbol)
                ]
                if len(filtered_snapshots) >= 2:
                    snapshots = filtered_snapshots
                    logger.debug(f"Filtered stability snapshots to symbol={symbol}: {len(filtered_snapshots)} snapshots")
                elif len(snapshots) >= 2:
                    # Fallback: use all snapshots but warn
                    logger.warning(
                        f"⚠️  Stability computation: Could not filter snapshots by symbol={symbol}. "
                        f"Using all {len(snapshots)} snapshots (may include cross-symbol comparisons). "
                        f"Low overlap may be due to symbol heterogeneity, not instability."
                    )
            
            # Compute stability metrics (with filtering enabled)
            metrics = compute_stability_metrics(snapshots, top_k=20, filter_by_universe_id=True)
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
        symbol: Optional[str] = None,
        resolved_mode: Optional[str] = None
    ) -> str:
        """
        Load leakage status for target (and optionally symbol).
        
        Escalation policy: If leakage is BLOCKED but confirmed quarantine exists,
        downgrade to SUSPECT (allow with quarantine) since the issue has been addressed.
        
        Args:
            target: Target name
            symbol: Optional symbol name
            resolved_mode: Resolved mode (CROSS_SECTIONAL, SYMBOL_SPECIFIC, etc.)
        
        Returns:
            Leakage status string (SAFE, SUSPECT, BLOCKED, UNKNOWN)
        """
        # Look for leakage detection outputs
        # This would depend on where leakage detection stores its results
        # For now, default to UNKNOWN
        
        # Could check for:
        # - leakage_detection/{target}/results.json
        # - feature_selections/{target}/leakage_status.json
        # etc.
        
        leakage_status = "UNKNOWN"  # Placeholder
        
        # Small-panel leniency: Check if we're in a small panel scenario
        # Load small-panel config and check n_symbols from run context
        n_symbols = None
        try:
            from TRAINING.orchestration.utils.run_context import load_run_context
            context = load_run_context(self.output_dir)
            if context:
                n_symbols = context.get("n_symbols")
        except Exception as e:
            logger.debug(f"Could not load n_symbols from run context: {e}")
        
        # Load small-panel config
        small_panel_cfg = {}
        try:
            from CONFIG.config_loader import get_safety_config
            safety_cfg = get_safety_config()
            safety_section = safety_cfg.get('safety', {})
            leakage_cfg = safety_section.get('leakage_detection', {})
            small_panel_cfg = leakage_cfg.get('small_panel', {})
        except Exception as e:
            logger.debug(f"Could not load small-panel config: {e}")
        
        # Apply small-panel leniency if enabled and conditions are met
        if (small_panel_cfg.get('enabled', False) and 
            n_symbols is not None and 
            leakage_status in ["BLOCKED", "HIGH_SCORE", "SUSPICIOUS"]):
            min_symbols_threshold = small_panel_cfg.get('min_symbols_threshold', 10)
            downgrade_enabled = small_panel_cfg.get('downgrade_block_to_suspect', True)
            log_warning = small_panel_cfg.get('log_warning', True)
            
            if n_symbols < min_symbols_threshold and downgrade_enabled:
                if log_warning:
                    logger.warning(
                        f"🔒 Small panel detected (n_symbols={n_symbols} < {min_symbols_threshold}), "
                        f"downgrading leakage severity from {leakage_status} to SUSPECT. "
                        f"This allows dominance quarantine to attempt recovery before blocking."
                    )
                # Downgrade to SUSPECT (allows training to proceed, but with warning)
                leakage_status = "SUSPECT"
        
        # Escalation policy: Check for confirmed quarantine
        # If confirmed quarantine exists, leakage has been addressed via feature-level quarantine
        # Downgrade BLOCKED to SUSPECT (or allow) to prevent blocking target/view
        if leakage_status == "BLOCKED" and self.output_dir:
            try:
                from TRAINING.ranking.utils.dominance_quarantine import load_confirmed_quarantine
                
                # Determine view from resolved_mode or default to CROSS_SECTIONAL
                view = resolved_mode if resolved_mode in ["CROSS_SECTIONAL", "SYMBOL_SPECIFIC"] else "CROSS_SECTIONAL"
                
                confirmed_quarantine = load_confirmed_quarantine(
                    out_dir=self.output_dir,
                    target=target,
                    view=view,
                    symbol=symbol
                )
                
                if confirmed_quarantine:
                    # Confirmed quarantine exists - leakage has been addressed
                    # Downgrade BLOCKED to SUSPECT (allow with quarantine)
                    logger.info(
                        f"🔒 Escalation policy: Leakage BLOCKED for {target}/{view}/{symbol or 'ALL'}, "
                        f"but confirmed quarantine exists ({len(confirmed_quarantine)} features). "
                        f"Downgrading to SUSPECT (allow with quarantine)."
                    )
                    return "SUSPECT"  # Allow with quarantine, don't block
            except Exception as e:
                logger.debug(f"Could not check for confirmed quarantine: {e}")
        
        return leakage_status
    
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
            # Use globals/routing/ (new structure)
            from TRAINING.orchestration.utils.target_first_paths import run_root, globals_dir
            # Find base run directory
            base_dir = self.output_dir
            while base_dir.name in ["FEATURE_SELECTION", "TARGET_RANKING", "REPRODUCIBILITY", "feature_selections", "target_rankings"]:
                if not base_dir.parent.exists():
                    break
                base_dir = base_dir.parent
            
            run_root_dir = run_root(base_dir)
            routing_dir = globals_dir(run_root_dir, "routing")
            routing_dir.mkdir(parents=True, exist_ok=True)
            output_path = routing_dir / "routing_candidates.parquet"
        
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
