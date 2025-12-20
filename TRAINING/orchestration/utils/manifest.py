"""
Manifest Generation Utilities

Creates and updates manifest.json at run root with run metadata and target index.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import subprocess

logger = logging.getLogger(__name__)


def get_git_sha() -> Optional[str]:
    """Get current git SHA."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()[:12]  # Short SHA
    except Exception:
        pass
    return None


def create_manifest(
    output_dir: Path,
    run_id: Optional[str] = None,
    config_digest: Optional[str] = None,
    targets: Optional[List[str]] = None,
    experiment_config: Optional[Dict[str, Any]] = None,
    run_metadata: Optional[Dict[str, Any]] = None
) -> Path:
    """
    Create manifest.json at run root with comprehensive run metadata.
    
    Args:
        output_dir: Base run output directory
        run_id: Run identifier (defaults to directory name or timestamp)
        config_digest: Config digest/hash for reproducibility
        targets: List of targets processed
        experiment_config: Optional experiment config dict (name, data_dir, symbols, etc.)
        run_metadata: Optional additional run metadata (data_dir, symbols, n_effective, etc.)
    
    Returns:
        Path to manifest.json
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate run_id if not provided
    if run_id is None:
        if output_dir.name.startswith("run_"):
            run_id = output_dir.name
        else:
            run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Get git SHA
    git_sha = get_git_sha()
    
    # Build manifest structure
    manifest = {
        "run_id": run_id,
        "git_sha": git_sha,
        "config_digest": config_digest,
        "created_at": datetime.now().isoformat(),
        "targets": targets or []
    }
    
    # Add experiment config if provided
    if experiment_config:
        manifest["experiment"] = {
            "name": experiment_config.get("name") or experiment_config.get("experiment", {}).get("name"),
            "description": experiment_config.get("description") or experiment_config.get("experiment", {}).get("description"),
            "data_dir": str(experiment_config.get("data_dir")) if experiment_config.get("data_dir") else None,
            "symbols": experiment_config.get("symbols") or experiment_config.get("data", {}).get("symbols"),
            "interval": experiment_config.get("interval") or experiment_config.get("data", {}).get("bar_interval"),
            "max_samples_per_symbol": experiment_config.get("max_samples_per_symbol") or experiment_config.get("data", {}).get("max_rows_per_symbol")
        }
    
    # Add run metadata if provided
    if run_metadata:
        manifest["run_metadata"] = {
            k: v for k, v in run_metadata.items()
            if k not in ["experiment_config", "targets"]  # Avoid duplication
        }
    
    # Add target index if targets directory exists
    targets_dir = output_dir / "targets"
    if targets_dir.exists():
        target_index = {}
        for target_dir in targets_dir.iterdir():
            if target_dir.is_dir():
                target_name = target_dir.name
                target_info = {
                    "decision": _find_files(target_dir / "decision"),
                    "models": _find_model_families(target_dir / "models"),
                    "metrics": _find_files(target_dir / "metrics"),
                    "trends": _find_files(target_dir / "trends"),
                    "reproducibility": _find_files(target_dir / "reproducibility")
                }
                target_index[target_name] = target_info
        manifest["target_index"] = target_index
    
    # Add trend reports references if they exist
    # trend_reports is at RESULTS/trend_reports/ (outside run directories)
    # Find RESULTS directory by walking up from output_dir
    results_dir = output_dir
    for _ in range(10):
        if results_dir.name == "RESULTS":
            break
        if not results_dir.parent.exists():
            break
        results_dir = results_dir.parent
    
    if results_dir.name == "RESULTS":
        trend_reports_dir = results_dir / "trend_reports"
        if trend_reports_dir.exists():
            trend_index = {}
            by_target_dir = trend_reports_dir / "by_target"
            if by_target_dir.exists():
                for target_dir in by_target_dir.iterdir():
                    if target_dir.is_dir():
                        target_name = target_dir.name
                        trend_files = {}
                        for trend_file in ["performance_timeseries.parquet", "routing_score_timeseries.parquet", 
                                         "feature_importance_timeseries.parquet"]:
                            trend_path = target_dir / trend_file
                            if trend_path.exists():
                                # Store relative path from RESULTS directory
                                trend_files[trend_file.replace(".parquet", "")] = str(trend_path.relative_to(results_dir))
                        if trend_files:
                            trend_index[target_name] = trend_files
            
            # Also add by_run references
            by_run_dir = trend_reports_dir / "by_run"
            if by_run_dir.exists():
                run_snapshots = {}
                for run_dir in by_run_dir.iterdir():
                    if run_dir.is_dir():
                        snapshot_file = run_dir / f"{run_dir.name}_summary.json"
                        if snapshot_file.exists():
                            run_snapshots[run_dir.name] = str(snapshot_file.relative_to(results_dir))
                if run_snapshots:
                    trend_index["_by_run"] = run_snapshots
            
            if trend_index:
                manifest["trend_reports"] = trend_index
    
    # Write manifest
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2, default=str)
    
    logger.info(f"✅ Created manifest.json: {manifest_path}")
    return manifest_path


def update_manifest(
    output_dir: Path,
    target: str,
    selected_feature_set_digest: Optional[str] = None,
    split_ids: Optional[List[str]] = None,
    model_artifact_paths: Optional[Dict[str, List[str]]] = None,
    decision_paths: Optional[Dict[str, str]] = None
) -> None:
    """
    Update manifest.json with target-specific information.
    
    Args:
        output_dir: Base run output directory
        target: Target name
        selected_feature_set_digest: Feature set digest
        split_ids: List of split IDs
        model_artifact_paths: Dict mapping family -> list of artifact paths
        decision_paths: Dict mapping decision type -> path
    """
    manifest_path = output_dir / "manifest.json"
    
    if not manifest_path.exists():
        # Create initial manifest if it doesn't exist
        create_manifest(output_dir)
    
    # Load existing manifest
    with open(manifest_path) as f:
        manifest = json.load(f)
    
    # Initialize target_index if needed
    if "target_index" not in manifest:
        manifest["target_index"] = {}
    
    target_name_clean = target.replace('/', '_').replace('\\', '_')
    
    # Update or create target entry
    if target_name_clean not in manifest["target_index"]:
        manifest["target_index"][target_name_clean] = {}
    
    target_entry = manifest["target_index"][target_name_clean]
    
    # Update target-specific fields
    if selected_feature_set_digest:
        target_entry["selected_feature_set_digest"] = selected_feature_set_digest
    if split_ids:
        target_entry["split_ids"] = split_ids
    if model_artifact_paths:
        target_entry["model_artifact_paths"] = model_artifact_paths
    if decision_paths:
        target_entry["decision_paths"] = decision_paths
    
    # Update timestamp
    manifest["updated_at"] = datetime.now().isoformat()
    
    # Write updated manifest
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2, default=str)
    
    logger.debug(f"Updated manifest.json for target {target_name_clean}")


def _find_files(directory: Path) -> List[str]:
    """Find all files in a directory recursively."""
    if not directory.exists():
        return []
    files = []
    for path in directory.rglob("*"):
        if path.is_file():
            files.append(str(path.relative_to(directory.parent.parent)))
    return sorted(files)


def _find_model_families(models_dir: Path) -> Dict[str, List[str]]:
    """Find model families and their artifacts."""
    if not models_dir.exists():
        return {}
    
    families = {}
    for family_dir in models_dir.iterdir():
        if family_dir.is_dir():
            family_name = family_dir.name
            artifacts = []
            for artifact_path in family_dir.rglob("*"):
                if artifact_path.is_file():
                    artifacts.append(str(artifact_path.relative_to(models_dir.parent.parent)))
            if artifacts:
                families[family_name] = sorted(artifacts)
    
    return families


def create_target_metadata(
    output_dir: Path,
    target_name: str
) -> Path:
    """
    Create per-target metadata.json that aggregates information from all cohorts.
    
    This file provides a single source of truth for all metadata related to a target,
    aggregating information from:
    - All cohort directories (CROSS_SECTIONAL and SYMBOL_SPECIFIC)
    - Decision files
    - Metrics summaries
    - Feature selection results
    
    Args:
        output_dir: Base run output directory
        target_name: Target name (will be cleaned)
    
    Returns:
        Path to target metadata.json
    """
    import json
    from TRAINING.orchestration.utils.target_first_paths import (
        get_target_reproducibility_dir, get_target_decision_dir, get_target_metrics_dir
    )
    
    target_name_clean = target_name.replace('/', '_').replace('\\', '_')
    target_dir = output_dir / "targets" / target_name_clean
    
    if not target_dir.exists():
        logger.warning(f"Target directory does not exist: {target_dir}")
        return None
    
    target_metadata = {
        "target_name": target_name,
        "target_name_clean": target_name_clean,
        "created_at": datetime.now().isoformat(),
        "cohorts": {},
        "views": {},
        "decisions": {},
        "metrics_summary": {}
    }
    
    # Collect cohort metadata from reproducibility directory
    repro_dir = get_target_reproducibility_dir(output_dir, target_name_clean)
    if repro_dir.exists():
        for view_dir in repro_dir.iterdir():
            if view_dir.is_dir() and view_dir.name in ["CROSS_SECTIONAL", "SYMBOL_SPECIFIC"]:
                view_name = view_dir.name
                target_metadata["views"][view_name] = {
                    "cohorts": []
                }
                
                # For SYMBOL_SPECIFIC, need to check symbol subdirectories
                if view_name == "SYMBOL_SPECIFIC":
                    for symbol_dir in view_dir.iterdir():
                        if symbol_dir.is_dir() and symbol_dir.name.startswith("symbol="):
                            symbol = symbol_dir.name.replace("symbol=", "")
                            # Find all cohort directories under symbol
                            for cohort_dir in symbol_dir.iterdir():
                                if cohort_dir.is_dir() and cohort_dir.name.startswith("cohort="):
                                    cohort_id = cohort_dir.name.replace("cohort=", "")
                                    metadata_file = cohort_dir / "metadata.json"
                                    
                                    cohort_info = {
                                        "cohort_id": cohort_id,
                                        "symbol": symbol,
                                        "path": str(cohort_dir.relative_to(output_dir))
                                    }
                                    
                                    # Load metadata if available
                                    if metadata_file.exists():
                                        try:
                                            with open(metadata_file) as f:
                                                cohort_metadata = json.load(f)
                                                cohort_info["metadata"] = {
                                                    "stage": cohort_metadata.get("stage"),
                                                    "route_type": cohort_metadata.get("route_type"),
                                                    "n_effective": cohort_metadata.get("N_effective") or cohort_metadata.get("n_effective"),
                                                    "n_symbols": cohort_metadata.get("n_symbols"),
                                                    "date_start": cohort_metadata.get("date_start") or cohort_metadata.get("date_range_start"),
                                                    "date_end": cohort_metadata.get("date_end") or cohort_metadata.get("date_range_end"),
                                                    "universe_id": cohort_metadata.get("universe_id"),
                                                    "min_cs": cohort_metadata.get("min_cs"),
                                                    "max_cs_samples": cohort_metadata.get("max_cs_samples")
                                                }
                                        except Exception as e:
                                            logger.debug(f"Failed to load metadata from {metadata_file}: {e}")
                                    
                                    target_metadata["views"][view_name]["cohorts"].append(cohort_info)
                                    target_metadata["cohorts"][cohort_id] = {
                                        "view": view_name,
                                        "symbol": symbol,
                                        "path": str(cohort_dir.relative_to(output_dir))
                                    }
                else:
                    # CROSS_SECTIONAL: cohorts directly under view
                    for cohort_dir in view_dir.iterdir():
                        if cohort_dir.is_dir() and cohort_dir.name.startswith("cohort="):
                            cohort_id = cohort_dir.name.replace("cohort=", "")
                            metadata_file = cohort_dir / "metadata.json"
                            
                            cohort_info = {
                                "cohort_id": cohort_id,
                                "path": str(cohort_dir.relative_to(output_dir))
                            }
                            
                            # Load metadata if available
                            if metadata_file.exists():
                                try:
                                    with open(metadata_file) as f:
                                        cohort_metadata = json.load(f)
                                        cohort_info["metadata"] = {
                                            "stage": cohort_metadata.get("stage"),
                                            "route_type": cohort_metadata.get("route_type"),
                                            "n_effective": cohort_metadata.get("N_effective") or cohort_metadata.get("n_effective"),
                                            "n_symbols": cohort_metadata.get("n_symbols"),
                                            "date_start": cohort_metadata.get("date_start") or cohort_metadata.get("date_range_start"),
                                            "date_end": cohort_metadata.get("date_end") or cohort_metadata.get("date_range_end"),
                                            "universe_id": cohort_metadata.get("universe_id"),
                                            "min_cs": cohort_metadata.get("min_cs"),
                                            "max_cs_samples": cohort_metadata.get("max_cs_samples")
                                        }
                                except Exception as e:
                                    logger.debug(f"Failed to load metadata from {metadata_file}: {e}")
                            
                            target_metadata["views"][view_name]["cohorts"].append(cohort_info)
                            target_metadata["cohorts"][cohort_id] = {
                                "view": view_name,
                                "path": str(cohort_dir.relative_to(output_dir))
                            }
    
    # Collect decision files
    decision_dir = get_target_decision_dir(output_dir, target_name_clean)
    if decision_dir.exists():
        for decision_file in decision_dir.iterdir():
            if decision_file.is_file() and decision_file.suffix in [".json", ".yaml"]:
                decision_type = decision_file.stem
                target_metadata["decisions"][decision_type] = {
                    "path": str(decision_file.relative_to(output_dir)),
                    "format": decision_file.suffix[1:]  # Remove leading dot
                }
    
    # Collect metrics summary
    metrics_dir = get_target_metrics_dir(output_dir, target_name_clean)
    if metrics_dir.exists():
        for view_dir in metrics_dir.iterdir():
            if view_dir.is_dir() and view_dir.name.startswith("view="):
                view_name = view_dir.name.replace("view=", "")
                metrics_file = view_dir / "metrics.json"
                if metrics_file.exists():
                    try:
                        with open(metrics_file) as f:
                            metrics_data = json.load(f)
                            target_metadata["metrics_summary"][view_name] = {
                                "path": str(metrics_file.relative_to(output_dir)),
                                "mean_score": metrics_data.get("mean_score"),
                                "std_score": metrics_data.get("std_score"),
                                "composite_score": metrics_data.get("composite_score"),
                                "metric_name": metrics_data.get("metric_name")
                            }
                    except Exception as e:
                        logger.debug(f"Failed to load metrics from {metrics_file}: {e}")
    
    # Write target metadata
    target_metadata_file = target_dir / "metadata.json"
    with open(target_metadata_file, 'w') as f:
        json.dump(target_metadata, f, indent=2, default=str)
    
    logger.debug(f"Created target metadata: {target_metadata_file}")
    return target_metadata_file

