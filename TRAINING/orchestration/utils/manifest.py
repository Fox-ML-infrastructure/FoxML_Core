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
    targets: Optional[List[str]] = None
) -> Path:
    """
    Create manifest.json at run root.
    
    Args:
        output_dir: Base run output directory
        run_id: Run identifier (defaults to directory name or timestamp)
        config_digest: Config digest/hash for reproducibility
        targets: List of targets processed
    
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

