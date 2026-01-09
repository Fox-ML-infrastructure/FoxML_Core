# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
Artifact Mirror Generation

Optional utility to generate family-first symlinks or manifest for browsing.
This is a derived view only - canonical location is targets/<target>/models/...
"""

from pathlib import Path
from typing import Optional, Dict, List
import logging
import pandas as pd

logger = logging.getLogger(__name__)


def generate_family_first_mirrors(
    run_root: Path,
    create_symlinks: bool = False
) -> Optional[Path]:
    """
    Generate derived mirrors for family-first browsing.
    
    If create_symlinks=True: Create symlinks in training_results/
    If False: Only update globals/manifests/models_manifest.parquet
    
    Args:
        run_root: Base run output directory
        create_symlinks: If True, create symlinks (default: False, only manifest)
    
    Returns:
        Path to manifest file if created, None otherwise
    """
    manifest = []
    targets_dir = run_root / "targets"
    
    if not targets_dir.exists():
        logger.debug("No targets directory found, skipping mirror generation")
        return None
    
    for target_dir in targets_dir.iterdir():
        if not target_dir.is_dir():
            continue
        target = target_dir.name
        models_dir = target_dir / "models"
        if not models_dir.exists():
            continue
        
        # Walk models directory
        for view_dir in models_dir.iterdir():
            if not view_dir.is_dir() or not view_dir.name.startswith("view="):
                continue
            view = view_dir.name.replace("view=", "")
            
            # Handle SYMBOL_SPECIFIC
            if view == "SYMBOL_SPECIFIC":
                for symbol_dir in view_dir.iterdir():
                    if not symbol_dir.is_dir() or not symbol_dir.name.startswith("symbol="):
                        continue
                    symbol = symbol_dir.name.replace("symbol=", "")
                    for family_dir in symbol_dir.iterdir():
                        if not family_dir.is_dir() or not family_dir.name.startswith("family="):
                            continue
                        family = family_dir.name.replace("family=", "")
                        canonical_path = family_dir
                        
                        # Add to manifest
                        manifest.append({
                            'target': target,
                            'view': view,
                            'symbol': symbol,
                            'family': family,
                            'canonical_path': str(canonical_path.relative_to(run_root)),
                            'absolute_path': str(canonical_path)
                        })
                        
                        # Create symlink if requested
                        if create_symlinks:
                            try:
                                mirror_dir = run_root / "training_results" / family / f"view={view}" / f"symbol={symbol}"
                                mirror_dir.mkdir(parents=True, exist_ok=True)
                                symlink_path = mirror_dir / f"target={target}"
                                
                                # Remove existing symlink if it exists
                                if symlink_path.exists() or symlink_path.is_symlink():
                                    symlink_path.unlink()
                                
                                # Create symlink
                                symlink_path.symlink_to(canonical_path, target_is_directory=True)
                                logger.debug(f"Created symlink: {symlink_path} -> {canonical_path}")
                            except Exception as e:
                                logger.warning(f"Failed to create symlink for {target}/{family}/{symbol}: {e}")
            else:
                # CROSS_SECTIONAL
                for family_dir in view_dir.iterdir():
                    if not family_dir.is_dir() or not family_dir.name.startswith("family="):
                        continue
                    family = family_dir.name.replace("family=", "")
                    canonical_path = family_dir
                    
                    # Add to manifest
                    manifest.append({
                        'target': target,
                        'view': view,
                        'symbol': None,
                        'family': family,
                        'canonical_path': str(canonical_path.relative_to(run_root)),
                        'absolute_path': str(canonical_path)
                    })
                    
                    # Create symlink if requested
                    if create_symlinks:
                        try:
                            mirror_dir = run_root / "training_results" / family / f"view={view}"
                            mirror_dir.mkdir(parents=True, exist_ok=True)
                            symlink_path = mirror_dir / f"target={target}"
                            
                            # Remove existing symlink if it exists
                            if symlink_path.exists() or symlink_path.is_symlink():
                                symlink_path.unlink()
                            
                            # Create symlink
                            symlink_path.symlink_to(canonical_path, target_is_directory=True)
                            logger.debug(f"Created symlink: {symlink_path} -> {canonical_path}")
                        except Exception as e:
                            logger.warning(f"Failed to create symlink for {target}/{family}: {e}")
    
    # Write manifest
    if manifest:
        manifest_df = pd.DataFrame(manifest)
        manifest_path = run_root / "globals" / "manifests" / "models_manifest.parquet"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_df.to_parquet(manifest_path)
        logger.info(f"✅ Generated models manifest: {len(manifest)} entries -> {manifest_path}")
        return manifest_path
    else:
        logger.debug("No models found, skipping manifest generation")
        return None

