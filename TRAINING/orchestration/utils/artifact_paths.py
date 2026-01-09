# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
Artifact Paths Builder (Single Source of Truth)

Provides canonical paths for all training artifacts.
All model saves/loads should use these paths to ensure consistency.
"""

from pathlib import Path
from typing import Optional, Union

# SST: Import View enum for consistent view handling
from TRAINING.orchestration.utils.scope_resolution import View


def _get_default_extension(family: str) -> str:
    """Get default file extension for model family."""
    family_lower = family.lower()
    if family_lower in ['lightgbm', 'lgb']:
        return 'txt'
    elif family_lower in ['xgboost', 'xgb']:
        return 'json'
    elif family_lower in ['neural_network', 'mlp', 'cnn1d', 'lstm', 'transformer', 'vae', 'gan', 'multitask']:
        return 'keras'
    elif family_lower in ['pytorch', 'torch']:
        return 'pt'
    else:
        return 'joblib'  # Default for scikit-learn models


class ArtifactPaths:
    """Single Source of Truth for all artifact paths."""
    
    @staticmethod
    def model_dir(
        run_root: Path, 
        target: str, 
        view: Union[str, View], 
        family: str, 
        symbol: Optional[str] = None,
        universe_sig: Optional[str] = None
    ) -> Path:
        """
        Get canonical model directory.
        
        Args:
            run_root: Base run output directory
            target: Target name
            view: View enum or "CROSS_SECTIONAL" or "SYMBOL_SPECIFIC" string
            family: Model family name
            symbol: Optional symbol name (required if view is View.SYMBOL_SPECIFIC)
            universe_sig: Optional universe signature for cross-run reproducibility
        
        Returns:
            Path to targets/<target>/models/view=<view>/[universe=<universe_sig>/][symbol=<symbol>/]family=<family>/
        
        Raises:
            ValueError: If view is View.SYMBOL_SPECIFIC but symbol is None
        """
        # Normalize view to enum for comparison and path construction
        view_enum = View.from_string(view) if isinstance(view, str) else view
        if view_enum == View.SYMBOL_SPECIFIC and symbol is None:
            raise ValueError("symbol parameter is required when view=View.SYMBOL_SPECIFIC")
        
        # Use enum value for path construction (ensures consistent string representation)
        view_str = str(view_enum)  # View enum's __str__ returns .value
        base = run_root / "targets" / target / "models" / f"view={view_str}"
        if universe_sig:
            base = base / f"universe={universe_sig}"
        if view_enum == View.SYMBOL_SPECIFIC and symbol:
            base = base / f"symbol={symbol}"
        return base / f"family={family}"
    
    @staticmethod
    def model_file(model_dir: Path, family: str, extension: Optional[str] = None) -> Path:
        """
        Get model file path (no target/symbol in filename - path encodes it).
        
        Args:
            model_dir: Model directory (from model_dir())
            family: Model family name (for extension inference)
            extension: Optional file extension (defaults based on family)
        
        Returns:
            Path to model file (e.g., model.joblib, model.keras, model.txt)
        """
        if extension is None:
            extension = _get_default_extension(family)
        return model_dir / f"model.{extension}"
    
    @staticmethod
    def metadata_file(model_dir: Path) -> Path:
        """
        Get metadata file path.
        
        Args:
            model_dir: Model directory (from model_dir())
        
        Returns:
            Path to model_meta.json
        """
        return model_dir / "model_meta.json"
    
    @staticmethod
    def metrics_file(model_dir: Path) -> Path:
        """
        Get metrics file path.
        
        Args:
            model_dir: Model directory (from model_dir())
        
        Returns:
            Path to metrics.json
        """
        return model_dir / "metrics.json"
    
    @staticmethod
    def fingerprints_file(model_dir: Path) -> Path:
        """
        Get fingerprints file path.
        
        Contains feature_fingerprint and routing_fingerprint for reproducibility.
        
        Args:
            model_dir: Model directory (from model_dir())
        
        Returns:
            Path to fingerprints.json
        """
        return model_dir / "fingerprints.json"
    
    @staticmethod
    def reproducibility_file(model_dir: Path) -> Path:
        """
        Get reproducibility metadata file path.
        
        Args:
            model_dir: Model directory (from model_dir())
        
        Returns:
            Path to reproducibility.json
        """
        return model_dir / "reproducibility.json"
    
    @staticmethod
    def scaler_file(model_dir: Path, family: str) -> Path:
        """
        Get scaler file path.
        
        Args:
            model_dir: Model directory (from model_dir())
            family: Model family name (for extension)
        
        Returns:
            Path to scaler file (e.g., scaler.joblib)
        """
        return model_dir / "scaler.joblib"
    
    @staticmethod
    def imputer_file(model_dir: Path, family: str) -> Path:
        """
        Get imputer file path.
        
        Args:
            model_dir: Model directory (from model_dir())
            family: Model family name (for extension)
        
        Returns:
            Path to imputer file (e.g., imputer.joblib)
        """
        return model_dir / "imputer.joblib"

