"""
Copyright (c) 2025 Fox ML Infrastructure

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
Family Router for Sequential vs Cross-Sectional Models
=====================================================

Routes different model families to appropriate data processing pipelines.
"""


import logging
from typing import Dict, List, Set, Any, Optional, Tuple
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

from features.seq_builder import build_sequences_panel, validate_sequences
from datasets.seq_dataset import SeqDataset, SeqDataModule, create_seq_dataloader

logger = logging.getLogger(__name__)

# Define sequence-based model families
SEQUENCE_FAMILIES: Set[str] = {
    "CNN1D", "LSTM", "Transformer", "TabLSTM", "TabTransformer", 
    "VAE", "MultiTask", "CNN1DTrainer", "LSTMTrainer", "TransformerTrainer",
    "TabLSTMTrainer", "TabTransformerTrainer", "VAETrainer", "MultiTaskTrainer"
}

# Define cross-sectional model families
CROSS_SECTIONAL_FAMILIES: Set[str] = {
    "LightGBM", "XGBoost", "MLP", "NeuralNetwork", "Ensemble", 
    "RewardBased", "QuantileLightGBM", "NGBoost", "GMMRegime", 
    "ChangePoint", "FTRLProximal", "GAN", "MetaLearning",
    "LightGBMTrainer", "XGBoostTrainer", "MLPTrainer", "NeuralNetworkTrainer",
    "EnsembleTrainer", "RewardBasedTrainer", "QuantileLightGBMTrainer",
    "NGBoostTrainer", "GMMRegimeTrainer", "ChangePointTrainer",
    "FTRLProximalTrainer", "GANTrainer", "MetaLearningTrainer"
}

class FamilyRouter:
    """
    Router to determine data processing pipeline based on model family.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize family router.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.sequential_config = self.config.get('sequential', {})
        
        # Get configuration parameters
        self.lookback_T = self.sequential_config.get('lookback_T', 60)
        self.stride = self.sequential_config.get('stride', 1)
        self.horizon_map = self.sequential_config.get('horizon_map', {})
        
        logger.info(f"FamilyRouter initialized: lookback_T={self.lookback_T}, "
                   f"stride={self.stride}, horizon_map={self.horizon_map}")
    
    def is_sequence_family(self, family: str) -> bool:
        """
        Check if a model family requires sequential data.
        
        Args:
            family: Model family name
        
        Returns:
            True if sequence-based, False if cross-sectional
        """
        return family in SEQUENCE_FAMILIES
    
    def is_cross_sectional_family(self, family: str) -> bool:
        """
        Check if a model family uses cross-sectional data.
        
        Args:
            family: Model family name
        
        Returns:
            True if cross-sectional, False if sequence-based
        """
        return family in CROSS_SECTIONAL_FAMILIES
    
    def get_horizon_bars(self, target_name: str) -> int:
        """
        Get horizon bars for a target.
        
        Args:
            target_name: Target name (e.g., 'fwd_ret_5m', 'fwd_ret_15m')
        
        Returns:
            Number of horizon bars
        """
        # Extract horizon from target name
        if 'fwd_ret_' in target_name:
            try:
                horizon_str = target_name.split('_')[-1]
                if horizon_str.endswith('m'):
                    return int(horizon_str[:-1])
                elif horizon_str.endswith('h'):
                    return int(horizon_str[:-1]) * 60  # Convert hours to minutes
            except (ValueError, IndexError):
                pass
        
        # Use horizon map if available
        if target_name in self.horizon_map:
            return self.horizon_map[target_name]
        
        # Default horizon
        return 1
    
    def create_dataset(self, family: str, panel: Dict[str, pd.DataFrame],
                      feature_cols: List[str], target_col: str,
                      **kwargs) -> Tuple[Dataset, Optional[DataLoader]]:
        """
        Create appropriate dataset based on model family.
        
        Args:
            family: Model family name
            panel: Data panel {symbol: DataFrame}
            feature_cols: List of feature columns
            target_col: Target column name
            **kwargs: Additional arguments
        
        Returns:
            Dataset and optional DataLoader
        """
        if self.is_sequence_family(family):
            return self._create_sequence_dataset(panel, feature_cols, target_col, **kwargs)
        else:
            return self._create_cross_sectional_dataset(panel, feature_cols, target_col, **kwargs)
    
    def _create_sequence_dataset(self, panel: Dict[str, pd.DataFrame],
                               feature_cols: List[str], target_col: str,
                               **kwargs) -> Tuple[SeqDataset, Optional[DataLoader]]:
        """
        Create sequential dataset for sequence-based models.
        
        Args:
            panel: Data panel
            feature_cols: Feature columns
            target_col: Target column
            **kwargs: Additional arguments
        
        Returns:
            Sequential dataset and optional DataLoader
        """
        # Get horizon for this target
        horizon_bars = self.get_horizon_bars(target_col)
        
        # Build sequences
        X, y, ts, syms = build_sequences_panel(
            panel=panel,
            feature_cols=feature_cols,
            target_col=target_col,
            lookback_T=self.lookback_T,
            horizon_bars=horizon_bars,
            stride=self.stride
        )
        
        # Validate sequences
        if not validate_sequences(X, y, ts, self.lookback_T, feature_cols):
            logger.error("Sequence validation failed")
            return None, None
        
        # Create dataset
        dataset = SeqDataset(X, y, ts, syms)
        
        # Create DataLoader if requested
        dataloader = None
        if kwargs.get('create_dataloader', False):
            batch_size = kwargs.get('batch_size', 32)
            shuffle = kwargs.get('shuffle', True)
            dataloader = create_seq_dataloader(dataset, batch_size, shuffle)
        
        logger.info(f"Created sequential dataset: {len(dataset)} samples, "
                   f"shape {X.shape}, target '{target_col}'")
        
        return dataset, dataloader
    
    def _create_cross_sectional_dataset(self, panel: Dict[str, pd.DataFrame],
                                      feature_cols: List[str], target_col: str,
                                      **kwargs) -> Tuple[Any, Optional[DataLoader]]:
        """
        Create cross-sectional dataset for tabular models.
        
        Args:
            panel: Data panel
            feature_cols: Feature columns
            target_col: Target column
            **kwargs: Additional arguments
        
        Returns:
            Cross-sectional dataset and optional DataLoader
        """
        # This would use your existing cross-sectional dataset
        # For now, return None as placeholder
        logger.info(f"Cross-sectional dataset creation not implemented yet for target '{target_col}'")
        return None, None
    
    def get_model_adapter(self, family: str, input_dim: int, output_dim: int = 1,
                         **kwargs) -> Any:
        """
        Get appropriate model adapter for the family.
        
        Args:
            family: Model family name
            input_dim: Input dimension
            output_dim: Output dimension
            **kwargs: Additional arguments
        
        Returns:
            Model adapter
        """
        if not self.is_sequence_family(family):
            logger.info(f"Cross-sectional model '{family}' - no adapter needed")
            return None
        
        # Import adapters
        from models.seq_adapters import (
            CNN1DHead, LSTMHead, TransformerHead, 
            TabLSTMHead, TabTransformerHead
        )
        
        # Route to appropriate adapter
        if family in ["CNN1D", "CNN1DTrainer"]:
            return CNN1DHead(input_dim, **kwargs)
        elif family in ["LSTM", "LSTMTrainer"]:
            return LSTMHead(input_dim, **kwargs)
        elif family in ["Transformer", "TransformerTrainer"]:
            return TransformerHead(input_dim, **kwargs)
        elif family in ["TabLSTM", "TabLSTMTrainer"]:
            tabular_dim = kwargs.get('tabular_dim', input_dim // 2)
            return TabLSTMHead(input_dim, tabular_dim, **kwargs)
        elif family in ["TabTransformer", "TabTransformerTrainer"]:
            tabular_dim = kwargs.get('tabular_dim', input_dim // 2)
            return TabTransformerHead(input_dim, tabular_dim, **kwargs)
        else:
            logger.warning(f"No adapter found for sequence family '{family}'")
            return None
    
    def get_training_config(self, family: str, **kwargs) -> Dict[str, Any]:
        """
        Get training configuration for a model family.
        
        Args:
            family: Model family name
            **kwargs: Additional arguments
        
        Returns:
            Training configuration
        """
        config = {
            'family': family,
            'is_sequence': self.is_sequence_family(family),
            'is_cross_sectional': self.is_cross_sectional_family(family)
        }
        
        if self.is_sequence_family(family):
            config.update({
                'lookback_T': self.lookback_T,
                'stride': self.stride,
                'requires_sequences': True,
                'input_shape': f'(batch_size, {self.lookback_T}, features)'
            })
        else:
            config.update({
                'requires_sequences': False,
                'input_shape': '(batch_size, features)'
            })
        
        return config

def create_family_router(config: Optional[Dict[str, Any]] = None) -> FamilyRouter:
    """
    Create a family router instance.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        FamilyRouter instance
    """
    return FamilyRouter(config)

# Convenience functions
def is_sequence_model(family: str) -> bool:
    """Check if model family requires sequences."""
    return family in SEQUENCE_FAMILIES

def is_cross_sectional_model(family: str) -> bool:
    """Check if model family uses cross-sectional data."""
    return family in CROSS_SECTIONAL_FAMILIES

def get_sequence_families() -> Set[str]:
    """Get all sequence-based model families."""
    return SEQUENCE_FAMILIES.copy()

def get_cross_sectional_families() -> Set[str]:
    """Get all cross-sectional model families."""
    return CROSS_SECTIONAL_FAMILIES.copy()
