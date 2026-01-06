# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial

"""
PyTorch Dataset for Sequential Data
===================================

Dataset classes for sequence-based models (CNN1D, LSTM, Transformer, etc.).
Handles (N, T, F) sequences with proper collation and masking.
"""


import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger(__name__)

class SeqDataset(Dataset):
    """
    PyTorch Dataset for sequential data.
    
    Handles (N, T, F) sequences where:
    - N: number of samples
    - T: sequence length (lookback bars)
    - F: number of features
    """
    
    def __init__(self, X: np.ndarray, y: np.ndarray, 
                 ts: Optional[np.ndarray] = None, 
                 syms: Optional[np.ndarray] = None):
        """
        Initialize sequential dataset.
        
        Args:
            X: [N, T, F] sequences
            y: [N] labels
            ts: [N] timestamps (optional)
            syms: [N] symbols (optional)
        """
        assert X.ndim == 3, f"X should be 3D [N, T, F], got shape {X.shape}"
        assert y.ndim == 1, f"y should be 1D [N], got shape {y.shape}"
        assert len(X) == len(y), f"X and y length mismatch: {len(X)} vs {len(y)}"
        
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float().view(-1, 1)  # [N, 1]
        self.ts = ts
        self.syms = syms
        
        logger.info(f"SeqDataset created: {len(self)} samples, shape {X.shape}")
    
    def __len__(self) -> int:
        return self.X.shape[0]
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample."""
        sample = {
            "x": self.X[idx],      # [T, F]
            "y": self.y[idx],      # [1]
        }
        
        if self.ts is not None:
            # Convert datetime64 to string to avoid DataLoader issues
            sample["ts"] = str(self.ts[idx])
        if self.syms is not None:
            sample["sym"] = str(self.syms[idx])
            
        return sample

class VariableSeqDataset(Dataset):
    """
    PyTorch Dataset for variable-length sequences.
    
    Handles sequences of different lengths with padding and masking.
    """
    
    def __init__(self, sequences: List[np.ndarray], labels: List[float],
                 timestamps: Optional[List] = None,
                 symbols: Optional[List] = None):
        """
        Initialize variable-length sequential dataset.
        
        Args:
            sequences: List of [T_i, F] arrays of different lengths
            labels: List of scalar labels
            timestamps: List of timestamps (optional)
            symbols: List of symbols (optional)
        """
        assert len(sequences) == len(labels), "Sequences and labels length mismatch"
        
        self.sequences = [torch.from_numpy(seq).float() for seq in sequences]
        self.labels = torch.tensor(labels, dtype=torch.float32).view(-1, 1)
        self.timestamps = timestamps
        self.symbols = symbols
        
        logger.info(f"VariableSeqDataset created: {len(self)} samples")
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample."""
        sample = {
            "x": self.sequences[idx],  # [T_i, F]
            "y": self.labels[idx],     # [1]
        }
        
        if self.timestamps is not None:
            sample["ts"] = self.timestamps[idx]
        if self.symbols is not None:
            sample["sym"] = self.symbols[idx]
            
        return sample

def pad_collate(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Collate function for variable-length sequences.
    
    Pads sequences to the same length and creates attention masks.
    
    Args:
        batch: List of samples from VariableSeqDataset
    
    Returns:
        Batched tensors with padding and masks
    """
    # Extract sequences and labels
    xs = [sample["x"] for sample in batch]  # [T_i, F]
    ys = torch.stack([sample["y"] for sample in batch], dim=0)  # [B, 1]
    
    # Find maximum sequence length
    T_max = max(x.shape[0] for x in xs)
    F = xs[0].shape[1]
    B = len(xs)
    
    # Create padded tensor and mask
    X_padded = torch.zeros(B, T_max, F, dtype=xs[0].dtype)
    mask = torch.zeros(B, T_max, dtype=torch.bool)
    
    for i, x in enumerate(xs):
        T_i = x.shape[0]
        X_padded[i, -T_i:, :] = x  # Right-align sequences
        mask[i, -T_i:] = True      # True for actual data, False for padding
    
    result = {
        "x": X_padded,      # [B, T_max, F]
        "mask": mask,       # [B, T_max]
        "y": ys,            # [B, 1]
    }
    
    # Add optional metadata
    if "ts" in batch[0]:
        result["ts"] = [sample["ts"] for sample in batch]
    if "sym" in batch[0]:
        result["sym"] = [sample["sym"] for sample in batch]
    
    return result

def create_seq_dataloader(dataset: Dataset, 
                         batch_size: int = 32,
                         shuffle: bool = True,
                         num_workers: int = 0,
                         collate_fn: Optional[callable] = None,
                         pin_memory: bool = True) -> DataLoader:
    """
    Create DataLoader for sequential data with deterministic settings.
    
    Args:
        dataset: Sequential dataset
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of worker processes (0 for deterministic)
        collate_fn: Custom collate function
        pin_memory: Whether to pin memory
    
    Returns:
        DataLoader configured for sequential data
    """
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=False  # For deterministic behavior
    )

class SeqDataModule:
    """
    Data module for sequential training with train/val/test splits.
    """
    
    def __init__(self, X: np.ndarray, y: np.ndarray,
                 ts: Optional[np.ndarray] = None,
                 syms: Optional[np.ndarray] = None,
                 train_ratio: float = 0.7,
                 val_ratio: float = 0.15,
                 test_ratio: float = 0.15,
                 batch_size: int = 32,
                 lookback_T: int = 60):
        """
        Initialize sequential data module.
        
        Args:
            X: [N, T, F] sequences
            y: [N] labels
            ts: [N] timestamps
            syms: [N] symbols
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
            batch_size: Batch size
            lookback_T: Sequence length
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
        
        self.lookback_T = lookback_T
        self.batch_size = batch_size
        
        # Time-based split (chronological)
        N = len(X)
        train_end = int(N * train_ratio)
        val_end = int(N * (train_ratio + val_ratio))
        
        # Split data
        self.X_train = X[:train_end]
        self.y_train = y[:train_end]
        self.X_val = X[train_end:val_end]
        self.y_val = y[train_end:val_end]
        self.X_test = X[val_end:]
        self.y_test = y[val_end:]
        
        # Split metadata
        if ts is not None:
            self.ts_train = ts[:train_end]
            self.ts_val = ts[train_end:val_end]
            self.ts_test = ts[val_end:]
        else:
            self.ts_train = self.ts_val = self.ts_test = None
            
        if syms is not None:
            self.syms_train = syms[:train_end]
            self.syms_val = syms[train_end:val_end]
            self.syms_test = syms[val_end:]
        else:
            self.syms_train = self.syms_val = self.syms_test = None
        
        logger.info(f"Data split: train={len(self.X_train)}, val={len(self.X_val)}, test={len(self.X_test)}")
    
    def get_train_dataloader(self, shuffle: bool = True) -> DataLoader:
        """Get training dataloader."""
        dataset = SeqDataset(self.X_train, self.y_train, self.ts_train, self.syms_train)
        return create_seq_dataloader(dataset, self.batch_size, shuffle)
    
    def get_val_dataloader(self, shuffle: bool = False) -> DataLoader:
        """Get validation dataloader."""
        dataset = SeqDataset(self.X_val, self.y_val, self.ts_val, self.syms_val)
        return create_seq_dataloader(dataset, self.batch_size, shuffle)
    
    def get_test_dataloader(self, shuffle: bool = False) -> DataLoader:
        """Get test dataloader."""
        dataset = SeqDataset(self.X_test, self.y_test, self.ts_test, self.syms_test)
        return create_seq_dataloader(dataset, self.batch_size, shuffle)
