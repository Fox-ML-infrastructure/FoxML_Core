# MIT License - see LICENSE file

"""
CNN1D PyTorch Trainer
=====================

PyTorch implementation of CNN1D for sequential data.
"""

import logging
import numpy as np
from model_fun.seq_torch_base import SeqTorchTrainerBase
from models.seq_adapters import CNN1DHead

logger = logging.getLogger(__name__)

class CNN1DTrainerTorch:
    """CNN1D trainer using PyTorch."""
    
    def __init__(self, config=None):
        self.config = {**{
            "batch_size": 512,
            "epochs": 40,
            "lr": 1e-3,
            "num_threads": 1,
            "hidden_dims": [128, 64],
            "dropout": 0.2
        }, **(config or {})}
        self.core = None

    def train(self, X_seq, y_seq):
        """
        Train CNN1D model on sequential data.
        
        Args:
            X_seq: (N, T, D) sequential features
            y_seq: (N,) targets
        """
        _, T, D = X_seq.shape
        
        model = CNN1DHead(
            input_dim=D,
            hidden_dims=self.config["hidden_dims"],
            output_dim=1,
            dropout=self.config["dropout"]
        )
        
        self.core = SeqTorchTrainerBase(model, self.config)
        self.core.train(X_seq, y_seq)
        return self

    def predict(self, X_seq):
        """Make predictions on sequential data."""
        return self.core.predict(X_seq)
