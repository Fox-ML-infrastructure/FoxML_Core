# MIT License - see LICENSE file

"""
TabCNN PyTorch Trainer
=====================

PyTorch implementation of TabCNN for tabular + sequential data.
"""

import logging
import numpy as np
from model_fun.seq_torch_base import SeqTorchTrainerBase
from models.seq_adapters import TabCNNHead

logger = logging.getLogger(__name__)

class TabCNNTrainerTorch:
    """TabCNN trainer using PyTorch."""
    
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
        Train TabCNN model on tabular + sequential data.
        
        Args:
            X_seq: (N, T, D) features (tabular + sequential)
            y_seq: (N,) targets
        """
        _, T, D = X_seq.shape
        
        # Assume first half are tabular features
        tabular_dim = D // 2
        
        model = TabCNNHead(
            input_dim=D,
            tabular_dim=tabular_dim,
            hidden_dims=self.config["hidden_dims"],
            output_dim=1,
            dropout=self.config["dropout"]
        )
        
        self.core = SeqTorchTrainerBase(model, self.config)
        self.core.train(X_seq, y_seq)
        return self

    def predict(self, X_seq):
        """Make predictions on tabular + sequential data."""
        return self.core.predict(X_seq)
