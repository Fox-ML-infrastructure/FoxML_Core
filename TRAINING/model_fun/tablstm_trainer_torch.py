# MIT License - see LICENSE file

"""
TabLSTM PyTorch Trainer
======================

PyTorch implementation of TabLSTM for tabular + sequential data.
"""

import logging
import numpy as np
from model_fun.seq_torch_base import SeqTorchTrainerBase
from models.seq_adapters import TabLSTMHead

logger = logging.getLogger(__name__)

class TabLSTMTrainerTorch:
    """TabLSTM trainer using PyTorch."""
    
    def __init__(self, config=None):
        self.config = {**{
            "batch_size": 512,
            "epochs": 50,
            "lr": 1e-3,
            "num_threads": 1,
            "hidden_dim": 128,
            "num_layers": 2,
            "dropout": 0.1
        }, **(config or {})}
        self.core = None

    def train(self, X_seq, y_seq):
        """
        Train TabLSTM model on tabular + sequential data.
        
        Args:
            X_seq: (N, T, D) features (tabular + sequential)
            y_seq: (N,) targets
        """
        _, T, D = X_seq.shape
        
        # Assume first half are tabular features
        tabular_dim = D // 2
        
        model = TabLSTMHead(
            input_dim=D,
            tabular_dim=tabular_dim,
            hidden_dim=self.config["hidden_dim"],
            num_layers=self.config["num_layers"],
            dropout=self.config["dropout"],
            output_dim=1
        )
        
        self.core = SeqTorchTrainerBase(model, self.config)
        self.core.train(X_seq, y_seq)
        return self

    def predict(self, X_seq):
        """Make predictions on tabular + sequential data."""
        return self.core.predict(X_seq)
