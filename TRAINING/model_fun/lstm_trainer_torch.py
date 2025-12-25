# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial

"""
LSTM PyTorch Trainer
===================

PyTorch implementation of LSTM for sequential data.
"""

import logging
import numpy as np
from model_fun.seq_torch_base import SeqTorchTrainerBase
from models.seq_adapters import LSTMHead

logger = logging.getLogger(__name__)

class LSTMTrainerTorch:
    """LSTM trainer using PyTorch."""
    
    def __init__(self, config=None):
        self.config = {**{
            "lookback": 64,
            "hidden_dim": 128,
            "num_layers": 2,
            "dropout": 0.1,
            "batch_size": 512,
            "epochs": 50,
            "lr": 1e-3,
            "num_threads": 1
        }, **(config or {})}
        self.core = None

    def train(self, X_seq, y_seq):
        """
        Train LSTM model on sequential data.
        
        Args:
            X_seq: (N, T, D) sequential features
            y_seq: (N,) targets
        """
        _, T, D = X_seq.shape
        
        model = LSTMHead(
            input_dim=D,
            hidden_dim=self.config["hidden_dim"],
            num_layers=self.config["num_layers"],
            dropout=self.config["dropout"],
            output_dim=1
        )
        
        self.core = SeqTorchTrainerBase(model, self.config)
        self.core.train(X_seq, y_seq)
        return self

    def predict(self, X_seq):
        """Make predictions on sequential data."""
        return self.core.predict(X_seq)
