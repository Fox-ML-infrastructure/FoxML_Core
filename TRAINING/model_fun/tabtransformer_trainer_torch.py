# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
TabTransformer PyTorch Trainer
============================

PyTorch implementation of TabTransformer for tabular + sequential data.
"""

import logging
import numpy as np
from model_fun.seq_torch_base import SeqTorchTrainerBase
from models.seq_adapters import TabTransformerHead

logger = logging.getLogger(__name__)

class TabTransformerTrainerTorch:
    """TabTransformer trainer using PyTorch."""
    
    def __init__(self, config=None):
        self.config = {**{
            "d_model": 128,
            "nhead": 8,
            "num_layers": 3,
            "dropout": 0.1,
            "batch_size": 384,
            "epochs": 40,
            "lr": 2e-4,
            "num_threads": 1
        }, **(config or {})}
        self.core = None

    def train(self, X_seq, y_seq):
        """
        Train TabTransformer model on tabular + sequential data.
        
        Args:
            X_seq: (N, T, D) features (tabular + sequential)
            y_seq: (N,) targets
        """
        _, T, D = X_seq.shape
        
        # Assume first half are tabular features
        tabular_dim = D // 2
        
        model = TabTransformerHead(
            input_dim=D,
            tabular_dim=tabular_dim,
            d_model=self.config["d_model"],
            nhead=self.config["nhead"],
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