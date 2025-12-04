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
