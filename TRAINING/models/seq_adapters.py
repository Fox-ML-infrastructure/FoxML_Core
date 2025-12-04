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
Model Adapters for Sequential Data
==================================

Adapters to handle different sequence model architectures and their input/output shapes.
"""


import os
import math
import logging
from typing import Dict, List, Optional, Tuple, Any

# CRITICAL: Guard torch import to prevent libiomp5 in CPU-only children
_TORCH_DISABLED = os.getenv("TRAINER_CHILD_NO_TORCH", "0") == "1"
if not _TORCH_DISABLED:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
else:
    # Sentinel values so module loads but torch is None
    torch = None
    nn = None
    F = None

logger = logging.getLogger(__name__)

class CNN1DHead(nn.Module):
    """
    CNN1D head for sequence data.
    
    Expects input shape (B, T, F) and converts to (B, F, T) for Conv1D.
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 64], 
                 output_dim: int = 1, dropout: float = 0.3):
        """
        Initialize CNN1D head.
        
        Args:
            input_dim: Number of input features (F)
            hidden_dims: List of hidden dimensions
            output_dim: Output dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Build CNN layers
        layers = []
        in_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.extend([
                nn.Conv1d(in_dim, hidden_dim, kernel_size=5, padding=2),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim
        
        # Global average pooling
        layers.append(nn.AdaptiveAvgPool1d(1))
        
        self.cnn = nn.Sequential(*layers)
        
        # Final linear layer
        self.fc = nn.Linear(hidden_dims[-1], output_dim)
        
        logger.info(f"CNN1DHead created: input_dim={input_dim}, hidden_dims={hidden_dims}, output_dim={output_dim}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, T, F)
        
        Returns:
            Output tensor of shape (B, output_dim)
        """
        # Convert (B, T, F) to (B, F, T) for Conv1D
        x = x.transpose(1, 2)  # [B, F, T]
        
        # Apply CNN
        x = self.cnn(x)  # [B, hidden_dim, 1]
        x = x.squeeze(-1)  # [B, hidden_dim]
        
        # Final linear layer
        x = self.fc(x)  # [B, output_dim]
        
        return x

class LSTMHead(nn.Module):
    """
    LSTM head for sequence data.
    
    Expects input shape (B, T, F).
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, 
                 num_layers: int = 2, output_dim: int = 1, 
                 dropout: float = 0.3, bidirectional: bool = False):
        """
        Initialize LSTM head.
        
        Args:
            input_dim: Number of input features (F)
            hidden_dim: LSTM hidden dimension
            num_layers: Number of LSTM layers
            output_dim: Output dimension
            dropout: Dropout rate
            bidirectional: Whether to use bidirectional LSTM
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Calculate LSTM output dimension
        lstm_output_dim = hidden_dim * (2 if bidirectional else 1)
        
        # Final layers
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        logger.info(f"LSTMHead created: input_dim={input_dim}, hidden_dim={hidden_dim}, "
                   f"num_layers={num_layers}, bidirectional={bidirectional}, output_dim={output_dim}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, T, F)
        
        Returns:
            Output tensor of shape (B, output_dim)
        """
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)  # lstm_out: [B, T, hidden_dim]
        
        # Use the last output
        x = lstm_out[:, -1, :]  # [B, hidden_dim]
        
        # Final layers
        x = self.fc(x)  # [B, output_dim]
        
        return x

class TransformerHead(nn.Module):
    """
    Transformer head for sequence data.
    
    Expects input shape (B, T, F).
    """
    
    def __init__(self, input_dim: int, d_model: int = 128, 
                 nhead: int = 8, num_layers: int = 3, 
                 output_dim: int = 1, dropout: float = 0.1):
        """
        Initialize Transformer head.
        
        Args:
            input_dim: Number of input features (F)
            d_model: Transformer model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            output_dim: Output dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_dim)
        )
        
        logger.info(f"TransformerHead created: input_dim={input_dim}, d_model={d_model}, "
                   f"nhead={nhead}, num_layers={num_layers}, output_dim={output_dim}")
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, T, F)
            mask: Optional attention mask of shape (B, T)
        
        Returns:
            Output tensor of shape (B, output_dim)
        """
        # Input projection
        x = self.input_proj(x)  # [B, T, d_model]
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Transformer forward pass
        x = self.transformer(x, src_key_padding_mask=mask)  # [B, T, d_model]
        
        # Global average pooling
        if mask is not None:
            # Masked average pooling
            mask_expanded = mask.unsqueeze(-1).expand_as(x)  # [B, T, d_model]
            x_masked = x * mask_expanded
            x = x_masked.sum(dim=1) / mask_expanded.sum(dim=1)  # [B, d_model]
        else:
            x = x.mean(dim=1)  # [B, d_model]
        
        # Output projection
        x = self.output_proj(x)  # [B, output_dim]
        
        return x

class PositionalEncoding(nn.Module):
    """
    Positional encoding for Transformer.
    """
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)

class TabLSTMHead(nn.Module):
    """
    TabLSTM head combining tabular and sequential processing.
    
    Expects input shape (B, T, F) with both tabular and sequential features.
    """
    
    def __init__(self, input_dim: int, tabular_dim: int, 
                 lstm_hidden: int = 128, output_dim: int = 1,
                 dropout: float = 0.3):
        """
        Initialize TabLSTM head.
        
        Args:
            input_dim: Total input features (F)
            tabular_dim: Number of tabular features
            lstm_hidden: LSTM hidden dimension
            output_dim: Output dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.tabular_dim = tabular_dim
        self.sequential_dim = input_dim - tabular_dim
        
        # Tabular processing
        self.tabular_fc = nn.Sequential(
            nn.Linear(tabular_dim, lstm_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Sequential processing (LSTM)
        self.lstm = nn.LSTM(
            input_size=self.sequential_dim,
            hidden_size=lstm_hidden // 2,
            num_layers=2,
            dropout=dropout,
            batch_first=True
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(lstm_hidden, lstm_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden // 2, output_dim)
        )
        
        logger.info(f"TabLSTMHead created: input_dim={input_dim}, tabular_dim={tabular_dim}, "
                   f"sequential_dim={self.sequential_dim}, output_dim={output_dim}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, T, F)
        
        Returns:
            Output tensor of shape (B, output_dim)
        """
        # Split tabular and sequential features
        x_tab = x[:, :, :self.tabular_dim]  # [B, T, tabular_dim]
        x_seq = x[:, :, self.tabular_dim:]  # [B, T, sequential_dim]
        
        # Process tabular features (use last timestep)
        x_tab = x_tab[:, -1, :]  # [B, tabular_dim]
        x_tab = self.tabular_fc(x_tab)  # [B, lstm_hidden//2]
        
        # Process sequential features
        lstm_out, _ = self.lstm(x_seq)  # [B, T, lstm_hidden//2]
        x_seq = lstm_out[:, -1, :]  # [B, lstm_hidden//2]
        
        # Fusion
        x = torch.cat([x_tab, x_seq], dim=1)  # [B, lstm_hidden]
        x = self.fusion(x)  # [B, output_dim]
        
        return x

class TabTransformerHead(nn.Module):
    """
    TabTransformer head combining tabular and sequential processing.
    
    Expects input shape (B, T, F) with both tabular and sequential features.
    """
    
    def __init__(self, input_dim: int, tabular_dim: int,
                 d_model: int = 128, nhead: int = 8, 
                 num_layers: int = 3, output_dim: int = 1,
                 dropout: float = 0.1):
        """
        Initialize TabTransformer head.
        
        Args:
            input_dim: Total input features (F)
            tabular_dim: Number of tabular features
            d_model: Transformer model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            output_dim: Output dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.tabular_dim = tabular_dim
        self.sequential_dim = input_dim - tabular_dim
        
        # Guard head dims for TabTransformer so odd nhead doesn't explode
        nhead_tab = max(1, nhead // 2)
        d_half = max(8, (d_model // 2))
        
        # Tabular processing
        self.tabular_fc = nn.Sequential(
            nn.Linear(tabular_dim, d_half),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Sequential processing (Transformer)
        self.seq_proj = nn.Linear(self.sequential_dim, d_half)
        self.pos_encoding = PositionalEncoding(d_half, dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_half,
            nhead=nhead_tab,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_dim)
        )
        
        logger.info(f"TabTransformerHead created: input_dim={input_dim}, tabular_dim={tabular_dim}, "
                   f"sequential_dim={self.sequential_dim}, output_dim={output_dim}")
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, T, F)
            mask: Optional attention mask
        
        Returns:
            Output tensor of shape (B, output_dim)
        """
        # Split tabular and sequential features
        x_tab = x[:, :, :self.tabular_dim]  # [B, T, tabular_dim]
        x_seq = x[:, :, self.tabular_dim:]  # [B, T, sequential_dim]
        
        # Process tabular features (use last timestep)
        x_tab = x_tab[:, -1, :]  # [B, tabular_dim]
        x_tab = self.tabular_fc(x_tab)  # [B, d_model//2]
        
        # Process sequential features
        x_seq = self.seq_proj(x_seq)  # [B, T, d_model//2]
        x_seq = self.pos_encoding(x_seq)
        x_seq = self.transformer(x_seq, src_key_padding_mask=mask)  # [B, T, d_model//2]
        x_seq = x_seq.mean(dim=1)  # [B, d_model//2]
        
        # Fusion
        x = torch.cat([x_tab, x_seq], dim=1)  # [B, d_model]
        x = self.fusion(x)  # [B, output_dim]
        
        return x

# Import math for positional encoding
import math
