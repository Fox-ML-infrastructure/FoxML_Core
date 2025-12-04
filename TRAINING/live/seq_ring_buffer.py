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
Live Rolling Buffers for Sequential Models
==========================================

Ring buffers for maintaining rolling windows of sequential data during live inference.
"""


import numpy as np
import torch
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import time

logger = logging.getLogger(__name__)

class SeqRingBuffer:
    """
    Ring buffer for maintaining rolling sequences of features.
    
    Maintains a (T, F) buffer where T is the sequence length and F is the number of features.
    """
    
    def __init__(self, T: int, F: int, ttl_seconds: float = 300.0):
        """
        Initialize ring buffer.
        
        Args:
            T: Sequence length (lookback bars)
            F: Number of features
            ttl_seconds: Time-to-live for data validity
        """
        self.T = T
        self.F = F
        self.ttl_seconds = ttl_seconds
        
        # Initialize buffer
        self.buffer = np.zeros((T, F), dtype=np.float32)
        self.fill_count = 0
        self.last_update = None
        self.valid_mask = np.zeros(T, dtype=bool)
        
        logger.debug(f"SeqRingBuffer initialized: T={T}, F={F}, ttl={ttl_seconds}s")
    
    def push(self, x_row: np.ndarray, timestamp: Optional[datetime] = None) -> bool:
        """
        Push new feature row into the buffer.
        
        Args:
            x_row: Feature row of shape (F,)
            timestamp: Optional timestamp for TTL tracking
        
        Returns:
            True if successful, False if invalid
        """
        # Validate input
        if x_row.shape != (self.F,):
            logger.error(f"Invalid feature shape: {x_row.shape}, expected ({self.F},)")
            return False
        
        # Check for NaN or infinite values
        if np.isnan(x_row).any() or np.isinf(x_row).any():
            logger.warning("Invalid features (NaN/inf), skipping update")
            return False
        
        # Update timestamp
        if timestamp is None:
            timestamp = datetime.now()
        self.last_update = timestamp
        
        # Roll buffer and add new data
        self.buffer = np.roll(self.buffer, -1, axis=0)
        self.buffer[-1, :] = x_row
        
        # Update fill count and validity
        self.fill_count = min(self.fill_count + 1, self.T)
        self.valid_mask = np.roll(self.valid_mask, -1)
        self.valid_mask[-1] = True
        
        logger.debug(f"Pushed features: fill_count={self.fill_count}/{self.T}")
        return True
    
    def ready(self) -> bool:
        """
        Check if buffer is ready for inference.
        
        Returns:
            True if buffer is full and data is fresh
        """
        # Check if buffer is full
        if self.fill_count < self.T:
            return False
        
        # Check TTL
        if self.last_update is None:
            return False
        
        age_seconds = (datetime.now() - self.last_update).total_seconds()
        if age_seconds > self.ttl_seconds:
            logger.warning(f"Buffer data too old: {age_seconds:.1f}s > {self.ttl_seconds}s")
            return False
        
        # Check for any invalid data in the buffer
        if np.isnan(self.buffer).any() or np.isinf(self.buffer).any():
            logger.warning("Buffer contains invalid data (NaN/inf)")
            return False
        
        return True
    
    def view(self) -> np.ndarray:
        """
        Get current buffer view.
        
        Returns:
            Buffer copy of shape (T, F)
        """
        return self.buffer.copy()
    
    def get_sequence(self) -> torch.Tensor:
        """
        Get sequence as PyTorch tensor.
        
        Returns:
            Tensor of shape (1, T, F) for inference
        """
        if not self.ready():
            raise RuntimeError("Buffer not ready for inference")
        
        # Convert to tensor and add batch dimension
        sequence = torch.from_numpy(self.buffer).float().unsqueeze(0)  # [1, T, F]
        return sequence
    
    def reset(self):
        """Reset buffer to empty state."""
        self.buffer.fill(0.0)
        self.fill_count = 0
        self.last_update = None
        self.valid_mask.fill(False)
        logger.debug("Buffer reset")
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get buffer status information.
        
        Returns:
            Status dictionary
        """
        age_seconds = None
        if self.last_update is not None:
            age_seconds = (datetime.now() - self.last_update).total_seconds()
        
        return {
            'fill_count': self.fill_count,
            'capacity': self.T,
            'is_ready': self.ready(),
            'last_update': self.last_update,
            'age_seconds': age_seconds,
            'ttl_seconds': self.ttl_seconds,
            'has_nan': np.isnan(self.buffer).any(),
            'has_inf': np.isinf(self.buffer).any()
        }

class SeqBufferManager:
    """
    Manager for multiple symbol ring buffers.
    """
    
    def __init__(self, T: int, F: int, ttl_seconds: float = 300.0):
        """
        Initialize buffer manager.
        
        Args:
            T: Sequence length
            F: Number of features
            ttl_seconds: TTL for data validity
        """
        self.T = T
        self.F = F
        self.ttl_seconds = ttl_seconds
        self.buffers: Dict[str, SeqRingBuffer] = {}
        
        logger.info(f"SeqBufferManager initialized: T={T}, F={F}, ttl={ttl_seconds}s")
    
    def get_buffer(self, symbol: str) -> SeqRingBuffer:
        """
        Get or create buffer for a symbol.
        
        Args:
            symbol: Symbol name
        
        Returns:
            Ring buffer for the symbol
        """
        if symbol not in self.buffers:
            self.buffers[symbol] = SeqRingBuffer(self.T, self.F, self.ttl_seconds)
            logger.debug(f"Created buffer for symbol: {symbol}")
        
        return self.buffers[symbol]
    
    def push_features(self, symbol: str, features: np.ndarray, 
                     timestamp: Optional[datetime] = None) -> bool:
        """
        Push features for a symbol.
        
        Args:
            symbol: Symbol name
            features: Feature array
            timestamp: Optional timestamp
        
        Returns:
            True if successful
        """
        buffer = self.get_buffer(symbol)
        return buffer.push(features, timestamp)
    
    def is_ready(self, symbol: str) -> bool:
        """
        Check if symbol buffer is ready.
        
        Args:
            symbol: Symbol name
        
        Returns:
            True if ready
        """
        if symbol not in self.buffers:
            return False
        
        return self.buffers[symbol].ready()
    
    def get_sequence(self, symbol: str) -> Optional[torch.Tensor]:
        """
        Get sequence for a symbol.
        
        Args:
            symbol: Symbol name
        
        Returns:
            Sequence tensor or None if not ready
        """
        if not self.is_ready(symbol):
            return None
        
        return self.buffers[symbol].get_sequence()
    
    def get_ready_symbols(self) -> List[str]:
        """
        Get list of symbols with ready buffers.
        
        Returns:
            List of ready symbols
        """
        return [symbol for symbol, buffer in self.buffers.items() if buffer.ready()]
    
    def get_status(self) -> Dict[str, Dict[str, Any]]:
        """
        Get status of all buffers.
        
        Returns:
            Status dictionary for all symbols
        """
        return {symbol: buffer.get_status() for symbol, buffer in self.buffers.items()}
    
    def reset_symbol(self, symbol: str):
        """Reset buffer for a symbol."""
        if symbol in self.buffers:
            self.buffers[symbol].reset()
    
    def reset_all(self):
        """Reset all buffers."""
        for buffer in self.buffers.values():
            buffer.reset()
        logger.info("All buffers reset")

class LiveSeqInference:
    """
    Live inference handler for sequential models.
    """
    
    def __init__(self, model, buffer_manager: SeqBufferManager, device: str = 'cpu'):
        """
        Initialize live inference handler.
        
        Args:
            model: Trained sequential model
            buffer_manager: Buffer manager
            device: Device for inference
        """
        self.model = model
        self.buffer_manager = buffer_manager
        self.device = device
        
        # Move model to device
        if hasattr(model, 'to'):
            self.model = model.to(device)
        
        logger.info(f"LiveSeqInference initialized on device: {device}")
    
    def predict(self, symbol: str) -> Optional[float]:
        """
        Make prediction for a symbol.
        
        Args:
            symbol: Symbol name
        
        Returns:
            Prediction or None if not ready
        """
        # Get sequence
        sequence = self.buffer_manager.get_sequence(symbol)
        if sequence is None:
            return None
        
        try:
            # Move to device and predict
            sequence = sequence.to(self.device)
            with torch.no_grad():
                prediction = self.model(sequence)
                if isinstance(prediction, torch.Tensor):
                    prediction = prediction.cpu().numpy()
                    if prediction.ndim > 1:
                        prediction = prediction.squeeze()
                    return float(prediction)
                return float(prediction)
        
        except Exception as e:
            logger.error(f"Prediction failed for {symbol}: {e}")
            return None
    
    def predict_batch(self, symbols: List[str]) -> Dict[str, Optional[float]]:
        """
        Make predictions for multiple symbols.
        
        Args:
            symbols: List of symbol names
        
        Returns:
            Dictionary of {symbol: prediction}
        """
        results = {}
        for symbol in symbols:
            results[symbol] = self.predict(symbol)
        return results
    
    def predict_ready_symbols(self) -> Dict[str, float]:
        """
        Make predictions for all ready symbols.
        
        Returns:
            Dictionary of {symbol: prediction} for ready symbols
        """
        ready_symbols = self.buffer_manager.get_ready_symbols()
        return self.predict_batch(ready_symbols)
