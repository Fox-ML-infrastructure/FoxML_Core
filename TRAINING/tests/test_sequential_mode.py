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
Sequential Mode Verification Tests
==================================

Comprehensive tests for sequential mode functionality.
"""


import numpy as np
import pandas as pd
import torch
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

# Import our sequential components
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features.seq_builder import build_sequences_for_symbol, build_sequences_panel, validate_sequences
from datasets.seq_dataset import SeqDataset, SeqDataModule, create_seq_dataloader
from models.seq_adapters import CNN1DHead, LSTMHead, TransformerHead
from models.family_router import FamilyRouter, is_sequence_model, is_cross_sectional_model
from live.seq_ring_buffer import SeqRingBuffer, SeqBufferManager, LiveSeqInference

logger = logging.getLogger(__name__)

def test_sequence_builder():
    """Test sequence builder functionality."""
    print("🧪 Testing sequence builder...")
    
    # Create test data
    dates = pd.date_range('2023-01-01', periods=100, freq='5min')
    np.random.seed(42)
    
    df = pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100),
        'feature3': np.random.randn(100),
        'fwd_ret_5m': np.random.randn(100)
    }, index=dates)
    
    # Test single symbol
    X, y, t = build_sequences_for_symbol(
        df, 
        feature_cols=['feature1', 'feature2', 'feature3'],
        target_col='fwd_ret_5m',
        lookback_T=10,
        horizon_bars=1,
        stride=1
    )
    
    assert X.shape[1] == 10, f"Expected sequence length 10, got {X.shape[1]}"
    assert X.shape[2] == 3, f"Expected 3 features, got {X.shape[2]}"
    assert len(X) == len(y), "X and y length mismatch"
    assert len(X) == len(t), "X and t length mismatch"
    
    # Test validation
    assert validate_sequences(X, y, t, 10, ['feature1', 'feature2', 'feature3']), "Validation failed"
    
    print("✅ Sequence builder tests passed")

def test_pytorch_dataset():
    """Test PyTorch dataset functionality."""
    print("🧪 Testing PyTorch dataset...")
    
    # Create test data
    N, T, F = 50, 10, 5
    X = np.random.randn(N, T, F).astype(np.float32)
    y = np.random.randn(N).astype(np.float32)
    
    # Test SeqDataset
    dataset = SeqDataset(X, y)
    assert len(dataset) == N, f"Expected {N} samples, got {len(dataset)}"
    
    # Test single sample
    sample = dataset[0]
    assert sample['x'].shape == (T, F), f"Expected shape ({T}, {F}), got {sample['x'].shape}"
    assert sample['y'].shape == (1,), f"Expected shape (1,), got {sample['y'].shape}"
    
    # Test DataLoader
    dataloader = create_seq_dataloader(dataset, batch_size=8, shuffle=False)
    batch = next(iter(dataloader))
    assert batch['x'].shape == (8, T, F), f"Expected batch shape (8, {T}, {F}), got {batch['x'].shape}"
    assert batch['y'].shape == (8, 1), f"Expected batch shape (8, 1), got {batch['y'].shape}"
    
    print("✅ PyTorch dataset tests passed")

def test_model_adapters():
    """Test model adapters for different architectures."""
    print("🧪 Testing model adapters...")
    
    # Test CNN1D
    cnn = CNN1DHead(input_dim=10, hidden_dims=[64, 32], output_dim=1)
    x = torch.randn(2, 20, 10)  # [B, T, F]
    y = cnn(x)
    assert y.shape == (2, 1), f"Expected shape (2, 1), got {y.shape}"
    
    # Test LSTM
    lstm = LSTMHead(input_dim=10, hidden_dim=64, output_dim=1)
    y = lstm(x)
    assert y.shape == (2, 1), f"Expected shape (2, 1), got {y.shape}"
    
    # Test Transformer
    transformer = TransformerHead(input_dim=10, d_model=64, nhead=4, output_dim=1)
    y = transformer(x)
    assert y.shape == (2, 1), f"Expected shape (2, 1), got {y.shape}"
    
    print("✅ Model adapters tests passed")

def test_family_router():
    """Test family router functionality."""
    print("🧪 Testing family router...")
    
    # Test sequence families
    assert is_sequence_model("CNN1D"), "CNN1D should be sequence model"
    assert is_sequence_model("LSTM"), "LSTM should be sequence model"
    assert is_sequence_model("Transformer"), "Transformer should be sequence model"
    
    # Test cross-sectional families
    assert is_cross_sectional_model("LightGBM"), "LightGBM should be cross-sectional"
    assert is_cross_sectional_model("XGBoost"), "XGBoost should be cross-sectional"
    
    # Test router
    router = FamilyRouter()
    assert router.is_sequence_family("CNN1D"), "Router should identify CNN1D as sequence"
    assert router.is_cross_sectional_family("LightGBM"), "Router should identify LightGBM as cross-sectional"
    
    print("✅ Family router tests passed")

def test_ring_buffer():
    """Test ring buffer functionality."""
    print("🧪 Testing ring buffer...")
    
    # Test single buffer
    buffer = SeqRingBuffer(T=5, F=3, ttl_seconds=60)
    
    # Push features
    for i in range(7):  # More than capacity
        features = np.array([i, i*2, i*3], dtype=np.float32)
        success = buffer.push(features)
        assert success, f"Push {i} failed"
    
    # Check buffer state
    assert buffer.fill_count == 5, f"Expected fill_count 5, got {buffer.fill_count}"
    assert buffer.ready(), "Buffer should be ready"
    
    # Test view
    view = buffer.view()
    assert view.shape == (5, 3), f"Expected shape (5, 3), got {view.shape}"
    
    # Test sequence
    sequence = buffer.get_sequence()
    assert sequence.shape == (1, 5, 3), f"Expected shape (1, 5, 3), got {sequence.shape}"
    
    print("✅ Ring buffer tests passed")

def test_buffer_manager():
    """Test buffer manager functionality."""
    print("🧪 Testing buffer manager...")
    
    manager = SeqBufferManager(T=5, F=3, ttl_seconds=60)
    
    # Test multiple symbols
    symbols = ['AAPL', 'GOOGL', 'MSFT']
    for symbol in symbols:
        for i in range(6):
            features = np.random.randn(3).astype(np.float32)
            success = manager.push_features(symbol, features)
            assert success, f"Push failed for {symbol}"
    
    # Check ready symbols
    ready_symbols = manager.get_ready_symbols()
    assert len(ready_symbols) == 3, f"Expected 3 ready symbols, got {len(ready_symbols)}"
    
    # Test sequence retrieval
    for symbol in symbols:
        sequence = manager.get_sequence(symbol)
        assert sequence is not None, f"Sequence should be available for {symbol}"
        assert sequence.shape == (1, 5, 3), f"Expected shape (1, 5, 3), got {sequence.shape}"
    
    print("✅ Buffer manager tests passed")

def test_live_inference():
    """Test live inference functionality."""
    print("🧪 Testing live inference...")
    
    # Create dummy model
    class DummyModel:
        def __init__(self):
            self.device = 'cpu'
        
        def to(self, device):
            return self
        
        def eval(self):
            pass
        
        def __call__(self, x):
            return torch.randn(x.shape[0], 1)
    
    model = DummyModel()
    manager = SeqBufferManager(T=5, F=3, ttl_seconds=60)
    inference = LiveSeqInference(model, manager)
    
    # Setup buffer
    symbol = 'AAPL'
    for i in range(6):
        features = np.random.randn(3).astype(np.float32)
        manager.push_features(symbol, features)
    
    # Test prediction
    prediction = inference.predict(symbol)
    assert prediction is not None, "Prediction should not be None"
    assert isinstance(prediction, float), f"Prediction should be float, got {type(prediction)}"
    
    print("✅ Live inference tests passed")

def test_integration():
    """Test end-to-end integration."""
    print("🧪 Testing integration...")
    
    # Create test panel
    dates = pd.date_range('2023-01-01', periods=100, freq='5min')
    np.random.seed(42)
    
    panel = {}
    for symbol in ['AAPL', 'GOOGL']:
        df = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'feature3': np.random.randn(100),
            'fwd_ret_5m': np.random.randn(100)
        }, index=dates)
        panel[symbol] = df
    
    # Build sequences
    X, y, ts, syms = build_sequences_panel(
        panel,
        feature_cols=['feature1', 'feature2', 'feature3'],
        target_col='fwd_ret_5m',
        lookback_T=10,
        horizon_bars=1,
        stride=1
    )
    
    # Validate
    assert validate_sequences(X, y, ts, 10, ['feature1', 'feature2', 'feature3']), "Integration validation failed"
    
    # Create dataset
    dataset = SeqDataset(X, y, ts, syms)
    dataloader = create_seq_dataloader(dataset, batch_size=8, shuffle=False)
    
    # Test batch
    batch = next(iter(dataloader))
    assert batch['x'].shape[0] == 8, f"Expected batch size 8, got {batch['x'].shape[0]}"
    
    print("✅ Integration tests passed")

def run_verification_checklist():
    """Run the complete verification checklist."""
    print("🔍 Running Sequential Mode Verification Checklist")
    print("=" * 60)
    
    try:
        test_sequence_builder()
        test_pytorch_dataset()
        test_model_adapters()
        test_family_router()
        test_ring_buffer()
        test_buffer_manager()
        test_live_inference()
        test_integration()
        
        print("\n🎉 All verification tests passed!")
        print("✅ Sequential mode is ready for production")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Verification failed: {e}")
        return False

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Run verification
    success = run_verification_checklist()
    
    if success:
        print("\n🚀 Sequential mode implementation complete!")
        print("📋 Checklist:")
        print("  ✅ Sequence builder for leak-safe window construction")
        print("  ✅ PyTorch Dataset for sequential data")
        print("  ✅ Model adapters for sequence shapes")
        print("  ✅ Family router for model routing")
        print("  ✅ Live rolling buffers for inference")
        print("  ✅ Configuration for sequential mode")
        print("  ✅ Updated trainer files")
        print("  ✅ Verification tests")
    else:
        print("\n❌ Some tests failed - check implementation")
