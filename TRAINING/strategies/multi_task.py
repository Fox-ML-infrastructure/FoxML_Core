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
Multi-Task Strategy

Option B: Shared encoder + separate heads for different targets.
Best for: Neural networks, when targets are related, limited data scenarios.
"""


import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import logging
from .base import BaseTrainingStrategy

logger = logging.getLogger(__name__)

class MultiTaskStrategy(BaseTrainingStrategy):
    """Multi-task learning with shared encoder and separate heads"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.shared_model = None
        self.target_types = {}
        
    def train(self, X: np.ndarray, y_dict: Dict[str, np.ndarray], 
              feature_names: List[str], **kwargs) -> Dict[str, Any]:
        """Train multi-task model"""
        logger.info("🧠 Training multi-task model (Option B)")
        
        self.validate_data(X, y_dict)
        
        # Determine target types
        for target_name, y in y_dict.items():
            self.target_types[target_name] = self._determine_target_type(target_name, y)
        
        # Create multi-task model
        self.shared_model = self._create_multi_task_model(
            X.shape[1], 
            list(y_dict.keys()),
            feature_names
        )
        
        # Train the model
        training_results = self._train_multi_task_model(X, y_dict)
        
        # Store results
        results = {
            'shared_model': self.shared_model,
            'target_types': self.target_types,
            'feature_names': feature_names,
            'n_features': X.shape[1],
            'n_samples': len(X),
            'training_results': training_results
        }
        
        return results
    
    def predict(self, X: np.ndarray, **kwargs) -> Dict[str, np.ndarray]:
        """Make predictions using multi-task model"""
        if self.shared_model is None:
            raise ValueError("Model not trained yet")
        
        predictions = {}
        
        try:
            # Get predictions from shared model
            model_predictions = self.shared_model.predict(X)
            
            # Organize predictions by target
            target_names = list(self.target_types.keys())
            for i, target_name in enumerate(target_names):
                if i < len(model_predictions):
                    pred = model_predictions[i]
                    
                    # Apply appropriate transformation based on target type
                    if self.target_types[target_name] == 'classification':
                        # For classification, apply sigmoid to get probabilities
                        pred = self._sigmoid(pred)
                    
                    predictions[target_name] = pred
                else:
                    logger.warning(f"No prediction for target {target_name}")
                    predictions[target_name] = np.zeros(len(X))
                    
        except Exception as e:
            logger.error(f"Error in multi-task prediction: {e}")
            # Return zero predictions for all targets
            for target_name in self.target_types.keys():
                predictions[target_name] = np.zeros(len(X))
        
        return predictions
    
    def get_target_types(self) -> Dict[str, str]:
        """Return target types for each target"""
        return self.target_types.copy()
    
    def _determine_target_type(self, target_name: str, y: np.ndarray) -> str:
        """Determine if target is regression or classification"""
        
        # Check target name patterns
        if target_name.startswith('fwd_ret_'):
            return 'regression'
        elif any(target_name.startswith(prefix) for prefix in 
                ['will_peak', 'will_valley', 'mdd', 'mfe', 'y_will_']):
            return 'classification'
        
        # Check data characteristics
        unique_values = np.unique(y[~np.isnan(y)])
        
        if len(unique_values) <= 2 and all(v in [0, 1] for v in unique_values):
            return 'classification'
        else:
            return 'regression'
    
    def _create_multi_task_model(self, input_dim: int, target_names: List[str], 
                                feature_names: List[str]):
        """Create multi-task neural network model"""
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            logger.error("PyTorch required for multi-task learning")
            raise ImportError("Install PyTorch: pip install torch")
        
        # Get model configuration
        model_config = self.config.get('model', {})
        shared_dim = model_config.get('shared_dim', 128)
        head_dims = model_config.get('head_dims', {})
        
        return MultiTaskNeuralNetwork(
            input_dim=input_dim,
            shared_dim=shared_dim,
            target_names=target_names,
            target_types=self.target_types,
            head_dims=head_dims
        )
    
    def _train_multi_task_model(self, X: np.ndarray, y_dict: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Train the multi-task model"""
        try:
            import torch
            import torch.optim as optim
            from torch.utils.data import DataLoader, TensorDataset
        except ImportError:
            logger.error("PyTorch required for multi-task learning")
            raise ImportError("Install PyTorch: pip install torch")
        
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X)
        y_tensors = {name: torch.FloatTensor(y) for name, y in y_dict.items()}
        
        # Create data loader
        dataset = MultiTaskDataset(X_tensor, y_tensors)
        batch_size = self.config.get('batch_size', 32)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Training setup
        optimizer = optim.Adam(self.shared_model.parameters(), lr=self.config.get('learning_rate', 0.001))
        criterion = MultiTaskLoss(self.target_types, self.config.get('loss_weights', {}))
        
        # Training loop with early stopping
        n_epochs = self.config.get('n_epochs', 100)
        patience = self.config.get('patience', 10)
        self.shared_model.train()
        
        training_losses = []
        best_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(n_epochs):
            # Training phase - Dropout is ACTIVE (model is in train() mode)
            self.shared_model.train()
            epoch_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                
                # Forward pass (Dropout will randomly drop units)
                predictions = self.shared_model(batch_X)
                
                # Compute loss
                loss = criterion(predictions, batch_y)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            training_losses.append(epoch_loss)
            
            # Early stopping check
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                patience_counter = 0
                # Save best model state
                best_model_state = {
                    'encoder': self.shared_model.encoder.state_dict(),
                    'heads': {name: head.state_dict() for name, head in self.shared_model.heads.items()}
                }
            else:
                patience_counter += 1
            
            if epoch % 20 == 0:
                logger.info(f"Epoch {epoch}, Loss: {epoch_loss:.4f}, Best: {best_loss:.4f}")
            
            # Early stopping
            if patience_counter >= patience:
                logger.info(f"✅ Early stopping at epoch {epoch} (patience={patience})")
                # Restore best model
                if best_model_state:
                    self.shared_model.encoder.load_state_dict(best_model_state['encoder'])
                    for name, head in self.shared_model.heads.items():
                        head.load_state_dict(best_model_state['heads'][name])
                break
        
        return {
            'training_losses': training_losses,
            'final_loss': training_losses[-1] if training_losses else 0,
            'best_loss': best_loss,
            'stopped_at_epoch': epoch
        }
    
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Apply sigmoid function"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # Clip to prevent overflow

class MultiTaskNeuralNetwork:
    """Multi-task neural network with shared encoder and separate heads"""
    
    def __init__(self, input_dim: int, shared_dim: int, target_names: List[str], 
                 target_types: Dict[str, str], head_dims: Dict[str, int] = None):
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            raise ImportError("PyTorch required")
        
        self.target_names = target_names
        self.target_types = target_types
        self.head_dims = head_dims or {}
        
        # Shared encoder (Dropout is properly handled via train()/eval())
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),  # Active during training, disabled during eval
            nn.Linear(256, shared_dim),
            nn.ReLU(),
            nn.Dropout(0.2)   # Active during training, disabled during eval
        )
        
        # Separate heads
        self.heads = nn.ModuleDict()
        for target_name in target_names:
            head_dim = self.head_dims.get(target_name, 64)
            self.heads[target_name] = nn.Sequential(
                nn.Linear(shared_dim, head_dim),
                nn.ReLU(),
                nn.Linear(head_dim, 1)
            )
    
    def forward(self, x):
        """Forward pass through the network"""
        # Shared representation
        shared = self.encoder(x)
        
        # Separate predictions
        predictions = {}
        for target_name in self.target_names:
            predictions[target_name] = self.heads[target_name](shared)
        
        return [predictions[target_name] for target_name in self.target_names]
    
    def predict(self, X):
        """Make predictions (numpy interface)"""
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch required")
        
        # CRITICAL: Set model to eval mode (disables Dropout)
        self.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            predictions = self.forward(X_tensor)
            return [pred.numpy() for pred in predictions]
    
    def train(self):
        """Set model to training mode"""
        self.encoder.train()
        for head in self.heads.values():
            head.train()
    
    def eval(self):
        """Set model to evaluation mode"""
        self.encoder.eval()
        for head in self.heads.values():
            head.eval()

class MultiTaskLoss:
    """Multi-task loss function with weighting"""
    
    def __init__(self, target_types: Dict[str, str], loss_weights: Dict[str, float] = None):
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            raise ImportError("PyTorch required")
        
        self.target_types = target_types
        self.loss_weights = loss_weights or {}
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
    
    def __call__(self, predictions, targets):
        """Compute multi-task loss"""
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch required")
        
        total_loss = 0
        target_names = list(self.target_types.keys())
        
        for i, (pred, target) in enumerate(zip(predictions, targets)):
            target_name = target_names[i] if i < len(target_names) else f"target_{i}"
            
            # Determine loss type based on target type
            if self.target_types.get(target_name, 'regression') == 'classification':
                # Binary classification
                loss = self.bce_loss(pred.squeeze(), target)
            else:
                # Regression
                loss = self.mse_loss(pred.squeeze(), target)
            
            # Apply weighting
            weight = self.loss_weights.get(target_name, 1.0)
            total_loss += weight * loss
        
        return total_loss

class MultiTaskDataset:
    """Dataset for multi-task learning"""
    
    def __init__(self, X, y_dict):
        self.X = X
        self.y_dict = y_dict
        self.length = len(X)
        self.target_names = list(y_dict.keys())
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        x = self.X[idx]
        y = [self.y_dict[target][idx] for target in self.target_names]
        return x, y
