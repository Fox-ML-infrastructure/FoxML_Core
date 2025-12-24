# MIT License - see LICENSE file

"""
Model Training Functions

Individual model training functions for each model family.
Uses conditional imports to avoid loading TF/Torch in CPU-only child processes.
"""


import os as _os

# ---- CPU-only families (safe to import everywhere) ----
from .lightgbm_trainer import LightGBMTrainer
from .quantile_lightgbm_trainer import QuantileLightGBMTrainer
from .xgboost_trainer import XGBoostTrainer
from .reward_based_trainer import RewardBasedTrainer
from .ngboost_trainer import NGBoostTrainer
from .ensemble_trainer import EnsembleTrainer
from .gmm_regime_trainer import GMMRegimeTrainer
from .change_point_trainer import ChangePointTrainer
from .ftrl_proximal_trainer import FTRLProximalTrainer
from .base_trainer import BaseModelTrainer

__all__ = [
    'LightGBMTrainer',
    'QuantileLightGBMTrainer',
    'XGBoostTrainer',
    'RewardBasedTrainer',
    'NGBoostTrainer',
    'EnsembleTrainer',
    'GMMRegimeTrainer',
    'ChangePointTrainer',
    'FTRLProximalTrainer',
    'BaseModelTrainer',
]

# ---- TensorFlow families (only import if TF is allowed and available) ----
if _os.getenv("TRAINER_CHILD_NO_TF", "0") != "1":
    try:
        # Import TensorFlow - show warnings so user knows if GPU isn't working
        import tensorflow as tf
        _TF_AVAILABLE = True
    except ImportError:
        _TF_AVAILABLE = False
    
    if _TF_AVAILABLE:
        try:
            from .mlp_trainer import MLPTrainer
            from .cnn1d_trainer import CNN1DTrainer
            from .lstm_trainer import LSTMTrainer
            from .transformer_trainer import TransformerTrainer
            from .tabcnn_trainer import TabCNNTrainer
            from .tablstm_trainer import TabLSTMTrainer
            from .tabtransformer_trainer import TabTransformerTrainer
            from .vae_trainer import VAETrainer
            from .gan_trainer import GANTrainer
            from .meta_learning_trainer import MetaLearningTrainer
            from .multi_task_trainer import MultiTaskTrainer
            
            __all__.extend([
                'MLPTrainer',
                'CNN1DTrainer',
                'LSTMTrainer',
                'TransformerTrainer',
                'TabCNNTrainer',
                'TabLSTMTrainer',
                'TabTransformerTrainer',
                'VAETrainer',
                'GANTrainer',
                'MetaLearningTrainer',
                'MultiTaskTrainer',
            ])
        except ImportError as e:
            # TensorFlow is available but trainer import failed
            import logging
            logging.getLogger(__name__).warning(f"TensorFlow trainers not available: {e}")

# ---- PyTorch families (only import if Torch is allowed) ----
# Currently all sequence models use TensorFlow, so this is empty
# Add pure PyTorch trainers here when implemented
if _os.getenv("TRAINER_CHILD_NO_TORCH", "0") != "1":
    pass  # No pure PyTorch trainers yet