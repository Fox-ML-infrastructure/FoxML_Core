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

# ---- TensorFlow families (only import if TF is allowed) ----
if _os.getenv("TRAINER_CHILD_NO_TF", "0") != "1":
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

# ---- PyTorch families (only import if Torch is allowed) ----
# Currently all sequence models use TensorFlow, so this is empty
# Add pure PyTorch trainers here when implemented
if _os.getenv("TRAINER_CHILD_NO_TORCH", "0") != "1":
    pass  # No pure PyTorch trainers yet