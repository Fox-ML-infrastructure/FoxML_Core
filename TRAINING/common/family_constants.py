"""
Copyright (c) 2025-2026 Fox ML Infrastructure LLC

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
Model Family Constants

Centralized definitions of model family classifications.
"""

# TensorFlow families
TF_FAMS = {"MLP", "VAE", "GAN", "MetaLearning", "MultiTask"}

# PyTorch families
TORCH_FAMS = {"CNN1D", "LSTM", "Transformer", "TabCNN", "TabLSTM", "TabTransformer"}

# CPU-only families (no GPU required)
CPU_FAMS = {
    "LightGBM",
    "QuantileLightGBM",
    "RewardBased",
    "NGBoost",
    "GMMRegime",
    "ChangePoint",
    "FTRLProximal",
    "Ensemble"
}

# GPU TensorFlow families (alternative naming)
GPU_TF_FAMS = {"MLP", "CNN1D", "LSTM", "Transformer", "TabCNN", "TabLSTM", "TabTransformer", 
                "VAE", "GAN", "MetaLearning", "MultiTask"}

# GPU PyTorch families (alternative naming)
GPU_TORCH = {"CNN1D", "LSTM", "Transformer", "TabCNN", "TabLSTM", "TabTransformer"}

# PyTorch sequential families (for better performance)
TORCH_SEQ_FAMILIES = {"CNN1D", "LSTM", "Transformer", "TabCNN", "TabLSTM", "TabTransformer"}

