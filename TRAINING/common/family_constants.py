# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial

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

