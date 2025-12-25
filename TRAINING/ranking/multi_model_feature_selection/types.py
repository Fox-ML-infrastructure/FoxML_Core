# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial

"""
Multi-Model Feature Selection Types

Data classes for multi-model feature selection pipeline.
"""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class ModelFamilyConfig:
    """Configuration for a model family"""
    name: str
    importance_method: str  # 'native', 'shap', 'permutation'
    enabled: bool
    config: Dict[str, Any]
    weight: float = 1.0  # Weight in final aggregation


@dataclass
class ImportanceResult:
    """Result from a single model's feature importance calculation"""
    model_family: str
    symbol: str
    importance_scores: Any  # pd.Series
    method: str
    train_score: float

