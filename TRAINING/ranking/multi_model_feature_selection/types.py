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

