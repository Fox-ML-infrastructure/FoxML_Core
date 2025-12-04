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
Schema validator with support for optional interaction features.
Allows downgrading interaction features from fatal errors to warnings.
"""


from dataclasses import dataclass
from typing import Iterable, Set, List
import logging

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class SchemaExpectations:
    """Schema validation expectations"""
    required: Set[str]
    optional_interactions: Set[str] = None
    
    def __post_init__(self):
        if self.optional_interactions is None:
            object.__setattr__(self, 'optional_interactions', set())

def validate_schema(cols: Iterable[str], spec: SchemaExpectations, 
                   strict_interactions: bool = True) -> List[str]:
    """
    Validate schema against expectations.
    
    Args:
        cols: Available column names
        spec: Schema expectations
        strict_interactions: If True, missing interactions are fatal; if False, they're warnings
    
    Returns:
        List of validation warnings (empty if all checks pass)
        
    Raises:
        ValueError: If required columns are missing or interactions are missing in strict mode
    """
    cols = set(cols)
    warnings = []

    # Check required columns
    missing_required = spec.required - cols
    if missing_required:
        raise ValueError(f"Missing required columns: {sorted(missing_required)}")

    # Check interaction columns
    missing_interactions = spec.optional_interactions - cols
    if missing_interactions:
        if strict_interactions:
            raise ValueError(f"Missing interaction columns (strict mode): {sorted(missing_interactions)}")
        else:
            warning_msg = f"Optional interaction columns missing: {sorted(missing_interactions)}"
            warnings.append(warning_msg)
            logger.warning(warning_msg)

    return warnings

def create_schema_expectations_from_features(feature_list: List[str], 
                                           interaction_features: Set[str] = None) -> SchemaExpectations:
    """
    Create schema expectations from a feature list, automatically categorizing interaction features.
    
    Args:
        feature_list: List of all expected features
        interaction_features: Set of known interaction features (auto-detected if None)
    
    Returns:
        SchemaExpectations with required and optional_interactions sets
    """
    if interaction_features is None:
        # Auto-detect interaction features (contain '_x_' pattern)
        interaction_features = {f for f in feature_list if '_x_' in f}
    
    required = set(feature_list) - interaction_features
    optional_interactions = interaction_features & set(feature_list)
    
    return SchemaExpectations(
        required=required,
        optional_interactions=optional_interactions
    )

def validate_feature_completeness(built_features: Set[str], expected_features: List[str],
                                strict_interactions: bool = True) -> bool:
    """
    Validate that all expected features were built successfully.
    
    Args:
        built_features: Set of actually built feature names
        expected_features: List of expected feature names
        strict_interactions: Whether to treat missing interactions as fatal
    
    Returns:
        True if validation passes, False otherwise
    """
    try:
        spec = create_schema_expectations_from_features(expected_features)
        warnings = validate_schema(built_features, spec, strict_interactions)
        
        if warnings:
            logger.info(f"Schema validation completed with {len(warnings)} warnings")
            for warning in warnings:
                logger.info(f"  - {warning}")
        
        return True
        
    except ValueError as e:
        logger.error(f"Schema validation failed: {e}")
        return False
