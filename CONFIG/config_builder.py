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
Configuration Builder

Builds typed configuration objects by merging experiment configs
with module-specific configs. Ensures clean separation and prevents
config "crossing" between pipeline components.
"""

import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from .config_schemas import (
    ExperimentConfig,
    FeatureSelectionConfig,
    TargetRankingConfig,
    TrainingConfig,
    LeakageConfig,
    SystemConfig,
    DataConfig,
    LoggingConfig,
    ModuleLoggingConfig,
    BackendLoggingConfig
)

logger = logging.getLogger(__name__)

# Resolve CONFIG directory
CONFIG_DIR = Path(__file__).resolve().parent


def load_yaml(path: Path) -> Dict[str, Any]:
    """Load YAML file, return empty dict if not found"""
    if not path.exists():
        logger.warning(f"Config file not found: {path}")
        return {}
    
    try:
        with open(path, 'r') as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        logger.error(f"Failed to load config {path}: {e}")
        return {}


def load_experiment_config(experiment_name: str) -> ExperimentConfig:
    """
    Load experiment configuration from YAML file.
    
    Supports multiple input formats:
    - Just name: "honest_baseline_test" → CONFIG/experiments/honest_baseline_test.yaml
    - Relative path: "experiments/honest_baseline_test.yaml" → CONFIG/experiments/honest_baseline_test.yaml
    - Full path: "CONFIG/experiments/honest_baseline_test.yaml" → CONFIG/experiments/honest_baseline_test.yaml
    - Absolute path: "/path/to/CONFIG/experiments/honest_baseline_test.yaml" → uses as-is
    
    Args:
        experiment_name: Name of experiment, path to experiment, or full path
    
    Returns:
        ExperimentConfig object
    
    Raises:
        FileNotFoundError: If experiment config file doesn't exist
        ValueError: If required fields are missing or invalid
    """
    # Normalize input
    experiment_name = experiment_name.strip()
    
    # Convert to Path for easier manipulation
    input_path = Path(experiment_name)
    
    # Determine if it's a path or just a name
    exp_path = None
    
    # Case 1: Absolute path - use as-is
    if input_path.is_absolute():
        exp_path = input_path
        if not exp_path.suffix:
            exp_path = exp_path.with_suffix('.yaml')
    
    # Case 2: Relative path (contains / or \)
    elif '/' in experiment_name or '\\' in experiment_name:
        # Normalize path separators
        normalized = experiment_name.replace('\\', '/')
        
        # Remove .yaml extension if present (before processing)
        if normalized.endswith('.yaml') or normalized.endswith('.yml'):
            normalized = normalized[:-5] if normalized.endswith('.yaml') else normalized[:-4]
        
        # Remove CONFIG/experiments/ prefix if present (common mistake)
        if normalized.startswith('CONFIG/experiments/'):
            normalized = normalized[len('CONFIG/experiments/'):]
        elif normalized.startswith('experiments/'):
            normalized = normalized[len('experiments/'):]
        elif normalized.startswith('CONFIG/'):
            # If it's just CONFIG/something, assume it's already in CONFIG_DIR
            normalized = normalized[len('CONFIG/'):]
        
        # Build path relative to CONFIG_DIR/experiments
        exp_path = CONFIG_DIR / "experiments" / normalized
        if not exp_path.suffix:
            exp_path = exp_path.with_suffix('.yaml')
        
        # If not found in CONFIG_DIR/experiments, try as relative to current working directory
        if not exp_path.exists():
            exp_path = Path(experiment_name)
            if not exp_path.suffix:
                exp_path = exp_path.with_suffix('.yaml')
    
    # Case 3: Just a name (default: look in CONFIG/experiments/)
    else:
        # Remove .yaml extension if present
        name = experiment_name
        if name.endswith('.yaml') or name.endswith('.yml'):
            name = name[:-5] if name.endswith('.yaml') else name[:-4]
        exp_path = CONFIG_DIR / "experiments" / f"{name}.yaml"
    
    # Final check: ensure file exists
    if not exp_path.exists():
        available = [f.stem for f in (CONFIG_DIR / 'experiments').glob('*.yaml') if f.is_file()]
        raise FileNotFoundError(
            f"Experiment config not found: {exp_path}\n"
            f"Available experiments: {available}"
        )
    
    data = load_yaml(exp_path)
    exp_data = data.get('experiment', {})
    data_data = data.get('data', {})
    targets_data = data.get('targets')  # Don't default to {} - need to check if None
    if targets_data is None:
        targets_data = {}
    
    # Check if auto_targets is enabled (targets.primary not required if auto_targets=true)
    intel_training = data.get('intelligent_training', {})
    auto_targets = intel_training.get('auto_targets', False) if intel_training else False
    
    # Validate required fields
    if not data_data.get('data_dir'):
        raise ValueError(f"Experiment config missing required field: data.data_dir")
    if not data_data.get('symbols'):
        raise ValueError(f"Experiment config missing required field: data.symbols")
    
    # targets.primary is only required if auto_targets is false
    if not auto_targets and not targets_data.get('primary'):
        raise ValueError(
            f"Experiment config missing required field: targets.primary "
            f"(required when auto_targets=false). "
            f"Either set targets.primary or set intelligent_training.auto_targets=true"
        )
    
    # Build DataConfig from data section
    # Support both old format (interval) and new format (bar_interval)
    bar_interval = data_data.get('bar_interval') or data_data.get('interval', '5m')
    data_config = DataConfig(
        timestamp_column=data_data.get('timestamp_column', 'ts'),
        bar_interval=bar_interval,
        max_samples_per_symbol=data_data.get('max_samples_per_symbol', 50000),
        validation_split=data_data.get('validation_split', 0.2),
        random_state=data_data.get('random_state', 42)
    )
    
    # Build ExperimentConfig (validation happens in __post_init__)
    return ExperimentConfig(
        name=exp_data.get('name', experiment_name),
        data_dir=Path(data_data['data_dir']),
        symbols=data_data['symbols'],
        target=targets_data['primary'],
        data=data_config,
        max_samples_per_symbol=data_data.get('max_samples_per_symbol', 5000),
        description=exp_data.get('description'),
        feature_selection_overrides=data.get('feature_selection', {}),
        target_ranking_overrides=data.get('target_ranking', {}),
        training_overrides=data.get('training', {})
    )


def build_feature_selection_config(
    experiment_cfg: ExperimentConfig,
    module_cfg_path: Optional[Path] = None
) -> FeatureSelectionConfig:
    """
    Build FeatureSelectionConfig by merging experiment config with module config.
    
    Args:
        experiment_cfg: Experiment configuration
        module_cfg_path: Optional path to module config (default: feature_selection/multi_model.yaml)
    
    Returns:
        FeatureSelectionConfig object
    
    Raises:
        ValueError: If config validation fails
    """
    # Load module config
    if module_cfg_path is None:
        module_cfg_path = CONFIG_DIR / "feature_selection" / "multi_model.yaml"
    
    # Fallback to legacy location if new doesn't exist
    if not module_cfg_path.exists():
        legacy_path = CONFIG_DIR / "multi_model_feature_selection.yaml"
        if legacy_path.exists():
            logger.warning(f"Using legacy config: {legacy_path}")
            module_data = load_yaml(legacy_path)
        else:
            logger.warning(f"Module config not found: {module_cfg_path}, using defaults")
            module_data = {}
    else:
        module_data = load_yaml(module_cfg_path)
    
    # Extract model families BEFORE applying overrides (so we can filter)
    model_families = module_data.get('model_families', {})
    if not isinstance(model_families, dict):
        model_families = {}
    
    # Check if experiment wants to filter model families (list override)
    enabled_families_list = experiment_cfg.feature_selection_overrides.get('model_families', None)
    if enabled_families_list and isinstance(enabled_families_list, list):
        # Filter to only specified families
        filtered_families = {}
        for family_name in enabled_families_list:
            if family_name in model_families:
                family_cfg = model_families[family_name].copy()
                family_cfg['enabled'] = True
                filtered_families[family_name] = family_cfg
        model_families = filtered_families
        # Remove from overrides so we don't overwrite it later
        overrides_copy = experiment_cfg.feature_selection_overrides.copy()
        overrides_copy.pop('model_families', None)
    else:
        overrides_copy = experiment_cfg.feature_selection_overrides
    
    # Apply other experiment overrides (excluding model_families if it was a list)
    if overrides_copy:
        # Deep merge: update nested dicts
        for key, value in overrides_copy.items():
            if key in module_data and isinstance(module_data[key], dict) and isinstance(value, dict):
                module_data[key].update(value)
            else:
                module_data[key] = value
    
    # Build config (validation happens in __post_init__)
    try:
        return FeatureSelectionConfig(
            top_n=experiment_cfg.feature_selection_overrides.get('top_n', module_data.get('top_n', 30)),
            model_families=model_families,
            aggregation=module_data.get('aggregation', {}),
            sampling=module_data.get('sampling', {}),
            shap=module_data.get('shap', {}),
            permutation=module_data.get('permutation', {}),
            cross_validation=module_data.get('cross_validation', {}),
            output=module_data.get('output', {}),
            compute=module_data.get('compute', {}),
            target=experiment_cfg.target,
            data_dir=experiment_cfg.data_dir,
            symbols=experiment_cfg.symbols,
            max_samples_per_symbol=experiment_cfg.max_samples_per_symbol
        )
    except ValueError as e:
        logger.error(f"FeatureSelectionConfig validation failed: {e}")
        raise


def build_target_ranking_config(
    experiment_cfg: ExperimentConfig,
    module_cfg_path: Optional[Path] = None
) -> TargetRankingConfig:
    """
    Build TargetRankingConfig by merging experiment config with module config.
    
    Raises:
        ValueError: If config validation fails
    """
    """
    Build TargetRankingConfig by merging experiment config with module config.
    
    Args:
        experiment_cfg: Experiment configuration
        module_cfg_path: Optional path to module config
    
    Returns:
        TargetRankingConfig object
    """
    # Load module config (if it exists)
    if module_cfg_path is None:
        module_cfg_path = CONFIG_DIR / "target_ranking" / "multi_model.yaml"
    
    module_data = load_yaml(module_cfg_path) if module_cfg_path.exists() else {}
    
    # Apply experiment overrides
    if experiment_cfg.target_ranking_overrides:
        for key, value in experiment_cfg.target_ranking_overrides.items():
            if key in module_data and isinstance(module_data[key], dict) and isinstance(value, dict):
                module_data[key].update(value)
            else:
                module_data[key] = value
    
    # Build config (validation happens in __post_init__)
    try:
        return TargetRankingConfig(
            model_families=module_data.get('model_families', {}),
            ranking=module_data.get('ranking', {}),
            sampling=module_data.get('sampling', {}),
            cross_validation=module_data.get('cross_validation', {}),
            min_samples=module_data.get('min_samples', 100),
            min_class_samples=module_data.get('min_class_samples', 10),
            data_dir=experiment_cfg.data_dir,
            symbols=experiment_cfg.symbols,
            max_samples_per_symbol=experiment_cfg.max_samples_per_symbol
        )
    except ValueError as e:
        logger.error(f"TargetRankingConfig validation failed: {e}")
        raise


def build_training_config(
    experiment_cfg: ExperimentConfig,
    module_cfg_path: Optional[Path] = None
) -> TrainingConfig:
    """
    Build TrainingConfig by merging experiment config with module config.
    
    Raises:
        ValueError: If config validation fails
    """
    """
    Build TrainingConfig by merging experiment config with module config.
    
    Args:
        experiment_cfg: Experiment configuration
        module_cfg_path: Optional path to module config
    
    Returns:
        TrainingConfig object
    """
    # Load module configs
    if module_cfg_path is None:
        models_path = CONFIG_DIR / "training" / "models.yaml"
        pipeline_path = CONFIG_DIR / "training_config" / "pipeline_config.yaml"
    else:
        models_path = module_cfg_path
        pipeline_path = None
    
    models_data = load_yaml(models_path) if models_path.exists() else {}
    pipeline_data = load_yaml(pipeline_path) if pipeline_path and pipeline_path.exists() else {}
    
    # For training, model_families might come from feature_selection config (shared)
    # Try to load from feature_selection config as fallback
    if not models_data.get('model_families'):
        feature_selection_path = CONFIG_DIR / "feature_selection" / "multi_model.yaml"
        legacy_path = CONFIG_DIR / "multi_model_feature_selection.yaml"
        
        if feature_selection_path.exists():
            fs_data = load_yaml(feature_selection_path)
            models_data['model_families'] = fs_data.get('model_families', {})
        elif legacy_path.exists():
            fs_data = load_yaml(legacy_path)
            models_data['model_families'] = fs_data.get('model_families', {})
    
    # Apply experiment overrides
    if experiment_cfg.training_overrides:
        for key, value in experiment_cfg.training_overrides.items():
            if key == 'model_families' and isinstance(value, list):
                # Filter model families if list provided
                all_families = models_data.get('model_families', {})
                models_data['model_families'] = {
                    k: v for k, v in all_families.items() if k in value
                }
            elif key in models_data and isinstance(models_data[key], dict) and isinstance(value, dict):
                models_data[key].update(value)
            else:
                models_data[key] = value
    
    # Build config (validation happens in __post_init__)
    try:
        return TrainingConfig(
            model_families=models_data.get('model_families', {}),
            cv_folds=experiment_cfg.training_overrides.get('cv_folds', pipeline_data.get('pipeline', {}).get('cv_folds', 5)),
            pipeline=pipeline_data,
            gpu=load_yaml(CONFIG_DIR / "training_config" / "gpu_config.yaml"),
            memory=load_yaml(CONFIG_DIR / "training_config" / "memory_config.yaml"),
            preprocessing=load_yaml(CONFIG_DIR / "training_config" / "preprocessing_config.yaml"),
            threading=load_yaml(CONFIG_DIR / "training_config" / "threading_config.yaml"),
            callbacks=load_yaml(CONFIG_DIR / "training_config" / "callbacks_config.yaml"),
            optimizer=load_yaml(CONFIG_DIR / "training_config" / "optimizer_config.yaml"),
            target=experiment_cfg.target,
            data_dir=experiment_cfg.data_dir,
            symbols=experiment_cfg.symbols,
            max_samples_per_symbol=experiment_cfg.max_samples_per_symbol
        )
    except ValueError as e:
        logger.error(f"TrainingConfig validation failed: {e}")
        raise


def build_leakage_config(module_cfg_path: Optional[Path] = None) -> LeakageConfig:
    """
    Build LeakageConfig from module config.
    
    Args:
        module_cfg_path: Optional path to module config (default: training_config/safety_config.yaml)
    
    Returns:
        LeakageConfig object
    """
    if module_cfg_path is None:
        module_cfg_path = CONFIG_DIR / "training_config" / "safety_config.yaml"
    
    data = load_yaml(module_cfg_path)
    
    return LeakageConfig(
        safety=data.get('safety', {}),
        auto_fix=data.get('auto_fix', {}),
        auto_rerun=data.get('auto_rerun', {}),
        pre_scan=data.get('pre_scan', {}),
        warning_thresholds=data.get('warning_thresholds', {})
    )


def build_system_config(module_cfg_path: Optional[Path] = None) -> SystemConfig:
    """
    Build SystemConfig from module config.
    
    Args:
        module_cfg_path: Optional path to module config (default: training_config/system_config.yaml)
    
    Returns:
        SystemConfig object
    """
    if module_cfg_path is None:
        module_cfg_path = CONFIG_DIR / "training_config" / "system_config.yaml"
    
    data = load_yaml(module_cfg_path)
    
    return SystemConfig(
        paths=data.get('system', {}).get('paths', {}),
        logging=data.get('system', {}).get('logging', {}),
        defaults=data.get('system', {}).get('defaults', {}),
        backup=data.get('system', {}).get('backup', {})
    )


def build_data_config(experiment_cfg: Optional[ExperimentConfig] = None) -> DataConfig:
    """
    Build DataConfig from experiment config or defaults.
    
    Args:
        experiment_cfg: Optional experiment config
    
    Returns:
        DataConfig object
    """
    if experiment_cfg:
        return DataConfig(
            timestamp_column="ts",  # Could come from system config
            interval=experiment_cfg.interval,
            max_samples_per_symbol=experiment_cfg.max_samples_per_symbol,
            validation_split=0.2,
            random_state=42
        )
    else:
        return DataConfig()


def build_logging_config(
    config_path: Optional[Path] = None,
    profile: Optional[str] = None
) -> LoggingConfig:
    """
    Build LoggingConfig from YAML file.
    
    Args:
        config_path: Optional path to logging config (default: logging_config.yaml)
        profile: Optional profile name to apply (default: "default")
    
    Returns:
        LoggingConfig object
    """
    if config_path is None:
        config_path = CONFIG_DIR / "logging_config.yaml"
    
    data = load_yaml(config_path)
    logging_data = data.get('logging', {})
    
    # Apply profile if specified
    if profile and profile != "default":
        profiles = logging_data.get('profiles', {})
        if profile in profiles:
            profile_data = profiles[profile]
            # Merge profile into base config
            if 'global_level' in profile_data:
                logging_data['global_level'] = profile_data['global_level']
            if 'modules' in profile_data:
                for module_name, module_overrides in profile_data['modules'].items():
                    if module_name not in logging_data.get('modules', {}):
                        logging_data.setdefault('modules', {})[module_name] = {}
                    logging_data['modules'][module_name].update(module_overrides)
    
    # Build module configs
    modules = {}
    for module_name, module_data in logging_data.get('modules', {}).items():
        modules[module_name] = ModuleLoggingConfig(
            level=module_data.get('level', logging_data.get('global_level', 'INFO')),
            gpu_detail=module_data.get('gpu_detail', False),
            cv_detail=module_data.get('cv_detail', False),
            edu_hints=module_data.get('edu_hints', False),
            detail=module_data.get('detail', False)
        )
    
    # Build backend configs
    backends = {}
    for backend_name, backend_data in logging_data.get('backends', {}).items():
        backends[backend_name] = BackendLoggingConfig(
            native_verbosity=backend_data.get('native_verbosity', -1),
            show_sparse_warnings=backend_data.get('show_sparse_warnings', True)
        )
    
    return LoggingConfig(
        global_level=logging_data.get('global_level', 'INFO'),
        modules=modules,
        backends=backends,
        profiles=logging_data.get('profiles', {})
    )

