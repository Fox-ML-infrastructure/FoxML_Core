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


def _load_config_with_fallback(new_path: str, old_path: str) -> Dict[str, Any]:
    """Load config from new location, fallback to old location."""
    new_file = CONFIG_DIR / new_path
    old_file = CONFIG_DIR / old_path
    
    if new_file.exists():
        return load_yaml(new_file)
    elif old_file.exists():
        logger.debug(f"Using legacy config: {old_path} (consider migrating to {new_path})")
        return load_yaml(old_file)
    else:
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
    
    # Support both max_samples_per_symbol and max_rows_per_symbol (backward compatibility)
    max_samples = data_data.get('max_samples_per_symbol') or data_data.get('max_rows_per_symbol', 50000)
    
    # NEW: Parse base_interval_minutes and related multi-interval settings
    # Precedence: base_interval_minutes (explicit) > bar_interval (legacy) > default
    base_interval_minutes = data_data.get('base_interval_minutes')
    base_interval_source = data_data.get('base_interval_source', 'auto')  # 'config' or 'auto'
    
    # If base_interval_minutes is provided as string (e.g., "5m"), convert to float
    if base_interval_minutes is not None:
        if isinstance(base_interval_minutes, str):
            from TRAINING.utils.data_interval import normalize_interval
            try:
                base_interval_minutes = float(normalize_interval(base_interval_minutes))
            except ValueError as e:
                raise ValueError(
                    f"Invalid base_interval_minutes format '{base_interval_minutes}'. "
                    f"Expected: float (minutes) or string like '5m', '15m', '1h'. Error: {e}"
                )
        elif isinstance(base_interval_minutes, (int, float)):
            base_interval_minutes = float(base_interval_minutes)
        else:
            raise ValueError(
                f"base_interval_minutes must be float or string, got {type(base_interval_minutes)}"
            )
    
    # If base_interval_source='config' but base_interval_minutes not provided, hard-fail
    if base_interval_source == 'config' and base_interval_minutes is None:
        raise ValueError(
            f"base_interval_source='config' requires base_interval_minutes to be set. "
            f"Either set base_interval_minutes in experiment config or use base_interval_source='auto'."
        )
    
    # NEW: Parse embargo and staleness settings
    default_embargo_minutes = float(data_data.get('default_embargo_minutes', 0.0))
    default_max_staleness_minutes = data_data.get('default_max_staleness_minutes')
    if default_max_staleness_minutes is not None:
        default_max_staleness_minutes = float(default_max_staleness_minutes)
    
    asof_strategy = data_data.get('asof_strategy', 'backward')  # Only 'backward' supported for now
    if asof_strategy != 'backward':
        raise ValueError(
            f"asof_strategy must be 'backward' (only supported strategy), got '{asof_strategy}'"
        )
    
    data_config = DataConfig(
        timestamp_column=data_data.get('timestamp_column', 'ts'),
        bar_interval=bar_interval,
        base_interval_minutes=base_interval_minutes,
        base_interval_source=base_interval_source,
        asof_strategy=asof_strategy,
        default_embargo_minutes=default_embargo_minutes,
        default_max_staleness_minutes=default_max_staleness_minutes,
        max_samples_per_symbol=max_samples,
        validation_split=data_data.get('validation_split', 0.2),
        random_state=data_data.get('random_state', 42)
    )
    
    # Build ExperimentConfig (validation happens in __post_init__)
    # Use targets.primary if available, otherwise use empty string (will be overridden by auto_targets)
    primary_target = targets_data.get('primary', '') if targets_data else ''
    
    # Support both max_samples_per_symbol and max_rows_per_symbol for ExperimentConfig too
    exp_max_samples = data_data.get('max_samples_per_symbol') or data_data.get('max_rows_per_symbol', 5000)
    
    return ExperimentConfig(
        name=exp_data.get('name', experiment_name),
        data_dir=Path(data_data['data_dir']),
        symbols=data_data['symbols'],
        target=primary_target,
        data=data_config,
        max_samples_per_symbol=exp_max_samples,
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
    # Load module config - try new location first (ranking/features/), then old (feature_selection/)
    if module_cfg_path is None:
        # Try new location first
        module_cfg_path = CONFIG_DIR / "ranking" / "features" / "multi_model.yaml"
        if not module_cfg_path.exists():
            # Fallback to old location
            module_cfg_path = CONFIG_DIR / "feature_selection" / "multi_model.yaml"
    
    # Fallback to legacy location if new doesn't exist
    if not module_cfg_path.exists():
        legacy_path = CONFIG_DIR / "multi_model_feature_selection.yaml"
        if legacy_path.exists():
            logger.debug(f"Using legacy config: {legacy_path}")
            module_data = load_yaml(legacy_path)
        else:
            logger.debug(f"Module config not found: {module_cfg_path}, using defaults")
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
    # Load module config - try new location first (ranking/targets/), then old (target_ranking/)
    if module_cfg_path is None:
        # Try new location first
        module_cfg_path = CONFIG_DIR / "ranking" / "targets" / "multi_model.yaml"
        if not module_cfg_path.exists():
            # Fallback to old location
            module_cfg_path = CONFIG_DIR / "target_ranking" / "multi_model.yaml"
    
    module_data = load_yaml(module_cfg_path) if module_cfg_path.exists() else {}
    
    # Apply experiment overrides
    # Priority: target_ranking_overrides > training_overrides (for model configs) > module config > defaults
    if experiment_cfg.target_ranking_overrides:
        for key, value in experiment_cfg.target_ranking_overrides.items():
            if key in module_data and isinstance(module_data[key], dict) and isinstance(value, dict):
                module_data[key].update(value)
            else:
                module_data[key] = value
    
    # Also apply training_overrides.model_families if provided (allows experiment config to override model configs)
    # This allows users to override CatBoost/XGBoost/etc configs in training section of experiment config
    if experiment_cfg.training_overrides and 'model_families' in experiment_cfg.training_overrides:
        training_model_families = experiment_cfg.training_overrides['model_families']
        if isinstance(training_model_families, dict):
            # Deep merge model family configs from training_overrides into target_ranking model_families
            if 'model_families' not in module_data:
                module_data['model_families'] = {}
            for family_name, family_config in training_model_families.items():
                if family_name in module_data['model_families']:
                    # Deep merge: update nested config dict
                    if isinstance(module_data['model_families'][family_name], dict) and isinstance(family_config, dict):
                        module_data['model_families'][family_name].update(family_config)
                    else:
                        module_data['model_families'][family_name] = family_config
                else:
                    # Add new family config
                    module_data['model_families'][family_name] = family_config
    
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
    # Load module configs - try new locations first
    if module_cfg_path is None:
        models_path = CONFIG_DIR / "training" / "models.yaml"
        # Try new location first (pipeline/), then old (training_config/)
        pipeline_path = CONFIG_DIR / "pipeline" / "pipeline.yaml"
        if not pipeline_path.exists():
            pipeline_path = CONFIG_DIR / "training_config" / "pipeline_config.yaml"
    else:
        models_path = module_cfg_path
        pipeline_path = None
    
    models_data = load_yaml(models_path) if models_path.exists() else {}
    pipeline_data = load_yaml(pipeline_path) if pipeline_path and pipeline_path.exists() else {}
    
    # For training, model_families might come from feature_selection config (shared)
    # Try to load from feature_selection config as fallback
    if not models_data.get('model_families'):
        # Try new location first (ranking/features/), then old (feature_selection/)
        feature_selection_path = CONFIG_DIR / "ranking" / "features" / "multi_model.yaml"
        if not feature_selection_path.exists():
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
            gpu=_load_config_with_fallback("pipeline/gpu.yaml", "training_config/gpu_config.yaml"),
            memory=_load_config_with_fallback("pipeline/memory.yaml", "training_config/memory_config.yaml"),
            preprocessing=_load_config_with_fallback("pipeline/training/preprocessing.yaml", "training_config/preprocessing_config.yaml"),
            threading=_load_config_with_fallback("pipeline/threading.yaml", "training_config/threading_config.yaml"),
            callbacks=_load_config_with_fallback("pipeline/training/callbacks.yaml", "training_config/callbacks_config.yaml"),
            optimizer=_load_config_with_fallback("pipeline/training/optimizer.yaml", "training_config/optimizer_config.yaml"),
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
        # Try new location first (pipeline/training/), then old (training_config/)
        module_cfg_path = CONFIG_DIR / "pipeline" / "training" / "safety.yaml"
        if not module_cfg_path.exists():
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
        # Try new location first (core/), then old (training_config/)
        module_cfg_path = CONFIG_DIR / "core" / "system.yaml"
        if not module_cfg_path.exists():
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

