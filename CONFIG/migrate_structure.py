#!/usr/bin/env python3
"""
Config Structure Migration Script

Moves config files to a more human-usable structure while maintaining
backward compatibility through symlinks.
"""

import shutil
import sys
from pathlib import Path
from typing import List, Tuple

# Get CONFIG directory
CONFIG_DIR = Path(__file__).resolve().parent

# File mappings: (old_path, new_path, create_symlink)
MIGRATIONS: List[Tuple[Path, Path, bool]] = [
    # Core system configs
    (CONFIG_DIR / "logging_config.yaml", CONFIG_DIR / "core" / "logging.yaml", True),
    (CONFIG_DIR / "training_config" / "system_config.yaml", CONFIG_DIR / "core" / "system.yaml", True),
    
    # Data configs
    (CONFIG_DIR / "feature_registry.yaml", CONFIG_DIR / "data" / "feature_registry.yaml", True),
    (CONFIG_DIR / "excluded_features.yaml", CONFIG_DIR / "data" / "excluded_features.yaml", True),
    (CONFIG_DIR / "feature_target_schema.yaml", CONFIG_DIR / "data" / "feature_target_schema.yaml", True),
    (CONFIG_DIR / "feature_groups.yaml", CONFIG_DIR / "data" / "feature_groups.yaml", True),
    
    # Model configs (move entire directory)
    # Handled separately below
    
    # Pipeline training configs
    (CONFIG_DIR / "training_config" / "intelligent_training_config.yaml", CONFIG_DIR / "pipeline" / "training" / "intelligent.yaml", True),
    (CONFIG_DIR / "training_config" / "safety_config.yaml", CONFIG_DIR / "pipeline" / "training" / "safety.yaml", True),
    (CONFIG_DIR / "training_config" / "preprocessing_config.yaml", CONFIG_DIR / "pipeline" / "training" / "preprocessing.yaml", True),
    (CONFIG_DIR / "training_config" / "optimizer_config.yaml", CONFIG_DIR / "pipeline" / "training" / "optimizer.yaml", True),
    (CONFIG_DIR / "training_config" / "callbacks_config.yaml", CONFIG_DIR / "pipeline" / "training" / "callbacks.yaml", True),
    (CONFIG_DIR / "training_config" / "routing_config.yaml", CONFIG_DIR / "pipeline" / "training" / "routing.yaml", True),
    (CONFIG_DIR / "training_config" / "stability_config.yaml", CONFIG_DIR / "pipeline" / "training" / "stability.yaml", True),
    (CONFIG_DIR / "training_config" / "decision_policies.yaml", CONFIG_DIR / "pipeline" / "training" / "decisions.yaml", True),
    (CONFIG_DIR / "training_config" / "family_config.yaml", CONFIG_DIR / "pipeline" / "training" / "families.yaml", True),
    (CONFIG_DIR / "training_config" / "sequential_config.yaml", CONFIG_DIR / "pipeline" / "training" / "sequential.yaml", True),
    (CONFIG_DIR / "training_config" / "first_batch_specs.yaml", CONFIG_DIR / "pipeline" / "training" / "first_batch.yaml", True),
    
    # Pipeline system configs
    (CONFIG_DIR / "training_config" / "gpu_config.yaml", CONFIG_DIR / "pipeline" / "gpu.yaml", True),
    (CONFIG_DIR / "training_config" / "memory_config.yaml", CONFIG_DIR / "pipeline" / "memory.yaml", True),
    (CONFIG_DIR / "training_config" / "threading_config.yaml", CONFIG_DIR / "pipeline" / "threading.yaml", True),
    (CONFIG_DIR / "training_config" / "pipeline_config.yaml", CONFIG_DIR / "pipeline" / "pipeline.yaml", True),
    
    # Ranking configs
    (CONFIG_DIR / "target_ranking" / "multi_model.yaml", CONFIG_DIR / "ranking" / "targets" / "multi_model.yaml", True),
    (CONFIG_DIR / "target_configs.yaml", CONFIG_DIR / "ranking" / "targets" / "configs.yaml", True),
    (CONFIG_DIR / "feature_selection" / "multi_model.yaml", CONFIG_DIR / "ranking" / "features" / "multi_model.yaml", True),
    (CONFIG_DIR / "feature_selection_config.yaml", CONFIG_DIR / "ranking" / "features" / "config.yaml", True),
    
    # Archive unused files
    (CONFIG_DIR / "comprehensive_feature_ranking.yaml", CONFIG_DIR / "archive" / "comprehensive_feature_ranking.yaml", False),
    (CONFIG_DIR / "fast_target_ranking.yaml", CONFIG_DIR / "archive" / "fast_target_ranking.yaml", False),
    (CONFIG_DIR / "multi_model_feature_selection.yaml.deprecated", CONFIG_DIR / "archive" / "multi_model_feature_selection.yaml.deprecated", False),
]


def migrate_file(old_path: Path, new_path: Path, create_symlink: bool) -> bool:
    """Migrate a single file."""
    if not old_path.exists():
        print(f"  ⚠️  Source not found: {old_path}")
        return False
    
    # Create parent directory if needed
    new_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if destination already exists
    if new_path.exists():
        print(f"  ⚠️  Destination exists: {new_path} (skipping)")
        return False
    
    # Move file
    try:
        shutil.move(str(old_path), str(new_path))
        print(f"  ✅ Moved: {old_path.name} → {new_path.relative_to(CONFIG_DIR)}")
        
        # Create symlink for backward compatibility
        if create_symlink:
            try:
                old_path.symlink_to(new_path)
                print(f"     🔗 Created symlink: {old_path.name} → {new_path.relative_to(CONFIG_DIR)}")
            except OSError as e:
                print(f"     ⚠️  Could not create symlink: {e}")
        
        return True
    except Exception as e:
        print(f"  ❌ Error moving {old_path}: {e}")
        return False


def migrate_model_configs():
    """Migrate model_config directory to models."""
    old_dir = CONFIG_DIR / "model_config"
    new_dir = CONFIG_DIR / "models"
    
    if not old_dir.exists():
        print("  ⚠️  model_config directory not found")
        return
    
    if new_dir.exists() and any(new_dir.iterdir()):
        print("  ⚠️  models directory already has files (skipping)")
        return
    
    # Create models directory
    new_dir.mkdir(parents=True, exist_ok=True)
    
    # Move all YAML files
    moved = 0
    for yaml_file in old_dir.glob("*.yaml"):
        new_path = new_dir / yaml_file.name
        if not new_path.exists():
            shutil.move(str(yaml_file), str(new_path))
            print(f"  ✅ Moved: model_config/{yaml_file.name} → models/{yaml_file.name}")
            moved += 1
    
    if moved > 0:
        # Create symlink for backward compatibility
        try:
            if not old_dir.exists() or not any(old_dir.iterdir()):
                # Only create symlink if old dir is empty
                if old_dir.exists():
                    old_dir.rmdir()
                old_dir.symlink_to(new_dir)
                print(f"     🔗 Created symlink: model_config/ → models/")
        except OSError as e:
            print(f"     ⚠️  Could not create symlink: {e}")


def create_readme_files():
    """Create README files in new directories."""
    readmes = {
        CONFIG_DIR / "core" / "README.md": """# Core System Configs

Core system configuration files for logging, system resources, and paths.

## Files

- `logging.yaml` - Logging configuration
- `system.yaml` - System resources and paths
""",
        CONFIG_DIR / "data" / "README.md": """# Data Configs

Data-related configuration files for features, schemas, and exclusions.

## Files

- `feature_registry.yaml` - Feature registry (allowed/excluded features)
- `excluded_features.yaml` - Always-excluded features
- `feature_target_schema.yaml` - Feature-target schema definitions
- `feature_groups.yaml` - Feature groups (if used)
""",
        CONFIG_DIR / "models" / "README.md": """# Model Configs

Model-specific hyperparameter configurations.

## Files

Each YAML file corresponds to a model family (e.g., `lightgbm.yaml`, `xgboost.yaml`).

See individual files for model-specific settings.
""",
        CONFIG_DIR / "pipeline" / "README.md": """# Pipeline Configs

Pipeline execution configuration files.

## Structure

- `training/` - Training pipeline configs
- `gpu.yaml` - GPU settings
- `memory.yaml` - Memory management
- `threading.yaml` - Threading policy
- `pipeline.yaml` - Main pipeline configuration
""",
        CONFIG_DIR / "pipeline" / "training" / "README.md": """# Training Pipeline Configs

Training-specific configuration files.

## Files

- `intelligent.yaml` - Intelligent training (main config)
- `safety.yaml` - Safety & temporal configs
- `preprocessing.yaml` - Data preprocessing
- `optimizer.yaml` - Optimizer settings
- `callbacks.yaml` - Training callbacks
- `routing.yaml` - Target routing
- `stability.yaml` - Stability analysis
- `decisions.yaml` - Decision policies
- `families.yaml` - Model family configs
- `sequential.yaml` - Sequential training
- `first_batch.yaml` - First batch specs
""",
        CONFIG_DIR / "ranking" / "README.md": """# Ranking & Selection Configs

Configuration files for target ranking and feature selection.

## Structure

- `targets/` - Target ranking configs
- `features/` - Feature selection configs
""",
        CONFIG_DIR / "ranking" / "targets" / "README.md": """# Target Ranking Configs

Configuration files for target ranking.

## Files

- `multi_model.yaml` - Multi-model target ranking
- `configs.yaml` - Target configs (legacy)
""",
        CONFIG_DIR / "ranking" / "features" / "README.md": """# Feature Selection Configs

Configuration files for feature selection.

## Files

- `multi_model.yaml` - Multi-model feature selection
- `config.yaml` - Feature selection config (legacy)
""",
        CONFIG_DIR / "archive" / "README.md": """# Archive

Archived and deprecated configuration files.

These files are no longer actively used but are kept for reference.

## Files

- `comprehensive_feature_ranking.yaml` - Legacy feature ranking config
- `fast_target_ranking.yaml` - Legacy fast ranking config
- `multi_model_feature_selection.yaml.deprecated` - Deprecated feature selection config
"""
    }
    
    for readme_path, content in readmes.items():
        if not readme_path.exists():
            readme_path.parent.mkdir(parents=True, exist_ok=True)
            readme_path.write_text(content)
            print(f"  📝 Created: {readme_path.relative_to(CONFIG_DIR)}")


def main():
    """Run the migration."""
    print("=" * 80)
    print("CONFIG Structure Migration")
    print("=" * 80)
    print()
    
    # Create README files first
    print("Creating README files...")
    create_readme_files()
    print()
    
    # Migrate individual files
    print("Migrating individual files...")
    migrated = 0
    for old_path, new_path, create_symlink in MIGRATIONS:
        if migrate_file(old_path, new_path, create_symlink):
            migrated += 1
    print()
    
    # Migrate model configs
    print("Migrating model configs...")
    migrate_model_configs()
    print()
    
    print("=" * 80)
    print(f"Migration complete! Migrated {migrated} files.")
    print("=" * 80)
    print()
    print("Next steps:")
    print("1. Update config loaders to check new locations first")
    print("2. Test that everything still works")
    print("3. After migration period, remove old symlinks")


if __name__ == "__main__":
    main()

