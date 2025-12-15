#!/usr/bin/env python3
"""
Show Config Hierarchy - What Controls What

This script helps you understand which config file controls which settings.
"""
import yaml
from pathlib import Path
from typing import Dict, Any, List, Tuple

CONFIG_DIR = Path(__file__).parent.parent

def find_symlinks() -> List[Tuple[str, str]]:
    """Find all symlinks and their targets"""
    symlinks = []
    for item in CONFIG_DIR.rglob('*'):
        if item.is_symlink():
            target = item.resolve()
            rel_old = item.relative_to(CONFIG_DIR)
            try:
                rel_new = target.relative_to(CONFIG_DIR)
                symlinks.append((str(rel_old), str(rel_new)))
            except ValueError:
                # Target is outside CONFIG_DIR
                symlinks.append((str(rel_old), str(target)))
    return symlinks

def show_experiment_config_structure(exp_name: str = "e2e_full_targets_test"):
    """Show what an experiment config controls"""
    exp_file = CONFIG_DIR / "experiments" / f"{exp_name}.yaml"
    if not exp_file.exists():
        print(f"❌ Experiment config not found: {exp_file}")
        return
    
    with open(exp_file) as f:
        exp = yaml.safe_load(f) or {}
    
    print(f"\n📋 EXPERIMENT CONFIG: {exp_name}.yaml")
    print("=" * 80)
    
    if 'data' in exp:
        print("\n📊 DATA SECTION (data.*):")
        data = exp['data']
        for key in ['data_dir', 'symbols', 'min_cs', 'max_cs_samples', 'max_rows_per_symbol', 'max_rows_train']:
            if key in data:
                print(f"  • {key}: {data[key]}")
    
    if 'intelligent_training' in exp:
        print("\n🎯 INTELLIGENT TRAINING SECTION (intelligent_training.*):")
        intel = exp['intelligent_training']
        for key in ['auto_targets', 'top_n_targets', 'max_targets_to_evaluate', 
                    'auto_features', 'top_m_features', 'strategy']:
            if key in intel:
                print(f"  • {key}: {intel[key]}")
    
    if 'training' in exp:
        print("\n🏋️ TRAINING SECTION (training.*):")
        training = exp['training']
        if 'model_families' in training:
            print(f"  • model_families: {training['model_families']}")

def show_config_precedence():
    """Show config loading precedence"""
    print("\n" + "=" * 80)
    print("CONFIG PRECEDENCE (Highest to Lowest Priority)")
    print("=" * 80)
    print("""
1. CLI Arguments (--top-n-targets, --data-dir, etc.)
   └─ Highest priority, overrides everything

2. Experiment Config (experiments/*.yaml)
   └─ Used when: --experiment-config <name>
   └─ Overrides: intelligent training config

3. Intelligent Training Config (pipeline/training/intelligent.yaml)
   └─ Used when: NOT using --experiment-config
   └─ Base defaults for runs

4. Pipeline Configs (pipeline/training/*.yaml)
   └─ Training workflow configs (safety, routing, preprocessing, etc.)

5. Defaults (defaults.yaml)
   └─ Single Source of Truth for common settings
   └─ Lowest priority, used as fallback
    """)

def main():
    import sys
    
    print("=" * 80)
    print("CONFIG HIERARCHY GUIDE")
    print("=" * 80)
    
    # Show symlinks
    symlinks = find_symlinks()
    if symlinks:
        print("\n📌 SYMLINKS (Legacy → New Location):")
        for old, new in sorted(symlinks):
            print(f"  {old:50s} → {new}")
    
    # Show experiment config
    exp_name = sys.argv[1] if len(sys.argv) > 1 else "e2e_full_targets_test"
    show_experiment_config_structure(exp_name)
    
    # Show precedence
    show_config_precedence()
    
    print("\n" + "=" * 80)
    print("QUICK REFERENCE")
    print("=" * 80)
    print("""
To change settings for a run:
  • Edit: CONFIG/experiments/<your_experiment>.yaml
  • Run with: --experiment-config <your_experiment>

To change base defaults:
  • Edit: CONFIG/pipeline/training/intelligent.yaml

To change global defaults:
  • Edit: CONFIG/defaults.yaml
    """)

if __name__ == "__main__":
    main()

