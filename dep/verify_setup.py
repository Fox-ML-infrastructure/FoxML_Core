#!/usr/bin/env python3

# MIT License - see LICENSE file

"""
Verify Setup Script
Checks that data, configs, and environment are properly set up.
"""


import sys
from pathlib import Path
import polars as pl

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def check_directories():
    """Check required directories exist"""
    print("🔍 Checking directory structure...")
    
    required_dirs = [
        "CONFIG",
        "CONFIG/model_config",
        "DATA_PROCESSING",
        "DATA_PROCESSING/features",
        "DATA_PROCESSING/targets",
        "DATA_PROCESSING/pipeline",
        "DATA_PROCESSING/utils",
        "TRAINING",
        "TRAINING/model_fun",
        "INFORMATION",
        "data/data_labeled/interval=5m"
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        full_path = PROJECT_ROOT / dir_path
        if full_path.exists():
            print(f"  ✅ {dir_path}")
        else:
            print(f"  ❌ {dir_path} - MISSING!")
            all_exist = False
    
    return all_exist

def check_data_files():
    """Check data files are readable"""
    print("\n🔍 Checking data files...")
    
    data_dir = PROJECT_ROOT / "data/data_labeled/interval=5m"
    if not data_dir.exists():
        print(f"  ❌ Data directory not found: {data_dir}")
        return False
    
    parquet_files = list(data_dir.glob("*.parquet"))
    if not parquet_files:
        print(f"  ❌ No parquet files found in {data_dir}")
        return False
    
    print(f"  ✅ Found {len(parquet_files)} parquet files")
    
    # Test read first file
    test_file = parquet_files[0]
    try:
        df = pl.read_parquet(test_file)
        print(f"  ✅ Successfully read {test_file.name}")
        print(f"     Rows: {len(df):,}")
        print(f"     Columns: {len(df.columns)}")
        
        # Check required columns
        required_cols = ['ts', 'open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"  ❌ Missing required columns: {missing_cols}")
            return False
        else:
            print(f"  ✅ All required columns present")
        
        return True
    except Exception as e:
        print(f"  ❌ Failed to read {test_file.name}: {e}")
        return False

def check_configs():
    """Check configuration files exist"""
    print("\n🔍 Checking configuration files...")
    
    config_files = [
        "CONFIG/model_config/lightgbm.yaml",
        "CONFIG/model_config/xgboost.yaml",
        "CONFIG/model_config/ensemble.yaml",
        "CONFIG/config_loader.py"
    ]
    
    all_exist = True
    for config_file in config_files:
        full_path = PROJECT_ROOT / config_file
        if full_path.exists():
            print(f"  ✅ {config_file}")
        else:
            print(f"  ❌ {config_file} - MISSING!")
            all_exist = False
    
    return all_exist

def check_imports():
    """Check key imports work"""
    print("\n🔍 Checking Python imports...")
    
    imports_to_test = [
        ("polars", "Polars (for data processing)"),
        ("pandas", "Pandas"),
        ("numpy", "NumPy"),
        ("sklearn", "Scikit-learn"),
        ("lightgbm", "LightGBM"),
        ("xgboost", "XGBoost"),
    ]
    
    all_ok = True
    for module_name, description in imports_to_test:
        try:
            __import__(module_name)
            print(f"  ✅ {description}")
        except ImportError:
            print(f"  ❌ {description} - NOT INSTALLED!")
            all_ok = False
    
    # Test project imports
    try:
        from CONFIG.config_loader import load_model_config
        print(f"  ✅ CONFIG.config_loader")
    except ImportError as e:
        print(f"  ❌ CONFIG.config_loader - {e}")
        all_ok = False
    
    try:
        from DATA_PROCESSING.utils import MemoryManager
        print(f"  ✅ DATA_PROCESSING.utils")
    except ImportError as e:
        print(f"  ❌ DATA_PROCESSING.utils - {e}")
        all_ok = False
    
    return all_ok

def main():
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                        SETUP VERIFICATION                                    ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")
    
    checks = [
        ("Directories", check_directories),
        ("Data Files", check_data_files),
        ("Configurations", check_configs),
        ("Python Imports", check_imports)
    ]
    
    results = {}
    for check_name, check_func in checks:
        results[check_name] = check_func()
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    all_passed = True
    for check_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{check_name:20s} {status}")
        if not passed:
            all_passed = False
    
    print("="*80)
    
    if all_passed:
        print("\n✅ All checks passed! You're ready to run the pipeline.")
        print("\nNext step: python SCRIPTS/process_single_symbol.py AAPL")
        return 0
    else:
        print("\n❌ Some checks failed. Please fix the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

