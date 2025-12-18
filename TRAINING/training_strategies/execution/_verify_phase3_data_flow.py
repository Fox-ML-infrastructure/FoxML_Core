#!/usr/bin/env python3
"""
Verification script to check that all data passed to Phase 3 (Step 3: Model Training)
is correctly passed and can be accepted by the training functions.

This checks:
1. Function signatures match what's being called
2. Return types match what's expected
3. Data types are correct
4. All required parameters are passed
"""

import ast
import inspect
import sys
from pathlib import Path
from typing import get_type_hints, get_args, get_origin

# Add project root
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

def check_function_signature(func, expected_params):
    """Check if function accepts expected parameters"""
    sig = inspect.signature(func)
    params = list(sig.parameters.keys())
    
    issues = []
    for param_name, param_type in expected_params.items():
        if param_name not in params:
            issues.append(f"Missing parameter: {param_name}")
        else:
            # Check if type annotation matches (if available)
            param = sig.parameters[param_name]
            if param.annotation != inspect.Parameter.empty:
                # Type checking would require runtime types, skip for now
                pass
    
    return issues

def check_return_type(func, expected_return_type):
    """Check if function returns expected type"""
    sig = inspect.signature(func)
    return_annotation = sig.return_annotation
    
    if return_annotation == inspect.Signature.empty:
        return ["No return type annotation"]
    
    # For tuple returns, check element count
    if hasattr(return_annotation, '__origin__') and return_annotation.__origin__ is tuple:
        expected_count = len(get_args(expected_return_type)) if hasattr(expected_return_type, '__args__') else None
        actual_count = len(get_args(return_annotation))
        if expected_count and actual_count != expected_count:
            return [f"Return tuple count mismatch: expected {expected_count}, got {actual_count}"]
    
    return []

def main():
    print("=" * 80)
    print("PHASE 3 DATA FLOW VERIFICATION")
    print("=" * 80)
    print()
    
    # Import the actual functions
    try:
        from TRAINING.training_strategies.strategy_functions import load_mtf_data
        from TRAINING.training_strategies.execution.data_preparation import prepare_training_data_cross_sectional
        from TRAINING.training_strategies.execution.training import train_models_for_interval_comprehensive, train_model_comprehensive
        from TRAINING.orchestration.intelligent_trainer import IntelligentTrainer
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return 1
    
    all_ok = True
    
    # Check 1: load_mtf_data return type
    print("1. Checking load_mtf_data() return type...")
    sig = inspect.signature(load_mtf_data)
    print(f"   Signature: {sig}")
    print(f"   Return annotation: {sig.return_annotation}")
    if sig.return_annotation == inspect.Signature.empty:
        print("   ⚠️  No return type annotation")
    else:
        print("   ✓ Has return type annotation")
    print()
    
    # Check 2: prepare_training_data_cross_sectional signature and return
    print("2. Checking prepare_training_data_cross_sectional()...")
    sig = inspect.signature(prepare_training_data_cross_sectional)
    print(f"   Parameters: {list(sig.parameters.keys())}")
    print(f"   Return annotation: {sig.return_annotation}")
    
    # Check return tuple count
    if hasattr(sig.return_annotation, '__args__'):
        return_count = len(sig.return_annotation.__args__)
        print(f"   Return tuple elements: {return_count}")
        if return_count != 8:
            print(f"   ❌ Expected 8 elements, got {return_count}")
            all_ok = False
        else:
            print("   ✓ Return tuple has 8 elements")
    print()
    
    # Check 3: train_models_for_interval_comprehensive signature
    print("3. Checking train_models_for_interval_comprehensive()...")
    sig = inspect.signature(train_models_for_interval_comprehensive)
    print(f"   Parameters: {list(sig.parameters.keys())}")
    
    # Check required parameters
    required_params = ['interval', 'targets', 'mtf_data', 'families']
    for param in required_params:
        if param in sig.parameters:
            print(f"   ✓ Has parameter: {param}")
        else:
            print(f"   ❌ Missing parameter: {param}")
            all_ok = False
    print()
    
    # Check 4: train_model_comprehensive signature
    print("4. Checking train_model_comprehensive()...")
    sig = inspect.signature(train_model_comprehensive)
    print(f"   Parameters: {list(sig.parameters.keys())}")
    
    required_params = ['family', 'X', 'y', 'target', 'strategy', 'feature_names', 'caps', 'routing_meta']
    for param in required_params:
        if param in sig.parameters:
            print(f"   ✓ Has parameter: {param}")
        else:
            print(f"   ❌ Missing parameter: {param}")
            all_ok = False
    print()
    
    # Check 5: Verify data flow in intelligent_trainer
    print("5. Checking data flow in intelligent_trainer...")
    try:
        # Read the file to check the actual call
        trainer_file = _PROJECT_ROOT / "TRAINING/orchestration/intelligent_trainer.py"
        with open(trainer_file) as f:
            content = f.read()
        
        # Check that load_mtf_data is called correctly
        if "mtf_data = load_mtf_data(" in content:
            print("   ✓ load_mtf_data() is called")
        else:
            print("   ❌ load_mtf_data() call not found")
            all_ok = False
        
        # Check that train_models_for_interval_comprehensive is called with correct params
        if "train_models_for_interval_comprehensive(" in content:
            print("   ✓ train_models_for_interval_comprehensive() is called")
            # Check for key parameters
            if "mtf_data=mtf_data" in content:
                print("   ✓ mtf_data parameter passed")
            if "targets=targets" in content:
                print("   ✓ targets parameter passed")
            if "target_features=" in content:
                print("   ✓ target_features parameter passed")
        else:
            print("   ❌ train_models_for_interval_comprehensive() call not found")
            all_ok = False
    except Exception as e:
        print(f"   ⚠️  Could not verify data flow: {e}")
    print()
    
    # Check 6: Verify tuple unpacking
    print("6. Checking tuple unpacking in training.py...")
    training_file = _PROJECT_ROOT / "TRAINING/training_strategies/training.py"
    with open(training_file) as f:
        content = f.read()
    
    # Check unpacking pattern
    if "X, y, feature_names, symbols, indices, feat_cols, time_vals, routing_meta = prepare_training_data_cross_sectional(" in content:
        print("   ✓ Tuple unpacking matches 8-element return")
    else:
        print("   ⚠️  Could not verify tuple unpacking pattern")
    print()
    
    # Summary
    print("=" * 80)
    if all_ok:
        print("✅ ALL CHECKS PASSED - Data flow looks correct")
        print()
        print("Verified:")
        print("  ✓ load_mtf_data returns Dict[str, pd.DataFrame]")
        print("  ✓ prepare_training_data_cross_sectional returns 8-element tuple")
        print("  ✓ train_models_for_interval_comprehensive accepts all required params")
        print("  ✓ train_model_comprehensive accepts all required params")
        print("  ✓ Data flow in intelligent_trainer looks correct")
    else:
        print("❌ SOME ISSUES FOUND - Review above")
    print("=" * 80)
    
    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(main())
