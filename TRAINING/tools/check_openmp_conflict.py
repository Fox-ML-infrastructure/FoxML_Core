#!/usr/bin/env python3

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
Quick check for OpenMP runtime conflicts that cause segfaults.

Run this before training to verify you don't have both libgomp and libiomp loaded.
"""

import sys

def check_openmp_runtimes():
    """Check for dangerous OpenMP runtime conflicts."""
    try:
        from threadpoolctl import threadpool_info
    except ImportError:
        print("❌ threadpoolctl not installed - cannot check OpenMP conflicts")
        print("   Install with: pip install threadpoolctl")
        return False
    
    print("🔍 Checking for OpenMP runtime conflicts...")
    print()
    
    infos = threadpool_info()
    openmp_runtimes = []
    
    for pool in infos:
        if pool.get('user_api') == 'openmp':
            internal = pool.get('internal_api', 'unknown')
            filepath = pool.get('filepath', 'unknown')
            version = pool.get('version', 'unknown')
            num_threads = pool.get('num_threads', 'unknown')
            
            openmp_runtimes.append({
                'internal': internal,
                'filepath': filepath,
                'version': version,
                'threads': num_threads
            })
            
            print(f"  OpenMP runtime: {internal}")
            print(f"    File: {filepath}")
            print(f"    Version: {version}")
            print(f"    Threads: {num_threads}")
            print()
    
    if not openmp_runtimes:
        print("✅ No OpenMP runtimes detected (OK if using CPU-only)")
        return True
    
    # Check for conflicts
    internal_apis = set(r['internal'] for r in openmp_runtimes)
    
    if len(internal_apis) > 1:
        print("❌ CONFLICT DETECTED: Multiple OpenMP runtimes loaded!")
        print(f"   Found: {', '.join(internal_apis)}")
        print()
        print("This will cause segfaults! Fix by setting:")
        print("  export MKL_THREADING_LAYER=GNU  # Force MKL to use libgomp")
        print("  export JOBLIB_START_METHOD=spawn  # Never fork with OpenMP")
        print()
        return False
    
    elif 'iomp' in internal_apis or 'intelem' in internal_apis:
        print("⚠️  WARNING: Using Intel OpenMP (libiomp5)")
        print("   This can conflict with libgomp from LightGBM/XGBoost")
        print()
        print("Recommended fix:")
        print("  export MKL_THREADING_LAYER=GNU  # Use libgomp instead")
        print()
        return True  # Warning but not fatal
    
    else:
        print("✅ Single OpenMP runtime detected (libgomp)")
        print("   Safe for LightGBM, XGBoost, and sklearn")
        print()
        return True

def check_blas_backends():
    """Check which BLAS backends are loaded."""
    try:
        from threadpoolctl import threadpool_info
    except ImportError:
        return
    
    print("🔍 Checking BLAS backends...")
    print()
    
    infos = threadpool_info()
    blas_libs = []
    
    for pool in infos:
        if pool.get('user_api') == 'blas':
            internal = pool.get('internal_api', 'unknown')
            filepath = pool.get('filepath', 'unknown')
            num_threads = pool.get('num_threads', 'unknown')
            
            blas_libs.append({
                'internal': internal,
                'filepath': filepath,
                'threads': num_threads
            })
            
            print(f"  BLAS backend: {internal}")
            print(f"    File: {filepath}")
            print(f"    Threads: {num_threads}")
            print()
    
    if not blas_libs:
        print("  No BLAS backends detected")
        print()

def main():
    print("=" * 60)
    print("OpenMP Runtime Conflict Checker")
    print("=" * 60)
    print()
    
    openmp_ok = check_openmp_runtimes()
    check_blas_backends()
    
    print("=" * 60)
    
    if not openmp_ok:
        print("❌ CRITICAL: Fix OpenMP conflicts before training!")
        sys.exit(1)
    else:
        print("✅ OpenMP configuration looks safe")
        sys.exit(0)

if __name__ == "__main__":
    main()

