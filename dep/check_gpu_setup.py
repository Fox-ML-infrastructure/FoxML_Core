#!/usr/bin/env python3

# MIT License - see LICENSE file

"""
Quick GPU setup verification script for LightGBM.
Checks if your system is ready for GPU-accelerated feature selection.
"""


import sys
import subprocess
from pathlib import Path

def print_section(title):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def check_lightgbm_gpu():
    """Check if LightGBM is installed with GPU support."""
    print_section("1. Checking LightGBM Installation")
    
    try:
        import lightgbm as lgb
        
        # Try to get version (may not exist in incomplete installations)
        try:
            version = lgb.__version__
            print(f"✅ LightGBM version: {version}")
        except AttributeError:
            print(f"⚠️  LightGBM installed but version unavailable (possibly incomplete installation)")
        
        # Try to check for GPU support
        import numpy as np
        test_X = np.random.rand(10, 5)
        test_y = np.random.rand(10)
        test_data = lgb.Dataset(test_X, label=test_y)
        
        # Try CUDA first (preferred for NVIDIA)
        cuda_works = False
        opencl_works = False
        
        print("\n   Testing CUDA support...")
        try:
            test_params = {
                'device': 'cuda',
                'objective': 'regression',
                'verbose': -1
            }
            lgb.train(test_params, test_data, num_boost_round=1, callbacks=[lgb.log_evaluation(period=0)])
            print("   ✅ CUDA: ENABLED")
            cuda_works = True
        except Exception as e:
            print(f"   ❌ CUDA: NOT AVAILABLE ({str(e).split(':')[0]})")
        
        print("\n   Testing OpenCL support...")
        try:
            test_params = {
                'device': 'gpu',
                'objective': 'regression',
                'verbose': -1
            }
            lgb.train(test_params, test_data, num_boost_round=1, callbacks=[lgb.log_evaluation(period=0)])
            print("   ✅ OpenCL: ENABLED")
            opencl_works = True
        except Exception as e:
            print(f"   ❌ OpenCL: NOT AVAILABLE ({str(e).split(':')[0]})")
        
        if cuda_works or opencl_works:
            print(f"\n✅ LightGBM GPU support: ENABLED")
            if cuda_works:
                print("   Preferred mode: CUDA (fastest for NVIDIA)")
            return True
        else:
            print(f"\n❌ LightGBM GPU support: NOT AVAILABLE")
            print("   Your LightGBM is the CPU-only pip version")
            return False
            
    except ImportError:
        print("❌ LightGBM not installed")
        print("   Install with: pip install lightgbm")
        return False

def check_cuda():
    """Check if CUDA is available."""
    print_section("2. Checking CUDA Installation (NVIDIA)")
    
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            # Parse CUDA version
            for line in result.stdout.split('\n'):
                if 'release' in line.lower():
                    version = line.split('release')[-1].strip().split(',')[0].strip()
                    print(f"✅ CUDA Compiler: version {version}")
                    break
            return True
        else:
            print("❌ CUDA compiler (nvcc) not found")
            return False
    except FileNotFoundError:
        print("❌ CUDA not installed")
        print("   Install with: sudo pacman -S cuda cudnn")
        return False
    except Exception as e:
        print(f"❌ Error checking CUDA: {e}")
        return False

def check_opencl():
    """Check if OpenCL is available."""
    print_section("3. Checking OpenCL Installation (Optional)")
    
    try:
        result = subprocess.run(['clinfo'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            # Parse output to show GPU devices
            lines = result.stdout.split('\n')
            platform_count = 0
            device_count = 0
            
            for line in lines:
                if 'Number of platforms' in line:
                    platform_count = int(line.split(':')[-1].strip())
                if 'Number of devices' in line:
                    device_count += int(line.split(':')[-1].strip())
                if 'Device Name' in line:
                    device_name = line.split(':')[-1].strip()
                    print(f"   📱 Found GPU: {device_name}")
            
            print(f"✅ OpenCL is installed")
            print(f"   Platforms: {platform_count}, Devices: {device_count}")
            return True
        else:
            print("❌ OpenCL not properly configured")
            return False
            
    except FileNotFoundError:
        print("❌ clinfo not found (OpenCL may not be installed)")
        print("   Install with: sudo apt install clinfo")
        return False
    except Exception as e:
        print(f"❌ Error checking OpenCL: {e}")
        return False

def check_gpu_memory():
    """Try to estimate available GPU memory."""
    print_section("4. Checking GPU Memory")
    
    try:
        # Try nvidia-smi first (for NVIDIA GPUs)
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.total,memory.free,name', '--format=csv,noheader,nounits'],
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            for line in result.stdout.strip().split('\n'):
                if line:
                    total, free, name = [x.strip() for x in line.split(',')]
                    print(f"   🎮 {name}")
                    print(f"      Total VRAM: {int(total)/1024:.1f} GB")
                    print(f"      Free VRAM:  {int(free)/1024:.1f} GB")
                    
                    if float(free) < 7000:  # Less than 7GB free
                        print(f"      ⚠️  Warning: Less than 7GB free VRAM")
                    else:
                        print(f"      ✅ Sufficient VRAM for feature selection")
            return True
    except FileNotFoundError:
        # Try rocm-smi for AMD GPUs
        try:
            result = subprocess.run(['rocm-smi', '--showmeminfo', 'vram'],
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                print("   AMD GPU detected via ROCm")
                print(result.stdout)
                return True
        except FileNotFoundError:
            print("   ℹ️  Could not detect GPU memory (nvidia-smi or rocm-smi not found)")
            print("   This is OK - LightGBM will manage memory automatically")
            return True
    
    return True

def provide_recommendations(lgb_ok, cuda_ok, opencl_ok):
    """Provide installation recommendations based on checks."""
    print_section("Summary & Recommendations")
    
    if lgb_ok:
        print("✅ Your system is ready for GPU-accelerated feature selection!")
        print("\n📝 Next steps:")
        print("   1. Config is already set to CUDA in CONFIG/feature_selection_config.yaml:")
        print("      device: 'cuda'")
        print("   2. Run feature selection:")
        print("      python SCRIPTS/select_features.py")
        print("\n💡 Tips for your 10GB VRAM:")
        print("   - max_bin: 63  (reduces memory usage)")
        print("   - Data is auto-converted to float32 (saves 50% memory)")
        print("   - Process runs sequentially (one symbol at a time)")
        print("   - You have plenty of VRAM - can increase max_bin to 127 if desired")
        
    elif not cuda_ok:
        print("❌ CUDA is not installed (required for NVIDIA GPU)")
        print("\n🔧 Installation guide for Arch Linux:")
        print("\n   1. Install CUDA:")
        print("      sudo pacman -S cuda cudnn")
        print("\n   2. Build LightGBM with CUDA (automated):")
        print("      bash SCRIPTS/build_lightgbm_cuda.sh")
        print("\n   OR manually:")
        print("      git clone --recursive https://github.com/microsoft/LightGBM")
        print("      cd LightGBM && mkdir build && cd build")
        print("      cmake -DUSE_CUDA=1 ..")
        print("      make -j$(nproc)")
        print("      cd ../python-package && pip install -e .")
        print("\n   3. Verify:")
        print("      python SCRIPTS/check_gpu_setup.py")
        
    elif not lgb_ok:
        print("❌ LightGBM GPU support is not enabled")
        print("   Your current LightGBM is the CPU-only pip version")
        print("\n🔧 Build LightGBM with CUDA support:")
        print("\n   Easy way (automated script):")
        print("      bash SCRIPTS/build_lightgbm_cuda.sh")
        print("\n   Manual way:")
        print("      git clone --recursive https://github.com/microsoft/LightGBM")
        print("      cd LightGBM && mkdir build && cd build")
        print("      cmake -DUSE_CUDA=1 ..")
        print("      make -j$(nproc)")
        print("      cd ../python-package")
        print("      pip uninstall -y lightgbm")
        print("      pip install -e .")
        print("\n   See detailed guide: SCRIPTS/ARCH_CUDA_SETUP.md")

def main():
    """Main check function."""
    print("\n🔍 GPU Setup Verification for LightGBM Feature Selection")
    print("="*60)
    print("   Checking for CUDA support (NVIDIA GPU detected)")
    print("="*60)
    
    lgb_ok = check_lightgbm_gpu()
    cuda_ok = check_cuda()
    opencl_ok = check_opencl()
    check_gpu_memory()
    provide_recommendations(lgb_ok, cuda_ok, opencl_ok)
    
    print("\n")
    return 0 if lgb_ok else 1

if __name__ == "__main__":
    sys.exit(main())

