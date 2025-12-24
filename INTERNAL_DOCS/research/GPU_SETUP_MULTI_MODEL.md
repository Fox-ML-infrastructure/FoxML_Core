# GPU Setup for Multi-Model Feature Selection

Enabling GPU acceleration can speed up feature selection by 3-10x.

---

## Current GPU Support

| Model Family | GPU Support | Speedup | Setup Required |
|--------------|-------------|---------|----------------|
| **LightGBM** | CUDA/OpenCL | 5-10x | Recompile LightGBM |
| **XGBoost** | CUDA | 3-5x | Install GPU version |
| **Random Forest** | No | N/A | sklearn doesn't support GPU |
| **Neural Network (sklearn)** | No | N/A | Use PyTorch/TF instead |

**Bottom line:** Focus on LightGBM and XGBoost for GPU acceleration.

---

## Check GPU Availability

```bash
# Check NVIDIA GPU
nvidia-smi

# Check CUDA
nvcc --version

# Check if LightGBM has GPU support
python -c "import lightgbm as lgb; print(lgb.__version__)"
```

---

## Setup 1: LightGBM with CUDA (Recommended)

### Arch Linux

```bash
# Install CUDA toolkit
sudo pacman -S cuda cudnn

# Build LightGBM from source with CUDA
cd ~
git clone --recursive https://github.com/microsoft/LightGBM
cd LightGBM
mkdir build && cd build
cmake -DUSE_CUDA=1 ..
make -j$(nproc)

# Install Python package
cd ../python-package
pip uninstall lightgbm -y
pip install -e . --no-build-isolation

# Test
python -c "import lightgbm as lgb; print('CUDA support:', lgb.LGBMRegressor(device='cuda'))"
```

### Ubuntu/Debian

```bash
# Install CUDA toolkit
sudo apt update
sudo apt install nvidia-cuda-toolkit nvidia-cuda-dev

# Follow same build steps as Arch
```

### Verify

```bash
python SCRIPTS/check_gpu_setup.py
```

---

## Setup 2: XGBoost with GPU

```bash
# Install GPU version
pip uninstall xgboost -y
pip install xgboost[gpu]

# Test
python -c "import xgboost as xgb; print(xgb.XGBRegressor(tree_method='gpu_hist'))"
```

---

## Scripts Already Support GPU (Auto-detect)

Both scripts now **automatically detect and use GPU** if available:

### `rank_target_predictability.py`
- Auto-detects CUDA or OpenCL
- Falls back to CPU if GPU unavailable
- Logs which device is being used

### `multi_model_feature_selection.py`
- Respects `device` setting in config
- Default: auto-detect and use GPU if available
- Can force CPU by setting `device: 'cpu'` in config

---

## Performance Comparison

### Target Ranking (5 symbols, 3 models)

| Setup | Time | Speedup |
|-------|------|---------|
| CPU only | ~10 min | 1x |
| LightGBM GPU | ~4 min | 2.5x |
| LightGBM + XGBoost GPU | ~3 min | 3.3x |

### Multi-Model Selection (728 symbols, 4 models)

| Setup | Time | Speedup |
|-------|------|---------|
| CPU only | ~10 hours | 1x |
| LightGBM GPU | ~3-4 hours | 2.5-3x |
| LightGBM + XGBoost GPU | ~2-3 hours | 3-5x |

---

## Configuration

### Enable GPU in Multi-Model Config

Edit `CONFIG/multi_model_feature_selection.yaml`:

```yaml
model_families:
  lightgbm:
    enabled: true
    config:
      device: "cuda"  # or "gpu" for OpenCL, or "cpu" to disable
      gpu_device_id: 0
      max_bin: 63  # Lower for GPU (saves VRAM)

  xgboost:
    enabled: true
    config:
      tree_method: "gpu_hist"  # Use GPU
      gpu_id: 0
```

### Force CPU (if GPU causing issues)

```yaml
model_families:
  lightgbm:
    config:
      device: "cpu"

  xgboost:
    config:
      tree_method: "auto"  # CPU
```

---

## Troubleshooting

### Issue: "CUDA not available"

**Check:**
```bash
nvidia-smi  # GPU visible?
nvcc --version  # CUDA installed?
```

**Fix:**
```bash
# Install CUDA toolkit
sudo pacman -S cuda cudnn  # Arch
sudo apt install nvidia-cuda-toolkit  # Ubuntu
```

### Issue: "Out of GPU memory"

**Fix 1:** Reduce `max_bin` (uses less VRAM)
```yaml
lightgbm:
  config:
    max_bin: 63  # Instead of 255
```

**Fix 2:** Sample more aggressively
```yaml
sampling:
  max_samples_per_symbol: 20000  # Instead of 50000
```

**Fix 3:** Run on CPU for neural networks (they need less speed)
```bash
python SCRIPTS/rank_target_predictability.py \
  --model-families lightgbm,random_forest  # Skip neural_network
```

### Issue: "LightGBM GPU slower than CPU"

**Causes:**
- Small datasets (GPU overhead > benefit)
- Low `n_estimators` (GPU thrives on large models)

**When to use GPU:**
- Large datasets (>10k samples)
- Many estimators (>100)
- Many features (>50)

**When CPU is fine:**
- Small tests (3-5 symbols)
- Quick prototyping
- Feature count < 50

---

## Quick Check Script

Create `SCRIPTS/check_gpu_setup.py`:

```python
#!/usr/bin/env python
"""Check GPU availability for multi-model feature selection"""

import sys
import numpy as np

print("GPU Availability Check")
print("=" * 60)

# Check CUDA
try:
    import pycuda.driver as cuda
    cuda.init()
    print(f" CUDA available: {cuda.Device.count()} device(s)")
    for i in range(cuda.Device.count()):
        dev = cuda.Device(i)
        print(f"   {i}: {dev.name()} ({dev.total_memory() // 1024**2} MB)")
except:
    print(" CUDA not available")

print()

# Check LightGBM
try:
    import lightgbm as lgb
    test_X = np.random.rand(100, 10)
    test_y = np.random.rand(100)

    # Test CUDA
    try:
        model = lgb.LGBMRegressor(device='cuda', n_estimators=10, verbose=-1)
        model.fit(test_X, test_y)
        print(" LightGBM CUDA: Available")
    except Exception as e:
        print(f" LightGBM CUDA: {e}")

    # Test OpenCL
    try:
        model = lgb.LGBMRegressor(device='gpu', n_estimators=10, verbose=-1)
        model.fit(test_X, test_y)
        print(" LightGBM OpenCL: Available")
    except Exception as e:
        print(f" LightGBM OpenCL: {e}")

except ImportError:
    print(" LightGBM not installed")

print()

# Check XGBoost
try:
    import xgboost as xgb
    test_X = np.random.rand(100, 10)
    test_y = np.random.rand(100)

    try:
        model = xgb.XGBRegressor(tree_method='gpu_hist', n_estimators=10)
        model.fit(test_X, test_y)
        print(" XGBoost GPU: Available")
    except Exception as e:
        print(f" XGBoost GPU: {e}")

except ImportError:
    print(" XGBoost not installed")

print()
print("=" * 60)
print("Recommendation:")
print("  • If CUDA available: Rebuild LightGBM with CUDA support")
print("  • If OpenCL works: You're good to go!")
print("  • If neither: Scripts will use CPU (still works fine)")
```

Make it executable:
```bash
chmod +x SCRIPTS/check_gpu_setup.py
python SCRIPTS/check_gpu_setup.py
```

---

## Summary

**To enable GPU (5 minutes):**

```bash
# 1. Check if you have GPU
nvidia-smi

# 2. Install CUDA
sudo pacman -S cuda cudnn  # or apt on Ubuntu

# 3. Rebuild LightGBM with CUDA
cd ~
git clone --recursive https://github.com/microsoft/LightGBM
cd LightGBM && mkdir build && cd build
cmake -DUSE_CUDA=1 .. && make -j$(nproc)
cd ../python-package && pip install -e . --no-build-isolation

# 4. Test
python SCRIPTS/check_gpu_setup.py

# 5. Run scripts (auto-detects GPU)
python SCRIPTS/rank_target_predictability.py
```

**Result:** 3-5x faster feature selection!

