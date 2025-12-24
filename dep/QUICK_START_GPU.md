# Quick Start: GPU-Accelerated Feature Selection (CUDA)

## TL;DR - 3 Steps to GPU Speed

```bash
# 1. Check if GPU is ready
python SCRIPTS/check_gpu_setup.py

# 2. Already enabled in config! (device: "cuda")
# 3. Run feature selection (will auto-detect GPU)
python SCRIPTS/select_features.py
```

## What Changed?

- GPU support added to `select_features.py`
- Memory optimized for 7GB VRAM
- Auto-fallback to CPU if GPU unavailable
- Smart processing: Sequential on GPU (faster than parallel CPU)

## Config Changes

Check `CONFIG/feature_selection_config.yaml`:

```yaml
lightgbm:
  device: "cuda" # ← CUDA enabled! (faster than OpenCL)
  max_bin: 63 # ← Optimized for 7GB VRAM
  gpu_use_dp: false # ← Uses float32 (saves 50% memory)
```

## Expected Performance

| Mode | Speed | Memory |
|------|-------|--------|
| CPU (12 workers) | 10-20 min for 20 symbols | 60-120GB RAM |
| GPU (sequential) | 1-2 min for 20 symbols | 3-5GB VRAM |

~10x faster.

## If GPU Fails

The script will automatically:
1. Test GPU availability
2. Show helpful error messages
3. Fall back to CPU mode
4. Continue processing normally

## Need to Disable GPU?

Edit `CONFIG/feature_selection_config.yaml`:

```yaml
lightgbm:
  device: "cpu" # ← Change to "cpu"
```

## GPU Not Set Up Yet?

See installation guides:
- Arch Linux (CUDA): `SCRIPTS/ARCH_CUDA_SETUP.md` (Recommended)
- General Guide: `SCRIPTS/GPU_FEATURE_SELECTION_GUIDE.md`

Quick install (Arch Linux):

```bash
# 1. Install CUDA
sudo pacman -S cuda cudnn

# 2. Build LightGBM with CUDA (automated script)
bash SCRIPTS/build_lightgbm_cuda.sh

# Or manually:
git clone --recursive https://github.com/microsoft/LightGBM
cd LightGBM && mkdir build && cd build
cmake -DUSE_CUDA=1 ..
make -j$(nproc)
cd ../python-package && pip install -e .
```
