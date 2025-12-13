# Installation Guide

System installation and setup for FoxML Core.

## Prerequisites

**⚠️ Important**: FoxML Core requires significant computational resources. This is **not suitable for laptop/desktop deployment** for production workloads.

### Minimum Requirements (Development/Testing Only)

- **OS**: Linux (Arch Linux, Ubuntu 22.04+, or similar)
  - Tested on: Arch Linux (kernel 6.17+)
- **Python**: 3.10 (as specified in `environment.yml`)
- **Package manager**: Conda (recommended) or pip
- **Git**: For cloning the repository
- **16 GB RAM minimum** (32 GB recommended for development)
- **Build tools**: GCC 11+ (provided via conda-forge if using conda)

**Note**: Minimum configuration is suitable for small-scale testing and development only. See [System Requirements](../../00_executive/SYSTEM_REQUIREMENTS.md) for production requirements.

### Production Requirements

- **OS**: Linux (Arch Linux, Ubuntu 22.04+, or similar)
  - Tested on: Arch Linux (kernel 6.17+)
- **Python**: 3.10 (as specified in `environment.yml`)
- **64 GB RAM minimum** (128 GB+ recommended)
- Multi-core processor (16+ cores)
- SSD storage (500 GB+)
- GPU optional but recommended (11 GB+ VRAM, CUDA 12.9)

**Verified Stable Range**: Up to 100 GB RAM (tested and verified)  
**Targeted Capacity**: 1 TB+ RAM (enterprise deployment)

See [System Requirements](../../00_executive/SYSTEM_REQUIREMENTS.md) for complete specifications.

## Installation Steps

### 1. Clone Repository

```bash
git clone https://github.com/Fox-ML-infrastructure/FoxML_Core.git
cd FoxML_Core
```

### 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
python -c "import sys; print(sys.version)"
python -c "import pandas, numpy, lightgbm; print('Core packages OK')"
```

## Optional: GPU Support

For GPU-accelerated training, see [GPU Setup](GPU_SETUP.md).

## Next Steps

- [Environment Setup](ENVIRONMENT_SETUP.md) - Configure Python environment
- [Quick Start](../../00_executive/QUICKSTART.md) - Get running in 5 minutes

