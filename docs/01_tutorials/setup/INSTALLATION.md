# Installation Guide

System installation and setup for Fox-v1-infra.

## Prerequisites

- Python 3.11 or higher
- pip package manager
- Git (for cloning the repository)
- 8GB+ RAM recommended
- Linux/macOS (Windows via WSL)

## Installation Steps

### 1. Clone Repository

```bash
git clone https://github.com/Fox-ML-infrastructure/Fox-v1-infra.git
cd Fox-v1-infra
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

