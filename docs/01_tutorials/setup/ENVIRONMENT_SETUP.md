# Environment Setup

Configure your Python environment for Fox-v1-infra.

## Virtual Environment

### Create Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Activate Environment

Always activate before working:

```bash
source venv/bin/activate
```

## Environment Variables

### Required Variables

Set these in your shell or `.env` file:

```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export DATA_DIR="./data"
export MODELS_DIR="./models"
```

### Optional Variables

```bash
export LOG_LEVEL="INFO"
export CUDA_VISIBLE_DEVICES="0"  # For GPU
```

## Python Path Configuration

The project uses dynamic path resolution. Ensure the project root is in your Python path:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
```

## Package Installation

### Core Packages

```bash
pip install pandas numpy scipy scikit-learn
pip install lightgbm xgboost
pip install torch  # For deep learning models
```

### Optional Packages

```bash
pip install yfinance  # For data fetching
pip install ib-insync  # For IBKR integration
pip install alpaca-trade-api  # For Alpaca integration
```

## Verification

Test your environment:

```bash
python -c "from CONFIG.config_loader import load_config; print('Config loader OK')"
python -c "from DATA_PROCESSING.pipeline import normalize_interval; print('Data processing OK')"
python -c "from TRAINING.model_fun import LightGBMTrainer; print('Training OK')"
```

## Troubleshooting

### Import Errors

If you see import errors, ensure:
1. Virtual environment is activated
2. PYTHONPATH includes project root
3. All dependencies are installed

### Path Issues

Use absolute paths in scripts or ensure you're running from the project root.

## Next Steps

- [GPU Setup](GPU_SETUP.md) - Configure GPU support
- [Configuration Basics](../configuration/CONFIG_BASICS.md) - Learn configuration

