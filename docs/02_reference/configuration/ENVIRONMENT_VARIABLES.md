# Environment Variables

Complete reference for environment variables used in Fox-v1-infra.

## Configuration Variables

### MODEL_VARIANT

Set default variant for all models:

```bash
export MODEL_VARIANT=conservative
```

### MODEL_CONFIG_DIR

Override configuration directory:

```bash
export MODEL_CONFIG_DIR=/custom/path/to/configs
```

## Data Variables

### DATA_DIR

Data directory path:

```bash
export DATA_DIR="./data"
```

### MODELS_DIR

Models directory path:

```bash
export MODELS_DIR="./models"
```

## Alpaca Trading

### ALPACA_API_KEY

Alpaca API key:

```bash
export ALPACA_API_KEY="your_api_key_id"
```

### ALPACA_SECRET_KEY

Alpaca secret key:

```bash
export ALPACA_SECRET_KEY="your_secret_key"
```

### ALPACA_BASE_URL

Alpaca API base URL:

```bash
export ALPACA_BASE_URL="https://paper-api.alpaca.markets"
```

## IBKR Trading

### IBKR_HOST

IBKR TWS/Gateway host:

```bash
export IBKR_HOST="127.0.0.1"
```

### IBKR_PORT

IBKR API port (7497 paper, 7496 live):

```bash
export IBKR_PORT=7497
```

### IBKR_CLIENT_ID

IBKR client ID:

```bash
export IBKR_CLIENT_ID=1
```

## GPU Configuration

### CUDA_VISIBLE_DEVICES

Select GPU device:

```bash
export CUDA_VISIBLE_DEVICES=0  # Use first GPU
export CUDA_VISIBLE_DEVICES=1  # Use second GPU
```

## Logging

### LOG_LEVEL

Set logging level:

```bash
export LOG_LEVEL="INFO"    # INFO, DEBUG, WARNING, ERROR
```

### LOG_DIR

Log directory:

```bash
export LOG_DIR="./logs"
```

## Python Path

### PYTHONPATH

Add project root to Python path:

```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

## Environment File

Create `.env` file:

```bash
# .env
MODEL_VARIANT=conservative
DATA_DIR=./data
MODELS_DIR=./models
ALPACA_API_KEY=your_key
ALPACA_SECRET_KEY=your_secret
LOG_LEVEL=INFO
```

Load with:

```python
from dotenv import load_dotenv
load_dotenv()
```

## See Also

- [Config Loader API](CONFIG_LOADER_API.md) - Config loading
- [Environment Setup](../../01_tutorials/setup/ENVIRONMENT_SETUP.md) - Setup guide

