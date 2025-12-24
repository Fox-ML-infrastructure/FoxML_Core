# Configuration Files

This directory contains all configuration files for the ALPACA trading system.

## Configuration Files

### `base.yaml` - Base Configuration
Base configuration with default settings for:
- Trading parameters
- Risk limits
- Feature settings
- Model settings

**Purpose:** Provides defaults that can be overridden by other config files.

### `models.yaml` - Model Registry Configuration
Defines all available ML models:
- Model paths and types
- Feature lists
- Target columns
- Task types (regression/classification)

**Format:**
```yaml
models:
  - name: "model_name"
    path: "models/model.pkl"
    kind: "pickle"
    features: ["feature1", "feature2"]
    target: "target_column"
    task_type: "regression"
```

### `paper_trading_config.json` - Paper Trading Settings
Main configuration for paper trading:
- Symbols to trade
- Risk parameters
- Strategy selection
- Broker settings
- Notification settings

**Key Settings:**
- `symbols`: List of symbols to trade
- `risk_profile`: Risk level (balanced, low, strict)
- `strategy`: Strategy to use
- `position_sizing`: Position sizing rules
- `max_drawdown`: Maximum drawdown limit

### `paper_config.json` - Paper Trading Configuration
Alternative/additional paper trading configuration.

### `paper_strict.yaml` - Strict Paper Trading Configuration
Strict risk management configuration:
- Lower position limits
- Tighter drawdown limits
- More conservative strategy parameters

**Use Case:** For testing with very conservative risk parameters.

### `enhanced_paper_trading_config.json` - Enhanced Configuration
Enhanced configuration with additional features:
- Advanced regime detection settings
- Feature reweighting parameters
- Performance targets
- Advanced notifications

### `enhanced_paper_trading_config_unified.json` - Unified Enhanced Configuration
Unified version combining all enhanced features.

### `paper-trading.env` - Environment Variables Template
Template for environment variables:
- `ALPACA_API_KEY` - Alpaca API key
- `ALPACA_SECRET_KEY` - Alpaca secret key
- `ALPACA_BASE_URL` - API base URL (optional)
- `DISCORD_WEBHOOK_URL` - Discord notifications (optional)

**Usage:** Copy to `.env` and fill in your values.

## Configuration Hierarchy

Configuration is loaded in this order (later overrides earlier):
1. `base.yaml` - Base defaults
2. `paper_trading_config.json` - Main config
3. Profile-specific config (if specified)
4. Environment variables
5. Command-line arguments

## Risk Profiles

Predefined risk profiles:
- **risk_balanced**: Balanced risk/return
- **risk_low**: Conservative, lower risk
- **risk_strict**: Very conservative, strict limits

## Validation

All configuration files are validated on load:
- Required fields checked
- Value ranges validated
- Type checking
- Dependency validation

## Security

**Important:** Never commit actual API keys or secrets to version control.
- Use `paper-trading.env` template
- Add `.env` to `.gitignore`
- Use environment variables for sensitive data

