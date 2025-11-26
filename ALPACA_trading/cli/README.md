# Command-Line Interface

This directory contains the CLI interface for interacting with the ALPACA trading system.

## Components

### `paper.py` - Paper Trading CLI
Command-line interface for paper trading operations.

**Features:**
- Interactive command interface
- Trade execution commands
- Position management
- Performance monitoring
- Configuration management

**Usage:**
```bash
python ALPACA_trading/cli/paper.py [command] [options]
```

**Commands:**
- `start` - Start paper trading
- `stop` - Stop paper trading
- `status` - Show current status
- `positions` - Show current positions
- `performance` - Show performance metrics
- `config` - Manage configuration

**Example:**
```bash
# Start trading
python ALPACA_trading/cli/paper.py start --symbols SPY,TSLA

# Check status
python ALPACA_trading/cli/paper.py status

# View positions
python ALPACA_trading/cli/paper.py positions
```

## Interactive Mode

The CLI supports interactive mode:
```bash
python ALPACA_trading/cli/paper.py
# Enters interactive shell
> help
> start --symbols SPY
> status
> exit
```

## Integration

The CLI interfaces with:
- `scripts/paper_runner.py` - Trading execution
- `core/paper.py` - Trading engine
- `brokers/paper.py` - Broker operations
- `core/performance.py` - Performance tracking

## Configuration

CLI uses configuration from:
- `config/paper_trading_config.json`
- Environment variables
- Command-line arguments (override config)

