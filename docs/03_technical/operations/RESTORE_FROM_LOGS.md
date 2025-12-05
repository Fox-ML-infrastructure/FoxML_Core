# Restore from Logs

Procedures for recovering system state from logs.

## Overview

Logs contain sufficient information to restore system state after failures or for analysis.

## Recovery Process

### 1. Identify Last Known State

```bash
# Find last successful operation
journalctl -u ibkr_trading | grep "SUCCESS"

# Find last position update
journalctl -u ibkr_trading | grep "position"
```

### 2. Extract State Information

Logs contain:
- Positions
- Orders
- Model predictions
- System state

### 3. Reconstruct State

Use log data to reconstruct:
- Current positions
- Pending orders
- Model states
- Configuration

## Log Analysis

### Extract Positions

```bash
journalctl -u ibkr_trading | grep "position" | tail -20
```

### Extract Orders

```bash
journalctl -u ibkr_trading | grep "order" | tail -20
```

### Extract Errors

```bash
journalctl -u ibkr_trading -p err | tail -50
```

## Automated Recovery

Scripts can parse logs and restore state:

```python
from scripts.restore_from_logs import restore_state

state = restore_state(log_file="logs/ibkr_trading.log")
```

## Best Practices

1. **Regular Backups**: Backup state regularly
2. **Structured Logs**: Use structured logging for easier parsing
3. **Monitor Logs**: Watch for issues in real-time
4. **Test Recovery**: Regularly test recovery procedures

## See Also

- [Journald Logging](JOURNALD_LOGGING.md) - Logging setup
- [Restore from Logs](../../RESTORE_FROM_LOGS.md) - Complete guide

