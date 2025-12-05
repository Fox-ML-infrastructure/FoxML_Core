# Journald Logging

Systemd journal logging setup and configuration.

## Overview

Journald provides centralized logging for the trading system with:
- Structured logs
- Log rotation
- Search and filtering
- Integration with systemd

## Configuration

### Enable Journald Logging

```python
import systemd.journal

logger = systemd.journal.JournalHandler()
```

### Log Levels

- **DEBUG**: Detailed debugging information
- **INFO**: General information
- **WARNING**: Warning messages
- **ERROR**: Error messages
- **CRITICAL**: Critical errors

## Usage

### View Logs

```bash
# All logs
journalctl -u ibkr_trading

# Recent logs
journalctl -u ibkr_trading -n 100

# Follow logs
journalctl -u ibkr_trading -f

# Filter by level
journalctl -u ibkr_trading -p err
```

### Search Logs

```bash
# Search by text
journalctl -u ibkr_trading | grep "ERROR"

# Search by time
journalctl -u ibkr_trading --since "2025-12-05 10:00:00"
```

## Best Practices

1. **Use Structured Logs**: Include context (symbol, timestamp, etc.)
2. **Appropriate Levels**: Use correct log levels
3. **Monitor Regularly**: Check logs for issues
4. **Rotate Logs**: Configure log rotation

## See Also

- [Journald Logging](../../JOURNALD_LOGGING.md) - Complete guide
- [Restore from Logs](RESTORE_FROM_LOGS.md) - Recovery procedures

