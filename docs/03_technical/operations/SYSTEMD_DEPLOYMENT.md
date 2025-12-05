# Systemd Deployment

Deploy trading system as a systemd service.

## Overview

Systemd service provides:
- Automatic startup on boot
- Service management
- Logging integration
- Process monitoring

## Service File

Create `/etc/systemd/system/ibkr-trading.service`:

```ini
[Unit]
Description=IBKR Trading System
After=network.target

[Service]
Type=simple
User=trading
WorkingDirectory=/home/trading/trader/IBKR_trading
ExecStart=/usr/bin/python3 run_trading_system.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

## Deployment

### 1. Install Service

```bash
sudo cp IBKR_trading/systemd/ibkr-trading.service /etc/systemd/system/
sudo systemctl daemon-reload
```

### 2. Enable Service

```bash
sudo systemctl enable ibkr-trading
```

### 3. Start Service

```bash
sudo systemctl start ibkr-trading
```

### 4. Check Status

```bash
sudo systemctl status ibkr-trading
```

## Management

### Start/Stop/Restart

```bash
sudo systemctl start ibkr-trading
sudo systemctl stop ibkr-trading
sudo systemctl restart ibkr-trading
```

### View Logs

```bash
sudo journalctl -u ibkr-trading -f
```

## See Also

- [Systemd Deployment Plan](../../../IBKR_trading/systemd/) - Service files
- [Journald Logging](JOURNALD_LOGGING.md) - Logging setup

