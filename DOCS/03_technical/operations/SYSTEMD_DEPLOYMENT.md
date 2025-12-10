# Systemd Deployment

Deploy trading system as a systemd service.

## Overview

Systemd service provides:
- Automatic startup on boot
- Service management
- Logging integration
- Process monitoring

## Service File

**Note:** Trading integration modules have been removed from the core repository. This section is for reference only.

Example service file structure:

```ini
[Unit]
Description=ML Training System
After=network.target

[Service]
Type=simple
User=trading
WorkingDirectory=/home/trading/trader
ExecStart=/usr/bin/python3 training_script.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

## Deployment

### 1. Install Service

```bash
sudo cp systemd/your-service.service /etc/systemd/system/
sudo systemctl daemon-reload
```

### 2. Enable Service

```bash
sudo systemctl enable your-service
```

### 3. Start Service

```bash
sudo systemctl start your-service
```

### 4. Check Status

```bash
sudo systemctl status your-service
```

## Management

### Start/Stop/Restart

```bash
sudo systemctl start your-service
sudo systemctl stop your-service
sudo systemctl restart your-service
```

### View Logs

```bash
sudo journalctl -u your-service -f
```

## See Also

- [Journald Logging](JOURNALD_LOGGING.md) - Logging setup

