#!/bin/bash
# Install IBKR Trading System as systemd service

set -e

# Configuration
SERVICE_NAME="ibkr-trading"
SERVICE_FILE="ibkr-trading.service"
SYSTEMD_DIR="/etc/systemd/system"
PROJECT_DIR="/home/Jennifer/secure/trader/IBKR_trading"
USER="trader"
GROUP="trader"

echo "🚀 Installing IBKR Trading System as systemd service..."

# Check if running as root
if [[ $EUID -ne 0 ]]; then
   echo "❌ This script must be run as root (use sudo)"
   exit 1
fi

# Create trader user if it doesn't exist
if ! id "$USER" &>/dev/null; then
    echo "👤 Creating user: $USER"
    useradd -r -s /bin/bash -d /home/$USER -m $USER
    usermod -aG sudo $USER
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p $PROJECT_DIR/{logs,state,config}
mkdir -p /var/log/ibkr-trading
chown -R $USER:$GROUP $PROJECT_DIR
chown -R $USER:$GROUP /var/log/ibkr-trading

# Copy service file
echo "📋 Installing service file..."
cp $SERVICE_FILE $SYSTEMD_DIR/$SERVICE_FILE

# Set permissions
chmod 644 $SYSTEMD_DIR/$SERVICE_FILE
chown root:root $SYSTEMD_DIR/$SERVICE_FILE

# Reload systemd
echo "🔄 Reloading systemd..."
systemctl daemon-reload

# Enable service
echo "✅ Enabling service..."
systemctl enable $SERVICE_NAME

echo "🎉 Installation complete!"
echo ""
echo "To start the service:"
echo "  sudo systemctl start $SERVICE_NAME"
echo ""
echo "To check status:"
echo "  sudo systemctl status $SERVICE_NAME"
echo ""
echo "To view logs:"
echo "  sudo journalctl -u $SERVICE_NAME -f"
echo ""
echo "To stop the service:"
echo "  sudo systemctl stop $SERVICE_NAME"
echo ""
echo "To disable the service:"
echo "  sudo systemctl disable $SERVICE_NAME"
