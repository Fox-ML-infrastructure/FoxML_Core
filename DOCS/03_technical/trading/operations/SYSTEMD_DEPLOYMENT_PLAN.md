# IBKR Trading System - Systemd Deployment Plan

## **Systemd Integration Strategy**

### **Service Architecture Overview**

The IBKR trading system will be deployed as a robust systemd service with the following characteristics:

- **High Availability**: Automatic restart on failure
- **Resource Management**: CPU and memory limits
- **Security**: Restricted permissions and sandboxing
- **Monitoring**: Comprehensive logging and health checks
- **Recovery**: Graceful shutdown and panic procedures

## ️ **Service Configuration**

### **1. Main Service (`ibkr-trading.service`)**

**Location**: `/etc/systemd/system/ibkr-trading.service`

**Key Features**:
- **User Isolation**: Runs as dedicated `trader` user
- **Resource Limits**: 8GB memory, 400% CPU quota
- **Security**: No new privileges, private tmp, protected system
- **Restart Policy**: Always restart with exponential backoff
- **Logging**: Journal integration with structured logging

### **2. Service Dependencies**

**Required Services**:
- `network.target` - Network connectivity
- `time-sync.target` - Time synchronization
- `ibkr-gateway.service` - IBKR Gateway (if running locally)

**Optional Dependencies**:
- `redis.service` - Redis for caching (if used)
- `postgresql.service` - Database (if used)

## **Installation Process**

### **Step 1: System Preparation**
```bash
# Create trader user
sudo useradd -r -s /bin/bash -d /home/trader -m trader
sudo usermod -aG sudo trader

# Create directories
sudo mkdir -p /home/trader/secure/trader/IBKR_trading/{logs,state,config}
sudo chown -R trader:trader /home/trader/secure/trader/IBKR_trading
```

### **Step 2: Service Installation**
```bash
# Copy service file
sudo cp IBKR_trading/systemd/ibkr-trading.service /etc/systemd/system/

# Set permissions
sudo chmod 644 /etc/systemd/system/ibkr-trading.service
sudo chown root:root /etc/systemd/system/ibkr-trading.service

# Reload systemd
sudo systemctl daemon-reload
```

### **Step 3: Service Activation**
```bash
# Enable service (start on boot)
sudo systemctl enable ibkr-trading

# Start service
sudo systemctl start ibkr-trading

# Check status
sudo systemctl status ibkr-trading
```

## **Monitoring & Health Checks**

### **1. Service Health Monitoring**

**Status Commands**:
```bash
# Check service status
sudo systemctl status ibkr-trading

# View recent logs
sudo journalctl -u ibkr-trading -f

# Check resource usage
sudo systemctl show ibkr-trading --property=MemoryCurrent,CPUUsageNSec
```

**Health Check Script**:
```bash
#!/bin/bash
# /home/trader/secure/trader/IBKR_trading/scripts/health_check.sh

# Check if service is running
if ! systemctl is-active --quiet ibkr-trading; then
    echo " Service is not running"
    exit 1
fi

# Check if panic flag exists
if [ -f "panic.flag" ]; then
    echo " Panic flag detected"
    exit 1
fi

# Check memory usage
MEMORY_USAGE=$(systemctl show ibkr-trading --property=MemoryCurrent --value)
if [ "$MEMORY_USAGE" -gt 8000000000 ]; then  # 8GB
    echo " Memory usage too high: $MEMORY_USAGE"
    exit 1
fi

echo " Service is healthy"
exit 0
```

### **2. Performance Monitoring**

**Metrics Collection**:
```bash
# CPU usage
systemctl show ibkr-trading --property=CPUUsageNSec

# Memory usage
systemctl show ibkr-trading --property=MemoryCurrent

# Process count
systemctl show ibkr-trading --property=TasksCurrent
```

**Log Analysis**:
```bash
# Error rate
sudo journalctl -u ibkr-trading --since "1 hour ago" | grep -c "ERROR"

# Performance metrics
sudo journalctl -u ibkr-trading --since "1 hour ago" | grep "latency_ms"
```

## **Emergency Procedures**

### **1. Panic Procedures**

**Immediate Stop**:
```bash
# Stop service immediately
sudo systemctl stop ibkr-trading

# Create panic flag
touch /home/trader/secure/trader/IBKR_trading/panic.flag

# Flatten all positions (if broker connected)
# This would be handled by the panic flag detection in the service
```

**Recovery**:
```bash
# Remove panic flag
rm /home/trader/secure/trader/IBKR_trading/panic.flag

# Restart service
sudo systemctl start ibkr-trading
```

### **2. Service Recovery**

**Automatic Recovery**:
- **Restart Policy**: Always restart on failure
- **Backoff**: Exponential backoff (10s, 20s, 40s)
- **Max Restarts**: 3 restarts per 60 seconds
- **Timeout**: 30 seconds for graceful shutdown

**Manual Recovery**:
```bash
# Restart service
sudo systemctl restart ibkr-trading

# Reload configuration
sudo systemctl reload ibkr-trading

# Check logs for issues
sudo journalctl -u ibkr-trading --since "5 minutes ago"
```

## **Security Configuration**

### **1. User Isolation**
- **Dedicated User**: `trader` user with minimal privileges
- **No Root Access**: Service runs as non-root user
- **Restricted Directories**: Only access to trading directories

### **2. System Protection**
- **No New Privileges**: Service cannot escalate privileges
- **Private Tmp**: Isolated temporary directory
- **Protected System**: Read-only system directories
- **Protected Home**: Restricted home directory access

### **3. Network Security**
- **Localhost Only**: IBKR connection to localhost only
- **Firewall**: Restrict outbound connections
- **VPN**: Use VPN for external connections

## **Performance Optimization**

### **1. Resource Limits**
- **Memory**: 8GB maximum
- **CPU**: 400% quota (4 cores)
- **File Descriptors**: 65536 maximum
- **Processes**: 4096 maximum

### **2. Scheduling**
- **Priority**: High priority for trading operations
- **CPU Affinity**: Pin to specific cores
- **I/O Priority**: High priority for disk I/O

### **3. Monitoring**
- **Metrics**: CPU, memory, disk I/O, network
- **Alerts**: Automated alerts for resource usage
- **Logging**: Structured logging for analysis

## ️ **Maintenance Procedures**

### **1. Regular Maintenance**

**Daily**:
- Check service status
- Review error logs
- Monitor resource usage
- Verify trading performance

**Weekly**:
- Update service configuration
- Review performance metrics
- Clean up old logs
- Test emergency procedures

**Monthly**:
- Full system health check
- Performance optimization review
- Security audit
- Backup verification

### **2. Update Procedures**

**Service Updates**:
```bash
# Stop service
sudo systemctl stop ibkr-trading

# Update code
git pull origin main

# Restart service
sudo systemctl start ibkr-trading
```

**Configuration Updates**:
```bash
# Update configuration
sudo cp new_config.yaml /home/trader/secure/trader/IBKR_trading/config/

# Reload service
sudo systemctl reload ibkr-trading
```

## **Deployment Checklist**

### **Pre-Deployment**
- [ ] System requirements met
- [ ] Dependencies installed
- [ ] User accounts created
- [ ] Directories created
- [ ] Permissions set

### **Deployment**
- [ ] Service file installed
- [ ] Configuration files copied
- [ ] Service enabled
- [ ] Service started
- [ ] Status verified

### **Post-Deployment**
- [ ] Health checks passing
- [ ] Performance metrics collected
- [ ] Logs monitored
- [ ] Emergency procedures tested
- [ ] Documentation updated

## **Success Metrics**

### **Operational Metrics**
- **Uptime**: 99.9% availability
- **Restart Rate**: <1 restart per day
- **Resource Usage**: <80% of limits
- **Error Rate**: <0.1% error rate

### **Trading Metrics**
- **Decision Latency**: <350ms (p99)
- **Order Latency**: <2s (p99)
- **Fill Rate**: >90%
- **Net P&L**: Positive after costs

---

**Next Steps**:
1. Install the systemd service using the provided script
2. Configure monitoring and alerting
3. Test emergency procedures
4. Begin performance optimization
