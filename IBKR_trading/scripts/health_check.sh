#!/bin/bash
# IBKR Trading System Health Check Script

set -e

# Configuration
SERVICE_NAME="ibkr-trading"
PROJECT_DIR="/home/Jennifer/secure/trader/IBKR_trading"
LOG_FILE="/var/log/ibkr-trading/health_check.log"
MAX_MEMORY_GB=8
MAX_CPU_PERCENT=80
MAX_ERROR_RATE=0.1

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

# Check if service is running
check_service_status() {
    log "🔍 Checking service status..."
    
    if ! systemctl is-active --quiet "$SERVICE_NAME"; then
        log "❌ Service is not running"
        return 1
    fi
    
    if ! systemctl is-enabled --quiet "$SERVICE_NAME"; then
        log "⚠️  Service is not enabled for auto-start"
        return 1
    fi
    
    log "✅ Service is running and enabled"
    return 0
}

# Check for panic flag
check_panic_flag() {
    log "🔍 Checking for panic flag..."
    
    if [ -f "$PROJECT_DIR/panic.flag" ]; then
        log "❌ Panic flag detected - system in emergency state"
        return 1
    fi
    
    log "✅ No panic flag detected"
    return 0
}

# Check memory usage
check_memory_usage() {
    log "🔍 Checking memory usage..."
    
    MEMORY_BYTES=$(systemctl show "$SERVICE_NAME" --property=MemoryCurrent --value)
    MEMORY_GB=$((MEMORY_BYTES / 1024 / 1024 / 1024))
    
    if [ "$MEMORY_GB" -gt "$MAX_MEMORY_GB" ]; then
        log "❌ Memory usage too high: ${MEMORY_GB}GB (max: ${MAX_MEMORY_GB}GB)"
        return 1
    fi
    
    log "✅ Memory usage OK: ${MEMORY_GB}GB"
    return 0
}

# Check CPU usage
check_cpu_usage() {
    log "🔍 Checking CPU usage..."
    
    # Get CPU usage from systemctl (this is a simplified check)
    CPU_USAGE=$(ps -p $(pgrep -f "ibkr_live_exec.py") -o %cpu --no-headers | awk '{sum+=$1} END {print sum}')
    
    if [ -z "$CPU_USAGE" ]; then
        log "⚠️  Could not determine CPU usage"
        return 0
    fi
    
    if (( $(echo "$CPU_USAGE > $MAX_CPU_PERCENT" | bc -l) )); then
        log "❌ CPU usage too high: ${CPU_USAGE}% (max: ${MAX_CPU_PERCENT}%)"
        return 1
    fi
    
    log "✅ CPU usage OK: ${CPU_USAGE}%"
    return 0
}

# Check error rate in logs
check_error_rate() {
    log "🔍 Checking error rate in logs..."
    
    # Count errors in the last hour
    ERROR_COUNT=$(journalctl -u "$SERVICE_NAME" --since "1 hour ago" | grep -c "ERROR" || true)
    TOTAL_LOGS=$(journalctl -u "$SERVICE_NAME" --since "1 hour ago" | wc -l)
    
    if [ "$TOTAL_LOGS" -eq 0 ]; then
        log "⚠️  No logs found in the last hour"
        return 0
    fi
    
    ERROR_RATE=$(echo "scale=4; $ERROR_COUNT / $TOTAL_LOGS" | bc -l)
    MAX_ERROR_RATE_DECIMAL=$(echo "scale=4; $MAX_ERROR_RATE" | bc -l)
    
    if (( $(echo "$ERROR_RATE > $MAX_ERROR_RATE_DECIMAL" | bc -l) )); then
        log "❌ Error rate too high: ${ERROR_RATE} (max: ${MAX_ERROR_RATE})"
        return 1
    fi
    
    log "✅ Error rate OK: ${ERROR_RATE} (${ERROR_COUNT}/${TOTAL_LOGS} logs)"
    return 0
}

# Check disk space
check_disk_space() {
    log "🔍 Checking disk space..."
    
    DISK_USAGE=$(df "$PROJECT_DIR" | awk 'NR==2 {print $5}' | sed 's/%//')
    
    if [ "$DISK_USAGE" -gt 90 ]; then
        log "❌ Disk usage too high: ${DISK_USAGE}%"
        return 1
    fi
    
    log "✅ Disk usage OK: ${DISK_USAGE}%"
    return 0
}

# Check network connectivity
check_network() {
    log "🔍 Checking network connectivity..."
    
    # Check if IBKR Gateway is reachable (assuming localhost:7496)
    if ! nc -z localhost 7496 2>/dev/null; then
        log "⚠️  IBKR Gateway not reachable on localhost:7496"
        return 1
    fi
    
    log "✅ Network connectivity OK"
    return 0
}

# Check recent trading activity
check_trading_activity() {
    log "🔍 Checking recent trading activity..."
    
    # Look for recent trading decisions in logs
    RECENT_DECISIONS=$(journalctl -u "$SERVICE_NAME" --since "10 minutes ago" | grep -c "decision" || true)
    
    if [ "$RECENT_DECISIONS" -eq 0 ]; then
        log "⚠️  No recent trading activity detected"
        return 1
    fi
    
    log "✅ Trading activity detected: ${RECENT_DECISIONS} recent decisions"
    return 0
}

# Main health check function
main() {
    log "🚀 Starting IBKR Trading System health check..."
    
    local exit_code=0
    
    # Run all health checks
    check_service_status || exit_code=1
    check_panic_flag || exit_code=1
    check_memory_usage || exit_code=1
    check_cpu_usage || exit_code=1
    check_error_rate || exit_code=1
    check_disk_space || exit_code=1
    check_network || exit_code=1
    check_trading_activity || exit_code=1
    
    if [ $exit_code -eq 0 ]; then
        log "✅ All health checks passed - system is healthy"
        echo -e "${GREEN}✅ System is healthy${NC}"
    else
        log "❌ Health checks failed - system needs attention"
        echo -e "${RED}❌ System needs attention${NC}"
    fi
    
    exit $exit_code
}

# Run main function
main "$@"
