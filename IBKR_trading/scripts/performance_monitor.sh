#!/bin/bash
# IBKR Trading System Performance Monitor

set -e

# Configuration
SERVICE_NAME="ibkr-trading"
PROJECT_DIR="/home/Jennifer/secure/trader/IBKR_trading"
LOG_FILE="/var/log/ibkr-trading/performance.log"
METRICS_FILE="/var/log/ibkr-trading/metrics.json"

# Create log directory
mkdir -p "$(dirname "$LOG_FILE")"
mkdir -p "$(dirname "$METRICS_FILE")"

# Logging function
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

# Get system metrics
get_system_metrics() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    # Service metrics
    local memory_bytes=$(systemctl show "$SERVICE_NAME" --property=MemoryCurrent --value 2>/dev/null || echo "0")
    local memory_gb=$(echo "scale=2; $memory_bytes / 1024 / 1024 / 1024" | bc -l)
    
    # CPU usage
    local cpu_usage=$(ps -p $(pgrep -f "ibkr_live_exec.py") -o %cpu --no-headers 2>/dev/null | awk '{sum+=$1} END {print sum+0}' || echo "0")
    
    # Process count
    local process_count=$(systemctl show "$SERVICE_NAME" --property=TasksCurrent --value 2>/dev/null || echo "0")
    
    # Disk usage
    local disk_usage=$(df "$PROJECT_DIR" | awk 'NR==2 {print $5}' | sed 's/%//')
    
    # Network connections
    local network_connections=$(ss -tuln | wc -l)
    
    # Log metrics to JSON
    cat >> "$METRICS_FILE" << EOF
{
  "timestamp": "$timestamp",
  "service": "$SERVICE_NAME",
  "metrics": {
    "memory_gb": $memory_gb,
    "cpu_percent": $cpu_usage,
    "process_count": $process_count,
    "disk_usage_percent": $disk_usage,
    "network_connections": $network_connections
  }
}
EOF
    
    log "📊 Metrics collected: Memory=${memory_gb}GB, CPU=${cpu_usage}%, Processes=${process_count}, Disk=${disk_usage}%"
}

# Get trading metrics
get_trading_metrics() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    # Count recent trading decisions
    local decisions_count=$(journalctl -u "$SERVICE_NAME" --since "1 minute ago" | grep -c "decision" || echo "0")
    
    # Count recent orders
    local orders_count=$(journalctl -u "$SERVICE_NAME" --since "1 minute ago" | grep -c "order submitted" || echo "0")
    
    # Count errors
    local errors_count=$(journalctl -u "$SERVICE_NAME" --since "1 minute ago" | grep -c "ERROR" || echo "0")
    
    # Count warnings
    local warnings_count=$(journalctl -u "$SERVICE_NAME" --since "1 minute ago" | grep -c "WARNING" || echo "0")
    
    # Log trading metrics
    log "📈 Trading metrics: Decisions=${decisions_count}, Orders=${orders_count}, Errors=${errors_count}, Warnings=${warnings_count}"
}

# Get latency metrics
get_latency_metrics() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    # Extract latency from logs (assuming logs contain "latency_ms" field)
    local avg_latency=$(journalctl -u "$SERVICE_NAME" --since "1 minute ago" | grep "latency_ms" | awk '{sum+=$NF; count++} END {if(count>0) print sum/count; else print 0}' || echo "0")
    
    # Extract decision time from logs
    local avg_decision_time=$(journalctl -u "$SERVICE_NAME" --since "1 minute ago" | grep "decision_time" | awk '{sum+=$NF; count++} END {if(count>0) print sum/count; else print 0}' || echo "0")
    
    log "⏱️  Latency metrics: Avg Latency=${avg_latency}ms, Avg Decision Time=${avg_decision_time}ms"
}

# Generate performance report
generate_report() {
    local report_file="/var/log/ibkr-trading/performance_report.txt"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    cat > "$report_file" << EOF
IBKR Trading System Performance Report
Generated: $timestamp
=====================================

Service Status:
$(systemctl status "$SERVICE_NAME" --no-pager)

Memory Usage:
$(systemctl show "$SERVICE_NAME" --property=MemoryCurrent,MemoryPeak --no-pager)

CPU Usage:
$(ps -p $(pgrep -f "ibkr_live_exec.py") -o pid,ppid,cmd,%cpu,%mem --no-headers 2>/dev/null || echo "Process not found")

Recent Logs (last 10 lines):
$(journalctl -u "$SERVICE_NAME" --no-pager -n 10)

Disk Usage:
$(df -h "$PROJECT_DIR")

Network Connections:
$(ss -tuln | grep -E "(7496|7497)" || echo "No IBKR connections found")

Recent Errors (last 5):
$(journalctl -u "$SERVICE_NAME" --since "1 hour ago" | grep "ERROR" | tail -5 || echo "No recent errors")

Recent Warnings (last 5):
$(journalctl -u "$SERVICE_NAME" --since "1 hour ago" | grep "WARNING" | tail -5 || echo "No recent warnings")
EOF
    
    log "📋 Performance report generated: $report_file"
}

# Monitor performance in real-time
monitor_realtime() {
    log "🔄 Starting real-time performance monitoring..."
    
    while true; do
        get_system_metrics
        get_trading_metrics
        get_latency_metrics
        
        # Generate report every 5 minutes
        if [ $(($(date +%s) % 300)) -eq 0 ]; then
            generate_report
        fi
        
        sleep 10
    done
}

# Generate one-time report
generate_onetime_report() {
    log "📊 Generating one-time performance report..."
    
    get_system_metrics
    get_trading_metrics
    get_latency_metrics
    generate_report
    
    log "✅ Performance report complete"
}

# Main function
main() {
    case "${1:-report}" in
        "monitor")
            monitor_realtime
            ;;
        "report")
            generate_onetime_report
            ;;
        "help")
            echo "Usage: $0 [monitor|report|help]"
            echo "  monitor - Start real-time monitoring"
            echo "  report  - Generate one-time report"
            echo "  help    - Show this help"
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use '$0 help' for usage information"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"
