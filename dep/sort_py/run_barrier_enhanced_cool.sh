#!/bin/bash
# Run Smart Barrier Processing with Enhanced Targets in "Cool Mode"
# Outputs to: barrier_Target_5m_cool
# Includes: Basic barriers + ZigZag + MFE/MDD + TTH + Ordinal + Path Quality + Asymmetric

set -e

echo "❄️ Starting Enhanced Barrier Processing in Cool Mode"
echo "================================================================="
echo "📊 New target families:"
echo "  ✅ Time-to-hit (TTH) - regression on time to barrier"
echo "  ✅ Ordinal buckets - multiclass return magnitude"
echo "  ✅ Path quality - MFE share, time in profit, flip count"
echo "  ✅ Asymmetric barriers - separate TP/SL targets"
echo ""

# Configuration for cooler operation
# Use absolute path to trader/data directory
TRADER_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${DATA_DIR:-${TRADER_ROOT}/data/5m_comprehensive_features_hft_with_5m_10m}"
OUTPUT_DIR="${OUTPUT_DIR:-${TRADER_ROOT}/data/data_labeled}"
HORIZONS="5 10 15 30 60"
BARRIER_SIZES="0.3 0.5 0.8"
N_WORKERS=4         # Reduced from 8
BATCH_SIZE=10       # Reduced from 20
THROTTLE_DELAY=0.5  # Added delay between symbol processing

# Create logs directory
mkdir -p logs

echo "📊 Processing configuration (Cool Mode):"
echo "  Data directory: $DATA_DIR"
echo "  Output directory: $OUTPUT_DIR (NEW!)"
echo "  Horizons: $HORIZONS"
echo "  Barrier sizes: $BARRIER_SIZES"
echo "  Workers: $N_WORKERS"
echo "  Batch size: $BATCH_SIZE"
echo "  Throttle delay: ${THROTTLE_DELAY}s"
echo ""

# Check if output directory exists and has partial data
if [ -d "$OUTPUT_DIR" ]; then
    echo "📁 Output directory exists - will resume from last processed symbol"
else
    echo "📁 Creating new output directory: $OUTPUT_DIR"
fi

# Run with lower priority and in background
# Use 'nice' to set a lower priority (19 is lowest)
# Use 'nohup' to allow it to run after terminal closes
# Redirect stdout/stderr to a log file
nice -n 19 nohup python3 smart_barrier_processing.py \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --horizons $HORIZONS \
    --barrier-sizes $BARRIER_SIZES \
    --n-workers $N_WORKERS \
    --batch-size $BATCH_SIZE \
    --throttle-delay $THROTTLE_DELAY \
    > logs/barrier_enhanced_cool_$(date +%Y%m%d_%H%M%S).log 2>&1 &

PROCESS_PID=$!

echo "✅ Enhanced barrier processing started in background!"
echo "🆔 Process ID: $PROCESS_PID"
echo "📁 Output directory: $OUTPUT_DIR"
echo "📊 Log file: logs/barrier_enhanced_cool_*.log"
echo ""
echo "🔍 Monitor with: tail -f logs/barrier_enhanced_cool_*.log"
echo "🛑 Stop with: kill $PROCESS_PID"
echo "📊 Check progress: ps aux | grep smart_barrier_processing"
echo ""
echo "Note: The script will automatically resume from where it left off if restarted."
echo "      Enhanced targets include: TTH, ordinal, path quality, asymmetric barriers"

