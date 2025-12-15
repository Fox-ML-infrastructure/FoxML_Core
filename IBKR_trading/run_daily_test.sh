#!/bin/bash

# Run IBKR Daily Model Test
# This tests the IBKR stack with your current daily models

echo "🚀 IBKR Daily Model Test"
echo "========================"

# Create logs directory
mkdir -p logs

# Run the test
cd /home/Jennifer/secure/trader
python IBKR_trading/test_daily_models.py

echo ""
echo "✅ Daily model test completed!"
echo "📊 Check logs/daily_model_test.log for details"
echo ""
echo "🔄 When your intraday models are ready, run:"
echo "   python IBKR_trading/test_daily_models.py"
echo "   tester.switch_to_intraday()"
