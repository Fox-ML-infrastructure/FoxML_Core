# Core Trading Engine Components

This directory contains the core components of the ALPACA paper trading system.

## Components

### `paper.py` - Core Paper Trading Engine
The main trading engine that orchestrates all trading operations. Handles:
- Trading state management
- Order execution coordination
- Integration with brokers, strategies, and ML models
- Regime detection and adaptive feature reweighting
- Performance tracking and logging

**Key Features:**
- Regime-aware trading with automatic strategy selection
- Adaptive feature reweighting based on market conditions
- Integration with Discord notifications
- Growth target calculation and tracking

### `regime_detector.py` - Market Regime Detection
Detects current market regime (trending, choppy, volatile) to adapt trading strategies.

**Regimes:**
- **Trending**: Strong directional movement
- **Choppy**: Sideways/range-bound movement
- **Volatile**: High volatility, uncertain direction

**Usage:** Used by strategies to adjust parameters and signal weights based on market conditions.

### `strategy_selector.py` - Strategy Selection Logic
Selects the appropriate trading strategy based on:
- Current market regime
- Historical performance
- Risk parameters
- Market conditions

**Strategies Supported:**
- Regime-aware ensemble
- Trend-following
- Mean-reversion
- Custom strategies via factory pattern

### `performance.py` - Performance Tracking
Tracks and calculates trading performance metrics:
- Returns (total, daily, monthly)
- Sharpe ratio
- Maximum drawdown
- Win rate
- Growth targets

**Features:**
- Real-time performance monitoring
- Historical performance analysis
- Growth target calculation

### `risk/guardrails.py` - Risk Management Guardrails
Optional risk management system that enforces:
- Position size limits
- Maximum drawdown limits
- Daily loss limits
- Exposure limits

**Safety Features:**
- Automatic position reduction on risk violations
- Circuit breakers for extreme market conditions
- Real-time risk monitoring

### `telemetry/snapshot.py` - Performance Telemetry
Optional telemetry system for capturing system state snapshots:
- Trading state snapshots
- Performance metrics at specific points
- System health monitoring

### `data_sanity.py` - Data Validation
Validates incoming market data for:
- Missing values
- Outliers and anomalies
- Data quality checks
- Timestamp validation

**Purpose:** Ensures data quality before making trading decisions.

### `feature_reweighter.py` - Adaptive Feature Reweighting
Dynamically adjusts feature weights based on:
- Recent performance
- Market regime
- Feature importance over time

**Components:**
- `FeatureReweighter`: Base reweighting logic
- `AdaptiveFeatureEngine`: Advanced adaptive reweighting

### `notifications.py` - Notification System
Sends notifications for:
- Trade executions
- Performance milestones
- Error alerts
- System status updates

**Supported Channels:**
- Discord (via webhooks)
- Extensible to other channels

### `enhanced_logging.py` - Enhanced Logging System
Structured logging with:
- Color-coded console output
- Separate log files for trades, performance, errors, system
- Automatic log rotation
- Detailed trade logging

### `utils.py` - Utility Functions
Common utility functions for:
- Price normalization
- Performance metric calculation
- Directory management
- Logging setup

## Integration

All core components work together through the `PaperTradingEngine`:
1. **Regime Detection** → Identifies market conditions
2. **Strategy Selection** → Chooses appropriate strategy
3. **Feature Reweighting** → Adjusts feature weights
4. **Data Validation** → Ensures data quality
5. **Order Execution** → Executes trades via broker
6. **Performance Tracking** → Monitors results
7. **Risk Management** → Enforces guardrails
8. **Logging & Notifications** → Records and alerts

## Configuration

Core components are configured via:
- `config/base.yaml` - Base configuration
- `config/paper_trading_config.json` - Paper trading settings
- Environment variables for sensitive data

