# Broker Integration

This directory contains broker interface implementations for connecting to trading platforms.

## Components

### `interface.py` - Broker Protocol
Defines the abstract interface that all broker implementations must follow.

**Protocol Methods:**
- `submit_order()` - Submit buy/sell orders
- `cancel_order()` - Cancel existing orders
- `get_positions()` - Get current positions
- `get_cash()` - Get available cash balance
- `get_account_info()` - Get account information
- `get_market_data()` - Get real-time market data

**Purpose:** Provides a consistent interface regardless of the underlying broker API.

### `paper.py` - Alpaca Paper Trading Broker
Implementation of the broker interface for Alpaca Markets paper trading API.

**Features:**
- Paper trading (no real money)
- Order submission and cancellation
- Position and account management
- Market data retrieval
- Real-time quote access

**API:** Uses `alpaca-trade-api` or `alpaca-py` Python libraries.

**Configuration:**
- `ALPACA_API_KEY` - API key (from Alpaca dashboard)
- `ALPACA_SECRET_KEY` - Secret key
- `ALPACA_BASE_URL` - API base URL (default: paper trading endpoint)

### `data_provider.py` - Data Provider Interface
Provides market data for backtesting and analysis.

**Features:**
- Historical data retrieval
- Real-time data streaming
- Multiple data sources support

**Current Implementation:** `IBKRDataProvider` - Provides data from IBKR (if configured)

### `ibkr_broker.py` - IBKR Broker Integration
Optional integration with Interactive Brokers (IBKR) for:
- Live trading (if configured)
- Data access
- Order execution

**Note:** This is an optional component. The primary broker for ALPACA trading is the Alpaca paper broker.

## Usage

```python
from brokers.paper import PaperBroker
from brokers.interface import Broker

# Initialize broker
broker = PaperBroker(
    api_key="your_key",
    secret_key="your_secret",
    base_url="https://paper-api.alpaca.markets"
)

# Submit order
order = broker.submit_order(
    symbol="SPY",
    side="BUY",
    qty=10,
    order_type="market"
)

# Get positions
positions = broker.get_positions()
```

## Broker Selection

The trading engine automatically selects the appropriate broker based on configuration:
- **Paper Trading**: Uses `PaperBroker` (Alpaca)
- **Live Trading**: Would use `IBKRBroker` (if configured)

## Error Handling

All broker implementations include:
- Connection retry logic
- Error logging
- Order status validation
- Timeout handling

## Testing

Brokers can be tested in isolation:
- Paper broker uses Alpaca's paper trading environment
- No real money at risk
- Full API functionality available

