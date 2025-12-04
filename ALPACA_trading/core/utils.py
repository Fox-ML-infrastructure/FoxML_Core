"""
Copyright (c) 2025 Fox ML Infrastructure

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

"""
Utility functions for the trading system.
Consolidates common functionality to reduce code duplication.
"""


import json
import logging
import os
from typing import Any

import numpy as np
import pandas as pd

# Constants
DEFAULT_COMMISSION_BPS = 5.0
DEFAULT_SLIPPAGE_BPS = 2.0
MIN_HISTORY_DAYS = 60
MAX_POSITION_SIZE_PCT = 0.1
DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def setup_logging(
    log_file: str = "trading.log",
    level: int = logging.INFO,
    format_string: str | None = None,
) -> logging.Logger:
    """
    Setup logging configuration with file and console handlers.

    Args:
        log_file: Path to log file
        level: Logging level
        format_string: Custom format string (uses default if None)

    Returns:
        Configured logger
    """
    # Ensure log directory exists
    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    # Use default format if none provided
    if format_string is None:
        format_string = DEFAULT_LOG_FORMAT

    # Configure logging
    logging.basicConfig(
        level=level,
        format=format_string,
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )

    return logging.getLogger(__name__)


def normalize_prices(prices: pd.Series | np.ndarray) -> pd.Series | np.ndarray:
    """
    Normalize price data by handling NaN and infinite values.

    Args:
        prices: Price series or array

    Returns:
        Normalized prices with NaN/inf values replaced
    """
    if isinstance(prices, pd.Series):
        return prices.fillna(method="ffill").fillna(method="bfill")
    else:
        prices = np.asarray(prices)
        prices = np.nan_to_num(prices, nan=np.nan, posinf=np.nan, neginf=np.nan)
        # Forward fill then backward fill
        mask = np.isnan(prices)
        if mask.any():
            prices = pd.Series(prices).fillna(method="ffill").fillna(method="bfill").values
        return prices


def apply_slippage(
    price: float, quantity: float, slippage_bps: float = DEFAULT_SLIPPAGE_BPS
) -> float:
    """
    Apply slippage to trade execution.

    Args:
        price: Base price
        quantity: Trade quantity (positive for buy, negative for sell)
        slippage_bps: Slippage in basis points

    Returns:
        Adjusted price with slippage
    """
    slippage_multiplier = 1 + (slippage_bps / 10000)
    if quantity > 0:  # Buy
        return price * slippage_multiplier
    else:  # Sell
        return price / slippage_multiplier


def calculate_drawdown(equity_curve: pd.Series | np.ndarray) -> float:
    """
    Calculate maximum drawdown from equity curve.

    Args:
        equity_curve: Series of portfolio values

    Returns:
        Maximum drawdown as a percentage
    """
    if isinstance(equity_curve, pd.Series):
        equity_curve = equity_curve.values

    peak = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - peak) / peak
    return float(np.min(drawdown))


def validate_trade(
    symbol: str,
    quantity: float,
    price: float,
    cash: float,
    action: str | None = None,
    positions: dict[str, float] | None = None,
    risk_limits: dict[str, float] | None = None,
) -> tuple[bool, str]:
    """
    Comprehensive trade validation function.

    Args:
        symbol: Trading symbol
        quantity: Trade quantity
        price: Trade price
        cash: Available cash
        action: Trade action ("BUY" or "SELL")
        positions: Current positions dictionary
        risk_limits: Risk limit dictionary

    Returns:
        (is_valid, error_message)
    """
    # Basic validation
    if not symbol or not symbol.strip():
        return False, "Invalid symbol"

    if quantity == 0:
        return False, "Zero quantity"

    if price <= 0:
        return False, "Invalid price"

    # Cash validation
    trade_value = abs(quantity * price)
    if trade_value > cash:
        return False, f"Insufficient cash: {trade_value} > {cash}"

    # Position validation for sell orders
    if action == "SELL" and positions is not None:
        current_position = positions.get(symbol, 0)
        if current_position < quantity:
            return (
                False,
                f"Insufficient shares for sell: {quantity} > {current_position}",
            )

    # Risk limit validation
    if risk_limits is not None:
        max_position_size = risk_limits.get("max_position_size", MAX_POSITION_SIZE_PCT)
        max_position_value = cash * max_position_size
        if trade_value > max_position_value:
            return (
                False,
                f"Position size exceeds limit: {trade_value} > {max_position_value}",
            )

    return True, ""


def calculate_returns(close: pd.Series, shift: int = -1) -> pd.Series:
    """
    Calculate returns from close prices.

    Args:
        close: Close price series
        shift: Number of periods to shift (default -1 for next-period returns)

    Returns:
        Returns series
    """
    return close.pct_change().shift(shift)


def calculate_performance_metrics(
    equity: pd.Series, start_date: str, end_date: str
) -> dict[str, float]:
    """
    Calculate comprehensive performance metrics.

    Args:
        equity: Equity curve series
        start_date: Start date string
        end_date: End date string

    Returns:
        Dictionary of performance metrics
    """
    if len(equity) < 2:
        return {
            "total_return": 0.0,
            "cagr": 0.0,
            "sharpe": 0.0,
            "max_dd": 0.0,
            "volatility": 0.0,
        }

    # Calculate returns
    rets = equity.pct_change().dropna()

    # Total return
    total_return = (equity.iloc[-1] / equity.iloc[0]) - 1

    # CAGR
    years = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days / 365.25
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1 if years > 0 else 0.0

    # Sharpe ratio
    sharpe = rets.mean() / (rets.std() + 1e-6) * np.sqrt(252)

    # Max drawdown
    rolling_max = equity.expanding().max()
    drawdown = (equity - rolling_max) / rolling_max
    max_dd = drawdown.min()

    # Volatility
    volatility = rets.std() * np.sqrt(252)

    return {
        "total_return": total_return,
        "cagr": cagr,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "volatility": volatility,
    }


def ensure_directories(base_dir: str = ".") -> None:
    """
    Ensure required directories exist.

    Args:
        base_dir: Base directory path
    """
    folders = [
        os.path.join(base_dir, "data"),
        os.path.join(base_dir, "models"),
        os.path.join(base_dir, "features"),
        os.path.join(base_dir, "results"),
        os.path.join(base_dir, "logs"),
        os.path.join(base_dir, "signals"),
        os.path.join(base_dir, "dashboard"),
    ]

    for folder in folders:
        os.makedirs(folder, exist_ok=True)


def _safe_len(x) -> int:
    """Safely get the length of any sequence-like object."""
    return 0 if x is None else (len(x) if hasattr(x, "__len__") else 0)


def _last(x):
    """Safely get the last element of any sequence-like object."""
    if x is None or _safe_len(x) == 0:
        return None
    if hasattr(x, "iloc"):
        return x.iloc[-1]
    return x[-1]


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, handling division by zero.

    Args:
        numerator: Numerator
        denominator: Denominator
        default: Default value if denominator is zero

    Returns:
        Division result or default value
    """
    if denominator == 0 or np.isnan(denominator):
        return default
    return numerator / denominator


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean DataFrame by handling common data issues.

    Args:
        df: Input DataFrame

    Returns:
        Cleaned DataFrame
    """
    # Handle MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)

    # Fill infinite values
    df = df.replace([np.inf, -np.inf], np.nan)

    # Fill NaN values with 0 for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)

    return df


def validate_strategy_params(params: dict[str, Any], required_params: list[str]) -> bool:
    """
    Validate strategy parameters.

    Args:
        params: Parameter dictionary
        required_params: List of required parameter names

    Returns:
        True if valid, False otherwise
    """
    return all(param in params for param in required_params)


def format_percentage(value: float, decimals: int = 2) -> str:
    """
    Format value as percentage string.

    Args:
        value: Value to format
        decimals: Number of decimal places

    Returns:
        Formatted percentage string
    """
    return f"{value:.{decimals}%}"


def format_currency(value: float, decimals: int = 2) -> str:
    """
    Format value as currency string.

    Args:
        value: Value to format
        decimals: Number of decimal places

    Returns:
        Formatted currency string
    """
    return f"${value:.{decimals}f}"


def load_config(config_file: str) -> dict[str, Any]:
    """
    Load configuration from JSON file with error handling.

    Args:
        config_file: Path to configuration file

    Returns:
        Configuration dictionary
    """
    try:
        with open(config_file) as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        logging.error(f"Config file not found: {config_file}")
        return {}
    except json.JSONDecodeError as e:
        logging.error(f"Invalid JSON in config: {e}")
        return {}


def validate_dataframe(df: pd.DataFrame, required_columns: list[str]) -> bool:
    """
    Validate DataFrame has required columns.

    Args:
        df: DataFrame to validate
        required_columns: List of required column names

    Returns:
        True if valid, False otherwise
    """
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        logging.error(f"Missing columns: {missing_columns}")
        return False
    return True


def validate_numeric_range(value: float, min_val: float, max_val: float, name: str) -> bool:
    """
    Validate numeric value is within range.

    Args:
        value: Value to validate
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        name: Name of the parameter for error messages

    Returns:
        True if valid, False otherwise
    """
    if not (min_val <= value <= max_val):
        logging.error(f"{name} must be between {min_val} and {max_val}, got {value}")
        return False
    return True
