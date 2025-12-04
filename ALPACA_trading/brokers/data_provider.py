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
IBKR Data Provider
Provides market data from Interactive Brokers with caching and fallback support
"""


import logging
import pickle
import time
from datetime import date as date_class
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from .ibkr_broker import IBKRBroker, IBKRConfig

logger = logging.getLogger(__name__)


class IBKRDataProvider:
    """
    Data provider for Interactive Brokers market data.
    Includes caching, fallback to yfinance, and data validation.
    """

    def __init__(
        self,
        config: IBKRConfig = None,
        cache_dir: str = "data/ibkr",
        use_cache: bool = True,
        fallback_to_yfinance: bool = True,
    ):
        """
        Initialize IBKR data provider.

        Args:
            config: IBKR configuration
            cache_dir: Directory for caching data
            use_cache: Whether to use data caching
            fallback_to_yfinance: Whether to fallback to yfinance if IBKR fails
        """
        self.config = config or IBKRConfig()
        self.cache_dir = Path(cache_dir)
        self.use_cache = use_cache
        self.fallback_to_yfinance = fallback_to_yfinance

        # Create cache directory
        if self.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize IBKR broker
        self.broker = None
        self._init_broker()

        # Cache for real-time data
        self._price_cache = {}
        self._cache_timeout = 30  # seconds

        logger.info(f"Initialized IBKRDataProvider with cache: {self.use_cache}")

    def _init_broker(self):
        """Initialize IBKR broker connection."""
        try:
            self.broker = IBKRBroker(config=self.config, auto_connect=False)
            logger.info("IBKR broker initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize IBKR broker: {e}")
            self.broker = None

    def _get_cache_path(self, symbol: str, duration: str, bar_size: str) -> Path:
        """Get cache file path for data."""
        # Create a safe filename
        safe_symbol = symbol.replace("/", "_").replace("\\", "_")
        safe_duration = duration.replace(" ", "_")
        safe_bar_size = bar_size.replace(" ", "_")

        filename = f"{safe_symbol}_{safe_duration}_{safe_bar_size}.pkl"
        return self.cache_dir / filename

    def _load_from_cache(self, symbol: str, duration: str, bar_size: str) -> pd.DataFrame | None:
        """Load data from cache with DataSanity validation."""
        if not self.use_cache:
            return None

        try:
            cache_path = self._get_cache_path(symbol, duration, bar_size)
            if cache_path.exists():
                # Check if cache is recent (within 1 hour for daily data, 5 minutes for intraday)
                cache_age = time.time() - cache_path.stat().st_mtime
                max_age = 3600 if "day" in bar_size else 300  # 1 hour for daily, 5 min for intraday

                if cache_age < max_age:
                    # Use DataSanity wrapper for loading and validation
                    from core.data_sanity import get_data_sanity_wrapper

                    wrapper = get_data_sanity_wrapper()
                    data = wrapper.load_and_validate(str(cache_path), symbol)
                    logger.debug(f"Loaded {symbol} data from cache with validation")
                    return data
                else:
                    logger.debug(f"Cache for {symbol} is stale, will refresh")

            return None

        except Exception as e:
            logger.warning(f"Failed to load cache for {symbol}: {e}")
            return None

    def _save_to_cache(self, symbol: str, duration: str, bar_size: str, data: pd.DataFrame):
        """Save data to cache."""
        if not self.use_cache:
            return

        try:
            cache_path = self._get_cache_path(symbol, duration, bar_size)
            with open(cache_path, "wb") as f:
                pickle.dump(data, f)
            logger.debug(f"Saved {symbol} data to cache")

        except Exception as e:
            logger.warning(f"Failed to save cache for {symbol}: {e}")

    def _fallback_to_yfinance(
        self, symbol: str, start_date: datetime, end_date: datetime
    ) -> pd.DataFrame | None:
        """Fallback to yfinance if IBKR fails."""
        if not self.fallback_to_yfinance:
            return None

        try:
            import yfinance as yf

            logger.info(f"Falling back to yfinance for {symbol}")
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date)

            if not data.empty:
                # Ensure column names match expected format
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)

                # Rename columns to match IBKR format
                column_mapping = {
                    "Open": "Open",
                    "High": "High",
                    "Low": "Low",
                    "Close": "Close",
                    "Volume": "Volume",
                }

                data = data.rename(columns=column_mapping)
                data = data[["Open", "High", "Low", "Close", "Volume"]]

                logger.info(f"Successfully fetched {symbol} data from yfinance")
                return data
            else:
                logger.warning(f"No data available from yfinance for {symbol}")
                return None

        except Exception as e:
            logger.error(f"Failed to fetch data from yfinance for {symbol}: {e}")
            return None

    def get_historical_data(
        self, symbol: str, duration: str = "1 Y", bar_size: str = "1 day"
    ) -> pd.DataFrame | None:
        """
        Get historical data for a symbol.

        Args:
            symbol: Stock symbol
            duration: Data duration (e.g., '1 D', '1 W', '1 M', '1 Y')
            bar_size: Bar size (e.g., '1 min', '5 mins', '1 hour', '1 day')

        Returns:
            DataFrame with OHLCV data
        """
        # Try cache first
        cached_data = self._load_from_cache(symbol, duration, bar_size)
        if cached_data is not None:
            return cached_data

        # Try IBKR
        if self.broker and self.broker.is_connected():
            try:
                logger.info(f"Fetching {symbol} data from IBKR")
                data = self.broker.get_market_data(symbol, duration, bar_size)

                if data is not None and not data.empty:
                    # Save to cache
                    self._save_to_cache(symbol, duration, bar_size, data)
                    return data

            except Exception as e:
                logger.warning(f"Failed to get data from IBKR for {symbol}: {e}")

        # Try to connect to IBKR if not connected
        if self.broker and not self.broker.is_connected():
            try:
                if self.broker.connect():
                    data = self.broker.get_market_data(symbol, duration, bar_size)
                    if data is not None and not data.empty:
                        self._save_to_cache(symbol, duration, bar_size, data)
                        return data
            except Exception as e:
                logger.warning(f"Failed to connect to IBKR: {e}")

        # Fallback to yfinance
        try:
            # Calculate start and end dates for yfinance
            end_date = datetime.now()

            # Parse duration for yfinance
            if "D" in duration:
                days = int(duration.split()[0])
                start_date = end_date - timedelta(days=days)
            elif "W" in duration:
                weeks = int(duration.split()[0])
                start_date = end_date - timedelta(weeks=weeks)
            elif "M" in duration:
                months = int(duration.split()[0])
                start_date = end_date - timedelta(days=months * 30)
            elif "Y" in duration:
                years = int(duration.split()[0])
                start_date = end_date - timedelta(days=years * 365)
            else:
                start_date = end_date - timedelta(days=365)  # Default to 1 year

            fallback_data = self._fallback_to_yfinance(symbol, start_date, end_date)
            if fallback_data is not None:
                self._save_to_cache(symbol, duration, bar_size, fallback_data)
                return fallback_data

        except Exception as e:
            logger.error(f"Failed to get fallback data for {symbol}: {e}")

        logger.error(f"Failed to get data for {symbol} from all sources")
        return None

    def get_real_time_price(self, symbol: str) -> float | None:
        """Get real-time price for a symbol."""
        # Check cache first
        if symbol in self._price_cache:
            cache_time, price = self._price_cache[symbol]
            if time.time() - cache_time < self._cache_timeout:
                return price

        # Try IBKR
        if self.broker and self.broker.is_connected():
            try:
                price = self.broker.get_real_time_price(symbol)
                if price is not None:
                    self._price_cache[symbol] = (time.time(), price)
                    return price
            except Exception as e:
                logger.warning(f"Failed to get real-time price from IBKR for {symbol}: {e}")

        # Try to connect to IBKR if not connected
        if self.broker and not self.broker.is_connected():
            try:
                if self.broker.connect():
                    price = self.broker.get_real_time_price(symbol)
                    if price is not None:
                        self._price_cache[symbol] = (time.time(), price)
                        return price
            except Exception as e:
                logger.warning(f"Failed to connect to IBKR: {e}")

        # Fallback to yfinance
        if self.fallback_to_yfinance:
            try:
                import yfinance as yf

                ticker = yf.Ticker(symbol)
                price = ticker.info.get("regularMarketPrice")
                if price is not None:
                    self._price_cache[symbol] = (time.time(), price)
                    return price
            except Exception as e:
                logger.warning(f"Failed to get real-time price from yfinance for {symbol}: {e}")

        return None

    def get_multiple_symbols_data(
        self, symbols: list[str], duration: str = "1 Y", bar_size: str = "1 day"
    ) -> dict[str, pd.DataFrame]:
        """
        Get historical data for multiple symbols.

        Args:
            symbols: List of stock symbols
            duration: Data duration
            bar_size: Bar size

        Returns:
            Dict mapping symbols to DataFrames
        """
        results = {}

        for symbol in symbols:
            try:
                data = self.get_historical_data(symbol, duration, bar_size)
                if data is not None:
                    results[symbol] = data
                else:
                    logger.warning(f"No data available for {symbol}")
            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {e}")

        logger.info(f"Successfully fetched data for {len(results)}/{len(symbols)} symbols")
        return results

    def get_daily_data(
        self, symbol: str, start_date: date_class, end_date: date_class
    ) -> pd.DataFrame | None:
        """
        Get daily data for a specific date range.

        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with daily OHLCV data
        """
        # Calculate duration
        days = (end_date - start_date).days
        duration = f"{days} D"

        return self.get_historical_data(symbol, duration, "1 day")

    def clear_cache(self, symbol: str | None = None):
        """Clear data cache."""
        if not self.use_cache:
            return

        try:
            if symbol:
                # Clear specific symbol cache
                pattern = f"{symbol}_*.pkl"
                for cache_file in self.cache_dir.glob(pattern):
                    cache_file.unlink()
                logger.info(f"Cleared cache for {symbol}")
            else:
                # Clear all cache
                for cache_file in self.cache_dir.glob("*.pkl"):
                    cache_file.unlink()
                logger.info("Cleared all cache")

        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")

    def get_cache_info(self) -> dict[str, Any]:
        """Get cache information."""
        if not self.use_cache:
            return {"cache_enabled": False}

        try:
            cache_files = list(self.cache_dir.glob("*.pkl"))
            total_size = sum(f.stat().st_size for f in cache_files)

            return {
                "cache_enabled": True,
                "cache_dir": str(self.cache_dir),
                "file_count": len(cache_files),
                "total_size_mb": total_size / (1024 * 1024),
                "files": [f.name for f in cache_files],
            }

        except Exception as e:
            logger.error(f"Failed to get cache info: {e}")
            return {"cache_enabled": True, "error": str(e)}

    def is_connected(self) -> bool:
        """Check if connected to IBKR."""
        return self.broker is not None and self.broker.is_connected()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self.broker:
            self.broker.disconnect()
        # Return False to re-raise any exceptions
        return False


def test_data_provider():
    """Test the data provider."""
    try:
        # Initialize data provider
        provider = IBKRDataProvider(use_cache=True, fallback_to_yfinance=True)

        print("Testing IBKR Data Provider...")

        # Test connection
        print(f"IBKR Connected: {provider.is_connected()}")

        # Test historical data
        symbols = ["SPY", "AAPL", "NVDA"]
        for symbol in symbols:
            print(f"\nFetching data for {symbol}...")
            data = provider.get_historical_data(symbol, "1 M", "1 day")

            if data is not None:
                print(
                    f"✅ {symbol}: {len(data)} rows, date range: {data.index[0]} to "
                    f"{data.index[-1]}"
                )
                print(f"   Columns: {list(data.columns)}")
                print(f"   Last close: ${data['Close'].iloc[-1]:.2f}")
            else:
                print(f"❌ {symbol}: No data available")

        # Test real-time prices
        print("\nTesting real-time prices...")
        for symbol in symbols:
            price = provider.get_real_time_price(symbol)
            if price:
                print(f"✅ {symbol}: ${price:.2f}")
            else:
                print(f"❌ {symbol}: No real-time price")

        # Test cache info
        cache_info = provider.get_cache_info()
        print(f"\nCache Info: {cache_info}")

        return True

    except Exception as e:
        print(f"❌ Error testing data provider: {e}")
        return False


if __name__ == "__main__":
    test_data_provider()
