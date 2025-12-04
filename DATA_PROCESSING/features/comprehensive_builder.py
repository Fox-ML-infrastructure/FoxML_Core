#!/usr/bin/env python3

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
Comprehensive Feature Builder - Builds 200+ features for ranking pipeline
"""


import argparse
import gc
import logging
import sys
import time
import yaml
from pathlib import Path
from typing import List, Dict, Any
import polars as pl
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ml.memory import MemoryManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ComprehensiveFeatureBuilder:
    """Builds comprehensive feature set for ranking pipeline"""
    
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.feature_config = self.config.get('features', {})
        self.memory_manager = MemoryManager()
        
    def build_features(self, input_paths: List[str], output_dir: str, universe_config: str):
        """Build comprehensive features"""
        logger.info(f"Building comprehensive features for {len(input_paths)} input files")
        logger.info(f"Output directory: {output_dir}")
        
        # Load universe
        with open(universe_config, 'r') as f:
            universe = yaml.safe_load(f)
        symbols = universe.get('symbols', universe.get('universe', []))
        
        # Restrict to symbols in universe
        input_symbols = self._extract_symbols_from_paths(input_paths)
        symbols_to_process = [s for s in input_symbols if s in symbols]
        
        logger.info(f"Processing {len(symbols_to_process)} symbols (restricted from {len(symbols)} universe)")
        logger.info(f"Symbols to process: {symbols_to_process[:10]}{'...' if len(symbols_to_process) > 10 else ''}")
        
        # Create output directory
        output_path = Path(f"storage/features/{output_dir}")
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Process each symbol
        for i, symbol in enumerate(symbols_to_process):
            logger.info(f"🔄 Processing symbol: {symbol}")
            self._process_symbol(symbol, input_paths, output_path)
            
            # Memory cleanup
            if i % 10 == 0:
                gc.collect()
                logger.info(f"📊 Memory at feature building batch {i//10 + 1}: {self.memory_manager.get_system_memory_usage()}")
        
        logger.info(f"✅ Feature building completed! Processed {len(symbols_to_process)}/{len(symbols_to_process)} symbols")
        
    def _extract_symbols_from_paths(self, input_paths: List[str]) -> List[str]:
        """Extract symbols from input paths"""
        symbols = set()
        for path in input_paths:
            if 'symbol=' in path:
                symbol = path.split('symbol=')[1].split('/')[0]
                symbols.add(symbol)
            elif '/symbol=' in path:
                symbol = path.split('/symbol=')[1].split('/')[0]
                symbols.add(symbol)
        return sorted(list(symbols))
    
    def _process_symbol(self, symbol: str, input_paths: List[str], output_path: Path):
        """Process a single symbol"""
        # Find files for this symbol
        symbol_files = [p for p in input_paths if f'symbol={symbol}' in p or f'/symbol={symbol}/' in p]
        if not symbol_files:
            logger.warning(f"No files found for symbol {symbol}")
            return
        
        # Load and process data
        features = self._load_symbol_data(symbol_files, symbol)
        if features is None:
            return
        
        # Build comprehensive features
        features = self._build_comprehensive_features(features)
        
        # Write output
        self._write_features(features, symbol, output_path)
        
        logger.info(f"✅ {symbol}: comprehensive features built successfully")
    
    def _load_symbol_data(self, symbol_files: List[str], symbol: str) -> pl.LazyFrame:
        """Load data for a symbol"""
        try:
            # Use glob pattern to find all files for this symbol
            pattern = f"data/liquid/bars/interval=1h/symbol={symbol}/date=*/*.parquet"
            scan = pl.scan_parquet(pattern, hive_partitioning=True, low_memory=True)
            
            # Basic validation
            sample = scan.limit(100).collect()
            if len(sample) == 0:
                logger.warning(f"No data found for {symbol}")
                return None
            
            logger.info(f"Loaded {len(sample)} sample rows for {symbol}")
            return scan
            
        except Exception as e:
            logger.error(f"Error loading data for {symbol}: {e}")
            return None
    
    def _build_comprehensive_features(self, scan: pl.LazyFrame) -> pl.LazyFrame:
        """Build comprehensive feature set"""
        return (
            scan
            .with_columns([
                # Basic price features
                (pl.col("high") - pl.col("low")).alias("range_frac").cast(pl.Float32),
                (pl.col("close") / pl.col("open") - 1).alias("ret_oc").cast(pl.Float32),
                (pl.col("open") / pl.col("close").shift(1) - 1).alias("gap_co").cast(pl.Float32),
                
                # VWAP features
                ((pl.col("close") - pl.col("vwap")) / pl.col("vwap")).alias("vwap_dev").cast(pl.Float32),
                ((pl.col("high") - pl.col("vwap")) / pl.col("vwap")).alias("vwap_dev_high").cast(pl.Float32),
                ((pl.col("low") - pl.col("vwap")) / pl.col("vwap")).alias("vwap_dev_low").cast(pl.Float32),
                
                # Overnight features
                (pl.col("open") / pl.col("close").shift(1)).log().abs().alias("overnight_vol").cast(pl.Float32),
                (pl.col("open") / pl.col("close").shift(1) - 1).alias("overnight_return").cast(pl.Float32),
                
                # Range features
                (pl.col("close") / pl.col("high")).alias("close_loc_1d").cast(pl.Float32),
                (pl.col("open") / pl.col("high")).alias("open_loc_1d").cast(pl.Float32),
                (pl.col("low") / pl.col("high")).alias("low_loc_1d").cast(pl.Float32),
                
                # Volume features
                (pl.col("close") * pl.col("volume")).alias("dollar_volume").cast(pl.Float32),
                (pl.col("volume") / pl.col("volume").rolling_mean(20)).alias("turnover_20").cast(pl.Float32),
                (pl.col("volume") / pl.col("volume").rolling_mean(5)).alias("turnover_5").cast(pl.Float32),
                
                # Price momentum
                pl.col("close").pct_change().alias("ret_1m").cast(pl.Float32),
                pl.col("close").pct_change(5).alias("ret_5m").cast(pl.Float32),
                pl.col("close").pct_change(15).alias("ret_15m").cast(pl.Float32),
                pl.col("close").pct_change(30).alias("ret_30m").cast(pl.Float32),
                pl.col("close").pct_change(60).alias("ret_60m").cast(pl.Float32),
                
                # Returns
                pl.col("close").pct_change().alias("returns_1d").cast(pl.Float32),
                pl.col("close").pct_change(5).alias("returns_5d").cast(pl.Float32),
                pl.col("close").pct_change(20).alias("returns_20d").cast(pl.Float32),
                
                # Volatility
                pl.col("close").pct_change().rolling_std(5).alias("vol_5m").cast(pl.Float32),
                pl.col("close").pct_change().rolling_std(15).alias("vol_15m").cast(pl.Float32),
                pl.col("close").pct_change().rolling_std(30).alias("vol_30m").cast(pl.Float32),
                pl.col("close").pct_change().rolling_std(60).alias("vol_60m").cast(pl.Float32),
                pl.col("close").pct_change().rolling_std(20).alias("volatility_20d").cast(pl.Float32),
                pl.col("close").pct_change().rolling_std(60).alias("volatility_60d").cast(pl.Float32),
                
                # Momentum
                pl.col("close").pct_change(1).alias("mom_1d").cast(pl.Float32),
                pl.col("close").pct_change(3).alias("mom_3d").cast(pl.Float32),
                pl.col("close").pct_change(5).alias("mom_5d").cast(pl.Float32),
                pl.col("close").pct_change(10).alias("mom_10d").cast(pl.Float32),
                pl.col("close").pct_change(5).alias("price_momentum_5d").cast(pl.Float32),
                pl.col("close").pct_change(20).alias("price_momentum_20d").cast(pl.Float32),
                pl.col("close").pct_change(60).alias("price_momentum_60d").cast(pl.Float32),
                
                # Technical indicators - SMA
                pl.col("close").rolling_mean(5).alias("sma_5").cast(pl.Float32),
                pl.col("close").rolling_mean(10).alias("sma_10").cast(pl.Float32),
                pl.col("close").rolling_mean(20).alias("sma_20").cast(pl.Float32),
                pl.col("close").rolling_mean(50).alias("sma_50").cast(pl.Float32),
                pl.col("close").rolling_mean(200).alias("sma_200").cast(pl.Float32),
                
                # Technical indicators - EMA
                pl.col("close").ewm_mean(span=5).alias("ema_5").cast(pl.Float32),
                pl.col("close").ewm_mean(span=10).alias("ema_10").cast(pl.Float32),
                pl.col("close").ewm_mean(span=20).alias("ema_20").cast(pl.Float32),
                pl.col("close").ewm_mean(span=50).alias("ema_50").cast(pl.Float32),
                
                # RSI
                self._rsi(pl.col("close"), 7).cast(pl.Float32).alias("rsi_7"),
                self._rsi(pl.col("close"), 14).cast(pl.Float32).alias("rsi_14"),
                self._rsi(pl.col("close"), 21).cast(pl.Float32).alias("rsi_21"),
                
                # MACD
                self._macd(pl.col("close"))[0],
                self._macd(pl.col("close"))[1], 
                self._macd(pl.col("close"))[2],
                
                # Bollinger Bands  
                self._bollinger_bands(pl.col("close"))[0],
                self._bollinger_bands(pl.col("close"))[1],
                self._bollinger_bands(pl.col("close"))[2],
                self._bollinger_bands(pl.col("close"))[3],
                
                # Stochastic
                self._stochastic(pl.col("high"), pl.col("low"), pl.col("close"))[0],
                self._stochastic(pl.col("high"), pl.col("low"), pl.col("close"))[1],
                
                # Williams %R
                self._williams_r(pl.col("high"), pl.col("low"), pl.col("close")),
                
                # CCI
                self._cci(pl.col("high"), pl.col("low"), pl.col("close")),
                
                # ATR
                self._atr(pl.col("high"), pl.col("low"), pl.col("close")),
                
                # ADX
                self._adx(pl.col("high"), pl.col("low"), pl.col("close")),
                
                # Volume features
                pl.col("volume").pct_change(5).alias("volume_momentum_5d").cast(pl.Float32),
                pl.col("volume").pct_change(20).alias("volume_momentum_20d").cast(pl.Float32),
                (pl.col("volume").rolling_mean(5) / pl.col("volume").rolling_mean(20)).alias("volume_ratio_5d").cast(pl.Float32),
                (pl.col("volume").rolling_mean(20) / pl.col("volume").rolling_mean(60)).alias("volume_ratio_20d").cast(pl.Float32),
                pl.col("volume").rolling_std(20).alias("volume_volatility_20d").cast(pl.Float32),
                
                # Volume SMAs
                pl.col("volume").rolling_mean(5).alias("volume_sma_5").cast(pl.Float32),
                pl.col("volume").rolling_mean(20).alias("volume_sma_20").cast(pl.Float32),
                pl.col("volume").rolling_mean(50).alias("volume_sma_50").cast(pl.Float32),
                
                # Volume EMAs
                pl.col("volume").ewm_mean(span=5).alias("volume_ema_5").cast(pl.Float32),
                pl.col("volume").ewm_mean(span=20).alias("volume_ema_20").cast(pl.Float32),
                
                # Time features
                pl.col("ts").dt.hour().alias("_hour").cast(pl.Int16),
                pl.col("ts").dt.weekday().alias("_weekday").cast(pl.Int16),
                pl.col("ts").dt.day().alias("_day").cast(pl.Int16),
                pl.col("ts").dt.month().alias("_month").cast(pl.Int16),
                pl.col("ts").dt.quarter().alias("_quarter").cast(pl.Int16),
                pl.col("ts").dt.year().alias("_year").cast(pl.Int16),
            ])
            .with_columns([
                # Hour dummies
                (pl.col("_hour") == 9).cast(pl.UInt8).alias("hour_9"),
                (pl.col("_hour") == 10).cast(pl.UInt8).alias("hour_10"),
                (pl.col("_hour") == 11).cast(pl.UInt8).alias("hour_11"),
                (pl.col("_hour") == 12).cast(pl.UInt8).alias("hour_12"),
                (pl.col("_hour") == 13).cast(pl.UInt8).alias("hour_13"),
                (pl.col("_hour") == 14).cast(pl.UInt8).alias("hour_14"),
                (pl.col("_hour") == 15).cast(pl.UInt8).alias("hour_15"),
                
                # Weekday dummies
                (pl.col("_weekday") == 0).cast(pl.UInt8).alias("wd_0"),
                (pl.col("_weekday") == 1).cast(pl.UInt8).alias("wd_1"),
                (pl.col("_weekday") == 2).cast(pl.UInt8).alias("wd_2"),
                (pl.col("_weekday") == 3).cast(pl.UInt8).alias("wd_3"),
                (pl.col("_weekday") == 4).cast(pl.UInt8).alias("wd_4"),
                
                # Month dummies
                (pl.col("_month") == 1).cast(pl.UInt8).alias("month_1"),
                (pl.col("_month") == 2).cast(pl.UInt8).alias("month_2"),
                (pl.col("_month") == 3).cast(pl.UInt8).alias("month_3"),
                (pl.col("_month") == 4).cast(pl.UInt8).alias("month_4"),
                (pl.col("_month") == 5).cast(pl.UInt8).alias("month_5"),
                (pl.col("_month") == 6).cast(pl.UInt8).alias("month_6"),
                (pl.col("_month") == 7).cast(pl.UInt8).alias("month_7"),
                (pl.col("_month") == 8).cast(pl.UInt8).alias("month_8"),
                (pl.col("_month") == 9).cast(pl.UInt8).alias("month_9"),
                (pl.col("_month") == 10).cast(pl.UInt8).alias("month_10"),
                (pl.col("_month") == 11).cast(pl.UInt8).alias("month_11"),
                (pl.col("_month") == 12).cast(pl.UInt8).alias("month_12"),
                
                # Quarter dummies
                (pl.col("_quarter") == 1).cast(pl.UInt8).alias("quarter_1"),
                (pl.col("_quarter") == 2).cast(pl.UInt8).alias("quarter_2"),
                (pl.col("_quarter") == 3).cast(pl.UInt8).alias("quarter_3"),
                (pl.col("_quarter") == 4).cast(pl.UInt8).alias("quarter_4"),
                
                # Advanced features
                (pl.col("sma_5") / pl.col("sma_20")).alias("sma_ratio_5_20").cast(pl.Float32),
                (pl.col("sma_10") / pl.col("sma_50")).alias("sma_ratio_10_50").cast(pl.Float32),
                (pl.col("sma_20") / pl.col("sma_200")).alias("sma_ratio_20_200").cast(pl.Float32),
                (pl.col("ema_5") / pl.col("ema_20")).alias("ema_ratio_5_20").cast(pl.Float32),
                (pl.col("ema_10") / pl.col("ema_50")).alias("ema_ratio_10_50").cast(pl.Float32),
                
                # Price ratios
                (pl.col("close") / pl.col("sma_20")).alias("price_ma_ratio").cast(pl.Float32),
                (pl.col("close") / pl.col("ema_20")).alias("price_ema_ratio").cast(pl.Float32),
                
                # Volatility ratios
                (pl.col("vol_5m") / pl.col("vol_15m")).alias("vol_ratio_5_15").cast(pl.Float32),
                (pl.col("vol_15m") / pl.col("vol_60m")).alias("vol_ratio_15_60").cast(pl.Float32),
                
                # Return/volatility ratios
                (pl.col("ret_5m") / pl.col("vol_5m")).alias("ret_vol_ratio_5m").cast(pl.Float32),
                (pl.col("ret_15m") / pl.col("vol_15m")).alias("ret_vol_ratio_15m").cast(pl.Float32),
                (pl.col("ret_60m") / pl.col("vol_60m")).alias("ret_vol_ratio_60m").cast(pl.Float32),
                
                # Non-linear features
                (pl.col("ret_5m") ** 2).alias("ret2_5m").cast(pl.Float32),
                (pl.col("ret_15m") ** 2).alias("ret2_15m").cast(pl.Float32),
                (pl.col("ret_60m") ** 2).alias("ret2_60m").cast(pl.Float32),
                
                # Interaction features
                (pl.col("vol_15m") * pl.col("ret_5m")).alias("vol_x_ret_5m").cast(pl.Float32),
                (pl.col("vol_15m") * pl.col("ret_15m")).alias("vol_x_ret_15m").cast(pl.Float32),
                (pl.col("vol_15m") * pl.col("ret_60m")).alias("vol_x_ret_60m").cast(pl.Float32),
                (pl.col("ret_5m") * pl.col("mom_5d")).alias("ret_x_mom_5d").cast(pl.Float32),
                (pl.col("ret_20d") * pl.col("mom_20d")).alias("ret_x_mom_20d").cast(pl.Float32),
                (pl.col("vol_5m") * pl.col("mom_5d")).alias("vol_x_mom_5d").cast(pl.Float32),
                (pl.col("vol_20d") * pl.col("mom_20d")).alias("vol_x_mom_20d").cast(pl.Float32),
                
                # Technical interaction features
                (pl.col("rsi_14") * pl.col("vol_15m")).alias("rsi_x_vol").cast(pl.Float32),
                (pl.col("bb_percent_b_20") * pl.col("vol_15m")).alias("bb_x_vol").cast(pl.Float32),
                
                # Forward returns for target
                pl.col("close").pct_change(-1).alias("fwd_ret_1d").cast(pl.Float32),
            ])
            .select([
                "symbol", "ts", "open", "high", "low", "close", "volume", "vwap",
                "range_frac", "ret_oc", "gap_co", "vwap_dev", "vwap_dev_high", "vwap_dev_low",
                "overnight_vol", "overnight_return", "close_loc_1d", "open_loc_1d", "low_loc_1d",
                "dollar_volume", "turnover_20", "turnover_5",
                "ret_1m", "ret_5m", "ret_15m", "ret_30m", "ret_60m",
                "returns_1d", "returns_5d", "returns_20d",
                "vol_5m", "vol_15m", "vol_30m", "vol_60m", "volatility_20d", "volatility_60d",
                "mom_1d", "mom_3d", "mom_5d", "mom_10d", "price_momentum_5d", "price_momentum_20d", "price_momentum_60d",
                "sma_5", "sma_10", "sma_20", "sma_50", "sma_200",
                "ema_5", "ema_10", "ema_20", "ema_50",
                "rsi_7", "rsi_14", "rsi_21",
                "macd", "macd_signal", "macd_hist",
                "bb_upper_20", "bb_lower_20", "bb_width_20", "bb_percent_b_20",
                "stoch_k_14", "stoch_d_14", "williams_r_14", "cci_20", "atr_14", "adx_14",
                "volume_momentum_5d", "volume_momentum_20d", "volume_ratio_5d", "volume_ratio_20d", "volume_volatility_20d",
                "volume_sma_5", "volume_sma_20", "volume_sma_50", "volume_ema_5", "volume_ema_20",
                "hour_9", "hour_10", "hour_11", "hour_12", "hour_13", "hour_14", "hour_15",
                "wd_0", "wd_1", "wd_2", "wd_3", "wd_4",
                "month_1", "month_2", "month_3", "month_4", "month_5", "month_6",
                "month_7", "month_8", "month_9", "month_10", "month_11", "month_12",
                "quarter_1", "quarter_2", "quarter_3", "quarter_4",
                "sma_ratio_5_20", "sma_ratio_10_50", "sma_ratio_20_200",
                "ema_ratio_5_20", "ema_ratio_10_50",
                "price_ma_ratio", "price_ema_ratio",
                "vol_ratio_5_15", "vol_ratio_15_60",
                "ret_vol_ratio_5m", "ret_vol_ratio_15m", "ret_vol_ratio_60m",
                "ret2_5m", "ret2_15m", "ret2_60m",
                "vol_x_ret_5m", "vol_x_ret_15m", "vol_x_ret_60m",
                "ret_x_mom_5d", "ret_x_mom_20d", "vol_x_mom_5d", "vol_x_mom_20d",
                "rsi_x_vol", "bb_x_vol",
                "fwd_ret_1d"
            ])
        )
    
    def _write_features(self, features: pl.LazyFrame, symbol: str, output_path: Path):
        """Write features to output"""
        try:
            df = features.collect()
            if len(df) == 0:
                logger.warning(f"No data to write for {symbol}")
                return
            
            # Add partitioning columns
            df = df.with_columns([
                pl.lit("1h").alias("interval"),
                pl.col("ts").dt.date().alias("date")
            ])
            
            # Write to partitioned parquet
            symbol_path = output_path / f"interval=1h/symbol={symbol}"
            symbol_path.mkdir(parents=True, exist_ok=True)
            
            output_file = symbol_path / "features.parquet"
            df.write_parquet(output_file, compression="zstd")
            
            logger.info(f"Wrote {len(df)} rows for {symbol}")
            
        except Exception as e:
            logger.error(f"Error writing features for {symbol}: {e}")
    
    # Technical indicator helper methods
    def _rsi(self, close: pl.Expr, window: int) -> pl.Expr:
        """Calculate RSI"""
        delta = close.diff()
        gain = pl.when(delta > 0).then(delta).otherwise(0)
        loss = pl.when(delta < 0).then(-delta).otherwise(0)
        avg_gain = gain.rolling_mean(window)
        avg_loss = loss.rolling_mean(window)
        rs = avg_gain / avg_loss
        return (100 - (100 / (1 + rs))).alias(f"rsi_{window}")
    
    def _macd(self, close: pl.Expr) -> List[pl.Expr]:
        """Calculate MACD"""
        ema_12 = close.ewm_mean(span=12)
        ema_26 = close.ewm_mean(span=26)
        macd = ema_12 - ema_26
        signal = macd.ewm_mean(span=9)
        hist = macd - signal
        return [
            macd.alias("macd"),
            signal.alias("macd_signal"),
            hist.alias("macd_hist")
        ]
    
    def _bollinger_bands(self, close: pl.Expr) -> List[pl.Expr]:
        """Calculate Bollinger Bands"""
        sma = close.rolling_mean(20)
        std = close.rolling_std(20)
        upper = (sma + (std * 2)).alias("bb_upper_20")
        lower = (sma - (std * 2)).alias("bb_lower_20")
        width = (upper - lower).alias("bb_width_20")
        percent_b = ((close - lower) / (upper - lower)).alias("bb_percent_b_20")
        return [upper, lower, width, percent_b]
    
    def _stochastic(self, high: pl.Expr, low: pl.Expr, close: pl.Expr) -> List[pl.Expr]:
        """Calculate Stochastic Oscillator"""
        lowest_low = low.rolling_min(14)
        highest_high = high.rolling_max(14)
        k_percent = (100 * (close - lowest_low) / (highest_high - lowest_low)).alias("stoch_k_14")
        d_percent = k_percent.rolling_mean(3).alias("stoch_d_14")
        return [k_percent, d_percent]
    
    def _williams_r(self, high: pl.Expr, low: pl.Expr, close: pl.Expr) -> pl.Expr:
        """Calculate Williams %R"""
        highest_high = high.rolling_max(14)
        lowest_low = low.rolling_min(14)
        return (-100 * (highest_high - close) / (highest_high - lowest_low)).alias("williams_r_14")
    
    def _cci(self, high: pl.Expr, low: pl.Expr, close: pl.Expr) -> pl.Expr:
        """Calculate Commodity Channel Index"""
        typical_price = (high + low + close) / 3
        sma_tp = typical_price.rolling_mean(20)
        mean_dev = (typical_price - sma_tp).abs().rolling_mean(20)
        return ((typical_price - sma_tp) / (0.015 * mean_dev)).alias("cci_20")
    
    def _atr(self, high: pl.Expr, low: pl.Expr, close: pl.Expr) -> pl.Expr:
        """Calculate Average True Range"""
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        true_range = pl.max_horizontal([tr1, tr2, tr3])
        return true_range.rolling_mean(14).alias("atr_14")
    
    def _adx(self, high: pl.Expr, low: pl.Expr, close: pl.Expr) -> pl.Expr:
        """Calculate Average Directional Index (simplified)"""
        # Simplified ADX calculation
        plus_dm = pl.when(high.diff() > low.diff().abs()).then(high.diff()).otherwise(0)
        minus_dm = pl.when(low.diff().abs() > high.diff()).then(low.diff().abs()).otherwise(0)
        plus_di = 100 * (plus_dm.rolling_mean(14) / self._atr(high, low, close))
        minus_di = 100 * (minus_dm.rolling_mean(14) / self._atr(high, low, close))
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
        return dx.rolling_mean(14).alias("adx_14")

def main():
    """Main function"""
    memory_manager = MemoryManager()
    logger.info(f"System memory usage: {memory_manager.get_system_memory_usage()}")
    logger.info(f"GPU memory usage: {memory_manager.get_gpu_memory_usage()}")
    
    try:
        parser = argparse.ArgumentParser(description="Comprehensive Feature Builder")
        parser.add_argument("--config", default="config/feature_store_comprehensive.yaml", help="Feature store config")
        parser.add_argument("--universe", default="config/universe_liquid_1000.yaml", help="Universe config")
        parser.add_argument("--input", default="data/liquid/bars/interval=1h/symbol=*/date=*/*.parquet", help="Input pattern")
        parser.add_argument("--output", default="comprehensive_1h_features", help="Feature name (will create features/<name>/ structure)")
        
        args = parser.parse_args()
        
        # Find input files
        input_paths = list(Path(".").glob(args.input))
        if not input_paths:
            logger.error(f"No input files found matching pattern: {args.input}")
            return 1
        
        # Build features
        builder = ComprehensiveFeatureBuilder(args.config)
        builder.build_features([str(p) for p in input_paths], args.output, args.universe)
        
        logger.info(f"✅ Comprehensive features built successfully!")
        logger.info(f"📁 Output location: storage/features/{args.output}/")
        logger.info(f"🔍 Check: ls -la storage/features/{args.output}/")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error during feature building: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
