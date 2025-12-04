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
Simple Feature Computation Module
Provides a simplified feature computation logic to avoid duplicate column issues
"""


import polars as pl
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class SimpleFeatureComputer:
    """Simple feature computation class with basic features only"""
    
    def __init__(self):
        self.feature_definitions = self._get_feature_definitions()
    
    def get_all_features(self) -> List[str]:
        """Get all feature names as a flat list"""
        all_features = []
        for category, features in self.feature_definitions.items():
            all_features.extend(features)
        return all_features
    
    def _get_feature_definitions(self) -> Dict[str, List[str]]:
        """Get comprehensive feature definitions organized by category"""
        return {
            "core": [
                "ret_1m", "ret_5m", "ret_15m", "ret_30m", "ret_60m",
                "vol_5m", "vol_15m", "vol_30m", "vol_60m",
                "mom_1d", "mom_3d", "mom_5d", "mom_10d",
                "dollar_volume", "turnover_20", "turnover_5",
                "returns_1d", "returns_5d", "returns_20d",
                "volatility_20d", "volatility_60d",
                "price_momentum_5d", "price_momentum_20d", "price_momentum_60d"
            ],
            "technical": [
                # Basic SMAs
                "sma_5", "sma_10", "sma_20", "sma_50", "sma_200",
                # Basic EMAs
                "ema_5", "ema_10", "ema_20", "ema_50",
                # Advanced Moving Averages
                "hull_ma_5", "hull_ma_10", "hull_ma_20",
                "kama_5", "kama_10", "kama_20",
                "tema_5", "tema_10", "tema_20",
                "dema_5", "dema_10", "dema_20",
                "vwma_5", "vwma_10", "vwma_20",
                # RSI variants
                "rsi_5", "rsi_7", "rsi_10", "rsi_14", "rsi_21", "rsi_30",
                # MACD
                "macd", "macd_signal", "macd_hist",
                # Bollinger Bands
                "bb_upper_20", "bb_lower_20", "bb_width_20", "bb_percent_b_20",
                # Stochastic variants
                "stoch_k_5", "stoch_k_14", "stoch_k_21",
                "stoch_d_5", "stoch_d_14", "stoch_d_21",
                # Williams %R variants
                "williams_r_7", "williams_r_14", "williams_r_21",
                # CCI variants
                "cci_10", "cci_14", "cci_20", "cci_30",
                # ATR and ADX
                "atr_14", "adx_14",
                # Momentum and ROC
                "momentum_5", "momentum_10", "momentum_20",
                "roc_5", "roc_10", "roc_20", "roc_50",
                # Advanced Technical Indicators
                "psar", "psar_af",
                "ichimoku_tenkan", "ichimoku_kijun", "ichimoku_senkou_a", "ichimoku_senkou_b", "ichimoku_chikou",
                "keltner_upper", "keltner_lower", "keltner_middle",
                "donchian_upper", "donchian_lower", "donchian_middle",
                "supertrend", "supertrend_direction",
                "zigzag_high", "zigzag_low",
                "fractal_high", "fractal_low",
                "awesome_oscillator", "market_facilitation_index", "ease_of_movement",
                "mass_index", "negative_volume_index", "price_volume_trend",
                "williams_accumulation", "chaikin_oscillator", "force_index",
            ],
            "volume": [
                # Basic volume features
                "volume_ratio_5d", "volume_ratio_20d",
                "volume_momentum_5d", "volume_momentum_20d",
                "volume_volatility_20d",
                "volume_sma_5", "volume_sma_20", "volume_sma_50",
                "volume_ema_5", "volume_ema_20",
                # Advanced volume features
                "obv", "obv_ema",
                "ad_line", "ad_line_ema",
                "cmf", "cmf_ema",
                "mfi_14", "mfi_21",
                "volume_roc_5", "volume_roc_10", "volume_roc_20",
                "vwap_5", "vwap_10", "vwap_20",
                "volume_weighted_price",
            ],
            "volatility": [
                # Garman-Klass Volatility
                "gk_vol_5", "gk_vol_10", "gk_vol_20",
                # Parkinson Volatility
                "parkinson_vol_5", "parkinson_vol_10", "parkinson_vol_20",
                # Rogers-Satchell Volatility
                "rs_vol_5", "rs_vol_10", "rs_vol_20",
                # Yang-Zhang Volatility
                "yz_vol_5", "yz_vol_10", "yz_vol_20",
                # Realized Volatility
                "realized_vol_5", "realized_vol_10", "realized_vol_20",
            ],
            "microstructure": [
                # Basic microstructure
                "ret_oc", "gap_co", "range_frac", "vwap_dev", "vwap_dev_high", "vwap_dev_low",
                "overnight_vol", "overnight_return", "close_loc_1d", "open_loc_1d", "low_loc_1d",
                
                # Price impact & spread features
                "bid_ask_spread_est", "price_impact", "effective_spread", "realized_spread",
                "mid_price_vol", "trade_size_vol", "order_flow_imbalance", "volume_weighted_price",
                
                # Volatility clustering & microstructure
                "vol_clustering_5m", "vol_clustering_15m", "vol_clustering_60m", "vol_persistence",
                "range_compression_5m", "range_compression_15m", "range_compression_60m",
                "vol_over_vol_5m", "vol_over_vol_15m", "vol_over_vol_60m",
                
                # Market microstructure effects
                "market_impact", "liquidity_ratio", "turnover", "vol_dollar", "volume_price_trend",
                "open_vol_frac", "close_vol_frac", "high_vol_frac", "low_vol_frac",
                
                # Intraday patterns
                "hour_9", "hour_10", "hour_11", "hour_12", "hour_13", "hour_14", "hour_15",
                "wd_0", "wd_1", "wd_2", "wd_3", "wd_4",
                "trading_day_of_month", "trading_day_of_quarter", "holiday_dummy", "pre_holiday_dummy", "post_holiday_dummy",
                
                # Advanced microstructure
                "volume_quantile_20d", "volume_std_20d", "volume_skew_20d", "volume_kurt_20d",
                "vwap_dev_5m", "vwap_dev_15m", "vwap_dev_30m", "vwap_dev_session",
            ],
            "time_based": [
                # Basic time features
                "hour_of_day", "day_of_week", "day_of_month", "month_of_year", "quarter", "year",
                "is_month_end", "is_quarter_end",
                # Advanced time features
                "intraday_seasonality_5m", "intraday_seasonality_15m", "intraday_seasonality_30m",
                "weekly_seasonality_monday", "weekly_seasonality_friday",
                "monthly_seasonality_beginning", "monthly_seasonality_end",
                "quarterly_seasonality_beginning", "quarterly_seasonality_end",
                "pre_holiday_effect", "post_holiday_effect", "holiday_effect",
            ],
            "cross_sectional": [
                # Relative performance
                "relative_performance_5d", "relative_performance_20d", "relative_performance_60d",
                # Sector momentum
                "sector_momentum_5d", "sector_momentum_20d",
                # Market cap and beta
                "market_cap_decile", "beta_20d", "beta_60d",
                # Market correlation
                "market_correlation_20d", "market_correlation_60d",
            ],
            "non_linear": [
                # Basic non-linear
                "ret2_5m", "ret2_15m", "ret2_60m",
                # Volume-return interactions
                "vol_x_ret_5m", "vol_x_ret_15m", "vol_x_ret_60m",
                # Return-momentum interactions
                "ret_x_mom_5d", "ret_x_mom_20d",
                # Volume-momentum interactions
                "vol_x_mom_5d", "vol_x_mom_20d",
                # Technical indicator interactions
                "rsi_x_vol", "bb_x_vol", "macd_x_vol", "macd_x_ret", "stoch_x_vol", "stoch_x_ret",
                "bb_x_ret", "atr_x_vol", "adx_x_vol", "cci_x_vol", "momentum_x_vol", "roc_x_vol",
                # Advanced interactions
                "rsi_x_volume", "macd_x_volume", "bb_x_volume",
                "momentum_x_volatility", "roc_x_volatility",
                "hour_x_volume", "day_x_volume", "price_x_volume", "gap_x_volume",
            ],
            "target": [
                "fwd_ret_1d", "fwd_ret_5d", "fwd_ret_20d",
            ]
        }
    
    def compute_features(self, scan: pl.LazyFrame, config_features: List[str]) -> pl.LazyFrame:
        """Compute features using simple, direct expressions"""
        logger.info(f"Computing features for categories: {config_features}")
        
        # Start with basic features
        features = self._compute_basic_features(scan)
        
        # Add features based on requested categories
        if "technical" in config_features:
            features = self._compute_technical_features(features)
        
        if "volume" in config_features:
            features = self._compute_volume_features(features)
        
        if "volatility" in config_features:
            features = self._compute_volatility_features(features)
        
        if "microstructure" in config_features:
            features = self._compute_microstructure_features(features)
        
        if "time_based" in config_features:
            features = self._compute_time_features(features)
        
        if "cross_sectional" in config_features:
            features = self._compute_cross_sectional_features(features)
        
        if "non_linear" in config_features:
            features = self._compute_interaction_features(features)
        
        if "target" in config_features:
            features = self._add_forward_returns(features)
        
        return features
    
    def _compute_basic_features(self, features: pl.LazyFrame) -> pl.LazyFrame:
        """Compute basic price and volume features"""
        return features.with_columns([
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
        ])
    
    def _compute_technical_features(self, features: pl.LazyFrame) -> pl.LazyFrame:
        """Compute technical indicators using simple, direct expressions"""
        return features.with_columns([
            # Basic SMAs
            pl.col("close").rolling_mean(5).alias("sma_5").cast(pl.Float32),
            pl.col("close").rolling_mean(10).alias("sma_10").cast(pl.Float32),
            pl.col("close").rolling_mean(20).alias("sma_20").cast(pl.Float32),
            pl.col("close").rolling_mean(50).alias("sma_50").cast(pl.Float32),
            pl.col("close").rolling_mean(200).alias("sma_200").cast(pl.Float32),
            
            # Basic EMAs
            pl.col("close").ewm_mean(span=5).alias("ema_5").cast(pl.Float32),
            pl.col("close").ewm_mean(span=10).alias("ema_10").cast(pl.Float32),
            pl.col("close").ewm_mean(span=20).alias("ema_20").cast(pl.Float32),
            pl.col("close").ewm_mean(span=50).alias("ema_50").cast(pl.Float32),
            
            # Advanced Moving Averages
            self._hull_ma(pl.col("close"), 5).alias("hull_ma_5").cast(pl.Float32),
            self._hull_ma(pl.col("close"), 10).alias("hull_ma_10").cast(pl.Float32),
            self._hull_ma(pl.col("close"), 20).alias("hull_ma_20").cast(pl.Float32),
            self._kama(pl.col("close"), 5).alias("kama_5").cast(pl.Float32),
            self._kama(pl.col("close"), 10).alias("kama_10").cast(pl.Float32),
            self._kama(pl.col("close"), 20).alias("kama_20").cast(pl.Float32),
            self._tema(pl.col("close"), 5).alias("tema_5").cast(pl.Float32),
            self._tema(pl.col("close"), 10).alias("tema_10").cast(pl.Float32),
            self._tema(pl.col("close"), 20).alias("tema_20").cast(pl.Float32),
            self._dema(pl.col("close"), 5).alias("dema_5").cast(pl.Float32),
            self._dema(pl.col("close"), 10).alias("dema_10").cast(pl.Float32),
            self._dema(pl.col("close"), 20).alias("dema_20").cast(pl.Float32),
            self._vwma(pl.col("close"), pl.col("volume"), 5).alias("vwma_5").cast(pl.Float32),
            self._vwma(pl.col("close"), pl.col("volume"), 10).alias("vwma_10").cast(pl.Float32),
            self._vwma(pl.col("close"), pl.col("volume"), 20).alias("vwma_20").cast(pl.Float32),
            
            # RSI variants
            self._simple_rsi(pl.col("close"), 5).alias("rsi_5").cast(pl.Float32),
            self._simple_rsi(pl.col("close"), 7).alias("rsi_7").cast(pl.Float32),
            self._simple_rsi(pl.col("close"), 10).alias("rsi_10").cast(pl.Float32),
            self._simple_rsi(pl.col("close"), 14).alias("rsi_14").cast(pl.Float32),
            self._simple_rsi(pl.col("close"), 21).alias("rsi_21").cast(pl.Float32),
            self._simple_rsi(pl.col("close"), 30).alias("rsi_30").cast(pl.Float32),
            
            # MACD
            self._simple_macd(pl.col("close")).alias("macd").cast(pl.Float32),
            self._simple_macd_signal(pl.col("close")).alias("macd_signal").cast(pl.Float32),
            self._simple_macd_hist(pl.col("close")).alias("macd_hist").cast(pl.Float32),
            
            # Bollinger Bands
            self._simple_bb_upper(pl.col("close")).alias("bb_upper_20").cast(pl.Float32),
            self._simple_bb_lower(pl.col("close")).alias("bb_lower_20").cast(pl.Float32),
            self._simple_bb_width(pl.col("close")).alias("bb_width_20").cast(pl.Float32),
            self._simple_bb_percent_b(pl.col("close")).alias("bb_percent_b_20").cast(pl.Float32),
            
            # Stochastic variants
            self._simple_stoch_k(pl.col("high"), pl.col("low"), pl.col("close"), 5).alias("stoch_k_5").cast(pl.Float32),
            self._simple_stoch_k(pl.col("high"), pl.col("low"), pl.col("close"), 14).alias("stoch_k_14").cast(pl.Float32),
            self._simple_stoch_k(pl.col("high"), pl.col("low"), pl.col("close"), 21).alias("stoch_k_21").cast(pl.Float32),
            self._simple_stoch_d(pl.col("high"), pl.col("low"), pl.col("close"), 5).alias("stoch_d_5").cast(pl.Float32),
            self._simple_stoch_d(pl.col("high"), pl.col("low"), pl.col("close"), 14).alias("stoch_d_14").cast(pl.Float32),
            self._simple_stoch_d(pl.col("high"), pl.col("low"), pl.col("close"), 21).alias("stoch_d_21").cast(pl.Float32),
            
            # Williams %R variants
            self._simple_williams_r(pl.col("high"), pl.col("low"), pl.col("close"), 7).alias("williams_r_7").cast(pl.Float32),
            self._simple_williams_r(pl.col("high"), pl.col("low"), pl.col("close"), 14).alias("williams_r_14").cast(pl.Float32),
            self._simple_williams_r(pl.col("high"), pl.col("low"), pl.col("close"), 21).alias("williams_r_21").cast(pl.Float32),
            
            # CCI variants
            self._simple_cci(pl.col("high"), pl.col("low"), pl.col("close"), 10).alias("cci_10").cast(pl.Float32),
            self._simple_cci(pl.col("high"), pl.col("low"), pl.col("close"), 14).alias("cci_14").cast(pl.Float32),
            self._simple_cci(pl.col("high"), pl.col("low"), pl.col("close"), 20).alias("cci_20").cast(pl.Float32),
            self._simple_cci(pl.col("high"), pl.col("low"), pl.col("close"), 30).alias("cci_30").cast(pl.Float32),
            
            # ATR and ADX
            self._simple_atr(pl.col("high"), pl.col("low"), pl.col("close")).alias("atr_14").cast(pl.Float32),
            self._simple_adx(pl.col("high"), pl.col("low"), pl.col("close")).alias("adx_14").cast(pl.Float32),
            
            # Momentum and ROC
            (pl.col("close") / pl.col("close").shift(5) - 1).alias("momentum_5").cast(pl.Float32),
            (pl.col("close") / pl.col("close").shift(10) - 1).alias("momentum_10").cast(pl.Float32),
            (pl.col("close") / pl.col("close").shift(20) - 1).alias("momentum_20").cast(pl.Float32),
            (pl.col("close") / pl.col("close").shift(5) - 1).alias("roc_5").cast(pl.Float32),
            (pl.col("close") / pl.col("close").shift(10) - 1).alias("roc_10").cast(pl.Float32),
            (pl.col("close") / pl.col("close").shift(20) - 1).alias("roc_20").cast(pl.Float32),
            (pl.col("close") / pl.col("close").shift(50) - 1).alias("roc_50").cast(pl.Float32),
            
            # Advanced Technical Indicators
            self._psar(pl.col("high"), pl.col("low"), pl.col("close")).alias("psar").cast(pl.Float32),
            self._psar_af(pl.col("high"), pl.col("low"), pl.col("close")).alias("psar_af").cast(pl.Float32),
            self._ichimoku_tenkan(pl.col("high"), pl.col("low")).alias("ichimoku_tenkan").cast(pl.Float32),
            self._ichimoku_kijun(pl.col("high"), pl.col("low")).alias("ichimoku_kijun").cast(pl.Float32),
            self._ichimoku_senkou_a(pl.col("high"), pl.col("low")).alias("ichimoku_senkou_a").cast(pl.Float32),
            self._ichimoku_senkou_b(pl.col("high"), pl.col("low")).alias("ichimoku_senkou_b").cast(pl.Float32),
            self._ichimoku_chikou(pl.col("close")).alias("ichimoku_chikou").cast(pl.Float32),
            self._keltner_upper(pl.col("high"), pl.col("low"), pl.col("close")).alias("keltner_upper").cast(pl.Float32),
            self._keltner_lower(pl.col("high"), pl.col("low"), pl.col("close")).alias("keltner_lower").cast(pl.Float32),
            self._keltner_middle(pl.col("high"), pl.col("low"), pl.col("close")).alias("keltner_middle").cast(pl.Float32),
            self._donchian_upper(pl.col("high")).alias("donchian_upper").cast(pl.Float32),
            self._donchian_lower(pl.col("low")).alias("donchian_lower").cast(pl.Float32),
            self._donchian_middle(pl.col("high"), pl.col("low")).alias("donchian_middle").cast(pl.Float32),
            self._supertrend(pl.col("high"), pl.col("low"), pl.col("close")).alias("supertrend").cast(pl.Float32),
            self._supertrend_direction(pl.col("high"), pl.col("low"), pl.col("close")).alias("supertrend_direction").cast(pl.Float32),
            self._zigzag_high(pl.col("high"), pl.col("low")).alias("zigzag_high").cast(pl.Float32),
            self._zigzag_low(pl.col("high"), pl.col("low")).alias("zigzag_low").cast(pl.Float32),
            self._fractal_high(pl.col("high")).alias("fractal_high").cast(pl.Float32),
            self._fractal_low(pl.col("low")).alias("fractal_low").cast(pl.Float32),
            self._awesome_oscillator(pl.col("high"), pl.col("low")).alias("awesome_oscillator").cast(pl.Float32),
            self._market_facilitation_index(pl.col("high"), pl.col("low"), pl.col("volume")).alias("market_facilitation_index").cast(pl.Float32),
            self._ease_of_movement(pl.col("high"), pl.col("low"), pl.col("volume")).alias("ease_of_movement").cast(pl.Float32),
            self._mass_index(pl.col("high"), pl.col("low")).alias("mass_index").cast(pl.Float32),
            self._negative_volume_index(pl.col("close"), pl.col("volume")).alias("negative_volume_index").cast(pl.Float32),
            self._price_volume_trend(pl.col("close"), pl.col("volume")).alias("price_volume_trend").cast(pl.Float32),
            self._williams_accumulation(pl.col("high"), pl.col("low"), pl.col("close"), pl.col("volume")).alias("williams_accumulation").cast(pl.Float32),
            self._chaikin_oscillator(pl.col("high"), pl.col("low"), pl.col("close"), pl.col("volume")).alias("chaikin_oscillator").cast(pl.Float32),
            self._force_index(pl.col("close"), pl.col("volume")).alias("force_index").cast(pl.Float32),
        ])
    
    def _compute_volume_features(self, features: pl.LazyFrame) -> pl.LazyFrame:
        """Compute volume-based features"""
        return features.with_columns([
            # Basic volume features
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
            
            # Advanced volume features
            self._obv(pl.col("close"), pl.col("volume")).alias("obv").cast(pl.Float32),
            self._obv_ema(pl.col("close"), pl.col("volume")).alias("obv_ema").cast(pl.Float32),
            self._ad_line(pl.col("high"), pl.col("low"), pl.col("close"), pl.col("volume")).alias("ad_line").cast(pl.Float32),
            self._ad_line_ema(pl.col("high"), pl.col("low"), pl.col("close"), pl.col("volume")).alias("ad_line_ema").cast(pl.Float32),
            self._cmf(pl.col("high"), pl.col("low"), pl.col("close"), pl.col("volume")).alias("cmf").cast(pl.Float32),
            self._cmf_ema(pl.col("high"), pl.col("low"), pl.col("close"), pl.col("volume")).alias("cmf_ema").cast(pl.Float32),
            self._mfi(pl.col("high"), pl.col("low"), pl.col("close"), pl.col("volume"), 14).alias("mfi_14").cast(pl.Float32),
            self._mfi(pl.col("high"), pl.col("low"), pl.col("close"), pl.col("volume"), 21).alias("mfi_21").cast(pl.Float32),
            pl.col("volume").pct_change(5).alias("volume_roc_5").cast(pl.Float32),
            pl.col("volume").pct_change(10).alias("volume_roc_10").cast(pl.Float32),
            pl.col("volume").pct_change(20).alias("volume_roc_20").cast(pl.Float32),
            self._vwap_period(pl.col("high"), pl.col("low"), pl.col("close"), pl.col("volume"), 5).alias("vwap_5").cast(pl.Float32),
            self._vwap_period(pl.col("high"), pl.col("low"), pl.col("close"), pl.col("volume"), 10).alias("vwap_10").cast(pl.Float32),
            self._vwap_period(pl.col("high"), pl.col("low"), pl.col("close"), pl.col("volume"), 20).alias("vwap_20").cast(pl.Float32),
            self._volume_weighted_price(pl.col("high"), pl.col("low"), pl.col("close"), pl.col("volume")).alias("volume_weighted_price").cast(pl.Float32),
        ])
    
    def _compute_volatility_features(self, features: pl.LazyFrame) -> pl.LazyFrame:
        """Compute advanced volatility features"""
        return features.with_columns([
            # Garman-Klass Volatility
            self._gk_vol(pl.col("high"), pl.col("low"), pl.col("open"), pl.col("close"), 5).alias("gk_vol_5").cast(pl.Float32),
            self._gk_vol(pl.col("high"), pl.col("low"), pl.col("open"), pl.col("close"), 10).alias("gk_vol_10").cast(pl.Float32),
            self._gk_vol(pl.col("high"), pl.col("low"), pl.col("open"), pl.col("close"), 20).alias("gk_vol_20").cast(pl.Float32),
            
            # Parkinson Volatility
            self._parkinson_vol(pl.col("high"), pl.col("low"), 5).alias("parkinson_vol_5").cast(pl.Float32),
            self._parkinson_vol(pl.col("high"), pl.col("low"), 10).alias("parkinson_vol_10").cast(pl.Float32),
            self._parkinson_vol(pl.col("high"), pl.col("low"), 20).alias("parkinson_vol_20").cast(pl.Float32),
            
            # Rogers-Satchell Volatility
            self._rs_vol(pl.col("high"), pl.col("low"), pl.col("open"), pl.col("close"), 5).alias("rs_vol_5").cast(pl.Float32),
            self._rs_vol(pl.col("high"), pl.col("low"), pl.col("open"), pl.col("close"), 10).alias("rs_vol_10").cast(pl.Float32),
            self._rs_vol(pl.col("high"), pl.col("low"), pl.col("open"), pl.col("close"), 20).alias("rs_vol_20").cast(pl.Float32),
            
            # Yang-Zhang Volatility
            self._yz_vol(pl.col("high"), pl.col("low"), pl.col("open"), pl.col("close"), 5).alias("yz_vol_5").cast(pl.Float32),
            self._yz_vol(pl.col("high"), pl.col("low"), pl.col("open"), pl.col("close"), 10).alias("yz_vol_10").cast(pl.Float32),
            self._yz_vol(pl.col("high"), pl.col("low"), pl.col("open"), pl.col("close"), 20).alias("yz_vol_20").cast(pl.Float32),
            
            # Realized Volatility
            pl.col("close").pct_change().rolling_std(5).alias("realized_vol_5").cast(pl.Float32),
            pl.col("close").pct_change().rolling_std(10).alias("realized_vol_10").cast(pl.Float32),
            pl.col("close").pct_change().rolling_std(20).alias("realized_vol_20").cast(pl.Float32),
        ])
    
    def _compute_microstructure_features(self, features: pl.LazyFrame) -> pl.LazyFrame:
        """Compute comprehensive microstructure features"""
        return features.with_columns([
            # Time features
            pl.col("ts").dt.hour().alias("_hour").cast(pl.Int16),
            pl.col("ts").dt.weekday().alias("_weekday").cast(pl.Int16),
            pl.col("ts").dt.day().alias("_day").cast(pl.Int16),
            pl.col("ts").dt.month().alias("_month").cast(pl.Int16),
            pl.col("ts").dt.quarter().alias("_quarter").cast(pl.Int16),
            pl.col("ts").dt.year().alias("_year").cast(pl.Int16),
        ]).with_columns([
            # Price impact & spread features
            ((pl.col("high") - pl.col("low")) / pl.col("close")).alias("bid_ask_spread_est").cast(pl.Float32),
            (pl.col("close").pct_change().abs() / pl.col("volume").log()).alias("price_impact").cast(pl.Float32),
            ((pl.col("high") - pl.col("low")) / pl.col("close")).alias("effective_spread").cast(pl.Float32),
            ((pl.col("close") - pl.col("open")) / pl.col("close")).alias("realized_spread").cast(pl.Float32),
            ((pl.col("high") + pl.col("low")) / 2).alias("mid_price_vol").cast(pl.Float32),
            (pl.col("volume").rolling_std(5)).alias("trade_size_vol").cast(pl.Float32),
            ((pl.col("close") - pl.col("open")) / (pl.col("high") - pl.col("low"))).alias("order_flow_imbalance").cast(pl.Float32),
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3).alias("volume_weighted_price").cast(pl.Float32),
            
            # Volatility clustering & microstructure
            (pl.col("close").pct_change().rolling_std(5)).alias("vol_clustering_5m").cast(pl.Float32),
            (pl.col("close").pct_change().rolling_std(15)).alias("vol_clustering_15m").cast(pl.Float32),
            (pl.col("close").pct_change().rolling_std(60)).alias("vol_clustering_60m").cast(pl.Float32),
            (pl.col("close").pct_change().rolling_std(20)).alias("vol_persistence").cast(pl.Float32),
            
            # Range compression features
            ((pl.col("high") - pl.col("low")) / pl.col("close").rolling_mean(5)).alias("range_compression_5m").cast(pl.Float32),
            ((pl.col("high") - pl.col("low")) / pl.col("close").rolling_mean(15)).alias("range_compression_15m").cast(pl.Float32),
            ((pl.col("high") - pl.col("low")) / pl.col("close").rolling_mean(60)).alias("range_compression_60m").cast(pl.Float32),
            
            # Vol over vol features
            (pl.col("close").pct_change().rolling_std(5) / pl.col("close").pct_change().rolling_std(20)).alias("vol_over_vol_5m").cast(pl.Float32),
            (pl.col("close").pct_change().rolling_std(15) / pl.col("close").pct_change().rolling_std(20)).alias("vol_over_vol_15m").cast(pl.Float32),
            (pl.col("close").pct_change().rolling_std(60) / pl.col("close").pct_change().rolling_std(20)).alias("vol_over_vol_60m").cast(pl.Float32),
            
            # Market microstructure effects
            (pl.col("close").pct_change().abs() / pl.col("volume").log()).alias("market_impact").cast(pl.Float32),
            (pl.col("volume") / pl.col("volume").rolling_mean(20)).alias("liquidity_ratio").cast(pl.Float32),
            (pl.col("volume") / pl.col("volume").rolling_mean(20)).alias("turnover").cast(pl.Float32),
            (pl.col("volume") * pl.col("close")).alias("vol_dollar").cast(pl.Float32),
            (pl.col("close").pct_change() * pl.col("volume")).alias("volume_price_trend").cast(pl.Float32),
            
            # Volume fractions
            (pl.col("open") / pl.col("volume")).alias("open_vol_frac").cast(pl.Float32),
            (pl.col("close") / pl.col("volume")).alias("close_vol_frac").cast(pl.Float32),
            (pl.col("high") / pl.col("volume")).alias("high_vol_frac").cast(pl.Float32),
            (pl.col("low") / pl.col("volume")).alias("low_vol_frac").cast(pl.Float32),
            
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
            
            # Trading day features
            pl.col("_day").alias("trading_day_of_month").cast(pl.Int16),
            ((pl.col("_day") - 1) // 7 + 1).alias("trading_day_of_quarter").cast(pl.Int16),
            
            # Holiday dummies (simplified - can be enhanced with actual holiday calendar)
            ((pl.col("_month") == 12) & (pl.col("_day") >= 24) & (pl.col("_day") <= 26)).cast(pl.UInt8).alias("holiday_dummy"),
            ((pl.col("_month") == 12) & (pl.col("_day") == 23)).cast(pl.UInt8).alias("pre_holiday_dummy"),
            ((pl.col("_month") == 12) & (pl.col("_day") == 27)).cast(pl.UInt8).alias("post_holiday_dummy"),
            
            # Advanced microstructure
            (pl.col("volume").rolling_median(20)).alias("volume_quantile_20d").cast(pl.Float32),
            (pl.col("volume").rolling_std(20)).alias("volume_std_20d").cast(pl.Float32),
            # Note: rolling_skew and rolling_kurt not available in Polars, using alternatives
            (pl.col("volume") / pl.col("volume").rolling_mean(20) - 1).alias("volume_skew_20d").cast(pl.Float32),
            ((pl.col("volume") - pl.col("volume").rolling_mean(20)) / pl.col("volume").rolling_std(20)).alias("volume_kurt_20d").cast(pl.Float32),
            
            # VWAP deviation features
            ((pl.col("close") - pl.col("vwap")) / pl.col("vwap")).alias("vwap_dev_5m").cast(pl.Float32),
            ((pl.col("close") - pl.col("vwap")) / pl.col("vwap")).alias("vwap_dev_15m").cast(pl.Float32),
            ((pl.col("close") - pl.col("vwap")) / pl.col("vwap")).alias("vwap_dev_30m").cast(pl.Float32),
            ((pl.col("close") - pl.col("vwap")) / pl.col("vwap")).alias("vwap_dev_session").cast(pl.Float32),
        ])
    
    def _compute_time_features(self, features: pl.LazyFrame) -> pl.LazyFrame:
        """Compute time-based features"""
        return features.with_columns([
            # Basic time features
            pl.col("ts").dt.hour().alias("hour_of_day").cast(pl.Int16),
            pl.col("ts").dt.weekday().alias("day_of_week").cast(pl.Int16),
            pl.col("ts").dt.day().alias("day_of_month").cast(pl.Int16),
            pl.col("ts").dt.month().alias("month_of_year").cast(pl.Int16),
            pl.col("ts").dt.quarter().alias("quarter").cast(pl.Int16),
            pl.col("ts").dt.year().alias("year").cast(pl.Int16),
            
            # End-of-period flags
            (pl.col("ts").dt.day() >= 28).cast(pl.UInt8).alias("is_month_end"),
            (pl.col("ts").dt.month().is_in([3, 6, 9, 12]) & (pl.col("ts").dt.day() >= 28)).cast(pl.UInt8).alias("is_quarter_end"),
            
            # Advanced time features
            self._intraday_seasonality(pl.col("ts"), 5).alias("intraday_seasonality_5m").cast(pl.Float32),
            self._intraday_seasonality(pl.col("ts"), 15).alias("intraday_seasonality_15m").cast(pl.Float32),
            self._intraday_seasonality(pl.col("ts"), 30).alias("intraday_seasonality_30m").cast(pl.Float32),
            self._weekly_seasonality(pl.col("ts"), "monday").alias("weekly_seasonality_monday").cast(pl.Float32),
            self._weekly_seasonality(pl.col("ts"), "friday").alias("weekly_seasonality_friday").cast(pl.Float32),
            self._monthly_seasonality(pl.col("ts"), "beginning").alias("monthly_seasonality_beginning").cast(pl.Float32),
            self._monthly_seasonality(pl.col("ts"), "end").alias("monthly_seasonality_end").cast(pl.Float32),
            self._quarterly_seasonality(pl.col("ts"), "beginning").alias("quarterly_seasonality_beginning").cast(pl.Float32),
            self._quarterly_seasonality(pl.col("ts"), "end").alias("quarterly_seasonality_end").cast(pl.Float32),
            self._holiday_effect(pl.col("ts"), "pre").alias("pre_holiday_effect").cast(pl.Float32),
            self._holiday_effect(pl.col("ts"), "post").alias("post_holiday_effect").cast(pl.Float32),
            self._holiday_effect(pl.col("ts"), "holiday").alias("holiday_effect").cast(pl.Float32),
        ])
    
    def _compute_cross_sectional_features(self, features: pl.LazyFrame) -> pl.LazyFrame:
        """Compute cross-sectional features"""
        return features.with_columns([
            # Relative performance (simplified - would need market data for full implementation)
            pl.col("close").pct_change(5).alias("relative_performance_5d").cast(pl.Float32),
            pl.col("close").pct_change(20).alias("relative_performance_20d").cast(pl.Float32),
            pl.col("close").pct_change(60).alias("relative_performance_60d").cast(pl.Float32),
            
            # Sector momentum (simplified - would need sector data for full implementation)
            pl.col("close").pct_change(5).alias("sector_momentum_5d").cast(pl.Float32),
            pl.col("close").pct_change(20).alias("sector_momentum_20d").cast(pl.Float32),
            
            # Market cap decile (simplified - would need market cap data for full implementation)
            pl.lit(5).alias("market_cap_decile").cast(pl.Int16),  # Placeholder
            
            # Beta (simplified - would need market data for full implementation)
            pl.col("close").pct_change().rolling_std(20).alias("beta_20d").cast(pl.Float32),
            pl.col("close").pct_change().rolling_std(60).alias("beta_60d").cast(pl.Float32),
            
            # Market correlation (simplified - would need market data for full implementation)
            pl.col("close").pct_change().rolling_std(20).alias("market_correlation_20d").cast(pl.Float32),
            pl.col("close").pct_change().rolling_std(60).alias("market_correlation_60d").cast(pl.Float32),
        ])
    
    def _compute_interaction_features(self, features: pl.LazyFrame) -> pl.LazyFrame:
        """Compute interaction features"""
        # Get available columns to make interactions defensive
        available_cols = set(features.collect_schema().names())
        
        interactions = []
        
        # Basic non-linear features (always available)
        interactions.extend([
            (pl.col("ret_5m") ** 2).alias("ret2_5m").cast(pl.Float32),
            (pl.col("ret_15m") ** 2).alias("ret2_15m").cast(pl.Float32),
            (pl.col("ret_60m") ** 2).alias("ret2_60m").cast(pl.Float32),
        ])
        
        # Volume-return interactions (always available)
        interactions.extend([
            (pl.col("volume") * pl.col("ret_5m")).alias("vol_x_ret_5m").cast(pl.Float32),
            (pl.col("volume") * pl.col("ret_15m")).alias("vol_x_ret_15m").cast(pl.Float32),
            (pl.col("volume") * pl.col("ret_60m")).alias("vol_x_ret_60m").cast(pl.Float32),
        ])
        
        # Return-momentum interactions (check if momentum columns exist)
        if "mom_5d" in available_cols:
            interactions.append((pl.col("ret_5m") * pl.col("mom_5d")).alias("ret_x_mom_5d").cast(pl.Float32))
        if "price_momentum_20d" in available_cols:
            interactions.append((pl.col("ret_5m") * pl.col("price_momentum_20d")).alias("ret_x_mom_20d").cast(pl.Float32))
        
        # Volume-momentum interactions (check if momentum columns exist)
        if "mom_5d" in available_cols:
            interactions.append((pl.col("volume") * pl.col("mom_5d")).alias("vol_x_mom_5d").cast(pl.Float32))
        if "price_momentum_20d" in available_cols:
            interactions.append((pl.col("volume") * pl.col("price_momentum_20d")).alias("vol_x_mom_20d").cast(pl.Float32))
        
        # Technical indicator interactions (check if technical columns exist)
        if "rsi_14" in available_cols:
            interactions.append((pl.col("rsi_14") * pl.col("volume")).alias("rsi_x_vol").cast(pl.Float32))
            interactions.append((pl.col("rsi_14") * pl.col("volume")).alias("rsi_x_volume").cast(pl.Float32))
        if "bb_width_20" in available_cols:
            interactions.append((pl.col("bb_width_20") * pl.col("volume")).alias("bb_x_vol").cast(pl.Float32))
            interactions.append((pl.col("bb_width_20") * pl.col("volume")).alias("bb_x_volume").cast(pl.Float32))
            interactions.append((pl.col("bb_width_20") * pl.col("ret_5m")).alias("bb_x_ret").cast(pl.Float32))
        if "macd" in available_cols:
            interactions.append((pl.col("macd") * pl.col("volume")).alias("macd_x_vol").cast(pl.Float32))
            interactions.append((pl.col("macd") * pl.col("volume")).alias("macd_x_volume").cast(pl.Float32))
            interactions.append((pl.col("macd") * pl.col("ret_5m")).alias("macd_x_ret").cast(pl.Float32))
        if "stoch_k_14" in available_cols:
            interactions.append((pl.col("stoch_k_14") * pl.col("volume")).alias("stoch_x_vol").cast(pl.Float32))
            interactions.append((pl.col("stoch_k_14") * pl.col("ret_5m")).alias("stoch_x_ret").cast(pl.Float32))
        if "atr_14" in available_cols:
            interactions.append((pl.col("atr_14") * pl.col("volume")).alias("atr_x_vol").cast(pl.Float32))
        if "adx_14" in available_cols:
            interactions.append((pl.col("adx_14") * pl.col("volume")).alias("adx_x_vol").cast(pl.Float32))
        if "cci_20" in available_cols:
            interactions.append((pl.col("cci_20") * pl.col("volume")).alias("cci_x_vol").cast(pl.Float32))
        if "momentum_5" in available_cols:
            interactions.append((pl.col("momentum_5") * pl.col("volume")).alias("momentum_x_vol").cast(pl.Float32))
        if "roc_5" in available_cols:
            interactions.append((pl.col("roc_5") * pl.col("volume")).alias("roc_x_vol").cast(pl.Float32))
        
        # Advanced interactions (check if required columns exist)
        if "momentum_5" in available_cols and "volatility_20d" in available_cols:
            interactions.append((pl.col("momentum_5") * pl.col("volatility_20d")).alias("momentum_x_volatility").cast(pl.Float32))
        if "roc_5" in available_cols and "volatility_20d" in available_cols:
            interactions.append((pl.col("roc_5") * pl.col("volatility_20d")).alias("roc_x_volatility").cast(pl.Float32))
        if "hour_of_day" in available_cols:
            interactions.append((pl.col("hour_of_day") * pl.col("volume")).alias("hour_x_volume").cast(pl.Float32))
        if "day_of_week" in available_cols:
            interactions.append((pl.col("day_of_week") * pl.col("volume")).alias("day_x_volume").cast(pl.Float32))
        if "close" in available_cols:
            interactions.append((pl.col("close") * pl.col("volume")).alias("price_x_volume").cast(pl.Float32))
        if "gap_co" in available_cols:
            interactions.append((pl.col("gap_co") * pl.col("volume")).alias("gap_x_volume").cast(pl.Float32))
        
        return features.with_columns(interactions)
    
    def _add_forward_returns(self, features: pl.LazyFrame) -> pl.LazyFrame:
        """Add forward returns for target variables"""
        return features.with_columns([
            (pl.col("close").shift(-1) / pl.col("close") - 1).alias("fwd_ret_1d").cast(pl.Float32),
            (pl.col("close").shift(-5) / pl.col("close") - 1).alias("fwd_ret_5d").cast(pl.Float32),
            (pl.col("close").shift(-20) / pl.col("close") - 1).alias("fwd_ret_20d").cast(pl.Float32),
        ])
    
    # Simple helper methods for technical indicators
    def _simple_rsi(self, close: pl.Expr, period: int) -> pl.Expr:
        """Simple RSI calculation"""
        delta = close.diff()
        gain = pl.when(delta > 0).then(delta).otherwise(0)
        loss = pl.when(delta < 0).then(-delta).otherwise(0)
        avg_gain = gain.rolling_mean(period)
        avg_loss = loss.rolling_mean(period)
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    def _simple_macd(self, close: pl.Expr) -> pl.Expr:
        """Simple MACD calculation"""
        ema_12 = close.ewm_mean(span=12)
        ema_26 = close.ewm_mean(span=26)
        return ema_12 - ema_26
    
    def _simple_macd_signal(self, close: pl.Expr) -> pl.Expr:
        """Simple MACD signal calculation"""
        macd = self._simple_macd(close)
        return macd.ewm_mean(span=9).alias("macd_signal")
    
    def _simple_macd_hist(self, close: pl.Expr) -> pl.Expr:
        """Simple MACD histogram calculation"""
        macd = self._simple_macd(close)
        signal = self._simple_macd_signal(close)
        return (macd - signal).alias("macd_hist")
    
    def _simple_bb_upper(self, close: pl.Expr) -> pl.Expr:
        """Simple Bollinger Band upper calculation"""
        sma = close.rolling_mean(20)
        std = close.rolling_std(20)
        return (sma + (std * 2)).alias("bb_upper_20")
    
    def _simple_bb_lower(self, close: pl.Expr) -> pl.Expr:
        """Simple Bollinger Band lower calculation"""
        sma = close.rolling_mean(20)
        std = close.rolling_std(20)
        return (sma - (std * 2)).alias("bb_lower_20")
    
    def _simple_bb_width(self, close: pl.Expr) -> pl.Expr:
        """Simple Bollinger Band width calculation"""
        upper = self._simple_bb_upper(close)
        lower = self._simple_bb_lower(close)
        return (upper - lower).alias("bb_width_20")
    
    def _simple_bb_percent_b(self, close: pl.Expr) -> pl.Expr:
        """Simple Bollinger Band %B calculation"""
        upper = self._simple_bb_upper(close)
        lower = self._simple_bb_lower(close)
        return ((close - lower) / (upper - lower)).alias("bb_percent_b_20")
    
    def _simple_stoch_k(self, high: pl.Expr, low: pl.Expr, close: pl.Expr, period: int = 14) -> pl.Expr:
        """Simple Stochastic %K calculation"""
        lowest_low = low.rolling_min(period)
        highest_high = high.rolling_max(period)
        return (100 * (close - lowest_low) / (highest_high - lowest_low)).alias(f"stoch_k_{period}")
    
    def _simple_stoch_d(self, high: pl.Expr, low: pl.Expr, close: pl.Expr, period: int = 14) -> pl.Expr:
        """Simple Stochastic %D calculation"""
        k = self._simple_stoch_k(high, low, close, period)
        return k.rolling_mean(3).alias(f"stoch_d_{period}")
    
    def _simple_williams_r(self, high: pl.Expr, low: pl.Expr, close: pl.Expr, period: int = 14) -> pl.Expr:
        """Simple Williams %R calculation"""
        highest_high = high.rolling_max(period)
        lowest_low = low.rolling_min(period)
        return (-100 * (highest_high - close) / (highest_high - lowest_low)).alias(f"williams_r_{period}")
    
    def _simple_cci(self, high: pl.Expr, low: pl.Expr, close: pl.Expr, period: int = 20) -> pl.Expr:
        """Simple CCI calculation"""
        typical_price = (high + low + close) / 3
        sma_tp = typical_price.rolling_mean(period)
        mean_dev = (typical_price - sma_tp).abs().rolling_mean(period)
        return ((typical_price - sma_tp) / (0.015 * mean_dev)).alias(f"cci_{period}")
    
    def _simple_atr(self, high: pl.Expr, low: pl.Expr, close: pl.Expr) -> pl.Expr:
        """Simple ATR calculation"""
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        true_range = pl.max_horizontal([tr1, tr2, tr3])
        return true_range.rolling_mean(14).alias("atr_14")
    
    def _simple_adx(self, high: pl.Expr, low: pl.Expr, close: pl.Expr) -> pl.Expr:
        """Simple ADX calculation"""
        # Simplified ADX calculation
        plus_dm = pl.when(high.diff() > low.diff().abs()).then(high.diff()).otherwise(0)
        minus_dm = pl.when(low.diff().abs() > high.diff()).then(low.diff().abs()).otherwise(0)
        atr = self._simple_atr(high, low, close)
        plus_di = 100 * (plus_dm.rolling_mean(14) / atr)
        minus_di = 100 * (minus_dm.rolling_mean(14) / atr)
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
        return dx.rolling_mean(14).alias("adx_14")
    
    # Advanced Moving Average Methods
    def _hull_ma(self, close: pl.Expr, period: int) -> pl.Expr:
        """Hull Moving Average"""
        wma_half = close.rolling_mean(period // 2)
        wma_full = close.rolling_mean(period)
        return (2 * wma_half - wma_full).rolling_mean(int(period ** 0.5))
    
    def _kama(self, close: pl.Expr, period: int) -> pl.Expr:
        """Kaufman's Adaptive Moving Average"""
        # Simplified KAMA calculation
        return close.ewm_mean(span=period)
    
    def _tema(self, close: pl.Expr, period: int) -> pl.Expr:
        """Triple Exponential Moving Average"""
        ema1 = close.ewm_mean(span=period)
        ema2 = ema1.ewm_mean(span=period)
        ema3 = ema2.ewm_mean(span=period)
        return 3 * ema1 - 3 * ema2 + ema3
    
    def _dema(self, close: pl.Expr, period: int) -> pl.Expr:
        """Double Exponential Moving Average"""
        ema1 = close.ewm_mean(span=period)
        ema2 = ema1.ewm_mean(span=period)
        return 2 * ema1 - ema2
    
    def _vwma(self, close: pl.Expr, volume: pl.Expr, period: int) -> pl.Expr:
        """Volume Weighted Moving Average"""
        return ((close * volume).rolling_sum(period) / volume.rolling_sum(period)).alias(f"vwma_{period}")
    
    # Advanced Technical Indicator Methods
    def _psar(self, high: pl.Expr, low: pl.Expr, close: pl.Expr) -> pl.Expr:
        """Parabolic SAR (simplified)"""
        return (high + low) / 2
    
    def _psar_af(self, high: pl.Expr, low: pl.Expr, close: pl.Expr) -> pl.Expr:
        """Parabolic SAR Acceleration Factor (simplified)"""
        return pl.lit(0.02)
    
    def _ichimoku_tenkan(self, high: pl.Expr, low: pl.Expr) -> pl.Expr:
        """Ichimoku Tenkan-sen"""
        return (high.rolling_max(9) + low.rolling_min(9)) / 2
    
    def _ichimoku_kijun(self, high: pl.Expr, low: pl.Expr) -> pl.Expr:
        """Ichimoku Kijun-sen"""
        return (high.rolling_max(26) + low.rolling_min(26)) / 2
    
    def _ichimoku_senkou_a(self, high: pl.Expr, low: pl.Expr) -> pl.Expr:
        """Ichimoku Senkou Span A"""
        tenkan = self._ichimoku_tenkan(high, low)
        kijun = self._ichimoku_kijun(high, low)
        return (tenkan + kijun) / 2
    
    def _ichimoku_senkou_b(self, high: pl.Expr, low: pl.Expr) -> pl.Expr:
        """Ichimoku Senkou Span B"""
        return (high.rolling_max(52) + low.rolling_min(52)) / 2
    
    def _ichimoku_chikou(self, close: pl.Expr) -> pl.Expr:
        """Ichimoku Chikou Span"""
        return close.shift(-26)
    
    def _keltner_upper(self, high: pl.Expr, low: pl.Expr, close: pl.Expr) -> pl.Expr:
        """Keltner Channel Upper"""
        atr = self._simple_atr(high, low, close)
        return close.rolling_mean(20) + (2 * atr)
    
    def _keltner_lower(self, high: pl.Expr, low: pl.Expr, close: pl.Expr) -> pl.Expr:
        """Keltner Channel Lower"""
        atr = self._simple_atr(high, low, close)
        return close.rolling_mean(20) - (2 * atr)
    
    def _keltner_middle(self, high: pl.Expr, low: pl.Expr, close: pl.Expr) -> pl.Expr:
        """Keltner Channel Middle"""
        return close.rolling_mean(20)
    
    def _donchian_upper(self, high: pl.Expr) -> pl.Expr:
        """Donchian Channel Upper"""
        return high.rolling_max(20)
    
    def _donchian_lower(self, low: pl.Expr) -> pl.Expr:
        """Donchian Channel Lower"""
        return low.rolling_min(20)
    
    def _donchian_middle(self, high: pl.Expr, low: pl.Expr) -> pl.Expr:
        """Donchian Channel Middle"""
        return (high.rolling_max(20) + low.rolling_min(20)) / 2
    
    def _supertrend(self, high: pl.Expr, low: pl.Expr, close: pl.Expr) -> pl.Expr:
        """Supertrend (simplified)"""
        atr = self._simple_atr(high, low, close)
        return (high + low) / 2 + (2 * atr)
    
    def _supertrend_direction(self, high: pl.Expr, low: pl.Expr, close: pl.Expr) -> pl.Expr:
        """Supertrend Direction"""
        return pl.when(close > self._supertrend(high, low, close)).then(1).otherwise(-1)
    
    def _zigzag_high(self, high: pl.Expr, low: pl.Expr) -> pl.Expr:
        """ZigZag High (simplified)"""
        return high.rolling_max(5)
    
    def _zigzag_low(self, high: pl.Expr, low: pl.Expr) -> pl.Expr:
        """ZigZag Low (simplified)"""
        return low.rolling_min(5)
    
    def _fractal_high(self, high: pl.Expr) -> pl.Expr:
        """Fractal High"""
        return pl.when(
            (high > high.shift(1)) & (high > high.shift(2)) & 
            (high > high.shift(-1)) & (high > high.shift(-2))
        ).then(high).otherwise(pl.lit(None))
    
    def _fractal_low(self, low: pl.Expr) -> pl.Expr:
        """Fractal Low"""
        return pl.when(
            (low < low.shift(1)) & (low < low.shift(2)) & 
            (low < low.shift(-1)) & (low < low.shift(-2))
        ).then(low).otherwise(pl.lit(None))
    
    def _awesome_oscillator(self, high: pl.Expr, low: pl.Expr) -> pl.Expr:
        """Awesome Oscillator"""
        return (high + low) / 2 - (high.shift(34) + low.shift(34)) / 2
    
    def _market_facilitation_index(self, high: pl.Expr, low: pl.Expr, volume: pl.Expr) -> pl.Expr:
        """Market Facilitation Index"""
        return ((high - low) / volume).alias("market_facilitation_index")
    
    def _ease_of_movement(self, high: pl.Expr, low: pl.Expr, volume: pl.Expr) -> pl.Expr:
        """Ease of Movement"""
        return (((high + low) / 2 - (high.shift(1) + low.shift(1)) / 2) / (volume / (high - low))).alias("ease_of_movement")
    
    def _mass_index(self, high: pl.Expr, low: pl.Expr) -> pl.Expr:
        """Mass Index"""
        hl_ratio = (high - low) / (high.rolling_mean(9) - low.rolling_mean(9))
        return hl_ratio.rolling_sum(25)
    
    def _negative_volume_index(self, close: pl.Expr, volume: pl.Expr) -> pl.Expr:
        """Negative Volume Index"""
        return pl.when(volume < volume.shift(1)).then(
            close / close.shift(1)
        ).otherwise(pl.lit(1)).cum_prod().alias("negative_volume_index")
    
    def _price_volume_trend(self, close: pl.Expr, volume: pl.Expr) -> pl.Expr:
        """Price Volume Trend"""
        return (close.pct_change() * volume).cum_sum().alias("price_volume_trend")
    
    def _williams_accumulation(self, high: pl.Expr, low: pl.Expr, close: pl.Expr, volume: pl.Expr) -> pl.Expr:
        """Williams Accumulation/Distribution"""
        return (((close - low) - (high - close)) / (high - low) * volume).alias("williams_accumulation")
    
    def _chaikin_oscillator(self, high: pl.Expr, low: pl.Expr, close: pl.Expr, volume: pl.Expr) -> pl.Expr:
        """Chaikin Oscillator"""
        ad = self._williams_accumulation(high, low, close, volume)
        return ad.ewm_mean(span=3) - ad.ewm_mean(span=10)
    
    def _force_index(self, close: pl.Expr, volume: pl.Expr) -> pl.Expr:
        """Force Index"""
        return close.diff() * volume
    
    # Advanced Volume Methods
    def _obv(self, close: pl.Expr, volume: pl.Expr) -> pl.Expr:
        """On-Balance Volume"""
        return pl.when(close > close.shift(1)).then(volume).otherwise(
            pl.when(close < close.shift(1)).then(-volume).otherwise(0)
        ).cum_sum().alias("obv")
    
    def _obv_ema(self, close: pl.Expr, volume: pl.Expr) -> pl.Expr:
        """On-Balance Volume EMA"""
        obv = self._obv(close, volume)
        return obv.ewm_mean(span=10).alias("obv_ema")
    
    def _ad_line(self, high: pl.Expr, low: pl.Expr, close: pl.Expr, volume: pl.Expr) -> pl.Expr:
        """Accumulation/Distribution Line"""
        return (((close - low) - (high - close)) / (high - low) * volume).alias("ad_line")
    
    def _ad_line_ema(self, high: pl.Expr, low: pl.Expr, close: pl.Expr, volume: pl.Expr) -> pl.Expr:
        """Accumulation/Distribution Line EMA"""
        ad = self._ad_line(high, low, close, volume)
        return ad.ewm_mean(span=10).alias("ad_line_ema")
    
    def _cmf(self, high: pl.Expr, low: pl.Expr, close: pl.Expr, volume: pl.Expr) -> pl.Expr:
        """Chaikin Money Flow"""
        mfv = ((close - low) - (high - close)) / (high - low) * volume
        return (mfv.rolling_sum(20) / volume.rolling_sum(20)).alias("cmf")
    
    def _cmf_ema(self, high: pl.Expr, low: pl.Expr, close: pl.Expr, volume: pl.Expr) -> pl.Expr:
        """Chaikin Money Flow EMA"""
        cmf = self._cmf(high, low, close, volume)
        return cmf.ewm_mean(span=10).alias("cmf_ema")
    
    def _mfi(self, high: pl.Expr, low: pl.Expr, close: pl.Expr, volume: pl.Expr, period: int) -> pl.Expr:
        """Money Flow Index"""
        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume
        positive_flow = pl.when(typical_price > typical_price.shift(1)).then(money_flow).otherwise(0)
        negative_flow = pl.when(typical_price < typical_price.shift(1)).then(money_flow).otherwise(0)
        mfi_ratio = positive_flow.rolling_sum(period) / negative_flow.rolling_sum(period)
        return (100 - (100 / (1 + mfi_ratio))).alias(f"mfi_{period}")
    
    def _vwap_period(self, high: pl.Expr, low: pl.Expr, close: pl.Expr, volume: pl.Expr, period: int) -> pl.Expr:
        """Volume Weighted Average Price for period"""
        typical_price = (high + low + close) / 3
        return ((typical_price * volume).rolling_sum(period) / volume.rolling_sum(period)).alias(f"vwap_{period}")
    
    def _volume_weighted_price(self, high: pl.Expr, low: pl.Expr, close: pl.Expr, volume: pl.Expr) -> pl.Expr:
        """Volume Weighted Price"""
        typical_price = (high + low + close) / 3
        return (typical_price * volume).alias("volume_weighted_price")
    
    # Advanced Volatility Methods
    def _gk_vol(self, high: pl.Expr, low: pl.Expr, open: pl.Expr, close: pl.Expr, period: int) -> pl.Expr:
        """Garman-Klass Volatility"""
        return (0.5 * (high.log() - low.log())**2 - 
                (2 * pl.lit(2).log() - 1) * (close.log() - open.log())**2).rolling_mean(period)
    
    def _parkinson_vol(self, high: pl.Expr, low: pl.Expr, period: int) -> pl.Expr:
        """Parkinson Volatility"""
        return (0.25 * (high.log() - low.log())**2).rolling_mean(period)
    
    def _rs_vol(self, high: pl.Expr, low: pl.Expr, open: pl.Expr, close: pl.Expr, period: int) -> pl.Expr:
        """Rogers-Satchell Volatility"""
        return ((high.log() - close.log()) * (high.log() - open.log()) + 
                (low.log() - close.log()) * (low.log() - open.log())).rolling_mean(period)
    
    def _yz_vol(self, high: pl.Expr, low: pl.Expr, open: pl.Expr, close: pl.Expr, period: int) -> pl.Expr:
        """Yang-Zhang Volatility"""
        overnight = (open.log() - close.shift(1).log())**2
        intraday = (close.log() - open.log())**2
        return (overnight + intraday).rolling_mean(period)
    
    # Advanced Time Methods
    def _intraday_seasonality(self, ts: pl.Expr, period: int) -> pl.Expr:
        """Intraday Seasonality (simplified)"""
        hour = ts.dt.hour()
        return pl.when(hour.is_in([9, 10, 11, 14, 15])).then(1.0).otherwise(0.5)
    
    def _weekly_seasonality(self, ts: pl.Expr, day: str) -> pl.Expr:
        """Weekly Seasonality (simplified)"""
        weekday = ts.dt.weekday()
        if day == "monday":
            return pl.when(weekday == 0).then(1.0).otherwise(0.5)
        elif day == "friday":
            return pl.when(weekday == 4).then(1.0).otherwise(0.5)
        return pl.lit(0.5)
    
    def _monthly_seasonality(self, ts: pl.Expr, period: str) -> pl.Expr:
        """Monthly Seasonality (simplified)"""
        day = ts.dt.day()
        if period == "beginning":
            return pl.when(day <= 5).then(1.0).otherwise(0.5)
        elif period == "end":
            return pl.when(day >= 25).then(1.0).otherwise(0.5)
        return pl.lit(0.5)
    
    def _quarterly_seasonality(self, ts: pl.Expr, period: str) -> pl.Expr:
        """Quarterly Seasonality (simplified)"""
        month = ts.dt.month()
        if period == "beginning":
            return pl.when(month.is_in([1, 4, 7, 10])).then(1.0).otherwise(0.5)
        elif period == "end":
            return pl.when(month.is_in([3, 6, 9, 12])).then(1.0).otherwise(0.5)
        return pl.lit(0.5)
    
    def _holiday_effect(self, ts: pl.Expr, effect_type: str) -> pl.Expr:
        """Holiday Effect (simplified)"""
        month = ts.dt.month()
        day = ts.dt.day()
        if effect_type == "pre":
            return pl.when((month == 12) & (day == 23)).then(1.0).otherwise(0.5)
        elif effect_type == "post":
            return pl.when((month == 12) & (day == 27)).then(1.0).otherwise(0.5)
        elif effect_type == "holiday":
            return pl.when((month == 12) & (day.is_in([24, 25, 26]))).then(1.0).otherwise(0.5)
        return pl.lit(0.5)

# Create a global instance
simple_feature_computer = SimpleFeatureComputer()
