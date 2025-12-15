#pragma once

#include "common.h"
#include <unordered_map>
#include <memory>
#include <thread>
#include <future>

namespace ibkr_trading {

// Forward declarations
class FeatureCache;
class SIMDFeatureCalculator;

// High-performance feature pipeline
class FeaturePipeline {
public:
    explicit FeaturePipeline(const FeatureConfig& config);
    ~FeaturePipeline();
    
    // Compute features for a single symbol
    FeatureVector compute_features(const MarketData& market_data);
    
    // Batch feature computation
    std::unordered_map<std::string, FeatureVector>
    compute_features_batch(const std::unordered_map<std::string, MarketData>& market_data);
    
    // Incremental feature updates
    void update_features(const std::string& symbol, const MarketData& market_data);
    
    // Get cached features
    FeatureVector get_cached_features(const std::string& symbol);
    
    // Clear feature cache
    void clear_cache();
    
    // Get performance metrics
    PerformanceMetrics get_metrics() const;

private:
    FeatureConfig config_;
    std::unique_ptr<FeatureCache> cache_;
    std::unique_ptr<SIMDFeatureCalculator> simd_calc_;
    
    // Performance tracking
    mutable PerformanceMetrics metrics_;
    mutable std::mutex metrics_mutex_;
    
    // Feature computation helpers
    FeatureVector compute_price_features(const MarketData& md);
    FeatureVector compute_volume_features(const MarketData& md);
    FeatureVector compute_volatility_features(const MarketData& md);
    FeatureVector compute_momentum_features(const MarketData& md);
    FeatureVector compute_technical_features(const MarketData& md);
    FeatureVector compute_microstructure_features(const MarketData& md);
    
    // SIMD-optimized feature computation
    void compute_features_simd(const MarketData& md, float* output);
    void compute_price_features_simd(const MarketData& md, float* output);
    void compute_volume_features_simd(const MarketData& md, float* output);
    void compute_volatility_features_simd(const MarketData& md, float* output);
    void compute_momentum_features_simd(const MarketData& md, float* output);
    void compute_technical_features_simd(const MarketData& md, float* output);
    void compute_microstructure_features_simd(const MarketData& md, float* output);
};

// Feature cache for incremental updates
class FeatureCache {
public:
    explicit FeatureCache(size_t max_cache_size = 1000);
    ~FeatureCache();
    
    // Cache operations
    void store(const std::string& symbol, const FeatureVector& features);
    FeatureVector retrieve(const std::string& symbol);
    bool exists(const std::string& symbol) const;
    void remove(const std::string& symbol);
    void clear();
    
    // Cache statistics
    size_t size() const { return cache_.size(); }
    size_t max_size() const { return max_size_; }
    double hit_rate() const { return hit_rate_; }
    
    // Incremental updates
    void update_incremental(const std::string& symbol, const MarketData& md);
    
private:
    size_t max_size_;
    std::unordered_map<std::string, FeatureVector> cache_;
    std::unordered_map<std::string, std::chrono::high_resolution_clock::time_point> timestamps_;
    
    // Cache statistics
    mutable size_t hits_;
    mutable size_t misses_;
    mutable double hit_rate_;
    mutable std::mutex stats_mutex_;
    
    // LRU eviction
    void evict_lru();
    void update_stats(bool hit);
};

// SIMD-optimized feature calculator
class SIMDFeatureCalculator {
public:
    SIMDFeatureCalculator();
    ~SIMDFeatureCalculator();
    
    // Price-based features
    void calculate_returns(const float* prices, float* returns, size_t n);
    void calculate_log_returns(const float* prices, float* returns, size_t n);
    void calculate_volatility(const float* returns, float* volatility, size_t n, size_t window);
    void calculate_momentum(const float* prices, float* momentum, size_t n, size_t lookback);
    void calculate_rsi(const float* prices, float* rsi, size_t n, size_t period);
    void calculate_macd(const float* prices, float* macd, size_t n, size_t fast, size_t slow, size_t signal);
    
    // Volume-based features
    void calculate_volume_ratio(const float* volumes, float* ratio, size_t n, size_t window);
    void calculate_volume_momentum(const float* volumes, float* momentum, size_t n, size_t lookback);
    void calculate_volume_volatility(const float* volumes, float* volatility, size_t n, size_t window);
    
    // Microstructure features
    void calculate_spread_features(const float* bids, const float* asks, float* features, size_t n);
    void calculate_imbalance_features(const float* bid_sizes, const float* ask_sizes, float* features, size_t n);
    void calculate_impact_features(const float* prices, const float* volumes, float* features, size_t n);
    
    // Technical indicators
    void calculate_sma(const float* prices, float* sma, size_t n, size_t window);
    void calculate_ema(const float* prices, float* ema, size_t n, float alpha);
    void calculate_bollinger_bands(const float* prices, float* upper, float* lower, size_t n, size_t window, float std_dev);
    void calculate_stochastic(const float* high, const float* low, const float* close, float* k, float* d, size_t n, size_t k_period, size_t d_period);
    
    // Statistical features
    void calculate_skewness(const float* data, float* skewness, size_t n, size_t window);
    void calculate_kurtosis(const float* data, float* kurtosis, size_t n, size_t window);
    void calculate_autocorrelation(const float* data, float* autocorr, size_t n, size_t max_lag);
    
    // Cross-sectional features
    void calculate_rank_features(const float* data, float* ranks, size_t n);
    void calculate_percentile_features(const float* data, float* percentiles, size_t n, const std::vector<float>& percentiles);
    void calculate_zscore_features(const float* data, float* zscores, size_t n, size_t window);
    
    // Performance info
    bool has_avx2() const { return has_avx2_; }
    bool has_avx512() const { return has_avx512_; }

private:
    bool has_avx2_;
    bool has_avx512_;
    
    // AVX2 implementations
    void calculate_returns_avx2(const float* prices, float* returns, size_t n);
    void calculate_volatility_avx2(const float* returns, float* volatility, size_t n, size_t window);
    void calculate_momentum_avx2(const float* prices, float* momentum, size_t n, size_t lookback);
    void calculate_sma_avx2(const float* prices, float* sma, size_t n, size_t window);
    void calculate_ema_avx2(const float* prices, float* ema, size_t n, float alpha);
    
    // AVX512 implementations
    void calculate_returns_avx512(const float* prices, float* returns, size_t n);
    void calculate_volatility_avx512(const float* returns, float* volatility, size_t n, size_t window);
    void calculate_momentum_avx512(const float* prices, float* momentum, size_t n, size_t lookback);
    void calculate_sma_avx512(const float* prices, float* sma, size_t n, size_t window);
    void calculate_ema_avx512(const float* prices, float* ema, size_t n, float alpha);
    
    // Fallback implementations
    void calculate_returns_fallback(const float* prices, float* returns, size_t n);
    void calculate_volatility_fallback(const float* returns, float* volatility, size_t n, size_t window);
    void calculate_momentum_fallback(const float* prices, float* momentum, size_t n, size_t lookback);
    void calculate_sma_fallback(const float* prices, float* sma, size_t n, size_t window);
    void calculate_ema_fallback(const float* prices, float* ema, size_t n, float alpha);
};

// Feature engineering utilities
namespace feature_utils {
    // Rolling window operations
    void rolling_mean(const float* data, float* result, size_t n, size_t window);
    void rolling_std(const float* data, float* result, size_t n, size_t window);
    void rolling_min(const float* data, float* result, size_t n, size_t window);
    void rolling_max(const float* data, float* result, size_t n, size_t window);
    void rolling_median(const float* data, float* result, size_t n, size_t window);
    void rolling_quantile(const float* data, float* result, size_t n, size_t window, float quantile);
    
    // Lag features
    void create_lag_features(const float* data, float* result, size_t n, const std::vector<size_t>& lags);
    void create_diff_features(const float* data, float* result, size_t n, const std::vector<size_t>& diffs);
    
    // Interaction features
    void create_interaction_features(const float* data1, const float* data2, float* result, size_t n);
    void create_polynomial_features(const float* data, float* result, size_t n, size_t degree);
    
    // Time-based features
    void create_time_features(const int64_t* timestamps, float* result, size_t n);
    void create_cyclical_features(const int64_t* timestamps, float* result, size_t n);
    
    // Cross-sectional features
    void calculate_cross_sectional_rank(const float* data, float* ranks, size_t n);
    void calculate_cross_sectional_percentile(const float* data, float* percentiles, size_t n, float percentile);
    void calculate_cross_sectional_zscore(const float* data, float* zscores, size_t n);
}

} // namespace ibkr_trading
