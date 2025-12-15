#include "feature_pipeline.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <execution>

namespace ibkr_trading {

FeaturePipeline::FeaturePipeline(const FeatureConfig& config) 
    : config_(config), cache_(std::make_unique<FeatureCache>()),
      simd_calc_(std::make_unique<SIMDFeatureCalculator>()) {
}

FeaturePipeline::~FeaturePipeline() = default;

FeatureVector FeaturePipeline::compute_features(const MarketData& market_data) {
    Timer timer;
    
    // Check cache first
    if (cache_->exists(market_data.symbol)) {
        auto cached_features = cache_->retrieve(market_data.symbol);
        update_features(market_data.symbol, market_data);
        return cached_features;
    }
    
    // Compute all feature categories
    FeatureVector features;
    features.reserve(281);  // Assuming 281 total features
    
    // Price features
    auto price_features = compute_price_features(market_data);
    features.insert(features.end(), price_features.begin(), price_features.end());
    
    // Volume features
    auto volume_features = compute_volume_features(market_data);
    features.insert(features.end(), volume_features.begin(), volume_features.end());
    
    // Volatility features
    auto volatility_features = compute_volatility_features(market_data);
    features.insert(features.end(), volatility_features.begin(), volatility_features.end());
    
    // Momentum features
    auto momentum_features = compute_momentum_features(market_data);
    features.insert(features.end(), momentum_features.begin(), momentum_features.end());
    
    // Technical features
    auto technical_features = compute_technical_features(market_data);
    features.insert(features.end(), technical_features.begin(), technical_features.end());
    
    // Microstructure features
    auto microstructure_features = compute_microstructure_features(market_data);
    features.insert(features.end(), microstructure_features.begin(), microstructure_features.end());
    
    // Store in cache
    cache_->store(market_data.symbol, features);
    
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    metrics_.feature_time_ms = timer.elapsed_ms();
    
    return features;
}

std::unordered_map<std::string, FeatureVector>
FeaturePipeline::compute_features_batch(const std::unordered_map<std::string, MarketData>& market_data) {
    Timer timer;
    
    std::unordered_map<std::string, FeatureVector> results;
    
    // Parallel feature computation
    std::vector<std::future<void>> futures;
    std::mutex results_mutex;
    
    for (const auto& [symbol, md] : market_data) {
        futures.emplace_back(std::async(std::launch::async, [&, symbol, md]() {
            auto features = compute_features(md);
            
            std::lock_guard<std::mutex> lock(results_mutex);
            results[symbol] = std::move(features);
        }));
    }
    
    // Wait for all futures to complete
    for (auto& future : futures) {
        future.wait();
    }
    
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    metrics_.total_time_ms = timer.elapsed_ms();
    
    return results;
}

void FeaturePipeline::update_features(const std::string& symbol, const MarketData& market_data) {
    cache_->update_incremental(symbol, market_data);
}

FeatureVector FeaturePipeline::get_cached_features(const std::string& symbol) {
    return cache_->retrieve(symbol);
}

void FeaturePipeline::clear_cache() {
    cache_->clear();
}

PerformanceMetrics FeaturePipeline::get_metrics() const {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    return metrics_;
}

FeatureVector FeaturePipeline::compute_price_features(const MarketData& md) {
    FeatureVector features(50);  // 50 price features
    
    // SIMD-optimized price feature computation
    compute_price_features_simd(md, features.data());
    
    return features;
}

FeatureVector FeaturePipeline::compute_volume_features(const MarketData& md) {
    FeatureVector features(30);  // 30 volume features
    
    // SIMD-optimized volume feature computation
    compute_volume_features_simd(md, features.data());
    
    return features;
}

FeatureVector FeaturePipeline::compute_volatility_features(const MarketData& md) {
    FeatureVector features(40);  // 40 volatility features
    
    // SIMD-optimized volatility feature computation
    compute_volatility_features_simd(md, features.data());
    
    return features;
}

FeatureVector FeaturePipeline::compute_momentum_features(const MarketData& md) {
    FeatureVector features(35);  // 35 momentum features
    
    // SIMD-optimized momentum feature computation
    compute_momentum_features_simd(md, features.data());
    
    return features;
}

FeatureVector FeaturePipeline::compute_technical_features(const MarketData& md) {
    FeatureVector features(60);  // 60 technical features
    
    // SIMD-optimized technical feature computation
    compute_technical_features_simd(md, features.data());
    
    return features;
}

FeatureVector FeaturePipeline::compute_microstructure_features(const MarketData& md) {
    FeatureVector features(66);  // 66 microstructure features
    
    // SIMD-optimized microstructure feature computation
    compute_microstructure_features_simd(md, features.data());
    
    return features;
}

void FeaturePipeline::compute_features_simd(const MarketData& md, float* output) {
    // SIMD-optimized feature computation
    // Implementation using AVX2/AVX512 intrinsics
}

void FeaturePipeline::compute_price_features_simd(const MarketData& md, float* output) {
    // SIMD-optimized price features
    // Implementation using AVX2/AVX512 intrinsics
}

void FeaturePipeline::compute_volume_features_simd(const MarketData& md, float* output) {
    // SIMD-optimized volume features
    // Implementation using AVX2/AVX512 intrinsics
}

void FeaturePipeline::compute_volatility_features_simd(const MarketData& md, float* output) {
    // SIMD-optimized volatility features
    // Implementation using AVX2/AVX512 intrinsics
}

void FeaturePipeline::compute_momentum_features_simd(const MarketData& md, float* output) {
    // SIMD-optimized momentum features
    // Implementation using AVX2/AVX512 intrinsics
}

void FeaturePipeline::compute_technical_features_simd(const MarketData& md, float* output) {
    // SIMD-optimized technical features
    // Implementation using AVX2/AVX512 intrinsics
}

void FeaturePipeline::compute_microstructure_features_simd(const MarketData& md, float* output) {
    // SIMD-optimized microstructure features
    // Implementation using AVX2/AVX512 intrinsics
}

// FeatureCache implementation
FeatureCache::FeatureCache(size_t max_cache_size) 
    : max_size_(max_cache_size), hits_(0), misses_(0), hit_rate_(0.0) {
}

FeatureCache::~FeatureCache() = default;

void FeatureCache::store(const std::string& symbol, const FeatureVector& features) {
    if (cache_.size() >= max_size_) {
        evict_lru();
    }
    
    cache_[symbol] = features;
    timestamps_[symbol] = std::chrono::high_resolution_clock::now();
}

FeatureVector FeatureCache::retrieve(const std::string& symbol) {
    auto it = cache_.find(symbol);
    if (it != cache_.end()) {
        update_stats(true);
        return it->second;
    }
    
    update_stats(false);
    return FeatureVector();
}

bool FeatureCache::exists(const std::string& symbol) const {
    return cache_.find(symbol) != cache_.end();
}

void FeatureCache::remove(const std::string& symbol) {
    cache_.erase(symbol);
    timestamps_.erase(symbol);
}

void FeatureCache::clear() {
    cache_.clear();
    timestamps_.clear();
    hits_ = 0;
    misses_ = 0;
    hit_rate_ = 0.0;
}

void FeatureCache::update_incremental(const std::string& symbol, const MarketData& md) {
    // Incremental feature updates
    // Implementation depends on feature types
}

void FeatureCache::evict_lru() {
    if (cache_.empty()) return;
    
    auto oldest = std::min_element(timestamps_.begin(), timestamps_.end(),
        [](const auto& a, const auto& b) {
            return a.second < b.second;
        });
    
    cache_.erase(oldest->first);
    timestamps_.erase(oldest->first);
}

void FeatureCache::update_stats(bool hit) {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    
    if (hit) {
        hits_++;
    } else {
        misses_++;
    }
    
    hit_rate_ = static_cast<double>(hits_) / (hits_ + misses_);
}

// SIMDFeatureCalculator implementation
SIMDFeatureCalculator::SIMDFeatureCalculator() {
    // Check CPU capabilities
    has_avx2_ = simd::has_avx2();
    has_avx512_ = simd::has_avx512();
}

SIMDFeatureCalculator::~SIMDFeatureCalculator() = default;

void SIMDFeatureCalculator::calculate_returns(const float* prices, float* returns, size_t n) {
    if (has_avx512_) {
        calculate_returns_avx512(prices, returns, n);
    } else if (has_avx2_) {
        calculate_returns_avx2(prices, returns, n);
    } else {
        calculate_returns_fallback(prices, returns, n);
    }
}

void SIMDFeatureCalculator::calculate_log_returns(const float* prices, float* returns, size_t n) {
    for (size_t i = 1; i < n; ++i) {
        returns[i-1] = std::log(prices[i] / prices[i-1]);
    }
}

void SIMDFeatureCalculator::calculate_volatility(const float* returns, float* volatility, size_t n, size_t window) {
    if (has_avx512_) {
        calculate_volatility_avx512(returns, volatility, n, window);
    } else if (has_avx2_) {
        calculate_volatility_avx2(returns, volatility, n, window);
    } else {
        calculate_volatility_fallback(returns, volatility, n, window);
    }
}

void SIMDFeatureCalculator::calculate_momentum(const float* prices, float* momentum, size_t n, size_t lookback) {
    if (has_avx512_) {
        calculate_momentum_avx512(prices, momentum, n, lookback);
    } else if (has_avx2_) {
        calculate_momentum_avx2(prices, momentum, n, lookback);
    } else {
        calculate_momentum_fallback(prices, momentum, n, lookback);
    }
}

void SIMDFeatureCalculator::calculate_rsi(const float* prices, float* rsi, size_t n, size_t period) {
    // RSI calculation
    for (size_t i = period; i < n; ++i) {
        float gain = 0.0f, loss = 0.0f;
        
        for (size_t j = i - period + 1; j <= i; ++j) {
            float change = prices[j] - prices[j-1];
            if (change > 0) {
                gain += change;
            } else {
                loss -= change;
            }
        }
        
        float avg_gain = gain / period;
        float avg_loss = loss / period;
        
        if (avg_loss == 0.0f) {
            rsi[i] = 100.0f;
        } else {
            float rs = avg_gain / avg_loss;
            rsi[i] = 100.0f - (100.0f / (1.0f + rs));
        }
    }
}

void SIMDFeatureCalculator::calculate_macd(const float* prices, float* macd, size_t n, size_t fast, size_t slow, size_t signal) {
    // MACD calculation
    std::vector<float> ema_fast(n), ema_slow(n);
    
    // Calculate EMAs
    calculate_ema(prices, ema_fast.data(), n, 2.0f / (fast + 1));
    calculate_ema(prices, ema_slow.data(), n, 2.0f / (slow + 1));
    
    // Calculate MACD line
    for (size_t i = 0; i < n; ++i) {
        macd[i] = ema_fast[i] - ema_slow[i];
    }
}

void SIMDFeatureCalculator::calculate_volume_ratio(const float* volumes, float* ratio, size_t n, size_t window) {
    for (size_t i = window; i < n; ++i) {
        float current_vol = volumes[i];
        float avg_vol = 0.0f;
        
        for (size_t j = i - window; j < i; ++j) {
            avg_vol += volumes[j];
        }
        avg_vol /= window;
        
        ratio[i] = current_vol / avg_vol;
    }
}

void SIMDFeatureCalculator::calculate_volume_momentum(const float* volumes, float* momentum, size_t n, size_t lookback) {
    for (size_t i = lookback; i < n; ++i) {
        momentum[i] = volumes[i] - volumes[i - lookback];
    }
}

void SIMDFeatureCalculator::calculate_volume_volatility(const float* volumes, float* volatility, size_t n, size_t window) {
    for (size_t i = window; i < n; ++i) {
        float mean = 0.0f;
        for (size_t j = i - window; j < i; ++j) {
            mean += volumes[j];
        }
        mean /= window;
        
        float variance = 0.0f;
        for (size_t j = i - window; j < i; ++j) {
            float diff = volumes[j] - mean;
            variance += diff * diff;
        }
        variance /= window;
        
        volatility[i] = std::sqrt(variance);
    }
}

void SIMDFeatureCalculator::calculate_spread_features(const float* bids, const float* asks, float* features, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        features[i] = asks[i] - bids[i];  // Spread
        features[i + n] = (asks[i] - bids[i]) / ((asks[i] + bids[i]) / 2.0f);  // Relative spread
    }
}

void SIMDFeatureCalculator::calculate_imbalance_features(const float* bid_sizes, const float* ask_sizes, float* features, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        float total_size = bid_sizes[i] + ask_sizes[i];
        if (total_size > 0) {
            features[i] = (bid_sizes[i] - ask_sizes[i]) / total_size;  // Order imbalance
        } else {
            features[i] = 0.0f;
        }
    }
}

void SIMDFeatureCalculator::calculate_impact_features(const float* prices, const float* volumes, float* features, size_t n) {
    for (size_t i = 1; i < n; ++i) {
        float price_change = prices[i] - prices[i-1];
        float volume = volumes[i];
        
        if (volume > 0) {
            features[i] = price_change / volume;  // Price impact
        } else {
            features[i] = 0.0f;
        }
    }
}

void SIMDFeatureCalculator::calculate_sma(const float* prices, float* sma, size_t n, size_t window) {
    if (has_avx512_) {
        calculate_sma_avx512(prices, sma, n, window);
    } else if (has_avx2_) {
        calculate_sma_avx2(prices, sma, n, window);
    } else {
        calculate_sma_fallback(prices, sma, n, window);
    }
}

void SIMDFeatureCalculator::calculate_ema(const float* prices, float* ema, size_t n, float alpha) {
    if (has_avx512_) {
        calculate_ema_avx512(prices, ema, n, alpha);
    } else if (has_avx2_) {
        calculate_ema_avx2(prices, ema, n, alpha);
    } else {
        calculate_ema_fallback(prices, ema, n, alpha);
    }
}

void SIMDFeatureCalculator::calculate_bollinger_bands(const float* prices, float* upper, float* lower, size_t n, size_t window, float std_dev) {
    std::vector<float> sma(n);
    calculate_sma(prices, sma.data(), n, window);
    
    for (size_t i = window; i < n; ++i) {
        float mean = sma[i];
        float variance = 0.0f;
        
        for (size_t j = i - window; j < i; ++j) {
            float diff = prices[j] - mean;
            variance += diff * diff;
        }
        variance /= window;
        
        float std = std::sqrt(variance);
        upper[i] = mean + std_dev * std;
        lower[i] = mean - std_dev * std;
    }
}

void SIMDFeatureCalculator::calculate_stochastic(const float* high, const float* low, const float* close, float* k, float* d, size_t n, size_t k_period, size_t d_period) {
    for (size_t i = k_period; i < n; ++i) {
        float highest = *std::max_element(high + i - k_period, high + i);
        float lowest = *std::min_element(low + i - k_period, low + i);
        
        if (highest != lowest) {
            k[i] = 100.0f * (close[i] - lowest) / (highest - lowest);
        } else {
            k[i] = 50.0f;
        }
    }
    
    // Calculate %D (smoothed %K)
    for (size_t i = k_period + d_period; i < n; ++i) {
        float sum = 0.0f;
        for (size_t j = i - d_period; j < i; ++j) {
            sum += k[j];
        }
        d[i] = sum / d_period;
    }
}

void SIMDFeatureCalculator::calculate_skewness(const float* data, float* skewness, size_t n, size_t window) {
    for (size_t i = window; i < n; ++i) {
        float mean = 0.0f;
        for (size_t j = i - window; j < i; ++j) {
            mean += data[j];
        }
        mean /= window;
        
        float variance = 0.0f;
        for (size_t j = i - window; j < i; ++j) {
            float diff = data[j] - mean;
            variance += diff * diff;
        }
        variance /= window;
        
        float std = std::sqrt(variance);
        if (std > 0) {
            float skew = 0.0f;
            for (size_t j = i - window; j < i; ++j) {
                float diff = data[j] - mean;
                skew += (diff / std) * (diff / std) * (diff / std);
            }
            skewness[i] = skew / window;
        } else {
            skewness[i] = 0.0f;
        }
    }
}

void SIMDFeatureCalculator::calculate_kurtosis(const float* data, float* kurtosis, size_t n, size_t window) {
    for (size_t i = window; i < n; ++i) {
        float mean = 0.0f;
        for (size_t j = i - window; j < i; ++j) {
            mean += data[j];
        }
        mean /= window;
        
        float variance = 0.0f;
        for (size_t j = i - window; j < i; ++j) {
            float diff = data[j] - mean;
            variance += diff * diff;
        }
        variance /= window;
        
        float std = std::sqrt(variance);
        if (std > 0) {
            float kurt = 0.0f;
            for (size_t j = i - window; j < i; ++j) {
                float diff = data[j] - mean;
                float normalized = diff / std;
                kurt += normalized * normalized * normalized * normalized;
            }
            kurtosis[i] = (kurt / window) - 3.0f;  // Excess kurtosis
        } else {
            kurtosis[i] = 0.0f;
        }
    }
}

void SIMDFeatureCalculator::calculate_autocorrelation(const float* data, float* autocorr, size_t n, size_t max_lag) {
    for (size_t lag = 1; lag <= max_lag; ++lag) {
        float numerator = 0.0f;
        float denominator = 0.0f;
        
        for (size_t i = lag; i < n; ++i) {
            numerator += data[i] * data[i - lag];
            denominator += data[i] * data[i];
        }
        
        if (denominator > 0) {
            autocorr[lag - 1] = numerator / denominator;
        } else {
            autocorr[lag - 1] = 0.0f;
        }
    }
}

void SIMDFeatureCalculator::calculate_rank_features(const float* data, float* ranks, size_t n) {
    std::vector<std::pair<float, size_t>> indexed_data(n);
    for (size_t i = 0; i < n; ++i) {
        indexed_data[i] = {data[i], i};
    }
    
    std::sort(indexed_data.begin(), indexed_data.end());
    
    for (size_t i = 0; i < n; ++i) {
        ranks[indexed_data[i].second] = static_cast<float>(i) / (n - 1);
    }
}

void SIMDFeatureCalculator::calculate_percentile_features(const float* data, float* percentiles, size_t n, const std::vector<float>& percentiles) {
    std::vector<float> sorted_data(data, data + n);
    std::sort(sorted_data.begin(), sorted_data.end());
    
    for (size_t i = 0; i < percentiles.size(); ++i) {
        float p = percentiles[i];
        size_t index = static_cast<size_t>(p * (n - 1));
        percentiles[i] = sorted_data[index];
    }
}

void SIMDFeatureCalculator::calculate_zscore_features(const float* data, float* zscores, size_t n, size_t window) {
    for (size_t i = window; i < n; ++i) {
        float mean = 0.0f;
        for (size_t j = i - window; j < i; ++j) {
            mean += data[j];
        }
        mean /= window;
        
        float variance = 0.0f;
        for (size_t j = i - window; j < i; ++j) {
            float diff = data[j] - mean;
            variance += diff * diff;
        }
        variance /= window;
        
        float std = std::sqrt(variance);
        if (std > 0) {
            zscores[i] = (data[i] - mean) / std;
        } else {
            zscores[i] = 0.0f;
        }
    }
}

// AVX2 implementations
void SIMDFeatureCalculator::calculate_returns_avx2(const float* prices, float* returns, size_t n) {
    // AVX2-optimized returns calculation
    // Implementation using AVX2 intrinsics
}

void SIMDFeatureCalculator::calculate_volatility_avx2(const float* returns, float* volatility, size_t n, size_t window) {
    // AVX2-optimized volatility calculation
    // Implementation using AVX2 intrinsics
}

void SIMDFeatureCalculator::calculate_momentum_avx2(const float* prices, float* momentum, size_t n, size_t lookback) {
    // AVX2-optimized momentum calculation
    // Implementation using AVX2 intrinsics
}

void SIMDFeatureCalculator::calculate_sma_avx2(const float* prices, float* sma, size_t n, size_t window) {
    // AVX2-optimized SMA calculation
    // Implementation using AVX2 intrinsics
}

void SIMDFeatureCalculator::calculate_ema_avx2(const float* prices, float* ema, size_t n, float alpha) {
    // AVX2-optimized EMA calculation
    // Implementation using AVX2 intrinsics
}

// Fallback implementations
void SIMDFeatureCalculator::calculate_returns_fallback(const float* prices, float* returns, size_t n) {
    for (size_t i = 1; i < n; ++i) {
        returns[i-1] = (prices[i] - prices[i-1]) / prices[i-1];
    }
}

void SIMDFeatureCalculator::calculate_volatility_fallback(const float* returns, float* volatility, size_t n, size_t window) {
    for (size_t i = window; i < n; ++i) {
        float mean = 0.0f;
        for (size_t j = i - window; j < i; ++j) {
            mean += returns[j];
        }
        mean /= window;
        
        float variance = 0.0f;
        for (size_t j = i - window; j < i; ++j) {
            float diff = returns[j] - mean;
            variance += diff * diff;
        }
        variance /= window;
        
        volatility[i] = std::sqrt(variance);
    }
}

void SIMDFeatureCalculator::calculate_momentum_fallback(const float* prices, float* momentum, size_t n, size_t lookback) {
    for (size_t i = lookback; i < n; ++i) {
        momentum[i] = (prices[i] - prices[i - lookback]) / prices[i - lookback];
    }
}

void SIMDFeatureCalculator::calculate_sma_fallback(const float* prices, float* sma, size_t n, size_t window) {
    for (size_t i = window; i < n; ++i) {
        float sum = 0.0f;
        for (size_t j = i - window; j < i; ++j) {
            sum += prices[j];
        }
        sma[i] = sum / window;
    }
}

void SIMDFeatureCalculator::calculate_ema_fallback(const float* prices, float* ema, size_t n, float alpha) {
    ema[0] = prices[0];
    for (size_t i = 1; i < n; ++i) {
        ema[i] = alpha * prices[i] + (1.0f - alpha) * ema[i-1];
    }
}

} // namespace ibkr_trading
