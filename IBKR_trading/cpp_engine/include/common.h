#pragma once

#include <vector>
#include <array>
#include <memory>
#include <string>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <immintrin.h>  // AVX2/AVX512 support

namespace ibkr_trading {

// Type aliases for better readability
using FeatureVector = std::vector<float>;
using FeatureMatrix = std::vector<std::vector<float>>;
using PredictionVector = std::vector<float>;
using SymbolVector = std::vector<std::string>;

// SIMD vector types
using AVXVector = __m256;
using AVX512Vector = __m512;

// Performance timing utilities
class Timer {
public:
    Timer() : start_(std::chrono::high_resolution_clock::now()) {}
    
    double elapsed_ms() const {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start_);
        return duration.count() / 1000.0;
    }
    
    void reset() {
        start_ = std::chrono::high_resolution_clock::now();
    }

private:
    std::chrono::high_resolution_clock::time_point start_;
};

// Memory alignment utilities
constexpr size_t ALIGNMENT = 32;  // AVX2 alignment

template<typename T>
T* aligned_alloc(size_t count) {
    return static_cast<T*>(std::aligned_alloc(ALIGNMENT, sizeof(T) * count));
}

template<typename T>
void aligned_free(T* ptr) {
    std::free(ptr);
}

// SIMD utilities
namespace simd {
    // Check if CPU supports AVX2
    bool has_avx2();
    
    // Check if CPU supports AVX512
    bool has_avx512();
    
    // Vectorized dot product
    float dot_product_avx2(const float* a, const float* b, size_t n);
    
    // Vectorized matrix-vector multiplication
    void matvec_mult_avx2(const float* matrix, const float* vector, 
                         float* result, size_t rows, size_t cols);
    
    // Vectorized softmax
    void softmax_avx2(const float* input, float* output, size_t n);
    
    // Vectorized normalization
    void normalize_avx2(float* data, size_t n);
}

// Model configuration
struct ModelConfig {
    std::string model_path;
    std::string model_type;
    size_t input_features;
    size_t output_classes;
    bool use_simd = true;
    bool use_gpu = false;
};

// Feature configuration
struct FeatureConfig {
    std::vector<std::string> feature_names;
    std::vector<float> feature_scales;
    std::vector<float> feature_means;
    std::vector<float> feature_stds;
    bool normalize = true;
    bool standardize = true;
};

// Market data structure
struct MarketData {
    std::string symbol;
    double bid;
    double ask;
    double last;
    double volume;
    double spread_bps;
    double vol_5m_bps;
    double impact_bps;
    int64_t timestamp_ms;
    int quote_age_ms;
    bool is_halted;
    bool luld_active;
    bool ssr_active;
};

// Performance metrics
struct PerformanceMetrics {
    double inference_time_ms;
    double feature_time_ms;
    double total_time_ms;
    size_t memory_usage_bytes;
    double cpu_usage_percent;
};

// Error handling
class TradingEngineError : public std::exception {
public:
    explicit TradingEngineError(const std::string& message) : message_(message) {}
    
    const char* what() const noexcept override {
        return message_.c_str();
    }

private:
    std::string message_;
};

// Utility functions
namespace utils {
    // Check if two floats are approximately equal
    bool float_equals(float a, float b, float epsilon = 1e-6f);
    
    // Clamp value between min and max
    template<typename T>
    T clamp(T value, T min_val, T max_val) {
        return std::max(min_val, std::min(value, max_val));
    }
    
    // Calculate correlation coefficient
    float correlation(const float* x, const float* y, size_t n);
    
    // Calculate rolling statistics
    void rolling_mean(const float* data, float* result, size_t n, size_t window);
    void rolling_std(const float* data, float* result, size_t n, size_t window);
    
    // Feature engineering utilities
    void calculate_returns(const float* prices, float* returns, size_t n);
    void calculate_volatility(const float* returns, float* volatility, size_t n, size_t window);
    void calculate_momentum(const float* prices, float* momentum, size_t n, size_t lookback);
}

} // namespace ibkr_trading
