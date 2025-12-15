#pragma once

#include "common.h"
#include <unordered_map>
#include <memory>
#include <thread>
#include <future>

namespace ibkr_trading {

// Forward declarations
class Model;
class MemoryPool;
class SIMDCalculator;

// High-performance model inference engine
class InferenceEngine {
public:
    explicit InferenceEngine(const std::string& model_dir);
    ~InferenceEngine();
    
    // Load models for all horizons and families
    void load_models(const std::vector<std::string>& horizons,
                    const std::vector<std::string>& families);
    
    // Batch inference for multiple symbols
    std::unordered_map<std::string, std::unordered_map<std::string, float>>
    predict_batch(const std::unordered_map<std::string, FeatureVector>& features);
    
    // Single symbol inference
    std::unordered_map<std::string, float>
    predict_single(const std::string& symbol, const FeatureVector& features);
    
    // Get performance metrics
    PerformanceMetrics get_metrics() const;
    
    // Warmup models (pre-compile, pre-allocate)
    void warmup();
    
    // Clear model cache
    void clear_cache();

private:
    std::string model_dir_;
    std::unordered_map<std::string, std::unique_ptr<Model>> models_;
    std::unique_ptr<MemoryPool> memory_pool_;
    std::unique_ptr<SIMDCalculator> simd_calc_;
    
    // Performance tracking
    mutable PerformanceMetrics metrics_;
    mutable std::mutex metrics_mutex_;
    
    // Model loading
    void load_model(const std::string& model_name, const std::string& model_path);
    std::string get_model_path(const std::string& family, const std::string& horizon);
    
    // Inference helpers
    std::vector<float> preprocess_features(const FeatureVector& features);
    std::vector<float> postprocess_predictions(const std::vector<float>& raw_predictions);
    
    // SIMD-optimized inference
    void inference_simd(const float* features, float* predictions, 
                        const std::string& model_name);
    
    // Parallel inference
    void inference_parallel(const std::unordered_map<std::string, FeatureVector>& features,
                           std::unordered_map<std::string, std::unordered_map<std::string, float>>& results);
};

// Individual model wrapper
class Model {
public:
    Model(const std::string& model_path, const ModelConfig& config);
    ~Model();
    
    // Single prediction
    std::vector<float> predict(const std::vector<float>& features);
    
    // Batch prediction
    std::vector<std::vector<float>> predict_batch(const std::vector<std::vector<float>>& features);
    
    // Get model info
    size_t input_size() const { return input_size_; }
    size_t output_size() const { return output_size_; }
    const std::string& model_type() const { return model_type_; }
    
    // Performance optimization
    void enable_simd(bool enable) { use_simd_ = enable; }
    void enable_gpu(bool enable) { use_gpu_ = enable; }

private:
    std::string model_path_;
    std::string model_type_;
    size_t input_size_;
    size_t output_size_;
    bool use_simd_;
    bool use_gpu_;
    
    // Model-specific data
    std::vector<float> weights_;
    std::vector<float> biases_;
    std::vector<std::vector<float>> layers_;
    
    // SIMD-optimized operations
    void predict_linear_simd(const float* input, float* output);
    void predict_mlp_simd(const float* input, float* output);
    void predict_tree_simd(const float* input, float* output);
    
    // GPU operations (if available)
    void predict_gpu(const float* input, float* output);
    
    // Model loading
    void load_lightgbm(const std::string& path);
    void load_xgboost(const std::string& path);
    void load_mlp(const std::string& path);
    void load_tabcnn(const std::string& path);
    void load_tablstm(const std::string& path);
    void load_tabtransformer(const std::string& path);
};

// Memory pool for efficient allocation
class MemoryPool {
public:
    explicit MemoryPool(size_t pool_size = 1024 * 1024 * 1024);  // 1GB default
    ~MemoryPool();
    
    // Allocate aligned memory
    float* allocate(size_t count);
    
    // Deallocate memory
    void deallocate(float* ptr);
    
    // Get pool statistics
    size_t total_allocated() const { return total_allocated_; }
    size_t peak_usage() const { return peak_usage_; }
    
    // Clear pool
    void clear();

private:
    size_t pool_size_;
    size_t total_allocated_;
    size_t peak_usage_;
    std::vector<std::unique_ptr<float[]>> blocks_;
    std::vector<size_t> free_blocks_;
    std::mutex pool_mutex_;
    
    // Block management
    std::unique_ptr<float[]> allocate_block(size_t size);
    void deallocate_block(std::unique_ptr<float[]> block);
};

// SIMD calculator for vectorized operations
class SIMDCalculator {
public:
    SIMDCalculator();
    ~SIMDCalculator();
    
    // Vectorized operations
    void dot_product(const float* a, const float* b, float* result, size_t n);
    void matrix_vector_mult(const float* matrix, const float* vector, 
                           float* result, size_t rows, size_t cols);
    void softmax(const float* input, float* output, size_t n);
    void normalize(const float* input, float* output, size_t n);
    void relu(const float* input, float* output, size_t n);
    void sigmoid(const float* input, float* output, size_t n);
    
    // Batch operations
    void batch_dot_product(const float* a, const float* b, float* result, 
                          size_t batch_size, size_t vector_size);
    void batch_matrix_vector_mult(const float* matrix, const float* vectors,
                                 float* result, size_t batch_size, size_t rows, size_t cols);
    
    // Performance info
    bool has_avx2() const { return has_avx2_; }
    bool has_avx512() const { return has_avx512_; }

private:
    bool has_avx2_;
    bool has_avx512_;
    
    // AVX2 implementations
    void dot_product_avx2(const float* a, const float* b, float* result, size_t n);
    void matrix_vector_mult_avx2(const float* matrix, const float* vector,
                                 float* result, size_t rows, size_t cols);
    void softmax_avx2(const float* input, float* output, size_t n);
    void normalize_avx2(const float* input, float* output, size_t n);
    
    // AVX512 implementations
    void dot_product_avx512(const float* a, const float* b, float* result, size_t n);
    void matrix_vector_mult_avx512(const float* matrix, const float* vector,
                                   float* result, size_t rows, size_t cols);
    void softmax_avx512(const float* input, float* output, size_t n);
    void normalize_avx512(const float* input, float* output, size_t n);
    
    // Fallback implementations
    void dot_product_fallback(const float* a, const float* b, float* result, size_t n);
    void matrix_vector_mult_fallback(const float* matrix, const float* vector,
                                    float* result, size_t rows, size_t cols);
    void softmax_fallback(const float* input, float* output, size_t n);
    void normalize_fallback(const float* input, float* output, size_t n);
};

} // namespace ibkr_trading
