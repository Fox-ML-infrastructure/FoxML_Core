#include "inference_engine.h"
#include <chrono>
#include <random>
#include <iostream>
#include <iomanip>

using namespace ibkr_trading;

class BenchmarkRunner {
public:
    BenchmarkRunner() : rng_(std::random_device{}()) {}
    
    void run_inference_benchmark() {
        std::cout << "🚀 IBKR Trading Engine - Inference Benchmark\n";
        std::cout << "==========================================\n\n";
        
        // Test parameters
        const size_t num_symbols = 100;
        const size_t num_features = 281;
        const size_t num_models = 16;
        const size_t num_horizons = 5;
        
        // Create test data
        auto features = generate_test_features(num_symbols, num_features);
        
        // Initialize inference engine
        InferenceEngine engine("models/");
        
        // Load models (mock)
        std::vector<std::string> horizons = {"5m", "10m", "15m", "30m", "60m"};
        std::vector<std::string> families = {
            "LightGBM", "XGBoost", "MLP", "TabCNN", "TabLSTM", "TabTransformer",
            "RewardBased", "QuantileLightGBM", "NGBoost", "GMMRegime",
            "ChangePoint", "FTRLProximal", "VAE", "GAN", "Ensemble", "MetaLearning"
        };
        
        std::cout << "📊 Benchmark Configuration:\n";
        std::cout << "  Symbols: " << num_symbols << "\n";
        std::cout << "  Features per symbol: " << num_features << "\n";
        std::cout << "  Models: " << num_models << "\n";
        std::cout << "  Horizons: " << num_horizons << "\n";
        std::cout << "  Total predictions: " << (num_symbols * num_models * num_horizons) << "\n\n";
        
        // Warmup
        std::cout << "🔥 Warming up models...\n";
        engine.warmup();
        
        // Benchmark single symbol inference
        benchmark_single_symbol(engine, features, num_features);
        
        // Benchmark batch inference
        benchmark_batch_inference(engine, features, num_symbols);
        
        // Benchmark memory usage
        benchmark_memory_usage(engine, features, num_symbols);
        
        // Benchmark SIMD performance
        benchmark_simd_performance();
        
        std::cout << "\n✅ Benchmark completed!\n";
    }
    
private:
    std::mt19937 rng_;
    
    std::unordered_map<std::string, FeatureVector> generate_test_features(size_t num_symbols, size_t num_features) {
        std::unordered_map<std::string, FeatureVector> features;
        std::normal_distribution<float> dist(0.0f, 1.0f);
        
        for (size_t i = 0; i < num_symbols; ++i) {
            std::string symbol = "SYMBOL_" + std::to_string(i);
            FeatureVector symbol_features(num_features);
            
            for (size_t j = 0; j < num_features; ++j) {
                symbol_features[j] = dist(rng_);
            }
            
            features[symbol] = std::move(symbol_features);
        }
        
        return features;
    }
    
    void benchmark_single_symbol(InferenceEngine& engine, const std::unordered_map<std::string, FeatureVector>& features, size_t num_features) {
        std::cout << "🔍 Single Symbol Inference Benchmark:\n";
        std::cout << "------------------------------------\n";
        
        const int num_iterations = 1000;
        std::vector<double> times;
        times.reserve(num_iterations);
        
        // Get first symbol for testing
        auto it = features.begin();
        const std::string& symbol = it->first;
        const FeatureVector& symbol_features = it->second;
        
        // Warmup
        for (int i = 0; i < 10; ++i) {
            engine.predict_single(symbol, symbol_features);
        }
        
        // Benchmark
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < num_iterations; ++i) {
            auto iter_start = std::chrono::high_resolution_clock::now();
            engine.predict_single(symbol, symbol_features);
            auto iter_end = std::chrono::high_resolution_clock::now();
            
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(iter_end - iter_start);
            times.push_back(duration.count() / 1000.0);  // Convert to milliseconds
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        // Calculate statistics
        std::sort(times.begin(), times.end());
        double mean = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
        double p50 = times[times.size() / 2];
        double p95 = times[static_cast<size_t>(times.size() * 0.95)];
        double p99 = times[static_cast<size_t>(times.size() * 0.99)];
        double min_time = *std::min_element(times.begin(), times.end());
        double max_time = *std::max_element(times.begin(), times.end());
        
        std::cout << "  Iterations: " << num_iterations << "\n";
        std::cout << "  Total time: " << total_duration.count() << " ms\n";
        std::cout << "  Mean time: " << std::fixed << std::setprecision(3) << mean << " ms\n";
        std::cout << "  Median (p50): " << p50 << " ms\n";
        std::cout << "  p95: " << p95 << " ms\n";
        std::cout << "  p99: " << p99 << " ms\n";
        std::cout << "  Min: " << min_time << " ms\n";
        std::cout << "  Max: " << max_time << " ms\n";
        std::cout << "  Throughput: " << std::fixed << std::setprecision(1) << (1000.0 / mean) << " predictions/sec\n\n";
    }
    
    void benchmark_batch_inference(InferenceEngine& engine, const std::unordered_map<std::string, FeatureVector>& features, size_t num_symbols) {
        std::cout << "📦 Batch Inference Benchmark:\n";
        std::cout << "------------------------------\n";
        
        const int num_iterations = 100;
        std::vector<double> times;
        times.reserve(num_iterations);
        
        // Warmup
        for (int i = 0; i < 5; ++i) {
            engine.predict_batch(features);
        }
        
        // Benchmark
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < num_iterations; ++i) {
            auto iter_start = std::chrono::high_resolution_clock::now();
            engine.predict_batch(features);
            auto iter_end = std::chrono::high_resolution_clock::now();
            
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(iter_end - iter_start);
            times.push_back(duration.count() / 1000.0);  // Convert to milliseconds
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        // Calculate statistics
        std::sort(times.begin(), times.end());
        double mean = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
        double p50 = times[times.size() / 2];
        double p95 = times[static_cast<size_t>(times.size() * 0.95)];
        double p99 = times[static_cast<size_t>(times.size() * 0.99)];
        
        std::cout << "  Iterations: " << num_iterations << "\n";
        std::cout << "  Symbols per batch: " << num_symbols << "\n";
        std::cout << "  Total time: " << total_duration.count() << " ms\n";
        std::cout << "  Mean time: " << std::fixed << std::setprecision(3) << mean << " ms\n";
        std::cout << "  Median (p50): " << p50 << " ms\n";
        std::cout << "  p95: " << p95 << " ms\n";
        std::cout << "  p99: " << p99 << " ms\n";
        std::cout << "  Throughput: " << std::fixed << std::setprecision(1) << (num_symbols * 1000.0 / mean) << " symbols/sec\n\n";
    }
    
    void benchmark_memory_usage(InferenceEngine& engine, const std::unordered_map<std::string, FeatureVector>& features, size_t num_symbols) {
        std::cout << "💾 Memory Usage Benchmark:\n";
        std::cout << "--------------------------\n";
        
        // Get initial memory usage
        auto initial_metrics = engine.get_metrics();
        
        // Run inference
        engine.predict_batch(features);
        
        // Get final memory usage
        auto final_metrics = engine.get_metrics();
        
        std::cout << "  Initial memory: " << (initial_metrics.memory_usage_bytes / 1024.0 / 1024.0) << " MB\n";
        std::cout << "  Final memory: " << (final_metrics.memory_usage_bytes / 1024.0 / 1024.0) << " MB\n";
        std::cout << "  Memory per symbol: " << ((final_metrics.memory_usage_bytes - initial_metrics.memory_usage_bytes) / num_symbols / 1024.0) << " KB\n\n";
    }
    
    void benchmark_simd_performance() {
        std::cout << "⚡ SIMD Performance Benchmark:\n";
        std::cout << "------------------------------\n";
        
        const size_t vector_size = 1000;
        const int num_iterations = 10000;
        
        // Generate test data
        std::vector<float> a(vector_size), b(vector_size), result(vector_size);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        
        for (size_t i = 0; i < vector_size; ++i) {
            a[i] = dist(rng_);
            b[i] = dist(rng_);
        }
        
        // Benchmark dot product
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < num_iterations; ++i) {
            float sum = 0.0f;
            for (size_t j = 0; j < vector_size; ++j) {
                sum += a[j] * b[j];
            }
            result[0] = sum;
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        double time_per_iteration = duration.count() / static_cast<double>(num_iterations);
        double operations_per_second = (vector_size * num_iterations) / (duration.count() / 1e6);
        
        std::cout << "  Vector size: " << vector_size << "\n";
        std::cout << "  Iterations: " << num_iterations << "\n";
        std::cout << "  Time per iteration: " << std::fixed << std::setprecision(3) << time_per_iteration << " μs\n";
        std::cout << "  Operations per second: " << std::fixed << std::setprecision(0) << operations_per_second << "\n";
        std::cout << "  AVX2 support: " << (simd::has_avx2() ? "Yes" : "No") << "\n";
        std::cout << "  AVX512 support: " << (simd::has_avx512() ? "Yes" : "No") << "\n\n";
    }
};

int main() {
    try {
        BenchmarkRunner runner;
        runner.run_inference_benchmark();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "❌ Benchmark failed: " << e.what() << std::endl;
        return 1;
    }
}
