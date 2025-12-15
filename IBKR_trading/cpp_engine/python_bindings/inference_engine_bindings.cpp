#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "inference_engine.h"

namespace py = pybind11;

PYBIND11_MODULE(ibkr_trading_engine_py, m) {
    m.doc() = "IBKR Trading Engine Python Bindings";
    
    // Common types
    py::class_<ibkr_trading::MarketData>(m, "MarketData")
        .def(py::init<>())
        .def_readwrite("symbol", &ibkr_trading::MarketData::symbol)
        .def_readwrite("bid", &ibkr_trading::MarketData::bid)
        .def_readwrite("ask", &ibkr_trading::MarketData::ask)
        .def_readwrite("last", &ibkr_trading::MarketData::last)
        .def_readwrite("volume", &ibkr_trading::MarketData::volume)
        .def_readwrite("spread_bps", &ibkr_trading::MarketData::spread_bps)
        .def_readwrite("vol_5m_bps", &ibkr_trading::MarketData::vol_5m_bps)
        .def_readwrite("impact_bps", &ibkr_trading::MarketData::impact_bps)
        .def_readwrite("timestamp_ms", &ibkr_trading::MarketData::timestamp_ms)
        .def_readwrite("quote_age_ms", &ibkr_trading::MarketData::quote_age_ms)
        .def_readwrite("is_halted", &ibkr_trading::MarketData::is_halted)
        .def_readwrite("luld_active", &ibkr_trading::MarketData::luld_active)
        .def_readwrite("ssr_active", &ibkr_trading::MarketData::ssr_active);
    
    py::class_<ibkr_trading::PerformanceMetrics>(m, "PerformanceMetrics")
        .def(py::init<>())
        .def_readwrite("inference_time_ms", &ibkr_trading::PerformanceMetrics::inference_time_ms)
        .def_readwrite("feature_time_ms", &ibkr_trading::PerformanceMetrics::feature_time_ms)
        .def_readwrite("total_time_ms", &ibkr_trading::PerformanceMetrics::total_time_ms)
        .def_readwrite("memory_usage_bytes", &ibkr_trading::PerformanceMetrics::memory_usage_bytes)
        .def_readwrite("cpu_usage_percent", &ibkr_trading::PerformanceMetrics::cpu_usage_percent);
    
    py::class_<ibkr_trading::ModelConfig>(m, "ModelConfig")
        .def(py::init<>())
        .def_readwrite("model_path", &ibkr_trading::ModelConfig::model_path)
        .def_readwrite("model_type", &ibkr_trading::ModelConfig::model_type)
        .def_readwrite("input_features", &ibkr_trading::ModelConfig::input_features)
        .def_readwrite("output_classes", &ibkr_trading::ModelConfig::output_classes)
        .def_readwrite("use_simd", &ibkr_trading::ModelConfig::use_simd)
        .def_readwrite("use_gpu", &ibkr_trading::ModelConfig::use_gpu);
    
    py::class_<ibkr_trading::FeatureConfig>(m, "FeatureConfig")
        .def(py::init<>())
        .def_readwrite("feature_names", &ibkr_trading::FeatureConfig::feature_names)
        .def_readwrite("feature_scales", &ibkr_trading::FeatureConfig::feature_scales)
        .def_readwrite("feature_means", &ibkr_trading::FeatureConfig::feature_means)
        .def_readwrite("feature_stds", &ibkr_trading::FeatureConfig::feature_stds)
        .def_readwrite("normalize", &ibkr_trading::FeatureConfig::normalize)
        .def_readwrite("standardize", &ibkr_trading::FeatureConfig::standardize);
    
    // InferenceEngine class
    py::class_<ibkr_trading::InferenceEngine>(m, "InferenceEngine")
        .def(py::init<const std::string&>())
        .def("load_models", &ibkr_trading::InferenceEngine::load_models,
             "Load models for all horizons and families",
             py::arg("horizons"), py::arg("families"))
        .def("predict_batch", &ibkr_trading::InferenceEngine::predict_batch,
             "Batch inference for multiple symbols",
             py::arg("features"))
        .def("predict_single", &ibkr_trading::InferenceEngine::predict_single,
             "Single symbol inference",
             py::arg("symbol"), py::arg("features"))
        .def("get_metrics", &ibkr_trading::InferenceEngine::get_metrics,
             "Get performance metrics")
        .def("warmup", &ibkr_trading::InferenceEngine::warmup,
             "Warmup models")
        .def("clear_cache", &ibkr_trading::InferenceEngine::clear_cache,
             "Clear model cache");
    
    // Model class
    py::class_<ibkr_trading::Model>(m, "Model")
        .def(py::init<const std::string&, const ibkr_trading::ModelConfig&>())
        .def("predict", &ibkr_trading::Model::predict,
             "Single prediction",
             py::arg("features"))
        .def("predict_batch", &ibkr_trading::Model::predict_batch,
             "Batch prediction",
             py::arg("features"))
        .def("input_size", &ibkr_trading::Model::input_size)
        .def("output_size", &ibkr_trading::Model::output_size)
        .def("model_type", &ibkr_trading::Model::model_type)
        .def("enable_simd", &ibkr_trading::Model::enable_simd,
             "Enable SIMD optimization",
             py::arg("enable"))
        .def("enable_gpu", &ibkr_trading::Model::enable_gpu,
             "Enable GPU acceleration",
             py::arg("enable"));
    
    // MemoryPool class
    py::class_<ibkr_trading::MemoryPool>(m, "MemoryPool")
        .def(py::init<size_t>(), py::arg("pool_size") = 1024 * 1024 * 1024)
        .def("allocate", &ibkr_trading::MemoryPool::allocate,
             "Allocate aligned memory",
             py::arg("count"))
        .def("deallocate", &ibkr_trading::MemoryPool::deallocate,
             "Deallocate memory",
             py::arg("ptr"))
        .def("total_allocated", &ibkr_trading::MemoryPool::total_allocated)
        .def("peak_usage", &ibkr_trading::MemoryPool::peak_usage)
        .def("clear", &ibkr_trading::MemoryPool::clear);
    
    // SIMDCalculator class
    py::class_<ibkr_trading::SIMDCalculator>(m, "SIMDCalculator")
        .def(py::init<>())
        .def("dot_product", &ibkr_trading::SIMDCalculator::dot_product,
             "Vectorized dot product",
             py::arg("a"), py::arg("b"), py::arg("result"), py::arg("n"))
        .def("matrix_vector_mult", &ibkr_trading::SIMDCalculator::matrix_vector_mult,
             "Matrix-vector multiplication",
             py::arg("matrix"), py::arg("vector"), py::arg("result"), py::arg("rows"), py::arg("cols"))
        .def("softmax", &ibkr_trading::SIMDCalculator::softmax,
             "Softmax activation",
             py::arg("input"), py::arg("output"), py::arg("n"))
        .def("normalize", &ibkr_trading::SIMDCalculator::normalize,
             "Normalize data",
             py::arg("input"), py::arg("output"), py::arg("n"))
        .def("relu", &ibkr_trading::SIMDCalculator::relu,
             "ReLU activation",
             py::arg("input"), py::arg("output"), py::arg("n"))
        .def("sigmoid", &ibkr_trading::SIMDCalculator::sigmoid,
             "Sigmoid activation",
             py::arg("input"), py::arg("output"), py::arg("n"))
        .def("batch_dot_product", &ibkr_trading::SIMDCalculator::batch_dot_product,
             "Batch dot product",
             py::arg("a"), py::arg("b"), py::arg("result"), py::arg("batch_size"), py::arg("vector_size"))
        .def("batch_matrix_vector_mult", &ibkr_trading::SIMDCalculator::batch_matrix_vector_mult,
             "Batch matrix-vector multiplication",
             py::arg("matrix"), py::arg("vectors"), py::arg("result"), py::arg("batch_size"), py::arg("rows"), py::arg("cols"))
        .def("has_avx2", &ibkr_trading::SIMDCalculator::has_avx2)
        .def("has_avx512", &ibkr_trading::SIMDCalculator::has_avx512);
    
    // Utility functions
    m.def("check_avx2_support", []() {
        return ibkr_trading::simd::has_avx2();
    }, "Check if CPU supports AVX2");
    
    m.def("check_avx512_support", []() {
        return ibkr_trading::simd::has_avx512();
    }, "Check if CPU supports AVX512");
    
    m.def("get_cpu_info", []() {
        return py::dict(
            py::arg("avx2") = ibkr_trading::simd::has_avx2(),
            py::arg("avx512") = ibkr_trading::simd::has_avx512()
        );
    }, "Get CPU information");
    
    // Performance utilities
    py::class_<ibkr_trading::Timer>(m, "Timer")
        .def(py::init<>())
        .def("elapsed_ms", &ibkr_trading::Timer::elapsed_ms)
        .def("reset", &ibkr_trading::Timer::reset);
    
    // Error handling
    py::register_exception<ibkr_trading::TradingEngineError>(m, "TradingEngineError");
    
    // Utility functions
    m.def("float_equals", &ibkr_trading::utils::float_equals,
          "Check if two floats are approximately equal",
          py::arg("a"), py::arg("b"), py::arg("epsilon") = 1e-6f);
    
    m.def("clamp", &ibkr_trading::utils::clamp<float>,
          "Clamp value between min and max",
          py::arg("value"), py::arg("min_val"), py::arg("max_val"));
    
    m.def("correlation", &ibkr_trading::utils::correlation,
          "Calculate correlation coefficient",
          py::arg("x"), py::arg("y"), py::arg("n"));
    
    m.def("rolling_mean", &ibkr_trading::utils::rolling_mean,
          "Calculate rolling mean",
          py::arg("data"), py::arg("result"), py::arg("n"), py::arg("window"));
    
    m.def("rolling_std", &ibkr_trading::utils::rolling_std,
          "Calculate rolling standard deviation",
          py::arg("data"), py::arg("result"), py::arg("n"), py::arg("window"));
    
    m.def("calculate_returns", &ibkr_trading::utils::calculate_returns,
          "Calculate returns",
          py::arg("prices"), py::arg("returns"), py::arg("n"));
    
    m.def("calculate_volatility", &ibkr_trading::utils::calculate_volatility,
          "Calculate volatility",
          py::arg("returns"), py::arg("volatility"), py::arg("n"), py::arg("window"));
    
    m.def("calculate_momentum", &ibkr_trading::utils::calculate_momentum,
          "Calculate momentum",
          py::arg("prices"), py::arg("momentum"), py::arg("n"), py::arg("lookback"));
}
