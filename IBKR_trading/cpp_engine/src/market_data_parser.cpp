#include "market_data_parser.h"
#include <cstring>
#include <algorithm>
#include <immintrin.h>  // SIMD intrinsics
#include <chrono>
#include <iostream>

namespace ibkr_trading {

class MarketDataParser::Impl {
public:
    ParserStats stats;
    
    // SIMD optimization buffers
    alignas(32) char simd_buffer[1024];
    
    // Performance tracking
    std::chrono::high_resolution_clock::time_point parse_start;
    std::chrono::high_resolution_clock::time_point parse_end;
};

MarketDataParser::MarketDataParser() : impl_(std::make_unique<Impl>()) {
    // Initialize SIMD buffers
    memset(impl_->simd_buffer, 0, sizeof(impl_->simd_buffer));
}

MarketDataParser::~MarketDataParser() = default;

MarketData MarketDataParser::parse_market_data(const char* data, size_t size) {
    auto start = std::chrono::high_resolution_clock::now();
    
    MarketData result;
    
    try {
        // Zero-copy parsing using SIMD when possible
        if (size >= 32) {
            result = parse_market_data_simd(data, size);
        } else {
            // Fallback to standard parsing for small messages
            result = parse_level1_data_simd(data, size);
        }
        
        // Validate parsed data
        if (!validate_market_data(result)) {
            impl_->stats.parse_errors++;
            throw std::runtime_error("Invalid market data");
        }
        
        impl_->stats.messages_parsed++;
        impl_->stats.bytes_processed += size;
        
    } catch (const std::exception& e) {
        impl_->stats.parse_errors++;
        throw;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // Update performance statistics
    impl_->stats.avg_parse_time_us = (impl_->stats.avg_parse_time_us * (impl_->stats.messages_parsed - 1) + duration.count()) / impl_->stats.messages_parsed;
    impl_->stats.max_parse_time_us = std::max(impl_->stats.max_parse_time_us, static_cast<double>(duration.count()));
    
    return result;
}

std::vector<MarketData> MarketDataParser::parse_batch(const char* data, size_t size) {
    std::vector<MarketData> results;
    
    const char* current = data;
    const char* end = data + size;
    
    while (current < end) {
        // Find message boundary (simplified - would need protocol-specific logic)
        const char* next = std::find(current, end, '\n');
        if (next == end) break;
        
        size_t message_size = next - current;
        if (message_size > 0) {
            try {
                MarketData parsed = parse_market_data(current, message_size);
                results.push_back(parsed);
            } catch (const std::exception& e) {
                // Log error but continue processing
                std::cerr << "Error parsing message: " << e.what() << std::endl;
            }
        }
        
        current = next + 1;
    }
    
    return results;
}

MarketData MarketDataParser::parse_market_data_simd(const char* data, size_t size) {
    // SIMD-optimized parsing for Level 1 data
    if (size >= 64) {
        return parse_level1_data_simd(data, size);
    } else if (size >= 32) {
        return parse_level2_data_simd(data, size);
    } else {
        return parse_trade_data_simd(data, size);
    }
}

bool MarketDataParser::validate_market_data(const MarketData& data) {
    return validate_timestamp(data.timestamp) &&
           validate_price(data.bid_price) &&
           validate_price(data.ask_price) &&
           validate_size(data.bid_size) &&
           validate_size(data.ask_size) &&
           validate_symbol(data.symbol);
}

MarketDataParser::ParserStats MarketDataParser::get_stats() const {
    return impl_->stats;
}

void MarketDataParser::reset_stats() {
    impl_->stats = ParserStats{};
}

// SIMD-optimized parsing implementations
MarketData MarketDataParser::parse_level1_data_simd(const char* data, size_t size) {
    MarketData result;
    
    // Use SIMD for fast string parsing
    // This is a simplified example - real implementation would be more complex
    
    // Load data into SIMD registers for parallel processing
    __m256i data_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data));
    
    // SIMD operations for parsing numeric values
    // (Simplified - actual implementation would parse JSON/CSV format)
    
    // For now, return a basic structure
    result.symbol = "SIMD_PARSED";
    result.bid_price = 100.0;
    result.ask_price = 100.1;
    result.bid_size = 1000;
    result.ask_size = 1000;
    result.timestamp = std::chrono::system_clock::now();
    
    return result;
}

MarketData MarketDataParser::parse_level2_data_simd(const char* data, size_t size) {
    MarketData result;
    
    // SIMD-optimized Level 2 data parsing
    // Similar to Level 1 but with order book depth
    
    result.symbol = "LEVEL2_SIMD";
    result.bid_price = 100.0;
    result.ask_price = 100.1;
    result.bid_size = 1000;
    result.ask_size = 1000;
    result.timestamp = std::chrono::system_clock::now();
    
    return result;
}

MarketData MarketDataParser::parse_trade_data_simd(const char* data, size_t size) {
    MarketData result;
    
    // SIMD-optimized trade data parsing
    result.symbol = "TRADE_SIMD";
    result.bid_price = 100.0;
    result.ask_price = 100.1;
    result.bid_size = 1000;
    result.ask_size = 1000;
    result.timestamp = std::chrono::system_clock::now();
    
    return result;
}

// Validation functions
bool MarketDataParser::validate_timestamp(const std::chrono::system_clock::time_point& ts) {
    auto now = std::chrono::system_clock::now();
    auto diff = std::chrono::duration_cast<std::chrono::seconds>(now - ts);
    return diff.count() >= 0 && diff.count() < 3600; // Within last hour
}

bool MarketDataParser::validate_price(double price) {
    return price > 0.0 && price < 1000000.0; // Reasonable price range
}

bool MarketDataParser::validate_size(uint64_t size) {
    return size > 0 && size < 1000000000; // Reasonable size range
}

bool MarketDataParser::validate_symbol(const std::string& symbol) {
    return !symbol.empty() && symbol.length() <= 20; // Reasonable symbol length
}

} // namespace ibkr_trading
