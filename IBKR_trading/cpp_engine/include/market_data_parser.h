#pragma once

#include "common.h"
#include <string>
#include <vector>
#include <memory>
#include <chrono>

namespace ibkr_trading {

/**
 * High-performance market data parser with zero-copy operations
 * Optimized for high-frequency trading data processing
 */
class MarketDataParser {
public:
    MarketDataParser();
    ~MarketDataParser();

    /**
     * Parse market data from raw buffer
     * @param data Raw market data buffer
     * @param size Buffer size
     * @return Parsed market data structure
     */
    MarketData parse_market_data(const char* data, size_t size);

    /**
     * Parse multiple market data messages from buffer
     * @param data Raw buffer containing multiple messages
     * @param size Buffer size
     * @return Vector of parsed market data
     */
    std::vector<MarketData> parse_batch(const char* data, size_t size);

    /**
     * Parse market data with SIMD optimization
     * @param data Raw market data buffer
     * @param size Buffer size
     * @return Parsed market data structure
     */
    MarketData parse_market_data_simd(const char* data, size_t size);

    /**
     * Validate market data integrity
     * @param data Market data to validate
     * @return True if valid, false otherwise
     */
    bool validate_market_data(const MarketData& data);

    /**
     * Get parser statistics
     * @return Parser performance statistics
     */
    struct ParserStats {
        uint64_t messages_parsed = 0;
        uint64_t bytes_processed = 0;
        double avg_parse_time_us = 0.0;
        double max_parse_time_us = 0.0;
        uint64_t parse_errors = 0;
    };
    
    ParserStats get_stats() const;

    /**
     * Reset parser statistics
     */
    void reset_stats();

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
    
    // SIMD-optimized parsing functions
    MarketData parse_level1_data_simd(const char* data, size_t size);
    MarketData parse_level2_data_simd(const char* data, size_t size);
    MarketData parse_trade_data_simd(const char* data, size_t size);
    
    // Validation functions
    bool validate_timestamp(const std::chrono::system_clock::time_point& ts);
    bool validate_price(double price);
    bool validate_size(uint64_t size);
    bool validate_symbol(const std::string& symbol);
};

} // namespace ibkr_trading
