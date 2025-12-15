#pragma once

#include "common.h"
#include <vector>
#include <memory>
#include <immintrin.h>  // SIMD intrinsics

namespace ibkr_trading {

/**
 * High-performance linear algebra engine for ensemble operations
 * Optimized with SIMD instructions for matrix operations
 */
class LinearAlgebraEngine {
public:
    LinearAlgebraEngine();
    ~LinearAlgebraEngine();

    /**
     * Matrix-vector multiplication with SIMD optimization
     * @param matrix Input matrix (row-major)
     * @param vector Input vector
     * @param result Output vector
     */
    void matrix_vector_multiply(const std::vector<std::vector<double>>& matrix,
                               const std::vector<double>& vector,
                               std::vector<double>& result);

    /**
     * SIMD-optimized dot product
     * @param a First vector
     * @param b Second vector
     * @return Dot product result
     */
    double dot_product_simd(const std::vector<double>& a, const std::vector<double>& b);

    /**
     * SIMD-optimized vector addition
     * @param a First vector
     * @param b Second vector
     * @param result Output vector
     */
    void vector_add_simd(const std::vector<double>& a, const std::vector<double>& b, std::vector<double>& result);

    /**
     * SIMD-optimized vector subtraction
     * @param a First vector
     * @param b Second vector
     * @param result Output vector
     */
    void vector_subtract_simd(const std::vector<double>& a, const std::vector<double>& b, std::vector<double>& result);

    /**
     * SIMD-optimized scalar multiplication
     * @param vector Input vector
     * @param scalar Scalar value
     * @param result Output vector
     */
    void scalar_multiply_simd(const std::vector<double>& vector, double scalar, std::vector<double>& result);

    /**
     * SIMD-optimized vector norm (L2)
     * @param vector Input vector
     * @return L2 norm
     */
    double vector_norm_simd(const std::vector<double>& vector);

    /**
     * SIMD-optimized correlation calculation
     * @param a First vector
     * @param b Second vector
     * @return Correlation coefficient
     */
    double correlation_simd(const std::vector<double>& a, const std::vector<double>& b);

    /**
     * SIMD-optimized covariance calculation
     * @param a First vector
     * @param b Second vector
     * @return Covariance
     */
    double covariance_simd(const std::vector<double>& a, const std::vector<double>& b);

    /**
     * SIMD-optimized matrix transpose
     * @param matrix Input matrix
     * @param result Output transposed matrix
     */
    void matrix_transpose_simd(const std::vector<std::vector<double>>& matrix,
                              std::vector<std::vector<double>>& result);

    /**
     * SIMD-optimized matrix multiplication
     * @param a First matrix
     * @param b Second matrix
     * @param result Output matrix
     */
    void matrix_multiply_simd(const std::vector<std::vector<double>>& a,
                             const std::vector<std::vector<double>>& b,
                             std::vector<std::vector<double>>& result);

    /**
     * SIMD-optimized eigenvalue calculation (simplified)
     * @param matrix Input matrix
     * @param eigenvalues Output eigenvalues
     * @param eigenvectors Output eigenvectors
     */
    void eigendecomposition_simd(const std::vector<std::vector<double>>& matrix,
                                std::vector<double>& eigenvalues,
                                std::vector<std::vector<double>>& eigenvectors);

    /**
     * SIMD-optimized Cholesky decomposition
     * @param matrix Input symmetric positive definite matrix
     * @param result Output lower triangular matrix
     */
    void cholesky_decomposition_simd(const std::vector<std::vector<double>>& matrix,
                                   std::vector<std::vector<double>>& result);

    /**
     * SIMD-optimized linear system solver (Ax = b)
     * @param A Coefficient matrix
     * @param b Right-hand side vector
     * @param x Solution vector
     */
    void solve_linear_system_simd(const std::vector<std::vector<double>>& A,
                                 const std::vector<double>& b,
                                 std::vector<double>& x);

    /**
     * Get engine statistics
     * @return Performance statistics
     */
    struct EngineStats {
        uint64_t operations_performed = 0;
        double avg_operation_time_us = 0.0;
        double max_operation_time_us = 0.0;
        uint64_t simd_operations = 0;
        uint64_t fallback_operations = 0;
    };
    
    EngineStats get_stats() const;

    /**
     * Reset engine statistics
     */
    void reset_stats();

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
    
    // SIMD helper functions
    void simd_vector_add(const double* a, const double* b, double* result, size_t size);
    void simd_vector_subtract(const double* a, const double* b, double* result, size_t size);
    void simd_scalar_multiply(const double* vector, double scalar, double* result, size_t size);
    double simd_dot_product(const double* a, const double* b, size_t size);
    double simd_vector_norm(const double* vector, size_t size);
    
    // Matrix operations helpers
    void simd_matrix_vector_multiply(const double* matrix, const double* vector, double* result, size_t rows, size_t cols);
    void simd_matrix_transpose(const double* matrix, double* result, size_t rows, size_t cols);
    void simd_matrix_multiply(const double* a, const double* b, double* result, size_t a_rows, size_t a_cols, size_t b_cols);
};

} // namespace ibkr_trading
