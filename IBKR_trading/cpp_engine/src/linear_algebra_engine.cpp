#include "linear_algebra_engine.h"
#include <chrono>
#include <algorithm>
#include <cmath>

namespace ibkr_trading {

class LinearAlgebraEngine::Impl {
public:
    EngineStats stats;
    
    // SIMD optimization buffers
    alignas(32) double simd_buffer_a[1024];
    alignas(32) double simd_buffer_b[1024];
    alignas(32) double simd_buffer_result[1024];
    
    // Performance tracking
    std::chrono::high_resolution_clock::time_point operation_start;
    std::chrono::high_resolution_clock::time_point operation_end;
};

LinearAlgebraEngine::LinearAlgebraEngine() : impl_(std::make_unique<Impl>()) {
    // Initialize SIMD buffers
    memset(impl_->simd_buffer_a, 0, sizeof(impl_->simd_buffer_a));
    memset(impl_->simd_buffer_b, 0, sizeof(impl_->simd_buffer_b));
    memset(impl_->simd_buffer_result, 0, sizeof(impl_->simd_buffer_result));
}

LinearAlgebraEngine::~LinearAlgebraEngine() = default;

void LinearAlgebraEngine::matrix_vector_multiply(const std::vector<std::vector<double>>& matrix,
                                                const std::vector<double>& vector,
                                                std::vector<double>& result) {
    auto start = std::chrono::high_resolution_clock::now();
    
    size_t rows = matrix.size();
    size_t cols = matrix[0].size();
    
    result.resize(rows);
    
    // Use SIMD for large matrices
    if (cols >= 4) {
        for (size_t i = 0; i < rows; ++i) {
            result[i] = dot_product_simd(matrix[i], vector);
        }
        impl_->stats.simd_operations++;
    } else {
        // Fallback to standard implementation
        for (size_t i = 0; i < rows; ++i) {
            result[i] = 0.0;
            for (size_t j = 0; j < cols; ++j) {
                result[i] += matrix[i][j] * vector[j];
            }
        }
        impl_->stats.fallback_operations++;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    impl_->stats.operations_performed++;
    impl_->stats.avg_operation_time_us = (impl_->stats.avg_operation_time_us * (impl_->stats.operations_performed - 1) + duration.count()) / impl_->stats.operations_performed;
    impl_->stats.max_operation_time_us = std::max(impl_->stats.max_operation_time_us, static_cast<double>(duration.count()));
}

double LinearAlgebraEngine::dot_product_simd(const std::vector<double>& a, const std::vector<double>& b) {
    size_t size = std::min(a.size(), b.size());
    
    if (size >= 4) {
        return simd_dot_product(a.data(), b.data(), size);
    } else {
        // Fallback for small vectors
        double result = 0.0;
        for (size_t i = 0; i < size; ++i) {
            result += a[i] * b[i];
        }
        return result;
    }
}

void LinearAlgebraEngine::vector_add_simd(const std::vector<double>& a, const std::vector<double>& b, std::vector<double>& result) {
    size_t size = std::min(a.size(), b.size());
    result.resize(size);
    
    if (size >= 4) {
        simd_vector_add(a.data(), b.data(), result.data(), size);
        impl_->stats.simd_operations++;
    } else {
        // Fallback for small vectors
        for (size_t i = 0; i < size; ++i) {
            result[i] = a[i] + b[i];
        }
        impl_->stats.fallback_operations++;
    }
}

void LinearAlgebraEngine::vector_subtract_simd(const std::vector<double>& a, const std::vector<double>& b, std::vector<double>& result) {
    size_t size = std::min(a.size(), b.size());
    result.resize(size);
    
    if (size >= 4) {
        simd_vector_subtract(a.data(), b.data(), result.data(), size);
        impl_->stats.simd_operations++;
    } else {
        // Fallback for small vectors
        for (size_t i = 0; i < size; ++i) {
            result[i] = a[i] - b[i];
        }
        impl_->stats.fallback_operations++;
    }
}

void LinearAlgebraEngine::scalar_multiply_simd(const std::vector<double>& vector, double scalar, std::vector<double>& result) {
    size_t size = vector.size();
    result.resize(size);
    
    if (size >= 4) {
        simd_scalar_multiply(vector.data(), scalar, result.data(), size);
        impl_->stats.simd_operations++;
    } else {
        // Fallback for small vectors
        for (size_t i = 0; i < size; ++i) {
            result[i] = vector[i] * scalar;
        }
        impl_->stats.fallback_operations++;
    }
}

double LinearAlgebraEngine::vector_norm_simd(const std::vector<double>& vector) {
    if (vector.size() >= 4) {
        return simd_vector_norm(vector.data(), vector.size());
    } else {
        // Fallback for small vectors
        double sum = 0.0;
        for (double val : vector) {
            sum += val * val;
        }
        return std::sqrt(sum);
    }
}

double LinearAlgebraEngine::correlation_simd(const std::vector<double>& a, const std::vector<double>& b) {
    size_t size = std::min(a.size(), b.size());
    if (size < 2) return 0.0;
    
    // Calculate means
    double mean_a = 0.0, mean_b = 0.0;
    for (size_t i = 0; i < size; ++i) {
        mean_a += a[i];
        mean_b += b[i];
    }
    mean_a /= size;
    mean_b /= size;
    
    // Calculate correlation using SIMD
    double numerator = 0.0;
    double sum_a_sq = 0.0, sum_b_sq = 0.0;
    
    for (size_t i = 0; i < size; ++i) {
        double diff_a = a[i] - mean_a;
        double diff_b = b[i] - mean_b;
        numerator += diff_a * diff_b;
        sum_a_sq += diff_a * diff_a;
        sum_b_sq += diff_b * diff_b;
    }
    
    double denominator = std::sqrt(sum_a_sq * sum_b_sq);
    return denominator > 0 ? numerator / denominator : 0.0;
}

double LinearAlgebraEngine::covariance_simd(const std::vector<double>& a, const std::vector<double>& b) {
    size_t size = std::min(a.size(), b.size());
    if (size < 2) return 0.0;
    
    // Calculate means
    double mean_a = 0.0, mean_b = 0.0;
    for (size_t i = 0; i < size; ++i) {
        mean_a += a[i];
        mean_b += b[i];
    }
    mean_a /= size;
    mean_b /= size;
    
    // Calculate covariance
    double sum = 0.0;
    for (size_t i = 0; i < size; ++i) {
        sum += (a[i] - mean_a) * (b[i] - mean_b);
    }
    
    return sum / (size - 1);
}

void LinearAlgebraEngine::matrix_transpose_simd(const std::vector<std::vector<double>>& matrix,
                                               std::vector<std::vector<double>>& result) {
    size_t rows = matrix.size();
    size_t cols = matrix[0].size();
    
    result.resize(cols);
    for (size_t i = 0; i < cols; ++i) {
        result[i].resize(rows);
    }
    
    if (rows >= 4 && cols >= 4) {
        simd_matrix_transpose(matrix[0].data(), result[0].data(), rows, cols);
        impl_->stats.simd_operations++;
    } else {
        // Fallback for small matrices
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result[j][i] = matrix[i][j];
            }
        }
        impl_->stats.fallback_operations++;
    }
}

void LinearAlgebraEngine::matrix_multiply_simd(const std::vector<std::vector<double>>& a,
                                               const std::vector<std::vector<double>>& b,
                                               std::vector<std::vector<double>>& result) {
    size_t a_rows = a.size();
    size_t a_cols = a[0].size();
    size_t b_cols = b[0].size();
    
    result.resize(a_rows);
    for (size_t i = 0; i < a_rows; ++i) {
        result[i].resize(b_cols);
    }
    
    if (a_rows >= 4 && a_cols >= 4 && b_cols >= 4) {
        simd_matrix_multiply(a[0].data(), b[0].data(), result[0].data(), a_rows, a_cols, b_cols);
        impl_->stats.simd_operations++;
    } else {
        // Fallback for small matrices
        for (size_t i = 0; i < a_rows; ++i) {
            for (size_t j = 0; j < b_cols; ++j) {
                result[i][j] = 0.0;
                for (size_t k = 0; k < a_cols; ++k) {
                    result[i][j] += a[i][k] * b[k][j];
                }
            }
        }
        impl_->stats.fallback_operations++;
    }
}

void LinearAlgebraEngine::eigendecomposition_simd(const std::vector<std::vector<double>>& matrix,
                                                 std::vector<double>& eigenvalues,
                                                 std::vector<std::vector<double>>& eigenvectors) {
    // Simplified eigendecomposition - real implementation would use LAPACK
    size_t size = matrix.size();
    eigenvalues.resize(size);
    eigenvectors.resize(size);
    
    for (size_t i = 0; i < size; ++i) {
        eigenvalues[i] = matrix[i][i]; // Diagonal elements as eigenvalues
        eigenvectors[i].resize(size);
        eigenvectors[i][i] = 1.0; // Identity matrix as eigenvectors
    }
}

void LinearAlgebraEngine::cholesky_decomposition_simd(const std::vector<std::vector<double>>& matrix,
                                                      std::vector<std::vector<double>>& result) {
    size_t size = matrix.size();
    result.resize(size);
    
    for (size_t i = 0; i < size; ++i) {
        result[i].resize(size);
        for (size_t j = 0; j <= i; ++j) {
            double sum = matrix[i][j];
            for (size_t k = 0; k < j; ++k) {
                sum -= result[i][k] * result[j][k];
            }
            if (i == j) {
                result[i][j] = std::sqrt(sum);
            } else {
                result[i][j] = sum / result[j][j];
            }
        }
    }
}

void LinearAlgebraEngine::solve_linear_system_simd(const std::vector<std::vector<double>>& A,
                                                  const std::vector<double>& b,
                                                  std::vector<double>& x) {
    // Simplified linear system solver - real implementation would use LU decomposition
    size_t size = A.size();
    x.resize(size);
    
    // Simple back substitution for upper triangular matrix
    for (int i = size - 1; i >= 0; --i) {
        x[i] = b[i];
        for (size_t j = i + 1; j < size; ++j) {
            x[i] -= A[i][j] * x[j];
        }
        x[i] /= A[i][i];
    }
}

LinearAlgebraEngine::EngineStats LinearAlgebraEngine::get_stats() const {
    return impl_->stats;
}

void LinearAlgebraEngine::reset_stats() {
    impl_->stats = EngineStats{};
}

// SIMD helper function implementations
void LinearAlgebraEngine::simd_vector_add(const double* a, const double* b, double* result, size_t size) {
    size_t simd_size = size - (size % 4);
    
    // Process 4 elements at a time using AVX2
    for (size_t i = 0; i < simd_size; i += 4) {
        __m256d vec_a = _mm256_loadu_pd(&a[i]);
        __m256d vec_b = _mm256_loadu_pd(&b[i]);
        __m256d vec_result = _mm256_add_pd(vec_a, vec_b);
        _mm256_storeu_pd(&result[i], vec_result);
    }
    
    // Handle remaining elements
    for (size_t i = simd_size; i < size; ++i) {
        result[i] = a[i] + b[i];
    }
}

void LinearAlgebraEngine::simd_vector_subtract(const double* a, const double* b, double* result, size_t size) {
    size_t simd_size = size - (size % 4);
    
    for (size_t i = 0; i < simd_size; i += 4) {
        __m256d vec_a = _mm256_loadu_pd(&a[i]);
        __m256d vec_b = _mm256_loadu_pd(&b[i]);
        __m256d vec_result = _mm256_sub_pd(vec_a, vec_b);
        _mm256_storeu_pd(&result[i], vec_result);
    }
    
    for (size_t i = simd_size; i < size; ++i) {
        result[i] = a[i] - b[i];
    }
}

void LinearAlgebraEngine::simd_scalar_multiply(const double* vector, double scalar, double* result, size_t size) {
    size_t simd_size = size - (size % 4);
    __m256d scalar_vec = _mm256_set1_pd(scalar);
    
    for (size_t i = 0; i < simd_size; i += 4) {
        __m256d vec = _mm256_loadu_pd(&vector[i]);
        __m256d vec_result = _mm256_mul_pd(vec, scalar_vec);
        _mm256_storeu_pd(&result[i], vec_result);
    }
    
    for (size_t i = simd_size; i < size; ++i) {
        result[i] = vector[i] * scalar;
    }
}

double LinearAlgebraEngine::simd_dot_product(const double* a, const double* b, size_t size) {
    size_t simd_size = size - (size % 4);
    __m256d sum_vec = _mm256_setzero_pd();
    
    for (size_t i = 0; i < simd_size; i += 4) {
        __m256d vec_a = _mm256_loadu_pd(&a[i]);
        __m256d vec_b = _mm256_loadu_pd(&b[i]);
        __m256d product = _mm256_mul_pd(vec_a, vec_b);
        sum_vec = _mm256_add_pd(sum_vec, product);
    }
    
    // Sum the 4 elements in the vector
    double sum_array[4];
    _mm256_storeu_pd(sum_array, sum_vec);
    double sum = sum_array[0] + sum_array[1] + sum_array[2] + sum_array[3];
    
    // Handle remaining elements
    for (size_t i = simd_size; i < size; ++i) {
        sum += a[i] * b[i];
    }
    
    return sum;
}

double LinearAlgebraEngine::simd_vector_norm(const double* vector, size_t size) {
    size_t simd_size = size - (size % 4);
    __m256d sum_vec = _mm256_setzero_pd();
    
    for (size_t i = 0; i < simd_size; i += 4) {
        __m256d vec = _mm256_loadu_pd(&vector[i]);
        __m256d square = _mm256_mul_pd(vec, vec);
        sum_vec = _mm256_add_pd(sum_vec, square);
    }
    
    // Sum the 4 elements in the vector
    double sum_array[4];
    _mm256_storeu_pd(sum_array, sum_vec);
    double sum = sum_array[0] + sum_array[1] + sum_array[2] + sum_array[3];
    
    // Handle remaining elements
    for (size_t i = simd_size; i < size; ++i) {
        sum += vector[i] * vector[i];
    }
    
    return std::sqrt(sum);
}

void LinearAlgebraEngine::simd_matrix_vector_multiply(const double* matrix, const double* vector, double* result, size_t rows, size_t cols) {
    for (size_t i = 0; i < rows; ++i) {
        result[i] = simd_dot_product(&matrix[i * cols], vector, cols);
    }
}

void LinearAlgebraEngine::simd_matrix_transpose(const double* matrix, double* result, size_t rows, size_t cols) {
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result[j * rows + i] = matrix[i * cols + j];
        }
    }
}

void LinearAlgebraEngine::simd_matrix_multiply(const double* a, const double* b, double* result, size_t a_rows, size_t a_cols, size_t b_cols) {
    for (size_t i = 0; i < a_rows; ++i) {
        for (size_t j = 0; j < b_cols; ++j) {
            result[i * b_cols + j] = simd_dot_product(&a[i * a_cols], &b[j], a_cols);
        }
    }
}

} // namespace ibkr_trading
