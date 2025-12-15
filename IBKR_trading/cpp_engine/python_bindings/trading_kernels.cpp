#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <vector>

namespace py = pybind11;

// Barrier gate batch computation
py::array_t<double> barrier_gate_batch(
    py::array_t<double> p_peak,
    py::array_t<double> p_valley,
    double g_min,
    double gamma,
    double delta
) {
    auto peak_buf = p_peak.unchecked<1>();
    auto valley_buf = p_valley.unchecked<1>();
    auto result = py::array_t<double>(p_peak.size());
    auto result_buf = result.mutable_unchecked<1>();
    
    for (py::ssize_t i = 0; i < p_peak.size(); ++i) {
        double p_p = std::min(1.0, std::max(0.0, peak_buf(i)));
        double p_v = std::min(1.0, std::max(0.0, valley_buf(i)));
        
        double peak_term = std::pow(1.0 - p_p, gamma);
        double valley_term = std::pow(0.5 + 0.5 * p_v, delta);
        
        double gate = peak_term * valley_term;
        result_buf(i) = std::max(g_min, std::min(1.0, gate));
    }
    
    return result;
}

// Simplex projection
py::array_t<double> project_simplex(py::array_t<double> weights) {
    auto w_buf = weights.unchecked<1>();
    auto result = py::array_t<double>(weights.size());
    auto result_buf = result.mutable_unchecked<1>();
    
    // Copy weights to vector for sorting
    std::vector<double> w_vec(w_buf.data(), w_buf.data() + weights.size());
    std::vector<double> sorted_w = w_vec;
    std::sort(sorted_w.begin(), sorted_w.end(), std::greater<double>());
    
    // Find threshold
    double cssv = 0.0;
    int rho = 0;
    for (int i = 0; i < static_cast<int>(sorted_w.size()); ++i) {
        cssv += sorted_w[i];
        if (sorted_w[i] * (i + 1) > cssv - 1) {
            rho = i;
        }
    }
    
    double theta = (cssv - 1.0) / (rho + 1);
    
    // Project to simplex
    for (py::ssize_t i = 0; i < weights.size(); ++i) {
        result_buf(i) = std::max(0.0, w_buf(i) - theta);
    }
    
    // Renormalize
    double sum = 0.0;
    for (py::ssize_t i = 0; i < weights.size(); ++i) {
        sum += result_buf(i);
    }
    
    if (sum > 0) {
        for (py::ssize_t i = 0; i < weights.size(); ++i) {
            result_buf(i) /= sum;
        }
    } else {
        double uniform = 1.0 / weights.size();
        for (py::ssize_t i = 0; i < weights.size(); ++i) {
            result_buf(i) = uniform;
        }
    }
    
    return result;
}

// Risk parity ridge optimization
py::array_t<double> risk_parity_ridge(
    py::array_t<double> z_scores,
    py::array_t<double> covariance,
    double lambda_reg
) {
    auto z_buf = z_scores.unchecked<1>();
    auto cov_buf = covariance.unchecked<2>();
    
    int n = z_scores.size();
    auto result = py::array_t<double>(n);
    auto result_buf = result.mutable_unchecked<1>();
    
    // Convert to Eigen matrices
    Eigen::Map<const Eigen::VectorXd> z(z_buf.data(), n);
    Eigen::Map<const Eigen::MatrixXd> cov(cov_buf.data(), n, n);
    
    // Add regularization: (Σ + λI)
    Eigen::MatrixXd reg_cov = cov + lambda_reg * Eigen::MatrixXd::Identity(n, n);
    
    // Solve: w = λ * (Σ + λI)^(-1) * z
    Eigen::VectorXd w = lambda_reg * reg_cov.ldlt().solve(z);
    
    // Copy result
    for (int i = 0; i < n; ++i) {
        result_buf(i) = w(i);
    }
    
    return result;
}

// Horizon softmax arbitration
std::pair<py::array_t<double>, int> horizon_softmax(
    py::array_t<double> alpha_matrix,
    py::array_t<double> vol_matrix,
    double beta
) {
    auto alpha_buf = alpha_matrix.unchecked<2>();
    auto vol_buf = vol_matrix.unchecked<2>();
    
    int n_symbols = alpha_matrix.shape(0);
    int n_horizons = alpha_matrix.shape(1);
    
    auto result = py::array_t<double>(n_symbols);
    auto result_buf = result.mutable_unchecked<1>();
    
    // Scale alphas by volatility
    Eigen::MatrixXd scaled_alphas(n_symbols, n_horizons);
    for (int i = 0; i < n_symbols; ++i) {
        for (int j = 0; j < n_horizons; ++j) {
            double vol_scale = vol_buf(i, j) + 1e-8;
            scaled_alphas(i, j) = alpha_buf(i, j) / vol_scale;
        }
    }
    
    // Compute softmax weights
    Eigen::MatrixXd exp_alphas = (beta * scaled_alphas).array().exp();
    Eigen::VectorXd row_sums = exp_alphas.rowwise().sum();
    Eigen::MatrixXd weights = exp_alphas.array().colwise() / row_sums.array();
    
    // Weighted combination
    for (int i = 0; i < n_symbols; ++i) {
        result_buf(i) = 0.0;
        for (int j = 0; j < n_horizons; ++j) {
            result_buf(i) += weights(i, j) * alpha_buf(i, j);
        }
    }
    
    // Find most weighted horizon
    Eigen::VectorXd avg_weights = weights.colwise().mean();
    int selected_horizon = 0;
    double max_weight = avg_weights(0);
    for (int j = 1; j < n_horizons; ++j) {
        if (avg_weights(j) > max_weight) {
            max_weight = avg_weights(j);
            selected_horizon = j;
        }
    }
    
    return std::make_pair(result, selected_horizon);
}

// EWMA volatility computation
py::array_t<double> ewma_vol(py::array_t<double> returns, double alpha) {
    auto ret_buf = returns.unchecked<1>();
    auto result = py::array_t<double>(returns.size());
    auto result_buf = result.mutable_unchecked<1>();
    
    double m2 = 0.0;
    for (py::ssize_t i = 0; i < returns.size(); ++i) {
        double x2 = ret_buf(i) * ret_buf(i);
        m2 = alpha * x2 + (1.0 - alpha) * m2;
        result_buf(i) = std::sqrt(m2);
    }
    
    return result;
}

// OFI computation
py::array_t<double> ofi_batch(
    py::array_t<double> pb0, py::array_t<double> pa0,
    py::array_t<double> sb0, py::array_t<double> sa0,
    py::array_t<double> pb1, py::array_t<double> pa1,
    py::array_t<double> sb1, py::array_t<double> sa1
) {
    auto pb0_buf = pb0.unchecked<1>();
    auto pa0_buf = pa0.unchecked<1>();
    auto sb0_buf = sb0.unchecked<1>();
    auto sa0_buf = sa0.unchecked<1>();
    auto pb1_buf = pb1.unchecked<1>();
    auto pa1_buf = pa1.unchecked<1>();
    auto sb1_buf = sb1.unchecked<1>();
    auto sa1_buf = sa1.unchecked<1>();
    
    auto result = py::array_t<double>(pb0.size());
    auto result_buf = result.mutable_unchecked<1>();
    
    for (py::ssize_t i = 0; i < pb0.size(); ++i) {
        double a = 0.0;
        if (pb1_buf(i) > pb0_buf(i)) a += sb1_buf(i);
        if (pa1_buf(i) < pa0_buf(i)) a -= sa1_buf(i);
        if (pb1_buf(i) == pb0_buf(i)) a += (sb1_buf(i) - sb0_buf(i));
        if (pa1_buf(i) == pa0_buf(i)) a -= (sa1_buf(i) - sa0_buf(i));
        result_buf(i) = a;
    }
    
    return result;
}

PYBIND11_MODULE(ibkr_trading_engine_py, m) {
    m.doc() = "IBKR Trading Engine - High Performance C++ Kernels";
    
    m.def("barrier_gate_batch", &barrier_gate_batch, 
          "Compute barrier gate for batch of probabilities",
          py::arg("p_peak"), py::arg("p_valley"), 
          py::arg("g_min"), py::arg("gamma"), py::arg("delta"));
    
    m.def("project_simplex", &project_simplex,
          "Project weights to probability simplex",
          py::arg("weights"));
    
    m.def("risk_parity_ridge", &risk_parity_ridge,
          "Solve ridge risk parity optimization",
          py::arg("z_scores"), py::arg("covariance"), py::arg("lambda_reg"));
    
    m.def("horizon_softmax", &horizon_softmax,
          "Softmax arbitration over horizons",
          py::arg("alpha_matrix"), py::arg("vol_matrix"), py::arg("beta"));
    
    m.def("ewma_vol", &ewma_vol,
          "Compute EWMA volatility",
          py::arg("returns"), py::arg("alpha"));
    
    m.def("ofi_batch", &ofi_batch,
          "Compute Order Flow Imbalance for batch",
          py::arg("pb0"), py::arg("pa0"), py::arg("sb0"), py::arg("sa0"),
          py::arg("pb1"), py::arg("pa1"), py::arg("sb1"), py::arg("sa1"));
}
