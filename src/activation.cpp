#include "activation.h"
#include <cmath>
#include <algorithm>

using namespace Activations;


Matrix Activations::activate(const Matrix& x, ActivationType type) {
    switch(type) {
        case ActivationType::RELU: return x.apply([](double v) { return std::max(0.0, v); });
        case ActivationType::LEAKY_RELU: {
            constexpr double alpha = 0.01;
            return x.apply([](double v) { return v > 0.0 ? v : alpha * v; });
        }

        case ActivationType::SOFTMAX: {
            size_t rows = x.getRows();
            size_t cols = x.getCols();
            Matrix result(rows, cols);
            const double* __restrict src = x.dataPtr();
            double* __restrict dst = result.dataPtr();

            for (size_t j = 0; j < cols; ++j) {
                double max_val = src[j];
                for (size_t i = 1; i < rows; ++i) if (src[i * cols + j] > max_val) max_val = src[i * cols + j];

                double exp_sum = 0.0;
                for (size_t i = 0; i < rows; ++i) {
                    dst[i * cols + j] = std::exp(src[i * cols + j] - max_val);
                    exp_sum += dst[i * cols + j];
                }

                const double inv_sum = 1.0 / exp_sum;
                for (size_t i = 0; i < rows; ++i) dst[i * cols + j] = std::max(dst[i * cols + j] * inv_sum, 1e-12);
            }

            return result;
        }

        default: throw std::runtime_error("Activation type unspecified");
    }
}

Matrix Activations::deriv_activate(const Matrix& x, ActivationType type) {
    switch(type) {
        case ActivationType::RELU: return x.apply([](double v) { return v > 0.0 ? 1.0 : 0.0; });
        case ActivationType::LEAKY_RELU: {
            constexpr double alpha = 0.01;
            return x.apply([](double v) {return v > 0.0 ? 1.0 : alpha; });
        }

        default: throw std::runtime_error("Activation type unspecified");
    }
}

std::string Activations::to_string(ActivationType type) {
    switch(type) {
        case ActivationType::RELU: return "RELU";
        case ActivationType::LEAKY_RELU: return "LEAKY RELU";
        case ActivationType::SOFTMAX: return "SOFTMAX";
        default: throw std::runtime_error("Activation type unspecified");
    }
}