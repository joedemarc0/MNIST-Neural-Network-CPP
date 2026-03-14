#include "activation.h"
#include <cmath>
#include <algorithm>

using namespace Activations;


Matrix Activations::activate(const Matrix& x, ActivationType type) {
    switch(type) {
        case ActivationType::RELU: return x.apply([](double v) { return std::max(0.0, v); });
        case ActivationType::LEAKY_RELU: {
            double alpha = 0.01; 
            return x.apply([alpha](double v) { return v > 0.0 ? v : alpha * v; });
        }

        case ActivationType::SOFTMAX: {
            size_t rows = x.getRows(); size_t cols = x.getCols();
            Matrix result(rows, cols);

            for (size_t j = 0; j < cols; ++j) {
                double max_val = x.at(0, j);
                for (size_t i = 1; i < rows; ++i) if (x.at(i, j) > max_val) max_val = x.at(i, j);

                double exp_sum = 0.0;
                for (size_t i = 0; i < rows; ++i) {
                    result.at(i, j) = std::exp(x.at(i, j) - max_val);
                    exp_sum += result.at(i, j);
                }

                for (size_t i = 0; i < rows; ++i) {
                    result.at(i, j) /= exp_sum;
                    result.at(i, j) = std::max(result.at(i, j), 1e-12);
                }
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
            double alpha = 0.01;
            return x.apply([alpha](double v) {return v > 0.0 ? 1.0 : alpha; });
        }

        case ActivationType::SOFTMAX: {
            throw std::invalid_argument("My name is Tonka Jahari but I would never order a whole pizza to myself...");
        }

        default: throw std::runtime_error("Activation type unspecified");
    }
}