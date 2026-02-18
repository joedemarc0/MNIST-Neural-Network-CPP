#include "activation.h"
#include <cmath>
#include <algorithm>


namespace Activations {
    Matrix activate(const Matrix& x, ActivationType type) {
        Matrix result(x.getRows(), x.getCols());

        switch (type) {
            case ActivationType::RELU: {
                for (size_t i = 0; i < x.getRows(); ++i) {
                    for (size_t j = 0; j < x.getCols(); ++j) {
                        result(i, j) = std::max(0.0, x(i, j));
                    }
                }
                break;
            }

            case ActivationType::LEAKY_RELU: {
                double alpha = 0.01;
                for (size_t i = 0; i < x.getRows(); ++i) {
                    for (size_t j = 0; j < x.getCols(); ++j) {
                        result(i, j) = x(i, j) > 0 ? x(i, j) : alpha * x(i, j);
                    }
                }
                break;
            }

            case ActivationType::SOFTMAX: {
                for (size_t j = 0; j < x.getCols(); ++j) {
                    double max_val = x(0, j);
                    for (size_t i = 1; i < x.getRows(); ++i) {
                        if (x(i, j) > max_val) max_val = x(i, j);
                    }

                    double exp_sum = 0.0;
                    for (size_t i = 0; i < x.getRows(); ++i) {
                        result(i, j) = std::exp(x(i, j) - max_val);
                        exp_sum += result(i, j);
                    }

                    for(size_t i = 0; i < x.getRows(); ++i) {
                        result(i, j) /= exp_sum;
                        result(i, j) = std::max(result(i, j), 1e-12);
                    }
                }
                break;
            }

            default: throw std::runtime_error("Layer Activation Function Type Unspecified");
        }

        return result;
    }

    Matrix deriv_activate(const Matrix& x, ActivationType type) {
        Matrix result(x.getRows(), x.getCols());
        switch (type) {
            case ActivationType::RELU: {
                for (size_t i = 0; i < x.getRows(); ++i) {
                    for (size_t j = 0; j < x.getCols(); ++j) {
                        result(i, j) = x(i, j) > 0 ? 1.0 : 0.0;
                    }
                }
                break;
            }

            case ActivationType::LEAKY_RELU: {
                double alpha = 0.01;
                for (size_t i = 0; i < x.getRows(); ++i) {
                    for (size_t j = 0; j < x.getCols(); ++j) {
                        result(i, j) = x(i, j) > 0 ? 1.0 : alpha;
                    }
                }
                break;
            }

            case ActivationType::SOFTMAX: {
                throw std::invalid_argument("This should theoretically not be called");
            }

            default: throw std::runtime_error("Layer Activation Function Type Unspecified");
        }

        return result;
    }
}