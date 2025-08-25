#include "activation.h"
#include <cmath>
#include <algorithm>


namespace Activations {
    Matrix activate(const Matrix& x, ActivationType type) {
        Matrix result(x.getRows(), x.getCols());

        switch (type) {
            case ActivationType::SIGMOID: {
                for (size_t i = 0; i < x.getRows(); ++i) {
                    for (size_t j = 0; j < x.getCols(); ++j) {
                        result(i, j) = 1.0 / (1.0 + std::exp(-x(i, j)));
                    }
                }
                break;
            }

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
                for (size_t i = 0; i < x.getRows(); ++i) {
                    double rowMax = x.getRow(i)[0];
                    for (size_t j = 1; j < x.getCols(); ++j) {
                        if (x(i, j) > rowMax) rowMax = x(i, j);
                    }

                    double sumExp = 0.0;
                    for (size_t j = 0; j < x.getCols(); ++j) {
                        result(i, j) = std::exp(x(i, j) - rowMax);
                        sumExp += result(i, j);
                    }

                    for (size_t j = 0; j < x.getCols(); ++j) {
                        result(i, j) /= sumExp;
                    }
                }
                break;
            }

            default: result = x;
        }

        return result;
    }

    Matrix deriv_activate(const Matrix& x, ActivationType type) {
        Matrix result(x.getRows(), x.getCols());
        switch (type) {
            case ActivationType::SIGMOID: {
                Matrix s(x.getRows(), x.getCols());
                for (size_t i = 0; i < x.getRows(); ++i) {
                    for (size_t j = 0; j < x.getCols(); ++j) {
                        s(i, j) = 1.0 / (1.0 + std::exp(-x(i, j)));
                        result(i, j) = s(i, j) * (1.0 - s(i, j));
                    }
                }
                break;
            }

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

            default: result = x;
        }

        return result;
    }
}