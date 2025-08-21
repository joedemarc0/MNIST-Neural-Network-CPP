#include "activation.h"
#include <cmath>
#include <algorithm>


namespace activations {
    // Sigmoid activations
    Matrix sigmoid(const Matrix& x) {
        Matrix result(x.getRows(), x.getCols());
        for (size_t i = 0; i < x.getRows(); ++i) {
            for (size_t j = 0; j < x.getCols(); ++j) {
                result(i, j) = 1.0 / (1.0 + std::exp(-x(i, j)));
            }
        }
        return result;
    }

    Matrix deriv_sigmoid(const Matrix& x) {
        Matrix s = sigmoid(x);
        Matrix ones(s.getRows(), s.getCols());
        ones.fill(1.0);

        return s.hadamard(ones - s);
    }

    // ReLU
    Matrix ReLU(const Matrix& x) {
        Matrix result(x.getRows(), x.getCols());
        for (size_t i = 0; i < x.getRows(); ++i) {
            for (size_t j = 0; j < x.getCols(); ++j) {
                result(i, j) = std::max(0.0, x(i, j));
            }
        }
        return result;
    }

    Matrix deriv_ReLU(const Matrix& x) {
        Matrix result(x.getRows(), x.getCols());
        for (size_t i = 0; i < x.getRows(); ++i) {
            for (size_t j = 0; j < x.getCols(); ++j) {
                result(i, j) = x(i, j) > 0 ? 1.0 : 0.0;
            }
        }
        return result;
    }

    // Leaky ReLU
    Matrix leaky_ReLU(const Matrix& x) {
        Matrix result(x.getRows(), x.getCols());
        double alpha = 0.01;

        for (size_t i = 0; i < x.getRows(); ++i) {
            for (size_t j = 0; j < x.getCols(); ++j) {
                result(i, j) = x(i, j) > 0 ? x(i, j) : alpha * x(i, j);
            }
        }
        return result;
    }

    Matrix deriv_leaky_ReLU(const Matrix& x) {
        Matrix result(x.getRows(), x.getCols());
        double alpha = 0.01;

        for (size_t i = 0; i < x.getRows(); ++i) {
            for (size_t j = 0; j < x.getCols(); ++j) {
                result(i, j) = x(i, j) > 0 ? 1.0 : alpha;
            }
        }
        return result;
    }

    // Softmax
    Matrix softmax(const Matrix& x) {
        Matrix result(x.getRows(), x.getCols());
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

        return result;
    }
}