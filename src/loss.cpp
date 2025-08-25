#include "loss.h"
#include <algorithm>
#include <iostream>


namespace Loss {
    double compute(const Matrix& y_true, const Matrix& y_pred, LossType type) {
        if (y_true.getRows() != y_pred.getRows() || y_true.getCols() != y_pred.getCols()) {
            throw std::invalid_argument("Shape mismatch in computing loss");
        }

        size_t m = y_true.getCols();
        double loss = 0.0;

        switch (type) {
            // Compute Loss for MSE
            case LossType::MSE: {
                Matrix diff = y_pred - y_true;

                for (size_t i = 0; i < diff.getRows(); ++i) {
                    for (size_t j = 0; j < diff.getCols(); ++j) {
                        loss += diff(i, j) * diff(i, j);
                    }
                }

                loss /= (2.0 * m);
                break;
            }

            // Compute Loss for Cross Entropy
            case LossType::CROSS_ENTROPY: {
                const double epsilon = 1e-12;

                for (size_t i = 0; i < y_true.getRows(); ++i) {
                    for (size_t j = 0; j < y_true.getCols(); ++j) {
                        double yt = y_true(i, j);
                        double yp = std::max(std::min(y_pred(i, j), 1.0 - epsilon), epsilon);
                        if (yt > 0.5) {
                            loss -= std::log(yp);
                        }
                    }
                }

                loss /= m;
                break;
            }

            default:
                throw std::invalid_argument("Unsupported loss type");
        }

        return loss;
    }

    Matrix derivative(const Matrix& y_true, const Matrix& y_pred, LossType type) {
        if (y_true.getRows() != y_pred.getRows() || y_true.getCols() != y_pred.getCols()) {
            throw std::invalid_argument("Shape mismatch in computing loss derivative");
        }

        size_t m = y_true.getCols();
        Matrix grad(y_true.getRows(), y_true.getCols());

        switch (type) {
            case LossType::MSE: {
                grad = (y_pred - y_true) * (1.0 / m);
                break;
            }

            case LossType::CROSS_ENTROPY: {
                grad = (y_pred - y_true) * (1.0 / m);
                break;
            }

            default:
                throw std::invalid_argument("Unsupported loss type");
        }

        return grad;
    }
}