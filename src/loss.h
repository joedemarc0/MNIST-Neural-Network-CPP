#ifndef LOSS_H
#define LOSS_H

#include "matrix.h"
#include <cstddef>
#include <cmath>
#include <stdexcept>


namespace Loss {
    enum class LossType { MSE, CROSS_ENTROPY };
    double compute(const Matrix& y_true, const Matrix& y_pred, LossType type);
    Matrix derivative(const Matrix& y_true, const Matrix& y_pred, LossType type);
}

#endif // LOSS_H