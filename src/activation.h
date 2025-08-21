/**
 * File to contain all of the activation functions and their derivatives
 * 
 * Store activation functions using the Matrix class in matrix.h
 * Contains wide range of activation functions implemented to be used
 * Keep in mind that not all of these functions will be used however the option to use another function will be
 * I will probably implement ReLU functions on the two hidden layers and softmax on the output since I achieved 97% accuracy 
 * on test sets with this method (784 -> 128 -> 64 -> 10)
 */

#ifndef ACTIVATION_H
#define ACTIVATION_H

#include "matrix.h"

namespace activations {
    // Sigmoid activations
    Matrix sigmoid(const Matrix& x);
    Matrix deriv_sigmoid(const Matrix& x);

    // ReLU
    Matrix ReLU(const Matrix& x);
    Matrix deriv_ReLU(const Matrix& x);

    // Leaky ReLU
    Matrix leaky_ReLU(const Matrix& x);
    Matrix deriv_leaky_ReLU(const Matrix& x);

    // Softmax
    Matrix softmax(const Matrix& x);
    // No deriv function needed
}

#endif // ACTIVATION_H