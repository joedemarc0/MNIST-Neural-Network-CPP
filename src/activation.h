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

// namespace of activation functions for use on each layer
namespace Activations {
    enum class ActivationType{ NONE, SIGMOID, RELU, LEAKY_RELU, SOFTMAX };
    Matrix activate(const Matrix& x, ActivationType type);
    Matrix deriv_activate(const Matrix& x, ActivationType type);
}

#endif // ACTIVATION_H