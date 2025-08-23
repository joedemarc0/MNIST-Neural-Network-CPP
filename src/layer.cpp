#include "layer.h"
#include <stdexcept>


Layer::Layer(size_t input_size,
            size_t output_size,
            Layer::ActivationType actType,
            Layer::InitType initType
        ) : inputSize(input_size), outputSize(output_size), weights(output_size, input_size), biases(output_size, 1), actType(actType)
{
    initialize(initType);
    biases.fill(0.0);
}

Matrix Layer::forward(const Matrix& X) {
    input = X;
    z = (weights * X) + biases;
    output = activate(z);
    return output;
}

Matrix Layer::backward(const Matrix& dA, double learning_rate) {
    Matrix dZ;
    if (actType == ActivationType::SOFTMAX) {
        dZ = dA;
    } else {
        dZ = dA.hadamard(activate_deriv(z));
    }

    Matrix dW = dZ * input.transpose();
    Matrix dB = dZ;

    weights -= dW * learning_rate;
    biases -= dB * learning_rate;

    return weights.transpose() * dZ;
}

Matrix Layer::activate(const Matrix& x) const {
    switch (actType) {
        case ActivationType::SIGMOID : return activations::sigmoid(x);
        case ActivationType::RELU : return activations::ReLU(x);
        case ActivationType::LEAKY_RELU : return activations::leaky_ReLU(x);
        case ActivationType::SOFTMAX : return activations::softmax(x);
        case ActivationType::NONE : return x;
        default : throw std::invalid_argument("Unknown activation type");
    }
}

Matrix Layer::activate_deriv(const Matrix& x) const {
    switch (actType) {
        case ActivationType::SIGMOID : return activations::deriv_sigmoid(x);
        case ActivationType::RELU : return activations::deriv_ReLU(x);
        case ActivationType::LEAKY_RELU : return activations::deriv_leaky_ReLU(x);
        case ActivationType::SOFTMAX :
            throw std::logic_error("Softmax derivation should be handled with loss");
        case ActivationType::NONE : 
            return Matrix(x.getRows(), x.getCols(), 1.0);
        default : throw std::invalid_argument("Unknown activation type");
    }
}

void Layer::initialize(InitType initType) {
    switch (initType) {
        case InitType::RANDOM : weights.randomize(-1.0, 1.0); break;
        case InitType::XAVIER : weights.xavierInit(); break;
        case InitType::HE : weights.heInit(); break;
        case InitType::NONE : weights.fill(0.0); break;
        default : throw std::invalid_argument("Unknown initialization type");
    }
}