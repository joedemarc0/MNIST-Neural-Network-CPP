#include "layer.h"
#include <stdexcept>


Layer::Layer(size_t input_size,
            size_t output_size,
            Activations::ActivationType actType,
            InitType initType
        ) : inputSize(input_size), outputSize(output_size), weights(output_size, input_size), biases(output_size, 1), actType(actType), initType(initType)
{
    initialize();
    biases.fill(0.0);
}

Matrix Layer::forward(const Matrix& X) {
    input = X;
    z = (weights * X) + biases;
    output = Activations::activate(z, actType);
    return output;
}

Matrix Layer::backward(const Matrix& dA, double learning_rate) {
    Matrix dZ;
    if (actType == Activations::ActivationType::SOFTMAX) {
        dZ = dA;
    } else {
        dZ = dA.hadamard(Activations::deriv_activate(z, actType));
    }

    Matrix dW = dZ * input.transpose();
    Matrix dB = dZ;

    weights -= dW * learning_rate;
    biases -= dB * learning_rate;

    return weights.transpose() * dZ;
}

void Layer::initialize() {
    switch (initType) {
        case InitType::RANDOM : weights.randomize(-1.0, 1.0); break;
        case InitType::XAVIER : weights.xavierInit(); break;
        case InitType::HE : weights.heInit(); break;
        case InitType::NONE : weights.fill(0.0); break;
        default : throw std::invalid_argument("Unknown initialization type");
    }
}