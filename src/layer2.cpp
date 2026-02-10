#include "layer2.h"


Layer::Layer(
    size_t input_size,
    size_t output_size,
    Activations::ActivationType act_type,
    InitType init_type
) : inputSize(input_size),
    outputSize(output_size),
    weights(output_size, input_size),
    biases(output_size, 1),
    actType(act_type),
    initType(init_type)
{
    initialize();
    biases.fill(0.0);
}

Matrix Layer::forward(const Matrix& X) {
    input = X;
    Matrix A = (weights * X) + biases;
    A = Activations::activate(A, actType);
    output = A;

    return A;
}

void Layer::initialize() {
    switch (initType) {
        case InitType::RANDOM: { weights.randomize(); }
        case InitType::XAVIER: { weights.xavierInit(); }
        case InitType::HE: { weights.heInit(); }
        case InitType::NONE: { weights.fill(1.0); }
        default: throw std::invalid_argument("USE InitType::NONE instead");
    }
}
