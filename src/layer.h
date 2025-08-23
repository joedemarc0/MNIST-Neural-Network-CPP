/**
 * 
 */

#ifndef LAYER_H
#define LAYER_H

#include "matrix.h"
#include "activation.h"
#include <functional>
#include <string>


class Layer {
    public:
        enum class ActivationType { NONE, SIGMOID, RELU, LEAKY_RELU, SOFTMAX };
        enum class InitType { NONE, RANDOM, XAVIER, HE };

        // Constructor
        Layer(
            size_t input_size,
            size_t output_size,
            ActivationType actType = ActivationType::RELU,
            InitType initType = InitType::XAVIER
        );

        Matrix forward(const Matrix& X);
        Matrix backward(const Matrix& dA, double learning_rate);

        // Getters
        const Matrix getWeights() const { return weights; }
        const Matrix getBiases() const { return biases; }
        const Matrix getOutput() const { return output; }
        const Matrix getZ() const { return z; }
    
    private:
        size_t inputSize;
        size_t outputSize;

        Matrix weights;
        Matrix biases;
        Matrix input;
        Matrix output;
        Matrix z;

        ActivationType actType;
        
        // Helpers for activation and derivative
        Matrix activate(const Matrix& x) const;
        Matrix activate_deriv(const Matrix& x) const;

        // Helper for init
        void initialize(InitType initType);
};

#endif // LAYER_H