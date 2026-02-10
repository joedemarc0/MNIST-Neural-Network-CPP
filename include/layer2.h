#ifndef LAYER_H
#define LAYER_H

#include "matrix.h"
#include "init.h"
#include "activation.h"


class Layer {
    private:
        size_t inputSize;
        size_t outputSize;

        Matrix weights;
        Matrix biases;
        Matrix input;
        Matrix output;

        InitType initType;
        Activations::ActivationType actType;

        void initialize();
    
    public:
        Layer(
            size_t input_size, 
            size_t output_size,
            Activations::ActivationType act_type,
            InitType init_type=InitType::XAVIER
        );

        Matrix forward(const Matrix& X);
        Matrix backward(const Matrix& dA, double learning_rate);

        // Getters
        Matrix getWeights() const { return weights; }
        Matrix getBiases() const { return biases; }
        Matrix getOutput() const { return output; }
        Activations::ActivationType getActivationType() const { return actType; }
        InitType getInitType() const { return initType; }
        size_t getInputSize() const { return inputSize; }
        size_t getOutputSize() const { return outputSize; }

        void setWeights(const Matrix& W) { weights = W; }
        void setBiases(const Matrix& b) { biases = b; }
};

#endif // LAYER_H