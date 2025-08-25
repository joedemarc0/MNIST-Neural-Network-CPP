/**
 * Layer implementation, representing a fully connected (dense) neural network with configurable activation
 * functions and weight initialization strategies
 */

#ifndef LAYER_H
#define LAYER_H

#include "matrix.h"
#include "activation.h"
#include <functional>
#include <string>


/**
 * @class Layer
 * Implements a dense neural network layer, including forward and backward propagation through
 * the layer, activation functions, and weight initialization
 */
class Layer {
    public:
        /**
         * @enum InitType
         * Specifies which weight initialization to apply
         */
        enum class InitType { NONE, RANDOM, XAVIER, HE };

        /**
         * Constructor
         * @param input_size Number of neurons feeding into this layer (input dimension)
         * @param output_size Number of neurons in this layer (output dimension)
         * @param actType Activation function type
         * @param InitType Weight initialization type
         * 
         * Constructs layer based on layer size, and previous layer size
         * Allows for flexibiltiy with activation functions and initializations
         */
        Layer(
            size_t input_size,
            size_t output_size,
            Activations::ActivationType actType,
            InitType initType=InitType::XAVIER
        );

        /**
         * Perform forward propagation through this layer
         * @param X input matrix of shape (input_size, output_size)
         * @return Output matrix after applying weights, biases, and activation of shape (output_size, batch_size)
         * 
         * Internally stores input, pre-activation values (z), and output for use in backprop
         */
        Matrix forward(const Matrix& X);

        /**
         * Perform backward propagation through this layer
         * @param dA Gradient of the loss with respect to the layer's output (same shape as output)
         * @param learning_rate Learning rate for gradient descent updates
         * @return Gradient of the loss with respect to the layer's input (same shape as input)
         * 
         * Updates this layer's weights and biases using gradient descent
         */
        Matrix backward(const Matrix& dA, double learning_rate);

        /**
         * Getters
         * @return private variables weights, biases, output, and Z, all Matrix class items
         */
        const Matrix getWeights() const { return weights; }
        const Matrix getBiases() const { return biases; }
        const Matrix getOutput() const { return output; }
        const Matrix getZ() const { return z; }
    
    private:
        size_t inputSize;           // Number of inputs to the layer
        size_t outputSize;          // Number of outputs in the layer

        Matrix weights;             // Weight matrix of shape (output_size, input_size)
        Matrix biases;              // Biases vector of shape (output_size, 1)
        Matrix input;               // Cached input values (for backprop)
        Matrix output;              // Cached output values (post-activation)
        Matrix z;                   // Cached linear transformation values (pre-activation)

        Activations::ActivationType actType;     // Chosen activation function for this layer
        InitType initType;          // Chosen weight initialization method

        /**
         * Initialize Matric weights
         * @return Weights matrix with properly initialized values ready to undergo backprop
         * 
         * Helper function that applies initialization to weights given Layer initType
         */
        void initialize();
};

#endif // LAYER_H