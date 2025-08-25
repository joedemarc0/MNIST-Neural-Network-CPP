/**
 * Neural Network Class Implementation. Represents fully configured neural network, ready to train,
 * predict, and evaluate data. Options to save and load models into /models/ folder
 */

#ifndef NETWORK_H
#define NETWORK_H

#include "layer.h"
#include "matrix.h"
#include "loss.h"
#include <vector>
#include <string>
#include <utility>
#include <fstream>
#include <stdexcept>


/**
 * @class Network
 * Implements Neural Network class that handles training, predicting, evaluating,
 * saving and loading model
 */
class Network {
    public:
        /**
         * @enum InitType
         * Specifies which initialization method to use
         */
        enum class InitType { NONE, RANDOM, XAVIER, HE };

        /**
         * Constructor
         * @param input_size Size of input data (784,)
         * @param learning_rate Eta in Gradient Descent, specifies how much weights should change by with each epoch
         * @param actType Type of activation function to use for each HIDDEN layer. Output layer is SOFTMAX by default
         * @param initType Type of weight initialization
         * @param lossType Type of loss function to utilize
         */
        Network(
            size_t input_size,
            double learning_rate,
            Activations::ActivationType actType,
            InitType initType,
            Loss::LossType lossType
        );

        /**
         * Training loop function
         * @param X Input image matrix, size (input_size, m), where m = 60000 for training set
         * @param y Matrix of labels for image comparison, size (m, 1)
         * @param epochs Number of epochs to iterate over
         * @param batch_size Number of batches to split data into (using mini-batches technique)
         * @param shuffle Whether to shuffle training set or not during training between epochs
         * 
         * Modifies weights and biases of Layer class objects using Gradient Descent
         * Shuffles training Matrix X if shuffle=true, and splits into n batches if
         * batch_size=n.
         */
        void train(const Matrix& X, const Matrix& y,
                   size_t epochs,
                   size_t batch_size=0,
                   bool shuffle=true);

        Matrix predict(const Matrix& X) const;
        double evaluate(const Matrix& X, const Matrix& y) const;

        void saveModel(const std::string& filename) const;
        void loadModel(const std::string& filename);
    
    private:
        /**
         * Forward Propagation Function
         * @param X Input matrix of size (784, m), where m is the size of the dataset
         * @return A, Matrix that has undergone all layers and their activation functions, is of size (10, m)
         * 
         * Sends input matrix through all layers and activation functions and returns 10xm matrix 
         * encoding the neural networks guess
         */
        Matrix forward(const Matrix& X);

        /**
         * Backward Propagation Function
         * @param y Labels Matrix, one-hot encoding all images
         * 
         * Cycles through layers backwards and updates layer weights and biases using layer.backward() function
         */
        void backward(const Matrix& y);

        std::vector<Layer> layers;
        size_t inputSize;
        double learningRate;
        Matrix last_output;

        Activations::ActivationType actType;
        InitType initType;
        Loss::LossType lossType;
};


#endif // NETWORK_H