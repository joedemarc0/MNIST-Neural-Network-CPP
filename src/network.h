/**
 * Neural Network Class Implementation. Represents fully configured neural network, ready to train,
 * predict, and evaluate data. Options to save and load models into /models/ folder
 */

#ifndef NETWORK_H
#define NETWORK_H

#include "layer.h"
#include "matrix.h"
#include "loss.h"
#include "init.h"
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

        /** Minimal Constructor: Defaults to RELU + RANDOM */
        Network(
            size_t input_size,
            double learning_rate,
            Loss::LossType lossType
        ) : inputSize(input_size),
        learningRate(learning_rate),
        actType(Activations::ActivationType::RELU),
        initType(InitType::RANDOM),
        lossType(lossType) {}

        /** 
         * Add Layer Functions
         * @param neurons Amount of neurons in the layer, equivalent to output size of the added layer
         * @param actType Activation function type
         * @param initType Layer initialization type
         * 
         * Adds layer to the end of the std::vector<Layer> layers variable. Ensures that added layer has
         * input size equal to output size of last layer in list, and output size equal to neurons param
         */
        void addLayer(size_t neurons);
        void addLayer(size_t neurons,
                      Activations::ActivationType actType,
                      InitType initType);

        /**
         * Training/Test Batch accuracy function
         * @param predictions Matrix of predictions encoded in one-hot
         * @param y Matrix of labels for training/test batch
         * @return double representing the accuracy of our model
         * 
         * Receives predictions of training/testing/evaluation batch and returns
         * percent of predictions that are correct
         * 
         * Aiming for 97% with mini-batching, shuffling, decaying learning rate, etc
         */
        double get_accuracy(const Matrix& predictions, const Matrix& y) const;

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
        
        /**
         * Prediction function
         * @param X Input matrix of size (input_size, m) where m is number of samples
         * @return Matrix of size (10, m) where each column represents probability vector
         * 
         * Performs a forward pass through the neural network without modifying weights
         * and returns final softmaxxed output for each sample
         * 
         * Useful for testing, validating, or inference
         */
        Matrix predict(const Matrix& X) const;

        /**
         * Evaluation function
         * @param X Input matrix of size (input_size, m) where m is number of samples
         * @param y Label matrix of size (10, m)
         * 
         * Runs predict() on a set of size m and evaluates the accuracy of the model on
         * the sample using get_accuracy()
         */
        double evaluate(const Matrix& X, const Matrix& y) const;

        /**
         * Save Model Function
         * @param filename Name of file for model to be saved to "../models/filename"
         * 
         * Saves all model data to filename, including (in order):
         *  Number of layers
         *  For each layer (in order of first hidden to output):
         *      Layer input size
         *      Layer output size
         *      Layer activation function type
         *      Layer initialization type
         *      Weights Matrix num rows
         *      Weights Matrix num cols
         *      Each element of Weights Matrix
         *      Biases Matrix num rows
         *      Biases Matrix num cols
         *      Each element of Biases Matrix
         */
        void saveModel(const std::string& filename) const;

        /**
         * Load Model Function
         * @param filename Name of file for model to be loaded from "../models/filename"
         * 
         * Loads all model data from filename, assuming file is already
         * saved using saveModel()
         */
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
        void backward(const Matrix& y_true);

        /**
         * One-hot encoding function (We'll see if I have to use this)
         * (Originally planning to use this in get_accuracy but I realized it was an 
         * inefficient implementation, however maybe I'll need it again)
         * 
         * @param predictions Matrix of size (10, m) with softmaxxed vectors
         * @return Matrix of size (10, m) one-hot encoded
         * 
         * Basically takes a (10, m) matrix, with each column representing a probability vector
         * and one-hot encodes it
         */
        Matrix onehot(const Matrix& predictions);

        std::vector<Layer> layers;
        size_t inputSize;
        double learningRate;
        Matrix last_output;

        Activations::ActivationType actType;
        InitType initType;
        Loss::LossType lossType;
};


#endif // NETWORK_H
