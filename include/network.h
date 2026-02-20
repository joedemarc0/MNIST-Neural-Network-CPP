/**
 * Neural Network Class Implementation. Represents fully configured neural network, ready to train,
 * predict, and evaluate data. Options to save and load models into /models/ folder
 * 
 * Contains Nested Layer class - Layer class was originally separated
 * Empty Constructor for Network creates Neural Network of the form:
 * * 784 -> 128 -> 64 -> 10
 * * RELU and HE Initialization for both hidden layers
 * * SOFTMAX and XAVIER Initialization on output layer
 * * CROSS ENTROPY Loss format
 * 
 * However the non-empty constructor allows for experimenting with different initializations, activation functions,
 * number of types of layers, etc.
 */

#ifndef NETWORK_H
#define NETWORK_H

#include "matrix.h"
#include "activation.h"
#include "init.h"
#include <vector>
#include <string>
#include <utility>
#include <fstream>
#include <stdexcept>

struct Sample {
    Matrix X;
    Matrix y;
};

class Network {
    private:
        // Nested Layer Class
        class Layer {
            private:
                size_t inputSize;
                size_t outputSize;

                Matrix weights;
                Matrix biases;
                Matrix input;
                Matrix output;
                Matrix preActivation;

                Activations::ActivationType actType;
                InitType initType;

                void initialize();
                void updateParams(const Matrix& dWeights, const Matrix& dbiases, double learning_rate);
            
            public:
                Layer(
                    size_t input_size,
                    size_t output_size,
                    Activations::ActivationType act_type,
                    InitType init_type
                );

                Matrix forward(const Matrix& X);
                Matrix backward(const Matrix& dA, size_t batch_size, double learning_rate);

                // Getters and Setters
                Matrix getWeights() const { return weights; }
                Matrix getBiases() const { return biases; }
                Matrix getZ() const { return preActivation; }
                Activations::ActivationType getActivationType() const { return actType; }
                InitType getInitType() const { return initType; }
                size_t getInputSize() const { return inputSize; }
                size_t getOutputSize() const { return outputSize; }

                void setWeights(const Matrix& W) { weights = W; }
                void setBiases(const Matrix& b) { biases = b; }
        };

        bool isCompiled = false;
        std::vector<Layer> layers;
        size_t networkInputSize;
        size_t batchSize;                       // This might need to change
        const double learningRate;
        const double decayRate;
        Matrix lastOutput;                      // As well as this

        Activations::ActivationType networkActType;
        InitType networkInitType;

        void addOutputLayer();

        Matrix forward(const Matrix& X);
        void backward(const Matrix& y_true, double learning_rate);
        Matrix onehot(const Matrix& predictions);

        std::vector<Sample> getBatches(
            const Matrix& X, const Matrix& y,
            size_t batch_size, bool shuffle
        );

        size_t getCorrectCount(const Matrix& predictions, const Matrix& y_true) const;
    
    public:
        Network();
        Network(
            size_t input_size,
            double learning_rate,
            Activations::ActivationType act_type,
            InitType init_type
        );

        // Getters
        std::vector<Layer> getLayers() const { return layers; }
        double getLearningRate() const { return learningRate; }
        double getDecayRate() const { return decayRate; }
        Activations::ActivationType getNetworkActType() const { return networkActType; }
        InitType getNetworkInitType() const { return networkInitType; }

        void addLayer(size_t neurons);
        void addLayer(
            size_t neurons,
            Activations::ActivationType actType,
            InitType initType
        );

        void compile();

        // train() needs to be reviewed
        void train(
            const Matrix& X, const Matrix& y,
            size_t epochs, size_t batch_size, bool shuffle=true,
            const Matrix& X_val=Matrix(), const Matrix& y_val=Matrix(),
            bool streamline = true
        );
        
        double get_accuracy(const Matrix& predictions, const Matrix& y_true) const;
        Matrix predict(const Matrix& X);
        double evaluate(const Matrix& X, const Matrix& y_true);
        double computeLoss(const Matrix& predictions, const Matrix& y_true) const;

        // These two functions need to be reviewed
        void saveModel(const std::string& filename) const;
        void loadModel(const std::string& filename);
};


#endif // NETWORK_H
