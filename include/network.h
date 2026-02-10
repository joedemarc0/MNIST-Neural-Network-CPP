/**
 * Neural Network Class Implementation. Represents fully configured neural network, ready to train,
 * predict, and evaluate data. Options to save and load models into /models/ folder
 */

#ifndef NETWORK_H
#define NETWORK_H

#include "matrix.h"
#include "activation.h"
#include "loss.h"
#include "init.h"
#include <vector>
#include <string>
#include <utility>
#include <fstream>
#include <stdexcept>


class Network {
    private:
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
        size_t batchSize;
        double learningRate;
        double decayRate;
        Matrix lastOutput;

        Activations::ActivationType networkActType;
        InitType networkInitType;
        Loss::LossType networkLossType;

        Matrix forward(const Matrix& X);
        void backward(const Matrix& y_true);
        Matrix onehot(const Matrix& predictions);
    
    public:
        Network();
        Network(
            size_t input_size,
            double learning_rate,
            Activations::ActivationType act_type,
            InitType init_type,
            Loss::LossType loss_type
        );

        void addLayer(size_t neurons);
        void addLayer(size_t neurons,
                      Activations::ActivationType actType,
                      InitType initType);

        double get_accuracy(const Matrix& predictions, const Matrix& y) const;

        void train(const Matrix& X, const Matrix& y,
                   size_t epochs, size_t batch_size, bool shuffle=true,
                   const Matrix& X_val=Matrix(), const Matrix& y_val=Matrix());
        
        Matrix predict(const Matrix& X) const;
        double evaluate(const Matrix& X, const Matrix& y) const;
        void saveModel(const std::string& filename) const;
        void loadModel(const std::string& filename);
};


#endif // NETWORK_H
