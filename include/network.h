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
#include "data-loader.h"
#include <vector>
#include <string>
#include <utility>
#include <fstream>
#include <stdexcept>


class Network {
    private:
        class Layer {
            private:
                const size_t inputSize;
                const size_t outputSize;

                Matrix weights;
                Matrix biases;
                Matrix input;
                Matrix preActivation;

                const Activations::ActivationType actType;
                const InitType initType;

                void initialize();
                void updateParams(const Matrix& dWeights, const Matrix& dbiases, double learning_rate);

            public:
                Layer(
                    size_t input_size,
                    size_t output_size,
                    Activations::ActivationType act_type,
                    InitType init_type,
                    bool skip_init
                );

                Layer(
                    size_t input_size,
                    size_t output_size,
                    Activations::ActivationType act_type,
                    InitType init_type
                );

                Matrix forward(const Matrix& X);
                Matrix backward(const Matrix& dA, size_t batch_size, double learning_rate);

                const Matrix& getWeights() const { return weights; }
                const Matrix& getBiases() const { return biases; }
                const Matrix& getZ() const { return preActivation; }
                Activations::ActivationType getActivationType() const { return actType; }
                InitType getInitType() const { return initType; }
                size_t getInputSize() const { return inputSize; }
                size_t getOutputSize() const { return outputSize; }

                void setWeights(const Matrix& W) { weights = W; }
                void setBiases(const Matrix& b) { biases = b; }
        };

        bool isCompiled = false;
        std::vector<Layer> layers;
        const size_t networkInputSize;
        const size_t numClasses;
        const double learningRate;
        const double decayRate;
        Matrix lastOutput;
        
        const Activations::ActivationType networkActType;
        const InitType networkInitType;

        void addOutputLayer();

        Matrix forward(const Matrix& X);
        void backward(const Matrix& y_true, double learning_rate);
        Matrix toOneHot(const std::vector<uint8_t>& labels) const;
        size_t computeCorrectCount(const Matrix& predictions, const std::vector<uint8_t>& labels) const;
        std::vector<MNISTDataset> createBatches(const Matrix& X, const std::vector<uint8_t>& labels, size_t batch_size, bool shuffle) const;

    public:
        Network();
        Network(
            size_t input_size,
            size_t num_classes,
            double learning_rate,
            double decay_rate,
            Activations::ActivationType act_type,
            InitType init_type
        );

        // Getters
        const std::vector<Layer>& getLayers() const { return layers; }
        size_t getInputSize() const { return networkInputSize; }
        size_t getNumClasses() const { return numClasses; }
        double getLearningRate() const { return learningRate; }
        double getDecayRate() const { return decayRate; }
        Activations::ActivationType getNetworkActType() const { return networkActType; }
        InitType getNetworkInitType() const { return networkInitType; }
        bool checkCompiled() const { return isCompiled; }

        void addLayer(size_t neurons);
        void addLayer(
            size_t neurons,
            Activations::ActivationType act_type,
            InitType init_type
        );

        void compile();

        void train(
            const Matrix& X, const std::vector<uint8_t>& labels,
            const Matrix& X_val, const std::vector<uint8_t>& labels_val,
            size_t epochs, size_t batch_size,
            bool shuffle, bool streamline, bool verbose=true
        );

        void train(
            const MNISTDataset& dataset,
            const MNISTDataset& val_dataset,
            size_t epochs, size_t batch_size,
            bool shuffle, bool streamline, bool verbose=true
        ) {
            train(
                dataset.X, dataset.labels,
                val_dataset.X, val_dataset.labels,
                epochs, batch_size,
                shuffle, streamline, verbose
            );
        }

        void train(
            const MNISTDataset& dataset,
            size_t val_size, size_t epochs, size_t batch_size,
            bool shuffle, bool streamline, bool verbose=true
        ) {
            auto [train_set, val_set] = val_size > 0 ? MNISTLoader::split(dataset, val_size) : std::make_pair(dataset, MNISTDataset{});
            train(
                train_set.X, train_set.labels,
                val_set.X, val_set.labels,
                epochs, batch_size,
                shuffle, streamline, verbose
            );
        }

        double computeAccuracy(const Matrix& predictions, const std::vector<uint8_t>& labels) const;
        double evaluate(const Matrix& X, const std::vector<uint8_t>& labels);
        double evaluate(const MNISTDataset& dataset) { return evaluate(dataset.X, dataset.labels); }
        Matrix predict(const Matrix& X) { return forward(X); }
        double computeLoss(const Matrix& predictions, const Matrix& y_true) const;

        void saveModel(const std::string& filename) const;
        static Network loadModel(const std::string& filename);
};


#endif // NETWORK_H