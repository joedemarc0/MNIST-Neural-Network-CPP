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


class Network {
    public:
        enum class InitType { NONE, RANDOM, XAVIER, HE };

        Network(
            size_t input_size,
            double learning_rate,
            Activations::ActivationType actType,
            InitType initType,
            Loss::LossType lossType
        );

        void train(const Matrix& X, const Matrix& y,
                   size_t epochs,
                   size_t batch_size=0,
                   bool shuffle=true);

        Matrix predict(const Matrix& X) const;
        double evaluate(const Matrix& X, const Matrix& y) const;

        void saveModel(const std::string& filename) const;
        void loadModel(const std::string& filename);
    
    private:
        Matrix forward(const Matrix& X);
        void backward(const Matrix& dA);

        std::vector<Layer> layers;
        size_t inputSize;
        double learningRate;
        Matrix last_output;

        Activations::ActivationType actType;
        InitType initType;
        Loss::LossType lossType;
};


#endif // NETWORK_H
