#include "network.h"
#include "loss.h"
#include <iostream>
#include <algorithm>
#include <random>


// Constructor
Network::Network(size_t input_size,
                double learning_rate,
                Activations::ActivationType actType,
                InitType initType,
                Loss::LossType lossType
            ) : inputSize(input_size),
                learningRate(learning_rate),
                actType(actType),
                initType(initType),
                lossType(lossType) {}


void Network::addLayer(size_t neurons) {
    if (neurons == 0) {
        throw std::invalid_argument("Layer must have at least one neuron");
    }

    size_t input_dim = layers.empty() ? inputSize : layers.back().getOutputSize();
    layers.emplace_back(input_dim, neurons, actType, initType);
}

void Network::addLayer(size_t neurons, Activations::ActivationType actType, InitType initType) {
    if (neurons == 0) {
        throw std::invalid_argument("Layer must have at least one neuron");
    }

    size_t input_dim = layers.empty() ? inputSize : layers.back().getOutputSize();
    layers.emplace_back(input_dim, neurons, actType, initType);
}

Matrix Network::forward(const Matrix& X) {
    Matrix A = X;
    for (auto& layer : layers) {
        A = layer.forward(A);
    }
    last_output = A;
    return A;
}

void Network::backward(const Matrix& y_true) {
    Matrix dA = Loss::derivative(y_true, last_output, lossType);

    for (int i = layers.size() - 1; i >= 0; --i) {
        dA = layers[i].backward(dA, learningRate);
    }
}

Matrix Network::onehot(const Matrix& predictions) {
    if (predictions.getRows() != 10) {
        throw std::invalid_argument("Predictions Matrix must be 10xm");
    }

    size_t rows = predictions.getRows();
    size_t cols = predictions.getCols();
    Matrix output(rows, cols);

    for (size_t j = 0; j < cols; ++j) {
        size_t max_index = 0;
        double max_val = predictions(0, j);

        for (size_t i = 0; i < rows; ++i) {
            if (predictions(i, j) > max_val) {
                max_val = predictions(i, j);
                max_index = i;
            }
        }

        output(max_index, j) = 1.0;
    }

    return output;
}

double Network::get_accuracy(const Matrix& predictions, const Matrix& y) const {
    if (predictions.getRows() != y.getRows() || predictions.getCols() != y.getCols()) {
        throw std::invalid_argument("Predictions Matrix and Labels Matrix must have same dimensions");
    }

    size_t rows = y.getRows();
    size_t cols = y.getCols();
    size_t count = 0;
    
    for (size_t j = 0; j < cols; ++j) {
        size_t pred_index = 0;
        double max_val = predictions(0, j);

        for (size_t i = 0; i < rows; ++i) {
            if (predictions(i, j) > max_val) {
                max_val = predictions(i, j);
                pred_index = i;
            }
        }

        size_t y_index = 0;
        for (size_t i = 0; i < rows; ++i) {
            if (y(i, j) == 1.0) {
                y_index = i;
                break;
            }
        }

        if (pred_index == y_index) {
            count++;
        }
    }
    
    double accuracy = static_cast<double>(count) / cols;
    return accuracy;
}

Matrix Network::predict(const Matrix& X) const {
    return const_cast<Network*>(this)->forward(X);
}

double Network::evaluate(const Matrix& X, const Matrix& y) const {
    Matrix predictions = predict(X);
    return get_accuracy(predictions, y);
}

void Network::train(const Matrix& X, const Matrix& y, size_t epochs, size_t batch_size, bool shuffle) {
    if (X.getCols() != y.getCols()) {
        throw std::invalid_argument("Image and Label sample sizes must match");
    }

    size_t m = X.getCols();
    std::vector<size_t> indices(m);
    std::iota(indices.begin(), indices.end(), 0);

    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        if (shuffle) {
            static std::random_device rd;
            static std::mt19937 g(rd());
            std::shuffle(indices.begin(), indices.end(), g);
        }

        Matrix X_epoch = X;
        Matrix y_epoch = y;

        Matrix predictions = forward(X_epoch);

        double loss = Loss::compute(y_epoch, predictions, lossType);

        backward(y_epoch);

        std::cout << "Epoch [" << epoch + 1 << "/" << epochs << "] - Loss: " << loss << std::endl;
    }    
}