/** 
 * At this stage, we need to:
 * 1. Build the Model
 *  Store layers in order std::vector<Layer>
 *  Provide a way to add layers
 * 
 * 2. Forward pass
 *  Input -> Each Layer -> Activation -> Output
 *  Return final output
 * 
 * 3. Backward pass
 *  Compute output error (loss gradient)
 *  Backpropagate through layers in reverse
 *  Update weights/biases using learning rate
 * 
 * 4. Training loop
 *  Iterate over epochs
 *  For each batch/sample
 *      Forward pass
 *      Compute loss
 *      Backpropagation
 *      Update weights/biases
 *  Track accuracy or loss for reporting
 * 
 * 5. Evaluation
 *  Run forward pass on test/validation set
 *  Compute accuracy/loss
 * 
 * 6. Model persistence
 *  Save weights/biases to file
 *  Load weights/biases from file
 */

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
                Loss::LossType lossType) : inputSize(input_size), learningRate(learning_rate), actType(actType), initType(initType), lossType(lossType) {}

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