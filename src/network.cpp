#include "network.h"
#include "loss.h"
#include <iostream>
#include <algorithm>
#include <random>


// Empty/Default Constructor
Network::Network(    
) : networkInputSize(784),
    learningRate(0.1),
    decayRate(0.99),
    networkActType(Activations::ActivationType::RELU),
    networkInitType(InitType::HE),
    networkLossType(Loss::LossType::CROSS_ENTROPY)
{
    addLayer(128);
    addLayer(64);
    addLayer(10, Activations::ActivationType::SOFTMAX, InitType::XAVIER, true);
    isCompiled = true;
}

// Non-empty Constructor
Network::Network(
    size_t input_size,
    double learning_rate,
    Activations::ActivationType act_type,
    InitType init_type,
    Loss::LossType loss_type
) : networkInputSize(input_size),
    learningRate(learning_rate),
    decayRate(0.99),
    networkActType(act_type),
    networkInitType(init_type),
    networkLossType(loss_type),
    batchSize(0),
    isCompiled(false)
{
    if (learning_rate <= 0.0) {
        throw std::invalid_argument("Learning Rate must be positive");
    }
    if (input_size <= 0) {
        throw std::invalid_argument("Network Input Size cannot be zero");
    }
}


// Nested Layer Class Implementation
Network::Layer::Layer(
    size_t input_size,
    size_t output_size,
    Activations::ActivationType act_type,
    InitType init_type
) : inputSize(input_size),
    outputSize(output_size),
    actType(act_type),
    initType(init_type)
{
    initialize();
    biases.fill(0.0);
}

Network::Layer::Layer(
    size_t input_size,
    size_t output_size,
    Activations::ActivationType act_type,
    InitType init_type,
    bool is_last_layer
) : inputSize(input_size),
    outputSize(output_size),
    actType(act_type),
    initType(init_type),
    isLastLayer(is_last_layer)
{
    initialize();
    biases.fill(0.0);
}

void Network::Layer::initialize() {
    switch(initType) {
        case InitType::RANDOM: { weights.randomize(); break; }
        case InitType::HE: { weights.heInit(); break; }
        case InitType::XAVIER: { weights.xavierInit(); break; }
        case InitType::NONE: { weights.fill(1.0); break; }
        default: throw std::invalid_argument("This shouldn't be called...");
    }
}

void Network::Layer::updateParams(const Matrix& dWeights, const Matrix& dbiases, double learning_rate) {
    weights -= learning_rate * dWeights;
    biases -= learning_rate * dbiases;
}

Matrix Network::Layer::forward(const Matrix& X) {
    input = X;
    Matrix Z = (weights * X) + biases;
    Matrix A = Activations::activate(Z, actType);
    preActivation = Z;
    output = A;
    return output;
}

Matrix Network::Layer::backward(const Matrix& dA, size_t batch_size, double learning_rate) {
    Matrix dZ(outputSize, batch_size);
    Matrix dWeights(outputSize, inputSize);
    Matrix dbiases(outputSize, 1);
    Matrix dA_return;

    if (isLastLayer) {
        dZ = dA;
    } else {
        Matrix sigma_prime;
        sigma_prime = Activations::deriv_activate(preActivation, actType);

        dZ = dA.hadamard(sigma_prime);
    }

    dWeights = (1.0 / batch_size) * (dZ * input.transpose());
    for (size_t i = 0; i < batch_size; ++i) {
        dbiases += dZ.getCol(i);
    }
    dbiases = dbiases / batch_size;

    dA_return = weights.transpose() * dZ;
    updateParams(dWeights, dbiases, learning_rate);

    return dA_return;
}


// Network Class Implementation
Matrix Network::forward(const Matrix& X) {
    if (!isCompiled) {
        if (layers.empty()) {
            throw std::runtime_error("Must have hidden layers to run forward pass");
        }

        size_t last_input_size = layers.back().getOutputSize();
        layers.emplace_back(last_input_size, 10, Activations::ActivationType::SOFTMAX, InitType::XAVIER);
        isCompiled = true;
    }

    if (X.getRows() != networkInputSize) {
        std::cout << "Network Input Size: " << networkInputSize << std::endl;
        std::cout << "Input Variable Size: " << X.getRows() << std::endl;
        throw std::invalid_argument("Network input size variable not equal to size of input");
    }

    Matrix A = X;
    for (auto& layer : layers) {
        A = layer.forward(A);
    }

    lastOutput = A;
    return lastOutput;
}

void Network::backward(const Matrix& y_true) {
    Matrix dA(y_true.getRows(), y_true.getCols());
    batchSize = y_true.getCols();

    Activations::ActivationType outputLayerActType;
    outputLayerActType = layers.back().getActivationType();

    switch(networkLossType) {
        case Loss::LossType::CROSS_ENTROPY: {
            if (outputLayerActType == Activations::ActivationType::SOFTMAX) {
                dA = (1.0 / batchSize) * (lastOutput - y_true);
            } else {
                Matrix dividend = y_true % lastOutput;
                Matrix psi_prime = Activations::deriv_activate(layers.back().getZ(), outputLayerActType);
                dA = - (1.0 / batchSize) * dividend.hadamard(psi_prime);
            }

            break;
        }

        case Loss::LossType::MSE: {
            if (outputLayerActType == Activations::ActivationType::SOFTMAX) {
                // WHOOOOO
                
            } else {
                Matrix difference = lastOutput - y_true;
                Matrix psi_prime = Activations::deriv_activate(layers.back().getZ(), outputLayerActType);
                dA = (1.0 / batchSize) * difference.hadamard(psi_prime); 
            }

            break;
        }
    }

    for (int i = static_cast<int>(layers.size()) - 1; i >= 0; --i) {
        dA = layers[i].backward(dA, batchSize, learningRate);
    }
}

void Network::addLayer(size_t neurons) {
    if (neurons == 0) {
        throw std::invalid_argument("Layer must have at least one neuron");
    }

    size_t input_dim = layers.empty() ? networkInputSize : layers.back().getOutputSize();
    layers.emplace_back(input_dim, neurons, networkActType, networkInitType);
}

void Network::addLayer(size_t neurons, Activations::ActivationType actType, InitType initType) {
    if (neurons == 0) {
        throw std::invalid_argument("Layer must have at least one neuron");
    }

    size_t input_dim = layers.empty() ? networkInputSize : layers.back().getOutputSize();
    layers.emplace_back(input_dim, neurons, actType, initType);
}

void Network::addLayer(size_t neurons, Activations::ActivationType actType, InitType initType, bool is_last_layer) {
    if (neurons == 0) {
        throw std::invalid_argument("Layer must have at least one neuron");
    }

    size_t input_dim = layers.empty() ? networkInputSize : layers.back().getOutputSize();
    layers.emplace_back(input_dim, neurons, actType, initType, is_last_layer);
}










// Everything Below needs to be reviewed and corrected
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

void Network::train(const Matrix& X, const Matrix& y,
                    size_t epochs, size_t batch_size, bool shuffle,
                    const Matrix& X_val, const Matrix& y_val) {
    
    if (X.getCols() != y.getCols()) {
        throw std::invalid_argument("Image and Label sample sizes must match");
    }

    size_t m = X.getCols();
    std::vector<size_t> indices(m);
    std::iota(indices.begin(), indices.end(), 0);

    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        double eta = learningRate * pow(decayRate, epoch);
        if (shuffle) {
            static std::random_device rd;
            static std::mt19937 g(rd());
            std::shuffle(indices.begin(), indices.end(), g);
        }

        for (size_t i = 0; i < m; i += batch_size) {
            size_t end = std::min(i + batch_size, m);

            Matrix X_batch = X.sliceCols(indices, i, end);
            Matrix y_batch = y.sliceCols(indices, i, end);

            Matrix predictions = forward(X_batch);
            backward(y_batch);
        }

        Matrix train_predictions = forward(X);
        double train_accuracy = get_accuracy(train_predictions, y);
        double train_loss = Loss::compute(y, train_predictions, networkLossType);

        double val_acc = -1;
        if (X_val.getCols() > 0) {
            Matrix val_predictions = forward(X_val);
            val_acc = get_accuracy(val_predictions, y_val);
        }

        if ((epoch + 1) % 10 == 0 || epoch + 1 == epochs - 1) {
            std::cout << "Epoch [" << epoch + 1 << "/" << epochs << "], "
                    << "Training Accuracy: " << train_accuracy << ", "
                    << "Validation Accuracy: " << (val_acc >= 0 ? std::to_string(val_acc) : "N/A")
                    << "Learning Rate: " << eta << ", "
                    << "Loss: " << train_loss << std::endl;
        }
    }    
}

void Network::saveModel(const std::string& filename) const {
    std::ofstream out(filename);
    if (!out.is_open()) throw std::runtime_error("Could not open file for saving model");

    out << layers.size() << std::endl;
    for (const auto& layer : layers) {
        out << layer.getInputSize() << " "
            << layer.getOutputSize() << " "
            << static_cast<int>(layer.getActivationType()) << " "
            << static_cast<int>(layer.getInitType()) << std::endl;

        const Matrix& W = layer.getWeights();
        out << W.getRows() << " " << W.getCols() << std::endl;
        for (size_t i = 0; i < W.getRows(); ++i) {
            for (size_t j = 0; j < W.getCols(); ++j) {
                out << W(i, j) << " ";
            }
            out << std::endl;
        }

        const Matrix& b = layer.getBiases();
        out << b.getRows() << " " << b.getCols() << std::endl;
        for (size_t i = 0; i < b.getRows(); ++i) {
            for (size_t j = 0; j < b.getCols(); ++j) {
                out << b(i, j) << " ";
            }
            out << std::endl;
        }
    }

    out.close();
}

void Network::loadModel(const std::string& filename) {
    std::ifstream in(filename);
    if (!in.is_open()) throw std::runtime_error("Could not open file for loading model");

    size_t num_layers;
    in >> num_layers;

    layers.clear();
    for (size_t l = 0; l < num_layers; ++l) {
        size_t in_size, out_size;
        int act, init;
        in >> in_size >> out_size >> act >> init;

        Layer layer(in_size, out_size,
                    static_cast<Activations::ActivationType>(act),
                    static_cast<InitType>(init));
        
        size_t w_rows, w_cols;
        in >> w_rows >> w_cols;
        Matrix W(w_rows, w_cols);
        for (size_t i = 0; i < w_rows; ++i) {
            for (size_t j = 0; j < w_cols; ++j) {
                in >> W(i, j);
            }
        }
        layer.setWeights(W);

        size_t b_rows, b_cols;
        in >> b_rows >> b_cols;
        Matrix b(b_rows, b_cols);
        for (size_t i = 0; i < b_rows; ++i) {
            for (size_t j = 0; j < b_cols; ++j) {
                in >> b(i, j);
            }
        }
        layer.setBiases(b);

        layers.push_back(layer);
    }

    in.close();
}