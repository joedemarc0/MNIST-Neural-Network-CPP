#include "network.h"
#include <iostream>
#include <algorithm>
#include <random>


// Empty/Default Network Constructor
Network::Network(
) : networkInputSize(784),
    learningRate(0.1),
    decayRate(0.99),
    networkActType(Activations::ActivationType::RELU),
    networkInitType(InitType::HE)
{
    addLayer(128);
    addLayer(64);
    compile();
}

// Non-empty Constructor
Network::Network(
    size_t input_size,
    double learning_rate,
    Activations::ActivationType act_type,
    InitType init_type
) : isCompiled(false),
    networkInputSize(input_size),
    learningRate(learning_rate),
    decayRate(0.99),
    networkActType(act_type),
    networkInitType(init_type)
{
    if (learning_rate <= 0.0) {
        throw std::invalid_argument("Learning Rate must be Positive");
    } else if (input_size == 0) {
        throw std::invalid_argument("Network Input Size must be Nonzero");
    } else if (act_type == Activations::ActivationType::SOFTMAX) {
        throw std::invalid_argument("Invalid Hidden Layer Activation Function Selection");
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
    weights = Matrix(outputSize, inputSize);
    biases = Matrix(outputSize, 1);
    initialize();
}

// Layer Class Private Functions
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

// Layer Class Public Functions
Matrix Network::Layer::forward(const Matrix& X) {
    if (X.getRows() != inputSize) {
        throw std::invalid_argument(
            "Dimension Mismatch: Forwarding matrix X has input size: " + std::to_string(X.getRows()) +
            ", Layer input size is: " + std::to_string(inputSize)
        );
    }

    input = X;
    Matrix Z = (weights * X) + biases;
    Matrix A = Activations::activate(Z, actType);
    preActivation = Z;
    output = A;
    return output;
}

Matrix Network::Layer::backward(const Matrix& dA, size_t batch_size, double learning_rate) {
    Matrix dZ;
    Matrix dbiases(outputSize, 1);

    if (actType == Activations::ActivationType::SOFTMAX) {
        dZ = dA;
    } else {
        Matrix sigma_prime;
        sigma_prime = Activations::deriv_activate(preActivation, actType);

        dZ = dA.hadamard(sigma_prime);
    }

    Matrix dWeights = (1.0 / batch_size) * (dZ * input.transpose());
    dbiases = dZ.sumCols();
    dbiases /= batch_size;

    Matrix dA_return = weights.transpose() * dZ;
    updateParams(dWeights, dbiases, learning_rate);

    return dA_return;
}


// Network Class Implementation
// Network Class Private Functions
void Network::addOutputLayer() {
    if (layers.empty()) {
        throw std::runtime_error("Network must have at least one hidden layer");
    }

    size_t input_dim = layers.back().getOutputSize();
    size_t output_dim = 10;
    layers.emplace_back(input_dim, output_dim, Activations::ActivationType::SOFTMAX, InitType::XAVIER);
}

Matrix Network::forward(const Matrix& X) {
    if (!isCompiled) {
        throw std::runtime_error("Network must be compiled");
    }

    if (X.getRows() != networkInputSize) {
        throw std::invalid_argument(
            "Network input size variable not equal to size of input - network input size: " +
            std::to_string(networkInputSize) + ", Input Variable Size: " + std::to_string(X.getRows())
        );
    }

    Matrix A = X;
    for (auto& layer : layers) {
        A = layer.forward(A);
    }

    lastOutput = A;
    return lastOutput;
}

void Network::backward(const Matrix& y_true, double learning_rate) {
    Matrix dA(y_true.getRows(), y_true.getCols());
    size_t batch_size = y_true.getCols();

    dA = lastOutput - y_true;

    for (int i = static_cast<int>(layers.size()) - 1; i >= 0; --i) {
        dA = layers[i].backward(dA, batch_size, learning_rate);
    }
}

Matrix Network::toOneHot(const MNISTDataset& dataset) const {
    Matrix result(dataset.num_classes, dataset.num_samples);
    for (size_t col = 0; col < dataset.num_samples; ++col) result(dataset.labels[col], col) = 1.0;
    return result;
}

Matrix Network::toOneHot(const std::vector<uint8_t>& labels, size_t num_classes) const {
    Matrix result(num_classes, labels.size());
    for (size_t col = 0; col < labels.size(); ++col) result(labels[col], col) = 1.0;
    return result;
}

size_t Network::computeCorrectCount(const Matrix& predictions, const std::vector<uint8_t>& labels, size_t num_classes) const {
    if (predictions.getCols() != labels.size()) {
        throw std::invalid_argument("Number of predictions not equal to number of labels");
    } else if (predictions.getRows() != num_classes) {
        throw std::invalid_argument("Predictions matrix does not match number of classes");
    }

    size_t batch_size = predictions.getCols();
    size_t count = 0;

    for (size_t sample = 0; sample < batch_size; ++sample) {
        double max_val = predictions(0, sample);
        size_t max_index = 0;

        for (size_t row = 1; row < num_classes; ++row) {
            if (predictions(row, sample) > max_val) {
                max_val = predictions(row, sample);
                max_index = row;
            }
        }

        if (max_index == labels[sample]) {
            count += 1;
        }
    }

    return count;
}

size_t Network::computeCorrectCount(const Matrix& predictions, const Matrix& y_true) const {
    if (!Matrix::matchDim(predictions, y_true)) {
        throw std::invalid_argument("Predictions matrix and labels matrix must have matching dimensions");
    }

    size_t batch_size = predictions.getCols();
    const double epsilon = 1e-6;
    size_t count = 0;

    for (size_t sample = 0; sample < batch_size; ++sample) {
        double max_val = predictions(0, sample);
        size_t max_index = 0;

        for (size_t row = 1; row < predictions.getRows(); ++row) {
            if (predictions(row, sample) > max_val) {
                max_val = predictions(row, sample);
                max_index = row;
            }
        }

        if (std::abs(y_true(max_index, sample) - 1.0) < epsilon) {
            count += 1;
        }
    }

    return count;
}

Matrix Network::sliceCols(const Matrix& X, const std::vector<size_t>& indices) const {
    size_t batch_size = indices.size();
    Matrix result(X.getRows(), batch_size);

    for (size_t i = 0; i < batch_size; ++i) {
        result.setCol(i, X.getCol(indices[i]));
    }

    return result;
}

std::vector<uint8_t> Network::sliceCols(const std::vector<uint8_t>& y, const std::vector<size_t>& indices) const {
    size_t batch_size = indices.size();
    std::vector<uint8_t> result(batch_size);

    for (size_t i = 0; i < batch_size; ++i) {
        result[i] = y[indices[i]];
    }

    return result;
}

std::vector<Sample> Network::createBatches(const Matrix& X, const std::vector<uint8_t> labels, size_t batch_size, bool shuffle) const {
    size_t training_size = X.getCols();
    std::vector<size_t> indices(training_size);
    std::iota(indices.begin(), indices.end(), 0);

    if (shuffle) {
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(indices.begin(), indices.end(), g);
    }

    std::vector<Sample> batches;
    for (size_t start = 0; start < training_size; start += batch_size) {
        size_t end = std::min(start + batch_size, training_size);

        std::vector<size_t> sliced_indices(
            indices.begin() + start,
            indices.begin() + end
        );

        Matrix X_batch = X.sliceCols(sliced_indices);
        std::vector<uint8_t> batch_labels = sliceCols(labels, sliced_indices);

        batches.emplace_back(X_batch, batch_labels);
    }

    return batches;
}

std::vector<Sample> Network::createBatches(const MNISTDataset& dataset, size_t batch_size, bool shuffle) const {
    return createBatches(dataset.X, dataset.labels, batch_size, shuffle);
}


// Network Class Public Functions
void Network::addLayer(size_t neurons) {
    if (neurons == 0) {
        throw std::invalid_argument("Number of neurons must be nonzero");
    } else if (isCompiled) {
        throw std::runtime_error("Cannot add layers once network is Compiled");
    }

    size_t input_dim = layers.empty() ? networkInputSize : layers.back().getOutputSize();
    layers.emplace_back(input_dim, neurons, networkActType, networkInitType);
}

void Network::addLayer(size_t neurons, Activations::ActivationType act_type, InitType init_type) {
    if (neurons == 0) {
        throw std::invalid_argument("Number of neurons must be nonzero");
    } else if (act_type == Activations::ActivationType::SOFTMAX) {
        throw std::invalid_argument("Hidden layer cannot have SOFTMAX activation");
    } else if (isCompiled) {
        throw std::runtime_error("Cannot add layers once network is compiled");
    }

    size_t input_dim = layers.empty() ? networkInputSize : layers.back().getOutputSize();
    layers.emplace_back(input_dim, neurons, act_type, init_type);
}

void Network::compile() {
    if (isCompiled) {
        throw std::runtime_error("Network is already compiled");
    } else if (layers.empty()) {
        throw std::runtime_error("Network cannot be compiled with no hidden layers");
    }

    size_t expectedInputSize = networkInputSize;
    for (size_t i = 0; i < layers.size(); ++i) {
        auto act = layers[i].getActivationType();

        if (act == Activations::ActivationType::SOFTMAX) {
            throw std::runtime_error(
                "Invalid Activation Function Type at Layer " + std::to_string(i) +
                ", Activation Function: " + std::to_string(static_cast<int>(layers[i].getActivationType()))
            );
        }

        if (layers[i].getInputSize() != expectedInputSize) {
            throw std::runtime_error(
                "Dimension Mismatch at Layer " + std::to_string(i) +
                ", Expected: " + std::to_string(expectedInputSize) +
                ", Got: " + std::to_string(layers[i].getInputSize()) + "."
            );
        }

        expectedInputSize = layers[i].getOutputSize();
    }

    addOutputLayer();
    isCompiled = true;
}

void Network::train(
    const Matrix& X, const std::vector<uint8_t>& labels,
    const Matrix& X_val, const std::vector<uint8_t>& labels_val,
    size_t epochs, size_t batch_size, size_t num_classes,
    bool shuffle, bool streamline
) {
    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        double eta = learningRate * pow(decayRate, epoch);
        size_t training_size = X.getCols();

        double epoch_loss = 0.0;
        size_t epoch_corr = 0;
        size_t total_samples = 0;

        if (streamline) {
            std::vector<size_t> indices(training_size);
            std::iota(indices.begin(), indices.end(), 0);

            if (shuffle) {
                std::random_device rd;
                std::mt19937 g(rd());
                std::shuffle(indices.begin(), indices.end(), g);
            }

            for (size_t start = 0; start < training_size; start += batch_size) {
                size_t end = std::min(start + batch_size, training_size);

                std::vector<size_t> sliced_indices(
                    indices.begin() + start,
                    indices.begin() + end
                );

                Matrix X_batch = X.sliceCols(sliced_indices);
                std::vector<uint8_t> batch_labels = sliceCols(labels, sliced_indices);
                Matrix y_batch = toOneHot(batch_labels, num_classes);

                Matrix predictions = forward(X_batch);
                backward(y_batch, eta);

                epoch_loss += computeLoss(predictions, y_batch) * X_batch.getCols();
                epoch_corr += computeCorrectCount(predictions, batch_labels, num_classes);
                total_samples += X_batch.getCols();
            }
        } else {
            /**
             * Realistically this method is going to be much much slower because you are allocating memory for previous
             * and future batches - quicker just to compute and move on
             * 
             * However I am incredibly impressed with myself for implementing this because this was a tough part of this project
             * Therefore I am not going to delete it. Bite me.
             */
            std::vector<Sample> batches = createBatches(X, labels, batch_size, shuffle);
            for (auto& batch : batches) {
                Matrix predictions = forward(batch.X);
                std::visit([&](auto& labels) {
                    Matrix y_true;
                    if constexpr (std::is_same_v<std::decay_t<decltype(labels)>, std::vector<uint8_t>>) {
                        y_true = toOneHot(labels, num_classes);
                    } else {
                        y_true = labels;
                    }

                    backward(y_true, eta);
                    epoch_loss += computeLoss(predictions, y_true) * batch.X.getCols();
                    epoch_corr += computeCorrectCount(predictions, y_true);

                }, batch.y);

                total_samples += batch.X.getCols();
            }
        }

        double train_accuracy = static_cast<double>(epoch_corr) / total_samples;
        double train_loss = epoch_loss / total_samples;

        double val_acc = -1.0;
        if (X_val.getCols() > 0) {
            Matrix val_predictions = forward(X_val);
            val_acc = computeAccuracy(val_predictions, labels_val, num_classes);
        }

        if ((epoch + 1) % 1 == 0 || epoch + 1 == epochs - 1) {
            std::cout << "Epoch [" << epoch + 1 << "/" << epochs << "]:" << std::endl;
            std::cout << "Total Samples Passed: " << total_samples << "/" << training_size << std::endl;
            std::cout << "Training Accuracy: " << train_accuracy << std::endl;
            std::cout << "Validation Accuracy: " << (val_acc >= 0.0 ? std::to_string(val_acc) : "N/A") << std::endl;
            std::cout << "Learning Rate: " << eta << std::endl;
            std::cout << "Loss: " << train_loss << std::endl;
            std::cout << std::endl;
        }
    }
}

double Network::computeAccuracy(const Matrix& predictions, const std::vector<uint8_t>& labels, size_t num_classes) const {
    size_t batch_size = predictions.getCols();
    return static_cast<double>(computeCorrectCount(predictions, labels, num_classes)) / batch_size;
}

double Network::computeAccuracy(const Matrix& predictions, const Matrix& y_true) const {
    size_t batch_size = predictions.getCols();
    return static_cast<double>(computeCorrectCount(predictions, y_true)) / batch_size;
}

double Network::computeLoss(const Matrix& predictions, const Matrix& y_true) const {
    if (!Matrix::matchDim(predictions, y_true)) {
        throw std::invalid_argument("How could this happen?!?!");
    }

    size_t batch_size = predictions.getCols();
    double loss = 0.0;
    const double epsilon = 1e-12;

    for (size_t sample = 0; sample < batch_size; ++sample) {
        for (size_t row = 0; row < predictions.getRows(); ++row) {
            double y_val = y_true(row, sample);

            if (std::abs(y_val - 1.0) < 1e-9) {
                double p = predictions(row, sample);
                loss -= std::log(p + epsilon);
            }
        }
    }

    return loss / batch_size;
}

double Network::computeLoss(const Matrix& predictions, const std::vector<uint8_t>& labels, size_t num_classes) const {
    Matrix y_true = toOneHot(labels, num_classes);
    return computeLoss(predictions, y_true);
}






// Everything Below needs to be reviewed and corrected
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