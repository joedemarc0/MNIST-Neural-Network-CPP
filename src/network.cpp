#include "network.h"
#include <iostream>
#include <algorithm>
#include <random>


// Empty/Default Constructorl
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
    batchSize(0),
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
    biases.fill(0.0);
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
            "Dimension Mismatch: Forwarding Matrix X has input size: " + std::to_string(X.getRows()) +
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
Matrix Network::forward(const Matrix& X) {
    if (!isCompiled) {
        throw std::runtime_error("Network must be compiled");
    }

    if (X.getRows() != networkInputSize) {
        throw std::invalid_argument(
            "Network Input Size Variable not equal to size of Input - Network Input Size: " +
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
    batchSize = y_true.getCols();

    dA = lastOutput - y_true;

    for (int i = static_cast<int>(layers.size()) - 1; i >= 0; --i) {
        dA = layers[i].backward(dA, batchSize, learning_rate);
    }
}

void Network::addOutputLayer() {
    if (layers.empty()) {
        throw std::runtime_error("Network must have at least one Hidden Layer");
    }

    size_t input_dim = layers.back().getOutputSize();
    layers.emplace_back(input_dim, 10, Activations::ActivationType::SOFTMAX, InitType::XAVIER);
}

Matrix Network::onehot(const Matrix& predictions) {
    Matrix result(predictions.getRows(), predictions.getCols());

    for (size_t j = 0; j < predictions.getCols(); ++j) {
        size_t max_index = 0;
        double max_val = predictions(0, j);

        for (size_t i = 1; i < predictions.getRows(); ++i) {
            if (predictions(i, j) > max_val) {
                max_val = predictions(i, j);
                max_index = i;
            }
        }

        result(max_index, j) = 1.0;
    }

    return result;
}

std::vector<Sample> Network::getBatches(
    const Matrix& X, const Matrix& y,
    size_t batch_size, bool shuffle
) {
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
        Matrix y_batch = y.sliceCols(sliced_indices);

        batches.emplace_back(X_batch, y_batch);
    }

    return batches;
}

size_t Network::getCorrectCount(const Matrix& predictions, const Matrix& y_true) const {
    if (predictions.getRows() != y_true.getRows() || predictions.getCols() != y_true.getCols()) {
        throw std::invalid_argument("Predictions and labels matrices must have equal dimensions");
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


// Network Class Public Functions
void Network::addLayer(size_t neurons) {
    if (neurons == 0) {
        throw std::invalid_argument("Number of Neurons must be Nonzero");
    } else if (isCompiled) {
        throw std::runtime_error("Cannot add Layers once Network is Compiled");
    }

    size_t input_dim = layers.empty() ? networkInputSize : layers.back().getOutputSize();
    layers.emplace_back(input_dim, neurons, networkActType, networkInitType);
}

void Network::addLayer(size_t neurons, Activations::ActivationType actType, InitType initType) {
    if (neurons == 0) {
        throw std::invalid_argument("Number of neurons must be Nonzero");
    } else if (actType == Activations::ActivationType::SOFTMAX) {
        throw std::invalid_argument("Hidden Layer cannot have Softmax Activation Function");
    } else if (isCompiled) {
        throw std::runtime_error("Cannot add Layers once Network is Compiled");
    }

    size_t input_dim = layers.empty() ? networkInputSize : layers.back().getOutputSize();
    layers.emplace_back(input_dim, neurons, actType, initType);
}

void Network::compile() {
    if (isCompiled) {
        throw std::runtime_error("Network is already Compiled");
    } else if (layers.empty()) {
        throw std::runtime_error("Network cannot be Compiled with no Hidden Layers");
    }

    size_t expectedInputSize = networkInputSize;
    for (size_t i = 0; i < layers.size(); ++i) {
        auto act = layers[i].getActivationType();

        if (act != Activations::ActivationType::RELU && act != Activations::ActivationType::LEAKY_RELU) {
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
    const Matrix& X, const Matrix& y_true,
    size_t epochs, size_t batch_size, bool shuffle,
    const Matrix& X_val, const Matrix& y_val,
    bool streamline
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
                Matrix y_batch = y_true.sliceCols(sliced_indices);

                Matrix predictions = forward(X_batch);
                backward(y_batch, eta);

                epoch_loss += computeLoss(predictions, y_batch) * X_batch.getCols();
                epoch_corr += getCorrectCount(predictions, y_batch) * X_batch.getCols();
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
            std::vector<Sample> batches = getBatches(X, y_true, batch_size, shuffle);
            for (auto& batch : batches) {
                Matrix predictions = forward(batch.X);
                backward(batch.y, eta);
            }
        }

        double train_accuracy = static_cast<double>(epoch_corr) / total_samples;
        double train_loss = epoch_loss / total_samples;

        double val_acc = -1.0;
        if (X_val.getCols() > 0) {
            Matrix val_predictions = forward(X_val);
            val_acc = getAccuracy(val_predictions, y_val);
        }

        if ((epoch + 1) % 10 == 0 || epoch + 1 == epochs - 1) {
            std::cout << "Epoch [" << epoch + 1 << "/" << epochs << "]:" << std::endl;
            std::cout << "Training Accuracy: " << train_accuracy << std::endl;
            std::cout << "Validation Accuracy: " << (val_acc >= 0.0 ? std::to_string(val_acc) : "N/A") << std::endl;
            std::cout << "Learning Rate: " << eta << std::endl;
            std::cout << "Loss: " << train_loss << std::endl;
            std::cout << std::endl;
        }
    }
}

Matrix Network::predict(const Matrix& X) {
    return forward(X);
}

double Network::getAccuracy(const Matrix& predictions, const Matrix& y_true) const {
    size_t batch_size = predictions.getCols();
    return static_cast<double>(getCorrectCount(predictions, y_true)) / batch_size;
}

double Network::evaluate(const Matrix& X, const Matrix& y_true) {
    Matrix predictions = predict(X);
    return getAccuracy(predictions, y_true);
}

double Network::computeLoss(const Matrix& predictions, const Matrix& y_true) const {
    if (predictions.getRows() != y_true.getRows() || predictions.getCols() != y_true.getCols()) {
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