#include <iostream>
#include <random>
#include <opencv2/opencv.hpp>
#include "data-loader.h"
#include "visualization.h"


int main() {
    try {
        std::cout << "=== MNISTLoader Test ===" << std::endl;

        MNISTLoader loader;

        std::cout << "Loading Training Data" << std::endl;
        MNISTDataset train_data = MNISTLoader::loadTrainingData();
        MNISTLoader::printDatasetInfo(train_data);

        if (!train_data.X.empty()) {
            const auto& first_sample = Sample(train_data.X.getCol(0), std::vector<uint8_t>(train_data.labels[0]));
            std::cout << "First Sample Details:" << std::endl;
            std::cout << "Label: " << first_sample.y[0] << std::endl;
            std::cout << "Number of pixels: " << first_sample.X.getRows();
        }

        std::cout << "Test Completed Successfully..." << std::endl;
    
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        std::cerr << "\nPossible Issues:" << std::endl;
        std::cerr << "1. Check that MNIST files are in ./data/ directory" << std::endl;
        std::cerr << "2. Verify filenames match exactly:" << std::endl;
        std::cerr << "   - train-images-idx3-ubyte" << std::endl;
        std::cerr << "   - train-labels-idx1-ubyte" << std::endl;
        std::cerr << "   - t10k-images-idx3-ubyte" << std::endl;
        std::cerr << "   - t10k-labels-idx1-ubyte" << std::endl;
        return 1;
    }
}