#include <iostream>
#include "data-loader.h"


int main() {
    try {
        std::cout << "=== MNIST Data Loader Test ===" << std::endl;

        MNISTLoader loader;

        std::cout << "\nTesting with training data" << std::endl;
        MNISTDataset train_data = loader.load_training_data("./data");

        loader.print_dataset_info(train_data);

        if (!train_data.samples.empty()) {
            const auto& first_sample = train_data.samples[0];
            std::cout << "First sample details:" << std::endl;
            std::cout << "  Label index: " << first_sample.label_index << std::endl;
            std::cout << "  Image size: " << first_sample.image.size() << std::endl;
            std::cout << "  Label size: " << first_sample.label.size() << std::endl;
            std::cout << "  First few pixels: ";
            for (int i = 0; i < 5 && i < first_sample.image.size(); ++i) {
                std::cout << first_sample.image[i] << " ";
            }
            std::cout << std::endl;
        }

        std::cout << "\n=== Test completed successfully ===" << std::endl;

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

    return 0;
}