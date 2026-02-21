#pragma once

#include <vector>
#include <string>
#include <cstdint>


struct MNISTSample {
    std::vector<double> image;
    std::vector<double> label;
    int label_index;
};

struct MNISTDataset {
    std::vector<MNISTSample> samples;
    int num_samples;
    int image_width;
    int image_height;
    int num_classes;

    double mean_pixel_value;
    double std_pixel_value;
};


class MNISTLoader {
    public:
        // Constructors
        MNISTLoader() = default;
        ~MNISTLoader() = default;

        MNISTDataset load_dataset(
                    const std::string& images_path,
                    const std::string& labels_path,
                    bool normalize=true,
                    bool one_hot=true
        );

        MNISTDataset load_training_data(const std::string& data_dir="./data/");
        MNISTDataset load_test_data(const std::string& data_dir="./data/");
        void print_dataset_info(const MNISTDataset& dataset);
        void save_dataset(const MNISTDataset& dataset, const std::string& filename);

        /**
         * Load previously saved dataset from binary format
         * @param filename Path to binary dataset file
         * @return MNISTDataset Complete dataset laoded from binary file
         * 
         * Deserializes a binary dataset file created by save_dataset(), providing
         * significantly faster loading compared to processing original ubyte files
         */
        MNISTDataset load_dataset_binary(const std::string& filename);
    
    private:
        /**
         * Read MNIST image file in idx3-ubyte format
         * @param path Path to MNIST images ubyte file
         * @return std::vector<std::vector<uint8_t>> 2D vector where each inner vector is 
         * flattened 28x28 image
         * 
         * Parses MNIST idx3-ubyte format, handles big-endian conversion, validates magic number,
         * and extracts raw pixel data (0-255 values) for all images
         */
        std::vector<std::vector<uint8_t>> read_mnist_images(const std::string& path);

        /**
         * Read MNIST label file in idx1-ubyte format
         * @param path Path to the MNIST labels ubyte file
         * @return std::vector<uint8_t> Vector containing integer labels (0-9) for each sample
         * 
         * Parses MNIST idx1-ubyte format, handles big-endian conversion, validates magic number,
         * and extracts label values corresponding to each image.
         */
        std::vector<uint8_t> read_mnist_labels(const std::string& path);

        /**
         * Convert 32-bit integer from big-endian to little-endian format
         * @param val 32-bit unsigned integer in big-endian format
         * @return uint32_t Same value converted to little-endian format
         * 
         * MNIST files use big-endian byte order, but most systems use little-endian
         * This function swaps byte order for proper integer interpretation
         */
        uint32_t swap_endian(uint32_t val);

        /**
         * Convert integer label to one-hot encoded vector with 10 classes
         * @param label Integer label value (0-9)
         * @param num_classes Total number of classes (default 10 for digits 0-9)
         * @return std::vector<double> One-hot encoded vector with 1.0 for label position, 0.0 elsewhere
         * 
         * Create a vector of zeros with a single 1.0 at the index corresponding to the label
         * Used for neural networks with categorical cross-entropy loss
         */
        std::vector<double> to_one_hot(int label, int num_classes=10);

        /**
         * Calculate and store statistical information for the MNISTDataset
         * @param dataset MNISTDataset for statistics to be updated
         * @return void
         * 
         * Computes mean and standard deviation of pixel values for a MNISTDataset
         * Useful for data analysis and possible normalization techniques
         */
        void calculate_statistics(MNISTDataset& dataset);
};