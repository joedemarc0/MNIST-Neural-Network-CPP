#pragma once

#include <vector>
#include <string>
#include <cstdint>


/**
 * Structure to hold a single MNIST image sample
 * Contains the pixel data, label information, and original label index
 */
struct MNISTSample {
    std::vector<double> image;
    std::vector<double> label;
    int label_index;
};

/**
 * Structure to hold entire MNIST dataset in form of MNISTSamples
 * Contains all samples plus metadata and statistical information
 */
struct MNISTDataset {
    std::vector<MNISTSample> samples;
    int num_samples;
    int image_width;
    int image_height;
    int num_classes;

    double mean_pixel_value;
    double std_pixel_value;
};


/**
 * MNIST Data Loader Class
 * Handles reading, processing, and managing MNIST ubyte data files
 */
class MNISTLoader {
    public:
        // Constructors
        MNISTLoader() = default;
        ~MNISTLoader() = default;

        /**
         * Load MNIST dataset from ubyte files and process data
         * @param images_path Path to the MNIST images ubyte file
         * @param labels_path Path to the MNIST labels ubyte file
         * @param normalize Whether to normalize pixel values from [0,255] to [0,1]
         * @param one_hot Whether to convert labels to one-hot encoding (10 classes) or keep as single values
         * @return MNISTDataset Complete processed dataset with all samples and metadata
         * 
         * Reads raw MNIST ubyte files, converts pixel values to doubles, optionally normalizes them,
         * converts labels to desired format, and calculates dataset statistics
         */
        MNISTDataset load_dataset(
                    const std::string& images_path,
                    const std::string& labels_path,
                    bool normalize=true,
                    bool one_hot=true
        );

        /**
         * Load MNIST training dataset using standard filenames
         * @param data_dir Directory containing MNIST data files
         * @return MNISTDataset Training dataset (60,000 samples)
         * 
         * Convenience function that loads training data using standard MNIST filenames
         * (train-images.idx3-ubyte and train-labels.idx1-ubyte)
         */
        MNISTDataset load_training_data(const std::string& data_dir="./data/");

        /**
         * Load MNIST test dataset using standard filenames  
         * @param data_dir Directory containing MNIST data files
         * @return MNISTDataset Test dataset (10,000 samples)
         * 
         * Convenience function that loads test data using standard MNIST filenames
         * (t10k-images.idx3-ubyte and t10k-labels.idx1-ubyte).
         */
        MNISTDataset load_test_data(const std::string& data_dir="./data/");

        /**
         * Print comprehensive information about a loaded dataset
         * @param dataset The MNISTDataset to analyze and display
         * @return void
         * 
         * Display dataset statistics including sample count, dimensions , class distribution,
         * mean/std pixel value, and samples per digit class
         */
        void print_dataset_info(const MNISTDataset& dataset);

        /**
         * Save processed dataset to binary format for faster future loading
         * @param dataset The MNISTDataset to save
         * @param filename Output filename for the binary dataset file
         * @return void
         * 
         * Serializes the entire dataset (samples, metadata, statistics) to a binary file
         * for much faster loading compared to processing ubyte files each time
         */
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