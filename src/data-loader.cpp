#include "data-loader.h"
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <iomanip>
#include <cmath>


// Main function - reads in training or testing images and labels, normalizes images and one hots labels
MNISTDataset MNISTLoader::load_dataset(
            const std::string& images_path,
            const std::string& labels_path,
            bool normalize,
            bool one_hot) {
    std::cout << "Loading MNIST data from:" << std::endl;
    std::cout << "  Images: " << images_path << std::endl;
    std::cout << "  Labels: " << labels_path << std::endl;

    auto raw_images = read_mnist_images(images_path);
    auto raw_labels = read_mnist_labels(labels_path);

    if (raw_images.size() != raw_labels.size()) {
        throw std::runtime_error("Number of images and labels does not match");
    }

    MNISTDataset dataset;
    dataset.num_samples = raw_images.size();
    dataset.image_width = 28;
    dataset.image_height = 28;
    dataset.num_classes = 10;
    dataset.samples.reserve(dataset.num_samples);

    std::cout << "Processing " << dataset.num_samples << " samples..." << std::endl;

    for (size_t i = 0; i < raw_images.size(); ++i) {
        MNISTSample sample;
        sample.label_index = static_cast<int>(raw_labels[i]);

        sample.image.reserve(784);
        for (uint8_t pixel : raw_images[i]) {
            double pixel_val = static_cast<double>(pixel);
            if (normalize) {
                pixel_val /= 255.0;
            }
            sample.image.push_back(pixel_val);
        }

        if (one_hot) {
            sample.label = to_one_hot(sample.label_index, dataset.num_classes);
        } else {
            sample.label = {static_cast<double>(sample.label_index)};
        }

        dataset.samples.push_back(sample);

        if ((i + 1) % 10000 == 0) {
            std::cout << "  Processed " << (i + 1) << " samples..." << std::endl;
        }
    }

    calculate_statistics(dataset);
    std::cout << "Dataset loaded successfully" << std::endl;
    return dataset;
}

// These Functions can be called as objects and return the clean, normalized training and test data respectively
// Function that (when called) returns training data
MNISTDataset MNISTLoader::load_training_data(const std::string& data_dir) {
    return load_dataset(data_dir + "/train-images.idx3-ubyte", data_dir + "/train-labels.idx1-ubyte");
}

// Function that (when called) returns test data
MNISTDataset MNISTLoader::load_test_data(const std::string& data_dir) {
    return load_dataset(data_dir + "/t10k-images.idx3-ubyte", data_dir + "/t10k-labels.idx1-ubyte");
}

// Function to open image files and read data to a vector
std::vector<std::vector<uint8_t>> MNISTLoader::read_mnist_images(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + path);
    }

    uint32_t magic_number, num_images, num_rows, num_cols;
    file.read(reinterpret_cast<char*>(&magic_number), 4);
    file.read(reinterpret_cast<char*>(&num_images), 4);
    file.read(reinterpret_cast<char*>(&num_rows), 4);
    file.read(reinterpret_cast<char*>(&num_cols), 4);

    // Convert from big-endian to little-endian
    magic_number = swap_endian(magic_number);
    num_images = swap_endian(num_images);
    num_rows = swap_endian(num_rows);
    num_cols = swap_endian(num_cols);

    std::cout << "Image file info" << std::endl;
    std::cout << "  Magic number: " << magic_number << std::endl;
    std::cout << "  Number of images: " << num_images << std::endl;
    std::cout << "  Image dimensions: " << num_rows << "x" << num_cols << std::endl;

    if (magic_number != 2051) {
        throw std::runtime_error("Invalid magic number in image file: " + std::to_string(magic_number));
    }

    std::vector<std::vector<uint8_t>> images;
    images.reserve(num_images);

    for (uint32_t i = 0; i < num_images; ++i) {
        std::vector<uint8_t> image(num_rows * num_cols);
        file.read(reinterpret_cast<char*>(image.data()), num_rows * num_cols);
        images.push_back(image);
    }

    file.close();
    return images;
}

// Function to open labels files and read data to a vector
std::vector<uint8_t> MNISTLoader::read_mnist_labels(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + path);
    }

    uint32_t magic_number, num_labels;
    file.read(reinterpret_cast<char*>(&magic_number), 4);
    file.read(reinterpret_cast<char*>(&num_labels), 4);

    magic_number = swap_endian(magic_number);
    num_labels = swap_endian(num_labels);

    std::cout << "Label file info:" << std::endl;
    std::cout << "  Magic number: " << magic_number << std::endl;
    std::cout << "  Number of labels: " << num_labels << std::endl;

    if (magic_number != 2049) {
        throw std::runtime_error("Invalid magic number in label file: " + std::to_string(magic_number));
    }

    std::vector<uint8_t> labels(num_labels);
    file.read(reinterpret_cast<char*>(labels.data()), num_labels);

    file.close();
    return labels;
}

uint32_t MNISTLoader::swap_endian(uint32_t val) {
    return ((val << 24) & 0xFF000000) |
           ((val << 8)  & 0x00FF0000) |
           ((val >> 8)  & 0x0000FF00) |
           ((val >> 24) & 0x000000FF);
}

// Converts labels to one hot vectors
std::vector<double> MNISTLoader::to_one_hot(int label, int num_classes) {
    std::vector<double> one_hot(num_classes, 0.0);
    if (label >= 0 && label < num_classes) {
        one_hot[label] = 1.0;
    }
    return one_hot;
}

// Calculates mean pixel value and standard deivation for a dataset struct
void MNISTLoader::calculate_statistics(MNISTDataset& dataset) {
    if (dataset.samples.empty()) return;

    // Calculate mean pixel value across all images
    double sum = 0.0;
    size_t total_pixels = 0;
    for (const auto& sample : dataset.samples) {
        for (double pixel : sample.image) {
            sum += pixel;
            total_pixels++;
        }
    }
    dataset.mean_pixel_value = sum / total_pixels;

    // Calculate standard deviation
    double variance_sum = 0.0;
    for (const auto& sample : dataset.samples) {
        for (double pixel : sample.image) {
            double diff = pixel - dataset.mean_pixel_value;
            variance_sum += diff * diff;
        }
    }
    dataset.std_pixel_value = std::sqrt(variance_sum / total_pixels);
}

// Function to print dataset information and statistics
void MNISTLoader::print_dataset_info(const MNISTDataset& dataset) {
    std::cout << "\n=== MNIST Dataset Information ===" << std::endl;
    std::cout << "Number of samples: " << dataset.num_samples << std::endl;
    std::cout << "Image dimensions: " << dataset.image_width << "x" << dataset.image_height << std::endl;
    std::cout << "Number of classes: " << dataset.num_classes << std::endl;
    std::cout << "Mean pixel value: " << std::fixed << std::setprecision(4) << dataset.mean_pixel_value << std::endl;
    std::cout << "Std pixel value: " << std::fixed << std::setprecision(4) << dataset.std_pixel_value << std::endl;

    std::vector<int> class_counts(dataset.num_classes, 0);
    for (const auto& sample : dataset.samples) {
        class_counts[sample.label_index]++;
    }

    std::cout << "\nSamples per class:" << std::endl;
    for (int i = 0; i < dataset.num_classes; ++i) {
        std::cout << "  Class " << i << ": " << class_counts[i] << " samples" << std::endl;
    }
    std::cout << "================================\n" << std::endl;
}

void MNISTLoader::save_dataset(const MNISTDataset& dataset, const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot create file: " + filename);
    }

    // Write dataset metadata
    file.write(reinterpret_cast<const char*>(&dataset.num_samples), sizeof(int));
    file.write(reinterpret_cast<const char*>(&dataset.image_width), sizeof(int));
    file.write(reinterpret_cast<const char*>(&dataset.image_height), sizeof(int));
    file.write(reinterpret_cast<const char*>(&dataset.num_classes), sizeof(int));
    file.write(reinterpret_cast<const char*>(&dataset.mean_pixel_value), sizeof(double));
    file.write(reinterpret_cast<const char*>(&dataset.std_pixel_value), sizeof(double));

    for (const auto& sample : dataset.samples) {
        // Write image data
        size_t image_size = sample.image.size();
        file.write(reinterpret_cast<const char*>(&image_size), sizeof(size_t));
        file.write(reinterpret_cast<const char*>(sample.image.data()), image_size * sizeof(double));

        // Write label data
        size_t label_size = sample.label.size();
        file.write(reinterpret_cast<const char*>(&label_size), sizeof(size_t));
        file.write(reinterpret_cast<const char*>(sample.label.data()), label_size * sizeof(double));

        // Write label index
        file.write(reinterpret_cast<const char*>(&sample.label_index), sizeof(int));
    }

    file.close();
    std::cout << "Dataset saved to: " << filename << std::endl;
}

MNISTDataset MNISTLoader::load_dataset_binary(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    MNISTDataset dataset;

    // Read dataset metadata
    file.read(reinterpret_cast<char*>(&dataset.num_samples), sizeof(int));
    file.read(reinterpret_cast<char*>(&dataset.image_width), sizeof(int));
    file.read(reinterpret_cast<char*>(&dataset.image_height), sizeof(int));
    file.read(reinterpret_cast<char*>(&dataset.num_classes), sizeof(int));
    file.read(reinterpret_cast<char*>(&dataset.mean_pixel_value), sizeof(double));
    file.read(reinterpret_cast<char*>(&dataset.std_pixel_value), sizeof(double));

    dataset.samples.reserve(dataset.num_samples);

    // Read samples
    for (int i = 0; i < dataset.num_samples; ++i) {
        MNISTSample sample;

        // Read image data
        size_t image_size;
        file.read(reinterpret_cast<char*>(&image_size), sizeof(size_t));
        sample.image.resize(image_size);
        file.read(reinterpret_cast<char*>(sample.image.data()), image_size * sizeof(double));

        // Read label data
        size_t label_size;
        file.read(reinterpret_cast<char*>(&label_size), sizeof(size_t));
        sample.label.resize(label_size);
        file.read(reinterpret_cast<char*>(sample.label.data()), label_size * sizeof(double));

        // Read label index
        file.read(reinterpret_cast<char*>(&sample.label_index), sizeof(int));

        dataset.samples.push_back(sample);
    }

    file.close();
    std::cout << "Dataset loaded from: " << filename << std::endl;
    return dataset;
}