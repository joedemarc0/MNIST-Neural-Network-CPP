#include "data-loader.h"
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <iomanip>
#include <cmath>
#include <random>


// MNISTLoader Public Functions
MNISTDataset MNISTLoader::loadDataset(
    const std::string& images_path,
    const std::string& labels_path,
    bool normalize
) {
    std::cout << "Loading MNIST data from: " << std::endl;
    std::cout << "  Images: " << images_path << std::endl;
    std::cout << "  Labels: " << labels_path << std::endl;

    RawImages raw_images = readImages(images_path);
    RawLabels raw_labels = readLabels(labels_path);

    if (raw_images.num_images != raw_labels.num_labels) {
        throw std::runtime_error(
            "Number of images and labels does not match (" +
            std::to_string(raw_images.num_images) + " images and " +
            std::to_string(raw_labels.num_labels) + " labels)"
        );
    }

    MNISTDataset dataset {
        toMatrix(raw_images, normalize),
        raw_labels.bytes
    };

    return dataset;
}

MNISTDataset MNISTLoader::loadTrainingData(const std::string& data_dir, bool normalize) {
    return loadDataset(
        data_dir + "/train-images.idx3-ubyte",
        data_dir + "/train-labels.idx1-ubyte",
        normalize
    );
}

MNISTDataset MNISTLoader::loadTestingData(const std::string& data_dir, bool normalize) {
    return loadDataset(
        data_dir + "/t10k-images.idx3-ubyte",
        data_dir + "/t10k-labels.idx1-ubyte",
        normalize
    );
}

std::pair<MNISTDataset, MNISTDataset> MNISTLoader::split(
    const MNISTDataset& dataset,
    size_t val_size,
    bool shuffle
) {
    size_t num_samples = dataset.num_samples;
    if (val_size == 0 || val_size >= num_samples) {
        throw std::invalid_argument("Invalid validation set size selection");
    }

    std::vector<size_t> index(num_samples);
    std::iota(index.begin(), index.end(), 0);

    if (shuffle) {
        uint32_t seed = 35;
        std::mt19937 g(seed);
        std::shuffle(index.begin(), index.end(), g);
    }

    std::vector<size_t> train_index(index.begin() + val_size, index.end());
    std::vector<size_t> val_index(index.begin(), index.begin() + val_size);

    MNISTDataset train_data {
        dataset.X.sliceCols(train_index),
        sliceLabels(dataset.labels, train_index),
    };

    MNISTDataset val_data {
        dataset.X.sliceCols(val_index),
        sliceLabels(dataset.labels, val_index),
    };

    return { std::move(train_data), std::move(val_data) };
}

// MNISTLoader Private Functions
uint32_t MNISTLoader::swapEndian(uint32_t val) {
    return ((val << 24) & 0xFF000000u) |
           ((val <<  8) & 0x00FF0000u) |
           ((val >>  8) & 0x0000FF00u) |
           ((val >> 24) & 0x000000FFu);
}

MNISTLoader::RawImages MNISTLoader::readImages(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + path);
    }

    uint32_t magic_number, num_images, num_rows, num_cols;
    file.read(reinterpret_cast<char*>(&magic_number), 4);
    file.read(reinterpret_cast<char*>(&num_images), 4);
    file.read(reinterpret_cast<char*>(&num_rows), 4);
    file.read(reinterpret_cast<char*>(&num_cols), 4);

    magic_number = swapEndian(magic_number);
    num_images = swapEndian(num_images);
    num_rows = swapEndian(num_rows);
    num_cols = swapEndian(num_cols);

    if (magic_number != 2051) {
        throw std::runtime_error("Invalid magic number in image file: " + std::to_string(magic_number));
    }

    size_t pixels_per_image = static_cast<size_t>(num_rows) * num_cols;
    size_t total_bytes = static_cast<size_t>(num_images) * pixels_per_image;

    RawImages raw;
    raw.num_images = num_images;
    raw.image_width = num_cols;
    raw.image_height = num_rows;
    raw.bytes.resize(total_bytes);
    file.read(reinterpret_cast<char*>(raw.bytes.data()), static_cast<std::streamsize>(total_bytes));

    if (!file) {
        throw std::runtime_error(
            "MNISTLoader: truncated IMAGE file (read " +
            std::to_string(file.gcount()) + " of " +
            std::to_string(total_bytes) + " bytes)"
        );
    }

    return raw;
}

MNISTLoader::RawLabels MNISTLoader::readLabels(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + path);
    }

    uint32_t magic_number, num_labels;
    file.read(reinterpret_cast<char*>(&magic_number), 4);
    file.read(reinterpret_cast<char*>(&num_labels), 4);
    
    magic_number = swapEndian(magic_number);
    num_labels = swapEndian(num_labels);

    if (magic_number != 2049) {
        throw std::runtime_error("Invalid magic number in labels file: " + std::to_string(magic_number));
    }

    RawLabels raw;
    raw.num_labels = num_labels;
    raw.bytes.resize(num_labels);
    file.read(reinterpret_cast<char*>(raw.bytes.data()), static_cast<std::streamsize>(num_labels));

    if (!file) {
        throw std::runtime_error(
            "MNISTLoader: truncated LABEL file (read " +
            std::to_string(file.gcount()) + " of " +
            std::to_string(num_labels) + " bytes)"
        );
    }

    return raw;
}

Matrix MNISTLoader::toMatrix(const RawImages& raw, bool normalize) {
    size_t num_images = raw.num_images;
    size_t num_pixels = raw.image_width * raw.image_height;
    double scale = normalize ? (1.0 / 255.0) : 1.0;

    Matrix X(num_pixels, num_images);
    for (size_t col = 0; col < num_images; ++col) {
        size_t base = col * num_pixels;
        for (size_t row = 0; row < num_pixels; ++row) {
            X(row, col) = static_cast<double>(raw.bytes[base + row]) * scale;
        }
    }

    return X;
}

std::vector<uint8_t> MNISTLoader::sliceLabels(const std::vector<uint8_t>& labels, const std::vector<size_t>& sliced_indices) {
    size_t batch_size = sliced_indices.size();
    std::vector<uint8_t> result(batch_size);

    for (size_t i = 0; i < batch_size; ++i) result[i] = labels[sliced_indices[i]];
    return result;
}