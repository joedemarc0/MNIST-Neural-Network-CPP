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
    std::cout << "MNISTLoader: reading images from " << images_path << std::endl;
    RawImages raw_images = readImages(images_path);

    std::cout << "MNISTLoader: reading labels from " << labels_path << std::endl;
    RawLabels raw_labels = readLabels(labels_path);

    if (raw_images.num_images != raw_labels.num_labels) {
        throw std::runtime_error(
            "MNISTLoader: image/label count mismatch (" +
            std::to_string(raw_images.num_images) + " images and " +
            std::to_string(raw_labels.num_labels) + " labels)"
        );
    }

    const size_t num_images = raw_images.num_images;
    const size_t PIXELS = 784;
    const size_t CLASSES = 10;
    const double scale = normalize ? (1.0 / 255.0) : 1.0;

    Matrix X(PIXELS, num_images);
    Matrix y(CLASSES, num_images);

    for (size_t col = 0; col < num_images; ++col) {
        const size_t base = col * PIXELS;
        for (size_t row = 0; row < PIXELS; ++row) {
            X(row, col) = static_cast<double>(raw_images.bytes[base + row]) * scale;
        }
    }

    for (size_t col = 0; col < num_images; ++col) {
        const size_t class_index = static_cast<size_t>(raw_labels.bytes[col]);
        if (class_index >= CLASSES) {
            throw std::runtime_error("MNISTLoader: label out of range: " + std::to_string(class_index));
        }

        y(class_index, col) = 1.0;
    }

    std::cout << "MNISTLoader: loaded " << num_images << " samples" << std::endl;
    return MNISTDataset{ std::move(X), std::move(y), num_images };
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
    const size_t num_samples = dataset.num_samples;
    if (val_size == 0 || val_size >= num_samples) {
        throw std::invalid_argument("MNISTLoader::split - invalid val_size selection");
    }

    std::vector<size_t> index(num_samples);
    std::iota(index.begin(), index.end(), 0);

    if (shuffle) {
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(index.begin(), index.end(), g);
    }

    std::vector<size_t> val_index(index.begin(), index.begin() + val_size);
    std::vector<size_t> train_index(index.begin() + val_size, index.end());

    const size_t train_size = num_samples - val_size;

    MNISTDataset train_data {
        dataset.X.sliceCols(train_index),
        dataset.y.sliceCols(train_index),
        train_size
    };

    MNISTDataset val_data {
        dataset.X.sliceCols(val_index),
        dataset.y.sliceCols(val_index),
        val_size
    };

    std::cout << "MNISTLoader::split - train: " << train_size << std::endl;
    std::cout << "MNISTLoader::split - val: " << val_size << std::endl;

    return { std::move(train_data), std::move(val_data) };
}

std::pair<MNISTDataset, MNISTDataset> MNISTLoader::split(
    const MNISTDataset& dataset,
    double val_percent,
    bool shuffle
) {
    if (val_percent <= 0.0 || val_percent >= 1.0) {
        throw std::invalid_argument("MNISTLoader::split - invalid val_percent selection");
    }

    const size_t val_size = std::max<size_t>(1, static_cast<size_t>(std::round(val_percent * dataset.num_samples)));
    return split(dataset, val_size, shuffle);
}

void MNISTLoader::printDatasetInfo(const MNISTDataset& dataset) {
    std::cout << "\n=== MNIST Dataset Information ===" << std::endl;
    std::cout << "Number of samples: " << dataset.num_samples << std::endl;
}

void MNISTLoader::saveDataset(const MNISTDataset& dataset, const std::string& filename) {}


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

    const size_t pixels_per_image = static_cast<size_t>(num_rows) * num_cols;
    const size_t total_bytes = static_cast<size_t>(num_images) * pixels_per_image;

    RawImages raw;
    raw.num_images = num_images;
    raw.bytes.resize(total_bytes);
    file.read(reinterpret_cast<char*>(raw.bytes.data()), static_cast<std::streamsize>(total_bytes));

    if (!file) {
        throw std::runtime_error(
            "MNISTLoader: truncated image file (read " +
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
            "MNISTLoader: truncated label file (read " +
            std::to_string(file.gcount()) + " of " +
            std::to_string(num_labels) + " bytes)"
        );
    }

    return raw;
}