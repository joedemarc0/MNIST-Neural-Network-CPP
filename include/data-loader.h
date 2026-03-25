#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#include "matrix.h"
#include <vector>
#include <utility>
#include <string>
#include <cstdint>
#include <variant>


/**
 * We should implement toOneHot in network.cpp and pass the labels data
 * as a vector of bytes rather than as a Matrix
 * 
 * That way we can easier implement getAccuracy, getCorrect for efficiency
 * because we don't NEED to onehot encode for that.
 * 
 * OneHot is really only useful for the loss calculation and whatnot
 * We can pass the labels as a vector for everything else
 */


struct MNISTDataset {
    Matrix X;
    std::vector<uint8_t> labels;
    size_t num_samples;

    MNISTDataset(
    ) : X(),
        labels(),
        num_samples(0)
    {}

    MNISTDataset(
        Matrix X_,
        std::vector<uint8_t> labels_
    ) : X(std::move(X_)),
        labels(std::move(labels_)),
        num_samples(labels.size())
    {
        if (X_.getCols() != labels_.size()) {
            throw std::invalid_argument("Number of samples does not match");
        }
    }
};


class MNISTLoader {
    public:
        static MNISTDataset loadDataset(
            const std::string& images_path,
            const std::string& labels_path,
            bool normalize = true
        );

        static MNISTDataset loadTrainingData(const std::string& data_dir="./data/", bool normalize=true);
        static MNISTDataset loadTestingData(const std::string& data_dir="./data/", bool normalize=true);

        static std::pair<MNISTDataset, MNISTDataset> split(
            const MNISTDataset& dataset,
            size_t val_size,
            bool shuffle = true
        );

        static std::vector<uint8_t> sliceLabels(
            const std::vector<uint8_t>& labels,
            const std::vector<size_t>& sliced_indices
        );
    
    private:
        struct RawImages {
            std::vector<uint8_t> bytes;
            uint32_t num_images;
            uint32_t image_width;
            uint32_t image_height;
        };

        struct RawLabels {
            std::vector<uint8_t> bytes;
            uint32_t num_labels;
        };

        static uint32_t swapEndian(uint32_t val);
        static RawImages readImages(const std::string& path);
        static RawLabels readLabels(const std::string& path);
        static Matrix toMatrix(const RawImages& raw, bool normalize);
};

#endif // DATA_LOADER_H