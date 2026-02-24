#ifndef DATA_LOADER2_H
#define DATA_LOADER2_H

#include "matrix.h"
#include <vector>
#include <utility>
#include <string>
#include <cstdint>


struct MNISTDataset {
    Matrix X;
    Matrix y;
    size_t num_samples;
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

        static std::pair<MNISTDataset, MNISTDataset> split(
            const MNISTDataset& dataset,
            double val_fraction,
            bool shuffle = true
        );

        void printDatasetInfo(const MNISTDataset& dataset);
        void saveDataset(const MNISTDataset& dataset, const std::string& filename);
    
    private:
        struct RawImages {
            std::vector<uint8_t> bytes;
            uint32_t num_images;
        };

        struct RawLabels {
            std::vector<uint8_t> bytes;
            uint32_t num_labels;
        };

        static RawImages readImages(const std::string& path);
        static RawLabels readLabels(const std::string& path);
        static uint32_t swapEndian(uint32_t val);
};


#endif // DATA_LOADER2_H