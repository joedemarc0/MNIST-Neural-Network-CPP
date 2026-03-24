#ifndef VISUALIZER_H
#define VISUALIZER_H


#include "network.h"
#include "data-loader.h"

namespace Visualizer {
    void viewSingle(const MNISTDataset& dataset, Network& net);
    void viewGrid(const MNISTDataset& dataset, Network& net);
}


#endif // VISUALIZER_H