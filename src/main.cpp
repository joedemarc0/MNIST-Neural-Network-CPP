#include "data-loader.h"
#include "network.h"
#include "visualizer.h"
#include <iostream>
#include <random>
#include <opencv2/opencv.hpp>
#include <chrono>


int main() {
    try {
        MNISTDataset train_data = MNISTLoader::loadTrainingData();
        Network net = Network::loadModel("models/mnist_model1.txt");

        if (!net.checkCompiled()) {
            std::cout << "Network Compile Error" << std::endl;
            return 1;
        }

        Visualizer::viewGrid(train_data, net);
    
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}