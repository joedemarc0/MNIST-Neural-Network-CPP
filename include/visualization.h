#pragma once
#include <opencv2/opencv.hpp>
#include "data-loader.h"

cv::Mat mnist_to_mat(const MNISTSample& sample, size_t width, size_t height);
void show_images(MNISTLoader loader);