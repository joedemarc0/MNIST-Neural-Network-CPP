/**
 * WIP - Visualization Implementation
 * 
 * Idea is to be able to display MNIST Images and their labels - as well as what the model predicts the image is displaying
 * Probably going to shuffle through images randomly and display them in a 3x5 grid of images.
 * Python version is implemented quite nicely however its hard without matplotlib :(
 */

#pragma once
#include <opencv2/opencv.hpp>
#include "data-loader.h"

cv::Mat mnist_to_mat(const MNISTSample& sample, size_t width, size_t height);
void show_images(MNISTLoader loader);