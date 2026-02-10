#include "data-loader.h"
#include "visualization.h"
#include <opencv2/opencv.hpp>
#include <random>


cv::Mat mnist_to_mat(const MNISTSample& sample, size_t width, size_t height) {
    cv::Mat img(height, width, CV_64F);

    for (size_t r = 0; r < height; ++r) {
        for (size_t c = 0; c < width; ++c) {
            img.at<double>(r, c) = sample.image[r * width + c];
        }
    }

    cv::Mat img_8u;
    img.convertTo(img_8u, CV_8U, 255.0);

    return img_8u;
}

void show_images(MNISTLoader loader) {
    MNISTDataset train = loader.load_training_data("./data");
    MNISTDataset test = loader.load_test_data("./data");

    std::vector<MNISTSample> all_samples = train.samples;
    all_samples.insert(all_samples.end(), test.samples.begin(), test.samples.end());

    const int num_images = 15;
    const int num_cols = 5;
    const int num_rows = 3;

    std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<> dist(0, all_samples.size() - 1);

    int img_w = train.image_width;
    int img_h = train.image_height;

    cv::Mat canvas(num_rows * img_h, num_cols * img_w, CV_8U, cv::Scalar(0));

    for (int i = 0; i < num_images; ++i) {
        int idx = dist(rng);
        const auto& sample = all_samples[idx];

        cv::Mat img = mnist_to_mat(sample, img_w, img_h);

        int r = i / num_cols;
        int c = i % num_cols;

        img.copyTo(
            canvas(cv::Rect(c * img_w, r * img_h, img_w, img_h))
        );

        cv::putText(
            canvas,
            std::to_string(sample.label_index),
            cv::Point(c * img_w + 2, r * img_h + 14),
            cv::FONT_HERSHEY_SIMPLEX,
            0.5,
            cv::Scalar(255)
        );
    }

    cv::imshow("MNIST Samples", canvas);
    cv::waitKey(0);
}