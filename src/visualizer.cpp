#include "visualizer.h"
#include "utils.h"
#include <opencv2/opencv.hpp>
#include <random>


// Constants
static constexpr int IMAGE_DIM = 28;
static constexpr int SCALE = 8;
static constexpr int CELL_WIDTH = IMAGE_DIM * SCALE;
static constexpr int CELL_HEIGHT = IMAGE_DIM * SCALE;
static constexpr int LABEL_HEIGHT = 48;
static constexpr int GRID_COLS = 5;
static constexpr int GRID_ROWS = 3;
static constexpr int GRID_SIZE = GRID_ROWS * GRID_COLS;
static constexpr int PADDING = 12;

// Key Inputs
static constexpr int KEY_A = 'a';
static constexpr int KEY_D = 'd';
static constexpr int KEY_ESC = 27;
static constexpr int KEY_Q = 'q';
static constexpr int KEY_R = 'r';

// Colors
static const cv::Scalar BG = {30, 30, 30};
static const cv::Scalar CORRECT_COL = {80, 200, 100};
static const cv::Scalar WRONG_COL = {80, 80, 220};
static const cv::Scalar TRUE_COL = {210, 210, 210};
static const cv::Scalar BORDER_COL = {60, 60, 60};


static cv::Mat colToMat(const MNISTDataset& dataset, size_t col) {
    cv::Mat img(IMAGE_DIM, IMAGE_DIM, CV_8U);
    for (size_t i = 0; i < IMAGE_DIM; ++i) {
        for (size_t j = 0; j < IMAGE_DIM; ++j) {
            img.at<uint8_t>(i, j) = static_cast<uint8_t>(dataset.X(i * IMAGE_DIM + j, col) * 255.0);
        }
    }

    return img;
}

// Consider rewriting in network.cpp
static std::pair<size_t, double> runPredict(Network& net, const MNISTDataset& dataset, size_t idx) {
    Matrix X(784, 1);
    for (size_t row = 0; row < 784; ++row) X(row, 0) = dataset.X.at(row, idx);

    Matrix prediction = net.predict(X);

    size_t max_idx = 0;
    double max_val = prediction(0, 0);
    for (size_t i = 0; i < prediction.getRows(); ++i) {
        if (prediction(i, 0) > max_val) {
            max_val = prediction(i, 0);
            max_idx = i;
        }
    }

    return { max_idx, max_val };
}

static cv::Mat upscale(const cv::Mat& mat) {
    cv::Mat large, bgr;
    cv::resize(mat, large, cv::Size(CELL_WIDTH, CELL_HEIGHT), 0, 0, cv::INTER_NEAREST);
    cv::cvtColor(large, bgr, cv::COLOR_GRAY2BGR);
    return bgr;
}

static void drawLabels(
    cv::Mat& canvas,
    int cell_x, int cell_y,
    size_t true_label, size_t pred_label
) {
    const cv::Scalar pred_col = (pred_label == true_label) ? CORRECT_COL : WRONG_COL;
    const double font = cv::FONT_HERSHEY_SIMPLEX;
    const double scale = 0.55;
    const int thick = 1;

    int text_y_true = cell_y + CELL_HEIGHT + 16;
    int text_y_pred = cell_y + CELL_HEIGHT + 36;

    cv::putText(
        canvas,
        "True: " + ts(true_label),
        cv::Point(cell_x + 4, text_y_true),
        font, scale, TRUE_COL, thick, cv::LINE_AA
    );

    cv::putText(
        canvas,
        "Pred: " + std::to_string(pred_label),
        cv::Point(cell_x + 4, text_y_pred),
        font, scale, pred_col, thick + 1, cv::LINE_AA
    );
}

void Visualizer::viewSingle(const MNISTDataset& dataset, Network& net) {
    const size_t N = dataset.num_samples;
    if (N == 0) return;

    const int WINDOW_WIDTH = CELL_WIDTH + PADDING * 2;
    const int WINDOW_HEIGHT = CELL_HEIGHT + LABEL_HEIGHT + PADDING * 2 + 30;

    size_t idx = 0;
    const std::string win = "MNIST Viewer | <- -> to navigate | Q, ESC to exit";
    cv::namedWindow(win, cv::WINDOW_AUTOSIZE);

    while (true) {
        cv::Mat canvas(WINDOW_HEIGHT, WINDOW_WIDTH, CV_8UC3, BG);

        cv::Mat img = colToMat(dataset, idx);
        cv::Mat scaled = upscale(img);
        scaled.copyTo(canvas(cv::Rect(PADDING, 30 + PADDING, CELL_WIDTH, CELL_HEIGHT)));

        cv::rectangle(
            canvas,
            cv::Point(PADDING - 1, 30 + PADDING - 1),
            cv::Point(PADDING + CELL_WIDTH, 30 + PADDING + CELL_WIDTH),
            BORDER_COL, 1
        );

        auto [pred_label, pred_prob] = runPredict(net, dataset, idx);
        size_t true_label = dataset.labels[idx];

        cv::putText(
            canvas,
            "Sample " + ts(idx + 1) + " / " + ts(N),
            cv::Point(PADDING, 22),
            cv::FONT_HERSHEY_SIMPLEX, 0.55, TRUE_COL, 1, cv::LINE_AA
        );

        drawLabels(canvas, PADDING, 30 + PADDING, true_label, pred_label);

        cv::imshow(win, canvas);

        int key = cv::waitKey(0) & 0xFF;
        if (key == KEY_ESC || key == KEY_Q) break;
        if (key == KEY_D) idx = (idx + 1) % N;
        if (key == KEY_A) idx = (idx == 0) ? N - 1 : idx - 1;
    }

    cv::destroyAllWindows();
}

void Visualizer::viewGrid(const MNISTDataset& dataset, Network& net) {
    const size_t N = dataset.num_samples;
    if (N == 0) return;

    const int CELL_TOTAL_WIDTH = CELL_WIDTH + PADDING;
    const int CELL_TOTAL_HEIGHT = CELL_HEIGHT + LABEL_HEIGHT + PADDING;
    const int WINDOW_WIDTH = GRID_COLS * CELL_TOTAL_WIDTH + PADDING;
    const int WINDOW_HEIGHT = GRID_ROWS * CELL_TOTAL_HEIGHT + PADDING + 30;

    std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<size_t> dist(0, N - 1);

    const std::string win = "MNIST Grid | R to reshuffle | Q, ESC to exit";
    cv::namedWindow(win, cv::WINDOW_AUTOSIZE);

    auto make_indices = [&]() {
        std::vector<size_t> indices(GRID_SIZE);
        for (auto& v : indices) v = dist(rng);
        return indices;
    };

    std::vector<size_t> indices = make_indices();

    while (true) {
        cv::Mat canvas(WINDOW_HEIGHT, WINDOW_WIDTH, CV_8UC3, BG);

        cv::putText(
            canvas, "R = reshuffle   |   Q / ESC = quit",
            cv::Point(PADDING, 22),
            cv::FONT_HERSHEY_SIMPLEX, 0.5, TRUE_COL, 1, cv::LINE_AA
        );

        for (int i = 0; i < GRID_SIZE; ++i) {
            const int grid_col = i % GRID_COLS;
            const int grid_row = i / GRID_COLS;

            const int cell_x = PADDING + grid_col * CELL_TOTAL_WIDTH;
            const int cell_y = 30 + PADDING + grid_row * CELL_TOTAL_HEIGHT;

            size_t idx = indices[i];

            cv::Mat img = colToMat(dataset, idx);
            cv::Mat scaled = upscale(img);
            scaled.copyTo(canvas(cv::Rect(cell_x, cell_y, CELL_WIDTH, CELL_HEIGHT)));

            cv::rectangle(
                canvas,
                cv::Point(cell_x - 1, cell_y - 1),
                cv::Point(cell_x + CELL_WIDTH, cell_y + CELL_HEIGHT),
                BORDER_COL, 1
            );

            auto [pred_label, pred_prob] = runPredict(net, dataset, idx);
            int true_label = static_cast<int>(dataset.labels[idx]);

            drawLabels(canvas, cell_x, cell_y, true_label, pred_label);
        }

        cv::imshow(win, canvas);

        int key = cv::waitKey(0) & 0xFF;
        if (key == KEY_ESC || key == KEY_Q) break;
        if (key == KEY_R || key == ' ') indices = make_indices();
    }

    cv::destroyAllWindows();
}