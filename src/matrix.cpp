#include "matrix.h"
#include <stdexcept>
#include <random>
#include <iomanip>
#include <cmath>


// Matrix Class Constructors
Matrix::Matrix() : rows(0), cols(0) {}

Matrix::Matrix(size_t rows, size_t cols)
    : rows(rows), cols(cols), data(rows * cols, 0.0) {}

Matrix::Matrix(size_t rows, size_t cols, double value)
    : rows(rows), cols(cols), data(rows * cols, value) {}

Matrix::Matrix(Matrix&& other) noexcept
    : rows(other.rows), cols(other.cols), data(std::move(other.data)) {
    other.rows = other.cols = 0;
}


// Assignment Operators
Matrix& Matrix::operator=(Matrix&& other) noexcept {
    if (this != &other) {
        rows = other.rows;
        cols = other.cols;
        data = std::move(other.data);
        other.rows = other.cols = 0;
    }

    return *this;
}


// Element Access Operators
double& Matrix::operator()(size_t row, size_t col) {
    if (row >= rows || col >= cols) {
        throw std::out_of_range("Matrix Operator (): Matrix index out of range");
    }

    return data[row * cols + col];
}

const double& Matrix::operator()(size_t row, size_t col) const {
    if (row >= rows || col >= cols) {
        throw std::out_of_range("Matrix Operator (): Matrix index out of range");
    }

    return data[row * cols + col];   
}


// Matrix Operations
Matrix Matrix::operator+(const Matrix& other) const {
    const double* __restrict a = data.data();
    const double* __restrict b = other.data.data();
    const size_t n = data.size();

    if (matchDim(*this, other)) {
        Matrix result(rows, cols);
        double* __restrict r = result.data.data();
        for (size_t i = 0; i < n; ++i) r[i] = a[i] + b[i];
        return result;
    } else if (rows == other.rows && other.cols == 1) {
        Matrix result(rows, cols);
        double* __restrict r = result.data.data();

        for (size_t row = 0; row < rows; ++row) {
            const double bias = b[row];
            const size_t base = row * cols;
            for (size_t col = 0; col < cols; ++col) r[base + col] = a[base + col] + bias;
        }

        return result;
    }

    throw std::invalid_argument("Matrix Operator (+): Matrix dimensions must match");
}

Matrix Matrix::operator-(const Matrix& other) const {
    if (!matchDim(*this, other)) {
        throw std::invalid_argument("Matrix Operator (-): Matrix dimensions must match");
    }

    Matrix result(rows, cols);
    const double* __restrict a = data.data();
    const double* __restrict b = other.data.data();
    double* __restrict r = result.data.data();

    const size_t n = data.size();
    for (size_t i = 0; i < n; ++i) r[i] = a[i] - b[i];
    return result;
}

Matrix Matrix::operator*(const Matrix& other) const {
    if (cols != other.rows) {
        throw std::invalid_argument("Matrix Operator (*): Matrix dimensions must be valid");
    }

    Matrix result(rows, other.cols);
    const double* __restrict a = data.data();
    const double* __restrict b = other.data.data();
    double* __restrict r = result.data.data();

    for (size_t i = 0; i < rows; ++i) {
        const double* a_row = a + i * cols;
        double* r_row = r + i * other.cols;

        for (size_t k = 0; k < cols; ++k) {
            const double a_ik = a_row[k];
            const double* b_row = b + k * other.cols;
            for (size_t j = 0; j < other.cols; ++j) r_row[j] += a_ik * b_row[j];
        }
    }

    return result;
}


// Scalar Operations
Matrix Matrix::operator*(double scalar) const {
    Matrix result(rows, cols);
    const double* __restrict a = data.data();
    double* __restrict r = result.data.data();

    const size_t n = data.size();
    for (size_t i = 0; i < n; ++i) r[i] = a[i] * scalar;
    return result;
}

Matrix Matrix::operator/(double scalar) const {
    if (std::abs(scalar) < 1e-12) {
        throw std::invalid_argument("Matrix Scalar Division: Divide by zero error");
    }

    return *this * (1.0 / scalar);
}


// In-place Matrix Operations
Matrix& Matrix::operator+=(const Matrix& other) {
    if (!matchDim(*this, other)) {
        throw std::invalid_argument("Matrix Operator (+=): Matrix dimensions must match");
    }

    double* __restrict a = data.data();
    const double* __restrict b = other.data.data();

    const size_t n = data.size();
    for (size_t i = 0; i < n; ++i) a[i] += b[i];
    return *this;
}

Matrix& Matrix::operator-=(const Matrix& other) {
    if (!matchDim(*this, other)) {
        throw std::invalid_argument("Matrix Operator (-=): Matrix dimensions must match");
    }

    double* __restrict a = data.data();
    const double* __restrict b = other.data.data();

    const size_t n = data.size();
    for (size_t i = 0; i < n; ++i) a[i] -= b[i];
    return *this;
}


// In-place Scalar Operations
Matrix& Matrix::operator*=(double scalar) {
    for (double &v : data) v *= scalar;
    return *this;
}

Matrix& Matrix::operator/=(double scalar) {
    if (std::abs(scalar) < 1e-12) {
        throw std::invalid_argument("Matrix Operator (/=): Divide by zero error");
    }

    const double inv = 1.0 / scalar;
    for (double &v : data) v *= inv;
    return *this;
}


// Boolean Operations
bool Matrix::operator==(const Matrix& other) const {
    if (!matchDim(*this, other)) return false;
    const double* __restrict a = data.data();
    const double* __restrict b = other.data.data();

    const size_t n = data.size();
    for (size_t i = 0; n; ++i) if (std::abs(a[i] - b[i]) > 1e-9) return false;
    return true;
}

bool Matrix::operator!=(const Matrix& other) const {
    return !(*this == other);
}


// Specialized Operations
Matrix Matrix::hadamard(const Matrix& other) const {
    if (!matchDim(*this, other)) {
        throw std::invalid_argument("Matrix::hadamard(): Matrix dimensions must match");
    }

    Matrix result(rows, cols);
    const double* __restrict a = data.data();
    const double* __restrict b = other.data.data();
    double* __restrict r = result.data.data();

    const size_t n = data.size();
    for (size_t i = 0; i < n; ++i) r[i] = a[i] * b[i];
    return result;
}

Matrix Matrix::transpose() const {
    Matrix result(cols, rows);
    const double* __restrict a = data.data();
    double* __restrict r = result.data.data();

    for (size_t row = 0; row < rows; ++row) {
        for (size_t col = 0; col < cols; ++col) {
            r[col * rows + row] = a[row * cols + col];
        }
    }

    return result;
}

Matrix Matrix::apply(std::function<double(double)> func) const {
    Matrix result(rows, cols);
    const double* __restrict a = data.data();
    double* __restrict r = result.data.data();

    const size_t n = data.size();
    for (size_t i = 0; i < n; ++i) r[i] = func(a[i]);
    return result;
}

Matrix Matrix::diag() const {
    const double* __restrict a = data.data();

    if (cols == rows) {
        Matrix result(rows, 1);
        double* __restrict r = result.data.data();
        for (size_t row = 0; row < rows; ++row) r[row] = a[row * cols + row];
        return result;
    } else if (cols == 1) {
        Matrix result(rows, rows);
        double* __restrict r = result.data.data();
        for (size_t row = 0; row < rows; ++row) r[row * rows + row] = a[row];
        return result;
    }
    
    throw std::invalid_argument("Matrix::diag(): Matrix must be square or column vector");
}


// Initialization Methods
void Matrix::randomize(double min, double max) {
    static std::random_device rd;
    static std::mt19937 g(rd());
    std::uniform_real_distribution<double> dis(min, max);
    for (double &v : data) v = dis(g);
}

void Matrix::xavierInit() {
    double limit = std::sqrt(6.0 / (rows + cols));
    randomize(-limit, limit);
}

void Matrix::heInit() {
    static std::random_device rd;
    static std::mt19937 g(rd());
    double stddev = std::sqrt(2.0 / cols);
    std::normal_distribution<double> dis(0.0, stddev);
    for (double &v : data) v = dis(g);
}

void Matrix::fill(double value) {
    std::fill(data.begin(), data.end(), value);
}

Matrix Matrix::identity(size_t dim) {
    Matrix result(dim, dim);
    for (size_t i = 0; i < dim; ++i) result.data[i * dim + i] = 1.0;
    return result;
}


// Utility Methods
bool Matrix::matchDim(const Matrix& a, const Matrix& b) {
    return (a.rows == b.rows && a.cols == b.cols);
}

double Matrix::sum() const {
    return std::accumulate(data.begin(), data.end(), 0.0);
}

Matrix Matrix::getRow(size_t row) const {
    if (row >= rows) {
        throw std::out_of_range("Matrix::getRow(): Row index out of range");
    }

    Matrix result(1, cols);
    const double* src = data.data() + row * cols;
    std::copy(src, src + cols, result.data.data());
    return result;
}

Matrix Matrix::getCol(size_t col) const {
    if (col >= cols) {
        throw std::out_of_range("Matrix::getCol(): Col index out of range");
    }

    Matrix result(rows, 1);
    const double* __restrict a = data.data();
    double* __restrict r = result.data.data();
    for (size_t i = 0; i < rows; ++i) r[i] = a[i * cols + col];
    return result;
}

Matrix Matrix::sumCols() const {
    Matrix result(rows, 1);
    const double* __restrict a = data.data();
    double* __restrict r = result.data.data();
    for (size_t row = 0; row < rows; ++row) {
        const double* row_ptr = a + row * cols;
        for (size_t col = 0; col < cols; ++col) r[row] += row_ptr[col];
    }

    return result;
}

Matrix Matrix::sliceCols(const std::vector<size_t>& sliced_indices) const {
    const size_t batch_size = sliced_indices.size();
    Matrix result(rows, batch_size);
    const double* __restrict a = data.data();
    double* __restrict r = result.data.data();

    for (size_t col = 0; col < batch_size; ++col) {
        for (size_t row = 0; row < rows; ++row) {
            r[row * batch_size + col] = a[row * cols + sliced_indices[col]];
        }
    }

    return result;
}

void Matrix::setCol(size_t col, const Matrix& colMatrix) {
    if (col >= cols) {
        throw std::out_of_range("Matrix::setCol(): Col index out of range");
    } else if (colMatrix.cols != 1 || colMatrix.rows != rows) {
        throw std::invalid_argument("Matrix::setCol(): colMatrix has invalid matrix dimensions");
    }

    const double* col_ptr = colMatrix.data.data();
    for (size_t row = 0; row < rows; ++row) {
        data[row * cols + col] = col_ptr[row];
    }
}

void Matrix::resize(size_t newRows, size_t newCols) {
    rows = newRows;
    cols = newCols;
    data.assign(rows * cols, 0.0);
}

void Matrix::print() const {
    std::cout << std::fixed << std::setprecision(6);
    for (size_t row = 0; row < rows; ++row) {
        std::cout << "[";
        const double* row_ptr = data.data() + row * cols;
        for (size_t col = 0; col < cols; ++col) {
            std::cout << std::setw(10) << row_ptr[col];
            if (col < cols - 1) std::cout << " ";
        }
        std::cout << "]" << std::endl;
    }
    std::cout << std::endl;
}

bool Matrix::hasNaNOrInf() const {
    for (const double v : data) {
        if (!std::isfinite(v)) return true;
    }

    return false;
}

void Matrix::assertFinite() const {
    if (hasNaNOrInf()) {
        throw std::runtime_error("NaN or Inf errors detected");
    }
}