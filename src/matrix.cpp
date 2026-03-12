#include "matrix.h"
#include <stdexcept>
#include <random>
#include <iomanip>
#include <cmath>


// Matrix Class Constructors
Matrix::Matrix() : rows(0), cols(0) {}

Matrix::Matrix(const Matrix& other) : rows(other.rows), cols(other.cols), data(other.data) {}

Matrix::Matrix(size_t rows, size_t cols) : rows(rows), cols(cols) {
    data.resize(rows * cols, 0.0);
}

Matrix::Matrix(size_t rows, size_t cols, double value) : rows(rows), cols(cols) {
    data.resize(rows * cols, value);
}


// Assigment and Element Access
Matrix& Matrix::operator=(const Matrix& other) {
    if (this != &other) {
        rows = other.rows;
        cols = other.cols;
        data = other.data;
    }

    return *this;
}

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

inline double& Matrix::at(size_t row, size_t col) {
    return data[row * cols + col];
}


// Matrix Operations
Matrix Matrix::operator+(const Matrix& other) const {
    if (matchDim(*this, other)) {
        Matrix result(rows, cols);
        for (size_t i = 0; i < data.size(); ++i) {
            result.data[i] = data[i] + other.data[i];
        }

        return result;
    } else if (rows == other.rows && other.cols == 1) {
        Matrix result(rows, cols);
        for (size_t i = 0; i < data.size(); ++i) {
            size_t row = i / cols;
            result.data[i] = data[i] + other.data[row];
        }

        return result;
    } else {
        throw std::invalid_argument("Matrix Operator (+): Matrix dimensions must match");
    }
}

Matrix Matrix::operator-(const Matrix& other) const {
    if (!matchDim(*this, other)) {
        throw std::invalid_argument("Matrix Operator (-): Matrix dimensions must match");
    }

    Matrix result(rows, cols);
    for (size_t i = 0; i < data.size(); ++i) {
        result.data[i] = data[i] - other.data[i];
    }

    return result;
}

Matrix Matrix::operator*(const Matrix& other) const {}


// Scalar Operations
Matrix Matrix::operator*(double scalar) const{
    Matrix result(rows, cols);
    for (size_t i = 0; i < data.size(); ++i) {
        result.data[i] = data[i] * scalar;
    }

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

    for (size_t i = 0; i < data.size(); ++i) {
        data[i] += other.data[i];
    }

    return *this;
}

Matrix& Matrix::operator-=(const Matrix& other) {
    if (!matchDim(*this, other)) {
        throw std::invalid_argument("Matrix Operator (-=): Matrix dimensions must match");
    }

    for (size_t i = 0; i < data.size(); ++i) {
        data[i] -= other.data[i];
    }

    return *this;
}


// In-place Scalar Operations
Matrix& Matrix::operator*=(double scalar) {
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] *= scalar;
    }

    return *this;
}

Matrix& Matrix::operator/=(double scalar) {
    if (std::abs(scalar) < 1e-12) {
        throw std::invalid_argument("Matrix Operator (/=): Divide by zero error");
    }

    for (size_t i = 0; i < data.size(); ++i) {
        data[i] /= scalar;
    }

    return *this;
}


// Boolean Operator
bool Matrix::operator==(const Matrix& other) const {
    if (!matchDim(*this, other)) {
        return false;
    }

    for (size_t i = 0; i < data.size(); ++i) {
        if (std::abs(data[i] - other.data[i]) > 1e-9) return false;
    }

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
    for (size_t i = 0; i < data.size(); ++i) {
        result.data[i] = data[i] * other.data[i];
    }

    return result;
}

Matrix Matrix::transpose() const {
    Matrix result(cols, rows);
    for (size_t row = 0; row < rows; ++row) {
        for (size_t col = 0; col < cols; ++col) {
            result(col, row) = (*this)(row, col);
        }
    }

    return result;
}

Matrix Matrix::apply(std::function<double(double)> func) const {
    Matrix result(rows, cols);
    for (size_t i = 0; i < data.size(); ++i) {
        result.data[i] = func(data[i]);
    }

    return result;
}

Matrix Matrix::diag() const {
    if (cols == rows) {
        Matrix result(rows, 1);
        for (size_t row = 0; row < rows; ++row) {
            result.data[row] = data[row * cols + row];
        }

        return result;
    } else if (cols == 1) {
        Matrix result(rows, rows);
        for (size_t row = 0; row < rows; ++row) {
            result.data[row * rows + row] = data[row];
        }

        return result;
    } else {
        throw std::invalid_argument("Matrix::diag(): Matrix must be square or column vector");
    }
}


// Initialization Methods
void Matrix::randomize(double min, double max) {
    static std::random_device rd;
    static std::mt19937 g(rd());
    std::uniform_real_distribution<double> dis(min, max);

    for (double &v : data) {
        v = dis(g);
    }
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

    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = dis(g);
    }
}

void Matrix::fill(double value) {
    std::fill(data.begin(), data.end(), value);
}

Matrix Matrix::identity(size_t dim) {
    Matrix result(dim, dim);
    for (size_t i = 0; i < dim; ++i) {
        result(i, i) = 1.0;
    }

    return result;
}


// Utility Methods
bool Matrix::matchDim(const Matrix& a, const Matrix& b) {
    return (a.rows == b.rows && a.cols == b.cols);
}

double Matrix::sum() const {
    double sum = 0.0;
    for (size_t i = 0; i < data.size(); ++i) {
        sum += data[i];
    }

    return sum;
}

Matrix Matrix::getRow(size_t row) const {
    if (row >= rows) {
        throw std::out_of_range("Matrix::getRow(): Row index out of range");
    }

    Matrix result(1, cols);
    for (size_t i = 0; i < cols; ++i) {
        result(0, i) = (*this)(row, i);
    }

    return result;
}

Matrix Matrix::getCol(size_t col) const {
    if (col >= cols) {
        throw std::out_of_range("Matrix::getCol(): Col index out of range");
    }

    Matrix result(rows, 1);
    for (size_t i = 0; i < rows; ++i) {
        result(i, 0) = (*this)(i, col);
    }

    return result;
}

Matrix Matrix::sumCols() const {
    Matrix result(rows, 1);
    for (size_t row = 0; row < rows; ++row) {
        for (size_t col = 0; col < cols; ++col) {
            result(row, 0) += (*this)(row, col);
        }
    }

    return result;
}

Matrix Matrix::sliceCols(const std::vector<size_t>& sliced_indices) const {
    size_t batch_size = sliced_indices.size();
    Matrix result(rows, batch_size);

    for (size_t col = 0; col < batch_size; ++col) {
        for (size_t row = 0; row < rows; ++row) {
            result.data[row * batch_size + col] = data[row * cols + sliced_indices[col]];
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

    for (size_t i = 0; i < rows; ++i) {
        (*this)(i, col) = colMatrix(i, 0);
    }
}


void Matrix::resize(size_t newRows, size_t newCols) {
    rows = newRows;
    cols = newCols;
    data.assign(rows * cols, 0.0);
}

void Matrix::print() const {
    std::cout << std::fixed << std::setprecision(6);
    for (size_t i = 0; i < rows; ++i) {
        std::cout << "[";
        for (size_t j = 0; j < cols; ++j) {
            std::cout << std::setw(10) << (*this)(i, j);
            if (j < cols - 1) std::cout << " ";
        }
        std::cout << "]" << std::endl;
    }
    std::cout << std::endl;
}

bool Matrix::hasNaNOrInf() const {
    for (size_t i = 0; i < data.size(); ++i) {
        if (!std::isfinite(data[i])) {
            return true;
        }
    }

    return false;
}

void Matrix::assertFinite() const {
    if (hasNaNOrInf()) {
        throw std::runtime_error("NaN or Inf errors detected");
    }
}