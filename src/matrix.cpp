#include "matrix.h"
#include <stdexcept>
#include <random>
#include <iomanip>
#include <cmath>


// Constructors
Matrix::Matrix() : rows(0), cols(0) {}

Matrix::Matrix(
    size_t rows,
    size_t cols
) : rows(rows),
    cols(cols)
{
    data.resize(rows, std::vector<double>(cols, 0.0));
}

Matrix::Matrix(
    size_t rows,
    size_t cols,
    double value
) : rows(rows),
    cols(cols)
{
    data.resize(rows, std::vector<double>(cols, value));
}

Matrix::Matrix(
    const Matrix& other
) : data(other.data),
    rows(other.rows),
    cols(other.cols)
{}


// Assignment operator
Matrix& Matrix::operator=(const Matrix& other) {
    if (this != &other) {
        data = other.data;
        rows = other.rows;
        cols = other.cols;
    }

    return *this;
}

// Element Access
double& Matrix::operator()(size_t row, size_t col) {
    if (row >= rows || col >= cols) {
        throw std::out_of_range("Matri index out of range");
    }

    return data[row][col];
}

const double& Matrix::operator()(size_t row, size_t col) const {
    if (row >= rows || col >= cols) {
        throw std::out_of_range("Matrix index out of range");
    }

    return data[row][col];
}

// Matrix operations
Matrix Matrix::operator+(const Matrix& other) const {
    if (rows == other.rows && cols == other.cols) {
        Matrix result(rows, cols);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result(i, j) = data[i][j] + other(i, j);
            }
        }
        return result;

    } else if (rows == other.rows && other.cols == 1) {
        Matrix result(rows, cols);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result(i, j) = data[i][j] + other(i, 0);
            }
        }
        return result;

    } else {
        throw std::invalid_argument("Matrix Dimensions do not Match");
    }
}

Matrix Matrix::operator-(const Matrix& other) const {
    if (rows != other.rows || cols != other.cols) {
        throw std::invalid_argument("Matrix dimensions do not match");
    }

    Matrix result(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result(i, j) = data[i][j] - other(i, j);
        }
    }
    return result;
}

Matrix Matrix::operator*(const Matrix& other) const {
    if (cols != other.rows) {
        throw std::invalid_argument("Invalid dimensions for matrix multiplication");
    }

    Matrix result(rows, other.cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < other.cols; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < cols; ++k) {
                sum += data[i][k] * other(k, j);
            }
            result(i, j) = sum;
        }
    }
    return result;
}

// Scalar operations
Matrix Matrix::operator*(double scalar) const {
    Matrix result(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result(i, j) = data[i][j] * scalar;
        }
    }
    return result;
}

Matrix Matrix::operator/(double scalar) const {
    if (std::abs(scalar) < 1e-12) {
        throw std::invalid_argument("Division by zero");
    }
    return *this * (1.0 / scalar);
}

// In-place Matrix operations
Matrix& Matrix::operator+=(const Matrix& other) {
    *this = *this + other;
    return *this;
}

Matrix& Matrix::operator-=(const Matrix& other) {
    *this = *this - other;
    return *this;
}

// In-place Scalar operations
Matrix& Matrix::operator*=(double scalar) {
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            data[i][j] *= scalar;
        }
    }
    
    return *this;
}

Matrix& Matrix::operator/=(double scalar) {
    if (std::abs(scalar) < 1e-12) {
        throw std::invalid_argument("Divide by zero error");
    }

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            data[i][j] /= scalar;
        }
    }

    return *this;
}

bool Matrix::operator==(const Matrix& other) const {
    if (rows != other.getRows() || cols != other.getCols()) {
        return false;
    }
    
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            if (std::abs(data[i][j] - other(i, j)) > 1e-9) {
                return false;
            }
        }
    }

    return true;
}

bool Matrix::operator!=(const Matrix& other) const {
    if (rows != other.getRows() || cols != other.getCols()) {
        return true;
    }

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            if (std::abs(data[i][j] - other(i ,j)) > 1e-9) {
                return true;
            }
        }
    }

    return false;
}

// Element-wise (Hadamard) product
Matrix Matrix::hadamard(const Matrix& other) const {
    if (rows != other.rows || cols != other.cols) {
        throw std::invalid_argument("Matrix dimensions do not match");
    }

    Matrix result(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result(i, j) = data[i][j] * other(i, j);
        }
    }
    return result;
}

// Transpose function
Matrix Matrix::transpose() const {
    Matrix result(cols, rows);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result(j, i) = data[i][j];
        }
    }
    return result;
}

// Apply function - input is a function R -> R and returns matrix with function applied to each element
Matrix Matrix::apply(std::function<double(double)> func) const {
    Matrix result(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result(i, j) = func(data[i][j]);
        }
    }
    return result;
}

Matrix Matrix::diag() const {
    if (cols == rows) {
        Matrix result(rows, 1);
        for (size_t i = 0; i < rows; ++i) {
            result(i, 0) = data[i][i];
        }
        return result;

    } else if (cols == 1) {
        Matrix result(rows, rows); 
        for (size_t i = 0; i < rows; ++i) {
            result(i, i) = data[i][0];
        }
        return result;

    } else {
        throw std::runtime_error("Matrix must be of dimension (n x 1) or (n x n)");
    }
}

// Initialize with random values
void Matrix::randomize(double min, double max) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(min, max);

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            data[i][j] = dis(gen);
        }
    }
}

void Matrix::xavierInit() {
    double limit = std::sqrt(6.0 / (rows + cols));
    randomize(-limit, limit);
}

void Matrix::heInit() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    double stddev = std::sqrt(2.0 / cols);
    std::normal_distribution<double> dis(0.0, stddev);

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            data[i][j] = dis(gen);
        }
    }
}

void Matrix::fill(double value) {
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            data[i][j] = value;
        }
    }
}

void Matrix::identity() {
    if (rows != cols) {
        throw std::invalid_argument("Indentity Matrix must be square");
    }

    fill(0.0);
    for (size_t i = 0; i < rows; ++i) {
        data[i][i] = 1.0;
    }
}

// Utility Methods
double Matrix::sum() const {
    double total = 0.0;
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            total += data[i][j];
        }
    }
    return total;
}

std::vector<double> Matrix::getRow(size_t row) const {
    if (row >= rows) {
        throw std::out_of_range("Row index out of range");
    }
    return data[row];
}

Matrix Matrix::getCol(size_t col) const {
    if (col >= cols) {
        throw std::out_of_range("Column index out of range");
    }

    Matrix result(rows, 1);
    for (size_t i = 0; i < rows; ++i) {
        result(i, 0) = data[i][col];
    }
    return result;
}

Matrix Matrix::sumCols() const {
    Matrix result(rows, 1);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result(i, 0) += data[i][j];
        }
    }
    return result;
}

// Probably Needed Once we implement Mini-Batch Training
Matrix Matrix::sliceCols(std::vector<size_t> indices, size_t begin, size_t end) const {
    /**
     * @param indices A vector of shuffled indices (for matrix with 6 cols: [2, 3, 6, 1, 5, 4] for example)
     * @param begin starting index
     * @param end ending index
     * 
     * The function should read [begin, end] and return the columns of the Matrix represented by the indices
     * in the vector indices - from [begin to end]
     * 
     * For example: indices = [2, 3, 6, 1, 5, 4] - begin = 2, end = 4
     * The function will return a Matrix of size (*this.rows)x(end - begin), where end - begin = 2, where the columns
     * of the matrix are column 6 and column 1 of the Matrix object
     * 
     * Exceptions: end cannot be greater than Matrix.getCols()
     */

    if (begin >= end || end > indices.size()) {
        throw std::out_of_range("Invalid slice range in sliceCols");
    }

    size_t newCols = end - begin;
    Matrix result(rows, newCols);

    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < newCols; ++c) {
            size_t colIdx = indices[begin + c];
            result(r, c) = data[r][colIdx];
        }
    }

    return result;
}

void Matrix::setCol(size_t col, const Matrix& colMatrix) {
    if (col >= cols) {
        throw std::out_of_range("Column index out of range");
    }
    if (colMatrix.getCols() != 1 || colMatrix.getRows() != rows) {
        throw std::invalid_argument("Matrix Dimnesions must match");
    }

    for (size_t i = 0; i < rows; ++i) {
        data[i][col] = colMatrix(i, 0);
    }
}

void Matrix::resize(size_t newRows, size_t newCols) {
    rows = newRows;
    cols = newCols;
    data.clear();
    data.resize(rows, std::vector<double>(cols, 0.0));
}

void Matrix::print() const {
    std::cout << std::fixed << std::setprecision(6);
    for (size_t i = 0; i < rows; ++i) {
        std::cout << "[";
        for (size_t j = 0; j < cols; ++j) {
            std::cout << std::setw(10) << data[i][j];
            if (j < cols - 1) std::cout << " ";
        }
        std::cout << "]" << std::endl;
    }
    std::cout << std::endl;
}

bool Matrix::hasNaNOrInf() const {
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            if(!std::isfinite(data[i][j])) {
                return true;
            }
        }
    }
    return false;
}

void Matrix::assertFinite() const {
    if (hasNaNOrInf()) {
        throw std::runtime_error("NaN or Inf Values Detected");
    }
}