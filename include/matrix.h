/**
 * A minimal "matrix" or "tensor" implementation - don't have to rely on std::vector<float> or whatever everywhere
 * 
 * Store 2D arrays (weights, biases, activations)
 * Basic ops: dot product, elementwise add/multiply, transpose, random init
 * Numpy-lite
 */

#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <iostream>
#include <functional>


class Matrix {
    private:
        std::vector<double> data;
        size_t rows, cols;
    
    public:
        Matrix();
        Matrix(size_t rows, size_t cols);
        Matrix(size_t rows, size_t cols, double value);
        Matrix(Matrix&& other) noexcept;
        Matrix(const Matrix& other) = default;

        // Getters
        size_t getRows() const { return rows; }
        size_t getCols() const { return cols; }

        // Assignment Operators
        Matrix& operator=(Matrix&& other) noexcept;
        Matrix& operator=(const Matrix& other) = default;

        // Element Access Operators
        double& operator()(size_t row, size_t col);
        const double& operator()(size_t row, size_t col) const;
        double& at(size_t row, size_t col) { return data[row * cols + col]; }
        const double& at(size_t row, size_t col) const { return data[row * cols + col]; }

        // Matrix Operations
        Matrix operator+(const Matrix& other) const;
        Matrix operator-(const Matrix& other) const;
        Matrix operator*(const Matrix& other) const;

        // Scalar Operations
        Matrix operator*(double scalar) const;
        Matrix operator/(double scalar) const;

        // In-place Matrix Operations
        Matrix& operator+=(const Matrix& other);
        Matrix& operator-=(const Matrix& other);

        // In-place Scalar Operations
        Matrix& operator*=(double scalar);
        Matrix& operator/=(double scalar);

        // Boolean Operations
        bool operator==(const Matrix& other) const;
        bool operator!=(const Matrix& other) const;

        // Specialized Operations
        Matrix hadamard(const Matrix& other) const;
        Matrix transpose() const;
        Matrix apply(std::function<double(double)> func) const;
        Matrix diag() const;

        // Initialization Methods
        void randomize(double min=0.0, double max=1.0);
        void xavierInit();
        void heInit();
        void fill(double value);
        static Matrix identity(size_t dim);

        // Utility Methods
        static bool matchDim(const Matrix& a, const Matrix& b);
        double sum() const;
        Matrix getRow(size_t row) const;
        Matrix getCol(size_t col) const;
        Matrix sumCols() const;
        Matrix sliceCols(const std::vector<size_t>& sliced_indices) const;
        void setCol(size_t col, const Matrix& colMatrix);
        void resize(size_t newRows, size_t newCols);
        void print() const;
        bool empty() const { return rows == 0 || cols == 0; }
        bool hasNaNOrInf() const;
        void assertFinite() const;
};

// External scalar multiplication (scalar * matrix)
inline Matrix operator*(const double scalar, const Matrix& matrix) {
    return matrix * scalar;
}

// Stream output operator
inline std::ostream& operator<<(std::ostream& os, const Matrix& matrix) {
    os << "Matrix(" << matrix.getRows() << ", " << matrix.getCols() << ")";
    return os;
}


#endif // MATRIX_H