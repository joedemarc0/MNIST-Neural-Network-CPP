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
        std::vector<std::vector<double>> data;
        size_t rows, cols;
    
    public:
        // Constructors
        Matrix();
        Matrix(size_t rows, size_t cols);
        Matrix(size_t rows, size_t cols, double value);
        Matrix(const Matrix& other);

        // Assignment operator
        Matrix& operator=(const Matrix& other);

        // Getters
        size_t getRows() const { return rows; }
        size_t getCols() const { return cols; }

        // Element access
        double& operator()(size_t row, size_t col);
        const double& operator()(size_t row, size_t col) const;

        // Matrix operations
        Matrix operator+(const Matrix& other) const;
        Matrix operator-(const Matrix& other) const;
        Matrix operator*(const Matrix& other) const;

        // Scalar operations
        Matrix operator*(double scalar) const;
        Matrix operator/(double scaler) const;

        // In-place operations
        Matrix& operator+=(const Matrix& other);
        Matrix& operator-=(const Matrix& other);
        Matrix& operator*=(double scalar);

        // Boolean operation
        bool operator==(const Matrix& other) const;

        // Specialized operations
        Matrix hadamard(const Matrix& other) const;
        Matrix transpose() const;
        Matrix apply(std::function<double(double)> func) const;

        // Initializations methods
        void randomize(double min=-1.0, double max=1.0);
        void xavierInit();
        void heInit();
        void fill(double value);
        void identity();

        // Utility methods
        double sum() const;
        std::vector<double> getRow(size_t row) const;
        Matrix getCol(size_t col) const;
        Matrix sliceCols(std::vector<size_t> indices, size_t begin, size_t end) const;
        void setCol(size_t col, const Matrix& colMatrix);
        void resize(size_t newRows, size_t newCols);
        void print() const;
        bool empty() const { return rows == 0 || cols == 0; }
};

// External scalar multiplication (scalar * matrix)
inline Matrix operator*(double scalar, const Matrix& matrix) {
    return matrix * scalar;
}

// Stream output operator
inline std::ostream& operator<<(std::ostream& os, const Matrix& matrix) {
    os << "Matrix(" << matrix.getRows() << "x" << matrix.getCols() << ")";
    return os;
}

#endif // MATRIX_H