//
// Created by Nikita on 6/24/2026.
//

#ifndef UNTITLED_MATRIX_H
#define UNTITLED_MATRIX_H
#include <iosfwd>
#include <stdexcept>
#include <vector>



class Matrix {

    private:
    int m_rows;
    int m_cols;

    std::vector<float> data;

    public:
    Matrix(int rows, int cols)
        :m_rows(rows), m_cols(cols), data(rows * cols, 0.f)
    {

    }

    float at(int row, int col) const {
        return data[m_cols * row + col];
    }

    void set(int row, int col, float val) {
        data[m_cols * row + col] = val;
    }

    int rows() const {return m_rows;}
    int cols() const {return m_cols;}

    // sum
    Matrix operator+(const Matrix& other) {
        if (rows() != other.rows() || cols() != other.cols()) {
            throw std::runtime_error("Incompatible matrix sizes");
        }

        Matrix result(rows(), cols());

        for (int i = 0; i < rows(); i++) {
            for (int j = 0; j < cols(); j++) {
                result.set(i, j, at(i, j) + other.at(i, j));
            }
        }
        return result;
    }

    // dot product
    Matrix operator*(const Matrix& other) {
        if (cols() != other.rows()) {
            throw std::runtime_error("Cannot multiply matrices");
        }
        Matrix result(rows(), other.cols());

        for (int i  = 0; i < rows(); i++) {
            for (int j = 0; j < other.cols(); j++) {
                float subresult = 0;
                for (int k = 0; k < cols(); k++) {
                    subresult += at(i, k) * other.at(k, j);
                }
                result.set(i, j, subresult);
            }
        }
        return result;
    }

    // multiply elementwise
    Matrix operator%(const Matrix& other) {
        if (rows() != other.rows() || cols() != other.cols()) {
            throw std::runtime_error("Incompatible matrix sizes");
        }

        Matrix result(rows(), cols());

        for (int i = 0; i < rows(); i++) {
            for (int j = 0; j < cols(); j++) {
                result.set(i, j, at(i, j) * other.at(i, j));
            }
        }
        return result;
    }

    // matrix transpose
    Matrix transpose() {
        Matrix result(cols(), rows());

        for (int i = 0; i < result.rows(); i++) {
            for (int j = 0; j < result.cols(); j++) {
                result.set(i, j, at(j, i));
            }
        }
        return result;
    }

    // func apply
    Matrix apply(float (*func)(float)) {
        Matrix result(rows(), cols());
        for (int i = 0; i < rows(); i++) {
            for (int j = 0; j < cols(); j++) {
                result.set(i, j, func(at(i, j)));
            }
        }
        return result;
    }

};

#endif //UNTITLED_MATRIX_H