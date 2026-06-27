//
// Created by Nikita on 6/25/2026.
//

#ifndef UNTITLED_NETWORK_H
#define UNTITLED_NETWORK_H
#include  <random>
#include <fstream>

#include "matrix.h"

float relu(float x) {
    return x > 0 ? x : 0;
}

float relu_derivative(float x) {
    return x > 0 ? 1.f : 0.f;
}

void softmax(Matrix& input) {
    float sum = 0;
    for (int i = 0; i < input.rows(); i++) {
        for (int j = 0; j < input.cols(); j++) {
            sum += exp(input.at(i, j));
        }
    }

    for (int i = 0; i < input.rows(); i++) {
        for (int j = 0; j < input.cols(); j++) {
            input.set(i, j, exp(input.at(i, j)) / sum);
        }
    }
}

float loss(const Matrix& output, uint8_t label) {
    return -log(output.at(label, 0));
}

Matrix one_hot(uint8_t label) {
    Matrix output(10, 1);
    output.set(label, 0, 1.f);
    return output;
}

class Network {

    private:
    Matrix weights1;
    Matrix weights2;
    Matrix bias1;
    Matrix bias2;
    Matrix hidden;
    float lr = 0.001;

    public:

    Network()
          :weights1(128, 784), weights2(10, 128), bias1(128, 1), bias2(10, 1), hidden(128, 1)
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

        // random weights 1
        for (int i = 0; i < weights1.rows(); i++) {
            for (int j = 0; j < weights1.cols(); j++) {
                weights1.set(i, j, dist(gen));
            }
        }

        // random weights 2
        for (int i = 0; i < weights2.rows(); i++) {
            for (int j = 0; j < weights2.cols(); j++) {
                weights2.set(i, j, dist(gen));
            }
        }
    }

    Matrix forward(Matrix input) {
        hidden = (weights1 * input + bias1).apply(relu);
        Matrix output = (weights2 * hidden + bias2);
        softmax(output);
        return output;
    }

    void backward(Matrix input, Matrix output, uint8_t label) {
        Matrix dOutput = output + one_hot(label) * -1.f;

        Matrix dWeights2 = dOutput * hidden.transpose();

        Matrix dBias2 = dOutput;

        Matrix dHidden = weights2.transpose() * dOutput;
        dHidden = dHidden % hidden.apply(relu_derivative);

        Matrix dWeights1 = dHidden * input.transpose();

        Matrix dBias1 = dHidden;

        // Update the weights
        weights1 = weights1 - dWeights1 * lr;
        weights2 = weights2 - dWeights2 * lr;
        bias1 = bias1 - dBias1 * lr;
        bias2 = bias2 - dBias2 * lr;
    }

    void save(const char *path) {
        std::ofstream file(path, std::ios::binary);

        for (int i = 0; i < weights1.rows(); i++) {
            for (int j = 0; j < weights1.cols(); j++) {
                float val = weights1.at(i, j);
                file.write((char*)&val, sizeof(float));
            }
        }

        for (int i = 0; i < weights2.rows(); i++) {
            for (int j = 0; j < weights2.cols(); j++) {
                float val = weights2.at(i, j);
                file.write((char*)&val, sizeof(float));
            }
        }

        for (int i = 0; i < bias1.rows(); i++) {
            for (int j = 0; j < bias1.cols(); j++) {
                float val = bias1.at(i, j);
                file.write((char*)&val, sizeof(float));
            }
        }

        for (int i = 0; i < bias2.rows(); i++) {
            for (int j = 0; j < bias2.cols(); j++) {
                float val = bias2.at(i, j);
                file.write((char*)&val, sizeof(float));
            }
        }

    }

    void load(const char *path) {
        std::ifstream file(path, std::ios::binary);

        for (int i = 0; i < weights1.rows(); i++) {
            for (int j = 0; j < weights1.cols(); j++) {
                float val;
                file.read((char*)&val, sizeof(float));
                weights1.set(i, j, val);
            }
        }

        for (int i = 0; i < weights2.rows(); i++) {
            for (int j = 0; j < weights2.cols(); j++) {
                float val;
                file.read((char*)&val, sizeof(float));
                weights2.set(i, j, val);
            }
        }

        for (int i = 0; i < bias1.rows(); i++) {
            for (int j = 0; j < bias1.cols(); j++) {
                float val;
                file.read((char*)&val, sizeof(float));
                bias1.set(i, j, val);
            }
        }

        for (int i = 0; i < bias2.rows(); i++) {
            for (int j = 0; j < bias2.cols(); j++) {
                float val;
                file.read((char*)&val, sizeof(float));
                bias2.set(i, j, val);
            }
        }

    }
};

#endif //UNTITLED_NETWORK_H
