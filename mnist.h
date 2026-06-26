//
// Created by Nikita on 6/25/2026.
//

#ifndef UNTITLED_MNIST_H
#define UNTITLED_MNIST_H
#include <cstdint>
#include <fstream>
#include <vector>

#include "matrix.h"

uint32_t swap32(uint32_t x) {
    return  ((x & 0xff000000) >> 24) |
            ((x & 0x00ff0000) >> 8) |
            ((x & 0x0000ff00) << 8) |
            ((x & 0x000000ff) << 24);
}

std::vector<Matrix> load_images(const std::string& path) {
    std::vector<Matrix> result;

    std::ifstream file(path, std::ios::binary);

    uint32_t magic;
    uint32_t count;
    uint32_t rows;
    uint32_t cols;

    file.read((char*)&magic, 4);
    file.read((char*)&count, 4);
    file.read((char*)&rows, 4);
    file.read((char*)&cols, 4);

    magic = swap32(magic);
    count = swap32(count);
    rows = swap32(rows);
    cols = swap32(cols);

    if (magic != 2051) {
        throw std::runtime_error("Invalid image file");
    }

    for (int i = 0; i < count; i++) {
        Matrix subResult((int)rows * (int)cols, 1);
        for (int j = 0; j < rows; j++) {
            for (int k = 0; k < cols; k++) {
                uint8_t pixel;
                file.read((char*)&pixel, 1);
                subResult.set(j * cols + k, 0, pixel/255.f);
            }
        }
        result.push_back(subResult);
    }

    return result;
}

std::vector<uint8_t> load_labels(const std::string& path) {
    std::vector<uint8_t> result;

    std::ifstream file(path, std::ios::binary);

    uint32_t magic;
    uint32_t count;

    file.read((char*)&magic, 4);
    file.read((char*)&count, 4);

    magic = swap32(magic);
    count = swap32(count);

    if (magic != 2049) {
        throw std::runtime_error("Invalid image file");
    }

    for (int i = 0; i < count; i++) {
        uint8_t label;
        file.read((char*)&label, 1);
        result.push_back(label);
    }

    return result;
}

#endif //UNTITLED_MNIST_H
