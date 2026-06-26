#include <iostream>
#include <string>

#include "matrix.h"
#include "mnist.h"

int main() {
    std::vector<Matrix> images = load_images("../dataset/train-images.idx3-ubyte");
    std::vector<uint8_t> labels = load_labels("../dataset/train-labels.idx1-ubyte");

    std::cout << (int)labels.back() << std::endl;
    std::cout << std::endl;

    for (int i = 0; i < 28; i++) {
        for (int j = 0; j < 28; j++) {
            std::cout << (images.back().at(i * 28 + j, 0) > 0.5f ? "#" : ".") << " ";
        }
        std::cout << std::endl;
    }


    return 0;
}